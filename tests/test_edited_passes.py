"""A pass over an instruction edit belongs to the checkpoint that drew it.

The reported bug: Upscale on a Qwen Edit result ran Qwen Edit again, with the
edit's order ("replace the head piece with jewelry. Keep everything else
exactly unchanged.") as its prompt -- and picking a checkpoint instead still
sent that order as the checkpoint's prompt. The edit is finished by then: the
upscale is the checkpoint's, and its prompt is the picture's own prompt with
the edit worked in (measured: the words still move details at upscale
strength). Fix faces and More like this copied the same two fields and had
the same bug.
"""

import io
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import main
from app.registry import CONFIG_PATH, load_registry
from app.sessions import Store

DRAWER = "oneobsession"      # a Danbooru checkpoint, untiled: 2048 at most
EDITOR = "qwen-edit"
ORDER = "replace the head piece with jewelry. Keep everything else exactly unchanged."
SCENE = "masterpiece, 1girl, ornate silver headdress, crimson gown"


@pytest.fixture
def client(tmp_path, monkeypatch):
    reg = load_registry(CONFIG_PATH)
    for spec in reg.models.values():
        spec.installed = True
    reg.upscaler = "4x_ultrasharp_f16.ckpt"
    monkeypatch.setattr(main, "registry", reg)
    monkeypatch.setattr(main, "store", Store(tmp_path))
    return TestClient(main.app)


def made(kind, model_key, prompt, parent=None, session=None, size=1024, neg=""):
    sid = session or main.store.create_session("x").id
    gen = main.store.create_generation(
        session_id=sid, kind=kind, model_key=model_key, prompt=prompt,
        negative_prompt=neg, seed=42, width=size, height=size, steps=20,
        cfg=4.0, sampler=None, parent_id=parent.id if parent else None,
    )
    buf = io.BytesIO()
    Image.new("RGB", (size, size), (30, 60, 90)).save(buf, format="PNG")
    main.store.save_image(gen, buf.getvalue())
    main.store.set_status(gen.id, "done", None)
    return main.store.get_generation(gen.id)


def offline(monkeypatch):
    """No request reaches the real Draw Things, whatever the job decides."""
    async def refuse(*a, **k):
        raise RuntimeError("stop here: no rendering in tests")

    for name in ("enlarge", "img2img", "txt2img", "renders"):
        monkeypatch.setattr(main.dt, name, refuse)


@pytest.fixture
def edited(client):
    final = made("final", DRAWER, SCENE, neg="lowres, bad hands")
    return final, made("edit", EDITOR, ORDER, parent=final, session=final.session_id)


def test_upscale_of_an_edit_is_drawn_by_the_checkpoint_not_the_editor(client, edited):
    final, edit = edited
    r = client.post(f"/api/generations/{edit.id}/upscale", json={"size": 2048})
    assert r.status_code == 200, r.text
    g = main.store.get_generation(r.json()["id"])
    assert g.model_key == DRAWER
    assert g.prompt == SCENE and ORDER not in g.prompt
    assert "lowres, bad hands" in g.negative_prompt
    assert g.parent_id == edit.id            # still the edited picture it enlarges
    assert not g.describe_first


def test_the_editor_is_not_offered_for_its_own_pictures(client, edited):
    _, edit = edited
    assert EDITOR not in main.registry.upscale_candidates(EDITOR)
    r = client.post(f"/api/generations/{edit.id}/upscale", json={"via": EDITOR, "size": 2048})
    assert r.status_code == 400


def test_another_checkpoint_picked_for_an_edit_still_gets_no_order(client, edited):
    _, edit = edited
    r = client.post(f"/api/generations/{edit.id}/upscale", json={"via": "bas-anima", "size": 2048})
    assert r.status_code == 200, r.text
    g = main.store.get_generation(r.json()["id"])
    assert g.model_key == "bas-anima" and g.prompt == SCENE


def test_the_walk_goes_through_edits_of_edits_and_a_2x_fast(client, edited):
    final, edit = edited
    again = made("edit", EDITOR, "make the sky pink", parent=edit, session=final.session_id)
    fast = made("esrgan", EDITOR, again.prompt, parent=again, session=final.session_id, size=1536)
    assert main.drawn_by(fast).id == final.id
    r = client.post(f"/api/generations/{fast.id}/upscale", json={"size": 2048})
    assert r.status_code == 200, r.text
    assert main.store.get_generation(r.json()["id"]).prompt == SCENE


def test_an_edit_with_nothing_above_it_is_refused_with_a_reason(client):
    orphan = made("edit", EDITOR, ORDER)
    r = client.post(f"/api/generations/{orphan.id}/upscale", json={"size": 2048})
    assert r.status_code == 400
    assert "which model drew it" in r.json()["detail"]


def test_a_plain_final_upscales_as_before(client):
    final = made("final", DRAWER, SCENE)
    r = client.post(f"/api/generations/{final.id}/upscale", json={"size": 2048})
    g = main.store.get_generation(r.json()["id"])
    assert (g.model_key, g.prompt, g.describe_first) == (DRAWER, SCENE, False)


def test_fix_faces_on_an_edit_uses_the_checkpoint_and_its_prompt(client, edited):
    _, edit = edited
    r = client.post(f"/api/generations/{edit.id}/detail", json={"box": [100, 100, 400, 400]})
    assert r.status_code == 200, r.text
    g = main.store.get_generation(r.json()["id"])
    assert (g.model_key, g.prompt) == (DRAWER, SCENE)


def test_more_like_an_edit_drafts_the_prompt_it_was_edited_from(client, edited, monkeypatch):
    monkeypatch.setattr(main, "job_queue", SimpleNamespace(put_nowait=lambda x: None))
    _, edit = edited
    r = client.post(f"/api/generations/{edit.id}/more", json={"count": 2})
    assert r.status_code == 200, r.text
    for d in r.json():
        assert d["model_key"] == DRAWER
        assert ORDER not in d["prompt"] and "ornate silver headdress" in d["prompt"]


async def test_finalize_plus_keeps_its_hint(client, monkeypatch):
    draft = made("draft", DRAWER, SCENE, size=512)
    r = client.post(f"/api/generations/{draft.id}/finalize", json={"mode": "describe"})
    assert r.status_code == 200, r.text
    seen = {}

    async def describe(image_b64, style, uses_neg, hint, *a, **k):
        seen["hint"] = hint
        raise RuntimeError("stop here")

    async def is_up():
        return True

    monkeypatch.setattr(main.llm, "describe", describe)
    monkeypatch.setattr(main.dt, "is_up", is_up)
    offline(monkeypatch)
    monkeypatch.setattr(main, "notify", lambda *a, **k: None)
    await main.run_job(r.json()["id"])
    assert SCENE in seen["hint"] and "edited" not in seen["hint"]


# -- the job folds the edits into the prompt ------------------------------------

def merging(monkeypatch, result="MERGED", fail=False):
    """Stub the LLM merge and Draw Things; returns what each was asked."""
    seen = {"refine": [], "rendered": []}

    async def refine(base, tweak, k, *a, **kw):
        seen["refine"].append((base, tweak))
        if fail:
            raise RuntimeError("ollama is not running")
        return [SimpleNamespace(prompt=result, negative_prompt="")]

    async def is_up():
        return True

    async def enlarge(init, **kw):
        return init

    async def render(init, strength, **kw):
        seen["rendered"].append(kw["prompt"])
        raise RuntimeError("stop here: no rendering in tests")

    monkeypatch.setattr(main.llm, "refine", refine)
    monkeypatch.setattr(main.dt, "is_up", is_up)
    offline(monkeypatch)
    monkeypatch.setattr(main.dt, "enlarge", enlarge)
    monkeypatch.setattr(main.dt, "img2img", render)
    monkeypatch.setattr(main, "notify", lambda *a, **k: None)
    return seen


async def test_the_job_merges_the_order_into_the_drawn_prompt(client, edited, monkeypatch):
    _, edit = edited
    gen_id = client.post(f"/api/generations/{edit.id}/upscale", json={"size": 2048}).json()["id"]
    seen = merging(monkeypatch)
    await main.run_job(gen_id)
    # the guard the app appends to every order is not part of what was asked
    assert seen["refine"] == [(SCENE, "replace the head piece with jewelry.")]
    assert main.store.get_generation(gen_id).prompt == "MERGED"
    assert seen["rendered"] == ["MERGED"]    # and that is what the pass was sent


async def test_a_retry_merges_from_the_drawn_prompt_again(client, edited, monkeypatch):
    _, edit = edited
    gen_id = client.post(f"/api/generations/{edit.id}/upscale", json={"size": 2048}).json()["id"]
    seen = merging(monkeypatch)
    await main.run_job(gen_id)
    await main.run_job(gen_id)
    assert [base for base, _ in seen["refine"]] == [SCENE, SCENE]


async def test_edits_of_edits_are_merged_oldest_first(client, edited, monkeypatch):
    final, edit = edited
    again = made("edit", EDITOR, "make the sky pink. " + main.EDIT_GUARD,
                 parent=edit, session=final.session_id)
    gen_id = client.post(f"/api/generations/{again.id}/upscale", json={"size": 2048}).json()["id"]
    seen = merging(monkeypatch)
    await main.run_job(gen_id)
    assert seen["refine"] == [
        (SCENE, "replace the head piece with jewelry. Then: make the sky pink.")]


async def test_without_the_llm_the_drawn_prompt_is_rendered(client, edited, monkeypatch):
    _, edit = edited
    gen_id = client.post(f"/api/generations/{edit.id}/upscale", json={"size": 2048}).json()["id"]
    seen = merging(monkeypatch, fail=True)
    await main.run_job(gen_id)
    assert main.store.get_generation(gen_id).prompt == SCENE
    assert seen["rendered"] == [SCENE]       # it went on to render, not held


async def test_a_plain_upscale_asks_the_llm_nothing(client, monkeypatch):
    final = made("final", DRAWER, SCENE)
    gen_id = client.post(f"/api/generations/{final.id}/upscale", json={"size": 2048}).json()["id"]
    seen = merging(monkeypatch)
    await main.run_job(gen_id)
    assert seen["refine"] == [] and main.store.get_generation(gen_id).prompt == SCENE


def test_a_combine_contributes_its_words_and_its_subject(client, edited):
    from app.detailer import combine_prompt
    final, _ = edited
    subject = made("final", "bas-anima", "1girl, emerald hair, white dress",
                   session=final.session_id)
    comb = main.store.create_generation(
        session_id=final.session_id, kind="edit", model_key=EDITOR,
        prompt=combine_prompt("put her on the steps"), negative_prompt="", seed=1,
        width=896, height=896, steps=24, cfg=1.0, sampler=None,
        parent_id=final.id, source_id=subject.id,
    )
    assert main.edit_orders(comb) == [
        "put her on the steps. (the subject: 1girl, emerald hair, white dress)"]
