"""Optional LoRAs: a tick per model, and what each picture was rendered with.

A LoRA option carries its own recipe -- an accelerator is useless without its
steps and cfg -- so most of what is worth asserting is that ticking one moves
all of it together, unticking puts back exactly what ships, a number the user
typed still wins, and a card queued under one setting renders under that
setting whatever the tick says by the time it runs.
"""

import io
import json

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import main
from app.registry import CONFIG_PATH, load_registry
from app.schemas import LoraOption, LoraSpec, ModelSpec, StageRecipe, StageSpec
from app.sessions import Store
from app.settings import SettingsStore

TURBO = "turbo.ckpt"
FIXED = "fixed.ckpt"


def spec(**extra) -> ModelSpec:
    return ModelSpec(
        key="m", label="M", dt_filename="m.ckpt", cfg=4.5, sampler="Euler a",
        uses_negative_prompt=True, prompt_style="tags",
        draft=StageSpec(size=640, steps=12, cfg=4.0, loras=[LoraSpec(file=FIXED, weight=0.6)]),
        final=StageSpec(size=1536, steps=24, strength=0.5),
        upscale=StageSpec(size=2048, steps=18),
        lora_options=[LoraOption(
            key="turbo", label="Turbo", file=TURBO, negative_prompt=False,
            opener="anime coloring",
            stages={"draft": StageRecipe(size=1024, steps=8, cfg=1.0, sampler="Euler A Trailing"),
                    "final": StageRecipe(steps=8, cfg=1.0)},
        )],
        **extra,
    )


@pytest.fixture
def reg():
    r = load_registry(CONFIG_PATH)
    m = spec()
    for stage in (m.draft, m.final, m.upscale):
        stage.default_steps = stage.steps
    r.models = {"m": m}
    r.apply_lora_choices({})
    return r


# -- the store ----------------------------------------------------------------


def test_only_ticks_set_by_hand_are_stored(tmp_path):
    s = SettingsStore(tmp_path)
    assert s.lora_choices() == {}
    s.set_lora_choice("m", "turbo", False)
    assert s.lora_choices() == {"m": {"turbo": False}}
    # None is "back to the shipped default", and an emptied model is dropped
    s.set_lora_choice("m", "turbo", None)
    assert s.lora_choices() == {}
    assert json.loads(s.path.read_text())["loras"] == {}


def test_a_hand_edited_non_bool_is_ignored(tmp_path):
    s = SettingsStore(tmp_path)
    s.path.write_text(json.dumps({"loras": {"m": {"turbo": "yes", "x": True}, "n": 3}}))
    assert s.lora_choices() == {"m": {"x": True}}


# -- the registry -------------------------------------------------------------


def test_a_default_option_brings_its_whole_recipe(reg):
    m = reg.models["m"]
    assert (m.draft.size, m.draft.steps, m.draft.cfg, m.draft.sampler) == (
        1024, 8, 1.0, "Euler A Trailing")
    assert (m.final.steps, m.final.cfg) == (8, 1.0)
    # added to the fixed LoRA, not in place of it; upscale is not named, so untouched
    assert [l.file for l in m.draft.loras] == [FIXED, TURBO]
    assert [l.file for l in m.final.loras] == [TURBO]
    assert m.upscale.loras == [] and m.upscale.steps == 18
    # CFG 1 has no negative pass
    assert m.uses_negative_prompt is False and m.default_negative == ""
    # the LoRA's own tags ride on the model's opener
    assert m.quality_opener.endswith(", anime coloring")
    assert m.faithful_steps == 16


def test_unticking_puts_back_exactly_what_ships(reg):
    reg.apply_lora_choices({"m": {"turbo": False}})
    m = reg.models["m"]
    assert (m.draft.size, m.draft.steps, m.draft.cfg, m.draft.sampler) == (640, 12, 4.0, None)
    assert [l.file for l in m.draft.loras] == [FIXED]
    assert (m.final.steps, m.final.cfg, m.final.loras) == (24, None, [])
    assert m.uses_negative_prompt is True
    assert "anime coloring" not in m.quality_opener
    # and ticking again is idempotent: the LoRA is not appended twice
    reg.apply_lora_choices({})
    reg.apply_lora_choices({})
    assert [l.file for l in m.draft.loras] == [FIXED, TURBO]
    assert m.quality_opener.count("anime coloring") == 1


def test_a_typed_step_count_wins_over_the_recipe_and_survives_the_tick(reg):
    reg.apply_step_overrides({"m": {"final": 12}})
    m = reg.models["m"]
    assert m.final.steps == 12 and m.final.default_steps == 8
    reg.apply_lora_choices({"m": {"turbo": False}})
    assert m.final.steps == 12 and m.final.default_steps == 24
    reg.apply_step_overrides({})
    assert m.final.steps == 24


def test_a_missing_optional_lora_leaves_the_model_usable(reg, monkeypatch):
    monkeypatch.setattr("app.registry.installed_files", lambda dirs: {"m.ckpt", FIXED})
    reg.resolve_installed()
    m = reg.models["m"]
    assert m.available
    assert m.lora_options[0].available is False and m.lora_options[0].on is False
    assert m.final.steps == 24 and m.final.loras == []


def test_a_missing_fixed_lora_still_hides_the_model(reg, monkeypatch):
    monkeypatch.setattr("app.registry.installed_files", lambda dirs: {"m.ckpt", TURBO})
    reg.resolve_installed()
    assert not reg.models["m"].available


def test_shipped_options_name_real_stages_and_default_on():
    r = load_registry(CONFIG_PATH)
    edit = r.models["qwen-edit"]
    assert edit.final.steps == 4 and [l.file for l in edit.final.loras] == [
        "qwen_image_edit_2511_lightning_4_step_v1.0_lora_f16.ckpt"]
    r.apply_lora_choices({"qwen-edit": {"lightning": False}})
    assert edit.final.steps == 16 and edit.final.loras == []


# -- the API and the record ----------------------------------------------------


@pytest.fixture
def client(tmp_path, monkeypatch, reg):
    # /api/models rescans the model folders on every call
    monkeypatch.setattr("app.registry.installed_files",
                        lambda dirs: {"m.ckpt", FIXED, TURBO})
    monkeypatch.setattr(main, "registry", reg)
    monkeypatch.setattr(main, "store", Store(tmp_path))
    monkeypatch.setattr(main, "settings_store", SettingsStore(tmp_path))
    monkeypatch.setattr(main, "job_queue",
                        type("Q", (), {"put_nowait": lambda self, x: None})())
    with TestClient(main.app) as c:
        yield c


def test_models_list_carries_the_options(client):
    m = next(m for m in client.get("/api/models").json() if m["key"] == "m")
    assert m["lora_options"] == [{"key": "turbo", "label": "Turbo", "on": True,
                                  "default": True, "available": True,
                                  "stages": ["draft", "final"]}]


def test_the_tick_is_saved_and_the_reply_says_what_changed(client):
    r = client.put("/api/models/m/loras/turbo", json={"on": False})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["final"] == {"steps": 24, "default": 24}
    assert body["uses_negative_prompt"] is True
    assert body["lora_options"][0]["on"] is False
    assert main.settings_store.lora_choices() == {"m": {"turbo": False}}
    assert client.put("/api/models/m/loras/nope", json={"on": True}).status_code == 404


def test_a_draft_records_its_loras_and_keeps_them_past_the_tick(client):
    sid = main.store.create_session("x").id
    r = client.post(f"/api/sessions/{sid}/drafts", json={
        "variants": [{"prompt": "1girl", "negative_prompt": ""}],
        "model_keys": ["m"], "seeds_per_variant": 1})
    assert r.status_code == 200, r.text
    gen = main.store.get_generation(r.json()[0]["id"])
    assert [(l.file, l.weight) for l in gen.loras] == [(FIXED, 0.6), (TURBO, 1.0)]
    assert (gen.steps, gen.cfg) == (8, 1.0)
    # the tick changes after it was queued; the card does not
    client.put("/api/models/m/loras/turbo", json={"on": False})
    assert [l.file for l in main.store.get_generation(gen.id).loras] == [FIXED, TURBO]


def a_done_draft(client):
    sid = main.store.create_session("x").id
    gen = client.post(f"/api/sessions/{sid}/drafts", json={
        "variants": [{"prompt": "1girl", "negative_prompt": "bad"}],
        "model_keys": ["m"], "seeds_per_variant": 1}).json()[0]
    main.store.set_status(gen["id"], "done", None)
    return gen


def test_a_final_follows_the_drafts_tick_not_the_chips(client):
    """Flipping the tick between a draft and its final changes nothing.

    The reported case: an untick 640 draft finished by the Turbo recipe's
    8-real-step Faithful is a soft picture (docs/decisions/DRAFT-STEPS.md).
    The chain inherits the state the draft recorded, both ways round.
    """
    ticked = a_done_draft(client)                       # made with Turbo on
    client.put("/api/models/m/loras/turbo", json={"on": False})
    unticked = a_done_draft(client)                     # made with it off
    assert [l["file"] for l in unticked["loras"]] == [FIXED]
    # chip is off now: the ticked draft's Faithful still runs the Turbo recipe
    f = client.post(f"/api/generations/{ticked['id']}/finalize", json={"mode": "faithful"}).json()
    assert ([l["file"] for l in f["loras"]], f["steps"], f["cfg"]) == ([TURBO], 16, 1.0)
    assert f["negative_prompt"] == ""                   # CFG 1: no negative
    # chip back on: the untick draft's Faithful and Finalize stay the model's own
    client.put("/api/models/m/loras/turbo", json={"on": True})
    f = client.post(f"/api/generations/{unticked['id']}/finalize", json={"mode": "faithful"}).json()
    assert (f["loras"], f["steps"], f["cfg"]) == ([], 48, 4.5)
    assert f["negative_prompt"] == "bad"
    r = client.post(f"/api/generations/{unticked['id']}/finalize", json={"mode": "regen"}).json()
    assert (r["loras"], r["steps"]) == ([], 24)


def test_a_row_from_before_the_record_takes_the_live_tick(client):
    gen = a_done_draft(client)
    # nothing writes that shape any more, so put it there by hand
    with main.store._connect() as c:
        c.execute("update generations set loras = NULL where id = ?", (gen["id"],))
    assert main.store.get_generation(gen["id"]).loras is None
    client.put("/api/models/m/loras/turbo", json={"on": False})
    f = client.post(f"/api/generations/{gen['id']}/finalize", json={"mode": "faithful"}).json()
    assert (f["loras"], f["steps"]) == ([], 48)


async def test_a_render_sends_the_recorded_loras_not_the_current_ones(client, monkeypatch):
    sid = main.store.create_session("x").id
    gen = main.store.create_generation(
        session_id=sid, kind="draft", model_key="m", prompt="1girl",
        negative_prompt="", seed=1, width=640, height=640, steps=8, cfg=1.0,
        sampler=None, loras=[{"file": TURBO, "weight": 0.9}],
    )
    sent = {}

    async def txt2img(**kw):
        sent.update(kw)
        buf = io.BytesIO()
        Image.new("RGB", (640, 640), (200, 100, 50)).save(buf, format="PNG")
        return buf.getvalue()

    async def up():
        return True

    monkeypatch.setattr(main.dt, "txt2img", txt2img)
    monkeypatch.setattr(main.dt, "is_up", up)
    main.registry.apply_lora_choices({"m": {"turbo": False}})
    await main.run_job(gen.id)
    assert [(l["file"], l["weight"]) for l in sent["loras"]] == [(TURBO, 0.9)]
    assert main.store.get_generation(gen.id).status == "done"


def test_a_row_from_before_the_record_has_none(tmp_path):
    store = Store(tmp_path)
    sid = store.create_session("x").id
    gen = store.create_generation(
        session_id=sid, kind="draft", model_key="m", prompt="p", negative_prompt="",
        seed=1, width=64, height=64, steps=4, cfg=1.0, sampler=None,
    )
    assert store.get_generation(gen.id).loras is None


def test_a_queued_prompt_carries_the_lora_opener_once():
    r = load_registry(CONFIG_PATH)
    lustrous = r.models["lustrousmix-anima"]
    main_registry = main.registry
    try:
        main.registry = r
        # an enriched prompt that already opens with the plain opener gets the
        # tag added, and one that already carries it is not doubled
        plain = main.prompt_for(lustrous, "masterpiece, best quality, 1girl, solo")
        assert plain.startswith("masterpiece, best quality, anime coloring, 1girl")
        assert main.prompt_for(lustrous, plain) == plain
        r.apply_lora_choices({"lustrousmix-anima": {"turbo": False}})
        assert main.prompt_for(lustrous, "1girl") == "masterpiece, best quality, 1girl"
    finally:
        main.registry = main_registry


def test_a_measurement_registry_has_every_option_unticked():
    """What the sweep scripts load: the models as they ship, no accelerator.

    Found in review: with the defaults applied, a plain step sweep of an Anima
    model rendered the turbo recipe and recorded it as the model's own.
    """
    from app.registry import lora_tag

    plain = load_registry(CONFIG_PATH, lora_options=False)
    for spec in plain.models.values():
        assert not any(o.on for o in spec.lora_options), spec.key
        assert lora_tag(spec) == ""
    lustrous = plain.models["lustrousmix-anima"]
    assert (lustrous.final.steps, lustrous.final.loras) == (24, [])
    assert lustrous.uses_negative_prompt and "anime coloring" not in lustrous.quality_opener
    # the app's own load keeps the shipped ticks, and names them for a filename
    shipped = load_registry(CONFIG_PATH)
    assert lora_tag(shipped.models["lustrousmix-anima"]) == "turbo"
    assert lora_tag(shipped.models["qwen-edit"]) == "lightning"
    qwen = shipped.models["qwen-image"]
    assert (qwen.final.size, qwen.final.steps, qwen.final.cfg, qwen.uses_negative_prompt) == (1024, 4, 1.0, False)
    assert (plain.models["qwen-image"].final.size, plain.models["qwen-image"].final.steps) == (1024, 12)
    # the tick reaches the upscale: its LoRA, cfg and its own 4 steps (measured
    # against the model's 8 in research/qwen-turbo/upscale.json)
    assert (qwen.upscale.cfg, qwen.upscale.steps, [l.file for l in qwen.upscale.loras]) == (
        1.0, 4, ["wuli_qwen_image_2512_turbo_lora_4steps_v1.0_bf16_comfyui_lora_f16.ckpt"])
    assert (plain.models["qwen-image"].upscale.cfg, plain.models["qwen-image"].upscale.loras) == (None, [])


def test_an_installed_variant_of_the_lora_file_is_the_one_sent(reg, monkeypatch):
    """Matched like a model: the quantisation suffix may differ from models.yaml."""
    monkeypatch.setattr("app.registry.installed_files",
                        lambda dirs: {"m.ckpt", "fixed_f16.ckpt", "turbo_lora_f16.ckpt"})
    reg.resolve_installed()
    m = reg.models["m"]
    assert m.available and m.lora_options[0].on
    assert [l.file for l in m.draft.loras] == ["fixed_f16.ckpt", "turbo_lora_f16.ckpt"]
    # and it survives the tick being rebuilt, which starts from the shipped stages
    reg.apply_lora_choices({"m": {"turbo": False}})
    assert [l.file for l in m.draft.loras] == ["fixed_f16.ckpt"]
