"""Combine: one picture's subject put into another picture's scene.

The instruction editor gets both pictures as one two-panel frame (Draw Things'
HTTP API takes a single init image), and the scene panel of the result is the
new picture. The new picture is an edit of the scene with the subject's
picture as a second origin, and that second link has to hold a draft the same
way parent_id does.
"""

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import main
from app.detailer import build_combine_frame, combine, combine_layout, combine_prompt
from app.sessions import Store


def png(w, h, color):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color).save(buf, format="PNG")
    return buf.getvalue()


# -- the frame ---------------------------------------------------------------

@pytest.mark.parametrize("scene,subject", [
    ((1536, 1536), (1536, 1536)),
    ((1792, 1344), (1024, 1024)),   # landscape scene
    ((768, 1344), (1024, 1024)),    # portrait scene
    ((2048, 1152), (768, 1344)),    # wide scene, tall subject
    ((1024, 1024), (2048, 512)),    # a strip of a subject
])
def test_layout_is_on_the_grid_and_inside_the_budget(scene, subject):
    area = 1344 * 896
    sw, bw, h = combine_layout(scene, subject, area=area)
    assert sw % 64 == 0 and bw % 64 == 0 and h % 64 == 0
    assert (sw + bw) * h <= area
    assert 0 < sw <= bw
    # the scene panel keeps the scene's shape to within one grid step
    assert abs(bw / h - scene[0] / scene[1]) <= 64 / h


def test_the_scene_gets_the_pixels():
    # a square scene and subject at half scale: the scene is two thirds of the
    # frame, which is the point -- it is the picture that comes back
    sw, bw, h = combine_layout((1024, 1024), (1024, 1024), area=1344 * 896, subject_scale=0.5)
    assert (sw, bw, h) == (448, 896, 896)


def test_frame_puts_the_subject_left_and_the_scene_right():
    frame, scene_w, height = build_combine_frame(
        png(1024, 1024, (0, 0, 255)), png(512, 512, (255, 0, 0)),
        area=1344 * 896, subject_scale=0.5,
    )
    img = Image.open(io.BytesIO(frame)).convert("RGB")
    assert img.size == (448 + 896, 896) and (scene_w, height) == (896, 896)
    assert img.getpixel((448 + 10, 10)) == (0, 0, 255)          # scene, every corner
    assert img.getpixel((img.width - 1, img.height - 1)) == (0, 0, 255)
    assert img.getpixel((224, 448)) == (255, 0, 0)              # subject, centred
    assert img.getpixel((224, 10)) == (255, 255, 255)           # padded above it


def test_a_wide_subject_is_fitted_whole_not_cropped():
    subject = Image.new("RGB", (2000, 500), (0, 255, 0))
    subject.paste((255, 0, 0), (0, 0, 20, 500))                 # a red left edge
    buf = io.BytesIO(); subject.save(buf, format="PNG")
    frame, scene_w, height = build_combine_frame(png(1024, 1024, (0, 0, 0)), buf.getvalue())
    img = Image.open(io.BytesIO(frame)).convert("RGB")
    row = height // 2
    assert img.getpixel((1, row))[0] > 200                      # the red edge survived


def test_prompt_ends_the_users_sentence_before_its_own():
    # a real render: "...in front of the café Light what comes..." when the
    # instruction had no full stop
    p = combine_prompt("put her in front of the café")
    assert "in front of the café. " in p
    assert "in front of the café.. " not in combine_prompt("put her in front of the café.")
    assert "look! " in combine_prompt("look!")


def test_prompt_asks_for_the_subject_to_stay_itself():
    # users name where and how, not "and keep her face"; without it the dress
    # and the eyes changed on the way across
    p = combine_prompt("put her on the steps")
    assert "face" in p and "clothing" in p


def test_prompt_keeps_the_look_but_not_the_pose():
    # "keep her face, hair and clothing as they are in the left panel" also
    # kept her sitting, with "standing" in the instruction; the pose has to be
    # handed to the instruction explicitly
    p = combine_prompt("put her on the steps")
    sentence = p.split("Keep the subject's", 1)[1].split(".", 1)[0]
    kept, _, handed_over = sentence.partition(" but ")
    assert "pose" not in kept
    assert "pose" in handed_over and "instruction" in handed_over


def test_prompt_names_the_panels_and_carries_the_words():
    p = combine_prompt("  put the woman on the bench, sitting  ")
    assert "put the woman on the bench, sitting" in p
    assert "left panel" in p and "right panel" in p
    assert "unchanged" in p


async def test_render_returns_the_scene_panel_only():
    frame, scene_w, height = build_combine_frame(
        png(1024, 1024, (0, 0, 255)), png(1024, 1024, (255, 0, 0)))
    seen = {}

    class DT:
        async def img2img(self, init, strength, *, width, height, **kw):
            seen.update(width=width, height=height, strength=strength, **kw)
            out = Image.new("RGB", (width, height), (255, 255, 255))
            out.paste((0, 255, 0), (width - scene_w, 0, width, height))
            b = io.BytesIO(); out.save(b, format="PNG")
            return b.getvalue()

    out = await combine(DT(), frame, scene_w=scene_w, prompt="p", steps=24)
    img = Image.open(io.BytesIO(out)).convert("RGB")
    assert img.size == (scene_w, height)
    assert img.getcolors() == [(scene_w * height, (0, 255, 0))]   # no subject column
    assert seen["strength"] == 1.0 and seen["steps"] == 24
    assert (seen["width"], seen["height"]) == Image.open(io.BytesIO(frame)).size


# -- the endpoint and the job -------------------------------------------------

@pytest.fixture
def store(tmp_path, monkeypatch):
    s = Store(data_dir=tmp_path)
    monkeypatch.setattr(main, "store", s)
    return s


@pytest.fixture
def client(store, monkeypatch):
    monkeypatch.setattr(main, "job_queue", type("Q", (), {"put_nowait": lambda self, x: None})())
    # the editor has to count as installed whatever this machine has
    monkeypatch.setattr(main.registry.models["qwen-edit"], "installed", True)
    with TestClient(main.app) as c:
        yield c


def picture(store, session_id, color, kind="final", size=(1024, 1024), status="done"):
    gen = store.create_generation(
        session_id=session_id, kind=kind, model_key="m", prompt=f"{kind} prompt",
        negative_prompt="", seed=7, width=size[0], height=size[1], steps=4,
        cfg=1.0, sampler=None,
    )
    if status == "done":
        store.save_image(gen, png(*size, color))
    store.set_status(gen.id, status)
    return store.get_generation(gen.id)


def test_combine_queues_an_edit_of_the_scene(client, store):
    s = store.create_session("x")
    scene = picture(store, s.id, (0, 0, 255), size=(1344, 1024))
    subject = picture(store, s.id, (255, 0, 0))
    r = client.post("/api/combine", json={
        "scene_id": scene.id, "subject_id": subject.id,
        "prompt": "put her on the steps", "model_key": "qwen-edit"})
    assert r.status_code == 200, r.text
    g = store.get_generation(r.json()["id"])
    assert (g.kind, g.parent_id, g.source_id) == ("edit", scene.id, subject.id)
    # no steps asked for: the editor's own, which is what its LoRA is built for
    assert g.steps == main.registry.models["qwen-edit"].final.steps
    assert g.strength == 1.0
    assert "put her on the steps" in g.prompt
    # the picture that comes back is the scene panel, and the frame is on disk
    frame = Image.open(store.init_path(g))
    assert frame.height == g.height and frame.width > g.width
    assert abs(g.width / g.height - 1344 / 1024) <= 64 / g.height


@pytest.mark.parametrize("case", ["same", "pending", "video", "other_session", "not_editor"])
def test_combine_refuses(client, store, case, monkeypatch):
    s = store.create_session("x")
    scene = picture(store, s.id, (0, 0, 255))
    subject = picture(store, s.id, (255, 0, 0))
    body = {"scene_id": scene.id, "subject_id": subject.id, "prompt": "p",
            "model_key": "qwen-edit"}
    if case == "same":
        body["subject_id"] = scene.id
    elif case == "pending":
        body["subject_id"] = picture(store, s.id, None, status="pending").id
    elif case == "video":
        body["subject_id"] = picture(store, s.id, (1, 1, 1), kind="video").id
    elif case == "other_session":
        body["subject_id"] = picture(store, store.create_session("y").id, (1, 1, 1)).id
    elif case == "not_editor":
        key = next(k for k, m in main.registry.models.items() if not m.edit_only)
        monkeypatch.setattr(main.registry.models[key], "installed", True)
        body["model_key"] = key
    r = client.post("/api/combine", json=body)
    assert r.status_code == 400, r.text
    assert len(store.get_session(s.id).generations) == (
        3 if case in ("pending", "video") else 2)


async def test_job_renders_the_frame_and_saves_the_scene_panel(store, monkeypatch):
    monkeypatch.setattr(main.registry.models["qwen-edit"], "installed", True)
    monkeypatch.setattr(main, "notify", lambda *a, **k: None)
    s = store.create_session("x")
    scene = picture(store, s.id, (0, 0, 255))
    subject = picture(store, s.id, (255, 0, 0))
    frame, scene_w, height = build_combine_frame(
        store.image_path(scene).read_bytes(), store.image_path(subject).read_bytes())
    gen = store.create_generation(
        session_id=s.id, kind="edit", model_key="qwen-edit", prompt="p",
        negative_prompt="", seed=1, width=scene_w, height=height, steps=24,
        cfg=1.0, sampler=None, parent_id=scene.id, source_id=subject.id, strength=1.0,
    )
    store.init_path(gen).write_bytes(frame)
    seen = {}

    async def is_up():
        return True

    async def img2img(init, strength, *, width, height, timeout=None, **kw):
        seen.update(init=init, size=(width, height), timeout=timeout)
        out = Image.new("RGB", (width, height), (255, 255, 255))
        out.paste((0, 255, 0), (width - scene_w, 0, width, height))
        b = io.BytesIO(); out.save(b, format="PNG")
        return b.getvalue()

    monkeypatch.setattr(main.dt, "is_up", is_up)
    monkeypatch.setattr(main.dt, "img2img", img2img)
    await main.run_job(gen.id)
    done = store.get_generation(gen.id)
    assert done.status == "done", done.error
    assert seen["init"] == frame                     # the frame, not the scene alone
    assert seen["size"] == Image.open(io.BytesIO(frame)).size
    assert seen["timeout"] >= main.REGION_INSTRUCT_TIMEOUT
    img = Image.open(store.image_path(done)).convert("RGB")
    assert img.size == (scene_w, height)
    assert img.getcolors() == [(scene_w * height, (0, 255, 0))]


# -- the second origin holds drafts like the first ----------------------------

def link(store, s, scene, subject, status="done"):
    g = store.create_generation(
        session_id=s.id, kind="edit", model_key="qwen-edit", prompt="p",
        negative_prompt="", seed=1, width=256, height=256, steps=4, cfg=1.0,
        sampler=None, parent_id=scene.id, source_id=subject.id,
    )
    store.save_image(g, b"\x89PNG")
    store.set_status(g.id, status)
    return g


def test_a_draft_used_as_the_subject_is_pruned_not_deleted(store):
    s = store.create_session("x")
    scene = picture(store, s.id, (0, 0, 255))
    subject = picture(store, s.id, (255, 0, 0), kind="draft")
    link(store, s, scene, subject)
    done = store.prune_drafts(s.id)
    assert done["deleted"] == [] and done["pruned"] == [subject.id]
    assert store.get_generation(subject.id).pruned_at


def test_a_draft_used_as_the_subject_of_a_queued_combine_is_kept(store):
    s = store.create_session("x")
    scene = picture(store, s.id, (0, 0, 255))
    subject = picture(store, s.id, (255, 0, 0), kind="draft")
    link(store, s, scene, subject, status="pending")
    done = store.prune_drafts(s.id)
    assert [k["id"] for k in done["kept"]] == [subject.id]


def test_deleting_the_combine_takes_its_pruned_subject_stub_with_it(store):
    s = store.create_session("x")
    scene = picture(store, s.id, (0, 0, 255))
    subject = picture(store, s.id, (255, 0, 0), kind="draft")
    g = link(store, s, scene, subject)
    store.prune_drafts(s.id)
    store.delete_generation(g.id)
    assert store.get_generation(subject.id) is None
    assert store.get_generation(scene.id) is not None


def test_combine_steps_asked_for_win(client, store):
    s = store.create_session("x")
    scene = picture(store, s.id, (0, 0, 255))
    subject = picture(store, s.id, (255, 0, 0))
    r = client.post("/api/combine", json={
        "scene_id": scene.id, "subject_id": subject.id,
        "prompt": "put her on the steps", "model_key": "qwen-edit", "steps": 12})
    assert r.status_code == 200, r.text
    assert store.get_generation(r.json()["id"]).steps == 12
