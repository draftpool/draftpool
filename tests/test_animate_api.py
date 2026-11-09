"""What the Animate endpoint queues for a model that is not on Wan's grid.

MiniMax H3 renders 17k+5 frames at 24 fps; Wan renders 4k+1 at 16. The page
used to send fps 16 and Wan's lengths for every model, so the server is what
holds a clip to its own model's grid and rate, whatever an open page sends.
"""

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import main
from app.registry import CONFIG_PATH, load_registry
from app.sessions import Store


@pytest.fixture
def client(tmp_path, monkeypatch):
    reg = load_registry(CONFIG_PATH)
    for spec in (*reg.models.values(), *reg.video_models.values()):
        spec.installed = True
    monkeypatch.setattr(main, "registry", reg)
    monkeypatch.setattr(main, "store", Store(tmp_path))
    # registry.resolve_installed() would rescan the real Draw Things folder
    monkeypatch.setattr(type(reg), "resolve_installed", lambda self: None)
    # no lifespan: the worker would try to render the queued clip
    return TestClient(main.app)


def a_still(client):
    sid = client.post("/api/sessions", json={"instruction": "a cat"}).json()["id"]
    gen = main.store.create_generation(
        session_id=sid, kind="final", model_key="juggernaut-ragnarok", prompt="a cat",
        negative_prompt="", seed=1, width=1024, height=1024, steps=20, cfg=4.0,
        sampler="DPM++ 2M Karras",
    )
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (30, 60, 90)).save(buf, format="PNG")
    main.store.save_image(gen, buf.getvalue())
    main.store.set_status(gen.id, "done", None)
    return gen


def test_h3_clip_is_snapped_to_its_grid_at_its_own_fps(client):
    g = a_still(client)
    # an old page: Wan's 2s length and Wan's rate
    r = client.post(f"/api/generations/{g.id}/animate",
                    json={"model_key": "minimax-h3", "num_frames": 49, "fps": 16})
    assert r.status_code == 200, r.text
    clip = r.json()
    assert clip["fps"] == 24
    assert (clip["num_frames"] - 5) % 17 == 0
    assert clip["num_frames"] == 56
    assert clip["sampler"] == "DDIM Trailing"
    assert [l["file"] for l in clip["loras"]]


def test_fusionx_clip_is_unchanged(client):
    g = a_still(client)
    r = client.post(f"/api/generations/{g.id}/animate",
                    json={"model_key": "fusionx", "num_frames": 33, "fps": 16})
    clip = r.json()
    assert (clip["num_frames"], clip["fps"]) == (33, 16)
    assert clip["sampler"] is None and not clip["loras"]


def test_video_models_offer_each_model_s_own_lengths(client):
    by_key = {m["key"]: m for m in client.get("/api/video-models").json()}
    assert by_key["fusionx"]["presets"] == [17, 33, 49, 81]
    assert by_key["minimax-h3"]["presets"] == [22, 56, 73, 124]
    assert by_key["minimax-h3"]["fps"] == 24
    assert set(by_key["minimax-h3"]["est_minutes"]) == {"22", "56", "73", "124"}


def test_a_request_past_the_longest_clip_is_capped(client):
    """The request may ask for up to 241 frames so H3's 124 fits; FusionX used
    to be held at 81 by the request itself, and must still be."""
    g = a_still(client)
    fx = client.post(f"/api/generations/{g.id}/animate",
                     json={"model_key": "fusionx", "num_frames": 241}).json()
    h3 = client.post(f"/api/generations/{g.id}/animate",
                     json={"model_key": "minimax-h3", "num_frames": 241}).json()
    assert fx["num_frames"] == 81
    assert h3["num_frames"] == 124


def test_motion_prompt_is_planned_for_the_models_own_clip_length(client, monkeypatch):
    """With the model named, the seconds the LLM plans for come from its grid
    and fps; without it (an older page), from the request as before."""
    seen = []

    async def motion(prompt, parent_prompt, image_b64, seconds):
        seen.append(seconds)
        from app.schemas import PromptVariant
        return PromptVariant(prompt="hair drifts")

    async def no_vision():
        return False

    monkeypatch.setattr(main.llm, "motion", motion)
    monkeypatch.setattr(main.llm, "has_vision", no_vision)
    g = a_still(client)
    client.post(f"/api/generations/{g.id}/animate-prompt",
                json={"num_frames": 56, "fps": 16, "model_key": "minimax-h3"})
    client.post(f"/api/generations/{g.id}/animate-prompt", json={"num_frames": 33, "fps": 16})
    assert seen[0] == pytest.approx(56 / 24)
    assert seen[1] == pytest.approx(33 / 16)


async def test_an_h3_clip_renders_with_the_recipe_it_was_queued_with(client, monkeypatch):
    """The render reads the card, not the live config: a Retry repeats it."""
    import base64
    import io as _io

    from PIL import Image as _Image

    sent = {}

    def png():
        buf = _io.BytesIO()
        _Image.new("RGB", (8, 8)).save(buf, format="PNG")
        return buf.getvalue()

    async def img2video(init_png, **kw):
        sent.update(kw)
        return [png(), png()]

    async def ready(gen):
        return None

    monkeypatch.setattr(main.dt, "img2video", img2video)
    monkeypatch.setattr(main, "wait_for_draw_things", ready)
    g = a_still(client)
    clip = client.post(f"/api/generations/{g.id}/animate",
                       json={"model_key": "minimax-h3", "num_frames": 22}).json()
    spec = main.registry.video_models["minimax-h3"]
    # the config changes after queueing; the card's recipe must still be used
    monkeypatch.setattr(spec, "sampler", "Euler a")
    monkeypatch.setattr(spec, "loras", [])
    await main.run_video_job(main.store.get_generation(clip["id"]))
    assert main.store.get_generation(clip["id"]).status == "done"
    assert sent["sampler"] == "DDIM Trailing"
    assert [l["file"] for l in sent["loras"]] == [clip["loras"][0]["file"]]
    assert sent["shift_for_audio"] == spec.shift_for_audio and sent["fps"] == 24
    assert sent["num_frames"] == 22
