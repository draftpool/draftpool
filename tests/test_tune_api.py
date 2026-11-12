"""/api/tune: the wizard's endpoints, with Draw Things answered in-process."""

import io
from types import SimpleNamespace
import time

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import main

KEY = "juggernaut-ragnarok"


def _png() -> bytes:
    out = io.BytesIO()
    Image.effect_noise((64, 64), 64).convert("RGB").save(out, "PNG")
    return out.getvalue()


@pytest.fixture
def client(monkeypatch):
    sent = []

    real = main.dt._build_payload

    async def txt2img(**kw):
        real(**kw)  # what Draw Things would be sent: a misspelt argument fails here
        sent.append(kw)
        return _png()

    written = []
    monkeypatch.setattr(main, "_tunable_keys", lambda: {KEY})
    monkeypatch.setattr(main.dt, "txt2img", txt2img)
    monkeypatch.setattr(main.tune, "write_tuning",
                        lambda key, state: written.append((key, state)) or f"{key}: tuned.")
    monkeypatch.setattr(main, "_reload_registry", lambda: written.append("reloaded"))
    monkeypatch.setattr(main.registry.models[KEY], "installed", True)
    with TestClient(main.app) as c:
        c.sent, c.written = sent, written
        yield c
        c.delete(f"/api/tune/{KEY}")


def _settled(client, phase):
    for _ in range(200):
        got = client.get(f"/api/tune/{KEY}").json()
        if got["phase"] == phase and not got["running"]:
            return got
        time.sleep(0.02)
    raise AssertionError(f"never reached {phase}: {got['phase']}")


def test_a_tuning_from_the_first_render_to_the_entry(client):
    r = client.post(f"/api/tune/{KEY}")
    assert r.status_code == 200 and r.json()["round"] == "sampler"
    got = _settled(client, "asking")
    assert len(client.sent) == 8
    # the model as it ships, with its own opener and negative floor
    spec = main.registry.models[KEY]
    assert {s["model_filename"] for s in client.sent} == {spec.dt_filename}
    assert {s["negative_prompt"] for s in client.sent} == {main.negative_for(spec, "")}
    assert {s["prompt"] for s in client.sent} == {main.prompt_for(spec, main.tune.SCENE)}
    # blind: what a picture is stays on the server until the review
    assert got["labels"] == {} and "label" not in got["questions"][0]["cells"][0]
    for q in got["questions"]:
        got = client.post(f"/api/tune/{KEY}/answer",
                          json={"question": q["id"], "answer": "cant_tell"}).json()
    assert got["round"] == "size"
    got = _settled(client, "asking")
    for q in got["questions"]:
        got = client.post(f"/api/tune/{KEY}/answer",
                          json={"question": q["id"], "answer": "one"}).json()
    assert got["round"] == "steps"
    got = client.post(f"/api/tune/{KEY}/skip").json()
    assert got["phase"] == "review" and got["labels"]
    assert got["questions"] == []
    # every frame is still there to look back at, now with what it was
    some = next(iter(got["labels"]))
    assert client.get(f"/api/tune/{KEY}/cells/{some}.png").status_code == 200
    r = client.post(f"/api/tune/{KEY}/apply")
    assert r.json() == {"message": f"{KEY}: tuned."}
    assert client.written[0][0] == KEY and client.written[1] == "reloaded"
    assert client.get(f"/api/tune/{KEY}").status_code == 404  # put away


def test_the_frames_are_served_while_they_are_asked_about(client):
    client.post(f"/api/tune/{KEY}")
    got = _settled(client, "asking")
    cell = got["questions"][0]["cells"][0]["id"]
    r = client.get(f"/api/tune/{KEY}/cells/{cell}.png")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    assert client.get(f"/api/tune/{KEY}/cells/..%2Fstate.png").status_code == 404


def test_what_cannot_be_tuned_or_answered_is_refused(client):
    assert client.post("/api/tune/flux1-schnell").status_code == 400
    assert client.get(f"/api/tune/{KEY}").status_code == 404
    client.post(f"/api/tune/{KEY}")
    got = _settled(client, "asking")
    bad = client.post(f"/api/tune/{KEY}/answer", json={"question": got["questions"][0]["id"],
                                                     "answer": 42})
    assert bad.status_code == 400
    assert client.post(f"/api/tune/{KEY}/apply").status_code == 400


def test_discarding_keeps_the_model_as_it_was(client):
    client.post(f"/api/tune/{KEY}")
    _settled(client, "asking")
    assert client.delete(f"/api/tune/{KEY}").json() == {"discarded": KEY}
    assert client.get(f"/api/tune/{KEY}").status_code == 404
    assert client.written == []


def test_the_model_list_says_which_models_can_be_tuned(client):
    rows = {m["key"]: m for m in client.get("/api/models").json()}
    assert rows[KEY]["can_tune"] in (True, False)  # installed or not on this machine
    assert all(not m["can_tune"] for k, m in rows.items() if k != KEY)
    client.post(f"/api/tune/{KEY}")
    assert {m["key"]: m for m in client.get("/api/models").json()}[KEY]["tuning"]


def test_a_model_on_an_sdxl_class_is_offered_all_three_rendered_their_own_way(tmp_path, monkeypatch):
    """The candidates come from the shipped profiles, each with the prompt
    the app would send that class -- a starved prompt is what made Animagine
    look soft when it was on the wrong one."""
    import yaml

    from app.private_models import load_profiles

    sdxl = load_profiles()["sdxl"]["settings"]
    private = tmp_path / "models.private.yaml"
    private.write_text(yaml.safe_dump({"models": {"mine": {
        "label": "Mine", "dt_filename": "mine_f16.ckpt", **sdxl}}}))
    monkeypatch.setattr(main, "private_path", lambda _: private)
    spec = SimpleNamespace(label="Mine", dt_filename="mine_f16.ckpt")
    current, options = main._tune_classes("mine", spec)
    assert current == "sdxl" and set(options) == {"sdxl", "illustrious", "animagine"}
    assert options["animagine"]["prompt"].startswith("masterpiece, best quality, very aesthetic")
    assert options["animagine"]["negative"] and options["sdxl"]["negative"] == ""
    assert options["illustrious"]["clip_skip"] == 2
    assert options["sdxl"]["prompt"] == main.tune.SCENE
    # edited by hand, or tuned before: no longer a class's, so no class round
    private.write_text(yaml.safe_dump({"models": {"mine": {
        "label": "Mine", "dt_filename": "mine_f16.ckpt", **{**sdxl, "cfg": 4.0}}}}))
    assert main._tune_classes("mine", spec) == (None, None)
