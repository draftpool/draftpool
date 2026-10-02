"""Which final honours `aspect_size`, and which does not.

The cap is the area a model composes an off-square frame at from scratch. A
Finalize composes and keeps to it; a Faithful is img2img over a draft that
already holds the figure, and capping it cost the SDXL models 57% of a 3:2
Faithful's pixels -- on a full-length figure, the face
(docs/decisions/DRAFT-STEPS.md, "Shaped Faithfuls"). qwen-image's cap is a
render ceiling instead, which every stage keeps to.
"""

import pytest
from fastapi.testclient import TestClient

from app import dims, main
from app.registry import CONFIG_PATH, load_registry
from app.sessions import Store
from app.settings import SettingsStore


@pytest.fixture
def client(tmp_path, monkeypatch):
    reg = load_registry(CONFIG_PATH)
    monkeypatch.setattr(main, "store", Store(tmp_path))
    monkeypatch.setattr(main, "settings_store", SettingsStore(tmp_path))
    monkeypatch.setattr(main, "registry", reg)
    with TestClient(main.app) as c:
        yield c


def a_finished_shaped_draft(client, model_key, aspect="3:2"):
    sid = client.post("/api/sessions", json={"instruction": "a queen"}).json()["id"]
    gen = client.post(f"/api/sessions/{sid}/drafts", json={
        "variants": [{"prompt": "1girl, full body, snow"}],
        "model_keys": [model_key], "seeds_per_combo": 1, "aspect": aspect,
    }).json()[0]
    main.store.set_status(gen["id"], "done", None)
    return gen


def finals(client, draft):
    faithful = client.post(f"/api/generations/{draft['id']}/finalize",
                           json={"mode": "faithful"}).json()
    final = client.post(f"/api/generations/{draft['id']}/finalize",
                        json={"mode": "regen"}).json()
    return faithful, final


def test_a_shaped_faithful_gets_the_finals_own_budget_where_a_finalize_is_capped(client):
    spec = main.registry.models["cyberrealistic-illustrious"]
    assert spec.aspect_size < spec.final.size and not spec.aspect_size_is_ceiling
    faithful, final = finals(client, a_finished_shaped_draft(client, spec.key))
    assert (faithful["width"], faithful["height"]) == (1856, 1280)
    assert (final["width"], final["height"]) == (1280, 832)
    # the shape is the draft's either way, only the budget differs
    assert faithful["aspect"] == final["aspect"] == "3:2"


def test_a_render_ceiling_holds_the_faithful_too(client):
    spec = main.registry.models["qwen-image"]
    assert spec.aspect_size_is_ceiling
    faithful, final = finals(client, a_finished_shaped_draft(client, spec.key, "16:9"))
    want = dims.stage_dims(spec.final.size, dims.ratio("16:9"), spec.aspect_size)
    assert (faithful["width"], faithful["height"]) == want
    assert (final["width"], final["height"]) == want
    assert want[0] <= 1152, "the frame this model returns nothing above"


def test_square_is_untouched_by_either_rule(client):
    spec = main.registry.models["cyberrealistic-illustrious"]
    faithful, final = finals(client, a_finished_shaped_draft(client, spec.key, "1:1"))
    assert (faithful["width"], faithful["height"]) == (final["width"], final["height"]) \
        == (spec.final.size, spec.final.size)


def test_a_faithful_size_of_its_own_applies_to_the_faithful_alone(client):
    """rinflanime's final is pinned at 1024 by what its txt2img does at 1536;
    a Faithful composes nothing and was measured clean there, so it names a
    budget of its own -- which the Finalize must not pick up, square or
    shaped, or the duplication the 1024 guards against comes back."""
    spec = main.registry.models["rinflanime"]
    assert spec.final.size == 1024 and spec.faithful_size == 1536
    faithful, final = finals(client, a_finished_shaped_draft(client, spec.key))
    assert (faithful["width"], faithful["height"]) == (1856, 1280)
    assert (final["width"], final["height"]) == (1280, 832)
    faithful, final = finals(client, a_finished_shaped_draft(client, spec.key, "1:1"))
    assert (faithful["width"], faithful["height"]) == (1536, 1536)
    assert (final["width"], final["height"]) == (1024, 1024)


def test_the_models_reply_says_what_a_faithful_honours(client):
    by_key = {m["key"]: m for m in client.get("/api/models").json()}
    assert by_key["cyberrealistic-illustrious"]["aspect_size"] == 1024
    assert by_key["cyberrealistic-illustrious"]["faithful_aspect_size"] is None
    assert by_key["qwen-image"]["faithful_aspect_size"] == by_key["qwen-image"]["aspect_size"] == 864
    # and at what budget: the final's, unless the model names a Faithful one
    assert by_key["cyberrealistic-illustrious"]["faithful_size"] == by_key["cyberrealistic-illustrious"]["final_size"] == 1536
    assert by_key["rinflanime"]["faithful_size"] == 1536 != by_key["rinflanime"]["final_size"]


def test_only_rinflanime_names_a_faithful_size():
    """A Faithful budget above the final's is a measured exception (nine
    one-figure frames on that model), not a default; a model that gains one
    gains the same measurement."""
    reg = load_registry(CONFIG_PATH)
    assert [k for k, m in reg.models.items() if m.faithful_size] == ["rinflanime"]


def test_only_qwen_image_declares_a_ceiling():
    """The SDXL entries' caps are composition budgets; if a new model's cap is
    a ceiling it has to say so, or its Faithfuls will ask for frames it cannot
    render. A model that gains the flag should also gain a measurement."""
    reg = load_registry(CONFIG_PATH)
    assert [k for k, m in reg.models.items() if m.aspect_size_is_ceiling] == ["qwen-image"]
