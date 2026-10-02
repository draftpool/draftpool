"""Per-model step counts: the store, the override, and the two ways to change one.

The point of the feature is that config/models.yaml stays the shipped default
and nothing writes to it, so most of what is worth asserting is about what
survives -- a reset finding its way back to the measured number, and an
override outliving the model being reloaded from disk.
"""

import json

import pytest
from fastapi.testclient import TestClient

from app import main
from app.registry import CONFIG_PATH, load_registry
from app.sessions import Store
from app.settings import STEP_MAX, SettingsStore


@pytest.fixture
def settings(tmp_path):
    return SettingsStore(tmp_path)


@pytest.fixture
def client(tmp_path, monkeypatch):
    """The app against a throwaway store, settings file and registry."""
    reg = load_registry(CONFIG_PATH)
    monkeypatch.setattr(main, "store", Store(tmp_path))
    monkeypatch.setattr(main, "settings_store", SettingsStore(tmp_path))
    monkeypatch.setattr(main, "registry", reg)
    with TestClient(main.app) as c:
        yield c


# -- the store ----------------------------------------------------------------


def test_nothing_saved_is_no_overrides(settings):
    assert settings.steps() == {}


def test_set_and_reset_a_single_stage(settings):
    settings.set_steps("bas-anima", {"draft": 12, "final": None})
    assert settings.steps() == {"bas-anima": {"draft": 12}}
    settings.set_steps("bas-anima", {"draft": None, "final": None})
    assert settings.steps() == {}
    # the empty model is dropped rather than left as an empty dict
    assert json.loads(settings.path.read_text())["steps"] == {}


def test_out_of_range_is_refused(settings):
    with pytest.raises(ValueError):
        settings.set_steps("bas-anima", {"draft": STEP_MAX + 1})
    with pytest.raises(ValueError):
        settings.set_steps("bas-anima", {"draft": 1})
    with pytest.raises(ValueError):
        settings.set_steps("bas-anima", {"upscale": 8})
    assert settings.steps() == {}


def test_a_refused_value_does_not_take_the_valid_one_with_it(settings):
    settings.set_steps("bas-anima", {"draft": 12})
    with pytest.raises(ValueError):
        settings.set_steps("bas-anima", {"draft": 14, "final": 999})
    assert settings.steps() == {"bas-anima": {"draft": 12}}


def test_a_hand_edited_file_cannot_smuggle_a_bad_value_in(settings, tmp_path):
    settings.path.write_text(json.dumps({"steps": {
        "bas-anima": {"draft": 999, "final": 12, "upscale": 4, "size": 512},
        "gone": "nonsense",
    }}))
    assert settings.steps() == {"bas-anima": {"final": 12}}


def test_unreadable_file_falls_back_to_the_shipped_defaults(settings):
    settings.path.write_text("{ not json")
    assert settings.steps() == {}


def test_clear_drops_everything(settings):
    settings.set_steps("bas-anima", {"draft": 12, "final": 30})
    settings.clear_steps()
    assert settings.steps() == {}


# -- the registry --------------------------------------------------------------


def test_defaults_are_captured_from_the_yaml():
    reg = load_registry(CONFIG_PATH)
    for spec in reg.models.values():
        for stage in (spec.draft, spec.final, spec.upscale):
            assert stage.default_steps == stage.steps


def test_override_applies_and_reverts():
    reg = load_registry(CONFIG_PATH)
    spec = reg.models["bas-anima"]
    shipped_draft, shipped_final = spec.draft.steps, spec.final.steps

    reg.apply_step_overrides({"bas-anima": {"draft": shipped_draft + 5}})
    assert spec.draft.steps == shipped_draft + 5
    assert spec.final.steps == shipped_final          # untouched stage
    assert spec.draft.default_steps == shipped_draft  # and the default survives

    # the whole answer, not a patch on the last one
    reg.apply_step_overrides({})
    assert spec.draft.steps == shipped_draft


def test_override_never_touches_the_upscale_stage():
    reg = load_registry(CONFIG_PATH)
    shipped = reg.models["bas-anima"].upscale.steps
    reg.apply_step_overrides({"bas-anima": {"upscale": 2}})
    assert reg.models["bas-anima"].upscale.steps == shipped


def test_override_for_a_model_that_is_gone_is_ignored():
    reg = load_registry(CONFIG_PATH)
    reg.apply_step_overrides({"no-such-model": {"draft": 12}})  # no raise


# -- the HTTP contract ---------------------------------------------------------


def shipped(client, key):
    m = next(m for m in client.get("/api/models").json() if m["key"] == key)
    return m["draft"]["default_steps"], m["final"]["default_steps"]


def test_put_changes_what_models_reports(client):
    draft_def, final_def = shipped(client, "bas-anima")
    resp = client.put("/api/models/bas-anima/steps",
                      json={"draft": draft_def + 2, "final": final_def + 2})
    assert resp.status_code == 200
    assert resp.json() == {
        "key": "bas-anima",
        "draft": {"steps": draft_def + 2, "default": draft_def},
        "final": {"steps": final_def + 2, "default": final_def},
    }
    m = next(m for m in client.get("/api/models").json() if m["key"] == "bas-anima")
    assert (m["draft"]["steps"], m["final"]["steps"]) == (draft_def + 2, final_def + 2)
    # the shipped numbers travel with them, so the UI can offer Reset
    assert (m["draft"]["default_steps"], m["final"]["default_steps"]) == (draft_def, final_def)


def test_null_resets_a_stage(client):
    draft_def, final_def = shipped(client, "bas-anima")
    client.put("/api/models/bas-anima/steps",
               json={"draft": draft_def + 2, "final": final_def + 2})
    body = client.put("/api/models/bas-anima/steps",
                      json={"draft": None, "final": final_def + 2}).json()
    assert body["draft"]["steps"] == draft_def
    assert body["final"]["steps"] == final_def + 2


def test_reset_all(client):
    draft_def, _ = shipped(client, "bas-anima")
    client.put("/api/models/bas-anima/steps", json={"draft": draft_def + 2})
    rows = client.delete("/api/settings/steps").json()
    assert {r["key"] for r in rows} == set(main.registry.models)
    assert all(r["draft"]["steps"] == r["draft"]["default"] for r in rows)


def test_out_of_range_and_unknown_model_are_rejected(client):
    assert client.put("/api/models/bas-anima/steps",
                      json={"draft": STEP_MAX + 1}).status_code == 422
    assert client.put("/api/models/no-such-model/steps",
                      json={"draft": 8}).status_code == 404


def test_a_saved_override_survives_a_registry_reload(client, tmp_path):
    draft_def, _ = shipped(client, "bas-anima")
    client.put("/api/models/bas-anima/steps", json={"draft": draft_def + 3})
    # what a server restart does
    reloaded = load_registry(CONFIG_PATH)
    reloaded.apply_step_overrides(SettingsStore(tmp_path).steps())
    assert reloaded.models["bas-anima"].draft.steps == draft_def + 3


def test_models_yaml_is_never_written_to(client):
    before = CONFIG_PATH.read_bytes()
    client.put("/api/models/bas-anima/steps", json={"draft": 11, "final": 21})
    client.delete("/api/settings/steps")
    assert CONFIG_PATH.read_bytes() == before


# -- what actually gets queued -------------------------------------------------


def a_draft(client):
    sid = client.post("/api/sessions", json={"instruction": "a cat"}).json()["id"]
    return client.post(f"/api/sessions/{sid}/drafts", json={
        "variants": [{"prompt": "1girl, park"}],
        "model_keys": ["bas-anima"],
        "seeds_per_combo": 1,
    }).json()[0]


def test_a_draft_is_queued_with_the_overridden_step_count(client):
    draft_def, _ = shipped(client, "bas-anima")
    client.put("/api/models/bas-anima/steps", json={"draft": draft_def + 4})
    assert a_draft(client)["steps"] == draft_def + 4


def test_a_final_follows_the_model_setting_by_default(client):
    _, final_def = shipped(client, "bas-anima")
    client.put("/api/models/bas-anima/steps", json={"final": final_def + 6})
    gen = a_draft(client)
    final = client.post(f"/api/generations/{gen['id']}/finalize",
                        json={"mode": "regen"}).json()
    assert final["steps"] == final_def + 6


def test_a_per_generation_override_beats_the_model_setting_and_leaves_it_alone(client):
    _, final_def = shipped(client, "bas-anima")
    gen = a_draft(client)
    final = client.post(f"/api/generations/{gen['id']}/finalize",
                        json={"mode": "regen", "steps": final_def + 20}).json()
    assert final["steps"] == final_def + 20
    # the next one, with nothing asked for, is back to the model's own number
    plain = client.post(f"/api/generations/{gen['id']}/finalize",
                        json={"mode": "regen"}).json()
    assert plain["steps"] == final_def
    assert main.settings_store.steps() == {}


def test_a_per_generation_override_out_of_range_is_rejected(client):
    gen = a_draft(client)
    resp = client.post(f"/api/generations/{gen['id']}/finalize",
                       json={"mode": "regen", "steps": STEP_MAX + 1})
    assert resp.status_code == 422


# -- "Faithful max": the number that makes an img2img pass do final.steps -----


def test_faithful_steps_undoes_the_halving_an_img2img_pass_does():
    reg = load_registry(CONFIG_PATH)
    m = reg.models["wai-illustrious"]
    assert m.final.strength == 0.5
    assert m.faithful_steps == m.final.steps * 2       # 28 -> 56, i.e. 28 real


def test_faithful_steps_never_exceeds_the_limit():
    """A model wanting more than the limit lands short, not over.

    Written against cyberrealistic-anima, which shipped 44 and wanted 88. It
    has since been measured down to 30 and no shipped model exceeds the limit
    any more, so the cap is exercised through an override rather than through
    whichever model happens to be highest this month.
    """
    reg = load_registry(CONFIG_PATH)
    reg.apply_step_overrides({"bas-anima": {"final": 40}})   # would want 80
    m = reg.models["bas-anima"]
    assert m.faithful_steps == STEP_MAX
    assert m.faithful_steps * m.final.strength < m.final.steps


def test_every_shipped_model_now_reaches_parity():
    """What GUIDE.md claims: the doubled figure fits under the limit for all.

    This is a property of the shipped numbers rather than of the code, and it
    became true only when cyberrealistic-anima came down from 44. If a future
    model ships above STEP_MAX/2 finals steps, its Faithful silently does less
    work than its Finalize and that sentence in the guide needs changing.
    """
    reg = load_registry(CONFIG_PATH)
    short = {
        k: (m.final.steps, m.faithful_steps)
        for k, m in reg.models.items()
        if m.supports_img2img and not m.edit_only
        and m.faithful_steps * m.final.strength < m.final.steps
    }
    assert not short, f"these cannot reach parity: {short}"


def test_a_full_strength_stage_asks_for_nothing_extra():
    """Nothing is halved at strength 1.0, so there is nothing to undo."""
    reg = load_registry(CONFIG_PATH)
    m = reg.models["qwen-edit"]
    assert m.final.strength == 1.0
    assert m.faithful_steps == m.final.steps


def test_faithful_steps_follows_an_override_rather_than_the_shipped_number():
    """It is derived, so re-measuring a model's finals moves it too."""
    reg = load_registry(CONFIG_PATH)
    m = reg.models["bas-anima"]
    reg.apply_step_overrides({"bas-anima": {"final": 20}})
    assert m.faithful_steps == 40


async def _noop():
    return None


def a_finished_draft(client):
    gen = a_draft(client)
    main.store.set_status(gen["id"], "done", None)
    return gen


def test_a_faithful_asks_for_double_so_the_pass_does_the_measured_count(client):
    _, final_def = shipped(client, "bas-anima")
    gen = a_finished_draft(client)
    faithful = client.post(f"/api/generations/{gen['id']}/finalize",
                           json={"mode": "faithful"}).json()
    assert faithful["steps"] == final_def * 2       # 8 -> 16, i.e. 8 real
    # ...and Finalize is txt2img, so it takes the number as measured
    final = client.post(f"/api/generations/{gen['id']}/finalize",
                        json={"mode": "regen"}).json()
    assert final["steps"] == final_def


def test_the_doubling_follows_the_models_own_setting(client):
    """Change the final count in settings and the Faithful moves with it."""
    client.put("/api/models/bas-anima/steps", json={"final": 20})
    gen = a_finished_draft(client)
    faithful = client.post(f"/api/generations/{gen['id']}/finalize",
                           json={"mode": "faithful"}).json()
    assert faithful["steps"] == 40


def test_finalize_plus_runs_the_same_count_as_finalize(client, monkeypatch):
    """Finalize+ only changes the prompt; the schedule is a Finalize's."""
    _, final_def = shipped(client, "bas-anima")
    monkeypatch.setattr(main, "_require_vision", _noop)
    gen = a_finished_draft(client)
    plus = client.post(f"/api/generations/{gen['id']}/finalize",
                       json={"mode": "describe"}).json()
    assert plus["steps"] == final_def
    assert plus["describe_first"] is True


def test_a_caller_can_still_name_its_own_count(client):
    """The box is gone from the UI; the endpoint is unchanged for scripts."""
    gen = a_finished_draft(client)
    faithful = client.post(f"/api/generations/{gen['id']}/finalize",
                           json={"mode": "faithful", "steps": 12}).json()
    assert faithful["steps"] == 12
