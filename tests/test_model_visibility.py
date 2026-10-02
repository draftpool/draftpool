"""Hiding a model from the UI: the store, the registry rule, and the endpoints.

The feature is a tidying of a crowded chip row and nothing more, so the
assertions that matter are about what hiding does *not* do -- it must not stop
a render, and it must not take the faithful Upscale off an existing card. The
rest is that the two ways to hide a model (the per-model tick and the
not-installed rule) stay independent, so turning the rule off gives back
exactly the ticks the user set by hand.
"""

import json

import pytest
from fastapi.testclient import TestClient

from app import main
from app.registry import CONFIG_PATH, load_registry
from app.sessions import Store
from app.settings import SettingsStore


@pytest.fixture
def settings(tmp_path):
    return SettingsStore(tmp_path)


@pytest.fixture
def reg():
    """A registry with everything installed, so `shown` is about the settings."""
    r = load_registry(CONFIG_PATH)
    for spec in (*r.models.values(), *r.video_models.values()):
        spec.installed = True
    return r


@pytest.fixture
def client(tmp_path, monkeypatch, reg):
    monkeypatch.setattr(main, "store", Store(tmp_path))
    monkeypatch.setattr(main, "settings_store", SettingsStore(tmp_path))
    monkeypatch.setattr(main, "registry", reg)
    with TestClient(main.app) as c:
        yield c


# -- the store ----------------------------------------------------------------


def test_everything_is_enabled_by_default(settings):
    assert settings.disabled_models() == set()
    assert settings.hide_uninstalled() is False


def test_tick_and_untick_one_model(settings):
    settings.set_model_enabled("bas-anima", False)
    assert settings.disabled_models() == {"bas-anima"}
    settings.set_model_enabled("bas-anima", True)
    assert settings.disabled_models() == set()


def test_disabled_keys_survive_a_step_change(settings):
    """The two settings share a file and must not overwrite each other."""
    settings.set_model_enabled("bas-anima", False)
    settings.set_steps("krea2-turbo", {"draft": 12, "final": None})
    settings.set_hide_uninstalled(True)
    assert settings.disabled_models() == {"bas-anima"}
    assert settings.steps() == {"krea2-turbo": {"draft": 12}}
    assert settings.hide_uninstalled() is True


def test_a_hand_edited_file_cannot_break_the_read(settings):
    settings.path.parent.mkdir(parents=True, exist_ok=True)
    settings.path.write_text(json.dumps({"disabled_models": "bas-anima"}))
    assert settings.disabled_models() == set()
    settings.path.write_text(json.dumps({"disabled_models": ["ok", 7, None]}))
    assert settings.disabled_models() == {"ok"}


def test_an_unknown_key_is_kept(settings):
    """A model dropped for a release and put back keeps its tick."""
    settings.set_model_enabled("model-from-a-later-release", False)
    settings.set_model_enabled("bas-anima", False)
    assert settings.disabled_models() == {"model-from-a-later-release", "bas-anima"}


# -- the registry rule --------------------------------------------------------


def test_unticking_hides_only_that_model(reg):
    reg.apply_model_visibility({"bas-anima"}, False)
    assert reg.models["bas-anima"].enabled is False
    assert reg.models["bas-anima"].shown is False
    assert all(m.shown for k, m in reg.models.items() if k != "bas-anima")


def test_applying_is_the_whole_answer_not_a_patch(reg):
    reg.apply_model_visibility({"bas-anima"}, False)
    reg.apply_model_visibility(set(), False)
    assert reg.models["bas-anima"].shown is True


def test_the_rule_and_the_tick_are_independent(reg):
    reg.models["krea2-turbo"].installed = False
    reg.apply_model_visibility({"bas-anima"}, True)
    # hidden by the rule, but still ticked: turning the rule off brings it back
    assert reg.models["krea2-turbo"].enabled is True
    assert reg.models["krea2-turbo"].shown is False
    reg.apply_model_visibility({"bas-anima"}, False)
    assert reg.models["krea2-turbo"].shown is True
    assert reg.models["bas-anima"].shown is False


def test_video_models_hide_the_same_way(reg):
    reg.apply_model_visibility({"fusionx"}, False)
    assert reg.video_models["fusionx"].shown is False


def test_hiding_does_not_stop_a_render(reg):
    """The endpoints go on serving a hidden model; only the UI leaves it out."""
    reg.upscaler = "remacri.ckpt"
    reg.apply_model_visibility({"bas-anima"}, False)
    assert reg.model_or_raise("bas-anima").key == "bas-anima"
    assert reg.upscale_block("bas-anima") is None


def test_a_hidden_model_keeps_upscaling_its_own_images(reg):
    reg.upscaler = "remacri.ckpt"
    reg.apply_model_visibility({"bas-anima"}, False)
    # its own card still offers it, and it is still the default
    assert reg.upscale_candidates("bas-anima")[0] == "bas-anima"
    # but it is gone from every other model's list of stand-ins
    assert "bas-anima" not in reg.upscale_candidates("krea2-turbo")


def test_a_rescan_does_not_lose_the_ticks(reg, monkeypatch):
    """resolve_installed() recomputes `shown`, and must not clear `enabled`."""
    reg.apply_model_visibility({"bas-anima"}, True)
    monkeypatch.setattr("app.registry.installed_files", lambda dirs: set())
    reg.resolve_installed()
    assert reg.models["bas-anima"].enabled is False
    assert reg.hide_uninstalled is True


# -- the endpoints ------------------------------------------------------------


def test_models_endpoint_lists_hidden_models_too(client):
    """The dialog has to be able to offer a hidden model back."""
    client.put("/api/models/bas-anima/enabled", json={"enabled": False})
    rows = {m["key"]: m for m in client.get("/api/models").json()}
    assert rows["bas-anima"]["enabled"] is False
    assert rows["bas-anima"]["shown"] is False
    assert rows["krea2-turbo"]["shown"] is True


def test_put_enabled_returns_every_model(client):
    res = client.put("/api/models/bas-anima/enabled", json={"enabled": False}).json()
    assert res["hide_uninstalled"] is False
    hidden = [m for m in res["models"] if not m["shown"]]
    assert [m["key"] for m in hidden] == ["bas-anima"]
    # video models are in the same reply, and addressable the same way
    assert any(m["key"] == "fusionx" for m in res["models"])


def test_put_enabled_accepts_a_video_model(client):
    res = client.put("/api/models/fusionx/enabled", json={"enabled": False}).json()
    assert {m["key"]: m["shown"] for m in res["models"]}["fusionx"] is False
    assert client.get("/api/video-models").json()[0]["shown"] is False


def test_unknown_key_is_a_404(client):
    assert client.put("/api/models/nope/enabled", json={"enabled": False}).status_code == 404


def test_hide_uninstalled_survives_a_restart(client, tmp_path, monkeypatch, reg):
    client.put("/api/settings/hide-uninstalled", json={"hide_uninstalled": True})
    client.put("/api/models/bas-anima/enabled", json={"enabled": False})
    assert client.get("/api/settings/models").json()["hide_uninstalled"] is True

    # what a fresh process does with the file the last one wrote
    fresh = load_registry(CONFIG_PATH)
    store = SettingsStore(tmp_path)
    fresh.apply_model_visibility(store.disabled_models(), store.hide_uninstalled())
    assert fresh.models["bas-anima"].enabled is False
    assert fresh.hide_uninstalled is True


def test_the_render_endpoints_ignore_the_tick(client):
    """Hiding is convenience: a card made before it keeps working."""
    client.put("/api/models/bas-anima/enabled", json={"enabled": False})
    # the step endpoint is the cheapest one that resolves a model by key
    assert client.put(
        "/api/models/bas-anima/steps", json={"draft": 12, "final": None}
    ).status_code == 200
