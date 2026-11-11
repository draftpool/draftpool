"""PUT /api/models/{key}/civitai: the settings dialog naming a model's Civitai version."""

import pytest
from fastapi.testclient import TestClient

from app import main


@pytest.fixture
def client(monkeypatch):
    done = []
    monkeypatch.setattr(main.migrate, "identify_model",
                        lambda key, vid: done.append((key, vid)) or f"{key}: credited.")
    monkeypatch.setattr(main, "_reload_registry", lambda: done.append("reloaded"))
    with TestClient(main.app) as c:
        c.done = done
        yield c


def test_the_page_can_send_what_the_civitai_page_shows(client):
    key = next(iter(main.registry.models))
    r = client.put(f"/api/models/{key}/civitai",
                   json={"ref": "urn:air:sdxl:checkpoint:civitai:1307857@3313575+3198988"})
    assert r.status_code == 200
    assert r.json() == {"message": f"{key}: credited.", "civitai_version_id": 3313575}
    # the model changes under the server, so the next request must see it
    assert client.done == [(key, 3313575), "reloaded"]


def test_a_reference_that_names_no_version_is_refused_and_changes_nothing(client):
    key = next(iter(main.registry.models))
    r = client.put(f"/api/models/{key}/civitai", json={"ref": "the unholy one"})
    assert r.status_code == 400 and "AIR" in r.json()["detail"]
    assert client.put("/api/models/no-such-model/civitai", json={"ref": "1"}).status_code == 404
    assert client.done == []


def test_the_model_list_says_which_models_can_be_given_one(client):
    rows = {m["key"]: m for m in client.get("/api/models").json()}
    for key, m in rows.items():
        spec = main.registry.models[key]
        assert m["civitai_version_id"] == spec.civitai_version_id
        # shipped models never: theirs is in models.yaml
        assert m["can_identify"] == (key in main.registry.private_keys and not spec.civitai_version_id)
