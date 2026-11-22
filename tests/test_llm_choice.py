"""The LLM dropdown names the model every LLM request uses.

It did not: the pick lived only in the server process, so a restart went back
to models.yaml's `llm.model` -- gemma3:12b, which this install's Ollama does
not even have -- and the dropdown, which lists only installed models, showed
its first option as selected while every request named the missing model.
"""

import pytest
from fastapi.testclient import TestClient

from app import main
from app.sessions import Store
from app.settings import SettingsStore

INSTALLED = [
    {"name": "a-text-only:7b", "vision": False},
    {"name": "qwen3-vl:8b", "vision": True},
    {"name": "z-vision:12b", "vision": True},
]


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "store", Store(tmp_path))
    monkeypatch.setattr(main, "settings_store", SettingsStore(tmp_path))

    async def installed():
        return INSTALLED

    monkeypatch.setattr(main.llm, "list_models", installed)
    before = main.llm.model
    with TestClient(main.app, raise_server_exceptions=False) as c:
        yield c
    main.llm.set_model(before)


def test_a_pick_is_saved_for_the_next_start(client, tmp_path):
    resp = client.post("/api/llms", json={"model": "z-vision:12b"})
    assert resp.status_code == 200
    assert SettingsStore(tmp_path).llm_model() == "z-vision:12b"


def test_a_missing_model_gives_way_to_an_installed_vision_one(client, tmp_path):
    main.llm.set_model("gemma3:12b")
    body = client.get("/api/llms").json()
    assert body["current"] == "qwen3-vl:8b"
    assert main.llm.model == "qwen3-vl:8b"
    # the stand-in is not a pick: nothing is saved
    assert SettingsStore(tmp_path).llm_model() is None


def test_an_installed_model_is_left_alone(client):
    main.llm.set_model("a-text-only:7b")
    assert client.get("/api/llms").json()["current"] == "a-text-only:7b"


def test_nothing_installed_leaves_the_model_alone(client, monkeypatch):
    async def none():
        return []

    monkeypatch.setattr(main.llm, "list_models", none)
    main.llm.set_model("gemma3:12b")
    assert client.get("/api/llms").json()["current"] == "gemma3:12b"


def test_the_setting_ignores_a_blank_or_hand_broken_value(tmp_path):
    s = SettingsStore(tmp_path)
    assert s.llm_model() is None
    s.path.write_text('{"llm_model": 42}')
    assert s.llm_model() is None
    s.path.write_text('{"llm_model": "  "}')
    assert s.llm_model() is None
    s.set_llm_model("qwen3-vl:8b")
    assert s.llm_model() == "qwen3-vl:8b"
