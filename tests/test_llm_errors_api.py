"""An Ollama failure has to reach the page as a reason, not a bare 500.

Every LLM route raised LlmError uncaught, so the browser saw "Internal Server
Error" and the cause stayed in the server log -- where, for a timeout, it was
logged as the empty string. These pin the wire contract the page's retry
notice is built on.
"""

import pytest
from fastapi.testclient import TestClient

from app import main
from app.llm import LlmError
from app.sessions import Store


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "store", Store(tmp_path))
    with TestClient(main.app, raise_server_exceptions=False) as c:
        yield c


def test_enrich_answers_503_with_the_reason(client, monkeypatch):
    async def boom(*a, **kw):
        raise LlmError("Ollama (gemma3:12b): no reply within 300s. Try again.")

    monkeypatch.setattr(main.llm, "enrich", boom)
    resp = client.post("/api/enrich", json={
        "instruction": "a cat", "k": 2, "uses_negative_prompt": False,
        "style": "prose", "model_keys": [], "mode": "explore",
    })
    assert resp.status_code == 503
    assert "no reply within 300s" in resp.json()["detail"]
    assert resp.json()["retry"] is True


def test_describe_answers_503_with_the_reason(client, monkeypatch):
    async def boom(*a, **kw):
        raise LlmError("Ollama (gemma3:12b): cannot reach it. Is `ollama serve` running?")

    monkeypatch.setattr(main.llm, "describe", boom)
    monkeypatch.setattr(main.llm, "has_vision", lambda *a, **kw: _true())
    resp = client.post("/api/describe", json={
        "image_b64": "AAAA", "style": "prose",
        "uses_negative_prompt": False, "hint": "", "model_keys": [],
    })
    assert resp.status_code == 503
    assert "ollama serve" in resp.json()["detail"]


async def _true():
    return True
