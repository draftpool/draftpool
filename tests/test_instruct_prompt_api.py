"""The Instruct dialog's optional LLM rewrite.

The action's contract is that the words in the box reach the edit model
verbatim, so the rewrite is a separate call that hands text back rather than a
step on the way to the queue. These pin that, and the two things the route
decides on the user's behalf: whether the picture itself is shown to the LLM,
and that there is no "surprise me" mode -- an LLM inventing a change to a
picture the user is happy with is not a feature.
"""

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import main
from app.llm import LlmError
from app.schemas import PromptVariant
from app.sessions import Store


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "store", Store(tmp_path))
    return TestClient(main.app, raise_server_exceptions=False)


def an_image(client, prompt="1girl in a park, a red mug on the table"):
    sid = client.post("/api/sessions", json={"instruction": "a cat"}).json()["id"]
    gen = main.store.create_generation(
        session_id=sid, kind="final", model_key="oneobsession", prompt=prompt,
        negative_prompt="blurry", seed=42, width=1024, height=1024, steps=20,
        cfg=4.0, sampler="DPM++ 2M Karras",
    )
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (30, 60, 90)).save(buf, format="PNG")
    main.store.save_image(gen, buf.getvalue())
    main.store.set_status(gen.id, "done", None)
    return main.store.get_generation(gen.id)


def stub_llm(monkeypatch, seen, vision=True, answer="Change the red mug to blue."):
    async def has_vision(*a, **kw):
        return vision

    async def edit_instruction(wish, source_prompt, image_b64=None):
        seen.update(wish=wish, source_prompt=source_prompt, image=image_b64)
        return PromptVariant(prompt=answer, negative_prompt="")

    monkeypatch.setattr(main.llm, "has_vision", has_vision)
    monkeypatch.setattr(main.llm, "edit_instruction", edit_instruction)


def test_the_rewrite_comes_back_and_nothing_is_queued(client, monkeypatch):
    seen: dict = {}
    stub_llm(monkeypatch, seen)
    g = an_image(client)
    resp = client.post(f"/api/generations/{g.id}/edit-prompt",
                       json={"prompt": "  make the mug blue  "})
    assert resp.status_code == 200
    assert resp.json() == {"prompt": "Change the red mug to blue.", "used_image": True}
    # the wish arrives trimmed, with the picture's own prompt for context
    assert seen["wish"] == "make the mug blue"
    assert "red mug" in seen["source_prompt"]
    # the edit itself is still the user's to press
    assert len(client.get(f"/api/sessions/{g.session_id}").json()["generations"]) == 1


def test_a_text_only_llm_is_never_sent_the_image(client, monkeypatch):
    """has_vision False means no images key at all; Ollama errors on one."""
    seen: dict = {}
    stub_llm(monkeypatch, seen, vision=False)
    g = an_image(client)
    resp = client.post(f"/api/generations/{g.id}/edit-prompt",
                       json={"prompt": "make the mug blue"})
    assert resp.status_code == 200
    assert resp.json()["used_image"] is False
    assert seen["image"] is None


def test_an_unknown_vision_answer_still_shows_the_picture(client, monkeypatch):
    """None is "could not ask Ollama", and is treated as a vision model.

    Same rule as animate-prompt: a capability lookup that failed must not
    silently downgrade the rewrite of a model that can in fact see.
    """
    seen: dict = {}
    stub_llm(monkeypatch, seen, vision=None)
    g = an_image(client)
    resp = client.post(f"/api/generations/{g.id}/edit-prompt",
                       json={"prompt": "make the mug blue"})
    assert resp.json()["used_image"] is True
    assert seen["image"] is not None


def test_an_empty_wish_is_refused_rather_than_guessed(client, monkeypatch):
    seen: dict = {}
    stub_llm(monkeypatch, seen)
    g = an_image(client)
    assert client.post(f"/api/generations/{g.id}/edit-prompt",
                       json={"prompt": ""}).status_code == 422
    assert client.post(f"/api/generations/{g.id}/edit-prompt",
                       json={"prompt": "   "}).status_code == 400
    assert not seen


def test_an_unfinished_parent_has_nothing_to_edit(client, monkeypatch):
    seen: dict = {}
    stub_llm(monkeypatch, seen)
    g = an_image(client)
    main.store.set_status(g.id, "running", None)
    resp = client.post(f"/api/generations/{g.id}/edit-prompt",
                       json={"prompt": "make the mug blue"})
    assert resp.status_code == 400
    assert not seen


def test_an_ollama_failure_reaches_the_page_as_a_reason(client, monkeypatch):
    async def has_vision(*a, **kw):
        return False

    async def boom(*a, **kw):
        raise LlmError("Ollama (gemma3:12b): no reply within 300s. Try again.")

    monkeypatch.setattr(main.llm, "has_vision", has_vision)
    monkeypatch.setattr(main.llm, "edit_instruction", boom)
    g = an_image(client)
    resp = client.post(f"/api/generations/{g.id}/edit-prompt",
                       json={"prompt": "make the mug blue"})
    assert resp.status_code == 503
    assert "no reply within 300s" in resp.json()["detail"]
