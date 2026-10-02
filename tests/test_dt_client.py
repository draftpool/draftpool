import base64
import io

import pytest
from PIL import Image

from app import dt_client
from app.dt_client import (
    DrawThingsClient,
    DrawThingsStalled,
    DrawThingsTimeout,
    fit_init_image,
)


def png_bytes(w: int, h: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (200, 30, 30)).save(buf, format="PNG")
    return buf.getvalue()


def test_resizes_to_target():
    out = fit_init_image(png_bytes(256, 256), 1024, 1024)
    assert Image.open(io.BytesIO(out)).size == (1024, 1024)


def test_passthrough_when_already_matching():
    src = png_bytes(512, 512)
    assert fit_init_image(src, 512, 512) is src


# -- a session that answers but does not render --------------------------------


class FakeResponse:
    status_code = 200

    def __init__(self, body):
        self._body = body

    def json(self):
        return self._body


def fake_httpx(monkeypatch, body):
    class FakeClient:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, url, json=None):
            return FakeResponse(body)

    monkeypatch.setattr(dt_client.httpx, "AsyncClient", FakeClient)


async def test_empty_images_raise_stalled_and_mark_the_client(monkeypatch):
    fake_httpx(monkeypatch, {"images": []})
    client = DrawThingsClient("http://x")
    with pytest.raises(DrawThingsStalled):
        await client.txt2img(
            prompt="p", negative_prompt="", model_filename="m.ckpt",
            width=512, height=512, steps=4, cfg=1.0, seed=1, sampler=None,
        )
    assert client.stalled is True
    assert await client.renders("m.ckpt", timeout=1) is False


async def test_an_image_clears_the_stall(monkeypatch):
    fake_httpx(monkeypatch, {"images": [base64.b64encode(png_bytes(8, 8)).decode()]})
    client = DrawThingsClient("http://x")
    client.stalled = True
    assert await client.renders("m.ckpt", timeout=1) is True
    assert client.stalled is False


async def test_a_timeout_says_so_instead_of_failing_with_no_message(monkeypatch):
    """httpx raises its timeouts empty, and the card showed exactly that."""

    class TimingOutClient:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, url, json=None):
            raise dt_client.httpx.ReadTimeout("")

    monkeypatch.setattr(dt_client.httpx, "AsyncClient", TimingOutClient)
    client = DrawThingsClient("http://x", request_timeout=900)
    with pytest.raises(DrawThingsTimeout) as caught:
        await client.txt2img(
            prompt="p", negative_prompt="", model_filename="m.ckpt",
            width=512, height=512, steps=4, cfg=1.0, seed=1, sampler=None,
        )
    assert "900s" in str(caught.value)
    # not a stall: the renderer never answered at all, so the health flag and
    # the hold-and-retry path both stay out of it
    assert client.stalled is False


async def test_txt2img_passes_its_timeout_to_the_request(monkeypatch):
    seen = {}

    class RecordingClient:
        def __init__(self, *a, **kw):
            seen["timeout"] = kw.get("timeout")

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, url, json=None):
            return FakeResponse({"images": [base64.b64encode(png_bytes(8, 8)).decode()]})

    monkeypatch.setattr(dt_client.httpx, "AsyncClient", RecordingClient)
    client = DrawThingsClient("http://x", request_timeout=900)
    await client.txt2img(
        timeout=3000,
        prompt="p", negative_prompt="", model_filename="m.ckpt",
        width=512, height=512, steps=4, cfg=1.0, seed=1, sampler=None,
    )
    assert seen["timeout"] == 3000


def test_only_file_and_weight_of_a_lora_reach_draw_things():
    """A recorded LoRA carries its Civitai version too; that is ours, not the API's."""
    payload = DrawThingsClient("http://x")._build_payload(
        prompt="p", negative_prompt="", model_filename="m.ckpt", width=64, height=64,
        steps=4, cfg=1.0, seed=1, sampler=None,
        loras=[{"file": "a.ckpt", "weight": 0.8, "civitai_version_id": 11},
               {"file": "b.ckpt"}],
    )
    assert payload["loras"] == [{"file": "a.ckpt", "weight": 0.8},
                                {"file": "b.ckpt", "weight": 1.0}]
    # and none is an explicit empty list, never absent
    assert DrawThingsClient("http://x")._build_payload(
        prompt="p", negative_prompt="", model_filename="m.ckpt", width=64, height=64,
        steps=4, cfg=1.0, seed=1, sampler=None)["loras"] == []
