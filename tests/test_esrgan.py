import io

import pytest
from PIL import Image

from app import esrgan


@pytest.mark.asyncio
async def test_upscale2x_doubles_dimensions():
    if not esrgan.available():
        pytest.skip("realesrgan binary not installed")
    img = Image.new("RGB", (96, 64), (120, 180, 90))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    out = await esrgan.upscale2x(buf.getvalue(), "anime")
    assert Image.open(io.BytesIO(out)).size == (192, 128)


@pytest.mark.asyncio
async def test_unknown_flavor_rejected():
    with pytest.raises(esrgan.EsrganError):
        await esrgan.upscale2x(b"", "watercolour")
