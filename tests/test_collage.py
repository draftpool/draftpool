import io

import pytest
from PIL import Image

from app.detailer import DetailerError, build_collage_init


def png(w, h, color):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color).save(buf, format="PNG")
    return buf.getvalue()


def test_pastes_reference_into_box():
    base = png(512, 512, (0, 0, 0))
    ref = png(200, 300, (255, 0, 0))
    out = build_collage_init(base, ref, (100, 100, 300, 300))
    img = Image.open(io.BytesIO(out))
    assert img.size == (512, 512)
    assert img.getpixel((200, 200)) == (255, 0, 0)   # inside box: reference
    assert img.getpixel((50, 50)) == (0, 0, 0)       # outside box: untouched


def test_ref_box_crop_applied():
    base = png(512, 512, (0, 0, 0))
    ref = Image.new("RGB", (400, 400), (0, 255, 0))
    ref.paste((0, 0, 255), (200, 0, 400, 400))       # right half blue
    buf = io.BytesIO(); ref.save(buf, format="PNG")
    out = build_collage_init(base, buf.getvalue(), (0, 0, 128, 128), (200, 0, 400, 400))
    assert Image.open(io.BytesIO(out)).getpixel((64, 64)) == (0, 0, 255)


def test_too_small_box_raises():
    with pytest.raises(DetailerError):
        build_collage_init(png(512, 512, (0,) * 3), png(64, 64, (255,) * 3), (0, 0, 10, 10))
