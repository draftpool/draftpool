import io
import os
import re
import types
from pathlib import Path

import pytest
from PIL import Image

from app.detailer import (
    DEFAULT_FACE_STRENGTH,
    DetailerError,
    _bounds_mask,
    _face_mask,
    _grid_size,
    _render_window,
    _square_crop_box,
    add_in_region,
    changed_mask,
    detect_faces,
    refine_faces,
)


def _png(w: int, h: int) -> bytes:
    """A flat image: nothing the face cascade can find."""
    return _png_solid(w, h, (120, 90, 70))


def _png_solid(w: int, h: int, colour: tuple[int, int, int]) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), colour).save(buf, "PNG")
    return buf.getvalue()

ROOT = Path(__file__).resolve().parent.parent
# a real render from scripts/probe.py, read from the install's own data
# directory (the test run itself uses a throwaway one, tests/conftest.py)
PROBE = Path(os.environ["DRAFTPOOL_CONFIGURED_DATA_DIR"]) / "probe" / "probe_anime_512.png"


def test_square_crop_box_clamps_to_image():
    box = _square_crop_box((900, 900, 100, 100), 1024, 1024)
    l, t, r, b = box
    assert r <= 1024 and b <= 1024 and l >= 0 and t >= 0
    assert r - l == b - t  # square


def test_square_crop_box_pads():
    l, t, r, b = _square_crop_box((400, 400, 100, 100), 1024, 1024)
    assert r - l == 200  # 100 * (1 + 2*0.5)


def test_detects_anime_face():
    if not PROBE.is_file():
        pytest.skip("probe image not present")
    faces = detect_faces(PROBE.read_bytes(), anime=True)
    assert len(faces) >= 1
    x, y, w, h = faces[0]
    assert w > 50 and h > 50


def test_ui_seeds_the_same_default_face_strength():
    # the Fix faces dialog pre-fills its strength box from a JS constant; if it
    # drifts from the server default the dialog lies about what "default" means
    html = (ROOT / "static" / "index.html").read_text()
    m = re.search(r"const FACE_STRENGTH_DEFAULT = ([0-9.]+);", html)
    assert m, "FACE_STRENGTH_DEFAULT missing from static/index.html"
    assert float(m.group(1)) == DEFAULT_FACE_STRENGTH


@pytest.mark.asyncio
async def test_a_hand_picked_box_skips_detection():
    """The fix for a face the cascade cannot see.

    lbpcascade_animeface keys on eyes, so a blindfold, visor or mask hides a
    face that is obvious to a person -- measured: covering only the eye band of
    a detected face drops detection from one to none, and no cascade setting
    recovers it. Passing the box explicitly must bypass detection entirely and
    still run the same crop/img2img/paste, including on an image where
    detection would find nothing at all.
    """
    png = _png(768, 768)          # flat image: the detector finds no face here
    assert detect_faces(png, anime=True) == []

    seen = {}

    async def fake_img2img(init, strength, **kw):
        seen["size"] = (kw["width"], kw["height"])
        seen["strength"] = strength
        return init

    dt = types.SimpleNamespace(img2img=fake_img2img)
    out = await refine_faces(
        dt, png, anime=True, strength=0.4,
        faces=[(200, 200, 160, 160)],
        prompt="p", negative_prompt="", model_filename="m.ckpt",
        steps=8, cfg=1.0, seed=1, sampler=None,
    )
    # it ran (detection would have raised) and used the square working size
    assert seen["size"] == (1024, 1024)
    assert seen["strength"] == 0.4
    with Image.open(io.BytesIO(out)) as im:
        assert im.size == (768, 768)   # pasted back into the original frame


@pytest.mark.asyncio
async def test_no_box_still_requires_a_detected_face():
    png = _png(768, 768)
    dt = types.SimpleNamespace(img2img=None)
    with pytest.raises(DetailerError):
        await refine_faces(dt, png, anime=True, prompt="p")


def test_changed_mask_finds_what_the_editor_touched():
    """The mask is the object, not the rectangle: it is built from the diff."""
    before = Image.new("RGB", (256, 256), (40, 40, 40))
    after = before.copy()
    after.paste(Image.new("RGB", (60, 60), (220, 30, 30)), (100, 100))

    mask, coverage = changed_mask(before, after)

    px = mask.load()
    assert px[130, 130] > 200          # inside the change
    assert px[10, 10] == 0             # far outside it
    # grown and feathered, so a little more than the 5.5% the square itself is
    assert 0.05 < coverage < 0.20


def test_changed_mask_reports_a_re_render_as_near_total_coverage():
    """What the fallback in add_in_region keys on.

    An instruction editor that repaints the whole crop instead of adding to it
    changes everything, and the caller has to notice: pasting that back through
    its own diff mask would replace the picture with a re-rendered one.
    """
    before = Image.new("RGB", (128, 128), (30, 30, 30))
    after = Image.new("RGB", (128, 128), (200, 200, 200))
    _, coverage = changed_mask(before, after)
    assert coverage > 0.95


def test_changed_mask_ignores_noise_below_the_threshold():
    """Texture and codec noise must not speckle the mask into a full paste."""
    import numpy as np

    rng = np.random.default_rng(0)
    base = np.full((128, 128, 3), 90, dtype=np.uint8)
    noisy = np.clip(base + rng.integers(-6, 7, base.shape), 0, 255).astype(np.uint8)
    _, coverage = changed_mask(Image.fromarray(base), Image.fromarray(noisy))
    assert coverage < 0.01


def test_nothing_outside_the_rectangle_can_be_written():
    """The promise the dialog makes, as arithmetic rather than as an estimate.

    Five of seven real region additions rewrote pixels outside the rectangle --
    up to 18,000 of them, 65px above the top edge -- because the mask was what
    the model changed and the model changed things it was not asked to. The
    mask is bounded by the rectangle now, and the bound is a hard multiply so
    "outside is untouched" cannot be approximately true.
    """
    import numpy as np

    box = (100, 100, 200, 200)
    mask = np.asarray(_bounds_mask((400, 400), box), float)
    outside = mask.copy()
    outside[box[1]:box[3], box[0]:box[2]] = 0
    assert outside.max() == 0                      # not one pixel of allowance
    assert mask[150, 150] == 255                   # and the middle is fully open


@pytest.mark.asyncio
async def test_an_addition_leaves_every_pixel_outside_the_box_byte_identical():
    """End to end, with an editor that returns something else entirely."""
    import numpy as np

    src = _png(768, 768)
    box = (200, 300, 400, 500)

    async def fake_img2img(init, strength, **kw):
        # the worst case: a "model" that ignores the instruction and hands back
        # a completely different picture at the requested size
        return _png_solid(kw["width"], kw["height"], (255, 0, 0))

    dt = types.SimpleNamespace(img2img=fake_img2img)
    out = await add_in_region(
        dt, src, box=box, work_size=512,
        prompt="Add a dog.", negative_prompt="", model_filename="m.ckpt",
        steps=8, cfg=1.0, seed=1, sampler=None,
    )
    before = np.asarray(Image.open(io.BytesIO(src)).convert("RGB"))
    after = np.asarray(Image.open(io.BytesIO(out)).convert("RGB"))
    assert after.shape == before.shape
    outside = np.abs(before.astype(int) - after.astype(int))
    outside[box[1]:box[3], box[0]:box[2]] = 0
    assert outside.max() == 0                      # byte for byte
    assert np.abs(before.astype(int) - after.astype(int)).max() > 0   # and it did edit


def test_the_render_window_keeps_the_crop_undistorted():
    """A 64px grid must not stretch what it renders.

    297x270 scaled to 1024 wants 1024x931 and the grid gives 1024x896; squashing
    the result back into the crop stretched everything in it by 4%, which is
    what a user sees as the edit changing the aspect. The window is grown to the
    render's shape instead, so the scale is uniform.
    """
    for box in [(415, 1261, 712, 1531), (100, 100, 500, 200), (0, 0, 333, 777)]:
        window, (tw, th) = _render_window(box, (1536, 1536), 1.5, 1024)
        w, h = window[2] - window[0], window[3] - window[1]
        assert abs((w / h) / (tw / th) - 1) < 0.005      # under half a percent
        assert tw % 64 == 0 and th % 64 == 0
        # and it still contains what the user selected
        assert window[0] <= box[0] and window[1] <= box[1]
        assert window[2] >= box[2] and window[3] >= box[3]


def test_the_render_frame_matches_the_rectangle_the_user_drew():
    """The frame the model composes in has to be the area that is kept.

    Taking the long side as work_size and snapping the short one to the grid
    puts a 2.8:1 rectangle in a 2:1 frame -- 100px of padding above it that the
    model composes into and the paste discards. Measured, that is the
    difference between a creature on the steps and four paws and a tuft of fur.
    """
    for w, h in [(712, 252), (297, 270), (400, 260), (333, 777), (400, 100)]:
        tw, th = _grid_size(w / h, 1024)
        assert tw % 64 == 0 and th % 64 == 0
        assert min(tw, th) >= 512                 # the renderer's own floor
        assert tw * th <= 2 * 1024 ** 2           # and the cost cap
        box = (100, 100, 100 + w, 100 + h)
        window, _ = _render_window(box, (2048, 2048), 1.0, 1024)
        pad_x = (window[2] - window[0]) - w
        pad_y = (window[3] - window[1]) - h
        # up to 4:1 the grid can match the shape, so the window is the rectangle
        assert pad_x <= 24 and pad_y <= 24, (w, h, pad_x, pad_y)


@pytest.mark.asyncio
async def test_a_low_contrast_addition_is_not_amputated():
    """The bug that lost a black dog's legs against dark stone.

    Pasting only the pixels that differ from the original sounds conservative
    and is not: "changed" and "visibly different" are not the same thing, and a
    dark object over a dark background falls under any workable threshold. The
    rectangle is what the user asked to have edited, so the rectangle is what
    gets written -- including the parts of the edit that barely register.
    """
    import numpy as np

    src = _png(768, 768)
    box = (200, 200, 456, 456)

    async def fake_img2img(init, strength, **kw):
        # a frame that differs from the original by 2 levels: under any
        # threshold that survives sensor and codec noise
        base = np.asarray(Image.open(io.BytesIO(init)).convert("RGB"), int)
        faint = np.clip(base + 2, 0, 255).astype("uint8")
        buf = io.BytesIO()
        Image.fromarray(faint).save(buf, "PNG")
        return buf.getvalue()

    dt = types.SimpleNamespace(img2img=fake_img2img)
    out = await add_in_region(
        dt, src, box=box, work_size=512,
        prompt="Add a black dog.", negative_prompt="", model_filename="m.ckpt",
        steps=8, cfg=1.0, seed=1, sampler=None,
    )
    after = np.asarray(Image.open(io.BytesIO(out)).convert("RGB"), int)
    before = np.asarray(Image.open(io.BytesIO(src)).convert("RGB"), int)
    middle = (box[0] + 40, box[1] + 40)
    assert after[middle[1], middle[0]].tolist() != before[middle[1], middle[0]].tolist()
    outside = np.abs(before - after)
    outside[box[1]:box[3], box[0]:box[2]] = 0
    assert outside.max() == 0


def test_a_face_pass_writes_the_face_and_not_the_crop_around_it():
    """Fix faces used to paste back the whole padded square it rendered.

    The crop is half as wide again as the face on purpose -- the model needs
    the head and its surroundings to redraw a face that belongs there -- but
    pasting all of it back re-rendered the hair, the wallpaper and the picture
    frame behind the head at the face's strength, and an img2img pass does not
    reproduce a pattern or a straight line exactly. Reported as the background
    coming back not quite matching.
    """
    import numpy as np

    face = (642, 144, 235, 235)                  # a real detected box
    crop = _square_crop_box(face, 1536, 1536)    # ... and the square it renders in
    size = (crop[2] - crop[0], crop[3] - crop[1])
    mask = np.asarray(_face_mask(size, face, (crop[0], crop[1])), float)

    cx = int(face[0] + face[2] / 2 - crop[0])
    cy = int(face[1] + face[3] / 2 - crop[1])
    assert mask[cy, cx] == 255                   # the face is written
    for corner in [(0, 0), (0, -1), (-1, 0), (-1, -1)]:
        assert mask[corner] == 0                 # its surroundings are not
    # and it covers more than the detector's box, which cuts off forehead and chin
    assert mask[cy - face[3] // 2, cx] > 0
