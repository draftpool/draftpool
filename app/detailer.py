"""ADetailer-style face refinement: detect faces, img2img each crop at high
resolution with the same model, feather-blend the result back in."""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter

log = logging.getLogger(__name__)

ANIME_CASCADE = Path(__file__).resolve().parent.parent / "config" / "lbpcascade_animeface.xml"

# How much of the face crop the img2img pass is allowed to rewrite. 0.4 adds
# detail while usually holding the likeness; below ~0.3 the expression (mouth,
# gaze) survives almost intact, above ~0.5 the model starts inventing a
# different face. The API and the UI both seed from this value.
DEFAULT_FACE_STRENGTH = 0.4


class DetailerError(RuntimeError):
    pass


def detect_faces(png: bytes, anime: bool) -> list[tuple[int, int, int, int]]:
    """Face boxes (x, y, w, h), largest first."""
    img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    if anime:
        cascade = cv2.CascadeClassifier(str(ANIME_CASCADE))
    else:
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
    faces = cascade.detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=5, minSize=(48, 48)
    )
    return sorted(
        (tuple(int(v) for v in f) for f in faces), key=lambda f: -(f[2] * f[3])
    )


def _square_crop_box(
    face: tuple[int, int, int, int], img_w: int, img_h: int, pad: float = 0.5
) -> tuple[int, int, int, int]:
    """Expand the face box by `pad` and square it, clamped to the image."""
    x, y, w, h = face
    size = min(int(max(w, h) * (1 + 2 * pad)), img_w, img_h)
    cx, cy = x + w // 2, y + h // 2
    left = max(0, min(cx - size // 2, img_w - size))
    top = max(0, min(cy - size // 2, img_h - size))
    return left, top, left + size, top + size


def eligible_faces(
    png: bytes, anime: bool, work_size: int = 1024
) -> tuple[list[tuple[int, int, int, int]], int]:
    """(faces worth refining, count skipped as already-large).

    A face whose padded crop is already close to work_size gains no
    resolution from the img2img pass; repainting it only changes the image.
    """
    img = Image.open(io.BytesIO(png))
    faces = detect_faces(png, anime)
    keep, skipped = [], 0
    for face in faces:
        l, t, r, b = _square_crop_box(face, img.width, img.height)
        if (r - l) >= work_size * 0.8:
            skipped += 1
        else:
            keep.append(face)
    return keep, skipped


def build_collage_init(
    base_png: bytes,
    ref_png: bytes,
    box: tuple[int, int, int, int],
    ref_box: tuple[int, int, int, int] | None = None,
) -> bytes:
    """Paste (aspect-fill) a reference crop into `box` of the base image.

    The result seeds a region repaint, so the reference's actual pixels drive
    the outcome instead of a text description.
    """
    base = Image.open(io.BytesIO(base_png)).convert("RGB")
    ref = Image.open(io.BytesIO(ref_png)).convert("RGB")
    if ref_box:
        l, t, r, b = (int(v) for v in ref_box)
        ref = ref.crop((max(0, l), max(0, t), min(ref.width, r), min(ref.height, b)))
    l, t, r, b = (int(v) for v in box)
    bw, bh = r - l, b - t
    if bw < 32 or bh < 32 or ref.width < 8 or ref.height < 8:
        raise DetailerError("selection is too small")
    s = max(bw / ref.width, bh / ref.height)
    ref = ref.resize((int(ref.width * s + 0.5), int(ref.height * s + 0.5)), Image.LANCZOS)
    ox, oy = (ref.width - bw) // 2, (ref.height - bh) // 2
    piece = ref.crop((ox, oy, ox + bw, oy + bh))
    # feathered paste: hard collage edges survive repainting as visible seams
    mask = Image.new("L", (bw, bh), 0)
    margin = max(6, min(bw, bh) // 12)
    mask.paste(255, (margin, margin, bw - margin, bh - margin))
    mask = mask.filter(ImageFilter.GaussianBlur(margin))
    base.paste(piece, (l, t), mask)
    out = io.BytesIO()
    base.save(out, format="PNG")
    return out.getvalue()


def _face_mask(
    size: tuple[int, int],
    face: tuple[int, int, int, int],
    origin: tuple[int, int],
    grow: float = 0.25,
) -> Image.Image:
    """A soft ellipse over `face` (x, y, w, h) inside a crop starting at `origin`.

    An ellipse rather than the crop's rectangle because that is the shape of
    the thing being refined: grown a quarter to take in the forehead and chin
    the detector's box cuts off, feathered enough that the boundary is nowhere
    a line, and leaving the hair and the background it sits in as they were.
    """
    x, y, w, h = face
    cx, cy = x + w / 2 - origin[0], y + h / 2 - origin[1]
    rx, ry = w / 2 * (1 + grow), h / 2 * (1 + grow)
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).ellipse(
        (cx - rx, cy - ry, cx + rx, cy + ry), fill=255
    )
    return mask.filter(ImageFilter.GaussianBlur(max(4, int(min(rx, ry) * 0.18))))


def _mult64(v: float) -> int:
    return max(512, int(round(v / 64)) * 64)


def _bounds_mask(
    size: tuple[int, int], box: tuple[int, int, int, int], spill: float = 0.0
) -> Image.Image:
    """A mask that is soft inside `box` and exactly zero outside it.

    `spill` widens the allowance by that fraction of the box; at 0 -- the
    default, and what the dialog promises -- the rectangle is the boundary and
    nothing beyond it can be written at all.

    The softness is put INSIDE the boundary rather than across it: a blurred
    rectangle has a tail wider than its radius, so blurring alone would let a
    little of every edit past the edge. The blurred mask is multiplied by the
    hard rectangle, which makes "outside is untouched" arithmetic rather than
    an estimate -- a paste through a zero mask leaves its destination byte for
    byte.
    """
    l, t, r, b = box
    mx, my = int((r - l) * spill), int((b - t) * spill)
    outer = (max(0, l - mx), max(0, t - my),
             min(size[0], r + mx), min(size[1], b + my))
    inset = max(2, min(outer[2] - outer[0], outer[3] - outer[1]) // 16)
    soft = Image.new("L", size, 0)
    soft.paste(255, (outer[0] + inset, outer[1] + inset,
                     max(outer[0] + inset + 1, outer[2] - inset),
                     max(outer[1] + inset + 1, outer[3] - inset)))
    soft = soft.filter(ImageFilter.GaussianBlur(inset))
    hard = Image.new("L", size, 0)
    hard.paste(255, outer)
    return ImageChops.multiply(soft, hard)


def _grid_size(aspect: float, work_size: int) -> tuple[int, int]:
    """Render dimensions on the 64px grid whose shape is closest to `aspect`.

    Taking the long side as work_size and snapping the short one is the obvious
    way and it is wrong for a strip: a 2.8:1 rectangle snapped that way comes
    out 1024x512, because 512 is the smallest side Draw Things will render, and
    a 2:1 frame around a 2.8:1 rectangle is 100px of padding the model composes
    into and the paste then throws away -- a decapitated dog, measured.
    Searching the grid instead finds 1472x512 (2.875:1), and the window barely
    has to grow at all. The cost is bounded by area rather than by shape: never
    more pixels than a square work_size render, and no side longer than twice
    one, which reaches 4:1 exactly (2048x512 is the pixel count of 1024x1024).
    The area cap is not a nicety -- a 1984x704 frame for this same rectangle
    ran past Draw Things' 900s request timeout and lost the render.
    """
    # A shape within `tolerance` is as good as exact: the window is grown to the
    # render's shape, so the error costs a few pixels of padding and never any
    # distortion. What is chosen among the close-enough shapes is the one
    # nearest work_size squared in area -- NOT the cheapest. These models have
    # a resolution they work at (an instruction editor barely functions below
    # about a megapixel, see the edit endpoint), so a strip has to be rendered
    # long rather than small: 1472x512 for a 2.8:1 rectangle, not 640x224.
    tolerance = 0.04
    want = work_size ** 2
    good, closest = [], None
    for short in range(512, work_size + 1, 64):
        for wide, tall in ((_mult64(short * aspect), short),
                           (short, _mult64(short / aspect))):
            if max(wide, tall) > work_size * 2 or wide * tall > work_size ** 2:
                continue
            err = abs((wide / tall) / aspect - 1)
            if closest is None or (err, wide * tall) < closest[0]:
                closest = ((err, wide * tall), (wide, tall))
            if err <= tolerance:
                good.append((abs(wide * tall - want), (wide, tall)))
    if good:
        return min(good)[1]
    return closest[1] if closest else (work_size, work_size)


def _render_window(
    box: tuple[int, int, int, int],
    size: tuple[int, int],
    context: float,
    work_size: int,
) -> tuple[tuple[int, int, int, int], tuple[int, int]]:
    """(window to crop, size to render it at) -- containing `box`, undistorted.

    Draw Things renders on a 64px grid, so a crop's own shape is not usually
    available: a 297x270 rectangle scaled to 1024 wants 1024x931 and gets
    1024x896, and squashing the result back into the crop stretches everything
    in it by 4%. Every edit shipped through here was distorted that way, which
    is what a user sees as "it changed the aspect".

    The fix is to choose the render size first and then GROW THE WINDOW to
    match its shape, so the scale is uniform in both directions. Growing is
    free: the extra pixels are context for the model and are never pasted back
    -- the paste is bounded by the rectangle itself (see `_bounds_mask`).
    """
    img_w, img_h = size
    l, t, r, b = box
    cx, cy = (l + r) / 2, (t + b) / 2
    w = min(float(img_w), max(r - l, (r - l) * context))
    h = min(float(img_h), max(b - t, (b - t) * context))
    tw, th = _grid_size(w / h, work_size)
    aspect = tw / th
    # grow the short side to the render aspect, then give back what the image
    # cannot supply -- a window at the edge of the frame cannot grow past it
    if w / h < aspect:
        w = min(float(img_w), h * aspect)
    else:
        h = min(float(img_h), w / aspect)
    if w / h > aspect:
        w = h * aspect
    else:
        h = w / aspect
    wl = int(round(min(max(cx - w / 2, 0), img_w - w)))
    wt = int(round(min(max(cy - h / 2, 0), img_h - h)))
    window = (wl, wt, wl + int(round(w)), wt + int(round(h)))
    # the window exists to hold the rectangle; if the aspect fit pushed an edge
    # off it, widen it back and accept the residual squash rather than crop
    window = (min(window[0], l), min(window[1], t),
              max(window[2], r), max(window[3], b))
    pad = max(window[2] - window[0] - (r - l), window[3] - window[1] - (b - t))
    if pad > max(24, 0.1 * max(r - l, b - t)):
        # a sliver past about 4:1 cannot be matched on the grid (512x3072 is
        # not on offer), so the frame is wider than the rectangle and the model
        # can compose into ground that will be discarded. Nothing is broken --
        # the paste is still bounded -- but a lost object is explained here.
        log.info("render frame is %dpx larger than the rectangle (shape %.2f:1 "
                 "is past what the grid offers)", pad, (r - l) / max(1, b - t))
    return window, (tw, th)


def changed_mask(
    before: Image.Image,
    after: Image.Image,
    *,
    threshold: float = 15,
    grow: int = 9,
    feather: int = 6,
) -> tuple[Image.Image, float]:
    """(mask of what `after` changed, share of the frame it covers).

    An instruction editor is told to leave everything but the new thing alone,
    and mostly does -- so the pixels it did change ARE the new thing, and they
    are a better mask than any rectangle: the object keeps its ears, its tail
    and the shadow it casts, none of which fit the box the user drew.

    Blur before thresholding, or film grain and texture speckle the mask; grow
    the result afterwards so a soft edge (fur, glow, shadow) comes with the
    object rather than being cut off it.
    """
    a = np.asarray(before.convert("RGB"), float)
    b = np.asarray(after.convert("RGB"), float)
    diff = np.abs(a - b).mean(axis=2).astype(np.uint8)
    smooth = Image.fromarray(diff).filter(ImageFilter.GaussianBlur(feather - 2))
    hard = Image.fromarray(
        ((np.asarray(smooth, float) > threshold) * 255).astype(np.uint8)
    )
    mask = hard.filter(ImageFilter.MaxFilter(grow)).filter(
        ImageFilter.GaussianBlur(feather)
    )
    coverage = float(np.asarray(mask, float).mean() / 255)
    return mask, coverage


async def add_in_region(
    dt: Any,
    png: bytes,
    *,
    box: tuple[int, int, int, int],
    work_size: int = 1024,
    context: float = 1.0,
    spill: float = 0.0,
    **gen_kwargs: Any,
) -> bytes:
    """Put something new near `box` using an instruction editor, keeping the rest.

    The other route to a new object is `repaint_region` at high strength, and
    it is the weaker one: a diffusion checkpoint given a crop and a prompt
    re-renders the crop, so the ground under the new thing is reinvented and
    its lighting comes from the prompt rather than from the picture
    (docs/decisions/EDITING.md, decisions 9 and 10). An instruction editor is
    built for the other job -- it reconstructs what it was not asked to change
    -- so here it renders the window with the addition in it and only the
    pixels it actually moved are taken back.

    **The rectangle is the boundary, and the whole of it is taken.** `spill`
    defaults to 0, so not one pixel outside `box` changes; inside it, the
    editor's frame is used as it came back.

    Both halves of that were learned the hard way. The first version let the
    change spill wherever the editor had touched, and five of seven real
    attempts rewrote pixels outside the rectangle -- up to 18,000 of them, 65px
    above the top edge. The second bounded the spill but still pasted only the
    pixels that differed from the original, and **that quietly amputates
    things**: a black dog's legs against dark stone differ by less than any
    workable threshold, so the mask never marked them and the composite came
    back with a dog missing its legs while the render in Draw Things had them.

    There is no threshold that fixes it, because the problem is a real one --
    "changed" and "visibly different" are not the same thing on a dark object
    over a dark background. The rectangle the user drew is the region they
    asked to be edited, so the rectangle is what is written.

    `context` is 1.0 -- the window IS the rectangle -- and that is not a matter
    of taste. The model composes inside the frame it is given, so any part of
    the frame that will not be pasted back is a place it can put the new thing
    and lose it: at 1.5x, the same instruction that produced a whole creature
    put its head 100px above the rectangle and the paste returned four paws and
    a tuft of fur. The frame the model sees and the area that is kept have to
    be the same rectangle. `_grid_size` is what makes that possible for shapes
    the 64px grid does not offer directly.

    """
    base = Image.open(io.BytesIO(png)).convert("RGB")
    l, t, r, b = (int(v) for v in box)
    if r - l < 32 or b - t < 32:
        raise DetailerError("selected region is too small")
    (wl, wt, wr, wb), (tw, th) = _render_window(
        (l, t, r, b), base.size, context, work_size
    )
    window = base.crop((wl, wt, wr, wb))
    buf = io.BytesIO()
    window.resize((tw, th), Image.LANCZOS).save(buf, format="PNG")
    out = await dt.img2img(buf.getvalue(), 1.0, width=tw, height=th, **gen_kwargs)
    painted = Image.open(io.BytesIO(out)).convert("RGB").resize(
        window.size, Image.LANCZOS
    )
    local = (l - wl, t - wt, r - wl, b - wt)
    # for the log only: how much of the frame the editor actually touched, so a
    # result that did nothing (or re-rendered everything) can be told apart
    # afterwards from one that added what was asked for
    _, coverage = changed_mask(window, painted)
    log.info("instruction editor changed %.0f%% of the frame", coverage * 100)
    base.paste(painted, (wl, wt), _bounds_mask(window.size, local, spill))
    result = io.BytesIO()
    base.save(result, format="PNG")
    return result.getvalue()


async def repaint_region(
    dt: Any,
    png: bytes,
    *,
    box: tuple[int, int, int, int],
    strength: float = 0.7,
    work_size: int = 1024,
    context: float = 2.5,
    **gen_kwargs: Any,
) -> bytes:
    """Repaint only `box` (l, t, r, b) toward the prompt, feathered back in.

    Two things make this work, and it did neither before (measured in
    docs/decisions/EDITING.md, decision 8):

    **The window is much larger than the box.** `context` is a multiple of the
    box, not a 15% margin. A tight crop tells the model nothing about the
    picture it is painting into, and at the strength a change needs it stops
    reconstructing the init and renders whatever the prompt describes -- a
    fresh room, correctly lit and completely wrong. With the surroundings in
    view it paints something that belongs where it is going. This is also why
    the window is not the whole image: everything in it is downscaled into
    `work_size` together, so context is bought with resolution on the part
    that is actually being repainted.

    This is for changing something that is already in the box -- a colour, a
    material, one object for another. It is NOT the way to add something to a
    picture: that took 0.85, and at 0.85 the crop is re-rendered rather than
    edited, so the ground under the new thing is invented, the light comes from
    the prompt instead of the picture, and the join with the untouched
    surroundings gives the whole thing away. `add_in_region` does additions;
    the withdrawal is decision 12 in docs/decisions/EDITING.md.

    **Only `box` is pasted back.** The model repaints the whole window; the
    surroundings it invents are discarded, and the pixels outside the box are
    the original ones, untouched. Draw Things' HTTP API has no mask parameter
    (checked: every plausible spelling is rejected as an unrecognized key), so
    the mask happens here.
    """
    base = Image.open(io.BytesIO(png)).convert("RGB")
    l, t, r, b = (int(v) for v in box)
    if r - l < 32 or b - t < 32:
        raise DetailerError("selected region is too small")
    # the context window: the box grown about its own centre, clamped to the
    # image and to the shape the renderer can actually produce (_render_window
    # -- without that last part everything in the crop comes back stretched)
    (wl, wt, wr, wb), (tw, th) = _render_window(
        (l, t, r, b), base.size, context, work_size
    )
    window = base.crop((wl, wt, wr, wb))
    buf = io.BytesIO()
    window.resize((tw, th), Image.LANCZOS).save(buf, format="PNG")
    out = await dt.img2img(buf.getvalue(), strength, width=tw, height=th, **gen_kwargs)
    painted = Image.open(io.BytesIO(out)).convert("RGB").resize(
        (wr - wl, wb - wt), Image.LANCZOS
    )
    base.paste(painted, (wl, wt), _bounds_mask(
        window.size, (l - wl, t - wt, r - wl, b - wt)))
    result = io.BytesIO()
    base.save(result, format="PNG")
    return result.getvalue()


# Combine: one picture's subject put into another picture's scene. Draw
# Things' HTTP API takes exactly one init image -- two are silently dropped for
# whatever is on the app's canvas -- so the instruction editor is handed both
# pictures as one frame, subject panel on the left and scene on the right, and
# the scene panel is what is kept.
#
# The frame's pixel budget, and how much of it the subject gets: see
# docs/decisions/COMBINE.md. The scene panel is the output, so it gets the
# pixels; the subject only has to be legible enough to be copied.
COMBINE_AREA = 1344 * 896
COMBINE_SUBJECT_SCALE = 0.5


def _floor64(v: float) -> int:
    return max(64, int(v) // 64 * 64)


def combine_layout(
    scene_size: tuple[int, int],
    subject_size: tuple[int, int],
    *,
    area: int = COMBINE_AREA,
    subject_scale: float = COMBINE_SUBJECT_SCALE,
) -> tuple[int, int, int]:
    """(subject panel width, scene panel width, height) of a Combine frame.

    Both panels share the frame's height. The scene panel keeps the scene's
    own shape (to the 64px grid) because it is the picture that comes back;
    the subject is shown at `subject_scale` of that height, in a column as wide
    as it needs, never wider than the scene. Everything sits on the 64px grid
    so the scene panel can be cut out of the result exactly, and the whole
    frame stays inside `area`.
    """
    scene_ar = scene_size[0] / scene_size[1]
    subject_ar = subject_size[0] / subject_size[1]
    height = _floor64((area / (scene_ar + subject_scale * subject_ar)) ** 0.5)
    while True:
        scene_w = max(256, int(round(scene_ar * height / 64)) * 64)
        subject_w = min(scene_w, max(128, int(round(subject_scale * subject_ar * height / 64)) * 64))
        if (scene_w + subject_w) * height <= area or height <= 256:
            return subject_w, scene_w, height
        height -= 64


def _cover(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    """`img` scaled to cover `size` and centre-cropped to it: never stretched."""
    s = max(size[0] / img.width, size[1] / img.height)
    img = img.resize((max(size[0], round(img.width * s)), max(size[1], round(img.height * s))),
                     Image.LANCZOS)
    left, top = (img.width - size[0]) // 2, (img.height - size[1]) // 2
    return img.crop((left, top, left + size[0], top + size[1]))


def build_combine_frame(
    scene_png: bytes,
    subject_png: bytes,
    *,
    area: int = COMBINE_AREA,
    subject_scale: float = COMBINE_SUBJECT_SCALE,
) -> tuple[bytes, int, int]:
    """(the two-panel frame as PNG, width of the scene panel on its right, height).

    The subject is fitted whole inside its panel on white -- cropping it could
    cut off the very thing being moved -- and centred vertically. The scene is
    covered, not stretched, so at most a grid step of its edge is lost.
    """
    scene = Image.open(io.BytesIO(scene_png)).convert("RGB")
    subject = Image.open(io.BytesIO(subject_png)).convert("RGB")
    subject_w, scene_w, height = combine_layout(
        scene.size, subject.size, area=area, subject_scale=subject_scale
    )
    frame = Image.new("RGB", (subject_w + scene_w, height), "white")
    frame.paste(_cover(scene, (scene_w, height)), (subject_w, 0))
    tall = height * subject_scale
    fit = min(subject_w / subject.width, tall / subject.height)
    piece = subject.resize(
        (max(1, round(subject.width * fit)), max(1, round(subject.height * fit))),
        Image.LANCZOS,
    )
    frame.paste(piece, ((subject_w - piece.width) // 2, (height - piece.height) // 2))
    buf = io.BytesIO()
    frame.save(buf, format="PNG")
    return buf.getvalue(), scene_w, height


COMBINE_LEAD = (
    "This image has two panels: the left panel shows a subject, the right "
    "panel is a scene. Edit the right panel: "
)
COMBINE_TAIL = " Keep the subject's face, hair and clothing"


def combine_words(prompt: str) -> str:
    """What the user typed, out of a prompt combine_prompt wrote."""
    if not prompt.startswith(COMBINE_LEAD):
        return prompt
    return prompt[len(COMBINE_LEAD):].split(COMBINE_TAIL, 1)[0].strip()


def combine_prompt(instruction: str) -> str:
    """The user's words, framed so the model knows which panel is which.

    The user writes about the pictures ("put the woman on the bench"); the
    panels are the app's device and are explained here rather than asked of
    them.
    """
    words = instruction.strip()
    if words and words[-1] not in ".!?":
        # otherwise the next sentence runs on: "...the café Light what..."
        words += "."
    return (
        f"{COMBINE_LEAD}{words} "
        # people write where and how, not "and keep her face": unasked, a
        # dress gained tiers and closed eyes opened on the way across. Asked
        # as "as they are in the left panel", the sofa pose came along too, so
        # the pose is handed to the instruction in so many words.
        f"{COMBINE_TAIL.strip()} the same as in the left "
        "panel, but give the subject the pose and position the instruction "
        "describes, fitted to the scene. Light it the way the scene is lit, "
        "with matching shadows. Keep the right panel's scene, framing and "
        "perspective exactly unchanged."
    )


async def combine(dt: Any, frame_png: bytes, *, scene_w: int, **gen_kwargs: Any) -> bytes:
    """Render a Combine frame and return its scene panel."""
    frame = Image.open(io.BytesIO(frame_png))
    width, height = frame.size
    out = await dt.img2img(frame_png, 1.0, width=width, height=height, **gen_kwargs)
    painted = Image.open(io.BytesIO(out)).convert("RGB")
    if painted.size != (width, height):
        # Draw Things answered at another size; scale back so the cut is exact
        painted = painted.resize((width, height), Image.LANCZOS)
    buf = io.BytesIO()
    painted.crop((width - scene_w, 0, width, height)).save(buf, format="PNG")
    return buf.getvalue()


async def refine_faces(
    dt: Any,
    png: bytes,
    *,
    anime: bool,
    strength: float = DEFAULT_FACE_STRENGTH,
    work_size: int = 1024,
    max_faces: int = 2,
    faces: list[tuple[int, int, int, int]] | None = None,
    **gen_kwargs: Any,
) -> bytes:
    """Return `png` with up to `max_faces` faces re-rendered at high detail.

    gen_kwargs are the usual dt.img2img args (prompt, model_filename, steps...)
    minus width/height, which are set to work_size for each crop.

    `faces` (x, y, w, h boxes) skips detection and refines exactly those. It is
    how a face the cascade cannot see gets refined anyway -- the detector keys
    on eyes, so a blindfold or mask hides one that is obvious to a person. A
    hand-picked box goes through the same padding, crop, img2img and feathered
    paste as a detected one, so the two paths give the same kind of result. The
    already-large check is skipped for it: the user pointed at this face, so
    repainting it is what they asked for.
    """
    if faces is None:
        faces, skipped = eligible_faces(png, anime, work_size)
        if not faces:
            raise DetailerError(
                "face already fills most of the frame; nothing to gain from a detail pass"
                if skipped
                else "no face detected in the image"
            )
    base = Image.open(io.BytesIO(png)).convert("RGB")
    for face in faces[:max_faces]:
        l, t, r, b = _square_crop_box(face, base.width, base.height)
        crop = base.crop((l, t, r, b))
        buf = io.BytesIO()
        crop.resize((work_size, work_size), Image.LANCZOS).save(buf, format="PNG")
        refined_png = await dt.img2img(
            buf.getvalue(), strength, width=work_size, height=work_size, **gen_kwargs
        )
        refined = (
            Image.open(io.BytesIO(refined_png))
            .convert("RGB")
            .resize((r - l, b - t), Image.LANCZOS)
        )
        # Write back the FACE, not the crop it was rendered in. The crop is
        # deliberately half as wide again as the face -- the model needs the
        # head and some surroundings to redraw a face that belongs there -- but
        # everything in that square was being pasted back too, so a detail pass
        # re-rendered the hair, the wall and the window behind the head at the
        # same strength as the face. Reported from use: the background inside
        # the square comes back not quite matching the picture around it, which
        # is exactly what an img2img pass does to a pattern or a straight line.
        base.paste(refined, (l, t), _face_mask(refined.size, face, (l, t)))
        log.info("refined face at (%d,%d) size %d", l, t, r - l)
    out = io.BytesIO()
    base.save(out, format="PNG")
    return out.getvalue()
