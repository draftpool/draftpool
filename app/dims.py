"""Grid geometry shared by the image and the video paths.

Draw Things floors each side of a request to a multiple of 64: asking for
624x624 returns 576x576, 416x832 returns 384x832, and the long-known 832x480
returns 832x448. An off-grid request therefore loses both area and the shape
it was computed for, so every size in here lands on the grid before it is sent.

Two callers, one idea. A generation stage's configured `size` and a video
model's configured width/height are both *pixel budgets* rather than fixed
shapes: diffusion cost tracks pixel count, so holding the area and taking the
shape from somewhere else (the chosen aspect ratio for a still, the source
still's own aspect for a clip) leaves generation time about where it was.
"""

from __future__ import annotations

import math

GRID = 64

# The ratios offered for generation. Square first because it is the default;
# the rest are ordered as landscape/portrait pairs, widening as they go, so the
# buttons read as a progression. Deliberately a closed list: past roughly 2:1
# even SDXL starts duplicating subjects, and the 64px grid gets coarse enough
# that the shape asked for is not the shape returned.
#
# 3:2 and 2:3 sit between the two pairs that were here first, so they are
# interior to the range already verified rather than an extension of it, and
# the grid resolves them better than it resolves 4:3: at a 1024 budget 3:2
# lands 1280x832 (+2.6% off the ratio) against 4:3's 1152x896 (-3.6%).
#
# 2:1 and 1:2 are NOT here on purpose. They are simultaneously at the
# composition limit above and the worst-resolved of any candidate (1472x704,
# +4.5%), and no model has been measured there. 21:9 is worse still: it gives
# up 6.2% of the pixel budget to reach 1536x640. Measure before adding either.
ASPECTS: dict[str, tuple[int, int]] = {
    "1:1": (1, 1),
    "4:3": (4, 3),
    "3:4": (3, 4),
    "3:2": (3, 2),
    "2:3": (2, 3),
    "16:9": (16, 9),
    "9:16": (9, 16),
}

ASPECT_LABELS: dict[str, str] = {
    "1:1": "Square",
    "4:3": "Landscape",
    "3:4": "Portrait",
    # the 35mm frame, and the one ratio here anybody has a camera for
    "3:2": "Photo",
    "2:3": "Photo tall",
    "16:9": "Wide",
    "9:16": "Tall",
}

DEFAULT_ASPECT = "1:1"


def snap(px: float) -> int:
    """Nearest multiple of the 64px grid, never below one step.

    Half-up rather than Python's banker's rounding, because static/index.html
    mirrors this with Math.round to say what a ratio will produce before it
    runs; tests/test_dims.py fails if the two disagree.
    """
    return max(GRID, math.floor(px / GRID + 0.5) * GRID)


def ratio(aspect: str) -> float:
    """Width over height for a named preset; unknown names fall back to square."""
    w, h = ASPECTS.get(aspect, ASPECTS[DEFAULT_ASPECT])
    return w / h


def stage_dims(size: int, r: float, aspect_size: int | None = None) -> tuple[int, int]:
    """Width and height for a `size`-budget render shaped to aspect ratio `r`.

    The budget is size x size pixels and it is held: at r = 4/3 a 1024 stage
    renders 1152x896, not 1365x1024, so picking a ratio costs no extra time.
    Each side is the equal-area solution snapped to the grid, which is where
    the familiar SDXL buckets come from -- 1024 at 3:4 gives 896x1152, at 9:16
    gives 768x1344, both bucket sizes exactly.

    `aspect_size` sets the budget for a non-square render, and binds where it
    is below the stage size: the SDXL entries in config/models.yaml keep a
    shaped final to 1024 against a 1536 square, qwen-image to 864 against
    1024. Which stages honour it is `ModelSpec.shaped_cap`: a from-scratch
    render always, a Faithful only where the cap is a render ceiling, an
    Upscale never -- an img2img pass over a frame that already holds the
    composition has nothing to compose, and capping it only costs pixels
    (docs/decisions/DRAFT-STEPS.md). Absent means "never measured, render
    square".

    An earlier version of this file claimed models compose badly off-square
    past ~1MP and capped four of them on that basis. That was wrong twice over:
    the portraits were never compared against a square control, and the whole
    batch ran on a Draw Things session that later stopped generating and had to
    be restarted. Re-running the identical model, size, seed and prompt on a
    healthy session gave a clean single subject. See docs/decisions/SHAPES.md.

    Square is exempt and returns exactly size x size, so the default ratio
    behaves exactly as it did before any of this existed, and every stage size
    in config/models.yaml is already a multiple of 64. See config/models.yaml
    for the per-model figures and docs/decisions/SHAPES.md for the Illustrious
    measurements.
    """
    if size <= 0:
        return GRID, GRID
    if r == 1.0:
        return snap(size), snap(size)
    budget = min(size, aspect_size) if aspect_size else size
    side = math.sqrt(r) if r > 0 else 1.0
    return snap(budget * side), snap(budget / side)


def budget(width: int, height: int) -> int:
    """The stage size an actual width x height picture was rendered at.

    The inverse of the rule `stage_dims` applies, and the only fair way to ask
    "is this target bigger than that picture": a stage size is an area, so it
    has to be compared against an area. A 1536 budget at 3:4 renders 1344x1792
    -- long edge above 1536, short edge below it -- so comparing either side
    against a target size gets the answer wrong in one direction or the other,
    and comparing the geometric mean gets it right for every shape.

    Grid snapping means this does not always return the budget exactly (that
    3:4 pair reads back as 1552), which is fine for the comparison it exists
    for and would not be fine for anything that needs the original number.

    floor(x + 0.5) rather than round() for the same reason as `snap`:
    static/index.html mirrors this with Math.round, which is half-up.
    """
    if width <= 0 or height <= 0:
        return 0
    return math.floor(math.sqrt(width * height) + 0.5)


def stage_size(width: int, height: int) -> int:
    """The stage size a width x height picture was rendered at, on the grid.

    `budget` recovers the area but not the number: 1344x1792 is a 1536 render
    and reads back as 1552, 1280x832 is a 1024 render and reads back as 1032.
    Snapping to the 64px grid puts the configured size back, which is what a
    caller asking "is this picture larger than a `size` stage" needs, or every
    3:2 render would count as larger than the stage that drew it. The same
    rule is in static/index.html (stageSize), which looks timings up by it and
    decides whether an edit resizes; tests/test_timings.py runs that copy.
    """
    return GRID * math.floor(math.sqrt(width * height) / GRID + 0.5)


def nearest_aspect(width: int, height: int) -> str:
    """The preset closest in shape to an actual width x height.

    The fallback for an image whose ratio was never recorded: rows predating
    the feature, and anything whose pixels came from elsewhere (a Real-ESRGAN
    2x, a region repaint). Compared in log space so 4:3 and 3:4 are the same
    distance from square.
    """
    if width <= 0 or height <= 0:
        return DEFAULT_ASPECT
    r = width / height
    return min(ASPECTS, key=lambda k: abs(math.log(ratio(k) / r)))


# 256px is the thinnest side worth asking a video model for, and 1024 the
# longest: 480p-class Wan models lose coherence beyond it.
_MIN_UNITS = 256 // GRID
_MAX_UNITS = 1024 // GRID


def clip_size(
    src_w: int, src_h: int, model_w: int, model_h: int
) -> tuple[int, int]:
    """Frame size for animating a src_w x src_h still with this model.

    Asking for the model's configured size verbatim squashes any still of a
    different aspect: a square still sent to 832x480 comes back visibly
    flattened, since the init image is resized to the requested frame. Holding
    the area and taking the aspect from the still keeps the composition and
    leaves generation time about the same.

    Unlike stage_dims, which answers to a ratio the user picked and can round
    it freely, this has to match a specific image, so it searches rather than
    solving in closed form: rounding each side independently can compound into
    a worse aspect than either side suggests (4:3 rounds to 704x576, i.e.
    1.22). Every grid width is tried and the pair closest to the still's aspect
    wins, ties going to the one nearest the pixel budget. Sizes far past the
    budget are rejected however well they fit: they are proportionally slower.
    """
    budget = max(1, model_w * model_h)
    aspect = (src_w / src_h) if src_w > 0 and src_h > 0 else 1.0
    best: tuple[float, float, tuple[int, int]] | None = None
    for w_units in range(_MIN_UNITS, _MAX_UNITS + 1):
        w = w_units * GRID
        h_units = max(_MIN_UNITS, min(_MAX_UNITS, round(w / aspect / GRID)))
        h = h_units * GRID
        area = w * h
        if not 0.6 * budget <= area <= 1.15 * budget:
            continue
        score = (abs((w / h) / aspect - 1), abs(area - budget))
        if best is None or score < best[:2]:
            best = (*score, (w, h))
    if best is None:
        # every grid pair was out of budget (a very extreme aspect): fall back
        # to the model's own shape rather than returning nothing
        return model_w, model_h
    return best[2]
