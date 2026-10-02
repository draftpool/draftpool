"""Grid geometry for generated images: the ratio buttons, and the sizes they
produce at each stage.

Two rules are checked here. The first is that a pixel budget is held: the two
sides come from the ratio at constant area, so picking a ratio does not quietly
cost extra time. The second is that aspect_size sets that budget for a
non-square render. Every shipped model sets it to its own final size, so the
lowering path is exercised here with synthetic values rather than real ones.

static/index.html mirrors both rules to say what a ratio will produce before it
runs, so the last test runs its copy and compares.
"""

import json
import math
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from app import dims
from app.registry import load_registry

ROOT = Path(__file__).resolve().parent.parent


def test_square_is_exactly_the_configured_size():
    # the default ratio must change nothing about what the app did before it
    # existed, at every stage size in config/models.yaml
    for size in (256, 512, 768, 1024, 1408, 1536, 2048, 3072, 4096):
        assert dims.stage_dims(size, 1.0) == (size, size)
        # and a lower off-square budget never applies to a square
        assert dims.stage_dims(size, 1.0, 1024) == (size, size)


def test_equal_area_matches_the_sdxl_buckets():
    """The shapes derived at constant area are the familiar buckets.

    Not a coincidence worth restating in code: SDXL's bucket list is itself
    roughly equal-area around 1024, so solving for it lands on the same sizes.
    """
    r34, r43 = dims.ratio("3:4"), dims.ratio("4:3")
    assert dims.stage_dims(1024, r34) == (896, 1152)
    assert dims.stage_dims(1024, r43) == (1152, 896)
    assert dims.stage_dims(1536, r34) == (1344, 1792)
    assert dims.stage_dims(1408, r34) == (1216, 1600)
    assert dims.stage_dims(1024, dims.ratio("9:16")) == (768, 1344)


def test_uncapped_shapes_hold_the_pixel_budget():
    for size in (768, 1024, 1408, 1536, 2048):
        for key in dims.ASPECTS:
            w, h = dims.stage_dims(size, dims.ratio(key))
            assert w % dims.GRID == 0 and h % dims.GRID == 0, (size, key)
            # time tracks pixel count, so a ratio must not change the bill
            assert 0.95 <= (w * h) / size**2 <= 1.05, (size, key, w, h)
            # and the shape asked for survives the grid
            assert abs((w / h) / dims.ratio(key) - 1) < 0.06, (size, key, w, h)


def test_a_lower_budget_keeps_the_shape_and_gives_up_area():
    # no shipped model sets a lower budget today, but the path has to work:
    # a 1536 stage held to a 1024 budget lands on SDXL's own training buckets
    assert dims.stage_dims(1536, dims.ratio("3:4"), 1024) == (896, 1152)
    assert dims.stage_dims(1536, dims.ratio("9:16"), 1024) == (768, 1344)
    assert dims.stage_dims(1536, dims.ratio("16:9"), 1024) == (1344, 768)
    assert dims.stage_dims(1536, dims.ratio("4:3"), 1024) == (1152, 896)
    for size in (1024, 1408, 1536, 2048):
        for key in dims.ASPECTS:
            w, h = dims.stage_dims(size, dims.ratio(key), 1024)
            assert w % dims.GRID == 0 and h % dims.GRID == 0
            if key != "1:1":
                # the area, not the long edge, is what is held down
                assert 0.95 <= (w * h) / 1024**2 <= 1.05, (size, key, w, h)
                assert abs((w / h) / dims.ratio(key) - 1) < 0.06, (size, key, w, h)


def test_the_off_square_budget_only_ever_shrinks():
    # a budget at or above the stage size must leave the equal-area answer alone
    for size in (768, 1024, 1536):
        for key in dims.ASPECTS:
            r = dims.ratio(key)
            uncapped = dims.stage_dims(size, r)
            assert dims.stage_dims(size, r, 8192) == uncapped, (size, key)
            assert dims.stage_dims(size, r, size) == uncapped, (size, key)
            capped = dims.stage_dims(size, r, 1024)
            assert capped[0] <= uncapped[0] and capped[1] <= uncapped[1]


def test_ratios_are_transposes_of_each_other():
    # Portrait must be exactly Landscape on its side, or the pair reads as two
    # unrelated shapes in the picker
    for size in (768, 1024, 1536):
        for a, b in (("4:3", "3:4"), ("3:2", "2:3"), ("16:9", "9:16")):
            for cap in (None, 1024):
                w, h = dims.stage_dims(size, dims.ratio(a), cap)
                assert dims.stage_dims(size, dims.ratio(b), cap) == (h, w)


def test_no_side_is_ever_degenerate():
    for size in (0, 64, 256, 4096):
        for key in dims.ASPECTS:
            for cap in (None, 64, 1024):
                w, h = dims.stage_dims(size, dims.ratio(key), cap)
                assert w >= dims.GRID and h >= dims.GRID, (size, key, cap)


def test_nearest_aspect_recovers_a_shape_from_pixels():
    # the fallback for rows predating the feature, and for pixels that came
    # from elsewhere (a Real-ESRGAN 2x, a region repaint)
    assert dims.nearest_aspect(1024, 1024) == "1:1"
    assert dims.nearest_aspect(896, 1152) == "3:4"
    assert dims.nearest_aspect(1152, 896) == "4:3"
    assert dims.nearest_aspect(1344, 768) == "16:9"
    assert dims.nearest_aspect(768, 1344) == "9:16"
    # a grid-rounded draft still resolves to the ratio it was asked for
    assert dims.nearest_aspect(640, 896) == "3:4"
    assert dims.nearest_aspect(1024, 1344) == "3:4"
    # doubled by an ESRGAN pass, and degenerate input
    assert dims.nearest_aspect(1792, 2304) == "3:4"
    assert dims.nearest_aspect(0, 0) == "1:1"


def test_every_shape_round_trips_through_nearest_aspect():
    for size in (768, 1024, 1408, 1536):
        for key in dims.ASPECTS:
            for cap in (None, 1024):
                w, h = dims.stage_dims(size, dims.ratio(key), cap)
                assert dims.nearest_aspect(w, h) == key, (size, key, cap, w, h)


def test_a_coarse_budget_cannot_resolve_every_ratio():
    """At a small budget the grid runs out of steps, and that is visible.

    FLUX drafts at 256, i.e. four grid steps a side, so 4:3 and 16:9 both round
    to 320x192 and the two buttons produce the same draft. Recorded rather than
    fixed: the fix would be to round the draft away from its budget, which
    makes the cheap preview stage cost more for no better preview, and the
    final the draft is picked for does resolve them (1152x896 vs 1344x768).
    """
    r43, r32, r169 = dims.ratio("4:3"), dims.ratio("3:2"), dims.ratio("16:9")
    # 3:2 falls in the same bucket, so at 256 three buttons give one draft
    assert dims.stage_dims(256, r43) == dims.stage_dims(256, r32) \
        == dims.stage_dims(256, r169) == (320, 192)
    # by 768 the grid has the resolution to tell them apart -- and 3:2 lands
    # exactly, 960x640, where the older pair still round
    assert dims.stage_dims(768, r43) != dims.stage_dims(768, r169)
    assert dims.stage_dims(768, r32) == (960, 640)
    assert dims.stage_dims(1024, r43) == (1152, 896)
    assert dims.stage_dims(1024, r169) == (1344, 768)


def test_ratio_falls_back_to_square_for_an_unknown_name():
    assert dims.ratio("2:1") == 1.0
    assert dims.ratio("") == 1.0


def test_configured_models_stay_inside_their_measured_budget():
    """The guard is only as good as the value it reads, so use the real config.

    Drafts and finals are compositions and must respect the budget. The
    upscale stage deliberately does not (it is img2img at low strength over an
    init that already holds the composition), so it is not checked here.

    The area tolerance is one grid step per side rather than a flat 5%: FLUX
    drafts at 256, where a step is a quarter of the budget and nothing lands
    closer. See test_a_coarse_budget_cannot_resolve_every_ratio.
    """
    reg = load_registry()
    for spec in reg.models.values():
        if not spec.supports_aspect:
            continue
        for stage in (spec.draft, spec.final):
            for key in dims.ASPECTS:
                w, h = dims.stage_dims(
                    stage.size, dims.ratio(key), spec.aspect_size
                )
                if key == "1:1":
                    assert (w, h) == (stage.size, stage.size), spec.key
                else:
                    budget = min(stage.size, spec.aspect_size)
                    tol = dims.GRID / budget
                    assert (
                        (1 - tol) ** 2 <= (w * h) / budget**2 <= (1 + tol) ** 2
                    ), (spec.key, key, w, h)


def test_budget_reads_back_the_stage_size_a_picture_was_rendered_at():
    """Round-trip through stage_dims: what went in comes back, give or take.

    The grid moves the sides, so the answer is not always exact -- 1536 at 3:4
    reads back as 1552 -- but it lands inside a grid step of the budget for
    every shape, which is what the Upscale offer needs from it.
    """
    for size in (512, 768, 1024, 1536, 2048, 3072):
        for key in dims.ASPECTS:
            w, h = dims.stage_dims(size, dims.ratio(key))
            assert abs(dims.budget(w, h) - size) <= dims.GRID, (size, key, w, h)


def test_budget_beats_the_long_edge_at_deciding_which_picture_is_larger():
    """The reason the comparison is an area and not a side.

    2304x4096 is a 9:16 picture at the 2048 budget after a 2x fast. Its long
    edge is already 4096, so a rule written on the long edge refuses the 4096
    pass -- which would in fact return 3072x5440, nearly twice the pixels. The
    long edge is never below the budget, so it can only ever refuse too much.
    """
    tall = (2304, 4096)
    assert dims.budget(*tall) == 3072     # what the picture is, as an area
    assert max(tall) == 4096              # what its long edge claims it is
    w, h = dims.stage_dims(4096, dims.ratio("9:16"))
    assert (w, h) == (3072, 5440)
    assert w * h > tall[0] * tall[1]      # so the pass really does enlarge it


def test_budget_of_nothing_is_zero_rather_than_an_error():
    """A row whose size is not known yet must not take the button away."""
    assert dims.budget(0, 0) == 0
    assert dims.budget(-1, 512) == 0


def test_labels_and_ratios_are_defined_for_every_preset():
    for key, (w, h) in dims.ASPECTS.items():
        assert key in dims.ASPECT_LABELS
        assert math.isclose(dims.ratio(key), w / h)
    assert dims.DEFAULT_ASPECT in dims.ASPECTS
    # square first: it is the default and the picker leads with it
    assert next(iter(dims.ASPECTS)) == dims.DEFAULT_ASPECT


# every combination the UI can be asked to quote, including the caps that bind
# and the coarse budgets where the grid rounds hardest
CASES = [
    (size, key, cap)
    for size in (256, 768, 1024, 1408, 1536, 2048)
    for key in ("1:1", "4:3", "3:4", "16:9", "9:16")
    for cap in (0, 1024, 1536)
]


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_ui_computes_the_same_sizes_as_the_server():
    html = (ROOT / "static" / "index.html").read_text()
    fn = re.search(r"^function stageDims\(size, r, aspectSize\) \{.*?^\}", html, re.S | re.M)
    assert fn, "stageDims missing from static/index.html"
    budget_fn = re.search(r"^function pixelBudget\(width, height\) \{.*?^\}", html, re.S | re.M)
    assert budget_fn, "pixelBudget missing from static/index.html"
    ratios = {k: dims.ratio(k) for k in dims.ASPECTS}
    script = (
        f"{fn.group(0)}\n"
        f"{budget_fn.group(0)}\n"
        f"const ratios = {json.dumps(ratios)};\n"
        f"const cases = {json.dumps(CASES)};\n"
        "console.log(JSON.stringify(cases.map(([size, key, cap]) => {"
        # the card filters the Upscale sizes with pixelBudget, so it has to
        # agree with the server that refuses the ones it lets through
        "const wh = stageDims(size, ratios[key], cap || null);"
        "return [wh[0], wh[1], pixelBudget(wh[0], wh[1])]; })));"
    )
    out = subprocess.run(
        ["node", "-e", script], capture_output=True, text=True, check=True
    ).stdout
    expected = [
        [*dims.stage_dims(size, dims.ratio(key), cap or None),
         dims.budget(*dims.stage_dims(size, dims.ratio(key), cap or None))]
        for size, key, cap in CASES
    ]
    assert json.loads(out) == expected
