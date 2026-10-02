"""A full-image edit runs at the model's working budget, in the picture's own
shape. The edit dialogs warn what size that comes back at; that figure is
computed in JavaScript from the same rule the endpoint applies. Two copies of
one rule drift, and the drift is invisible: the dialog would simply quote a
size the job never produces. This runs both and compares them."""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from app import dims
from app.main import edit_ceiling, edit_dims, edit_frame, edit_target
from app.registry import load_registry

ROOT = Path(__file__).resolve().parent.parent

SIZES = [2048, 3072, 4096]


class _Spec:
    """Just the fields the edit sizing reads."""

    def __init__(self, final_size, tiled, edit_only=False):
        self.final = type("S", (), {"size": final_size})()
        self.tiled_diffusion = tiled
        self.edit_only = edit_only


# squares, both landscape and portrait, sizes off the grid, a strip past any
# offered shape, and the shapes the app itself draws at 1024 and 1536
CASES = [
    (4096, 4096, 1024, 2048),
    (2048, 1152, 1024, 2048),
    (1152, 2048, 1536, 2048),
    (2048, 2048, 1408, 2048),
    (3072, 1728, 1536, 2048),
    (768, 768, 1024, 2048),
    (4096, 512, 1024, 2048),
    (1344, 768, 1024, 2048),
    (1280, 832, 1024, 2048),
    (4096, 2304, 3072, 4096),
    (4096, 2304, 3072, 2048),
]


def test_edit_dims_holds_the_budget_in_the_pictures_shape():
    for w, h, size, ceiling in CASES:
        ow, oh = edit_dims(w, h, size, ceiling)
        assert ow % 64 == 0 and oh % 64 == 0, (w, h, size)
        # the aspect survives to within one grid step on the short side
        assert abs(ow / oh - w / h) < 0.1, (w, h, size)
        assert max(ow, oh) <= ceiling, (w, h, size)
        if max(ow, oh) < ceiling:
            # the budget is an area: the frame is what the app draws that
            # shape at, not the long edge scaled down to the size
            assert (ow, oh) == dims.stage_dims(size, w / h), (w, h, size)
            assert dims.stage_size(ow, oh) == size, (w, h, size)


def test_the_apps_own_wide_frame_is_not_shrunk():
    # the case that exposed the long-edge rule: 1344x768 is what a 1024 model
    # draws 16:9 at, and it was being handed back as 1024x576 -- 56% of the
    # area, below what an instruction editor works at
    assert edit_dims(1344, 768, 1024, 2048) == (1344, 768)
    assert edit_dims(2048, 1152, 1024, 2048) == (1344, 768)
    assert edit_dims(1024, 576, 1024, 2048) == (1344, 768)


def test_a_strip_gives_up_budget_to_keep_the_long_edge_under_the_ceiling():
    w, h = edit_dims(4096, 512, 1024, 2048)
    assert max(w, h) == 2048
    assert dims.stage_size(w, h) < 1024
    # a picture over the ceiling comes back at the ceiling, in shape
    assert edit_dims(4096, 2304, 3072, 2048) == (2048, 1152)
    assert edit_dims(4096, 2304, 3072, 4096) == (4096, 2304)


def test_edit_frame_leaves_a_picture_at_the_budget_alone():
    editor = _Spec(1024, True, edit_only=True)
    generator = _Spec(1024, False)
    for spec in (editor, generator):
        for wh in [(1024, 1024), (1344, 768), (768, 1344), (1280, 832), (1152, 896)]:
            assert edit_frame(spec, *wh, False, SIZES) == wh, (spec.edit_only, wh)
    # larger is brought down to the budget, in shape
    assert edit_frame(editor, 2048, 1152, False, SIZES) == (1344, 768)
    assert edit_frame(generator, 4096, 4096, False, SIZES) == (1024, 1024)
    # an instruction editor scales a small draft up to what it works at; a
    # generator's img2img just runs at the draft's own size
    assert edit_frame(editor, 1024, 576, False, SIZES) == (1344, 768)
    assert edit_frame(editor, 768, 768, False, SIZES) == (1024, 1024)
    assert edit_frame(generator, 1024, 576, False, SIZES) == (1024, 576)


def test_full_size_is_the_pictures_own_budget_capped_at_what_renders():
    tiled = _Spec(1536, True)
    untiled = _Spec(1024, False)
    # a tiled model may go all the way up to the largest offered size
    assert edit_frame(tiled, 4096, 4096, True, SIZES) == (4096, 4096)
    assert edit_frame(tiled, 4096, 2304, True, SIZES) == (4096, 2304)
    # an untiled one stops at a 2048 long edge however large the picture is:
    # past its ceiling it returns a flat colour field and reports success
    assert edit_frame(untiled, 4096, 4096, True, SIZES) == (2048, 2048)
    assert edit_frame(untiled, 4096, 2304, True, SIZES) == (2048, 1152)
    # the flag never renders smaller than the working size, which is what an
    # instruct model needs from a small draft
    assert edit_target(tiled, 768, 768, True) == 1536
    # and off, it is always the working size
    for spec in (tiled, untiled):
        assert edit_target(spec, 4096, 4096, False) == spec.final.size


def test_no_installed_model_can_ask_for_more_than_it_renders():
    # the guard is only as good as the ceiling it reads, so check it against
    # the real config rather than a stand-in
    reg = load_registry()
    for spec in reg.models.values():
        w, h = edit_frame(spec, 8192, 4608, True, reg.upscale_sizes)
        assert max(w, h) <= edit_ceiling(spec, reg.upscale_sizes), spec.key
        if not spec.tiled_diffusion:
            assert max(w, h) <= 2048, spec.key


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_ui_computes_the_same_edit_size_as_the_server():
    html = (ROOT / "static" / "index.html").read_text()
    parts = []
    for sig in ("editDims(g, size, ceiling)", "stageDims(size, r, aspectSize)",
                "stageSize(width, height)", "fullSizeTarget(g, model)"):
        fn = re.search(rf"^function {re.escape(sig)} \{{.*?^\}}", html, re.S | re.M)
        assert fn, f"{sig} missing from static/index.html"
        parts.append(fn.group(0))
    script = (
        "\n".join(parts) + "\n"
        f"const cases = {json.dumps(CASES)};\n"
        "console.log(JSON.stringify(cases.map(([w, h, s, c]) => [\n"
        "  ...editDims({ width: w, height: h }, s, c),\n"
        "  fullSizeTarget({ width: w, height: h }, { allow_full_edit: true, edit_size: s })])));"
    )
    out = subprocess.run(
        ["node", "-e", script], capture_output=True, text=True, check=True
    ).stdout
    expected = [
        [*edit_dims(w, h, s, c), edit_target(_Spec(s, True), w, h, True)]
        for w, h, s, c in CASES
    ]
    assert json.loads(out) == expected
