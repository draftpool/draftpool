"""The measurement harness's own logic, minus the renders.

`scripts/add_model.py` decides three things before it ever calls Draw Things,
and all three are the kind of thing that fails quietly: whether two checkpoints
are the same architecture (so whether `--like` may inherit anything at all),
which cfgs to sweep, and how a contact sheet is cropped. None of that needs a
render, so all of it is tested here.

The sheet case is a regression test. `crop_sheet` centre-cropped at native
pixels for every sheet, which is right for reading an artefact and wrong for
the duplicate-subject sheet: a 512 crop of a 1536 render is a third of a frame,
so that sheet arrived as a torso and could not answer the only question it
existed to ask (docs/decisions/MODEL-MEASUREMENTS.md).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.add_model import (  # noqa: E402
    INHERITED,
    MEASURED_SOURCES,
    architecture,
    cfg_candidates,
    crop_sheet,
)


WHITE, GROUND = (255, 255, 255), (120, 60, 30)


def _png(path: Path, size: tuple[int, int]) -> Path:
    """A frame with a solid white top-left quarter on a dark ground.

    The marker is a solid block rather than a sparse pattern on purpose: it has
    to survive a LANCZOS downscale so that `whole=True` can be distinguished
    from a centre crop by pixel content, not only by tile size. A thin pattern
    averages into the ground and makes the test fail for the wrong reason.
    """
    img = Image.new("RGB", size, GROUND)
    img.paste(WHITE, (0, 0, size[0] // 4, size[1] // 4))
    img.save(path)
    return path


def _body(sheet_path: Path, crop: int = 512) -> Image.Image:
    """The first tile, with the label bar above it removed."""
    return Image.open(sheet_path).crop((0, 22, crop, crop + 22))


class TestCropSheet:
    def test_crop_mode_keeps_native_pixels_rather_than_downscaling(self, tmp_path):
        """1:1 is the whole point for an artefact sheet.

        Rainbow speckle is a per-pixel artefact and the numeric instruments for
        it were dropped, so the verdict is an eye on a 1:1 crop. Downscaling to
        fit would erase exactly what is being looked for, so a cell larger than
        the tile must be cropped, never resized.
        """
        big = _png(tmp_path / "big.png", (1536, 1536))
        out = crop_sheet([("a", big)], tmp_path / "s.png", crop=512)
        sheet = Image.open(out)
        assert sheet.size == (512, 512 + 22)
        # the centre 512 of a 1536 frame is x/y 512..1024, well outside the
        # top-left quarter, so the marker cannot appear
        assert _body(out).getpixel((5, 5)) == GROUND

    def test_whole_mode_fits_the_frame_so_the_composition_is_visible(self, tmp_path):
        """The regression. A duplicated subject is a whole-frame fact.

        The first version of the final-size sheet cropped 1:1 like the artefact
        sheet, which turned a 1536 render into a torso. `whole=True` must show
        the entire frame, including the corners a centre crop discards.
        """
        big = _png(tmp_path / "big.png", (1536, 1536))
        out = crop_sheet([("a", big)], tmp_path / "w.png", crop=512, whole=True)
        sheet = Image.open(out)
        assert sheet.size == (512, 512 + 22)
        # the marker is in frame now, which is exactly the difference: the
        # corners a centre crop discards are where a duplicated subject shows
        assert _body(out).getpixel((5, 5)) == WHITE

    def test_the_two_modes_actually_differ_on_the_same_input(self, tmp_path):
        """Guards against the flag being accepted and ignored.

        A mode flag that silently does nothing is how this bug shipped in the
        first place, so compare the bytes rather than trusting the parameter.
        """
        big = _png(tmp_path / "big.png", (1536, 1536))
        cropped = crop_sheet([("a", big)], tmp_path / "c.png", crop=512)
        whole = crop_sheet([("a", big)], tmp_path / "w.png", crop=512, whole=True)
        assert cropped.read_bytes() != whole.read_bytes()

    def test_a_frame_smaller_than_the_tile_is_scaled_up_in_both_modes(self, tmp_path):
        """A 256px smoke render must still fill its tile rather than sit in a
        corner, and must not make the sheet a different size from its siblings.
        """
        small = _png(tmp_path / "small.png", (256, 256))
        for whole in (False, True):
            out = crop_sheet([("a", small)], tmp_path / f"s{whole}.png",
                             crop=512, whole=whole)
            assert Image.open(out).size == (512, 512 + 22)

    def test_missing_cells_are_skipped_and_an_empty_sheet_is_none(self, tmp_path):
        """A failed render has no path, and the sheet still has to be written.

        Every cell here can come back blocked, flat or empty, so the sheet is
        built from whatever succeeded. No cells at all returns None rather than
        writing a black image that looks like a rendering failure.
        """
        ok = _png(tmp_path / "ok.png", (768, 768))
        out = crop_sheet([("a", ok), ("gone", tmp_path / "nope.png")],
                         tmp_path / "s.png", crop=512)
        # one tile, not two: the sheet is one column wide
        assert Image.open(out).size == (512, 512 + 22)
        assert crop_sheet([("gone", tmp_path / "nope.png")], tmp_path / "e.png") is None
        assert crop_sheet([], tmp_path / "e2.png") is None

    def test_cells_wrap_at_four_columns(self, tmp_path):
        """The sampler x cfg grid is eight cells and must not be one long row."""
        cells = [(f"c{i}", _png(tmp_path / f"{i}.png", (768, 768))) for i in range(8)]
        out = crop_sheet(cells, tmp_path / "s.png", crop=256)
        assert Image.open(out).size == (4 * 256, 2 * (256 + 22))


class TestArchitecture:
    def test_identical_checkpoints_report_identical_signatures(self):
        """What licenses --like at all.

        The two Illustrious/SDXL merges already in the registry are the same
        machine as each other, so their signatures must match exactly apart
        from where the signature was read from. If this stops holding, --like
        is comparing something that is not architecture.
        """
        a = architecture("cyberrealistic_illustriousanimev50_f16.ckpt")
        b = architecture("oneobsession_v24_f16.ckpt")
        if not a or not b:
            pytest.skip("Draw Things' custom.json is not present on this machine")
        assert a == b
        assert a["version"] == "sdxl_base_v0.9"

    def test_a_different_base_reports_a_different_signature(self):
        """The refusal has to have something to fire on.

        A Cosmos 2B checkpoint and an SDXL one must not compare equal, or the
        guard passes everything and inherits geometry across architectures.
        """
        sdxl = architecture("oneobsession_v24_f16.ckpt")
        cosmos = architecture("basbetteranimestyleplusanima_basev10_f16.ckpt")
        if not sdxl or not cosmos:
            pytest.skip("Draw Things' custom.json is not present on this machine")
        assert sdxl != cosmos
        assert sdxl["version"] != cosmos["version"]

    def test_an_unknown_file_has_no_signature_rather_than_a_default_one(self):
        """An empty dict is what makes the script refuse instead of guess.

        A signature that defaulted to something plausible would compare equal
        to a real one and wave the checkpoint through.
        """
        assert architecture("not_a_real_checkpoint_xyz_f16.ckpt") == {}

    def test_the_signature_covers_the_modifier(self):
        """v-prediction is the case this has to catch.

        rinFlanime's near-namesake on Civitai is a v-pred checkpoint that
        shares a filename stem, and a v-pred model needs a different recipe
        entirely -- so `modifier` has to be part of what is compared, not just
        the base version.
        """
        sig = architecture("oneobsession_v24_f16.ckpt")
        if not sig:
            pytest.skip("Draw Things' custom.json is not present on this machine")
        assert "modifier" in sig and "autoencoder" in sig and "version" in sig


class TestInheritanceContract:
    def test_sampler_cfg_and_draft_are_never_inheritable(self):
        """The three fields MODEL-MEASUREMENTS.md records as copied and wrong.

        janku then proved the point from the other direction: the unanimous
        four-of-four DPM++ 2M Karras prior predicted the wrong sampler for it,
        and only a structural refusal to inherit the field kept the measured
        value. If one of these ever appears in INHERITED, that protection is
        gone.
        """
        for field in ("sampler", "cfg", "draft"):
            assert field not in INHERITED, field

    def test_every_measured_source_is_a_real_registry_key(self):
        """A rename must not silently empty the allowlist.

        MEASURED_SOURCES is the gate on --like; a key that no longer exists
        cannot be selected, so the gate would quietly stop protecting anything.
        """
        from app.registry import CONFIG_PATH, load_registry

        keys = set(load_registry(CONFIG_PATH).models)
        assert MEASURED_SOURCES <= keys, sorted(MEASURED_SOURCES - keys)


class TestCfgCandidates:
    def test_the_card_value_and_about_a_third_lower(self):
        """The grid's shape, and it is this document's own experiment.

        Two cfgs are what separate CFG burn from a bad sampler: both look like
        speckle, and only the cfg column moving tells them apart. The pair for
        6.0 is 6.0/4.0 exactly, because that is the comparison that found the
        burn on the two Illustrious merges.
        """
        assert cfg_candidates(6.0) == [6.0, 4.0]
        assert cfg_candidates(5.0) == [5.0, 3.5]

    def test_always_highest_first_and_rounded_to_halves(self):
        """Order matters: the first entry is what the finals are rendered at,
        and halves are the resolution anything here is ever set at.
        """
        for card in (3.0, 4.0, 5.0, 6.0, 7.0, 4.5):
            got = cfg_candidates(card)
            assert got == sorted(got, reverse=True), card
            assert got[0] == card or abs(got[0] - card) <= 0.25, card
            assert all(v * 2 == int(v * 2) for v in got), got

    def test_a_distilled_cfg_1_recipe_is_not_swept(self):
        """The guard. A cfg-1 checkpoint carries a few-step distilled recipe:
        there is no classifier-free guidance to burn, nothing below 1 to sweep
        to, and this repo's standing advice is to leave the recipe alone --
        DPM++ 2M Karras on bas-anima renders pure noise rather than merely
        degrading. Sweeping it to 0.5 would be the same class of mistake.
        """
        assert cfg_candidates(1.0) == [1.0]
        assert cfg_candidates(0.8) == [0.8]

    def test_no_candidate_is_ever_below_one(self):
        """Belt and braces on the above, across the whole plausible range."""
        for card in [x / 2 for x in range(2, 31)]:
            assert all(v >= 1.0 for v in cfg_candidates(card)), card

    def test_the_sweep_is_never_more_than_two_renders_wide(self):
        """The whole point is a short batch; a third cfg doubles the grid."""
        for card in [x / 2 for x in range(2, 31)]:
            assert len(cfg_candidates(card)) <= 2, card
