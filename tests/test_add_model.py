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
    measured_sources,
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

        Two Anima finetunes in the registry are the same machine as each
        other, so their signatures must match exactly apart from where the
        signature was read from. If this stops holding, --like is comparing
        something that is not architecture.
        """
        a = architecture("hosekilustrousmixanima_animav10_f16.ckpt")
        b = architecture("cyberrealisticanima_v60aestheticv11_f16.ckpt")
        if not a or not b:
            pytest.skip("Draw Things' custom.json is not present on this machine")
        assert a == b
        assert a["version"] == "cosmos2.5_2b"

    def test_a_different_base_reports_a_different_signature(self):
        """The refusal has to have something to fire on.

        A Cosmos 2B checkpoint and an SDXL one must not compare equal, or the
        guard passes everything and inherits geometry across architectures.
        """
        sdxl = architecture("juggernaut_xl_ragnarok_f16.ckpt")
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

        One Illustrious finetune's near-namesake on Civitai is a v-pred checkpoint that
        shares a filename stem, and a v-pred model needs a different recipe
        entirely -- so `modifier` has to be part of what is compared, not just
        the base version.
        """
        sig = architecture("juggernaut_xl_ragnarok_f16.ckpt")
        if not sig:
            pytest.skip("Draw Things' custom.json is not present on this machine")
        assert "modifier" in sig and "autoencoder" in sig and "version" in sig


class TestInheritanceContract:
    def test_sampler_cfg_and_draft_are_never_inheritable(self):
        """The three fields MODEL-MEASUREMENTS.md records as copied and wrong.

        An Illustrious finetune then proved the point from the other direction: the unanimous
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
        The private models marked as measured count too, when the file is here.
        """
        from app.registry import CONFIG_PATH, load_registry

        keys = set(load_registry(CONFIG_PATH).models)
        assert MEASURED_SOURCES <= keys, sorted(MEASURED_SOURCES - keys)
        assert measured_sources() <= keys, sorted(measured_sources() - keys)


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


def test_a_sibling_is_measured_without_its_optional_loras():
    """--like copies the sibling as shipped, not with a default-on accelerator.

    lustrousmix-anima ships its Turbo LoRA ticked, and a confirm run against
    it printed `final: steps 8, cfg 1.0, loras [turbo]` as the geometry to
    inherit, then rendered the step-direction sheet at 8 x 1.5 = 12 steps.
    """
    from scripts.add_model import measuring_registry

    sib = measuring_registry().models["lustrousmix-anima"]
    assert sib.final.loras == []
    assert sib.final.steps == sib.final.default_steps == 24
    assert sib.final.cfg is None


def test_a_confirm_run_reaches_its_report(tmp_path, capsys):
    """The report at the end of a confirm run is what names the sheets.

    It used `args.key`, and `args` only exists in main(), so every confirm run
    rendered its whole grid, wrote the sheets and then died on a NameError
    before saying where they were or what to look for.
    """
    from types import SimpleNamespace

    from scripts.add_model import confirm

    frame = _png(tmp_path / "frame.png", (256, 256))

    class Stub:
        results: dict = {}

        def control(self):
            pass

        def render(self, name, **job):
            self.results[name] = {"ok": True, "path": str(frame), "seconds": 1.0}
            return self.results[name]

    stage = SimpleNamespace(size=256, steps=4)
    sib = SimpleNamespace(draft=stage, final=stage, clip_skip=1)
    assert confirm(Stub(), tmp_path, "m.ckpt", "new-key", "sib-key", sib,
                   ["Euler a"], [4.0, 2.5], 4) == 0
    assert "sweep_aspect.py --models new-key" in capsys.readouterr().out


# -- what a measured cell is sent -------------------------------------------------
#
# Every add_model.py cell went out as bare prose with an empty negative, which a
# Danbooru model never gets in the app (prompt_for, negative_for). Animagine's
# whole confirm run came back soft for it, and the same seed with its opener and
# negative is crisp (research/animagine-prompt/, MODEL-MEASUREMENTS.md).


class _FakeDrawThings:
    """httpx.Client for Renderer: records each payload, answers with a picture."""

    sent: list = []

    def __init__(self, *a, **kw):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def post(self, url, json):
        import base64
        import io

        type(self).sent.append(json)
        buf = io.BytesIO()
        Image.effect_noise((64, 64), 64).convert("RGB").save(buf, "PNG")

        class R:
            status_code = 200

            def json(self_inner):
                return {"images": [base64.b64encode(buf.getvalue()).decode()]}
        return R()


@pytest.fixture
def fake_dt(monkeypatch):
    import scripts.add_model as am

    _FakeDrawThings.sent = []
    monkeypatch.setattr(am.httpx, "Client", _FakeDrawThings)
    return _FakeDrawThings.sent


def test_a_measured_cell_gets_the_opener_and_negative_and_the_control_does_not(tmp_path, fake_dt):
    from scripts.add_model import Renderer

    r = Renderer(tmp_path, "control.ckpt", opener="masterpiece, best quality",
                 negative="lowres, worst quality")
    r.render("cell", model="m.ckpt", prompt="A woman in armour.", width=64, height=64, steps=4)
    r.render("_c", model="control.ckpt", prompt="a red cube", width=64, height=64, steps=4)
    cell, control = fake_dt
    assert cell["prompt"] == "masterpiece, best quality, A woman in armour."
    assert cell["negative_prompt"] == "lowres, worst quality"
    assert (control["prompt"], control["negative_prompt"]) == ("a red cube", "")


def test_a_cell_measured_with_another_prompt_is_rendered_again(tmp_path, fake_dt):
    from scripts.add_model import Renderer

    Renderer(tmp_path, None).render("cell", model="m.ckpt", prompt="p", width=64,
                                    height=64, steps=4)
    again = Renderer(tmp_path, None, opener="masterpiece", negative="lowres")
    rec = again.render("cell", model="m.ckpt", prompt="p", width=64, height=64, steps=4)
    assert len(fake_dt) == 2 and rec["opener"] == "masterpiece"
    # the same kit resumes as before
    again.render("cell", model="m.ckpt", prompt="p", width=64, height=64, steps=4)
    assert len(fake_dt) == 2


def test_the_prompt_kit_takes_the_card_then_the_model_then_the_tag_default():
    from types import SimpleNamespace

    from app.schemas import DEFAULT_TAG_NEGATIVE, DEFAULT_TAG_PREFIX
    from scripts.add_model import prompt_kit

    tag_spec = SimpleNamespace(prompt_style="tags", quality_opener="score_9",
                               default_negative="bad", uses_negative_prompt=True)
    prose_spec = SimpleNamespace(prompt_style="prose", quality_opener="",
                                 default_negative="", uses_negative_prompt=True)
    assert prompt_kit(tags=False, opener="card", negative="neg", spec=tag_spec) == ("card", "neg")
    assert prompt_kit(tags=False, opener=None, negative=None, spec=tag_spec) == ("score_9", "bad")
    assert prompt_kit(tags=True, opener=None, negative=None, spec=None) == (
        DEFAULT_TAG_PREFIX, DEFAULT_TAG_NEGATIVE)
    # a model set up as prose but measured as tags: the flag wins over its entry
    assert prompt_kit(tags=True, opener=None, negative=None, spec=prose_spec)[0] == DEFAULT_TAG_PREFIX
    assert prompt_kit(tags=False, opener=None, negative=None, spec=prose_spec) == ("", "")


def test_a_batch_always_finds_a_control_model_when_one_is_installed():
    """The control used to be the first model drafting at 512 or less. Every
    draft went to 768 (DRAFT-STEPS.md), the search found nothing, and a batch
    ran with no control renders at all -- the bracketing FALSE-SUCCESS.md
    exists for, silently gone. It only renders 256px at 4 steps, so the
    criterion is a fast model that is installed, never the one measured."""
    from types import SimpleNamespace

    from scripts.add_model import control_model

    def m(f, installed=True, edit=False):
        return SimpleNamespace(dt_filename=f, installed=installed, edit_only=edit)

    models = {"juggernaut-ragnarok": m("j.ckpt"), "z-image-turbo": m("z.ckpt"),
              "qwen-edit": m("q.ckpt", edit=True)}
    assert control_model(models, "new").dt_filename == "z.ckpt"
    assert control_model(models, "z-image-turbo").dt_filename == "j.ckpt"
    models["z-image-turbo"].installed = False
    assert control_model(models, "new").dt_filename == "j.ckpt"
    assert control_model({"qwen-edit": m("q.ckpt", edit=True)}, "new") is None


def test_the_control_render_names_its_sampler(tmp_path, fake_dt):
    """Sent without one, Draw Things uses whatever its UI is left on, and
    Z-Image Turbo answered every control of a batch with an empty image --
    the harness's own health check reporting a healthy session as stalled.
    The same request with its sampler renders."""
    from scripts.add_model import Renderer

    r = Renderer(tmp_path, "control.ckpt", control_sampler="Euler A Trailing")
    assert r.control()
    assert fake_dt[0]["model"] == "control.ckpt"
    assert fake_dt[0]["sampler_name"] == "Euler A Trailing"


def test_a_control_that_draws_its_red_square_flat_is_a_pass(tmp_path, monkeypatch):
    """The control asks for "a red square", and Z-Image Turbo draws exactly
    that: a solid red frame, standard deviation about 1. The flat-colour
    check, there for measured cells that came back as nothing, failed every
    control of a batch on it. A control proves the session renders; a flat
    frame of the colour it asked for does. A measured cell is still failed."""
    import base64
    import io

    import scripts.add_model as am
    from scripts.add_model import Renderer

    class Flat(_FakeDrawThings):
        def post(self, url, json):
            buf = io.BytesIO()
            Image.new("RGB", (64, 64), (206, 2, 7)).save(buf, "PNG")

            class R:
                status_code = 200

                def json(self_inner):
                    return {"images": [base64.b64encode(buf.getvalue()).decode()]}
            return R()

    monkeypatch.setattr(am.httpx, "Client", Flat)
    r = Renderer(tmp_path, "control.ckpt", control_sampler="Euler A Trailing")
    assert r.control()
    cell = r.render("cell", model="m.ckpt", prompt="p", width=64, height=64, steps=4)
    assert not cell["ok"] and "flat" in cell["why"]
