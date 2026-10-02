import re
from pathlib import Path

import pytest

from app.registry import CONFIG_PATH, load_registry
from app.schemas import DEFAULT_TAG_NEGATIVE, DEFAULT_TAG_PREFIX


def test_loads_default_config():
    """The shipped YAML parses into three usable stages.

    Deliberately no longer pins the sizes themselves. This asserted klein's
    draft was 256 and broke the day that was measured up to 512 -- which is a
    test failing for a config change that was the point of the change. The
    numbers are measured values that move; the shape is the invariant.
    """
    reg = load_registry(CONFIG_PATH)
    assert "flux2-klein" in reg.models
    spec = reg.models["flux2-klein"]
    assert spec.draft.size > 0 and spec.draft.steps > 0
    assert spec.final.size > 0 and spec.upscale.size > 0
    assert 0 < spec.upscale.strength < 1


def test_every_model_orders_its_stages():
    """draft <= final <= upscale, for all of them.

    The real invariant behind the sizes: a draft is a cheap preview of a final,
    and an upscale enlarges one. A model whose draft rendered larger than its
    final would be a typo nobody would notice from the app.
    """
    reg = load_registry(CONFIG_PATH)
    wrong = {
        k: (m.draft.size, m.final.size, m.upscale.size)
        for k, m in reg.models.items()
        if not 0 < m.draft.size <= m.final.size <= m.upscale.size
    }
    assert not wrong, f"stages out of order: {wrong}"


def test_unavailable_model_raises():
    reg = load_registry(CONFIG_PATH)
    empty = next((k for k, m in reg.models.items() if not m.dt_filename), None)
    if empty is None:
        pytest.skip("all models configured")
    with pytest.raises(ValueError):
        reg.model_or_raise(empty)


def test_unknown_model_raises():
    reg = load_registry(CONFIG_PATH)
    with pytest.raises(KeyError):
        reg.model_or_raise("nope")


def test_custom_config(tmp_path: Path):
    cfg = tmp_path / "models.yaml"
    cfg.write_text(
        """
llm: { model: test, host: "http://x", temperature: 0.5 }
draw_things: { host: "http://y" }
models:
  m1:
    label: Test
    dt_filename: test.ckpt
    cfg: 2.0
    draft: { size: 128, steps: 2 }
    final: { size: 512, steps: 10 }
    upscale: { size: 1024, steps: 8, strength: 0.4 }
"""
    )
    reg = load_registry(cfg)
    assert reg.model_or_raise("m1").available
    assert reg.llm.model == "test"


def _spec(dt_filename, loras=None, style="prose"):
    from app.schemas import ModelSpec

    return ModelSpec(
        key="m",
        label="M",
        dt_filename=dt_filename,
        prompt_style=style,
        draft={"size": 256, "steps": 4, "loras": loras or []},
        final={"size": 1024, "steps": 8},
        upscale={"size": 2048, "steps": 6, "strength": 0.3},
    )


def test_match_filename_exact_and_variant():
    from app.registry import match_filename

    files = {"krea_2_turbo_q8p.ckpt", "flux_1_schnell_q8p.ckpt"}
    assert match_filename("krea_2_turbo_q8p.ckpt", files) == "krea_2_turbo_q8p.ckpt"
    # same model, different quantisation picked at download time
    assert match_filename("krea_2_turbo_i8x.ckpt", files) == "krea_2_turbo_q8p.ckpt"
    assert match_filename("krea_2_turbo_f16.ckpt", files) == "krea_2_turbo_q8p.ckpt"
    # a different version is a different model, never matched
    assert match_filename("model_v11_f16.ckpt", {"model_v10_f16.ckpt"}) is None


def test_match_filename_prefers_higher_precision():
    from app.registry import match_filename

    files = {"m_q6p.ckpt", "m_f16.ckpt", "m_q8p.ckpt"}
    assert match_filename("m_i8x.ckpt", files) == "m_f16.ckpt"


def test_resolve_installed_marks_and_rewrites(tmp_path: Path):
    from app.registry import DrawThingsConfig, LlmConfig, Registry

    (tmp_path / "present_q8p.ckpt").touch()
    (tmp_path / "lora_a_f16.ckpt").touch()
    reg = Registry(
        llm=LlmConfig(),
        draw_things=DrawThingsConfig(models_dirs=[str(tmp_path)]),
        models={
            "ok": _spec("present_i8x.ckpt", loras=[{"file": "lora_a_q8p.ckpt", "weight": 0.6}]),
            "gone": _spec("absent_f16.ckpt"),
        },
    )
    reg.resolve_installed()
    ok = reg.models["ok"]
    assert ok.installed and ok.available
    assert ok.dt_filename == "present_q8p.ckpt"
    assert ok.draft.loras[0].file == "lora_a_f16.ckpt"
    assert reg.models["gone"].installed is False
    with pytest.raises(ValueError, match="not installed"):
        reg.model_or_raise("gone")


def test_resolve_installed_missing_lora_hides_model(tmp_path: Path):
    from app.registry import DrawThingsConfig, LlmConfig, Registry

    (tmp_path / "present_f16.ckpt").touch()
    reg = Registry(
        llm=LlmConfig(),
        draw_things=DrawThingsConfig(models_dirs=[str(tmp_path)]),
        models={"m": _spec("present_f16.ckpt", loras=[{"file": "missing_lora_f16.ckpt"}])},
    )
    reg.resolve_installed()
    assert reg.models["m"].installed is False


def test_resolve_installed_no_dirs_is_noop(tmp_path: Path):
    from app.registry import DrawThingsConfig, LlmConfig, Registry

    reg = Registry(
        llm=LlmConfig(),
        draw_things=DrawThingsConfig(models_dirs=[str(tmp_path / "nope")]),
        models={"m": _spec("whatever_f16.ckpt")},
    )
    reg.resolve_installed()
    assert reg.models["m"].installed is None
    assert reg.models["m"].available


def _vspec(dt_filename):
    from app.schemas import VideoModelSpec

    return VideoModelSpec(key="v", label="V", dt_filename=dt_filename)


def test_loads_video_models_from_default_config():
    reg = load_registry(CONFIG_PATH)
    assert "fusionx" in reg.video_models
    fx = reg.video_models["fusionx"]
    # FusionX is distilled: few steps at CFG 1
    assert fx.steps == 8
    assert fx.cfg == 1.0
    # the only video model that ships; AniSora was measured and dropped
    # (docs/decisions/VIDEO.md)
    assert list(reg.video_models) == ["fusionx"]


def test_video_estimate_grows_faster_than_frame_count():
    reg = load_registry(CONFIG_PATH)
    fx = reg.video_models["fusionx"]
    per_frame_33 = fx.estimate_seconds(33) / 33
    per_frame_81 = fx.estimate_seconds(81) / 81
    # measured 17.1 s/frame at 33 and 22.8 at 81: cost is superlinear
    assert per_frame_81 > per_frame_33
    # and the calibrated point should land near the 1844s that was measured
    assert 1500 < fx.estimate_seconds(81) < 2200


def test_unknown_video_model_raises():
    reg = load_registry(CONFIG_PATH)
    with pytest.raises(KeyError):
        reg.video_model_or_raise("nope")


def test_resolve_installed_rewrites_video_variant(tmp_path: Path):
    from app.registry import DrawThingsConfig, LlmConfig, Registry

    (tmp_path / "vid_q8p.ckpt").touch()
    reg = Registry(
        llm=LlmConfig(),
        draw_things=DrawThingsConfig(models_dirs=[str(tmp_path)]),
        models={},
        video_models={"v": _vspec("vid_f16.ckpt")},
    )
    reg.resolve_installed()
    assert reg.video_models["v"].installed
    assert reg.video_models["v"].dt_filename == "vid_q8p.ckpt"


def test_match_filename_packaging_variants():
    from app.registry import match_filename

    # official catalogue name vs a custom import of the same LoRA
    assert (
        match_filename(
            "dmd2_sdxl_4_step_lora_f16.ckpt",
            {"dmd2_sdxl_4step_lora_fp16_lora_f16.ckpt"},
        )
        == "dmd2_sdxl_4step_lora_fp16_lora_f16.ckpt"
    )


def test_quality_opener_follows_prompt_style():
    reg = load_registry(CONFIG_PATH)
    for key, spec in reg.models.items():
        if spec.prompt_style == "tags":
            assert spec.quality_opener, f"{key}: a tag model needs an opener"
        else:
            assert spec.quality_opener == "", f"{key}: prose models take no tag opener"
    # a tag model that names no opener of its own falls back to the default.
    # Every tag model shipped today dictates its own opener, so the fallback
    # has no subject in the config and is exercised directly instead.
    assert _spec("m_f16.ckpt", style="tags").quality_opener == DEFAULT_TAG_PREFIX
    assert _spec("m_f16.ckpt", style="prose").quality_opener == ""


def test_quality_opener_matches_each_model_s_own_guidance():
    """Where a model's guidance dictates its opener, the config must agree.

    The openers are not interchangeable between tag families, and the guidance
    text is where the requirement is recorded, so this keeps the two from
    drifting apart as models are added.
    """
    reg = load_registry(CONFIG_PATH)
    checked = 0
    for key, spec in reg.models.items():
        m = re.search(
            r'Start every prompt with exactly\s+"([^"]+)"',
            " ".join(spec.prompt_guidance.split()),
        )
        if not m:
            continue
        checked += 1
        # the shipped one: a ticked LoRA may add its own tags at queue time
        assert spec.shipped_opener == m.group(1), (
            f"{key}: guidance demands {m.group(1)!r} but config opens with "
            f"{spec.shipped_opener!r}"
        )
    assert checked >= 3, "expected several models to dictate their own opener"


def test_an_empty_scan_does_not_disable_upscale(tmp_path, monkeypatch):
    """A scan that sees no files must leave the previous answers alone.

    resolve_installed() used to null self.upscaler before checking whether the
    scan had found anything, so a single scan of a momentarily unreadable
    Draw Things container (it restarts, /api/models rescans on every page load)
    reported "no upscaler installed" for every model and rendered Upscale
    disabled on every card until the next refresh.
    """
    reg = load_registry()
    spec = reg.models["lustrousmix-anima"]
    good = {"4x_ultrasharp_f16.ckpt", spec.dt_filename} | {
        lora.file for stage in (spec.draft, spec.final, spec.upscale)
        for lora in stage.loras
    }
    monkeypatch.setattr("app.registry.installed_files", lambda dirs: good)
    reg.resolve_installed()
    assert reg.upscaler == "4x_ultrasharp_f16.ckpt"
    assert reg.upscale_block("lustrousmix-anima") is None

    # the container goes unreadable: the good answer must survive
    monkeypatch.setattr("app.registry.installed_files", lambda dirs: set())
    reg.resolve_installed()
    assert reg.upscaler == "4x_ultrasharp_f16.ckpt"
    assert reg.upscale_block("lustrousmix-anima") is None


def test_every_model_gets_its_own_opener_at_queue_time():
    """The mirror invariant to the negative-prompt floor below.

    Several checkpoints treat their opener as required, and the enrichment
    path only honours one when every model in the call agrees on it -- so two
    tag models with different openers, enriched together because they share a
    dialect, were sent whatever the LLM wrote freehand. Observed: "very
    aesthetic, absurdres 1girl full body", which is neither WAI's opener
    punctuated nor BAS+'s required trigger tags.
    """
    from app.main import prompt_for

    reg = load_registry()
    tag_models = [m for m in reg.models.values() if m.quality_opener]
    assert len(tag_models) > 1, "expected several models with openers"
    for spec in tag_models:
        for written in (
            "1girl, smile",                                   # no opener at all
            "masterpiece, best quality, 1girl, smile",        # another model's
            "masterpiece, best quality, amazing quality, very aesthetic, "
            "absurdres 1girl full body, smile",               # the reported bug
        ):
            out = prompt_for(spec, written)
            assert out.lower().startswith(spec.quality_opener.lower()), spec.key
            # the opener is punctuated off from what follows it, which is the
            # half of the bug that survived into the prompt as unreadable text
            rest = out[len(spec.quality_opener):]
            assert rest.startswith(", "), (spec.key, out)
            # and no second opener is left behind further in
            assert "masterpiece" not in rest.lower(), (spec.key, out)
            assert prompt_for(spec, out) == out, f"{spec.key} is not idempotent"
    # a prose model has no opener and its prompt must come through untouched
    prose = reg.models["z-image-turbo"]
    assert not prose.quality_opener
    assert prompt_for(prose, "A woman kneels on grass.") == "A woman kneels on grass."


def test_tag_models_have_a_negative_prompt_floor():
    """A model that uses a negative prompt must never be handed an empty one.

    The paths that skip the LLM had nothing to put in the field, so a third of
    the tag-model generations on record went out with no negative at all (found
    on Illustrious XL, which no longer ships; the invariant is not model
    specific and is checked here against every tag model that wants one).
    """
    from app.main import negative_for

    reg = load_registry()
    checked = 0
    for key, spec in reg.models.items():
        if spec.prompt_style != "tags" or not spec.uses_negative_prompt:
            continue
        checked += 1
        assert negative_for(spec, "") == spec.default_negative != "", key
        assert negative_for(spec, "   ") == spec.default_negative, key
        # whatever was actually written still wins
        assert negative_for(spec, "extra digits") == "extra digits", key
    assert checked, "expected at least one model to use a negative prompt"
    # a model documenting its own negative gets that one, not the shared floor
    # -- when it runs at its own CFG; its turbo LoRA takes the negative away
    lustrous = reg.models["lustrousmix-anima"]
    assert lustrous.default_negative == ""
    reg.apply_lora_choices({"lustrousmix-anima": {"turbo": False}})
    assert "score_1" in lustrous.default_negative
    assert lustrous.default_negative != DEFAULT_TAG_NEGATIVE
    # models that take no negative prompt are left alone
    assert negative_for(reg.models["bas-anima"], "") == ""
    assert negative_for(reg.models["flux2-klein"], "anything") == ""


def _yaml_blocks() -> dict[str, str]:
    """config/models.yaml split into the comment+body text of each model key.

    A comment block sitting above a key documents THAT key, not the one whose
    body it happens to follow, so pending comment lines are carried forward
    onto the next key rather than backwards onto the previous one. Getting this
    the other way round attributed a banner comment to `oneobsession` and made
    the marker check below fail on the wrong model.
    """
    text = CONFIG_PATH.read_text()
    body = text.split("\nmodels:\n", 1)[1].split("\nvideo_models:", 1)[0]
    blocks: dict[str, str] = {}
    key, pending = None, ""
    for line in body.splitlines():
        m = re.match(r"^  ([a-z0-9][a-z0-9-]*):\s*$", line)
        if m:
            key = m.group(1)
            blocks[key] = pending
            pending = ""
        elif re.match(r"^\s*(#.*)?$", line) and key is None:
            pending += line + "\n"
        elif re.match(r"^  #", line):
            # a top-level comment: belongs to whatever key comes next
            pending += line + "\n"
        elif key:
            blocks[key] += line + "\n"
    return blocks


def test_an_unconfirmed_entry_is_never_a_like_source():
    """The invariant that keeps `--level confirm` honest.

    `scripts/add_model.py --like` copies a sibling's geometry, and it may only
    ever copy from a checkpoint whose own sampler and cfg were established by
    rendering it. The incident that rule comes from is one entry that honestly
    labelled itself unmeasured being used as the source for three others
    (docs/decisions/MODEL-MEASUREMENTS.md) -- a value with a caveat next to it
    still reads as a value, so the caveat alone did not stop it.

    So this asserts the two lists cannot drift apart: an entry whose sampler
    is still marked UNCONFIRMED in the YAML must not appear in
    MEASURED_SOURCES, and the reverse -- promoting a key into that set without
    deleting the marker from its comment -- fails here rather than quietly
    seeding the next model with a guess.
    """
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from scripts.add_model import MEASURED_SOURCES

    reg = load_registry(CONFIG_PATH)
    unconfirmed = {k for k, text in _yaml_blocks().items() if "UNCONFIRMED" in text}
    # deliberately NOT asserting that anything is unconfirmed: an empty set is
    # the good end state, and asserting otherwise would make this test fail the
    # moment the last marker is earned away
    assert not (unconfirmed & MEASURED_SOURCES), (
        f"{sorted(unconfirmed & MEASURED_SOURCES)} are marked UNCONFIRMED in "
        f"config/models.yaml and are also listed as measured sources for "
        f"--like. One of the two is wrong."
    )
    # and every source named is a real key, so a rename cannot silently empty it
    assert MEASURED_SOURCES <= set(reg.models), (
        f"MEASURED_SOURCES names keys that are not in the registry: "
        f"{sorted(MEASURED_SOURCES - set(reg.models))}"
    )


def test_the_inherited_illustrious_entry_matches_the_architecture_it_copied():
    """A cheaply-added entry must actually carry the geometry it claims to.

    `--like` is only defensible while the inherited fields really are the
    sibling's, so the fields it copies are pinned against the measured source
    here. Not the sampler, cfg or draft: those are deliberately NOT inherited,
    and janku's draft steps sit between its siblings' rather than on any one
    of them.
    """
    reg = load_registry(CONFIG_PATH)
    source = reg.models["cyberrealistic-illustrious"]
    # rinflanime is left out: its final size is its own (MODEL-MEASUREMENTS.md)
    for key in ("janku",):
        spec = reg.models[key]
        assert spec.prompt_style == source.prompt_style == "tags", key
        assert spec.clip_skip == source.clip_skip == 2, key
        assert spec.prompt_budget == source.prompt_budget, key
        assert spec.final.size == source.final.size == 1536, key
        assert spec.upscale.size == source.upscale.size == 2048, key
        assert spec.draft.size == source.draft.size == 768, key
        # shapes at the SDXL buckets like the source, and 2048 unaided
        assert spec.aspect_size == source.aspect_size == 1024, key
        assert spec.tiled_diffusion is False, f"{key}: SDXL reaches 2048 unaided"
        assert spec.max_upscale_size is None, key
        # a tag model still needs its opener and its negative floor
        assert spec.quality_opener and spec.uses_negative_prompt, key


def test_rinflanime_keeps_the_smaller_final_it_was_measured_at():
    """A deliberate divergence from the family, pinned so it is not tidied away.

    Every other Illustrious/SDXL entry renders its final at 1536, so raising
    this one to match looks like obvious cleanup. It is not: at 1536 with 24
    steps this checkpoint puts a second figure behind the subject on 3 of 3
    seeds, where 1536 with 20 steps does so on none and 1024 is clean at both
    (docs/decisions/MODEL-MEASUREMENTS.md; frames in research/confirm/).

    1536/20 is measured clean and still rejected, because final.steps is
    overridable per install from the gear in the header -- so shipping a size
    that fails four steps away hands out a configuration a user can break
    without being told why. If this assert ever needs changing, the thing to
    change first is the measurement.
    """
    reg = load_registry(CONFIG_PATH)
    spec = reg.models["rinflanime"]
    assert spec.final.size == 1024, (
        "rinflanime's final size is measured, not inherited: 1536 duplicates "
        "the subject at 24 steps on every seed"
    )
    # the step count it was measured clean at, and the size it must not creep to
    assert spec.final.steps <= 24
    assert spec.draft.size == 768
    # and the cap is about area, so no shape may spend more of it than square
    assert spec.aspect_size is not None and spec.aspect_size <= spec.final.size, (
        "a shape budget above the final's would render shapes at an area this "
        "checkpoint was measured to duplicate at"
    )


def test_qwen_image_shapes_stay_under_its_step_ceiling():
    """Every shape's long edge stays on a row that rendered at the final's steps.

    Qwen Image returns an empty image past roughly long edge x steps ~ 14000
    (config/models.yaml); 1152x12 rendered 3/3 and 1344x12 1/2. So its shape
    budget sits below the square final, at the largest value whose widest
    ratio still lands a 1152 long edge -- raised to the final's own 1024 it
    puts 16:9 at 1344x768 and makes the wide shapes a coin flip.
    """
    from app.dims import ASPECTS, ratio, stage_dims

    spec = load_registry(CONFIG_PATH).models["qwen-image"]
    for aspect in ASPECTS:
        w, h = stage_dims(spec.final.size, ratio(aspect), spec.aspect_size)
        if aspect != "1:1":
            assert max(w, h) <= 1152, (aspect, w, h)


def test_every_model_credits_a_checkpoint_when_published():
    """A blank civitai_version_id publishes with no resource at all.

    Reported twice from use ("We weren't able to detect any resources used in
    the creation of this image"), both times a model left blank because it came
    from Draw Things' catalogue -- and all of those turned out to have official
    Civitai pages. scripts/check_civitai_ids.py says what each id resolves to;
    nothing here can check that, so this only holds the line that a model ships
    with one.
    """
    reg = load_registry(CONFIG_PATH)
    assert [k for k, m in reg.models.items() if not m.civitai_version_id] == []


def test_no_two_models_claim_the_same_civitai_version():
    """A copied entry would credit one checkpoint for another's pictures."""
    reg = load_registry(CONFIG_PATH)
    seen: dict[int, str] = {}
    for key, spec in reg.models.items():
        if spec.civitai_version_id:
            assert spec.civitai_version_id not in seen, (
                f"{key} and {seen[spec.civitai_version_id]} share a version id")
            seen[spec.civitai_version_id] = key


def test_every_model_that_composes_has_a_shape_budget():
    """No chip renders square whatever shape is picked.

    A missing aspect_size used to mean "never measured"; now a new model ships
    with a best guess and the sweep (scripts/sweep_aspect.py) confirms it.
    Edit-only models are exempt: they take the parent picture's size.
    """
    reg = load_registry(CONFIG_PATH)
    assert [k for k, m in reg.models.items()
            if not m.edit_only and m.aspect_size is None] == []


def test_the_widest_shape_renders_at_a_size_that_has_been_probed():
    """A shape budget is backed by a frame at the longest edge it produces.

    z-image-turbo shipped with aspect_size raised from 1024 to 1536 alongside
    its final, on the argument that SHAPES.md had found no resolution limit --
    its shapes had only ever been rendered at the old budget. The first 16:9
    final at the new one (2048x1152) drew every subject twice. That finding was
    about squares, and 4:3 and 3:2 at 1536 do hold; the two 2048-long-edge
    shapes do not (config/models.yaml, research/aspect/z-image-turbo).

    Every duplication on record is at the longest edge a model renders, so
    that is the probe: for each model that composes, 16:9 or 9:16 at the size
    `stage_dims` gives it must appear as a rendered row in
    research/aspect/<key>/results.json. One cell at low steps is enough
    (scripts/sweep_aspect.py --shapes 1:1,16:9 --seeds 1111 --steps 2).
    Change the final or the budget and the size moves, which fails here until
    the new size has been rendered and archived. Whether the frame is clean is
    still read by eye; this only proves somebody had the chance to look.
    """
    import json

    from app import ROOT
    from app.dims import ratio, stage_dims

    if not (ROOT / "research" / "aspect").is_dir():
        pytest.skip("research/ is not published with the public copy")
    reg = load_registry(CONFIG_PATH)
    unprobed: dict[str, str] = {}
    for key, m in reg.models.items():
        if m.edit_only or m.aspect_size is None:
            continue
        if key == "flux1-schnell":
            # not renderable on the measuring install: Draw Things answers
            # 422 "Missing files for model" for its checkpoint. Supported,
            # not recommended (docs/SETUP.md), and its 1344x768 is the size
            # klein, the same stack a generation on, probed clean at.
            continue
        log = ROOT / "research" / "aspect" / key / "results.json"
        rows = ({(r["width"], r["height"]) for r in json.loads(log.read_text())
                 if not r.get("empty")} if log.is_file() else set())
        wide = stage_dims(m.final.size, ratio("16:9"), m.aspect_size)
        tall = stage_dims(m.final.size, ratio("9:16"), m.aspect_size)
        if wide not in rows and tall not in rows:
            unprobed[key] = "%dx%d" % wide
    assert not unprobed, (
        "widest shape never rendered at the size it now produces -- run "
        "scripts/sweep_aspect.py --models <key> --shapes 1:1,16:9 --seeds 1111 "
        "--steps <few> --note '...' -- " + json.dumps(unprobed))


def test_every_draft_has_a_long_side_of_at_least_768_at_every_ratio():
    """A draft is what a Faithful enlarges, and its pixels are what it can carry.

    The rule is on the long side: 768 there, whatever the shape. A shaped
    draft's long side is never below its budget (dims.stage_dims spends the
    area on the long side as the ratio widens), so this is the same as no
    draft budget under 768 -- which four entries were: krea2-turbo at 512
    drafted 16:9 at 704x384, lustrousmix-anima and cyberrealistic-anima at
    640, flux1-schnell at 256. Their comments in config/models.yaml keep the
    measurements the smaller sizes were accepted on.
    """
    from app.dims import ASPECTS, ratio, stage_dims

    reg = load_registry(CONFIG_PATH)
    short = {
        (k, a): (w, h)
        for k, m in reg.models.items() if not m.edit_only
        for a in ASPECTS
        for w, h in [stage_dims(m.draft.size, ratio(a), m.aspect_size)]
        if max(w, h) < 768
    }
    assert short == {}
