"""Tune: the rounds, the rules that turn picks into settings, and the entry written."""

from __future__ import annotations

import asyncio
import io
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import tune  # noqa: E402
from app.private_models import added_keys, append_entry, entry_block, load_profiles  # noqa: E402
from app.registry import CONFIG_PATH, load_registry  # noqa: E402


def _spec(sampler="DPM++ 2M Karras", cfg=5.0, final=(1536, 20), draft=768, aspect=1024,
          clip_skip=1):
    return SimpleNamespace(
        label="Some Mix", sampler=sampler, cfg=cfg, aspect_size=aspect, clip_skip=clip_skip,
        supports_aspect=aspect is not None,
        final=SimpleNamespace(size=final[0], steps=final[1]),
        draft=SimpleNamespace(size=draft, steps=14, sampler=None, cfg=None))


def _png(noisy=True, grey=None) -> bytes:
    img = Image.effect_noise((64, 64), 64).convert("RGB") if noisy else \
        Image.new("RGB", (64, 64), grey or (20, 140, 30))
    out = io.BytesIO()
    img.save(out, "PNG")
    return out.getvalue()


@pytest.fixture
def tuner(tmp_path):
    return tune.Tuner(tmp_path / "tune", "some-mix")


def _render_all(t, png=None, fail=()):
    async def render(cell):
        if cell["id"] in fail:
            raise RuntimeError("Draw Things said no")
        return png() if callable(png) else (png or _png())
    asyncio.run(t.render_pending(render))


def _cell_of(state, q, **want):
    """The option index of the cell with these params, however it was shuffled."""
    return next(i for i, c in enumerate(q["cells"])
                if all(state["cells"][c][k] == v for k, v in want.items()))


# -- the plan ---------------------------------------------------------------------


def test_the_sampler_round_is_two_samplers_by_two_cfgs_on_two_seeds_at_draft_size():
    cells, questions = tune.plan_round("sampler", tune.settings_of(_spec()))
    assert len(cells) == 8 and len(questions) == 2
    assert {(c["sampler"], c["cfg"]) for c in cells} == {
        ("DPM++ 2M Karras", 5.0), ("DPM++ 2M Karras", 3.5), ("Euler a", 5.0), ("Euler a", 3.5)}
    assert {(c["width"], c["steps"]) for c in cells} == {(768, 20)}
    # blind: the model's own setting is not always the first picture
    firsts = {cells_by_id(cells)[q["cells"][0]]["label"] for q in questions}
    assert questions[0]["cells"] != sorted(questions[0]["cells"]) or len(firsts) > 1


def cells_by_id(cells):
    return {c["id"]: c for c in cells}


def test_an_euler_model_is_tried_against_dpm():
    cells, _ = tune.plan_round("sampler", tune.settings_of(_spec(sampler="Euler a")))
    assert {c["sampler"] for c in cells} == {"Euler a", "DPM++ 2M Karras"}


def test_the_size_round_counts_people_at_the_final_size_and_the_widest_shape():
    cells, questions = tune.plan_round("size", tune.settings_of(_spec()))
    assert [(c["width"], c["height"]) for c in cells][:2] == [(1536, 1536)] * 2
    wide = cells[2]
    assert wide["width"] > wide["height"] and wide["width"] * wide["height"] <= 1024 * 1024 * 1.05
    assert {q["kind"] for q in questions} == {"count"}
    # a model that renders square only has no shape to check
    _, qs = tune.plan_round("size", tune.settings_of(_spec(aspect=None)))
    assert [q["part"] for q in qs] == ["square", "square"]


# -- the rules --------------------------------------------------------------------


def _answered(name, s, answers, attempt=1):
    cells, questions = tune.plan_round(name, s, attempt)
    by = cells_by_id(cells)
    for q, a in zip(questions, answers):
        q["answer"] = a(by, q) if callable(a) else a
    return tune.decide(name, s, questions, by, attempt)


def _pick(**want):
    return lambda by, q: next(i for i, c in enumerate(q["cells"])
                              if all(by[c][k] == v for k, v in want.items()))


def test_a_sampler_picked_on_both_seeds_wins():
    s, notes, again = _answered("sampler", tune.settings_of(_spec()),
                                [_pick(sampler="Euler a", cfg=3.5)] * 2)
    assert (s.sampler, s.cfg, again) == ("Euler a", 3.5, False)
    assert "picked on both seeds" in notes[0]


def test_anything_short_of_both_seeds_keeps_the_model_as_it_is():
    before = tune.settings_of(_spec())
    for answers in ([_pick(sampler="Euler a", cfg=5.0), _pick(sampler="Euler a", cfg=3.5)],
                    [_pick(sampler="Euler a", cfg=5.0), "cant_tell"],
                    ["cant_tell", "cant_tell"]):
        s, notes, _ = _answered("sampler", before, answers)
        assert (s.sampler, s.cfg) == (before.sampler, before.cfg), answers
        assert "kept" in notes[0]


def test_two_figures_step_the_final_down_and_ask_again():
    s, notes, again = _answered("size", tune.settings_of(_spec()), ["more", "one", "one"])
    assert (s.final_size, again) == (1280, True)
    assert "trying 1280px" in notes[0]
    # one figure everywhere: settled
    s, _, again = _answered("size", s, ["one", "one", "one"], attempt=2)
    assert (s.final_size, again) == (1280, False)


def test_a_doubled_wide_shape_steps_the_shape_budget_down_a_grid_notch():
    s, _, again = _answered("size", tune.settings_of(_spec()), ["one", "one", "more"])
    assert (s.final_size, s.aspect_size, again) == (1536, 960, True)


def test_the_size_round_stops_at_the_draft_size_and_after_three_tries():
    s, notes, again = _answered("size", tune.settings_of(_spec(final=(1024, 20), draft=768)),
                                ["more", "one", "one"], attempt=3)
    assert (s.final_size, again) == (1024, False)
    assert "worth a look by hand" in notes[0]


def test_fewer_steps_win_unless_the_more_are_picked():
    s0 = tune.settings_of(_spec())
    s, _, _ = _answered("steps", s0, ["same", "same"])
    assert s.final_steps == 13
    s, _, _ = _answered("steps", s0, [_pick(steps=13), "same"])
    assert s.final_steps == 13
    s, notes, _ = _answered("steps", s0, [_pick(steps=20), _pick(steps=13)])
    assert s.final_steps == 20 and "kept" in notes[0]


# -- the flow ---------------------------------------------------------------------


def test_a_whole_tuning_from_start_to_review(tuner):
    state = tuner.start(_spec(), prompt="p", negative="n", today="2026-11-12")
    assert tuner.phase(state) == "rendering" and state["round"] == "sampler"
    assert tune.estimate(tuner.pending(state), [
        {"model_key": "some-mix", "kind": "draft", "size": 768, "sec_per_step": 1.0, "n": 5}],
        "some-mix") == 8 * 20
    _render_all(tuner)
    state = tuner.load()
    assert tuner.phase(state) == "asking"
    for q in state["questions"]:
        state = tuner.answer(q["id"], _cell_of(state, q, sampler="Euler a", cfg=5.0))
    assert state["round"] == "size" and state["settings"]["sampler"] == "Euler a"
    _render_all(tuner)
    for q in tuner.load()["questions"]:
        state = tuner.answer(q["id"], "one")
    assert state["round"] == "steps"
    state = tuner.skip_round()
    assert tuner.phase(state) == "review"
    assert tune.changes(state) == {"sampler": ("DPM++ 2M Karras", "Euler a")}


def test_a_distilled_model_skips_the_sampler_round(tuner):
    state = tuner.start(_spec(cfg=1.0), prompt="p", negative="n")
    assert state["round"] == "size" and "sampler" in state["skipped"]


def test_a_blank_or_failed_render_cannot_be_picked(tuner):
    state = tuner.start(_spec(), prompt="p", negative="n")
    q = state["questions"][0]
    bad = q["cells"][0]
    _render_all(tuner, fail={bad})
    state = tuner.load()
    assert state["cells"][bad]["status"] == "error"
    assert "Draw Things said no" in state["cells"][bad]["error"]
    assert tuner.phase(state) == "asking"  # the rest of the question can still be judged
    with pytest.raises(ValueError, match="failed"):
        tuner.answer(q["id"], 0)
    state = tuner.retry_failed()
    assert tuner.phase(state) == "rendering"


def test_flat_and_refused_renders_are_told_apart_from_pictures():
    assert tune.flat_or_blocked(_png()) is None
    assert "flat colour" in tune.flat_or_blocked(_png(noisy=False))
    assert "refused" in tune.flat_or_blocked(_png(noisy=False, grey=(128, 128, 128)))


def test_a_round_with_nothing_to_judge_says_so(tuner):
    state = tuner.start(_spec(), prompt="p", negative="n")
    _render_all(tuner, png=lambda: _png(noisy=False))
    assert tuner.phase(tuner.load()) == "failed"


def test_answers_are_checked(tuner):
    state = tuner.start(_spec(), prompt="p", negative="n")
    with pytest.raises(ValueError, match="not ready"):
        tuner.answer(state["questions"][0]["id"], 0)
    _render_all(tuner)
    with pytest.raises(ValueError, match="pick one"):
        tuner.answer(state["questions"][0]["id"], 9)
    with pytest.raises(ValueError, match="only the steps round"):
        tuner.skip_round()


def test_discarding_mid_render_drops_the_late_frame(tuner):
    tuner.start(_spec(), prompt="p", negative="n")

    async def render(cell):
        tuner.discard()
        return _png()
    asyncio.run(tuner.render_pending(render))
    assert tuner.load() is None and not tuner.dir.exists()


# -- what is written ------------------------------------------------------------


@pytest.fixture
def config(tmp_path):
    shutil.copy(CONFIG_PATH, tmp_path / "models.yaml")
    return tmp_path / "models.yaml"


def test_the_picks_go_into_the_entry_and_say_they_are_picks(config, tuner):
    profile = load_profiles()["illustrious"]["settings"]
    entry = {"label": "Some Mix", "dt_filename": "somemix_f16.ckpt", **profile}
    private = config.with_name("models.private.yaml")
    append_entry(private, "some-mix", entry_block(
        "some-mix", entry, ["Added 2026-11-11 on startup: installed in Draw Things, set up "
                            "from the illustrious profile"]))
    spec = load_registry(config).models["some-mix"]
    state = tuner.start(spec, prompt="p", negative="n")
    state["settings"].update(sampler="Euler a", cfg=4.0, final_size=1280, final_steps=14)
    state["notes"] = ["Sampler and cfg: Euler a, cfg 4, picked on both seeds over "
                      "DPM++ 2M Karras, cfg 5."]
    said = tune.write_tuning("some-mix", state, config_path=config, today="2026-11-12")
    assert "sampler DPM++ 2M Karras → Euler a" in said and "final size 1536 → 1280" in said
    got = load_registry(config).models["some-mix"]
    assert (got.sampler, got.cfg, got.final.size, got.final.steps) == ("Euler a", 4.0, 1280, 14)
    # the draft repeated the model's own sampler and cfg, so it follows them
    assert (got.draft.sampler, got.draft.cfg) == ("Euler a", 4.0)
    text = private.read_text()
    assert "Tuned 2026-11-12 in the settings dialog" in text and "Not a measurement" in text
    assert "some-mix" in added_keys(private)
    assert text.count("  some-mix:") == 1


def test_a_draft_sampler_of_its_own_is_left_alone():
    entry = {"sampler": "Euler a", "cfg": 4.5,
             "draft": {"size": 768, "steps": 12, "cfg": 4.5, "sampler": "UniPC Trailing"},
             "final": {"size": 1536, "steps": 24}}
    state = {"before": {"sampler": "Euler a", "cfg": 4.5, "final_size": 1536,
                        "final_steps": 24, "aspect_size": 1472, "draft_size": 768},
             "settings": {"sampler": "DPM++ 2M Karras", "cfg": 3.0, "final_size": 1536,
                          "final_steps": 24, "aspect_size": 1472, "draft_size": 768}}
    new = tune.tuned_entry(entry, state)
    assert new["draft"] == {"size": 768, "steps": 12, "cfg": 3.0, "sampler": "UniPC Trailing"}
    assert entry["sampler"] == "Euler a"  # the original is not touched


def test_only_a_model_you_added_can_be_tuned(config, tuner):
    state = {"before": {}, "settings": {}, "notes": [], "label": "x"}
    with pytest.raises(ValueError, match="only a model you added"):
        tune.write_tuning("juggernaut-ragnarok", state, config_path=config)
    assert not config.with_name("models.private.yaml").exists()
    assert yaml.safe_load(config.read_text())  # untouched


# -- the class round ----------------------------------------------------------------


def _classes():
    """Three SDXL classes as main builds them: each with its own prompt."""
    return {
        "sdxl": tune.class_option(_spec(final=(1024, 32)), prompt="SCENE, 85mm",
                                  negative="", entry={"prompt_style": "prose", "sampler": "DPM++ 2M Karras",
                                                      "cfg": 5.0, "final": {"size": 1024, "steps": 32}}),
        "illustrious": tune.class_option(_spec(clip_skip=2), prompt="masterpiece, SCENE",
                                         negative="bad quality",
                                         entry={"prompt_style": "tags", "clip_skip": 2,
                                                "sampler": "DPM++ 2M Karras", "cfg": 5.0,
                                                "draft": {"size": 768, "steps": 14, "cfg": 5.0,
                                                          "sampler": "DPM++ 2M Karras"},
                                                "final": {"size": 1536, "steps": 20}}),
        "animagine": tune.class_option(_spec(cfg=7.0, final=(1024, 28)),
                                       prompt="very aesthetic, SCENE", negative="lowres",
                                       entry={"prompt_style": "tags", "sampler": "DPM++ 2M Karras",
                                              "cfg": 7.0, "final": {"size": 1024, "steps": 28}}),
    }


def _start_classed(tuner):
    return tuner.start(_spec(final=(1024, 32)), prompt="SCENE, 85mm", negative="",
                       classes=_classes(), current_class="sdxl")


def test_a_model_on_a_class_of_a_shared_architecture_is_asked_its_kind_first(tuner):
    state = _start_classed(tuner)
    assert state["round"] == "class" and state["rounds"][0] == "class"
    cells = [state["cells"][c] for c in state["questions"][0]["cells"]]
    # each class is rendered with its own prompt, negative and clip skip
    assert {(c["class"], c["prompt"], c["negative"], c["clip_skip"]) for c in cells} == {
        ("sdxl", "SCENE, 85mm", "", 1), ("illustrious", "masterpiece, SCENE", "bad quality", 2),
        ("animagine", "very aesthetic, SCENE", "lowres", 1)}
    sent = []

    async def render(cell):
        sent.append(cell)
        return _png()
    asyncio.run(tuner.render_pending(render))
    assert {(s["prompt"], s["clip_skip"]) for s in sent if s["class"] == "illustrious"} == {
        ("masterpiece, SCENE", 2)}


def test_a_class_picked_on_both_seeds_moves_the_whole_model(tuner):
    _start_classed(tuner)
    _render_all(tuner)
    state = tuner.load()
    for q in state["questions"]:
        state = tuner.answer(q["id"], _cell_of(state, q, **{"class": "animagine"}))
    assert state["round"] == "sampler" and state["class"] == "animagine"
    # the rest is judged on the new class's prompt, from its settings
    assert (state["prompt"], state["settings"]["cfg"]) == ("very aesthetic, SCENE", 7.0)
    assert {c["cfg"] for c in tuner.pending(state)} == {7.0, 4.5}
    assert tune.changes(state)["class"] == ("sdxl", "animagine")
    entry = tune.tuned_entry({"label": "Some Mix", "dt_filename": "x.ckpt", "prompt_style": "prose",
                              "fingerprints": ["ab"]}, state)
    assert (entry["prompt_style"], entry["cfg"], entry["label"], entry["fingerprints"]) == (
        "tags", 7.0, "Some Mix", ["ab"])


def test_short_of_both_seeds_the_class_stays(tuner):
    _start_classed(tuner)
    _render_all(tuner)
    state = tuner.load()
    q1, q2 = state["questions"]
    tuner.answer(q1["id"], _cell_of(state, q1, **{"class": "animagine"}))
    state = tuner.answer(q2["id"], "cant_tell")
    assert state["class"] == "sdxl" and state["prompt"] == "SCENE, 85mm"
    assert "sdxl kept" in state["notes"][-1]


def test_no_class_round_without_a_choice(tuner):
    state = tuner.start(_spec(), prompt="p", negative="n")
    assert state["round"] == "sampler" and "class" not in state["rounds"]
    one = {"sdxl": _classes()["sdxl"]}
    state = tune.Tuner(tuner.dir.parent, "other").start(_spec(), prompt="p", negative="n",
                                                        classes=one, current_class="sdxl")
    assert state["round"] == "sampler"
