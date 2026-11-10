"""probe.py: its model report says what the app will find on disk and never
sends the user off to edit filenames the app matches by itself, and its test
render names an image model rather than inheriting the one Draw Things has
selected (after an Animate, the video model)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.probe import model_report, pick_test_model as pick_model  # noqa: E402

from app.registry import DrawThingsConfig, LlmConfig, Registry  # noqa: E402
from app.schemas import ModelSpec  # noqa: E402


def _spec(key, dt_filename, steps=4):
    return ModelSpec(key=key, label=key.title(), dt_filename=dt_filename,
                     draft={"size": 256, "steps": steps}, final={"size": 512, "steps": 8},
                     upscale={"size": 512, "steps": 8})


def _registry(dirs):
    return Registry(
        llm=LlmConfig(),
        draw_things=DrawThingsConfig(models_dirs=dirs, upscalers=["4x_ultrasharp_f16.ckpt"]),
        models={"slow": _spec("slow", "slow_f16.ckpt", steps=20),
                "here": _spec("here", "here_i8x.ckpt"),
                "gone": _spec("gone", "gone_f16.ckpt", steps=2)},
    )


def test_reports_the_variant_found_and_what_is_missing(tmp_path: Path):
    (tmp_path / "here_q8p.ckpt").touch()
    lines = model_report(_registry([str(tmp_path)]))
    assert "  found    Here  (as here_q8p.ckpt)" in lines
    assert any(line.startswith("  missing  Gone") for line in lines)
    assert lines[-1] == "Upscaler: none found -- Upscale stays disabled"
    assert not any("models.yaml" in line for line in lines)


def test_says_so_when_there_is_no_model_folder(tmp_path: Path):
    lines = model_report(_registry([str(tmp_path / "nope")]))
    assert lines == [f"No Draw Things model folder found (looked in {tmp_path / 'nope'})."]


def test_the_test_render_prefers_a_quick_model_over_fewest_steps(tmp_path: Path):
    (tmp_path / "here_q8p.ckpt").touch()
    (tmp_path / "slow_f16.ckpt").touch()
    (tmp_path / "zit_f16.ckpt").touch()
    reg = _registry([str(tmp_path)])
    # a Turbo-LoRA'd slow model has the fewest steps; it must not win
    reg.models["slow"].draft.steps = 1
    reg.models["z-image-turbo"] = _spec("z-image-turbo", "zit_f16.ckpt", steps=8)
    reg.resolve_installed()
    assert pick_model(reg).key == "z-image-turbo"
    assert pick_model(reg, "slow").key == "slow"


def test_without_a_quick_model_the_first_installed_is_used(tmp_path: Path):
    (tmp_path / "here_q8p.ckpt").touch()
    reg = _registry([str(tmp_path)])
    reg.resolve_installed()
    assert pick_model(reg).key == "here"
    assert pick_model(reg).dt_filename == "here_q8p.ckpt"


def test_no_installed_model_means_no_test_render(tmp_path: Path):
    reg = _registry([str(tmp_path)])
    (tmp_path / "unrelated_f16.ckpt").touch()
    reg.resolve_installed()
    assert pick_model(reg) is None
