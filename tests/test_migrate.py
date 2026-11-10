"""The startup migration: a model an earlier release shipped and this one
dropped comes back under its old key when the install still has its file.

Each test builds a real git clone with release tags -- the migration reads the
old configs from the clone's own history -- and a fake Draw Things folder.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from app import migrate
from app.registry import load_registry

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")


def _model(label, file, *, version=None, steps=10, extra=""):
    v = f"\n    civitai_version_id: {version}" if version else ""
    return (f"\n  {label}:\n    label: {label.title()}\n    dt_filename: {file}{v}\n"
            f"    draft: {{ size: 128, steps: 2 }}\n    final: {{ size: 512, steps: {steps} }}\n"
            f"    upscale: {{ size: 1024, steps: 8, strength: 0.4 }}{extra}\n")


def _config(models_dir: Path, *models: str) -> str:
    return (f"draw_things:\n  models_dirs: [\"{models_dir}\"]\n\nmodels:" + "".join(models))


KEPT = _model("kept", "kept_f16.ckpt")
OLD_ANIME = _model("old-anime", "oldanime_v1_f16.ckpt", version=111, steps=30)
OLD_REAL = _model("old-real", "oldreal_v2_f16.ckpt", version=222, steps=25)


@pytest.fixture
def install(tmp_path: Path):
    """A clone where v1.0.0 shipped old-anime and old-real and v1.1.0 dropped them."""
    root, models, data = tmp_path / "clone", tmp_path / "Models", tmp_path / "data"
    (root / "config").mkdir(parents=True)
    models.mkdir()
    for f in ("kept_f16.ckpt", "oldanime_v1_f16.ckpt", "oldreal_v2_f16.ckpt"):
        (models / f).touch()
    cfg = root / "config" / "models.yaml"

    def git(*args):
        subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "t@example.invalid")
    git("config", "user.name", "t")
    cfg.write_text(_config(models, KEPT, OLD_ANIME, OLD_REAL))
    git("add", ".")
    git("commit", "-qm", "1.0.0")
    git("tag", "v1.0.0")
    cfg.write_text(_config(models, KEPT))
    git("commit", "-qam", "1.1.0")
    git("tag", "v1.1.0")

    tuned = tmp_path / "tuned_models.yaml"
    tuned.write_text(yaml.safe_dump({"tuned": {111: {
        "draw_things_version": "sdxl_base_v0.9",
        "settings": {"prompt_style": "tags",
                     "draft": {"size": 768, "steps": 14},
                     "final": {"size": 1024, "steps": 28},
                     "upscale": {"size": 2048, "steps": 10, "strength": 0.35}}}}}))

    def run(**kw):
        return migrate.migrate_dropped_models(root=root, config_path=cfg, data_dir=data,
                                              tuned_path=tuned, today="2026-11-10", **kw)

    class Install:
        pass

    i = Install()
    i.root, i.cfg, i.models, i.data, i.tuned, i.run, i.git = root, cfg, models, data, tuned, run, git
    i.private = cfg.with_name("models.private.yaml")
    return i


def test_dropped_models_come_back_under_their_old_keys(install):
    assert sorted(install.run()) == ["old-anime", "old-real"]
    reg = load_registry(install.cfg)
    assert list(reg.models) == ["kept", "old-anime", "old-real"]
    anime, real = reg.models["old-anime"], reg.models["old-real"]
    # identity from the old config, so pictures and the publish credit still match
    assert (anime.label, anime.civitai_version_id) == ("Old-Anime", 111)
    assert real.dt_filename == "oldreal_v2_f16.ckpt" and real.civitai_version_id == 222
    # a version measured exactly gets those numbers; one that wasn't, the old entry's
    assert (anime.final.steps, anime.prompt_style) == (28, "tags")
    assert real.final.steps == 25
    text = install.private.read_text()
    assert "shipped until v1.0.0" in text and "measured on this exact version" in text


def test_it_runs_once_and_a_deleted_entry_stays_deleted(install):
    install.run()
    assert install.run() == []
    install.private.unlink()
    assert install.run() == []
    assert not install.private.exists()
    record = json.loads((install.data / migrate.MARKER).read_text())["dropped_models"]
    assert record["old-anime"] == {"from": "v1.0.0", "on": "2026-11-10"}


def test_an_existing_entry_is_never_touched(install):
    """This checkout's case: the dropped models already in the private file."""
    mine = "# mine\n\nmodels:" + _model("old-anime", "oldanime_v1_f16.ckpt", steps=7)
    install.private.write_text(mine)
    assert install.run() == ["old-real"]
    assert install.private.read_text().startswith(mine)
    assert load_registry(install.cfg).models["old-anime"].final.steps == 7


def test_a_file_the_user_already_added_under_another_key_is_left_alone(install):
    install.private.write_text("models:" + _model("my-anime", "oldanime_v1_q8p.ckpt"))
    assert install.run() == ["old-real"]
    assert "old-anime" not in load_registry(install.cfg).models


def test_a_model_no_longer_installed_is_not_added_until_it_is(install):
    (install.models / "oldreal_v2_f16.ckpt").unlink()
    assert install.run() == ["old-anime"]
    (install.models / "oldreal_v2_q8p.ckpt").touch()  # reinstalled, another variant
    assert install.run() == ["old-real"]


def test_an_entry_that_no_longer_loads_is_left_out_and_the_rest_go_in(install):
    """An old entry the current schema can't read must not block the others."""
    bad = _model("old-real", "oldreal_v2_f16.ckpt", extra="\n    final: { size: nope }")
    install.cfg.write_text(_config(install.models, KEPT, OLD_ANIME, bad))
    install.git("commit", "-qam", "1.1.1 with a bad entry")
    install.git("tag", "v1.1.1")
    install.cfg.write_text(_config(install.models, KEPT))
    install.git("commit", "-qam", "1.2.0")
    install.git("tag", "v1.2.0")
    assert install.run() == ["old-anime"]
    load_registry(install.cfg)


def test_no_tags_no_folder_no_git_means_nothing_happens(install, tmp_path):
    shutil.rmtree(install.models)
    assert install.run() == []          # Draw Things folder unreadable: decide nothing
    assert not (install.data / migrate.MARKER).exists()
    install.models.mkdir()
    (install.models / "oldanime_v1_f16.ckpt").touch()
    shutil.rmtree(install.root / ".git")
    assert install.run() == []          # a download, not a clone
    assert not install.private.exists()


def test_startup_survives_a_failing_migration(monkeypatch, caplog):
    def boom(**_):
        raise RuntimeError("disk on fire")

    monkeypatch.delenv("DRAFTPOOL_NO_MIGRATE", raising=False)
    monkeypatch.setattr(migrate, "migrate_dropped_models", boom)
    migrate.run_on_startup()
    assert "migration of dropped models failed" in caplog.text


def test_release_check_refuses_a_drop_it_cannot_migrate(install):
    previous = yaml.safe_load(_config(install.models, KEPT, OLD_ANIME, OLD_REAL))
    assert migrate.check_release(previous, install.cfg, install.tuned) == []
    bad = _model("old-real", "oldreal_v2_f16.ckpt", extra="\n    final: { size: nope }")
    previous = yaml.safe_load(_config(install.models, KEPT, bad))
    assert "'old-real' dropped" in migrate.check_release(previous, install.cfg, install.tuned)[0]
    previous["video_models"] = {"clip": {"label": "Clip", "dt_filename": "clip.ckpt"}}
    assert any("video model 'clip'" in p for p in migrate.check_release(previous, install.cfg, install.tuned))
