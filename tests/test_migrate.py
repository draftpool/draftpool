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
    """Any step failing is logged, and the others still run.

    All four are stubbed: run_on_startup acts on the real config otherwise.
    """
    def boom(**_):
        raise RuntimeError("disk on fire")

    ran = []
    monkeypatch.delenv("DRAFTPOOL_NO_MIGRATE", raising=False)
    monkeypatch.setattr(migrate, "refresh_measured_models", lambda **_: [])
    monkeypatch.setattr(migrate, "migrate_dropped_models", boom)
    monkeypatch.setattr(migrate, "reclass_guessed_models", lambda **_: ran.append(0) or [])
    monkeypatch.setattr(migrate, "wire_installed_models", lambda **_: ran.append(1) or [])
    migrate.run_on_startup()
    assert "migration of dropped models failed" in caplog.text and ran == [0, 1]
    monkeypatch.setattr(migrate, "migrate_dropped_models", lambda **_: ran.append(2) or [])
    monkeypatch.setattr(migrate, "reclass_guessed_models", boom)
    migrate.run_on_startup()
    assert "moving guessed models to their class failed" in caplog.text and ran == [0, 1, 2, 1]
    monkeypatch.setattr(migrate, "reclass_guessed_models", lambda **_: ran.append(0) or [])
    monkeypatch.setattr(migrate, "wire_installed_models", boom)
    migrate.run_on_startup()
    assert "wiring installed models failed" in caplog.text and ran == [0, 1, 2, 1, 2, 0]


def test_release_check_refuses_a_drop_it_cannot_migrate(install):
    previous = yaml.safe_load(_config(install.models, KEPT, OLD_ANIME, OLD_REAL))
    assert migrate.check_release(previous, install.cfg, install.tuned) == []
    bad = _model("old-real", "oldreal_v2_f16.ckpt", extra="\n    final: { size: nope }")
    previous = yaml.safe_load(_config(install.models, KEPT, bad))
    assert "'old-real' dropped" in migrate.check_release(previous, install.cfg, install.tuned)[0]
    previous["video_models"] = {"clip": {"label": "Clip", "dt_filename": "clip.ckpt"}}
    assert any("video model 'clip'" in p for p in migrate.check_release(previous, install.cfg, install.tuned))


# -- wiring what Draw Things has: recognised, or by class --------------------------

from app.private_models import class_of, entry_block, file_digest, load_profiles  # noqa: E402

PROFILES = {"profiles": {
    "illustrious": {"draw_things_version": "sdxl_base_v0.9", "name_hints": ["illustrious", "noob"],
                    "civitai_base_models": ["Illustrious"],
                    "settings": {"prompt_style": "tags", "draft": {"size": 768, "steps": 14},
                                 "final": {"size": 1536, "steps": 20},
                                 "upscale": {"size": 2048, "steps": 20, "strength": 0.3}}},
    "anima": {"draw_things_version": "cosmos2.5_2b", "civitai_base_models": ["Anima"],
              "settings": {"prompt_style": "tags", "draft": {"size": 768, "steps": 12},
                           "final": {"size": 1536, "steps": 24},
                           "upscale": {"size": 2048, "steps": 18, "strength": 0.3}}},
    "sdxl": {"draw_things_version": "sdxl_base_v0.9", "civitai_base_models": ["SDXL 1.0"],
             "settings": {"prompt_style": "prose", "draft": {"size": 1024, "steps": 14},
                          "final": {"size": 1280, "steps": 30},
                          "upscale": {"size": 2048, "steps": 20, "strength": 0.3}}},
    # listed after sdxl, the plain class of the same Civitai base, on purpose
    "animagine": {"draw_things_version": "sdxl_base_v0.9", "name_hints": ["animagine"],
                  "civitai_base_models": ["SDXL 1.0"],
                  "settings": {"prompt_style": "tags", "draft": {"size": 768, "steps": 14},
                               "final": {"size": 1024, "steps": 28},
                               "upscale": {"size": 2048, "steps": 20, "strength": 0.3}}},
}}


@pytest.fixture
def machine(tmp_path: Path):
    """A Draw Things folder with imports, a config using one of them, and a tuned file."""
    models, data, conf = tmp_path / "Models", tmp_path / "data", tmp_path / "config"
    models.mkdir(), conf.mkdir()
    imports = [
        ("kept_f16.ckpt", "Kept", "sdxl_base_v0.9"),
        ("secretmix_v3_f16.ckpt", "secretMix_v3", "sdxl_base_v0.9"),       # measured
        ("someillustrious_v1_f16.ckpt", "someIllustrious_v1", "sdxl_base_v0.9"),
        ("plainxl_v2_q8p.ckpt", "Plain XL v2", "sdxl_base_v0.9"),
        ("someflux_f16.ckpt", "Some Flux", "flux1"),                        # no class
        ("deleted_v1_f16.ckpt", "Deleted", "sdxl_base_v0.9"),               # listed, gone
    ]
    for file, *_ in imports[:-1]:
        (models / file).touch()
    (models / "custom.json").write_text(json.dumps(
        [{"file": f, "name": n, "version": v} for f, n, v in imports]))
    cfg = conf / "models.yaml"
    cfg.write_text(_config(models, KEPT))
    tuned = conf / "tuned_models.yaml"
    tuned.write_text(yaml.safe_dump({"tuned": {333: {
        "draw_things_version": "sdxl_base_v0.9",
        "files": [file_digest("secretmix_v3_f16.ckpt")],
        "settings": {"prompt_style": "tags", "draft": {"size": 768, "steps": 11},
                     "final": {"size": 1536, "steps": 27},
                     "upscale": {"size": 2048, "steps": 10, "strength": 0.35}}}}}))
    profiles = conf / "base_models.yaml"
    profiles.write_text(yaml.safe_dump(PROFILES, sort_keys=False))

    class M:
        pass

    m = M()
    m.models, m.data, m.cfg, m.tuned = models, data, cfg, tuned
    m.private = cfg.with_name("models.private.yaml")
    m.lookups = []

    def run(names=None):
        def lookup(vid):
            m.lookups.append(vid)
            return names
        return migrate.wire_installed_models(config_path=cfg, data_dir=data, tuned_path=tuned,
                                             profiles_path=profiles, today="2026-11-11",
                                             lookup=lookup)
    m.run = run
    return m


def test_every_installed_model_is_wired_measured_or_by_class(machine):
    added = machine.run({"model": "Secret Mix", "version": "v3.0", "creator": "someone"})
    assert sorted(added) == ["plainxl-v2", "secretmix-v3", "someillustrious-v1"]
    reg = load_registry(machine.cfg)
    secret = reg.models["secretmix-v3"]
    # recognised by its file: the exact version's settings and its credit
    assert (secret.civitai_version_id, secret.final.steps) == (333, 27)
    assert secret.label == "Secret Mix v3.0"
    # the rest by class; where two classes share SDXL, the name decides
    assert reg.models["someillustrious-v1"].prompt_style == "tags"
    plain = reg.models["plainxl-v2"]
    assert (plain.prompt_style, plain.label, plain.civitai_version_id) == ("prose", "Plain XL v2", None)
    text = machine.private.read_text()
    assert "measured\n  # on this exact version" in text and "Secret Mix v3.0, by someone." in text
    assert "set up from the sdxl profile" in text
    assert machine.lookups == [333]  # Civitai is asked only about a recognised version


def test_offline_a_recognised_model_takes_draw_things_name(machine):
    machine.run(None)
    secret = load_registry(machine.cfg).models["secretmix-v3"]
    assert (secret.label, secret.final.steps) == ("secretMix_v3", 27)


def test_it_runs_once_per_file_and_a_deleted_entry_stays_deleted(machine):
    machine.run()
    assert machine.run() == []
    machine.private.unlink()
    assert machine.run() == []
    (machine.models / "plainxl_v2_f16.ckpt").touch()  # another quantisation of the same
    assert machine.run() == []
    record = json.loads((machine.data / migrate.MARKER).read_text())["installed_models"]
    assert record["secretmix_v3_f16.ckpt"] == {"key": "secretmix-v3", "how": "measured",
                                              "on": "2026-11-11"}


def test_a_model_the_config_already_uses_is_left_alone(machine):
    # under another key and another quantisation; or the same version from another file
    machine.private.write_text("models:" + _model("mine", "plainxl_v2_f16.ckpt")
                               + _model("theirs", "renamed_f16.ckpt", version=333))
    assert machine.run() == ["someillustrious-v1"]


def test_a_recognised_name_on_other_weights_falls_back_to_its_class(machine):
    rows = json.loads((machine.models / "custom.json").read_text())
    rows[1]["version"] = "cosmos2.5_2b"  # same name, other weights: an Anima
    (machine.models / "custom.json").write_text(json.dumps(rows))
    assert "secretmix-v3" in machine.run()
    spec = load_registry(machine.cfg).models["secretmix-v3"]
    # its class's settings, and no claim to be the measured version
    assert (spec.civitai_version_id, spec.final.steps) == (None, 24)


def test_the_classes_split_sdxl_by_name_and_refuse_what_fits_none():
    profiles = load_profiles()
    assert class_of({"file": "x_noobai_f16.ckpt", "version": "sdxl_base_v0.9"}, profiles) == "illustrious"
    assert class_of({"file": "big_love_f16.ckpt", "version": "sdxl_base_v0.9"}, profiles) == "sdxl"
    # Danbooru-trained but not Illustrious: tags, at base SDXL's own size
    assert class_of({"file": "animagine_xl_v3.1_q6p_q8p.ckpt", "name": "Animagine XL v3.1 (8-bit)",
                     "version": "sdxl_base_v0.9"}, profiles) == "animagine"
    assert class_of({"file": "x_f16.ckpt", "version": "cosmos2.5_2b"}, profiles) == "anima"
    assert class_of({"file": "x_f16.ckpt", "version": "flux1"}, profiles) is None


def test_a_civitai_download_is_recognised_whatever_draw_things_calls_it():
    """Civitai serves "SOMEMIX v7.777.safetensors"; the import's name is Draw
    Things' rewrite of it, so case, separators and precision words must not
    decide whether a measured checkpoint is recognised."""
    same = ["SOMEMIX v7.777.safetensors", "somemix_v7.777_f16.ckpt", "somemix_v7777_q8p.ckpt",
            "somemix-v7.777.ckpt"]
    assert len({file_digest(n) for n in same}) == 1
    precision = ["othermix_v50f.fp16.safetensors", "othermix_v50f_fp16_f16.ckpt",
                 "othermix_v50f.fp16_f16.ckpt"]
    assert len({file_digest(n) for n in precision}) == 1
    assert file_digest("somemix_v7.777_f16.ckpt") != file_digest("somemix_v7.776_f16.ckpt")


# -- recognising the weights, whatever the import is called ---------------------------


def _ckpt(path: Path, seed: str) -> str:
    """A file shaped like Draw Things' .ckpt (SQLite, one `tensors` table); its fingerprint."""
    import sqlite3

    from app.private_models import fingerprint

    con = sqlite3.connect(path)
    con.execute("CREATE TABLE tensors (name TEXT PRIMARY KEY, type INTEGER, format INTEGER, "
                "datatype INTEGER, dim BLOB, data BLOB)")
    con.executemany("INSERT INTO tensors (name, data) VALUES (?, ?)",
                    [(f"norm-{i}", f"{seed}-{i}".encode()) for i in range(3)]
                    + [("big", b"x" * 70000)])
    con.commit()
    con.close()
    return fingerprint(path)


def _imports(machine, *rows):
    (machine.models / "custom.json").write_text(json.dumps(
        [{"file": f, "name": n, "version": v} for f, n, v in rows]))


@pytest.fixture
def weights(machine):
    """The machine with real fingerprints: one shipped-style entry that names its
    weights, and a measured version the tuned file recognises by them."""
    for f in ("kept_f16.ckpt", "secretmix_v3_f16.ckpt", "someillustrious_v1_f16.ckpt",
              "plainxl_v2_q8p.ckpt", "someflux_f16.ckpt"):
        (machine.models / f).unlink()
    old = machine.models / "named_oldname_f16.ckpt"
    machine.named_fp = _ckpt(old, "named")
    old.unlink()  # imported once under the setup script's name, since deleted
    machine.measured_fp = _ckpt(machine.models / "scratch.ckpt", "measured")
    (machine.models / "scratch.ckpt").unlink()
    machine.cfg.write_text(_config(machine.models, KEPT, _model(
        "named", "named_oldname_f16.ckpt", version=444,
        extra=f'\n    fingerprints: ["{machine.named_fp}"]')))
    row = yaml.safe_load(machine.tuned.read_text())["tuned"][333]
    row.update(key="measured-mix", label="Measured Mix v3", fingerprints=[machine.measured_fp])
    machine.tuned.write_text(yaml.safe_dump({"tuned": {333: row}}))
    return machine


def test_a_configured_model_imported_under_another_name_is_linked_not_added(weights):
    _ckpt(weights.models / "named_v1_f16.ckpt", "named")  # Civitai's name for it today
    _imports(weights, ("named_v1_f16.ckpt", "Named_v1", "sdxl_base_v0.9"))
    assert weights.run() == []
    assert migrate.file_aliases(weights.data) == {"named": "named_v1_f16.ckpt"}
    reg = load_registry(weights.cfg)
    reg.resolve_installed()
    assert not reg.models["named"].installed  # the name alone does not find it
    reg.file_aliases = migrate.file_aliases(weights.data)
    reg.resolve_installed()
    assert reg.models["named"].installed and reg.models["named"].dt_filename == "named_v1_f16.ckpt"


def test_a_second_import_of_weights_already_installed_is_not_a_second_model(weights):
    _ckpt(weights.models / "named_oldname_f16.ckpt", "named")
    _ckpt(weights.models / "named_v1_f16.ckpt", "named")
    _imports(weights, ("named_oldname_f16.ckpt", "x", "sdxl_base_v0.9"),
             ("named_v1_f16.ckpt", "Named_v1", "sdxl_base_v0.9"))
    assert weights.run() == []
    assert migrate.file_aliases(weights.data) == {}
    record = json.loads((weights.data / migrate.MARKER).read_text())["installed_models"]
    assert record["named_v1_f16.ckpt"]["how"] == "duplicate"


def test_measured_weights_are_recognised_under_any_name(weights):
    """The name hash would miss this one; the fingerprint does not."""
    _ckpt(weights.models / "whatever_civitai_calls_it_f16.ckpt", "measured")
    _imports(weights, ("whatever_civitai_calls_it_f16.ckpt", "W", "sdxl_base_v0.9"))
    assert weights.run() == ["measured-mix"]
    spec = load_registry(weights.cfg).models["measured-mix"]
    # a row that may name its model brings its key and label: no Civitai call
    assert (spec.label, spec.civitai_version_id, spec.final.steps) == ("Measured Mix v3", 333, 27)
    assert spec.fingerprints == [weights.measured_fp] and weights.lookups == []


def test_each_import_is_read_once_and_then_the_users_config_is_all_there_is(weights, monkeypatch):
    _ckpt(weights.models / "plain_f16.ckpt", "plain")
    _ckpt(weights.models / "flux_f16.ckpt", "flux")
    _imports(weights, ("plain_f16.ckpt", "Plain", "sdxl_base_v0.9"),
             ("flux_f16.ckpt", "Flux", "flux1"))
    assert weights.run() == ["plain"]
    reads = []
    monkeypatch.setattr(migrate, "fingerprint", lambda p: reads.append(p))
    assert weights.run() == [] and reads == []  # nothing new: nothing opened
    record = json.loads((weights.data / migrate.MARKER).read_text())["installed_models"]
    assert record["flux_f16.ckpt"]["how"] == "none" and record["plain_f16.ckpt"]["how"] == "sdxl"


# -- naming the Civitai version in the settings dialog ---------------------------------


def _civitai(base, model="Some Mix", version="v1"):
    return lambda vid: {"id": vid, "base_model": base, "model": model, "version": version,
                        "creator": "someone"}


def _identify(machine, key, vid, facts):
    return migrate.identify_model(key, vid, config_path=machine.cfg, tuned_path=machine.tuned,
                                  profiles_path=machine.cfg.with_name("base_models.yaml"),
                                  today="2026-11-11", facts=facts)


def test_a_guessed_class_is_corrected_by_the_version(machine):
    """Named nothing Illustrious, so set up as plain SDXL; Civitai knows better."""
    machine.run()
    assert load_registry(machine.cfg).models["plainxl-v2"].prompt_style == "prose"
    said = _identify(machine, "plainxl-v2", 555, _civitai("Illustrious"))
    spec = load_registry(machine.cfg).models["plainxl-v2"]
    assert (spec.prompt_style, spec.civitai_version_id) == ("tags", 555)
    assert "set up as illustrious (it was sdxl)" in said
    # the entry is replaced where it stood, not appended a second time
    assert machine.private.read_text().count("  plainxl-v2:") == 1


def _import(machine, file, name):
    (machine.models / file).touch()
    rows = json.loads((machine.models / "custom.json").read_text())
    rows.append({"file": file, "name": name, "version": "sdxl_base_v0.9"})
    (machine.models / "custom.json").write_text(json.dumps(rows))


def test_a_base_two_classes_share_does_not_undo_the_class_the_name_gave(machine):
    """Civitai calls Animagine "SDXL 1.0", which is the plain class's base too:
    crediting it must not move it onto prose and lens talk."""
    _import(machine, "animaginexl_v4_f16.ckpt", "Animagine XL 4.0")
    machine.run()
    assert load_registry(machine.cfg).models["animaginexl-v4"].prompt_style == "tags"
    said = _identify(machine, "animaginexl-v4", 556, _civitai("SDXL 1.0", "Animagine XL", "4.0"))
    spec = load_registry(machine.cfg).models["animaginexl-v4"]
    assert (spec.prompt_style, spec.final.size, spec.civitai_version_id) == ("tags", 1024, 556)
    assert "settings are kept" in said


def test_the_civitai_name_can_move_a_model_to_the_class_it_names(machine):
    machine.run()
    said = _identify(machine, "plainxl-v2", 557, _civitai("SDXL 1.0", "Animagine XL", "3.1"))
    assert load_registry(machine.cfg).models["plainxl-v2"].prompt_style == "tags"
    assert "set up as animagine (it was sdxl)" in said
    # and a photographic one stays where it is
    machine.run()
    said = _identify(machine, "someillustrious-v1", 558, _civitai("Illustrious"))
    assert "settings are kept" in said


def test_a_model_left_on_the_wrong_class_is_moved_once_on_startup(machine):
    """A release that teaches the classes a new name moves the models a
    previous start put on the wrong one -- only those still exactly on that
    profile, never one credited by Civitai or edited by hand, and only once."""
    profiles_path = machine.cfg.with_name("base_models.yaml")
    hinted = yaml.safe_load(profiles_path.read_text())
    unhinted = json.loads(json.dumps(hinted))
    unhinted["profiles"]["animagine"]["name_hints"] = ["nothing-matches-this"]
    profiles_path.write_text(yaml.safe_dump(unhinted, sort_keys=False))
    for file, name in (("animaginexl_v4_f16.ckpt", "Animagine XL 4.0"),
                       ("animagine_edited_f16.ckpt", "Animagine Edited"),
                       ("animagine_credited_f16.ckpt", "Animagine Credited")):
        _import(machine, file, name)
    machine.run()  # the release before: all three on plain sdxl
    text = machine.private.read_text()
    machine.private.write_text(text.replace(
        "  animagine-edited:\n", "  animagine-edited:\n    clip_skip: 2\n"))
    _identify(machine, "animagine-credited", 559, _civitai("SDXL 1.0", "Something"))
    profiles_path.write_text(yaml.safe_dump(hinted, sort_keys=False))  # the release after
    before = machine.private.read_text()

    def reclass():
        return migrate.reclass_guessed_models(config_path=machine.cfg, data_dir=machine.data,
                                             profiles_path=profiles_path, today="2026-11-12")
    assert reclass() == ["animaginexl-v4"]
    models = load_registry(machine.cfg).models
    assert models["animaginexl-v4"].prompt_style == "tags"
    assert models["animaginexl-v4"].label == "Animagine XL 4.0"
    assert (models["animagine-edited"].prompt_style, models["animagine-edited"].clip_skip) == ("prose", 2)
    assert models["animagine-credited"].prompt_style == "prose"
    assert "Moved 2026-11-12 on startup from the sdxl profile" in machine.private.read_text()
    # still written for the user, not measured: the rules for measured entries pass it by
    from app.private_models import added_keys
    assert "animaginexl-v4" in added_keys(machine.private)
    # put back by hand: left there
    machine.private.write_text(before)
    assert reclass() == []
    assert load_registry(machine.cfg).models["animaginexl-v4"].prompt_style == "prose"


def test_a_measured_version_brings_its_settings(machine):
    machine.run()
    said = _identify(machine, "plainxl-v2", 333, _civitai("Illustrious"))
    spec = load_registry(machine.cfg).models["plainxl-v2"]
    assert (spec.final.steps, spec.civitai_version_id) == (27, 333)
    assert "measured settings applied" in said


def test_settings_that_are_no_profile_are_kept_and_only_credited(machine):
    """Measured, or edited by hand: the version does not overwrite them."""
    machine.private.write_text("models:" + _model("mine", "plainxl_v2_q8p.ckpt", steps=17))
    said = _identify(machine, "mine", 555, _civitai("Illustrious"))
    spec = load_registry(machine.cfg).models["mine"]
    assert (spec.final.steps, spec.civitai_version_id) == (17, 555)
    assert "settings are kept" in said


def test_what_cannot_be_given_a_version_is_refused_with_the_reason(machine):
    machine.run()
    with pytest.raises(ValueError, match="only a model you added"):
        _identify(machine, "kept", 555, _civitai("Illustrious"))   # shipped
    with pytest.raises(ValueError, match="right file"):
        _identify(machine, "plainxl-v2", 555, _civitai("Anima"))   # another architecture
    def down(vid):
        raise OSError("offline")
    with pytest.raises(ValueError, match="could not ask Civitai"):
        _identify(machine, "plainxl-v2", 555, down)
    assert load_registry(machine.cfg).models["plainxl-v2"].civitai_version_id is None


def test_a_new_import_is_noticed_without_a_restart(machine):
    reg = load_registry(machine.cfg)
    assert migrate.has_new_imports(reg, machine.data)
    machine.run()
    assert not migrate.has_new_imports(reg, machine.data)
    (machine.models / "fresh_f16.ckpt").touch()
    rows = json.loads((machine.models / "custom.json").read_text())
    rows.append({"file": "fresh_f16.ckpt", "name": "Fresh", "version": "sdxl_base_v0.9"})
    (machine.models / "custom.json").write_text(json.dumps(rows))
    assert migrate.has_new_imports(reg, machine.data)


def test_every_form_the_civitai_page_shows_names_the_version():
    from app.private_models import version_from_ref

    for ref in ("3313575", " 3313575 ", "https://civitai.com/models/1307857/x?modelVersionId=3313575",
                "urn:air:sdxl:checkpoint:civitai:1307857@3313575+3198988", "civitai:1307857@3313575"):
        assert version_from_ref(ref) == 3313575, ref
    assert version_from_ref("AB" * 32, by_hash=lambda h: 7) == 7
    for bad in ("", "unholy desire", "civitai:1307857", "ab" * 31):
        with pytest.raises(ValueError):
            version_from_ref(bad, by_hash=lambda h: 7)


# -- a measured version whose measurement changed ------------------------------


def test_a_model_set_up_from_a_measurement_follows_it_when_it_changes(tmp_path):
    """Two measured Illustrious versions moved from Euler a to DPM++ after a control with
    their real prompt. Whoever imported one already has the old numbers in
    their own private file, which tuned_models.yaml no longer reaches -- so a
    start moves an entry still exactly on an earlier release's row to the
    current one. Never one edited (or tuned) since, never this checkout's own
    measured source, and once per change."""
    root, data = tmp_path / "clone", tmp_path / "data"
    (root / "config").mkdir(parents=True)

    def git(*args):
        subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)

    old = {"sampler": "Euler a", "cfg": 5.0, "prompt_style": "tags",
           "draft": {"size": 768, "steps": 14}, "final": {"size": 1024, "steps": 20},
           "upscale": {"size": 2048, "steps": 10, "strength": 0.3}}
    new = {**old, "sampler": "DPM++ 2M Karras"}
    row = {"draw_things_version": "sdxl_base_v0.9"}
    tuned_file = root / "config" / "tuned_models.yaml"
    git("init", "-q"); git("config", "user.email", "t@example.invalid"); git("config", "user.name", "t")
    tuned_file.write_text(yaml.safe_dump({"tuned": {111: {**row, "settings": old}}}))
    git("add", "."); git("commit", "-qm", "1.0.0"); git("tag", "v1.0.0")
    tuned_file.write_text(yaml.safe_dump({"tuned": {111: {**row, "settings": new}}}))

    cfg = root / "config" / "models.yaml"
    cfg.write_text(_config(tmp_path / "Models", KEPT))
    added = ["Added 2026-11-11 on startup: installed in Draw Things, with the settings measured",
             "on this exact version (config/tuned_models.yaml)."]
    blocks = [
        entry_block("mine", {"label": "Mine", "dt_filename": "mine_f16.ckpt",
                             "civitai_version_id": 111, **old, "fingerprints": ["ab"]}, added),
        entry_block("edited", {"label": "Edited", "dt_filename": "ed_f16.ckpt",
                               "civitai_version_id": 111, **{**old, "cfg": 4.0}}, added),
        # this checkout's own measured source: written by hand, no added mark
        entry_block("source", {"label": "Source", "dt_filename": "src_f16.ckpt",
                               "civitai_version_id": 111, **old}, ["MEASURED by hand."]),
    ]
    private = cfg.with_name("models.private.yaml")
    private.write_text("models:\n" + "\n".join(blocks))

    def run():
        return migrate.refresh_measured_models(root=root, config_path=cfg, data_dir=data,
                                               tuned_path=tuned_file, today="2026-11-12")
    assert run() == ["mine"]
    models = load_registry(cfg).models
    assert models["mine"].sampler == "DPM++ 2M Karras" and models["mine"].fingerprints == ["ab"]
    assert (models["edited"].sampler, models["source"].sampler) == ("Euler a", "Euler a")
    from app.private_models import added_keys
    assert "mine" in added_keys(private)
    assert "Updated 2026-11-12 on startup from the measured settings" in private.read_text()
    # put back by hand: left there
    private.write_text(private.read_text().replace("sampler: DPM++ 2M Karras", "sampler: Euler a", 1))
    assert run() == []
