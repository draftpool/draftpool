"""scripts/add_civitai_model.py: a config entry for any checkpoint, from its base model.

What matters is that the entry it writes is one the app loads and runs, that
the profile is the right one for what Draw Things says the import is, and that
it refuses rather than guesses when it cannot tell. Nothing here touches the
network or Draw Things: imports are dicts shaped like custom.json rows.
"""

import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts import add_civitai_model as acm  # noqa: E402

from app.registry import CONFIG_PATH, load_registry  # noqa: E402

PROFILES = acm.load_profiles()
SDXL = {"file": "somefinetune_v10_f16.ckpt", "name": "someFinetune_v10", "version": "sdxl_base_v0.9"}
ANIMA = {"file": "someanima_v2_f16.ckpt", "name": "someAnima_v2", "version": "cosmos2.5_2b"}


def civitai(base, model="Some Finetune", version="v1.0"):
    return {"id": 123456, "base_model": base, "model": model, "version": version, "creator": "someone"}


class TestChooseProfile:
    def test_an_anima_import_needs_no_help(self):
        assert acm.choose_profile(PROFILES, ANIMA, None, None) == "anima"

    def test_an_sdxl_import_is_ambiguous_without_civitai_or_a_flag(self):
        # Illustrious and plain SDXL are the same architecture to Draw Things
        with pytest.raises(SystemExit, match="--civitai"):
            acm.choose_profile(PROFILES, SDXL, None, None)

    def test_civitai_names_the_base(self):
        assert acm.choose_profile(PROFILES, SDXL, None, civitai("Illustrious")) == "illustrious"
        assert acm.choose_profile(PROFILES, SDXL, None, civitai("SDXL 1.0")) == "sdxl"

    def test_where_civitai_calls_two_classes_one_base_the_name_decides(self):
        """Civitai files Animagine under "SDXL 1.0", the base of the photographic
        class too; a Danbooru model there must not get prose and lens talk."""
        anim = {"file": "animagine_xl_v3.1_q6p_q8p.ckpt", "name": "Animagine XL v3.1 (8-bit)",
                "version": "sdxl_base_v0.9"}
        assert acm.choose_profile(PROFILES, anim, None, civitai("SDXL 1.0")) == "animagine"
        assert acm.choose_profile(PROFILES, SDXL, None, civitai("SDXL 1.0", "Animagine XL")) \
            == "animagine"

    def test_a_base_with_no_profile_is_refused_not_guessed(self):
        with pytest.raises(SystemExit, match="Pony"):
            acm.choose_profile(PROFILES, SDXL, None, civitai("Pony"))

    def test_the_flag_wins_but_not_against_the_architecture(self):
        assert acm.choose_profile(PROFILES, SDXL, "illustrious", civitai("Pony")) == "illustrious"
        with pytest.raises(SystemExit, match="would not fit"):
            acm.choose_profile(PROFILES, SDXL, "anima", None)
        with pytest.raises(SystemExit, match="would not fit"):
            acm.choose_profile(PROFILES, ANIMA, None, civitai("Illustrious"))


class TestPick:
    ROWS = [SDXL, ANIMA, {"file": "someanima_v3_f16.ckpt", "name": "someAnima_v3", "version": "cosmos2.5_2b"}]

    def test_exact_file_or_name(self):
        assert acm.pick(self.ROWS, "somefinetune_v10_f16.ckpt") is SDXL
        assert acm.pick(self.ROWS, "someAnima_v2") is ANIMA

    def test_a_unique_part(self):
        assert acm.pick(self.ROWS, "finetune") is SDXL

    def test_an_ambiguous_part_lists_the_candidates(self):
        with pytest.raises(SystemExit, match="several"):
            acm.pick(self.ROWS, "someanima")

    def test_nothing_matching(self):
        with pytest.raises(SystemExit, match="no imported model"):
            acm.pick(self.ROWS, "nope")


def test_imports_already_in_the_config_are_not_offered():
    rows = [SDXL, {"file": "juggernaut_xl_ragnarok_f16.ckpt", "version": "sdxl_base_v0.9"}]
    assert acm.unused(rows, ["juggernaut_xl_ragnarok_f16.ckpt"]) == [SDXL]


@pytest.fixture
def config(tmp_path):
    """A copy of the shipped config with no private file beside it."""
    shutil.copy(CONFIG_PATH, tmp_path / "models.yaml")
    return tmp_path / "models.yaml"


@pytest.mark.parametrize("profile,imported,info", [
    ("illustrious", SDXL, civitai("Illustrious")),
    ("sdxl", SDXL, civitai("SDXL 1.0", "Some Photo Model")),
    ("anima", ANIMA, None),
    ("animagine", SDXL, civitai("SDXL 1.0", "Animagine XL")),
])
def test_every_profile_writes_an_entry_the_app_loads(config, profile, imported, info):
    key, entry = acm.build_entry(imported, profile, PROFILES, info)
    text = acm.entry_text(key, entry, profile, info)
    private = config.parent / "models.private.yaml"
    acm.append_entry(private, key, text)
    reg = load_registry(config)
    spec = reg.models[key]
    assert spec.dt_filename == imported["file"]
    settings = PROFILES[profile]["settings"]
    assert spec.sampler == settings["sampler"] and spec.cfg == settings["cfg"]
    assert spec.prompt_style == settings["prompt_style"]
    assert spec.final.size == settings["final"]["size"]
    if info:
        assert spec.civitai_version_id == info["id"]
        assert spec.label.startswith(info["model"])
    # says where its numbers came from, so nobody mistakes them for a measurement
    assert "not measured on this" in private.read_text()


def test_the_anima_profile_carries_the_turbo_tick(config):
    key, entry = acm.build_entry(ANIMA, "anima", PROFILES)
    acm.append_entry(config.parent / "models.private.yaml", key, acm.entry_text(key, entry, "anima", None))
    spec = load_registry(config).models[key]
    assert [o.key for o in spec.lora_options] == ["turbo"]


def test_entries_accumulate_and_an_empty_file_is_filled(config):
    private = config.parent / "models.private.yaml"
    private.write_text("# mine\n\nmodels: {}\n")
    for imported in (SDXL, ANIMA):
        profile = "anima" if imported is ANIMA else "illustrious"
        key, entry = acm.build_entry(imported, profile, PROFILES)
        acm.append_entry(private, key, acm.entry_text(key, entry, profile, None))
    reg = load_registry(config)
    assert {"somefinetune-v10", "someanima-v2"} <= set(reg.models)
    assert private.read_text().startswith("# mine")


def test_a_key_already_in_the_file_is_refused(config):
    private = config.parent / "models.private.yaml"
    key, entry = acm.build_entry(ANIMA, "anima", PROFILES)
    acm.append_entry(private, key, acm.entry_text(key, entry, "anima", None))
    with pytest.raises(SystemExit, match="already has"):
        acm.append_entry(private, key, acm.entry_text(key, entry, "anima", None))


def test_profiles_name_no_model_kept_out_of_the_public_config():
    """The profiles and the tuned file ship; the red models they came from may
    not be named in them. The models the tuned file names itself (not
    adult-flagged) are the ones that may be."""
    from app.registry import private_path

    private = private_path(CONFIG_PATH)
    if not private.is_file():
        pytest.skip("no private models in this checkout")
    import yaml

    named = {r["key"] for r in TUNED.values() if r.get("key")}
    keys = set((yaml.safe_load(private.read_text()) or {}).get("models") or {}) - named
    for path in (acm.PROFILES_PATH, acm.TUNED_PATH):
        text = path.read_text()
        assert not [k for k in keys if k in text], path.name


def test_a_tuned_row_names_its_model_whole_or_not_at_all():
    for vid, row in TUNED.items():
        assert bool(row.get("key")) == bool(row.get("label")), vid
        assert row.get("fingerprints") or row.get("files"), vid


# -- versions measured here ---------------------------------------------------------

TUNED = acm.load_tuned()


def test_the_tuned_file_has_entries():
    assert TUNED, "config/tuned_models.yaml is empty or missing"


def test_a_measured_version_wins_over_its_base_profile():
    vid, row = next(iter(TUNED.items()))
    imported = {"file": "x_f16.ckpt", "name": "x", "version": row["draw_things_version"]}
    key, profiles = acm.resolve(PROFILES, TUNED, imported, None, vid, None)
    assert key == acm.MEASURED and profiles[key] is row


def test_base_asks_for_the_profile_even_for_a_measured_version():
    vid, row = next((v, r) for v, r in TUNED.items() if r["draw_things_version"] == "cosmos2.5_2b")
    imported = {"file": "x_f16.ckpt", "version": "cosmos2.5_2b"}
    key, _ = acm.resolve(PROFILES, TUNED, imported, "anima", vid, None)
    assert key == "anima"


def test_a_measured_version_on_the_wrong_architecture_is_refused():
    vid, row = next(iter(TUNED.items()))
    other = "cosmos2.5_2b" if row["draw_things_version"] != "cosmos2.5_2b" else "sdxl_base_v0.9"
    with pytest.raises(SystemExit, match="right file"):
        acm.resolve(PROFILES, TUNED, {"file": "x_f16.ckpt", "version": other}, None, vid, None)


def test_an_unmeasured_version_falls_back_to_the_profile():
    key, _ = acm.resolve(PROFILES, TUNED, ANIMA, None, 1, None)
    assert key == "anima"


@pytest.mark.parametrize("vid", sorted(TUNED))
def test_every_measured_entry_loads(config, vid):
    row = TUNED[vid]
    imported = {"file": f"m{vid}_f16.ckpt", "name": f"m{vid}", "version": row["draw_things_version"]}
    key, profiles = acm.resolve(PROFILES, TUNED, imported, None, vid, None)
    k, entry = acm.build_entry(imported, key, profiles, {"id": vid})
    private = config.parent / "models.private.yaml"
    acm.append_entry(private, k, acm.entry_text(k, entry, key, {"id": vid}))
    spec = load_registry(config).models[k]
    assert spec.civitai_version_id == vid
    assert spec.sampler == row["settings"]["sampler"]
    assert spec.final.size == row["settings"]["final"]["size"]
    assert "measured" in private.read_text() and "not measured" not in private.read_text()


def test_the_tuned_file_names_no_model():
    """Keyed by version id only: no label, file name, key or comment says which."""
    import re

    text = acm.TUNED_PATH.read_text()
    # an entry's own fields sit at six spaces; a LoRA tick's label is deeper
    assert not re.search(r"(?m)^      (label|dt_filename|civitai_version_id):", text)
    body = text.split("\ntuned:", 1)[1]
    assert "#" not in body


# -- the command itself, against a throwaway config ------------------------------


@pytest.fixture
def cli(config, monkeypatch):
    """main() pointed at a copied config, with Draw Things' import list faked."""
    monkeypatch.setattr(acm, "CONFIG_PATH", config)
    rows = [SDXL, ANIMA, {"file": "juggernaut_xl_ragnarok_f16.ckpt", "name": "Juggernaut",
                          "version": "sdxl_base_v0.9"}]
    monkeypatch.setattr(acm, "imported_models", lambda dirs: rows)
    return config.parent / "models.private.yaml"


def test_listing_offers_only_imports_the_app_does_not_use(cli, capsys):
    assert acm.main([]) == 0
    out = capsys.readouterr().out
    assert SDXL["file"] in out and ANIMA["file"] in out
    assert "juggernaut_xl_ragnarok" not in out
    assert not cli.exists()


def test_adding_writes_the_private_file_beside_the_config(cli, capsys):
    assert acm.main([ANIMA["file"]]) == 0
    assert "someanima-v2" in load_registry(acm.CONFIG_PATH).models
    assert "from the anima profile" in capsys.readouterr().out


def test_a_measured_version_needs_no_civitai_to_be_added(cli, monkeypatch, capsys):
    """Civitai unreachable: the label falls back, the measured settings do not."""
    vid, row = next((v, r) for v, r in TUNED.items() if r["draw_things_version"] == "sdxl_base_v0.9")

    def down(version_id):
        raise OSError("network is unreachable")

    monkeypatch.setattr(acm, "civitai_version", down)
    assert acm.main([SDXL["file"], "--civitai", str(vid)]) == 0
    # a row that may name its model brings its own key
    spec = load_registry(acm.CONFIG_PATH).models[row.get("key") or "somefinetune-v10"]
    assert spec.civitai_version_id == vid
    assert spec.sampler == row["settings"]["sampler"]
    assert spec.final.size == row["settings"]["final"]["size"]
    assert "measured on this version" in capsys.readouterr().out


def test_an_unmeasured_version_with_civitai_down_is_refused(cli, monkeypatch):
    def down(version_id):
        raise OSError("network is unreachable")

    monkeypatch.setattr(acm, "civitai_version", down)
    with pytest.raises(SystemExit, match="could not ask Civitai"):
        acm.main([SDXL["file"], "--civitai", "1"])
    assert not cli.exists()


def test_an_import_already_configured_is_refused(cli):
    with pytest.raises(SystemExit, match="already in the app's config"):
        acm.main(["juggernaut_xl_ragnarok_f16.ckpt", "--base", "sdxl"])


def test_the_version_can_be_named_by_its_air_or_its_hash(cli, monkeypatch):
    """What the Civitai page shows, copied as is: the AIR (with the file id the
    copy button adds) or the SHA256 of the downloaded file."""
    asked = []
    monkeypatch.setattr(acm, "civitai_version", lambda v: asked.append(v) or {
        "id": v, "base_model": "Anima", "model": "Some Anima", "version": "v2", "creator": None})
    monkeypatch.setattr(acm, "civitai_version_by_hash", lambda h: {"ab" * 32: 77}[h])
    assert acm.main([ANIMA["file"], "--air", "urn:air:anima:checkpoint:civitai:12@77+99",
                     "--dry-run"]) == 0
    assert acm.main([ANIMA["file"], "--hash", "ab" * 32, "--dry-run"]) == 0
    assert asked == [77, 77]
    with pytest.raises(SystemExit, match="not a Civitai AIR"):
        acm.main([ANIMA["file"], "--air", "civitai:77"])
    with pytest.raises(SystemExit, match="one way"):
        acm.main([ANIMA["file"], "--air", "civitai:12@77", "--civitai", "77"])
