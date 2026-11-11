"""Bring back, on startup, the models an earlier release shipped and this one
does not.

A release that drops a model from config/models.yaml would otherwise take it
away from everyone who was using it: its chip disappears and every picture
made with it loses the model behind Retry, Finalize and Upscale, because a
generation records only the model's key. 1.1.0 did exactly that to six
checkpoints, and told the user to add them back by hand -- information their
own install already had.

It still has it. An install is a git clone, so every release's models.yaml is
in its history, with the key, the Draw Things file and the Civitai version of
each model. For every model some earlier release shipped and the current one
does not, whose file is still in Draw Things and which nothing in the config
already claims (by key or by file), an entry is appended to
models.private.yaml under the old key -- so the pictures and the per-model
settings stored against that key keep working. Its settings are the ones
measured on that exact version (config/tuned_models.yaml) when there are any,
else the entry as it last shipped.

Rules that keep it from breaking a working install:

- It never edits or removes an existing entry, only appends.
- A key it migrated once is recorded in DATA_DIR/migrations.json and never
  migrated again, so an entry the user deletes stays deleted.
- Each entry is proven first by loading the whole registry with it on a
  copy; one that does not load is left out and logged, the rest go in, and
  the real file is written once.
- No git, no tags (a shallow clone, a download), no Draw Things folder: it
  does nothing. Any error is logged and startup carries on.

scripts/publish.sh runs `check_release` against the last published tag, so a
release cannot drop a model this cannot bring back.

The same startup then decides what each new Draw Things import is
(`wire_installed_models`), once per import: a configured model under another
name, a version measured here (recognised by its weights' fingerprint, or by
a hash of its file name; config/tuned_models.yaml), or a model of a class
(config/base_models.yaml) -- each added to models.private.yaml, which is
what the app uses from then on. Same rules: append only, a deleted entry
stays deleted, proven on a copy first, never fatal.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import tempfile
from datetime import date
from pathlib import Path

import yaml

from . import DATA_DIR, ROOT
from .private_models import (
    append_entry, civitai_version, class_of, entry_block, file_digest, fingerprint,
    imported_models, load_profiles, load_tuned, replace_entry, slug, PROFILES_PATH, TUNED_PATH,
)
from .registry import (
    _stem, _TRAILING_TOKEN, CONFIG_PATH, DEFAULT_MODELS_DIRS, installed_files, load_registry,
    match_filename, private_path,
)
from .schemas import ModelSpec

log = logging.getLogger(__name__)

MARKER = "migrations.json"
# Keys a release dropped on purpose (a model that stopped working, say) and
# that must not come back. Empty so far.
RETIRED: set[str] = set()
_TAG = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")
# the fields an entry carries that are not settings
_IDENTITY = ("label", "dt_filename", "civitai_version_id")


def _git(root: Path, *args: str) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(root), *args], capture_output=True,
                             text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 else None


def release_configs(root: Path = ROOT) -> list[tuple[str, dict]]:
    """(tag, models.yaml) for every release tag in the clone, oldest first."""
    if not (root / ".git").exists():
        return []
    tags = [t for t in (_git(root, "tag", "-l", "v*") or "").split() if _TAG.match(t)]
    tags.sort(key=lambda t: tuple(int(x) for x in _TAG.match(t).groups()))
    out = []
    for tag in tags:
        text = _git(root, "show", f"{tag}:config/models.yaml")
        if text:
            try:
                out.append((tag, yaml.safe_load(text) or {}))
            except yaml.YAMLError:
                continue
    return out


def dropped_models(history: list[tuple[str, dict]], shipped: set[str]) -> dict[str, tuple[str, dict]]:
    """Every image model a release shipped that `shipped` lacks: key -> (last tag, entry)."""
    out: dict[str, tuple[str, dict]] = {}
    for tag, cfg in history:  # oldest first, so the last release's entry wins
        for key, entry in (cfg.get("models") or {}).items():
            if key not in shipped and key not in RETIRED and isinstance(entry, dict):
                out[key] = (tag, entry)
    return out


def migrated_entry(entry: dict, tuned: dict[int, dict]) -> tuple[dict, bool]:
    """The entry to write, and whether its settings are the measured ones."""
    head = {k: entry[k] for k in _IDENTITY if entry.get(k) is not None}
    version = entry.get("civitai_version_id")
    if version in tuned:
        settings = json.loads(json.dumps(tuned[version]["settings"]))
        return {**head, **settings}, True
    return {**head, **{k: v for k, v in entry.items() if k not in _IDENTITY}}, False


def _with_entry(private_text: str | None, key: str, text: str, name: str) -> str:
    """private_text with the entry appended, as append_entry would write it."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / name
        if private_text is not None:
            path.write_text(private_text)
        append_entry(path, key, text)
        return path.read_text()


def _load_error(config_path: Path, private_text: str | None, key: str, text: str) -> str | None:
    """Why the registry would not load with this entry added, or None."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Path(tmp) / config_path.name
        cfg.write_text(config_path.read_text())
        try:
            private_path(cfg).write_text(_with_entry(private_text, key, text, private_path(cfg).name))
            load_registry(cfg)
        except Exception as exc:  # noqa: BLE001 - any reason is a reason to leave it out
            return str(exc)
    return None


def _read_marker(data_dir: Path) -> dict:
    try:
        return json.loads((data_dir / MARKER).read_text())
    except (OSError, ValueError):
        return {}


def migrate_dropped_models(*, root: Path = ROOT, config_path: Path = CONFIG_PATH,
                           data_dir: Path = DATA_DIR, tuned_path: Path = TUNED_PATH,
                           today: str | None = None) -> list[str]:
    """Append the dropped models this install still has; the keys added."""
    history = release_configs(root)
    if not history:
        return []
    shipped_raw = yaml.safe_load(config_path.read_text()) or {}
    dropped = dropped_models(history, set(shipped_raw.get("models") or {}))
    if not dropped:
        return []
    marker = _read_marker(data_dir)
    done = set(marker.get("dropped_models") or {})
    registry = load_registry(config_path)
    dirs = list(registry.draw_things.models_dirs or DEFAULT_MODELS_DIRS)
    files = installed_files(dirs)
    if not files:
        # Draw Things' folder unreadable right now: decide nothing, try next start
        return []
    taken_keys = set(registry.models) | set(registry.video_models)
    claimed = [s.dt_filename for s in (*registry.models.values(), *registry.video_models.values())]
    tuned = load_tuned(tuned_path)
    private = private_path(config_path)
    current = private.read_text() if private.is_file() else None

    blocks: list[tuple[str, str]] = []
    for key, (tag, entry) in dropped.items():
        if key in done or key in taken_keys:
            continue
        found = match_filename(str(entry.get("dt_filename") or ""), files)
        if not found:
            continue
        if any(match_filename(c, {found}) for c in claimed):
            # the user has it under another key already; theirs stands
            continue
        new, measured = migrated_entry(entry, tuned)
        how = ("the settings measured on this exact version (config/tuned_models.yaml)"
               if measured else "its settings as they shipped in that release")
        comment = [f"Migrated {today or date.today().isoformat()}: shipped until {tag}, "
                   "dropped from models.yaml after it,",
                   "and still installed in Draw Things. Kept under its old key, with",
                   f"{how}."]
        text = entry_block(key, new, comment)
        # each entry proven on its own, on a copy, so one this schema can no
        # longer read is left out instead of holding back the rest forever
        problem = _load_error(config_path, current, key, text)
        if problem:
            log.warning("migration: %s from %s does not load, left out: %s", key, tag, problem)
            continue
        current = _with_entry(current, key, text, private.name)
        blocks.append((key, tag))
    if not blocks:
        return []
    private.write_text(current)

    record = marker.get("dropped_models") or {}
    for key, tag in blocks:
        record[key] = {"from": tag, "on": today or date.today().isoformat()}
    marker["dropped_models"] = record
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / MARKER).write_text(json.dumps(marker, indent=2) + "\n")
    return [key for key, _ in blocks]


def _civitai_names(version_id: int) -> dict | None:
    """Civitai's names for a version, or None offline: a label is not worth a stall."""
    try:
        return civitai_version(version_id, timeout=8)
    except Exception:  # noqa: BLE001 - any failure falls back to Draw Things' name
        return None


def _key_for(file: str) -> str:
    """The app's key for an import: its file name without the packaging noise."""
    name = file.removesuffix(".ckpt").lower()
    while (m := _TRAILING_TOKEN.search(name)) is not None:
        name = name[: m.start()]
    return slug(name)


def wire_installed_models(*, config_path: Path = CONFIG_PATH, data_dir: Path = DATA_DIR,
                          tuned_path: Path = TUNED_PATH, profiles_path: Path = PROFILES_PATH,
                          today: str | None = None, lookup=_civitai_names) -> list[str]:
    """Decide, once, what each new import in Draw Things is; the keys added.

    Every import Draw Things lists and has on disk is looked at the first
    time a start sees it, and the outcome is recorded (DATA_DIR/MARKER), so
    after that the user's config is all there is: a later start costs
    nothing until something new is imported. In order:

    - already configured by name: nothing to do;
    - its fingerprint is a configured model's (`fingerprints`, or a measured
      version a config entry credits) whose own file is not on disk: the same
      weights imported under another name, linked to that entry
      (`file_aliases`) instead of becoming a second model;
    - the fingerprint, or failing it the file name, is a version measured
      here (config/tuned_models.yaml): an entry with those settings, under
      the row's key and label when the model may be named;
    - its architecture fits a model class (config/base_models.yaml): an
      entry with that class's settings;
    - otherwise left alone.

    `lookup(version_id)` gives Civitai's names for a measured version the
    row does not name, or None, and Draw Things' own name is used.
    """
    tuned = load_tuned(tuned_path)
    profiles = load_profiles(profiles_path)
    marker = _read_marker(data_dir)
    record = marker.get("installed_models") or {}
    seen = {_stem(f) for f in record}
    registry = load_registry(config_path)
    registry.file_aliases = file_aliases(data_dir)
    dirs = list(registry.draw_things.models_dirs or DEFAULT_MODELS_DIRS)
    files = installed_files(dirs)
    specs = list(registry.models.values())
    claimed = [s.dt_filename for s in (*specs, *registry.video_models.values())]
    by_version = {s.civitai_version_id: s for s in specs if s.civitai_version_id}
    # fingerprint -> the configured model it is, else the measured version
    known: dict[str, ModelSpec] = {fp: s for s in specs for fp in s.fingerprints}
    for vid, row in tuned.items():
        for fp in row.get("fingerprints") or ():
            if vid in by_version:
                known.setdefault(fp, by_version[vid])
    measured = {fp: vid for vid, row in tuned.items() for fp in row.get("fingerprints") or ()}
    by_name = {d: vid for vid, row in tuned.items() for d in row.get("files") or ()}
    taken_keys = set(registry.models) | set(registry.video_models)
    private = private_path(config_path)
    current = private.read_text() if private.is_file() else None
    stamp = today or date.today().isoformat()

    decided: dict[str, dict] = {}
    added: list[str] = []
    # Draw Things' list of imports, not the folder: it says what each file is,
    # and the text encoders, VAEs and LoRAs beside them are not on it
    for imported in sorted(imported_models(dirs), key=lambda e: e["file"]):
        file = imported["file"]
        if file not in files or _stem(file) in seen or file in decided:
            continue
        if any(match_filename(c, {file}) for c in claimed):
            decided[file] = {"how": "configured", "on": stamp}
            continue
        fp = fingerprint(Path(_folder_of(file, dirs)) / file)
        vid = measured.get(fp) if fp else None
        if vid is None:
            vid = by_name.get(file_digest(file))
        spec = (known.get(fp) if fp else None) or by_version.get(vid)
        if spec is not None:
            if match_filename(spec.dt_filename, files) or spec.key in registry.file_aliases:
                decided[file] = {"how": "duplicate", "key": spec.key, "on": stamp}
            else:
                decided[file] = {"how": "alias", "key": spec.key, "on": stamp}
                registry.file_aliases[spec.key] = file
            continue
        row = tuned.get(vid) if vid is not None else None
        if row and imported.get("version") == row["draw_things_version"]:
            how, settings = "measured", row["settings"]
        else:
            # no measurement, or the same name on other weights: its class
            # still might fit
            row, vid = None, None
            how = class_of(imported, profiles)
            if not how:
                decided[file] = {"how": "none", "on": stamp}
                continue
            settings = profiles[how]["settings"]
        key = (row or {}).get("key") or _key_for(file)
        if key in taken_keys:
            log.warning("installed model %s: key %r is taken, left for "
                        "scripts/add_civitai_model.py --key", file, key)
            decided[file] = {"how": "key taken", "key": key, "on": stamp}
            continue
        entry = {"label": str(imported.get("name") or Path(file).stem), "dt_filename": file}
        if row:
            names = {} if row.get("label") else (lookup(vid) or {})
            civitai_name = (f"{names['model']} {names.get('version') or ''}".strip()
                            if names.get("model") else None)
            entry["label"] = row.get("label") or civitai_name or entry["label"]
            entry["civitai_version_id"] = vid
            comment = [f"Added {stamp} on startup: installed in Draw Things, with the "
                       "settings measured",
                       "on this exact version (config/tuned_models.yaml)."]
            if civitai_name:
                who = f", by {names['creator']}" if names.get("creator") else ""
                comment.append(f"{civitai_name}{who}.")
        else:
            comment = [f"Added {stamp} on startup: installed in Draw Things, set up from "
                       f"the {how} profile",
                       "(config/base_models.yaml): its class's settings, not measured on this",
                       "checkpoint. scripts/add_civitai_model.py --air or --hash adds the credit."]
        entry.update(json.loads(json.dumps(settings)))
        if fp:
            entry["fingerprints"] = [fp]
        text = entry_block(key, entry, comment)
        problem = _load_error(config_path, current, key, text)
        if problem:
            log.warning("installed model %s does not load, left out: %s", file, problem)
            decided[file] = {"how": "does not load", "on": stamp}
            continue
        current = _with_entry(current, key, text, private.name)
        taken_keys.add(key)
        claimed.append(file)
        decided[file] = {"how": how, "key": key, "on": stamp}
        added.append(key)
    if not decided:
        return []
    if added:
        private.write_text(current)
    marker["installed_models"] = {**record, **decided}
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / MARKER).write_text(json.dumps(marker, indent=2) + "\n")
    return added


def identify_model(key: str, version_id: int, *, config_path: Path = CONFIG_PATH,
                   tuned_path: Path = TUNED_PATH, profiles_path: Path = PROFILES_PATH,
                   today: str | None = None, facts=civitai_version) -> str:
    """Give one of the user's models its Civitai version; what that changed, in a sentence.

    The settings dialog's answer to an import startup could only guess at.
    A version measured here brings its measured settings. Otherwise a model
    still on a class profile moves to the profile Civitai's base model names
    when that is a different one (an Illustrious checkpoint whose name did
    not say so, set up as plain SDXL). Settings that are no profile's --
    measured or edited by hand -- are kept, and only the credit is added.
    `facts(version_id)` is Civitai's description (civitai_version).

    Raises ValueError, with the reason for the user, when it cannot: a
    shipped model, a version of another architecture than the import, or
    Civitai unreachable for a version not measured here.
    """
    private = private_path(config_path)
    entries = (yaml.safe_load(private.read_text()) or {}).get("models") or {} \
        if private.is_file() else {}
    if key not in entries:
        raise ValueError("only a model you added can be given its Civitai version here")
    entry = entries[key]
    tuned = load_tuned(tuned_path)
    profiles = load_profiles(profiles_path)
    registry = load_registry(config_path)
    dirs = list(registry.draw_things.models_dirs or DEFAULT_MODELS_DIRS)
    imported = next((e for e in imported_models(dirs)
                     if match_filename(str(entry.get("dt_filename") or ""), {e["file"]})), {})
    arch = imported.get("version")
    try:
        info = facts(version_id)
    except Exception as exc:  # noqa: BLE001
        if version_id not in tuned:
            raise ValueError(f"could not ask Civitai about version {version_id}: {exc}") from exc
        info = {"id": version_id}
    name = (f"{info['model']} {info.get('version') or ''}".strip() if info.get("model") else None)

    identity = ("label", "dt_filename", "civitai_version_id", "fingerprints")
    current = {k: v for k, v in entry.items() if k not in identity}
    row = tuned.get(version_id)
    if row:
        if arch and arch != row["draw_things_version"]:
            raise ValueError(f"version {version_id} was measured as a {row['draw_things_version']} "
                             f"model, and this import is {arch}: is it the right file?")
        settings, how = row["settings"], "measured settings applied"
        label = row.get("label") or name or entry["label"]
    else:
        label = entry["label"]
        right = [k for k, p in profiles.items()
                 if info.get("base_model") in (p.get("civitai_base_models") or ())]
        on_profile = next((k for k, p in profiles.items() if p["settings"] == current), None)
        settings, how = current, "credited; its settings are kept"
        if right and arch and profiles[right[0]]["draw_things_version"] != arch:
            raise ValueError(f"Civitai says version {version_id} is a {info['base_model']} "
                             f"model, and this import is {arch}: is it the right file?")
        if right and on_profile and on_profile != right[0]:
            settings = profiles[right[0]]["settings"]
            how = f"set up as {right[0]} (it was {on_profile})"
        elif not right and info.get("base_model"):
            how = f"credited; no profile for {info['base_model']}, so its settings are kept"
    new = {"label": label, "dt_filename": entry["dt_filename"], "civitai_version_id": version_id,
           **({"fingerprints": entry["fingerprints"]} if entry.get("fingerprints") else {}),
           **json.loads(json.dumps(settings))}
    comment = [f"Civitai version {version_id} given in the settings dialog "
               f"{today or date.today().isoformat()}: {how}."]
    if name:
        who = f", by {info['creator']}" if info.get("creator") else ""
        comment.append(f"{name}{who}.")
    before = private.read_text()
    replace_entry(private, key, entry_block(key, new, comment))
    try:
        load_registry(config_path)
    except Exception as exc:  # put the file back rather than leave the app unable to start
        private.write_text(before)
        raise ValueError(f"the entry did not load, nothing was changed: {exc}") from exc
    return f"{label}: {how}."


def _folder_of(file: str, dirs: list[str]) -> str:
    """The models folder holding `file`, of the ones Draw Things uses."""
    for d in dirs:
        if (Path(d).expanduser() / file).is_file():
            return str(Path(d).expanduser())
    return str(Path(dirs[0]).expanduser())


def has_new_imports(registry, data_dir: Path = DATA_DIR) -> bool:
    """Whether Draw Things lists an installed import no start has decided yet.

    Cheap -- its import list, a folder listing and the record -- so the page
    can ask on every refresh, and a model imported while the server runs is
    set up without a restart.
    """
    dirs = list(registry.draw_things.models_dirs or DEFAULT_MODELS_DIRS)
    seen = {_stem(f) for f in _read_marker(data_dir).get("installed_models") or {}}
    files = installed_files(dirs)
    return any(e["file"] in files and _stem(e["file"]) not in seen for e in imported_models(dirs))


def file_aliases(data_dir: Path = DATA_DIR) -> dict[str, str]:
    """Model key -> the file its weights were found in under another name."""
    record = _read_marker(data_dir).get("installed_models") or {}
    return {v["key"]: f for f, v in record.items() if v.get("how") == "alias"}


def run_on_startup() -> None:
    """migrate_dropped_models, then wire_installed_models: logged, and never fatal."""
    if os.environ.get("DRAFTPOOL_NO_MIGRATE"):
        return
    try:
        added = migrate_dropped_models()
    except Exception:  # noqa: BLE001 - startup must survive a failed migration
        log.exception("migration of dropped models failed; config left as it was")
    else:
        if added:
            log.info("migrated %d model(s) an earlier release shipped: %s",
                     len(added), ", ".join(added))
    try:
        wired = wire_installed_models()
    except Exception:  # noqa: BLE001
        log.exception("wiring installed models failed; config left as it was")
        return
    if wired:
        log.info("added %d installed model(s) Draw Things has: %s", len(wired), ", ".join(wired))


def check_release(previous: dict, config_path: Path = CONFIG_PATH,
                  tuned_path: Path = TUNED_PATH) -> list[str]:
    """What stops a release: models `previous` shipped that cannot be migrated.

    Every image model dropped since `previous` (the last published
    models.yaml) must come back as a loadable entry beside the new config; a
    dropped video model cannot be migrated at all and must not happen.
    """
    shipped = yaml.safe_load(config_path.read_text()) or {}
    problems = [f"video model {k!r} dropped: the migration cannot bring a video model back"
                for k in (previous.get("video_models") or {})
                if k not in (shipped.get("video_models") or {}) and k not in RETIRED]
    dropped = dropped_models([("previous", previous)], set(shipped.get("models") or {}))
    tuned = load_tuned(tuned_path)
    for key, (_, entry) in dropped.items():
        new, _measured = migrated_entry(entry, tuned)
        problem = _load_error(config_path, None, key, entry_block(key, new, ["check"]))
        if problem:
            problems.append(f"model {key!r} dropped and its migrated entry does not load: {problem}")
    return problems
