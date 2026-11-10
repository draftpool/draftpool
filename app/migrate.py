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
from .private_models import append_entry, entry_block, load_tuned, TUNED_PATH
from .registry import (
    CONFIG_PATH, DEFAULT_MODELS_DIRS, installed_files, load_registry, match_filename,
    private_path,
)

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


def run_on_startup() -> None:
    """migrate_dropped_models for the server: logged, and never fatal."""
    if os.environ.get("DRAFTPOOL_NO_MIGRATE"):
        return
    try:
        added = migrate_dropped_models()
    except Exception:  # noqa: BLE001 - startup must survive a failed migration
        log.exception("migration of dropped models failed; config left as it was")
        return
    if added:
        log.info("migrated %d model(s) an earlier release shipped: %s",
                 len(added), ", ".join(added))


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
