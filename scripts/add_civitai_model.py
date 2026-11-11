"""Add a model you imported into Draw Things yourself, set up from its base model.

    uv run python scripts/add_civitai_model.py
        lists the models imported into Draw Things that the app doesn't use yet

    uv run python scripts/add_civitai_model.py NAME --civitai VERSION_ID
        adds one, NAME being its file or name from that list; the Civitai
        version id (the number after modelVersionId= on its page) names the
        base model and credits the checkpoint when you publish

    uv run python scripts/add_civitai_model.py NAME --air urn:air:sdxl:checkpoint:civitai:MODEL@VERSION
    uv run python scripts/add_civitai_model.py NAME --hash SHA256
        the same, naming the version by the AIR or the SHA256 its Civitai page
        shows (the hash of the file you downloaded; Draw Things' converted copy
        hashes differently)

    uv run python scripts/add_civitai_model.py NAME --base illustrious
        the same without Civitai, naming the base model yourself

The server sets up most imports by itself on the start after them; this is
for one it guessed wrong, or one you want credited when you publish. Delete
the entry the server added first.

Options: --key, --label to choose the app's names for it; --dry-run to print the
entry without writing it.

Some versions have settings measured on that exact file
(config/tuned_models.yaml); given one of those ids, the entry gets them instead
of the base profile's.

The entry goes into config/models.private.yaml, which is yours: setup and
updates never touch it, and it is not part of the published repository. The
settings come from config/base_models.yaml, the profile for the checkpoint's
base model -- what the measured models of that base agree on, not a
measurement of this one. Restart the server afterwards.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.private_models import (  # noqa: E402,F401
    PROFILES_PATH, TUNED_PATH, append_entry as _append_entry, civitai_version,
    civitai_version_by_hash, dump_entry, imported_models, load_profiles, load_tuned, slug,
    version_from_air,
)
from app.registry import CONFIG_PATH, load_registry, match_filename, private_path  # noqa: E402

# the profile name an entry gets when its version was measured here
MEASURED = "measured"


def resolve(profiles: dict, tuned: dict[int, dict], imported: dict, base: str | None,
            civitai_id: int | None, civitai: dict | None) -> tuple[str, dict]:
    """The profile name and the profiles to build from.

    A version measured here wins over its base profile unless --base asks for
    the profile; it is checked against the import's architecture the same way.
    """
    if civitai_id in tuned and not base:
        want = tuned[civitai_id]["draw_things_version"]
        if imported.get("version") != want:
            raise SystemExit(
                f"settings were measured for version {civitai_id} as Draw Things architecture "
                f"{want!r}, and this import is {imported.get('version')!r}: is it the right file?")
        return MEASURED, {**profiles, MEASURED: tuned[civitai_id]}
    return choose_profile(profiles, imported, base, civitai), profiles


def unused(imported: list[dict], known_files: list[str]) -> list[dict]:
    """Imported models no config entry already points at."""
    return [e for e in imported if not any(match_filename(k, {e["file"]}) for k in known_files)]


def pick(imported: list[dict], name: str) -> dict:
    """The one import NAME means: its file or its name, exact first, then a unique part."""
    low = name.lower()
    exact = [e for e in imported if low in (e["file"].lower(), str(e.get("name", "")).lower())]
    if len(exact) == 1:
        return exact[0]
    part = [e for e in imported if low in e["file"].lower() or low in str(e.get("name", "")).lower()]
    if len(part) == 1:
        return part[0]
    if not part:
        raise SystemExit(f"no imported model matches {name!r}; run with no arguments to list them")
    raise SystemExit(f"{name!r} matches several: " + ", ".join(e["file"] for e in part))


def choose_profile(profiles: dict, imported: dict, base: str | None, civitai: dict | None) -> str:
    """Which profile serves this checkpoint, refusing one Draw Things says it cannot be."""
    if base:
        if base not in profiles:
            raise SystemExit(f"no profile {base!r}; there are: {', '.join(profiles)}")
        key = base
    elif civitai and civitai.get("base_model"):
        hits = [k for k, p in profiles.items() if civitai["base_model"] in p["civitai_base_models"]]
        if not hits:
            raise SystemExit(
                f"Civitai says this is a {civitai['base_model']} model, and there is no profile "
                f"for that base ({', '.join(profiles)}). Pony, NoobAI and the rest are left out "
                "on purpose: no model of those bases was measured here. --base forces one if "
                "you know better.")
        key = hits[0]
    else:
        hits = [k for k, p in profiles.items() if p["draw_things_version"] == imported.get("version")]
        if len(hits) != 1:
            raise SystemExit(
                f"Draw Things records this as {imported.get('version')!r}, which fits "
                f"{', '.join(hits) or 'no profile'}. Say which with --civitai VERSION_ID "
                "or --base.")
        key = hits[0]
    want = profiles[key]["draw_things_version"]
    if imported.get("version") != want:
        raise SystemExit(
            f"the {key} profile is for Draw Things architecture {want!r}, and this import is "
            f"{imported.get('version')!r}: the settings would not fit it")
    return key


def build_entry(imported: dict, profile_key: str, profiles: dict,
                civitai: dict | None = None, key: str | None = None,
                label: str | None = None) -> tuple[str, dict]:
    if civitai and civitai.get("model"):
        default_label = f"{civitai['model']} {civitai.get('version') or ''}".strip()
    else:
        default_label = str(imported.get("name") or Path(imported["file"]).stem)
    entry = {"label": label or default_label, "dt_filename": imported["file"]}
    if civitai:
        entry["civitai_version_id"] = int(civitai["id"])
    entry.update(json.loads(json.dumps(profiles[profile_key]["settings"])))
    return key or slug(Path(imported["file"]).stem.removesuffix("_f16")), entry


def entry_text(key: str, entry: dict, profile_key: str, civitai: dict | None) -> str:
    if profile_key == MEASURED:
        lines = [f"  # Added {date.today().isoformat()} by scripts/add_civitai_model.py with the "
                 "settings measured",
                 "  # on this exact version (config/tuned_models.yaml)."]
    else:
        lines = [f"  # Added {date.today().isoformat()} by scripts/add_civitai_model.py from the "
                 f"{profile_key} profile",
                 "  # (config/base_models.yaml): its base model's settings, not measured on this",
                 "  # checkpoint."]
    if civitai and civitai.get("model"):
        who = f", by {civitai['creator']}" if civitai.get("creator") else ""
        lines.append(f"  # {civitai.get('model')} {civitai.get('version')}{who}.")
    body = "".join("    " + l + "\n" if l else "\n" for l in dump_entry(entry).splitlines())
    return "\n".join(lines) + f"\n  {key}:\n" + body


def append_entry(path: Path, key: str, text: str) -> None:
    try:
        _append_entry(path, key, text)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("name", nargs="?", help="the imported model's file or name")
    ap.add_argument("--civitai", type=int, help="its Civitai version id")
    ap.add_argument("--air", help="its Civitai AIR, instead of the version id")
    ap.add_argument("--hash", help="the SHA256 of the downloaded file, instead of the version id")
    ap.add_argument("--base", help="the profile to use, instead of asking Civitai")
    ap.add_argument("--key", help="the app's key for it (default: from the file name)")
    ap.add_argument("--label", help="the name the app shows (default: from Civitai or Draw Things)")
    ap.add_argument("--dry-run", action="store_true", help="print the entry, write nothing")
    args = ap.parse_args(argv)
    if sum(x is not None for x in (args.civitai, args.air, args.hash)) > 1:
        raise SystemExit("name the version one way: --civitai, --air or --hash")
    if args.air:
        try:
            args.civitai = version_from_air(args.air)
        except ValueError as exc:
            raise SystemExit(str(exc))
    elif args.hash:
        try:
            args.civitai = civitai_version_by_hash(args.hash)
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(f"Civitai knows no version with SHA256 {args.hash}: {exc}")

    registry = load_registry(CONFIG_PATH)
    profiles = load_profiles()
    tuned = load_tuned()
    known = [s.dt_filename for s in (*registry.models.values(), *registry.video_models.values())]
    imported = imported_models(registry.draw_things.models_dirs)
    if not imported:
        print("Draw Things has no imported models (no custom.json in "
              f"{', '.join(registry.draw_things.models_dirs)}). Import the checkpoint there "
              "first: Models list -> Manage -> Import Model.")
        return 1
    fits = {p["draw_things_version"] for p in profiles.values()}
    if not args.name:
        rows = [e for e in unused(imported, known) if e.get("version") in fits]
        if not rows:
            print("Every imported model Draw Things lists that a profile can serve is already in use.")
            return 0
        print("Imported into Draw Things and not used by the app yet:\n")
        for e in rows:
            print(f"  {e['file']:52} {e.get('name', '')}  [{e.get('version')}]")
        print("\nAdd one with: uv run python scripts/add_civitai_model.py FILE --civitai VERSION_ID")
        return 0

    chosen = pick(imported, args.name)
    if any(match_filename(k, {chosen["file"]}) for k in known):
        raise SystemExit(f"{chosen['file']} is already in the app's config")
    civitai = None
    if args.civitai:
        try:
            civitai = civitai_version(args.civitai)
        except Exception as exc:  # noqa: BLE001
            if args.civitai not in tuned or args.base:
                raise SystemExit(f"could not ask Civitai about version {args.civitai}: {exc}")
            # measured here: the settings need no lookup, only the label does
            civitai = {"id": args.civitai}
    profile_key, profiles = resolve(profiles, tuned, chosen, args.base, args.civitai, civitai)
    if profile_key == MEASURED:
        # a row that may name its model says what the app calls it
        args.key = args.key or tuned[args.civitai].get("key")
        args.label = args.label or tuned[args.civitai].get("label")
    key, entry = build_entry(chosen, profile_key, profiles, civitai, args.key, args.label)
    if key in registry.models or key in registry.video_models:
        raise SystemExit(f"the app already has a model {key!r}; choose another with --key")
    text = entry_text(key, entry, profile_key, civitai)
    if args.dry_run:
        print(text)
        return 0
    path = private_path(CONFIG_PATH)
    before = path.read_text() if path.is_file() else None
    append_entry(path, key, text)
    try:
        load_registry(CONFIG_PATH)
    except Exception as exc:  # put the file back rather than leave the app unable to start
        if before is None:
            path.unlink()
        else:
            path.write_text(before)
        raise SystemExit(f"the entry did not load, nothing was written: {exc}")
    how = ("with the settings measured on this version" if profile_key == MEASURED
           else f"from the {profile_key} profile")
    shown = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
    print(f"Added {key!r} ({entry['label']}) to {shown} {how}.\n"
          "Restart the server to see it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
