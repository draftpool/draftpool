"""The measurement frames: where a run writes, how it is archived, what is missing.

A rendered frame is expensive and a decision is re-read against it, so no
frame is ever regenerated or replaced. The lifecycle:

  1. A sweep writes into its own staging folder, `runs/<topic>__<stamp>/`
     (gitignored), created by `start_run`. Nothing there collides with an
     earlier run, and a run stopped halfway resumes in the same folder.
  2. When it is complete, `archive` moves the set into `research/<topic>/`
     -- every file through `keep`, which never overwrites -- merges the rows
     into that folder's `results.json`, and appends a dated entry with the
     command and the note saying what the run was for to `RUNS.md` beside
     them. The frames stay local (1.7 GB and growing, too much for git); the
     `results.json` and `RUNS.md` are committed and are the manifest.
  3. `check` lists every results.json row whose frame is not on this machine,
     so a lost frame is known before a verdict is trusted, not during.

    uv run python scripts/research_frames.py check            # missing frames
    uv run python scripts/research_frames.py check --all      # every folder
    uv run python scripts/research_frames.py archive runs/draft-size__20260919-070013 \\
        --note "LustrousMix 640/768/1024 untick drafts, Faithfuls read at 1:1"
    uv run python scripts/research_frames.py note research/draft-size/janku \\
        --files 'winter-queen*' --note "..."                  # annotate frames already there
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import ROOT

RESEARCH = ROOT / "research"
RUNS = ROOT / "runs"


def keep(path: Path, data: bytes) -> Path:
    """Write a frame without ever replacing one; the path actually written.

    A taken name gets `<stem>__<timestamp>.png` beside it, so the earlier
    frame -- the one a decision may already rest on -- stays.
    """
    if path.exists():
        path = path.with_name(f"{path.stem}__{_stamp()}{path.suffix}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def start_run(topic: str, resume: str | os.PathLike | None = None) -> Path:
    """The staging folder for a sweep: a fresh `runs/<topic>__<stamp>/`, or
    `resume` to carry on in one that was stopped. Records the command."""
    run = Path(resume) if resume else RUNS / f"{topic}__{_stamp()}"
    run.mkdir(parents=True, exist_ok=True)
    meta = run / "run.json"
    if not meta.is_file():
        try:
            rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                 capture_output=True, text=True).stdout.strip()
        except OSError:
            rev = ""
        meta.write_text(json.dumps({
            "topic": topic, "started": datetime.now().isoformat(timespec="seconds"),
            "command": shlex.join(["uv", "run", "python", *sys.argv]), "git": rev,
        }, indent=1))
    print(f"run folder: {run.relative_to(ROOT)}  (archive it when done: "
          f"scripts/research_frames.py archive {run.relative_to(ROOT)} --note '...')",
          flush=True)
    return run


def archived(topic: str, rel: Path | str) -> Path | None:
    """Where an earlier run's frame of this name is, if one was archived."""
    p = RESEARCH / topic / rel
    return p if p.is_file() else None


def archive(run: Path, note: str, topic: str | None = None) -> Path:
    """Move a finished run into research/<topic>/ and write down what it was.

    Files keep their paths relative to the run folder (so `<key>/x.png` lands
    at `research/<topic>/<key>/x.png`); a name that is taken gets keep()'s
    timestamped sibling and its results row is renamed to match. results.json
    rows are appended to the folder's own; RUNS.md gets the entry.
    """
    meta = json.loads((run / "run.json").read_text()) if (run / "run.json").is_file() else {}
    topic = topic or meta.get("topic") or run.name.split("__", 1)[0]
    into = RESEARCH / topic
    moved: dict[Path, list[tuple[str, str]]] = {}
    for src in sorted(p for p in run.rglob("*") if p.is_file()):
        if src.name in ("run.json", "results.json"):
            continue
        dst = keep(into / src.relative_to(run), src.read_bytes())
        src.unlink()
        moved.setdefault(dst.parent, []).append((src.name, dst.name))
    for log in sorted(run.rglob("results.json")):
        rows = json.loads(log.read_text())
        folder = into / log.parent.relative_to(run)
        renamed = dict(moved.get(folder, []))
        for r in rows:
            if isinstance(r, dict) and r.get("file") in renamed:
                r["file"] = renamed[r["file"]]
        target = folder / "results.json"
        have = json.loads(target.read_text()) if target.is_file() else []
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps([*have, *rows], indent=1))
        log.unlink()
    entry = [
        f"## {datetime.now():%Y-%m-%d %H:%M} — {run.name}", "",
        note.strip(), "",
        f"    {meta.get('command', '(command not recorded)')}", "",
    ]
    if meta.get("git"):
        entry += [f"Code at `{meta['git']}`.", ""]
    for folder, names in sorted(moved.items()):
        entry += [f"`{folder.relative_to(into)}/`: " +
                  ", ".join(d if s == d else f"{s} → {d}" for s, d in names), ""]
    _append_runs_md(into, "\n".join(entry))
    (run / "run.json").unlink(missing_ok=True)
    for leftover in sorted((p for p in run.rglob("*") if p.is_dir()), reverse=True):
        if not any(leftover.iterdir()):
            leftover.rmdir()
    if not any(run.iterdir()):
        run.rmdir()
    return into


def note(folder: Path, files: str, text: str) -> None:
    """Annotate frames that are already in research/ (made before runs/ existed)."""
    folder = folder.resolve()
    names = sorted(p.name for p in folder.glob(files))
    if not names:
        raise SystemExit(f"nothing in {folder} matches {files!r}")
    entry = [f"## {datetime.now():%Y-%m-%d %H:%M} — {folder.relative_to(RESEARCH)}", "",
             text.strip(), "", ", ".join(names), ""]
    _append_runs_md(folder, "\n".join(entry))


def _append_runs_md(folder: Path, entry: str) -> None:
    path = folder / "RUNS.md"
    head = "" if path.is_file() else (
        "# Runs archived here\n\nWhat each set of frames in this folder was rendered "
        "for, oldest first. The frames are local only; `results.json` names them.\n\n")
    with path.open("a") as f:
        f.write(head + entry + "\n")


def check(show_all: bool) -> int:
    lost = 0
    for log in sorted(RESEARCH.glob("**/results.json")):
        try:
            rows = json.loads(log.read_text())
        except ValueError:
            print(f"{log.relative_to(ROOT)}: not JSON")
            continue
        # a row that recorded an empty response never had a frame to keep
        names = sorted({r["file"] for r in rows
                        if isinstance(r, dict) and "file" in r and not r.get("empty")})
        missing = [n for n in names if not (log.parent / n).is_file()]
        lost += len(missing)
        if missing or show_all:
            print(f"{log.parent.relative_to(ROOT)}: {len(names) - len(missing)} of "
                  f"{len(names)} frames on disk")
            for n in missing:
                print(f"  missing  {n}")
    pending = sorted(p for p in RUNS.glob("*") if p.is_dir()) if RUNS.is_dir() else []
    if pending:
        print("\nnot yet archived (runs/):")
        for p in pending:
            print(f"  {p.name}")
    if lost:
        print(f"\n{lost} frame(s) named in a results.json are not on this machine: "
              "the verdicts they back cannot be re-read")
    return 1 if lost else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    c = sub.add_parser("check")
    c.add_argument("--all", action="store_true")
    a = sub.add_parser("archive")
    a.add_argument("run")
    a.add_argument("--note", required=True)
    a.add_argument("--topic", help="research/<topic>; default from the run folder's name")
    n = sub.add_parser("note")
    n.add_argument("folder")
    n.add_argument("--files", default="*.png")
    n.add_argument("--note", required=True)
    args = ap.parse_args()
    if args.cmd in (None, "check"):
        return check(getattr(args, "all", False))
    if args.cmd == "archive":
        into = archive(Path(args.run).resolve(), args.note, args.topic)
        print(f"archived into {into.relative_to(ROOT)}; see its RUNS.md")
        return 0
    note(Path(args.folder), args.files, args.note)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
