"""Writing entries into config/models.private.yaml, and the measured settings
an entry can take from config/tuned_models.yaml.

Shared by scripts/add_civitai_model.py (a model the user imported) and
app/migrate.py (a model an earlier release shipped and this one does not).
The file itself is the user's: setup and updates never overwrite it, so
everything here appends.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from .registry import CONFIG_PATH

TUNED_PATH = CONFIG_PATH.parent / "tuned_models.yaml"

HEADER = """\
# Models this checkout adds to the ones that ship.
#
# Same format as the `models` section of models.yaml, loaded after it
# (app/registry.py, private_path). scripts/add_civitai_model.py appends to it,
# and so does the startup migration for a model an earlier release shipped;
# entries can also be written by hand. It is never published and never
# overwritten by an update, so your models survive one.

models:
"""


def load_tuned(path: Path = TUNED_PATH) -> dict[int, dict]:
    """Settings measured on exact checkpoints, by Civitai version id."""
    if not path.is_file():
        return {}
    return {int(k): v for k, v in ((yaml.safe_load(path.read_text()) or {}).get("tuned") or {}).items()}


class _Block(str):
    pass


def dump_entry(entry: dict) -> str:
    """An entry as YAML, multi-line text as | blocks the way the shipped file has it."""
    class D(yaml.SafeDumper):
        pass

    D.add_representer(_Block, lambda d, s: d.represent_scalar("tag:yaml.org,2002:str", s, style="|"))

    def blocks(v):
        if isinstance(v, dict):
            return {k: blocks(x) for k, x in v.items()}
        if isinstance(v, list):
            return [blocks(x) for x in v]
        return _Block(v) if isinstance(v, str) and "\n" in v else v

    return yaml.dump(blocks(entry), Dumper=D, sort_keys=False, allow_unicode=True, width=88)


def entry_block(key: str, entry: dict, comment: list[str]) -> str:
    """`key: entry` indented under `models:`, after its comment lines."""
    body = "".join("    " + l + "\n" if l else "\n" for l in dump_entry(entry).splitlines())
    return "\n".join(f"  # {c}" if c else "  #" for c in comment) + f"\n  {key}:\n" + body


def append_entry(path: Path, key: str, text: str) -> None:
    """Add one entry's text under `models:`, creating the file if need be.

    Raises ValueError when the file already has `key`.
    """
    current = path.read_text() if path.is_file() else HEADER
    if re.search(rf"(?m)^  {re.escape(key)}:\s*$", current):
        raise ValueError(f"{path.name} already has a model {key!r}; choose another with --key")
    current = re.sub(r"(?m)^models:[ \t]*\{\}[ \t]*$", "models:", current)
    if not re.search(r"(?m)^models:[ \t]*$", current):
        current = current.rstrip("\n") + "\n\nmodels:\n"
    path.write_text(current.rstrip("\n") + "\n\n" + text)
