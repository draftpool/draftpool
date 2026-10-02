"""draftpool: local iterative image generation.

Project settings that cannot live in `config/models.yaml`, and where the data
lives.

`config/models.yaml` is tracked in git, so anything a user edits there is a
merge conflict or a silent overwrite on the next update. Settings that differ
per install therefore come from the environment, and from a gitignored `.env`
beside this project as the place to keep them. `scripts/setup.sh` writes that
file on a first run with this install's real data directory in it, rather than
shipping a template of commented-out lines to copy. A real environment variable
always wins over the file, so a one-off run can override it without editing
anything.

The parser here is deliberately about ten lines of stdlib rather than
python-dotenv: it keeps this module importable with nothing installed, which is
what lets `scripts/probe.py` and the tests ask where the data lives even when
the app itself cannot start. `export` prefixes, `#` comments and surrounding
quotes are handled; shell expansion, multi-line values and `${VAR}` references
are not, because a path and a token need none of them.

Everything under `DATA_DIR` is what an install cannot get back: the SQLite
database, and every image and clip ever generated. Everything else in the
project folder -- code, `.venv`, `tools/realesrgan` -- is rebuildable from the
package plus `scripts/setup.sh`, which is what makes updating a matter of
replacing the folder. So the directory is relocatable, and pointing it outside
the project is the advice in docs/SETUP.md: an update then cannot touch it,
whichever way the new version arrives.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / ".env"


def parse_env(text: str) -> dict[str, str]:
    """The `KEY=value` lines of a .env file, in order."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.removeprefix("export ").strip()
        if not key.replace("_", "").isalnum():
            continue
        value = value.strip()
        # a quoted value ends at its closing quote, so a # after that is a
        # comment and a # inside it is part of the path; unquoted, any " #"
        # starts a comment
        if value[:1] in ("\"", "'") and value.count(value[0]) >= 2:
            quote = value[0]
            value = value[1:value.index(quote, 1)]
        else:
            value = value.split(" #")[0].strip()
        out[key] = value
    return out


def load_env(path: Path = ENV_FILE) -> None:
    """Apply a .env file to the environment. Real variables win."""
    try:
        text = path.read_text()
    except OSError:
        return
    for key, value in parse_env(text).items():
        os.environ.setdefault(key, value)


load_env()

DATA_DIR = Path(os.environ.get("DRAFTPOOL_DATA_DIR") or ROOT / "data").expanduser().resolve()
