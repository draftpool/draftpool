"""Point the whole test run at a throwaway data directory.

`app.main` opens its Store and SettingsStore in DATA_DIR at import, so this
has to happen before any test module imports `app` -- which is why it is done
at conftest import rather than in a fixture. A real environment variable wins
over `.env` (app.load_env), so setting it here is enough.

The directory the install actually uses is kept in
DRAFTPOOL_CONFIGURED_DATA_DIR, for the few tests that only *read* something
a real run left there (test_detailer's probe render) and skip without it.
"""

import atexit
import os
import shutil
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


def _configured_data_dir() -> str:
    # parsed by hand rather than via `app`, which must not be imported yet
    value = os.environ.get("DRAFTPOOL_DATA_DIR")
    if value is None:
        try:
            for line in (_ROOT / ".env").read_text().splitlines():
                key, _, v = line.strip().partition("=")
                if key.removeprefix("export ").strip() == "DRAFTPOOL_DATA_DIR":
                    value = v.split(" #")[0].strip().strip("\"'")
        except OSError:
            pass
    return str(Path(value or _ROOT / "data").expanduser().resolve())


os.environ.setdefault("DRAFTPOOL_CONFIGURED_DATA_DIR", _configured_data_dir())
_TMP = tempfile.mkdtemp(prefix="draftpool-tests-")
atexit.register(shutil.rmtree, _TMP, ignore_errors=True)
os.environ["DRAFTPOOL_DATA_DIR"] = _TMP
