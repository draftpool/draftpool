"""The test run never opens the install's own data directory.

`app.main` creates its Store and SettingsStore at import, in DATA_DIR, so any
test that imports the app used to open -- and migrate -- the real database and
settings of whatever install it ran in. tests/conftest.py points
DRAFTPOOL_DATA_DIR at a throwaway directory before anything imports `app`.
"""

import os
from pathlib import Path

from app import DATA_DIR, ENV_FILE, ROOT, parse_env


def _configured() -> Path:
    """Where this install keeps its data when no test is running."""
    value = os.environ.get("DRAFTPOOL_CONFIGURED_DATA_DIR")
    if value is None:
        try:
            value = parse_env(ENV_FILE.read_text()).get("DRAFTPOOL_DATA_DIR")
        except OSError:
            value = None
    return Path(value or ROOT / "data").expanduser().resolve()


def test_the_tests_run_on_a_throwaway_data_directory():
    assert DATA_DIR != _configured()
    assert not DATA_DIR.is_relative_to(ROOT)


def test_the_app_stores_open_under_it():
    from app.main import settings_store, store

    assert Path(store._db_path).resolve().is_relative_to(DATA_DIR)
    assert Path(settings_store.path).resolve().is_relative_to(DATA_DIR)
