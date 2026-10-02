"""The .env parser: the one place an install's own settings can live."""

import os
from pathlib import Path

from app import load_env, parse_env


def test_parses_the_shapes_a_hand_written_file_takes():
    parsed = parse_env(
        "\n".join(
            [
                "# a comment",
                "",
                "DRAFTPOOL_DATA_DIR=/tmp/data",
                "export CIVITAI_TOKEN=abc123",
                'QUOTED="/path/with spaces"',
                "SINGLE='/other path'",
                'QUOTED_COMMENT="/tmp/y"  # and a note',
                'HASH_INSIDE="/tmp/a#b"',
                "TRAILING=/tmp/x  # where it lives",
                "not a setting at all",
            ]
        )
    )
    assert parsed == {
        "DRAFTPOOL_DATA_DIR": "/tmp/data",
        "CIVITAI_TOKEN": "abc123",
        "QUOTED": "/path/with spaces",
        "SINGLE": "/other path",
        "QUOTED_COMMENT": "/tmp/y",
        "HASH_INSIDE": "/tmp/a#b",
        # a # inside a quoted value would be part of the path, but bare it is a
        # comment: an unquoted trailing one must not end up in the directory name
        "TRAILING": "/tmp/x",
    }


def test_a_real_environment_variable_wins(tmp_path: Path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("DRAFTPOOL_DATA_DIR=/from/file\nONLY_IN_FILE=yes\n")
    monkeypatch.setenv("DRAFTPOOL_DATA_DIR", "/from/shell")
    load_env(env)
    assert os.environ["DRAFTPOOL_DATA_DIR"] == "/from/shell"
    assert os.environ["ONLY_IN_FILE"] == "yes"


def test_a_missing_file_is_not_an_error(tmp_path: Path):
    load_env(tmp_path / "nope.env")
