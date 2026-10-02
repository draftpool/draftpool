"""Per-install overrides of what config/models.yaml ships: step counts, and
which models the UI offers at all.

`config/models.yaml` is tracked in git, so a number the user changes from the
UI cannot be written back to it: the next update either conflicts with the edit
or silently reverts it, and either way the measurements and the reasoning
recorded in the comments above each value would be edited by a spinner. So the
file stays the shipped default and this holds what an install has decided
differently, as `settings.json` under `DATA_DIR` -- the one directory an update
does not touch (see the module docstring in `app/__init__.py`).

Only `draft` and `final` are overridable. `upscale` is not: its step count is
half of a pair with `strength`, and the pass is tuned as a whole.

An override for a model that is no longer in the registry is kept rather than
dropped, so removing a model from models.yaml for a release and putting it back
does not lose the setting.

It also holds which optional LoRAs are ticked (see LoraOption in
app/schemas.py): only the ticks set by hand, so an untouched option follows
whatever default the shipped config gives it.

The same file holds which models the chips, pickers and Animate list leave out
-- a per-model tick, plus one rule that sweeps up everything not installed.
That is a tidying of the UI and nothing more: the API still renders any model
asked for by key, so an old card's Upscale, Refine and Retry keep working after
its model is hidden. Storing the *disabled* keys rather than the enabled ones
is what makes a model added by a later release show up on its own.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from . import DATA_DIR

log = logging.getLogger(__name__)

# The range the UI offers and the API accepts. The floor is Draw Things': below
# 2 steps its sampler discretization crashes (see dt_client._build_payload).
# The ceiling is well past anything shipped -- the slowest model here finals at
# 44 -- and exists so a stray keystroke cannot queue an hour-long render.
STEP_MIN = 2
STEP_MAX = 64

STAGES = ("draft", "final")


class SettingsStore:
    """The `settings.json` file, read and written whole.

    It is a handful of integers touched by hand a few times an install, so
    every read goes to disk rather than to a cache that could go stale against
    an editor -- and every write rewrites the file through a temp file, so an
    interrupted save cannot leave a half-written one behind.
    """

    def __init__(self, data_dir: Path = DATA_DIR) -> None:
        self.path = Path(data_dir) / "settings.json"

    def _read(self) -> dict:
        try:
            data = json.loads(self.path.read_text())
        except FileNotFoundError:
            return {}
        except (OSError, ValueError) as exc:
            # a corrupt file is not worth taking the app down for: the shipped
            # defaults are a working configuration
            log.warning("ignoring unreadable %s: %s", self.path, exc)
            return {}
        return data if isinstance(data, dict) else {}

    def _write(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
        os.replace(tmp, self.path)

    def steps(self) -> dict[str, dict[str, int]]:
        """{model_key: {"draft": n, "final": n}}, only for what is overridden.

        Anything that is not an in-range integer for a known stage is dropped
        here rather than at the point of use, so a hand-edited file cannot put
        a value into a render that the API would have rejected.
        """
        raw = self._read().get("steps")
        if not isinstance(raw, dict):
            return {}
        out: dict[str, dict[str, int]] = {}
        for key, stages in raw.items():
            if not isinstance(stages, dict):
                continue
            clean = {
                stage: value
                for stage, value in stages.items()
                if stage in STAGES
                and isinstance(value, int)
                and not isinstance(value, bool)
                and STEP_MIN <= value <= STEP_MAX
            }
            if clean:
                out[key] = clean
        return out

    def set_steps(self, model_key: str, values: dict[str, int | None]) -> dict[str, int]:
        """Set (int) or reset (None) each named stage. Returns what is left.

        The caller states every stage it means to control, so resetting one is
        the same call as changing it -- there is no separate delete path to
        keep in step with the write path.
        """
        for stage, value in values.items():
            if stage not in STAGES:
                raise ValueError(f"unknown stage: {stage}")
            if value is not None and not STEP_MIN <= value <= STEP_MAX:
                raise ValueError(f"steps must be between {STEP_MIN} and {STEP_MAX}")
        data = self._read()
        steps = data.get("steps")
        if not isinstance(steps, dict):
            steps = {}
        data["steps"] = steps
        current = dict(steps.get(model_key) or {})
        for stage, value in values.items():
            if value is None:
                current.pop(stage, None)
            else:
                current[stage] = int(value)
        if current:
            steps[model_key] = current
        else:
            steps.pop(model_key, None)
        self._write(data)
        return current

    def disabled_models(self) -> set[str]:
        """Keys the user has unticked. Unknown keys ride along (see module doc)."""
        raw = self._read().get("disabled_models")
        if not isinstance(raw, list):
            return set()
        return {k for k in raw if isinstance(k, str)}

    def set_model_enabled(self, model_key: str, enabled: bool) -> set[str]:
        """Tick or untick one model. Returns the whole disabled set after."""
        data = self._read()
        disabled = self.disabled_models()
        if enabled:
            disabled.discard(model_key)
        else:
            disabled.add(model_key)
        data["disabled_models"] = sorted(disabled)
        self._write(data)
        return disabled

    def hide_uninstalled(self) -> bool:
        """Whether models missing from Draw Things are left out of the UI.

        A standing rule rather than a bulk untick of today's missing models:
        an install whose model folder grows and shrinks would otherwise need
        the box hit again after every change.
        """
        return self._read().get("hide_uninstalled") is True

    def set_hide_uninstalled(self, hide: bool) -> bool:
        data = self._read()
        data["hide_uninstalled"] = bool(hide)
        self._write(data)
        return bool(hide)

    def clear_steps(self) -> None:
        """Drop every step override; every model goes back to what it ships."""
        data = self._read()
        data.pop("steps", None)
        self._write(data)

    def lora_choices(self) -> dict[str, dict[str, bool]]:
        """{model_key: {option_key: on}}, only for ticks this install has set.

        A tick never set is not stored, so the option follows its shipped
        `default` -- including when a later release changes that default.
        """
        raw = self._read().get("loras")
        if not isinstance(raw, dict):
            return {}
        out: dict[str, dict[str, bool]] = {}
        for key, options in raw.items():
            if not isinstance(options, dict):
                continue
            clean = {k: v for k, v in options.items() if isinstance(v, bool)}
            if clean:
                out[key] = clean
        return out

    def set_lora_choice(self, model_key: str, option_key: str, on: bool | None) -> None:
        """Tick or untick one model's LoRA option; None goes back to its default."""
        data = self._read()
        loras = data.get("loras")
        if not isinstance(loras, dict):
            loras = {}
        data["loras"] = loras
        current = dict(loras.get(model_key) or {})
        if on is None:
            current.pop(option_key, None)
        else:
            current[option_key] = bool(on)
        if current:
            loras[model_key] = current
        else:
            loras.pop(model_key, None)
        self._write(data)
