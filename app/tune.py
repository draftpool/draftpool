"""Tune: a model's guessed settings, settled by renders the user picks between.

A model the app set up from its class (config/base_models.yaml) carries the
settings the measured models of that class agree on, and two of them are known
not to carry over between finetunes of one base: the sampler and the cfg. Four
of six Illustrious checkpoints measured best on one sampler and two on another,
and the family's cfg burned two merges that are clean a third lower
(docs/decisions/MODEL-MEASUREMENTS.md). The size a finetune composes one figure
at is the third thing a class cannot promise. None of these can be scored by a
number -- the one tried, Laplacian sharpness, recommended the accelerator that
ruined the picture -- so the renders are made here and the user judges them.

Up to four rounds, each a question a person can answer by looking:

  class    only where the architecture has more than one class and the model
           is still exactly on one: the scene as each class would prompt and
           render it (opener, negative, clip skip, sampler, cfg), two seeds.
           A class picked on both seeds becomes the start for the rest, which
           are judged on its prompt -- on the wrong class the prompt is a
           starved one, and every later pick would judge that.
  sampler  the class's sampler against the other common one, at its cfg and a
           third lower, two seeds, judged on 1:1 crops at draft size (the
           speckle is not a property of the output size). Blind and shuffled.
           A setting wins only when it is picked on both seeds; anything less,
           and "can't tell", keeps what the model has. Skipped at cfg 1: a
           distilled recipe has no guidance to burn and swapping its sampler
           is how one Anima model rendered pure noise.
  size     whole frames at the final size, two seeds, and one at the widest
           shape: "one person, or more?" -- not a preference, a count. More
           than one steps the final size (or the shape budget) down a notch
           and renders again, at most MAX_ATTEMPTS times.
  steps    the final's step count against two thirds of it, A/B on 1:1 crops;
           the fewer steps win unless the more are picked on a seed. Optional.

Speed LoRAs are deliberately not a round: four were tried on the Illustrious
class and every one drew the figure two or three times at its final size or
replaced its look with a poster's (FINAL-STEPS.md), and the Anima class ships
its Turbo tick already.

What it writes is the user's choice, not a measurement: the entry in
models.private.yaml gets the new values and a comment saying they were picked
in the settings dialog, and nothing goes near config/tuned_models.yaml. Once
the settings are not a profile's, startup's reclass leaves the model alone.

State lives in DATA_DIR/tune/<key>/state.json with the frames beside it, so a
restart resumes where the renders stopped and a page reload loses nothing.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import random
import shutil
import textwrap
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Awaitable, Callable, Optional

import yaml

from . import dims
from .private_models import entry_block, replace_entry
from .registry import CONFIG_PATH, load_registry, private_path

log = logging.getLogger(__name__)

# The add_model.py artefact scene: one figure, skin, loose hair and metal under
# warm light -- where cfg burn and leftover ancestral noise show first, and a
# subject count is unambiguous.
SCENE = (
    "A young woman in polished steel shoulder armour stands in a dim workshop, "
    "loose black hair falling across one cheek, a brass lamp at her shoulder "
    "throwing warm light along the metal and the stray strands of her hair."
)
SEEDS = (1111, 2222)
ROUNDS = ("class", "sampler", "size", "steps")
# the other sampler the measured classes split between
OTHER_SAMPLER = {"Euler a": "DPM++ 2M Karras"}
DEFAULT_OTHER = "Euler a"
# a grid notch for the shape budget (SHAPES.md), a bigger one for the final
SIZE_NOTCH = 256
ASPECT_NOTCH = dims.GRID
MAX_ATTEMPTS = 3
WIDEST = max(dims.ASPECTS, key=dims.ratio)

ANSWERS = {
    "pick": None,                       # an option index, or "cant_tell"
    "count": ("one", "more"),
    "ab": None,                         # an option index, or "same"
}


def lower_cfg(cfg: float) -> float:
    """About a third lower, to the half (add_model.cfg_candidates' rule)."""
    return max(1.0, round(cfg * 2 / 3 * 2) / 2)


def fewer_steps(steps: int) -> int:
    return max(4, round(steps * 2 / 3))


@dataclass
class Settings:
    sampler: str
    cfg: float
    final_size: int
    final_steps: int
    aspect_size: Optional[int]
    draft_size: int

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def settings_of(spec) -> Settings:
    """What the model renders with now, as the rounds see it."""
    return Settings(sampler=spec.sampler or "", cfg=float(spec.cfg),
                    final_size=spec.final.size, final_steps=spec.final.steps,
                    aspect_size=spec.aspect_size if spec.supports_aspect else None,
                    draft_size=spec.draft.size)


def class_option(spec, *, prompt: str, negative: str, entry: dict) -> dict:
    """One candidate of the class round: how that class would render the model.

    `spec` is a ModelSpec built from the class profile's settings on the
    model's own file; `prompt` and `negative` are what the app would send it
    (main.prompt_for, main.negative_for), so each class is judged on its own
    prompt style, opener and negative -- the thing a wrong class gets most
    wrong (MODEL-MEASUREMENTS.md, "animagine"). `entry` is the profile's
    settings as written, for the entry if this class is picked.
    """
    return {
        "prompt": prompt, "negative": negative, "clip_skip": spec.clip_skip,
        "settings": settings_of(spec).as_dict(), "entry": entry,
        "draft": {"size": spec.draft.size, "steps": spec.draft.steps,
                  "sampler": spec.draft.sampler or spec.sampler,
                  "cfg": spec.draft.cfg if spec.draft.cfg is not None else spec.cfg},
    }


def _cell(cid, *, sampler, cfg, steps, width, height, seed, label) -> dict:
    return {"id": cid, "sampler": sampler, "cfg": cfg, "steps": steps, "width": width,
            "height": height, "seed": seed, "label": label, "status": "pending"}


def plan_round(name: str, s: Settings, attempt: int = 1,
               classes: Optional[dict] = None) -> tuple[list[dict], list[dict]]:
    """The cells and the questions of one round, from the settings so far.

    Options in a question are shuffled by a fixed seed so that "the first
    one" is never the current setting, and a reload shows the same order.
    `classes` is the class round's candidates (class_options).
    """
    cells: list[dict] = []
    questions: list[dict] = []
    if name == "class":
        for seed in SEEDS:
            ids = []
            for cname, c in sorted((classes or {}).items()):
                d = c["draft"]
                cid = f"class_{cname}_{seed}"
                cells.append({**_cell(cid, sampler=d["sampler"], cfg=d["cfg"], steps=d["steps"],
                                      width=d["size"], height=d["size"], seed=seed,
                                      label=f"the {cname} settings"),
                              "class": cname, "prompt": c["prompt"], "negative": c["negative"],
                              "clip_skip": c["clip_skip"]})
                ids.append(cid)
            random.Random(seed + 3).shuffle(ids)
            questions.append({"id": f"class_{seed}", "round": name, "kind": "pick",
                              "cells": ids, "crop": False, "answer": None})
    elif name == "sampler":
        samplers = [s.sampler, OTHER_SAMPLER.get(s.sampler, DEFAULT_OTHER)]
        cfgs = [s.cfg, lower_cfg(s.cfg)]
        options = [(a, c) for a in dict.fromkeys(samplers) for c in dict.fromkeys(cfgs)]
        for seed in SEEDS:
            ids = []
            for i, (a, c) in enumerate(options):
                cid = f"sampler_{i}_{seed}"
                cells.append(_cell(cid, sampler=a, cfg=c, steps=s.final_steps,
                                   width=s.draft_size, height=s.draft_size, seed=seed,
                                   label=f"{a}, cfg {c:g}"))
                ids.append(cid)
            random.Random(seed).shuffle(ids)
            questions.append({"id": f"sampler_{seed}", "round": name, "kind": "pick",
                              "cells": ids, "crop": True, "answer": None})
    elif name == "size":
        n = f"size{attempt}"
        for seed in SEEDS:
            cid = f"{n}_square_{seed}"
            cells.append(_cell(cid, sampler=s.sampler, cfg=s.cfg, steps=s.final_steps,
                               width=s.final_size, height=s.final_size, seed=seed,
                               label=f"{s.final_size}px square"))
            questions.append({"id": cid, "round": name, "kind": "count", "cells": [cid],
                              "crop": False, "answer": None, "part": "square"})
        if s.aspect_size:
            w, h = dims.stage_dims(s.final_size, dims.ratio(WIDEST), s.aspect_size)
            cid = f"{n}_wide_{SEEDS[0]}"
            cells.append(_cell(cid, sampler=s.sampler, cfg=s.cfg, steps=s.final_steps,
                               width=w, height=h, seed=SEEDS[0],
                               label=f"{WIDEST} at {w}x{h}"))
            questions.append({"id": cid, "round": name, "kind": "count", "cells": [cid],
                              "crop": False, "answer": None, "part": "wide"})
    elif name == "steps":
        for seed in SEEDS:
            ids = []
            for i, steps in enumerate((s.final_steps, fewer_steps(s.final_steps))):
                cid = f"steps_{i}_{seed}"
                cells.append(_cell(cid, sampler=s.sampler, cfg=s.cfg, steps=steps,
                                   width=s.final_size, height=s.final_size, seed=seed,
                                   label=f"{steps} steps"))
                ids.append(cid)
            random.Random(seed + 7).shuffle(ids)
            questions.append({"id": f"steps_{seed}", "round": name, "kind": "ab",
                              "cells": ids, "crop": True, "answer": None})
    else:
        raise ValueError(f"no round {name!r}")
    return cells, questions


def decide(name: str, s: Settings, questions: list[dict], cells: dict[str, dict],
           attempt: int = 1) -> tuple[Settings, list[str], bool]:
    """What one round's answers change: (settings, notes, render the round again).

    Every rule leans to keeping what the model has: a change needs the same
    pick on both seeds, and "can't tell" is never a vote.
    """
    s = Settings(**s.as_dict())
    notes: list[str] = []
    if name == "class":
        raise ValueError("the class round is decided by Tuner._decide_class")
    if name == "sampler":
        picks = [q["answer"] for q in questions]
        chosen = {cells[q["cells"][a]]["label"] if isinstance(a, int) else None
                  for q, a in zip(questions, picks)}
        if len(chosen) == 1 and None not in chosen:
            cell = cells[questions[0]["cells"][picks[0]]]
            if (cell["sampler"], cell["cfg"]) != (s.sampler, s.cfg):
                notes.append(f"Sampler and cfg: {cell['sampler']}, cfg {cell['cfg']:g}, picked on "
                             f"both seeds over {s.sampler}, cfg {s.cfg:g}.")
                s.sampler, s.cfg = cell["sampler"], cell["cfg"]
            else:
                notes.append(f"Sampler and cfg: {s.sampler}, cfg {s.cfg:g} kept, picked on "
                             "both seeds.")
        else:
            notes.append(f"Sampler and cfg: {s.sampler}, cfg {s.cfg:g} kept -- no other was "
                         "picked on both seeds.")
        return s, notes, False
    if name == "size":
        square = [q for q in questions if q.get("part") == "square"]
        wide = [q for q in questions if q.get("part") == "wide"]
        again = False
        if any(q["answer"] == "more" for q in square):
            lower = s.final_size - SIZE_NOTCH
            if attempt < MAX_ATTEMPTS and lower >= s.draft_size:
                notes.append(f"Final size: more than one person at {s.final_size}px, "
                             f"trying {lower}px.")
                s.final_size, again = lower, True
            else:
                notes.append(f"Final size: still more than one person at {s.final_size}px; "
                             "kept, and worth a look by hand.")
        elif square:
            notes.append(f"Final size: one person at {s.final_size}px.")
        if s.aspect_size and any(q["answer"] == "more" for q in wide):
            lower = s.aspect_size - ASPECT_NOTCH
            if attempt < MAX_ATTEMPTS and lower >= s.draft_size:
                notes.append(f"Shape budget: more than one person at {WIDEST} "
                             f"({s.aspect_size}px budget), trying {lower}px.")
                s.aspect_size, again = lower, True
            else:
                notes.append(f"Shape budget: still more than one person at {WIDEST}; kept.")
        elif wide:
            notes.append(f"Shape budget: one person at {WIDEST}.")
        return s, notes, again
    if name == "steps":
        if any(q["answer"] == "skip" for q in questions):
            notes.append(f"Final steps: {s.final_steps} kept (round skipped).")
            return s, notes, False
        more_picked = False
        for q in questions:
            a = q["answer"]
            if isinstance(a, int) and cells[q["cells"][a]]["steps"] == s.final_steps:
                more_picked = True
        if more_picked:
            notes.append(f"Final steps: {s.final_steps} kept -- picked over "
                         f"{fewer_steps(s.final_steps)}.")
        else:
            few = fewer_steps(s.final_steps)
            notes.append(f"Final steps: {few}, not told apart from {s.final_steps}.")
            s.final_steps = few
        return s, notes, False
    raise ValueError(f"no round {name!r}")


def flat_or_blocked(png: bytes) -> Optional[str]:
    """Why a returned image is not a picture to judge, or None.

    add_model.py's check: one flat colour, or the mid-grey card a refused
    render comes back as (docs/decisions/FALSE-SUCCESS.md).
    """
    from PIL import Image, ImageStat

    img = Image.open(io.BytesIO(png)).convert("RGB")
    st = ImageStat.Stat(img)
    mean = sum(st.mean) / 3
    if max(st.stddev) < 20 and 100 < mean < 160 and max(st.mean) - min(st.mean) < 8:
        return "came back as a flat grey card, which is what a refused render looks like"
    if max(st.stddev) < 3.0:
        return "came back as one flat colour"
    return None


def estimate(cells: list[dict], timings: list[dict], model_key: str) -> Optional[float]:
    """Seconds the pending cells should take, from this install's own pace.

    Store.render_timings is seconds per step by model, kind and stage size;
    a size not on record is scaled from the nearest one by area. None when
    the model has no timings at all.
    """
    rows = [t for t in timings if t["model_key"] == model_key and t["kind"] in ("draft", "final")]
    if not rows:
        return None
    total = 0.0
    for c in cells:
        if c["status"] != "pending":
            continue
        size = dims.stage_size(c["width"], c["height"])
        near = min(rows, key=lambda t: abs(t["size"] - size))
        total += near["sec_per_step"] * (size / near["size"]) ** 2 * c["steps"]
    return round(total)


Render = Callable[[dict], Awaitable[bytes]]


class Tuner:
    """One model's tuning: its state on disk, its renders, its answers."""

    def __init__(self, root: Path, key: str) -> None:
        self.dir = root / key
        self.key = key
        self.path = self.dir / "state.json"

    # -- state ---------------------------------------------------------------

    def load(self) -> Optional[dict]:
        try:
            return json.loads(self.path.read_text())
        except (OSError, ValueError):
            return None

    def save(self, state: dict) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=1))
        tmp.replace(self.path)

    def discard(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def image(self, cell_id: str) -> Path:
        return self.dir / f"{cell_id}.png"

    def start(self, spec, *, prompt: str, negative: str, today: Optional[str] = None,
              classes: Optional[dict] = None, current_class: Optional[str] = None) -> dict:
        """`classes` (name -> class_option), with the model's own among them as
        `current_class`, adds the class round; without them it is skipped."""
        s = settings_of(spec)
        state = {
            "key": self.key, "label": spec.label, "started": today or date.today().isoformat(),
            "prompt": prompt, "negative": negative, "clip_skip": spec.clip_skip,
            "before": s.as_dict(), "settings": s.as_dict(), "notes": [],
            "round": None, "attempt": 1, "cells": {}, "questions": [], "skipped": [],
            "classes": classes if classes and len(classes) > 1 and current_class in classes
            else None,
            "class": current_class, "class_before": current_class,
        }
        state["rounds"] = [r for r in ROUNDS
                           if not (r == "class" and not state["classes"])
                           and not (r == "sampler" and s.cfg <= 1)]
        self._enter(state, self._next_round(state, None))
        self.save(state)
        return state

    def _next_round(self, state: dict, after: Optional[str]) -> Optional[str]:
        order = list(ROUNDS)
        rest = order[order.index(after) + 1:] if after else order
        for name in rest:
            if name == "class" and not state.get("classes"):
                continue
            if name == "sampler" and state["settings"]["cfg"] <= 1:
                state["skipped"].append("sampler")
                state["notes"].append("Sampler and cfg: not tried -- at cfg 1 the model runs "
                                      "a distilled recipe, which is left as it is.")
                continue
            return name
        return None

    def _enter(self, state: dict, name: Optional[str], attempt: int = 1) -> None:
        state["round"], state["attempt"] = name, attempt
        if name is None:
            state["questions"] = []
            return
        cells, questions = plan_round(name, Settings(**state["settings"]), attempt,
                                      classes=state.get("classes"))
        for c in cells:
            state["cells"].setdefault(c["id"], c)
        state["questions"] = questions

    def pending(self, state: dict) -> list[dict]:
        ids = {i for q in state["questions"] for i in q["cells"]}
        return [state["cells"][i] for i in sorted(ids) if state["cells"][i]["status"] == "pending"]

    def phase(self, state: dict) -> str:
        """rendering, asking, failed (nothing in this round can be judged) or review."""
        if state["round"] is None:
            return "review"
        if self.pending(state):
            return "rendering"
        judgeable = all(
            any(state["cells"][i]["status"] == "done" for i in q["cells"])
            for q in state["questions"])
        return "asking" if judgeable else "failed"

    # -- rendering -------------------------------------------------------------

    async def render_pending(self, render: Render, on_change: Callable[[], None] = lambda: None) -> None:
        """Render every cell the current round still lacks, one at a time."""
        while True:
            state = self.load()
            todo = self.pending(state) if state else []
            if not todo:
                return
            cell = todo[0]
            loop = asyncio.get_running_loop()
            began = loop.time()
            try:
                # a class-round cell carries its own class's prompt
                png = await render({"prompt": state["prompt"], "negative": state["negative"],
                                    "clip_skip": state.get("clip_skip", 1), **cell})
                why = flat_or_blocked(png)
            except Exception as exc:  # noqa: BLE001 - the cell fails, the round goes on
                png, why = None, str(exc)[:300] or type(exc).__name__
            state = self.load()
            if state is None or cell["id"] not in state["cells"]:
                return  # discarded while it rendered
            got = state["cells"][cell["id"]]
            if png is not None:
                self.dir.mkdir(parents=True, exist_ok=True)
                self.image(cell["id"]).write_bytes(png)
            got["seconds"] = round(loop.time() - began, 1)
            got["status"], got["error"] = ("error", why) if why else ("done", None)
            self.save(state)
            on_change()

    def retry_failed(self) -> dict:
        state = self.load()
        for q in state["questions"]:
            for i in q["cells"]:
                if state["cells"][i]["status"] == "error":
                    state["cells"][i].update(status="pending", error=None)
        self.save(state)
        return state

    # -- answers ---------------------------------------------------------------

    def answer(self, question_id: str, answer) -> dict:
        state = self.load()
        if state is None:
            raise ValueError("this model is not being tuned")
        if self.phase(state) != "asking":
            raise ValueError("the renders for this round are not ready")
        q = next((q for q in state["questions"] if q["id"] == question_id), None)
        if q is None:
            raise ValueError(f"no question {question_id!r} in this round")
        if q["kind"] == "count":
            if answer not in ANSWERS["count"]:
                raise ValueError("answer one or more")
        elif not (q["kind"] == "pick" and answer == "cant_tell"
                  or q["kind"] == "ab" and answer == "same"):
            if not isinstance(answer, int) or not 0 <= answer < len(q["cells"]):
                raise ValueError("pick one of the pictures")
            if state["cells"][q["cells"][answer]]["status"] != "done":
                raise ValueError("that render failed and cannot be picked")
        q["answer"] = answer
        self._advance(state)
        self.save(state)
        return state

    def skip_round(self) -> dict:
        """Only the steps round is optional."""
        state = self.load()
        if state is None or state["round"] != "steps":
            raise ValueError("only the steps round can be skipped")
        for q in state["questions"]:
            q["answer"] = "skip"
        state["skipped"].append("steps")
        self._advance(state)
        self.save(state)
        return state

    def _advance(self, state: dict) -> None:
        if any(q["answer"] is None for q in state["questions"]):
            return
        name = state["round"]
        if name == "class":
            self._decide_class(state)
            self._enter(state, self._next_round(state, name))
            return
        s, notes, again = decide(name, Settings(**state["settings"]), state["questions"],
                                 state["cells"], state["attempt"])
        state["settings"] = s.as_dict()
        state["notes"].extend(notes)
        if again:
            self._enter(state, name, state["attempt"] + 1)
        else:
            self._enter(state, self._next_round(state, name))


    def _decide_class(self, state: dict) -> None:
        """The class picked on both seeds, or the one the model has.

        A move takes the whole class: its prompt style, opener, negative and
        clip skip for the rounds after this, and its settings as their start.
        """
        picks = []
        for q in state["questions"]:
            a = q["answer"]
            picks.append(state["cells"][q["cells"][a]]["class"] if isinstance(a, int) else None)
        was = state["class"]
        if len(set(picks)) == 1 and picks[0] is not None and picks[0] != was:
            now = picks[0]
            c = state["classes"][now]
            state["class"] = now
            state["notes"].append(f"Kind of model: {now}, picked on both seeds over {was}. "
                                  f"Its prompt style and settings are the start for the rest.")
            state["prompt"], state["negative"] = c["prompt"], c["negative"]
            state["clip_skip"], state["settings"] = c["clip_skip"], dict(c["settings"])
        elif picks and len(set(picks)) == 1 and picks[0] == was:
            state["notes"].append(f"Kind of model: {was} kept, picked on both seeds.")
        else:
            state["notes"].append(f"Kind of model: {was} kept -- no other was picked on "
                                  "both seeds.")


def changes(state: dict) -> dict:
    """{field: (before, after)} for the settings the rounds moved, and the
    class when the class round moved it."""
    b, a = state["before"], state["settings"]
    out = {k: (b[k], a[k]) for k in a if a[k] != b[k]}
    if state.get("class") != state.get("class_before"):
        out = {"class": (state.get("class_before"), state.get("class")), **out}
    return out


def tuned_entry(entry: dict, state: dict) -> dict:
    """The models.private.yaml entry with the picked settings laid in.

    A stage that names its own sampler or cfg keeps it unless it was the
    model's own value (the Illustrious profile repeats them on the draft);
    a draft sampler of its own, like the Anima class's UniPC, is a measured
    choice for that stage and is left alone.
    """
    b, a = state["before"], state["settings"]
    if state.get("class") != state.get("class_before") and state.get("classes"):
        # another class: its whole entry, on this model's file, then the picks
        moved = state["classes"][state["class"]]
        entry = {**{k: entry[k] for k in ("label", "dt_filename", "civitai_version_id")
                    if k in entry},
                 **moved["entry"],
                 **({"fingerprints": entry["fingerprints"]} if entry.get("fingerprints") else {})}
        b = moved["settings"]
    new = json.loads(json.dumps(entry))
    new["sampler"], new["cfg"] = a["sampler"], a["cfg"]
    for stage in ("draft", "final", "upscale"):
        st = new.get(stage)
        if not isinstance(st, dict):
            continue
        if st.get("sampler") == b["sampler"]:
            st["sampler"] = a["sampler"]
        if st.get("cfg") is not None and float(st["cfg"]) == float(b["cfg"]):
            st["cfg"] = a["cfg"]
    new.setdefault("final", {})
    new["final"]["size"], new["final"]["steps"] = a["final_size"], a["final_steps"]
    if a["aspect_size"] is not None:
        new["aspect_size"] = a["aspect_size"]
    return new


def write_tuning(key: str, state: dict, *, config_path: Path = CONFIG_PATH,
                 today: Optional[str] = None) -> str:
    """Put the picked settings into the model's entry; what changed, in a sentence.

    The entry is replaced where it stands, with a comment that says these
    were picked by eye in the settings dialog -- so nobody later reads them
    as a measurement, and private_models.added_keys still counts the entry as
    written for the user. Proven by loading the registry; put back if not.
    """
    private = private_path(config_path)
    entries = ((yaml.safe_load(private.read_text()) or {}).get("models") or {}) \
        if private.is_file() else {}
    if key not in entries:
        raise ValueError("only a model you added can be tuned")
    moved = changes(state)
    stamp = today or date.today().isoformat()
    lines = [f"Tuned {stamp} in the settings dialog, from renders picked by eye: the",
             "class's settings with what the picks changed. Not a measurement."]
    for note in state["notes"]:
        lines += textwrap.wrap(note, 76)
    before = private.read_text()
    replace_entry(private, key, entry_block(key, tuned_entry(entries[key], state), lines))
    try:
        load_registry(config_path)
    except Exception as exc:  # put it back rather than leave the app unable to start
        private.write_text(before)
        raise ValueError(f"the entry did not load, nothing was changed: {exc}") from exc
    if not moved:
        return f"{state['label']}: kept as it was -- nothing the picks preferred."
    said = ", ".join(f"{FIELD_NAMES[k]} {v[0]:g} → {v[1]:g}" if isinstance(v[0], (int, float))
                     and not isinstance(v[0], bool) else f"{FIELD_NAMES[k]} {v[0]} → {v[1]}"
                     for k, v in moved.items())
    return f"{state['label']}: {said}."


FIELD_NAMES = {"sampler": "sampler", "cfg": "cfg", "final_size": "final size",
               "final_steps": "final steps", "aspect_size": "shape budget",
               "draft_size": "draft size"}
