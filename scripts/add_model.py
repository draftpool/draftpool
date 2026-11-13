"""Measure a newly installed Draw Things model and print its models.yaml entry.

    uv run python scripts/add_model.py --model z_image_turbo_1.0_i8x.ckpt --key z-image-turbo
    uv run python scripts/add_model.py --model <file> --key <key> --level quick
    uv run python scripts/add_model.py --model <file> --key <key> --like <sibling> \
        --level confirm --card-sampler "Euler a" --card-cfg 5 --card-steps 30

One resumable batch that prints an entry with the numbers filled in, leaving a
person only the parts that need eyes. docs/ADDING-A-MODEL.md is the how-to;
docs/decisions/MODEL-MEASUREMENTS.md is why it measures what it measures.

WHY THERE IS A --level confirm, AND WHEN IT IS HONEST
A full batch is ~30 renders and takes an hour on an SDXL checkpoint, and most
of it re-establishes facts that belong to the ARCHITECTURE rather than to the
checkpoint: that SDXL at 512 is a quarter of its native area and so drafts
badly, that these models reach 2048 untiled, that a dual-CLIP merge nobody has
probed gets the conservative prompt_budget default. Those are already the
shipped values for every Illustrious/SDXL entry in config/models.yaml, and
three of them are labelled there as convention rather than measurement --
honestly, in the file, where the next person can see it.

So for a checkpoint whose architecture is IDENTICAL to a sibling already in the
registry, --like inherits that geometry and --level confirm renders only what
cannot be inherited: the sampler and the cfg. Those are not the shortcut, they
are the whole point. MODEL-MEASUREMENTS.md records that copying them from a
neighbour is what gave three checkpoints the wrong sampler, and the artefact
they produce (rainbow speckle on skin, hair edges and metal, waxy unformed
faces) reads as "this model is bad" rather than as a setting.

--like REFUSES to inherit from an entry whose own sampler and cfg are not
measured, and refuses when the architecture does not match, because "an
unmeasured entry is not a source" is the actual lesson of that incident. What
it inherits it prints, field by field, so the entry it emits cannot quietly
launder a guess into a measurement.

WHAT THIS DECIDES WITHOUT RENDERING ANYTHING
Draw Things ships its own recommended settings for every model it offers, in
its cache (see catalogue() below): sampler, steps, cfg, clip skip and the text
encoder. Look there before probing for anything.

Some fields are reasoned rather than measured, because measuring them costs
renders and the reasoning is sound:
  uses_negative_prompt   cfg > 1 means real guidance, so a negative does
                         something; at cfg 1 it cannot.
  prompt_style           "tags" only for Danbooru/booru-trained checkpoints;
                         everything else is prose.
  max_upscale_size       the largest size actually measured good, which is a
                         floor on capability rather than a discovered ceiling.
  upscale.steps          the final's step count, scaled down for strength.

WHAT STILL NEEDS A PERSON
Duplicate subjects, whether a draft looks like its final, and whether an image
is any good are not things this can score. It writes contact sheets and tells
you which ones to look at. Everything numeric -- timings, flat/blocked/empty
detection, the prompt-length curves, upscale correlation -- is automatic.

WHY THERE ARE CONTROL RENDERS
A Draw Things session that has stopped rendering answers 200 with no image and
keeps passing GET /, and a whole batch of measurements was once taken in that
state and believed. So a 256px render on a known-good model proves the session
works before anything is measured, and another runs straight after any cell
that fails, to say whether the session or the cell is at fault. See
docs/decisions/FALSE-SUCCESS.md.
"""

from __future__ import annotations

import argparse
import base64
import fcntl
import io
import json
import math
import re
import sys
import time
from pathlib import Path

import httpx
import numpy as np
from PIL import Image, ImageStat

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import DATA_DIR  # noqa: E402
from app.registry import installed_files, load_registry, match_filename  # noqa: E402
from scripts.research_frames import keep  # noqa: E402

HOST = "http://127.0.0.1:7860"
CATALOGUE = Path.home() / (
    "Library/Containers/com.liuliu.draw-things/Data/Library/Caches/net"
)
# Draw Things stores its sampler as an enum; these are the names the HTTP API
# accepts. Only the ones current models actually use are mapped.
SAMPLERS = {10: "Euler A Trailing", 12: "DPM++ 2M Karras", 16: "DDIM Trailing",
            17: "UniPC Trailing", 19: "LCM"}

# The subject wears red and the marker is a yellow balloon, so both are scored
# off a colour mask rather than by eye. Keep the subject framed LARGE: the
# measurement is how far it falls, and a scene that starts small has no range.
# See docs/decisions/PROMPT-LENGTH.md.
MARKER = "A yellow balloon floats above her."
BASE = "A woman in a red coat stands on a gravel path in a park, looking towards the camera."
PAD = [
    "Wooden benches line the path on both sides.",
    "Clipped box hedges run along behind the benches.",
    "A black iron lamp post stands at the corner of the walk.",
    "A green litter bin sits beside it, with a folded newspaper on top.",
    "Fallen leaves have gathered along the kerb and between the gravel stones.",
    "A low brick wall edges the lawn beyond the hedges.",
    "Thin cloud drifts across a pale afternoon sky.",
    "Two pigeons pick at the gravel a little way off.",
    "A wooden signpost points towards the bandstand and the tea room.",
    "The path curves away to the left past a bed of dark ivy.",
    "A chain fence runs beside the lawn, its posts painted the same black as the lamp.",
    "An empty bandstand stands in the distance under bare trees.",
]
LEVELS = {"L0": 109, "L1": 236, "L2": 393, "L3": 592, "L4": 860}

# Skin, hair edges and metal in one frame, lit warm. Those are the three places
# CFG burn and leftover ancestral noise actually show (MODEL-MEASUREMENTS.md),
# and a scene without all three hides the thing this is looking for. Prose on
# purpose: the tag checkpoints here are prompted in prose too
# (docs/decisions/PROMPT-STYLE.md), so one scene serves every model.
ARTEFACT = (
    "A young woman in polished steel shoulder armour stands in a dim workshop, "
    "loose black hair falling across one cheek, a brass lamp at her shoulder "
    "throwing warm light along the metal and the stray strands of her hair."
)

# Keys whose sampler AND cfg were established by rendering this model, rather
# than inherited from a neighbour -- transcribed from
# docs/decisions/MODEL-MEASUREMENTS.md. --like will inherit from these and
# refuse anything else, because the incident that section records is exactly
# one unmeasured entry seeding three models' worth of wrong sampler: a value
# with a caveat next to it still reads as a value. ADD A KEY HERE ONLY AFTER
# ITS OWN RENDERS, and the file says which those were.
MEASURED_SOURCES = {
    "bas-anima",                   # sampler sweep: DPM++ renders pure noise here
    "lustrousmix-anima",
    "cyberrealistic-anima",
    "juggernaut-ragnarok",
}
# A model in config/models.private.yaml joins them by carrying this marker in
# its comments, so the gate covers it without its key being named in a
# published file (scripts/publish.sh refuses an export that names one).
MEASURED_MARKER = "MEASURED SOURCE"


def measured_sources() -> set[str]:
    """MEASURED_SOURCES and every private model whose entry carries the marker."""
    from app.registry import CONFIG_PATH, private_path

    path = private_path(CONFIG_PATH)
    if not path.is_file():
        return set(MEASURED_SOURCES)
    marked, pending, key = set(), "", None
    for line in path.read_text().splitlines():
        m = re.match(r"^  ([a-z0-9][a-z0-9-]*):\s*$", line)
        if m:
            key = m.group(1)
            if MEASURED_MARKER in pending:
                marked.add(key)
            pending = ""
        elif re.match(r"^  #", line):
            pending += line
        elif key and MEASURED_MARKER in line:
            marked.add(key)
    return set(MEASURED_SOURCES) | marked

# Every render goes under research/, never data/, and no frame is ever
# replaced (research_frames.keep): the harness used to write data/add_model/
# <key> and overwrite it per run, and data/ is what a user moves wholesale,
# which is how the first set of these was lost (see CLAUDE.md). Confirm-level
# runs, the ones models.yaml cites, keep their own folder.
RESEARCH = Path(__file__).resolve().parent.parent / "research" / "confirm"
RUNS = Path(__file__).resolve().parent.parent / "research" / "add_model"

# What is the same for every checkpoint of one architecture, so what --like may
# carry across. Deliberately NOT sampler, cfg or anything in `draft`: those are
# the three MODEL-MEASUREMENTS.md records as having been copied wrongly.
INHERITED = ("final", "upscale", "prompt_budget", "prompt_style", "clip_skip",
             "tiled_diffusion", "upscale_seconds", "max_upscale_size",
             "aspect_size", "supports_img2img")


def ladder() -> dict[str, str]:
    out = {}
    for level, target in LEVELS.items():
        text, i = BASE, 0
        while len(text) + 1 + len(MARKER) < target and i < len(PAD):
            text += " " + PAD[i]
            i += 1
        out[level] = text + " " + MARKER
    return out


def catalogue(dt_filename: str) -> dict:
    """Draw Things' own recommended settings for this model, or {}.

    Its cache holds two files: models.json describes each model (text encoder,
    autoencoder, a prose note that often states the recommended step count and
    licence), configs.json holds a default configuration per model. They are
    keyed by different filenames -- the catalogue names a q6p/q8p build where
    the user may have installed i8x -- so both are matched on the stem, the
    same way registry.match_filename does.
    """
    from app.registry import _stem

    want = _stem(dt_filename)
    out: dict = {}
    try:
        models = json.loads((CATALOGUE / "models.json").read_text())
        configs = json.loads((CATALOGUE / "configs.json").read_text())
    except OSError:
        return out
    for entry in models:
        if _stem(entry.get("file", "")) == want:
            out["note"] = entry.get("note", "")
            out["text_encoder"] = entry.get("text_encoder", "")
            out["autoencoder"] = entry.get("autoencoder", "")
            out["version"] = entry.get("version", "")
            break
    for entry in configs:
        c = entry.get("configuration", {})
        if _stem(c.get("model", "")) == want:
            out["name"] = entry.get("name", "")
            out["sampler"] = SAMPLERS.get(c.get("sampler"))
            out["sampler_id"] = c.get("sampler")
            out["steps"] = c.get("steps")
            out["cfg"] = c.get("guidanceScale")
            out["shift"] = c.get("shift")
            out["clip_skip"] = c.get("clipSkip", 1)
            break
    return out


def prompt_kit(*, tags: bool, opener: str | None, negative: str | None,
               spec=None) -> tuple[str, str]:
    """The opener and negative every measured cell is sent with.

    What the app sends the model (main.prompt_for, main.negative_for), because
    a cell measures the model as it will be used. Until 12 November 2026 every
    cell here went out as bare prose with an empty negative, and on Animagine
    -- which leans on its quality tags harder than any model here -- that made
    a whole confirm run soft and nearly misread the sampler
    (MODEL-MEASUREMENTS.md, "animagine"). In order: the card's own, given on
    the command line; the model's entry, when it is configured as a tag model;
    the tag default for --tags; else nothing, which is right for prose.
    """
    from app.schemas import DEFAULT_TAG_NEGATIVE, DEFAULT_TAG_PREFIX

    if opener is not None or negative is not None:
        return opener or "", negative or ""
    if spec is not None and spec.prompt_style == "tags":
        return spec.quality_opener, spec.default_negative if spec.uses_negative_prompt else ""
    if tags:
        return DEFAULT_TAG_PREFIX, DEFAULT_TAG_NEGATIVE
    return "", ""


# fastest first (scripts/probe.py's QUICK_MODELS): a control is 256px at 4 steps,
# and most of its cost is Draw Things switching checkpoints
CONTROL_MODELS = ("z-image-turbo", "flux2-klein", "flux3-klein")


def control_model(models: dict, key: str):
    """The model a control render uses: a fast installed one, never `key`.

    It used to be the first model drafting at 512 or less; once every draft
    was 768 there was none, and batches ran with no controls at all.
    """
    usable = {k: m for k, m in models.items()
              if k != key and m.installed is not False and not m.edit_only}
    for k in (*CONTROL_MODELS, *usable):
        if k in usable:
            return usable[k]
    return None


class Renderer:
    """Sequential, resumable, and suspicious of every response.

    Results are keyed by name in results.json, so a re-run skips finished
    cells. A blocked render is a valid PNG and would otherwise count as
    finished, so blocked cells are recorded as failures: they re-run rather
    than being scored as data.
    """

    def __init__(self, out: Path, control_model: str | None, *, opener: str = "",
                 negative: str = "", control_sampler: str | None = None):
        self.out = out
        # sent with every control: without one Draw Things renders with what
        # its UI is left on, and Z-Image Turbo answered every control of a
        # batch with an empty image that way -- a healthy session reported
        # as stalled
        self.control_sampler = control_sampler
        self.alive: bool | None = None  # the last control's answer, once one ran
        # what the app sends this model with every prompt (prompt_kit); the
        # control render is another model and goes out bare
        self.opener, self.negative = opener, negative
        self.out.mkdir(parents=True, exist_ok=True)
        self.path = out / "results.json"
        self.results: dict[str, dict] = {}
        if self.path.exists():
            self.results = {r["name"]: r for r in json.loads(self.path.read_text())}
        self.control_model = control_model
        self.controls = 0

    def _save(self):
        self.path.write_text(json.dumps(list(self.results.values()), indent=1))

    def render(self, name: str, **job) -> dict:
        measured = job["model"] != self.control_model
        opener, negative = (self.opener, self.negative) if measured else ("", "")
        done = self.results.get(name)
        # a cell rendered with another opener or negative is another
        # measurement, not this one finished: every add_model.py cell went out
        # bare until 12 November 2026, and those must not be resumed as these
        if (done and done.get("ok") and not job.get("fresh")
                and done.get("opener", "") == opener
                and done.get("negative", "") == negative):
            return done
        prompt = job.get("prompt", "")
        payload = {
            "prompt": f"{opener}, {prompt}" if opener and prompt else (opener or prompt),
            "negative_prompt": negative,
            "model": job["model"],
            "width": job["width"],
            "height": job["height"],
            "steps": max(2, job["steps"]),
            "cfg_scale": job.get("cfg", 1.0),
            "seed": job.get("seed", 4242),
            "batch_size": 1,
            "loras": [],
            "upscaler": job.get("upscaler"),
            "upscaler_scale": job.get("upscaler_scale", 0) if job.get("upscaler") else 0,
        }
        if job.get("sampler"):
            payload["sampler_name"] = job["sampler"]
        if job.get("clip_skip", 1) > 1:
            payload["clip_skip"] = job["clip_skip"]
        if max(job["width"], job["height"]) >= 2048:
            payload["tiled_decoding"] = True
        if job.get("tiled_diffusion"):
            payload.update({"tiled_diffusion": True, "diffusion_tile_width": 1024,
                            "diffusion_tile_height": 1024, "diffusion_tile_overlap": 128})
        endpoint = "/sdapi/v1/txt2img"
        if job.get("init"):
            init = Path(job["init"]).read_bytes()
            img = Image.open(io.BytesIO(init))
            if img.size != (job["width"], job["height"]):
                img = img.convert("RGB").resize((job["width"], job["height"]), Image.LANCZOS)
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                init = buf.getvalue()
            payload["init_images"] = [base64.b64encode(init).decode()]
            payload["denoising_strength"] = job.get("strength", 0.3)
            endpoint = "/sdapi/v1/img2img"

        t0 = time.time()
        try:
            with httpx.Client(timeout=job.get("timeout", 3600)) as c:
                resp = c.post(HOST + endpoint, json=payload)
        except httpx.HTTPError as exc:
            rec = {"name": name, "ok": False, "why": f"transport: {exc}",
                   "seconds": round(time.time() - t0, 1)}
            self.results[name] = rec
            self._save()
            return rec
        rec = {"name": name, "seconds": round(time.time() - t0, 1),
               "w": job["width"], "h": job["height"], "steps": job["steps"],
               "seed": job.get("seed", 4242), "opener": opener, "negative": negative}
        images = resp.json().get("images") or [] if resp.status_code == 200 else []
        if not images:
            # Empty is not one diagnosis. It could be a dead session, a render
            # cancelled by hand, or this model past its own work threshold --
            # the control render that follows is what tells them apart.
            rec.update(ok=False, why=f"empty response (HTTP {resp.status_code})")
        else:
            b64 = images[0]
            if "," in b64 and b64.strip().startswith("data:"):
                b64 = b64.split(",", 1)[1]
            data = base64.b64decode(b64)
            p = keep(self.out / f"{name}.png", data)
            img = Image.open(io.BytesIO(data)).convert("RGB")
            st = ImageStat.Stat(img)
            mean = sum(st.mean) / 3
            # not for a control: it asks for "a red square", and a model that
            # draws exactly that -- Z-Image Turbo, a solid red frame -- has
            # proved the session renders, which is all a control is for
            flat = max(st.stddev) < 3.0 and measured
            # a safety-filter refusal is a valid grey PNG with a caption set in
            # it; it is not a picture and must never be scored as one
            blocked = (max(st.stddev) < 20 and 100 < mean < 160
                       and max(st.mean) - min(st.mean) < 8)
            rec.update(ok=not (flat or blocked), path=str(p), out_size=list(img.size),
                       stddev=round(max(st.stddev), 2))
            if blocked:
                rec["why"] = "looks like a safety-filter card (flat mid-grey)"
            elif flat:
                rec["why"] = "flat colour field"
        self.results[name] = rec
        self._save()
        if measured and not rec["ok"] and self.control_model:
            # the one question a failed cell raises: is it the session or the cell?
            fine = self.control(explain=name)
            rec["why"] += ("; a control rendered right after it, so the session is fine "
                           "and this cell is the problem" if fine else
                           "; the control after it failed too: the session is not healthy")
            self._save()
        flag = "" if rec["ok"] else f"  <-- {rec['why']}"
        print(f"  {name:24} {rec['seconds']:7.1f}s{flag}", flush=True)
        return rec

    def control(self, *, explain: str | None = None) -> bool:
        """Is the renderer alive? Cheap, and on a different model on purpose.

        Rendered once per run, before anything is measured, and then only to
        `explain` a measured cell that failed (render() asks): a control before
        every cell made Draw Things swap checkpoints twice a cell for a check
        that tells nothing while cells succeed. Every call site still calls
        this before its cell; after the first it answers from that render.
        Never resumed from results.json -- it is about this session.
        """
        if not self.control_model:
            return True
        if self.alive is not None and explain is None:
            return self.alive
        self.controls += 1
        rec = self.render(f"_control{self.controls}", model=self.control_model,
                          prompt="a red square", width=256, height=256, steps=4,
                          cfg=1.0, seed=7, timeout=600, sampler=self.control_sampler,
                          fresh=True)
        self.alive = rec["ok"]
        if not rec["ok"]:
            print("\n  !! control render failed: the Draw Things session is not "
                  "healthy. Restart it and re-run; finished cells are kept.")
        return rec["ok"]


def score_ladder(out: Path, tag: str, texts: dict[str, str]) -> dict:
    """Subject scale and marker strength per rung, off colour masks.

    Subject is the LARGEST CONNECTED red region, not the whole red mask: park
    benches and autumn leaves pass a red threshold, and they take over the
    frame at exactly the long rungs, so the naive version reported the subject
    GROWING. Marker is the yellow mask's area in parts per million.
    """
    import cv2

    rows = {}
    for level in texts:
        cells = []
        for seed in (4242, 1111):
            p = out / f"{tag}_len{level}_{seed}.png"
            if not p.exists():
                continue
            a = np.asarray(Image.open(p).convert("RGB")).astype(int)
            r, g, b = a[..., 0], a[..., 1], a[..., 2]
            red = ((r > 110) & (r - g > 70) & (r - b > 70)).astype(np.uint8)
            yellow = (r > 150) & (g > 130) & (b < 120) & (r - b > 70) & (abs(r - g) < 60)
            h, w = red.shape
            n, _, stats, _ = cv2.connectedComponentsWithStats(red, connectivity=8)
            subj = 0.0
            if n > 1:
                big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
                subj = stats[big, cv2.CC_STAT_HEIGHT] / h
            cells.append((round(subj * 100), round(yellow.sum() / (h * w) * 1e6)))
        if cells:
            rows[level] = {"chars": len(texts[level]), "cells": cells}
    return rows


def budget_from(rows: dict) -> tuple[int, int, str]:
    """prompt_budget from the curves, plus the sentence explaining it.

    `full` is where the SUBJECT collapses, not where the marker dies: a long
    prompt does not lose its tail so much as render everything in it faintly,
    and the subject shrinking is the cost users actually feel. It is set to the
    last rung before the fall, rounded down -- deliberately below the last
    length that still worked.
    """
    levels = [k for k in LEVELS if k in rows]
    if len(levels) < 3:
        return 240, 600, "too few rungs scored; conservative default kept"
    subj = {k: sum(c[0] for c in rows[k]["cells"]) / len(rows[k]["cells"]) for k in levels}
    chars = {k: rows[k]["chars"] for k in levels}
    # ignore L0: at the shortest prompt a model often chooses a close portrait,
    # which inflates the subject share for reasons unrelated to length
    ranked = levels[1:]
    peak = max(subj[k] for k in ranked)
    last_good = ranked[0]
    # 0.85, and do not loosen it. A model whose subject declines steadily has
    # already begun collapsing at the first rung that drops; a slacker
    # threshold reads that decline as a plateau and returns a budget half
    # again too large. Only a model that genuinely sits flat for a rung should
    # carry its budget past it. Validated against Z-Image (260) and Qwen
    # (400), both scored by hand first.
    for k in ranked:
        if subj[k] >= peak * 0.85:
            last_good = k
        else:
            break
    full = int(chars[last_good] // 20 * 20)
    marker = {k: sum(c[1] for c in rows[k]["cells"]) / len(rows[k]["cells"]) for k in levels}
    top = max(marker.values()) or 1
    survives = [k for k in levels if marker[k] > top * 0.1]
    # fading is rounded harder than full: it comes off the marker, which is a
    # noisier instrument, and the shipped values are round numbers
    fading = int(chars[survives[-1]] // 100 * 100) if survives else full
    why = (f"subject holds to {chars[last_good]} chars then falls; marker still "
           f"present at {chars[survives[-1]] if survives else '?'}")
    return full, fading, why


def architecture(dt_filename: str) -> dict:
    """The checkpoint's architecture signature, free, off Draw Things' own records.

    Draw Things writes one of these per custom import into
    Models/custom.json -- base version, both text encoders, the autoencoder --
    and that is enough to say whether two checkpoints are the same machine
    under different weights. A model DT ships itself is not in custom.json, so
    the catalogue's `version`/`text_encoder` fields stand in.

    This is what makes --like checkable rather than a promise: the two SDXL
    merges it was written for report the identical signature, and a v-prediction
    checkpoint or a Cosmos 2B one does not.
    """
    from app.registry import _stem

    want = _stem(dt_filename)
    path = (Path.home() / "Library/Containers/com.liuliu.draw-things/Data"
            / "Documents/Models/custom.json")
    try:
        for entry in json.loads(path.read_text()):
            if _stem(entry.get("file", "")) == want:
                return {
                    "version": entry.get("version"),
                    "modifier": entry.get("modifier"),
                    "autoencoder": entry.get("autoencoder"),
                    # the per-model encoder FILES differ by name (they are
                    # converted alongside the weights); their shape does not
                    "clip": bool(entry.get("clip_encoder")),
                    "text": bool(entry.get("text_encoder")),
                    "source": "custom.json",
                }
    except OSError:
        pass
    cat = catalogue(dt_filename)
    if cat:
        return {"version": cat.get("version"), "modifier": None,
                "autoencoder": cat.get("autoencoder"),
                "clip": True, "text": bool(cat.get("text_encoder")),
                "source": "catalogue"}
    return {}


def crop_sheet(cells: list[tuple[str, Path]], out: Path, crop: int = 512,
               whole: bool = False) -> Path | None:
    """A contact sheet, cropped 1:1 by default or fitted whole with `whole`.

    The two questions these sheets answer need opposite views, and using one
    for both is a mistake that has now been made here once:

    `whole=False` centre-crops at NATIVE pixels, which is the only fair way to
    read an artefact. Laplacian variance cannot be compared across sizes and
    the chroma-noise metric was dropped, so the verdict is an eye on a 1:1 crop
    (MODEL-MEASUREMENTS.md) -- downscaling to fit would erase the very speckle
    being looked for.

    `whole=True` fits the entire frame instead. A duplicated subject is a
    composition fact about the WHOLE image, and a 512 centre crop of a 1536
    render is a third of the frame: the first version of this sheet cropped the
    final-size cells to a torso and could not answer the one question it
    existed to ask.
    """
    from PIL import ImageDraw

    have = [(label, p) for label, p in cells if p and Path(p).exists()]
    if not have:
        return None
    cols = min(4, len(have))
    rows = (len(have) + cols - 1) // cols
    bar = 22
    sheet = Image.new("RGB", (cols * crop, rows * (crop + bar)), "black")
    draw = ImageDraw.Draw(sheet)
    for i, (label, p) in enumerate(have):
        img = Image.open(p).convert("RGB")
        w, h = img.size
        if whole:
            tile = img.resize((crop, crop), Image.LANCZOS) if w == h else (
                img.copy().resize((crop, round(crop * h / w)), Image.LANCZOS))
        else:
            side = min(crop, w, h)
            left, top = (w - side) // 2, (h - side) // 2
            tile = img.crop((left, top, left + side, top + side))
            if tile.size != (crop, crop):
                tile = tile.resize((crop, crop), Image.LANCZOS)
        x, y = (i % cols) * crop, (i // cols) * (crop + bar)
        sheet.paste(tile, (x, y + bar))
        draw.text((x + 4, y + 5), label, fill="white")
    sheet.save(out)
    return out


def cfg_candidates(card_cfg: float) -> list[float]:
    """The cfgs to sweep: the card's value and about a third lower.

    A third lower is what separates CFG burn from a bad sampler -- both look
    like speckle, and only the cfg column moving tells them apart
    (MODEL-MEASUREMENTS.md). Rounded to halves because that is the resolution
    anything here is ever set at.

    Except at cfg 1, which is not swept at all: that is a distilled few-step
    recipe with no classifier-free guidance to burn, there is nothing below it
    to sweep to, and this file's standing advice is to leave such a recipe
    alone -- swapping the sampler on bas-anima produces pure noise rather than
    a degradation, and dropping its cfg would be the same class of mistake.
    """
    if card_cfg <= 1.0:
        return [card_cfg]
    return sorted({round(card_cfg * 2) / 2, round(card_cfg * 2 / 3 * 2) / 2},
                  reverse=True)


def confirm(r: "Renderer", out: Path, found: str, key: str, sib_key: str,
            sib, cand_samplers: list, cand_cfgs: list[float], steps: int) -> int:
    """Render only what --like cannot inherit: the sampler and the cfg.

    The grid is small on purpose, and its shape comes straight from
    MODEL-MEASUREMENTS.md: two cfgs (the card's value and about a third lower,
    which is what separates CFG burn from a bad sampler) by two samplers (the
    card's and the measured sibling's), two seeds, at DRAFT size. Draft size
    because the artefact sweep that produced the finding was run "at each
    model's own configured draft settings, one seed, face cropped at 1:1" --
    the speckle is not a property of the output resolution, and 768 costs ~25s
    against ~105s at 1536.

    Plus the one diagnostic that separates the two causes, which is worth more
    than any of the grid cells on its own: the same ancestral cell at half
    again the steps. Under-convergence gets BETTER with more steps; leftover
    ancestral noise gets WORSE. That render is what stops an evening being
    spent raising the step count, which is the obvious move and the wrong one.
    """
    draft_size = sib.draft.size
    print(f"\n-- 1. smoke test")
    if not r.render("smoke", model=found, prompt=ARTEFACT, width=256, height=256,
                    steps=steps, cfg=cand_cfgs[0], sampler=cand_samplers[0],
                    clip_skip=sib.clip_skip)["ok"]:
        print("   does not render at 256px; nothing else is worth running.")
        return 1

    print(f"\n-- 2. sampler x cfg at the inherited draft size ({draft_size}px, "
          f"{steps} steps, 2 seeds)")
    cells = []
    for sampler in cand_samplers:
        for cfg in cand_cfgs:
            for seed in (1111, 2222):
                tag = f"{sampler.replace('+','').replace(' ','_')}_cfg{cfg}_{seed}"
                r.control()
                rec = r.render(tag, model=found, prompt=ARTEFACT, width=draft_size,
                               height=draft_size, steps=steps, cfg=cfg, seed=seed,
                               sampler=sampler, clip_skip=sib.clip_skip)
                cells.append((f"{sampler} cfg{cfg} s{seed}", Path(rec["path"])
                              if rec.get("path") else None))

    print("\n-- 3. step-direction diagnostic (ancestral noise gets worse, "
          "under-convergence gets better)")
    high = max(6, int(steps * 1.5))
    diag = []
    for sampler in cand_samplers:
        rec = r.render(f"steps{high}_{sampler.replace('+','').replace(' ','_')}",
                       model=found, prompt=ARTEFACT, width=draft_size,
                       height=draft_size, steps=high, cfg=cand_cfgs[0], seed=1111,
                       sampler=sampler, clip_skip=sib.clip_skip)
        diag.append((f"{sampler} cfg{cand_cfgs[0]} {high}st", Path(rec["path"])
                     if rec.get("path") else None))

    # NOTE the step count: these use the SIBLING's final.steps, not a measured
    # one, and on an Illustrious finetune that is exactly what exposed a duplicated subject
    # at 1536/24 that 1536/20 does not show. So this cell tests one point in a
    # size x steps plane, and a duplicate seen here is a reason to sweep both
    # axes rather than to lower the size (MODEL-MEASUREMENTS.md).
    print(f"\n-- 4. inherited final size ({sib.final.size}px at {sib.final.steps} "
          f"steps, the sibling's), for duplicated subjects only")
    finals = []
    for seed in (1111, 2222):
        r.control()
        rec = r.render(f"final{sib.final.size}_{seed}", model=found, prompt=ARTEFACT,
                       width=sib.final.size, height=sib.final.size,
                       steps=sib.final.steps, cfg=cand_cfgs[0], seed=seed,
                       sampler=cand_samplers[0], clip_skip=sib.clip_skip)
        finals.append((f"final {sib.final.size} s{seed}", Path(rec["path"])
                       if rec.get("path") else None))

    grid = crop_sheet(cells, out / "sheet_sampler_cfg.png")
    step = crop_sheet(diag, out / "sheet_step_direction.png")
    # whole frames: a duplicated subject cannot be seen in a centre crop
    fin = crop_sheet(finals, out / "sheet_final_size.png", whole=True)

    rendered = [v for k, v in r.results.items()
                if not k.startswith("_control") and v.get("ok")]
    spent = sum(v["seconds"] for v in r.results.values() if v.get("seconds"))
    print("\n" + "=" * 72)
    print(f"{len(rendered)} renders, {spent / 60:.1f} min of Draw Things time.")
    print(f"""
YOUR EYES, ON THREE SHEETS -- 1:1 centre crops, in {out}
  * {grid and grid.name}
      Rainbow speckle on skin, hair edges and the armour; a waxy face with an
      unformed mouth. Pick the sampler and cfg with none of it. If a cfg column
      is speckled at the high value and clean at the low one, that is CFG burn;
      if BOTH cfgs are speckled under one sampler and clean under the other,
      the sampler is the cause. Judge across the two seeds, never one pair.
  * {step and step.name}
      Against the same cells at {steps} steps in the sheet above: worse at
      {high} means leftover ancestral noise, better or equal means the sampler
      is fine. These are NOT paired comparisons -- a different step count is a
      different sampling trajectory and so a different picture -- so read them
      for the presence of the artefact, never as a before-and-after.
  * {fin and fin.name}
      One subject, or two? That is the only question this sheet answers, and
      it is the one thing that has been got wrong twice here. If a second
      figure appears, do NOT simply lower the size: these cells are one point
      in a size x steps plane and the step count here is the sibling's, not a
      measured one. Sweep both axes -- one Illustrious finetune here
      duplicates at 1536/24 on every seed and at 1536/20 on none.

WHAT THIS RUN DELIBERATELY DID NOT MEASURE, AND WHY THAT IS DEFENSIBLE
  prompt_budget   inherited as the conservative 240/600 default, which is what
                  {sib_key} ships and labels unmeasured. It under-promises: if
                  this checkpoint is roomier the cost is a shorter prompt than
                  necessary, never a broken render. 10 renders to close.
  aspect_size     inherited from {sib_key} as a guess, NOT rendered here.
                  Confirm it on every ratio before shipping:
                  scripts/sweep_aspect.py --models {key}
  tiled_diffusion / max_upscale_size
                  absent, as on every SDXL entry here: they reach 2048 unaided.
  draft.steps     inherited from {sib_key}. This is the weakest of the
                  inherited numbers -- DRAFT-STEPS.md has four of five
                  checkpoints on one sampler disagreeing about it -- and
                  scripts/sweep_draft_size.py is what settles it, judged on the
                  Faithful rather than on the draft.
""")
    return 0


def measuring_registry():
    """The registry with every model as shipped: no optional LoRA ticked.

    What --like inherits and what the sheets render at is the sibling's own
    recipe. With the ticks at their defaults, lustrousmix-anima's draft and
    final are its Turbo LoRA's (1024 and 8 steps, cfg 1), and a confirm run
    printed that as the geometry to copy and rendered its sheets at it.
    """
    return load_registry(lora_options=False)


def sides(budget: int, w: int, h: int) -> tuple[int, int]:
    s = math.sqrt(budget * budget / (w * h))
    return max(64, round(w * s / 64) * 64), max(64, round(h * s / 64) * 64)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="Draw Things model filename")
    ap.add_argument("--key", required=True, help="key to use in models.yaml")
    ap.add_argument("--label", help="UI label (default: the catalogue name)")
    ap.add_argument("--level", choices=("quick", "full", "confirm"), default="full",
                    help="quick halves the ladder and the size grid; confirm "
                         "renders only sampler/cfg and needs --like")
    ap.add_argument("--final", type=int, default=1024, help="candidate final size")
    ap.add_argument("--tags", action="store_true",
                    help="Danbooru/booru checkpoint: prompt_style tags")
    ap.add_argument("--like", help="registry key of a MEASURED sibling of the same "
                                   "architecture to inherit geometry from")
    ap.add_argument("--card-sampler", help="sampler the model card names "
                                           "(a Civitai import is not in the catalogue)")
    ap.add_argument("--card-cfg", type=float, help="cfg the model card names")
    ap.add_argument("--card-steps", type=int, help="steps the model card names")
    ap.add_argument("--opener", help="the quality tags the card puts in front of every "
                                     "prompt (default: the model's entry, or the tag "
                                     "default with --tags)")
    ap.add_argument("--negative", help="the negative prompt the card gives (default: as "
                                       "for --opener)")
    args = ap.parse_args()

    out = (RESEARCH if args.level == "confirm" else RUNS) / args.key
    out.mkdir(parents=True, exist_ok=True)

    # Two batches sharing Draw Things distort every timing in both, and the
    # numbers look perfectly plausible afterwards, so nothing flags it -- one
    # whole round of measurements was invalidated that way
    # (MODEL-MEASUREMENTS.md). Both docs have claimed for a while that this
    # lock exists; it did not. It does now.
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    lock = open(DATA_DIR / "add_model.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        # Do NOT tell anyone to delete the lock file. flock lives on the open
        # descriptor and is released when the process exits, so the file is
        # ALWAYS left behind by a normal run and its presence means nothing.
        # Saying "remove it if nothing is running" invites deleting a file that
        # is not the lock, which fixes nothing and teaches the wrong model of
        # how this works.
        print("FAIL: another measurement batch holds the lock. Two batches "
              "sharing Draw Things distort each other's timings, and the result "
              "looks plausible afterwards -- so this refuses rather than warns. "
              "Wait for the other batch to finish; a run is resumable, so "
              "nothing you have already measured is lost. (The lock file itself "
              "is left behind by every run and is not what blocks you -- the "
              "lock is held on the open file descriptor and dies with the "
              "process, so there is nothing to clean up.)")
        return 1

    registry = measuring_registry()
    # without this every spec's `installed` is None and registry.upscaler is
    # unset, so the control model could be one that is not on disk and the
    # enlarge step would fall back to a hard-coded upscaler name
    registry.resolve_installed()
    files = installed_files(registry.draw_things.models_dirs)
    found = match_filename(args.model, files)
    if not found:
        print(f"FAIL: {args.model} is not installed in {registry.draw_things.models_dirs}")
        return 1

    cat = catalogue(found)
    sampler = cat.get("sampler")
    steps = cat.get("steps") or 20
    cfg = cat.get("cfg") if cat.get("cfg") is not None else 1.0
    print(f"== {found}")
    if cat:
        print(f"   Draw Things recommends: sampler={sampler} steps={steps} cfg={cfg} "
              f"shift={cat.get('shift')}")
        if cat.get("note"):
            print(f"   note: {cat['note'][:300]}")
    else:
        print("   not in the Draw Things catalogue; using generic defaults. Check its "
              "model card for sampler/steps/cfg before trusting this entry.")

    # a cheap, known-good model to bracket expensive cells with
    ctl = control_model(registry.models, args.key)
    control, control_sampler = (ctl.dt_filename, ctl.sampler) if ctl else (None, None)
    if control is None:
        print("   no other model installed to bracket cells with: running WITHOUT "
              "control renders (docs/decisions/FALSE-SUCCESS.md says why that matters)")
    opener, negative = prompt_kit(tags=args.tags, opener=args.opener, negative=args.negative,
                                  spec=registry.models.get(args.key))
    if opener or negative:
        print(f"   every cell opens with {opener!r}" + (f", negative {negative!r}" if negative else ""))
    else:
        print("   every cell is bare prose with no negative: right for a prose model, "
              "and a starved one for a Danbooru model -- pass --tags (or --opener/--negative)")
    r = Renderer(out, control, opener=opener, negative=negative, control_sampler=control_sampler)

    if args.level == "confirm":
        if not args.like:
            print("FAIL: --level confirm needs --like <sibling key>. It inherits "
                  "that model's geometry and renders only the sampler and cfg, "
                  "which are the two fields that must never be inherited.")
            return 1
        sib = registry.models.get(args.like)
        if sib is None:
            print(f"FAIL: --like {args.like} is not a key in config/models.yaml.")
            return 1
        if args.like not in measured_sources():
            print(f"FAIL: {args.like} is not a measured source (MEASURED_SOURCES, or"
                  f" a private entry marked {MEASURED_MARKER}), so its own "
                  f"sampler and cfg were never established by rendering it. An "
                  f"unmeasured entry is not a source -- that is how three "
                  f"checkpoints here got the wrong sampler "
                  f"(docs/decisions/MODEL-MEASUREMENTS.md). Pick a measured "
                  f"sibling, or run a full batch.")
            return 1

        mine, theirs = architecture(found), architecture(sib.dt_filename)
        if not mine or not theirs:
            print(f"FAIL: cannot read an architecture signature for "
                  f"{'this model' if not mine else args.like}, so --like cannot "
                  f"be checked. Run a full batch instead.")
            return 1
        if mine != {**theirs, "source": mine.get("source")}:
            print(f"FAIL: architecture mismatch, so nothing may be inherited.\n"
                  f"  {args.key:>14}: {mine}\n  {args.like:>14}: {theirs}\n"
                  f"A different base version, modifier (v-prediction!) or "
                  f"autoencoder is a different machine.")
            return 1
        print(f"\n== inheriting from {args.like} ({sib.label})")
        print(f"   architecture matches, per {mine['source']}: "
              f"{mine['version']}, modifier {mine['modifier']}, "
              f"{mine['autoencoder']}")
        for field in INHERITED:
            val = getattr(sib, field, None)
            if val is None or val is False:
                print(f"   {field:18} (absent -- and absent is a meaning here)")
            else:
                shown = val.model_dump(exclude_none=True, exclude_defaults=True) \
                    if hasattr(val, "model_dump") else val
                print(f"   {field:18} {shown}")
        print(f"   NOT inherited: sampler, cfg, draft -- these are measured below.")

        # The card's values against the sibling's. A Civitai import has no
        # catalogue entry, so the card is the only other source there is, and
        # MODEL-MEASUREMENTS.md says to spend the renders confirming it rather
        # than inheriting from the model next door.
        card_cfg = args.card_cfg if args.card_cfg is not None else sib.cfg
        steps = args.card_steps or sib.draft.steps
        samplers = list(dict.fromkeys(
            [x for x in (args.card_sampler, sib.sampler) if x]))
        cfgs = cfg_candidates(card_cfg)
        if len(cfgs) == 1:
            print("   cfg 1: distilled recipe, not swept (nothing below it, and "
                  "no guidance to burn)")
        print(f"   grid: samplers {samplers} x cfgs {cfgs} at "
              f"{sib.draft.size}px / {steps} steps")
        return confirm(r, out, found, args.key, args.like, sib, samplers, cfgs, steps)

    common = dict(model=found, sampler=sampler, cfg=cfg,
                  clip_skip=cat.get("clip_skip", 1))

    print("\n-- 1. smoke test")
    if not r.render("smoke", prompt=ladder()["L0"], width=256, height=256,
                    steps=steps, **common)["ok"]:
        print("   the model does not render at 256px; nothing else is worth running.")
        return 1

    print("\n-- 2. step sweep (does the recommended count earn itself?)")
    sweep = sorted({max(4, steps // 4), max(4, steps // 2), steps})
    for s in sweep:
        r.control()
        r.render(f"steps{s}", prompt=ladder()["L0"], width=args.final,
                 height=args.final, steps=s, **common)

    print("\n-- 3. draft candidates")
    for size in (384, 512, 768):
        r.render(f"draft{size}", prompt=ladder()["L0"], width=size, height=size,
                 steps=steps, **common)

    print("\n-- 4. size grid (look at these for duplicated subjects)")
    seeds = (1111, 2222, 4242) if args.level == "full" else (1111, 4242)
    for size in (args.final, int(args.final * 1.5) // 64 * 64):
        for seed in seeds:
            r.control()
            r.render(f"grid{size}_{seed}", prompt=ladder()["L0"], width=size,
                     height=size, steps=steps, seed=seed, **common)

    print("\n-- 5. prompt-length ladder")
    texts = ladder()
    ladder_seeds = (4242, 1111) if args.level == "full" else (4242,)
    for level, text in texts.items():
        for seed in ladder_seeds:
            r.control()
            rec = r.render(f"{args.key}_len{level}_{seed}", prompt=text,
                           width=args.final, height=args.final, steps=steps,
                           seed=seed, **common)
            if not rec["ok"] and "safety" in rec.get("why", ""):
                print("   this scene is blocked by the model's safety filter. Re-run "
                      "with a scene it accepts; see docs/decisions/FALSE-SUCCESS.md.")

    print("\n-- 6. shapes (the ladder's L0 cells are the square controls)")
    pw, ph = sides(args.final, 3, 4)
    lw, lh = sides(args.final, 16, 9)
    for seed in ladder_seeds:
        r.control()
        r.render(f"portrait_{seed}", prompt=texts["L0"], width=pw, height=ph,
                 steps=steps, seed=seed, **common)
    r.render("wide_4242", prompt=texts["L0"], width=lw, height=lh, steps=steps, **common)

    print("\n-- 7. upscale to 2048")
    src = out / f"{args.key}_lenL0_4242.png"
    tiled_needed = None
    upscale_seconds = corr = None
    if src.exists():
        up_steps = max(4, steps // 2)
        enl = r.render("enlarge2048", prompt="", width=args.final, height=args.final,
                       steps=2, init=str(src), strength=0.0,
                       upscaler=registry.upscaler or "4x_ultrasharp_f16.ckpt",
                       upscaler_scale=2, **common)
        if enl["ok"]:
            # BOTH are always rendered, and the winner is decided on
            # correlation with the source rather than on whether the plain one
            # merely came back. The flat colour field this flag exists for is
            # not the only failure it fixes: an untiled 2048 can come back
            # finely speckled over every smooth area, which passes any "did it
            # return something" test. Correlation separates that (0.984
            # untiled against 0.995 tiled on Z-Image); flatness does not.
            def corr_with_source(rec):
                if not rec["ok"]:
                    return None
                a = np.asarray(Image.open(src).convert("RGB").resize((2048, 2048)),
                               dtype=float).ravel()
                b = np.asarray(Image.open(rec["path"]).convert("RGB"), dtype=float).ravel()
                return float(np.corrcoef(a, b)[0, 1])

            plain = r.render("upscale2048_plain", prompt=texts["L0"], width=2048,
                             height=2048, steps=up_steps, init=enl["path"],
                             strength=0.3, **common)
            r.control()
            tiled = r.render("upscale2048_tiled", prompt=texts["L0"], width=2048,
                             height=2048, steps=up_steps, init=enl["path"],
                             strength=0.3, tiled_diffusion=True, **common)
            cp, ct = corr_with_source(plain), corr_with_source(tiled)
            if ct is not None and (cp is None or ct > cp + 0.003):
                tiled_needed, best, corr = True, tiled, ct
            elif cp is not None:
                tiled_needed, best, corr = False, plain, cp
            else:
                best = None
            if best:
                upscale_seconds = int(best["seconds"])
                print(f"  untiled corr {cp}, tiled corr {ct} -> "
                      f"tiled_diffusion: {str(bool(tiled_needed)).lower()}")

    rows = score_ladder(out, args.key, texts)
    full, fading, why = budget_from(rows)

    print("\n" + "=" * 72)
    print("LADDER (subject % of frame height / marker parts-per-million)")
    if rows and all(len(r["cells"]) < 2 for r in rows.values()):
        print("  (one seed only -- `full` is the number most worth a second "
              "seed; re-run without --level quick if it matters)")
    for level, row in rows.items():
        cells = "  ".join(f"{s:>3}/{m:>6}" for s, m in row["cells"])
        print(f"  {level}  {row['chars']:>4} chars   {cells}")
    print(f"  -> prompt_budget {{ full: {full}, fading: {fading} }}  ({why})")

    timings = {k: v["seconds"] for k, v in r.results.items() if v.get("ok")}
    print("\nTIMINGS")
    for k in sorted(timings, key=lambda k: timings[k]):
        if not k.startswith("_control"):
            print(f"  {k:24} {timings[k]:7.1f}s")

    label = args.label or cat.get("name") or args.key
    entry = f"""
  {args.key}:
    label: "{label}"
    dt_filename: "{found}"
    # Draw Things' own recommendation: {sampler} at {steps} steps, cfg {cfg}.
    sampler: {json.dumps(sampler)}
    cfg: {cfg}
    uses_negative_prompt: {str(cfg > 1).lower()}
    prompt_style: {"tags" if args.tags else "prose"}
    prompt_guidance: |
      TODO: write the dialect for this model.
    # Measured by scripts/add_model.py: {why}
    prompt_budget: {{ full: {full}, fading: {fading} }}
    # Best guess, the final size -- 1024 instead for SDXL, whose training
    # buckets are 1024-area. Confirm on every ratio with
    # scripts/sweep_aspect.py --models {args.key}; lower it if one duplicates.
    aspect_size: {args.final}
    draft: {{ size: TODO, steps: {steps} }}
    final: {{ size: {args.final}, steps: {steps}, strength: 0.5 }}
    upscale: {{ size: 2048, steps: {max(4, steps // 2)}, strength: 0.3 }}"""
    if tiled_needed:
        entry += "\n    tiled_diffusion: true"
    if upscale_seconds:
        entry += f"\n    upscale_seconds: {upscale_seconds}"
    entry += "\n    # nothing above 2048 was measured; this is a floor, not a ceiling"
    entry += "\n    max_upscale_size: 2048"

    print("\nDRAFT ENTRY (fill the TODOs after looking at the sheets)")
    print(entry)
    if corr is not None:
        print(f"\n  2048 upscale correlates {corr:.4f} with its source"
              f"{' (needed tiled_diffusion)' if tiled_needed else ''}")

    print(f"""
STILL NEEDS YOUR EYES -- images in {out}
  * grid*.png      duplicated subjects at each size -> which final size is safe
  * draft*.png     does the small one look like the big one? -> draft size
  * portrait_*, wide_4242 vs the {args.key}_lenL0_* squares -> a first look
    at aspect_size; scripts/sweep_aspect.py then covers every ratio
  * steps*.png     does the recommended step count beat half of it?

THEN, OR THE MODEL WILL NOT APPEAR
  1. paste the entry above into config/models.yaml
  2. RESTART THE SERVER -- it reads that file once, at startup

  A server left running keeps serving the model list it loaded when it
  started. Until it is restarted the new chip never appears, and a request
  naming {args.key} is rejected with 400 "unknown model key" before anything
  is queued -- so Draw Things sits idle and /api/health stays all-green.
  Renaming or removing a key bites the same way from the other side: the
  running server still offers the old chip, whose file it can no longer find,
  and rejects it with 400 "not installed in Draw Things".
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
