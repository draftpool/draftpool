"""Sweep the final stage's step count on one or more models, keeping every frame.

    uv run python scripts/sweep_final_steps.py --models perfectdeliberate
    uv run python scripts/sweep_final_steps.py --models all --seeds 1111,4242

Why this exists rather than the numbers already in config/models.yaml: eight of
the fourteen models are user-imported checkpoints that Draw Things has no
recommended step count for at all, and the calls that were made for them were
made on a face crop of a single cell. A crop cannot carry "A is better than B"
when A and B are not the same picture -- and they are not: changing the step
count moves the composition on a distilled model, which is visible in the one
surviving run under data/add_model/z-image-turbo (steps4.png against steps8.png
are different parks).

So this writes **whole frames** and a side-by-side sheet per row, and decides
nothing. It is the renders; the eyes are the measurement.

Everything is resumable: a cell whose PNG is already on disk is skipped, so an
interrupted run continues where it stopped.

**Timings are only valid on an idle Draw Things.** It renders serially, so a
sweep queued behind someone using the app measures their queue, not the cell.
The tell is numbers that do not rise with the work: a 256px draft reading slower
than a 384px one is wait time. The pixels are unaffected; only the seconds are.

**Probe one cell before you render a grid.** That applies to any new variable --
a LoRA, a cfg, a sampler -- not to the step ladders themselves, where the
comparison across rungs is the point. Thirty cells went into the first
accelerator-LoRA run and two would have shown the same answer; the probe that
then settled three configurations took a minute (`--models` one key, `--scenes`
one scene, `--steps` one rung).

Output goes to `research/final-steps/`, deliberately NOT under `data/`. The
renders behind the earlier step and size calls were written into
`data/add_model/<key>/` and are gone -- `data/` is the directory SETUP.md tells
people to move wholesale when they relocate their install, the cleanup tool
surveys it, and the harness overwrites a model's folder on the next run. The
PNGs are hundreds of megabytes and stay out of git; `results.json` is text and
is committed, so the numbers survive even if the frames do not.

The scenes are deliberately busy and full of small things -- lettering, lace,
jewellery, chalk on a board -- because a thin schedule loses fine detail first
and an empty scene would hide exactly what is being measured.

Control renders: a Draw Things session that has stopped rendering answers 200
with no image, and a whole batch of measurements was once taken in that state
and believed (docs/decisions/FALSE-SUCCESS.md). Every cell here is preceded by
a 256px render on a *different* model, so a failure can be blamed on the cell
rather than on the session.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# a script in scripts/ does not get the project root on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import ROOT
from app.dt_client import DrawThingsClient
from app.registry import load_registry

TOPIC = "final-steps"
OUT = ROOT / "research" / TOPIC   # archived frames; a run stages in runs/ (research_frames.py)

# The step ladders. Each is (below, shipped, above) for that model, so the
# shipped value is always in the middle with something either side of it to be
# judged against. Nothing here is a prediction -- they are the cells to look at.
LADDERS = {
    "flux2-klein": [4, 16, 32],          # 4 is Draw Things' own recommendation
    "flux1-schnell": [4, 6, 12],         # 4 is Draw Things' own recommendation
    "z-image-turbo": [8, 16, 24],        # ships at DT's 8; the question is above it
    # 12 added: 8 shows visible noise at 1:1 and 16 does not, so the floor is
    # somewhere between them and the shipped 16 wants a neighbour below it
    "krea2-turbo": [4, 8, 12, 16],
    "lustrousmix-anima": [16, 24, 40],
    # 24 added after the first pass: 8 was judged bad, 16 acceptable and 32
    # better, so the shipped value sits between two cells nobody had rendered
    "bas-anima": [8, 16, 24, 32],
    "cyberrealistic-anima": [30, 44, 60],
    "wai-illustrious": [20, 28, 40],     # the trio whose 20/28/40 call was crop-only
    # 24 added after the first pass, same reason as bas-anima: the verdict
    # landed between two rendered cells
    "cyberrealistic-illustrious": [20, 24, 28, 40],
    "oneobsession": [20, 28, 40],
    "perfectdeliberate": [20, 30, 45],
    "juggernaut-ragnarok": [20, 32, 48],
    # DT recommends 30 and this model returns an EMPTY image well before that
    # (see the table in config/models.yaml). 16 is in to confirm the threshold
    # on the current install rather than to be chosen.
    "qwen-image": [8, 12, 16],
}

# Two scenes, both crowded with small detail, in the two prompt dialects. Two
# rather than one because a single scene has hidden a break before: the sweep
# in DRAFT-STEPS.md found wai-illustrious clean on its market scene at 10 steps
# and broken on its portrait at the same cell. That is also the reason the
# default is two scenes at ONE seed rather than one scene at two -- of the two
# axes, this is the one with a break on the record.
SCENES = {
    "market": {
        "prose": (
            "A narrow night market alley in the rain, seen from the middle of the "
            "lane: paper lanterns strung overhead with painted characters on them, "
            "a chalkboard menu propped against a stall, trays of small silver fish "
            "on crushed ice, steam rising from a noodle pot, coins and folded notes "
            "in an open cash tin, a vendor in a canvas apron leaning on the counter "
            "with rolled sleeves and a wristwatch, wet cobbles reflecting red and "
            "green neon, a cat sitting under the table, telephone wires against a "
            "dark sky. Photographic, 35mm, shallow depth of field."
        ),
        "tags": (
            "1girl, solo, night market, food stall, paper lantern, chinese text, "
            "chalkboard, menu, fish, ice, steam, noodles, coins, apron, "
            "rolled sleeves, wristwatch, wet street, reflection, neon lights, "
            "cobblestone, cat, power lines, night, rain, depth of field, "
            "detailed background"
        ),
    },
    "portrait": {
        "prose": (
            "Close portrait of a woman at a workbench, three-quarter view, lit from "
            "a window on the left: a lace collar with an open weave, a small pearl "
            "earring, freckles across the nose, loose strands of hair catching the "
            "light, a thin chain necklace, an open pocket watch in her hand with "
            "roman numerals on the dial, brass tools and screws scattered on the "
            "bench, a magnifying loupe pushed up on her forehead. Photographic, "
            "85mm, soft window light."
        ),
        "tags": (
            "1girl, solo, upper body, portrait, lace trim, collared shirt, "
            "pearl earrings, freckles, jewelry, necklace, pocket watch, "
            "roman numeral, holding, tools, screws, workbench, magnifier, "
            "window light, looking at viewer, detailed face, detailed clothes"
        ),
    },
}


def analyse(png: bytes, want: tuple[int, int]) -> dict:
    """Whether this cell is an image at all, and a couple of cheap numbers.

    Not a quality score -- Laplacian variance rewards ringing and posterised
    colour breakdown, and it has already recommended a broken cell once
    (docs/decisions/DRAFT-STEPS.md). It is here to flag the blur end and the
    two failure shapes that report success: a flat field and a grey safety card.
    """
    import io

    img = Image.open(io.BytesIO(png)).convert("RGB")
    a = np.asarray(img, float)
    std = a.std(axis=(0, 1))
    grey = abs(a.mean() - 128) < 12 and float(std.max()) < 25
    lap = float(
        np.abs(np.diff(a.mean(axis=2), n=2, axis=0)).var()
        + np.abs(np.diff(a.mean(axis=2), n=2, axis=1)).var()
    )
    return {
        # NOT "size": the row already carries that as an int, and a dict spread
        # after it would replace the number with a [w, h] pair -- which is
        # exactly what shipped, and made every scale in viewer.html NaN
        "out_size": list(img.size),
        "size_ok": tuple(img.size) == want,
        "stddev": [round(float(v), 1) for v in std],
        "flat": bool(std.max() < 3),
        "maybe_blocked": bool(grey),
        "laplacian": round(lap, 1),
    }


def sheet(paths: list[Path], labels: list[str], out: Path, height: int = 720) -> None:
    """One row, whole frames, equal display height, labelled. No crops."""
    imgs = [Image.open(p).convert("RGB") for p in paths]
    scaled = [im.resize((round(im.width * height / im.height), height), Image.LANCZOS)
              for im in imgs]
    bar = 34
    sheet_img = Image.new("RGB", (sum(i.width for i in scaled), height + bar), (16, 17, 20))
    x = 0
    draw = ImageDraw.Draw(sheet_img)
    for im, label in zip(scaled, labels):
        sheet_img.paste(im, (x, bar))
        draw.text((x + 8, 10), label, fill=(235, 235, 240))
        x += im.width
    sheet_img.save(out)


async def run(keys: list[str], seeds: list[int], scenes: list[str],
              steps_override: list[int] | None = None,
              lora: dict | None = None, cfg: float | None = None,
              tag: str = "", run_dir: Path | None = None) -> Path:
    from scripts.research_frames import archived, keep, start_run
    run_dir = run_dir or start_run(TOPIC)
    # the models as they ship: an accelerator is measured through --lora and
    # a tag, never picked up silently from a default-on tick
    reg = load_registry(lora_options=False)
    dt = DrawThingsClient("http://127.0.0.1:7860")
    if not await dt.is_up():
        raise SystemExit("Draw Things is not answering on 127.0.0.1:7860")

    # a cheap, known-good model that is not the one under test, for the control
    control_key = next(k for k in ("z-image-turbo", "krea2-turbo", "bas-anima")
                       if reg.models[k].available)

    for key in keys:
        spec = reg.models[key]
        if not spec.available:
            print(f"{key}: not installed, skipping")
            continue
        # an explicit --steps replaces the ladder for this run: useful for
        # filling one rung, or for adding a scene without paying for a count
        # that has already been ruled out.
        #
        # NOT named `steps`: the innermost loop below binds that name, so a
        # parameter called `steps` survives the first model as an int and the
        # second model's `len(ladder)` dies. It did, mid-queue, after one model
        # had rendered -- and the symptom was Draw Things sitting idle.
        ladder = list(steps_override) if steps_override else LADDERS[key]
        out = run_dir / key
        out.mkdir(parents=True, exist_ok=True)
        results: list[dict] = []
        log = out / "results.json"
        if log.is_file():
            results = json.loads(log.read_text())
        kept_log = OUT / key / "results.json"
        kept_rows: list[dict] = json.loads(kept_log.read_text()) if kept_log.is_file() else []

        def have(name: str) -> Path | None:
            """This run's frame, or an archived one: a cell is never rendered twice."""
            return (out / name) if (out / name).is_file() else archived(TOPIC, f"{key}/{name}")

        size = spec.final.size
        print(f"\n=== {key}: {size}px, steps {ladder}, "
              f"{len(scenes)}x{len(seeds)}x{len(ladder)} cells ===")

        for scene in scenes:
            for seed in seeds:
                for steps in ladder:
                    # a tagged cell is a different experiment on the same row --
                    # an accelerator LoRA, a different cfg -- and has to sit
                    # beside the plain one rather than overwrite it
                    suffix = f"__{tag}" if tag else ""
                    name = f"{scene}__seed{seed}__steps{steps}{suffix}.png"
                    if have(name):
                        print(f"  {name}: already on disk ({have(name).parent.relative_to(ROOT)})")
                        continue
                    # control first: proves the session renders at all, on a
                    # different model, before spending minutes on this cell
                    ctl = reg.models[control_key]
                    t0 = time.monotonic()
                    ctl_png = await dt.txt2img(
                        prompt="a red cube on a wooden table", negative_prompt="",
                        model_filename=ctl.dt_filename, width=256, height=256,
                        steps=4, cfg=ctl.cfg, seed=7, sampler=ctl.sampler,
                        clip_skip=ctl.clip_skip, loras=[],
                    )
                    ctl_ok = not analyse(ctl_png, (256, 256))["flat"]
                    if not ctl_ok:
                        raise SystemExit(
                            "control render came back flat: Draw Things needs a "
                            "restart, and nothing measured after this is trustworthy"
                        )
                    prompt = SCENES[scene][spec.prompt_style]
                    if spec.quality_opener:
                        prompt = f"{spec.quality_opener}, {prompt}"
                    t0 = time.monotonic()
                    png = await dt.txt2img(
                        prompt=prompt,
                        negative_prompt=spec.default_negative,
                        model_filename=spec.dt_filename,
                        width=size, height=size, steps=steps,
                        cfg=cfg if cfg is not None else (
                            spec.final.cfg if spec.final.cfg is not None else spec.cfg
                        ),
                        seed=seed,
                        sampler=spec.final.sampler or spec.sampler,
                        clip_skip=spec.clip_skip,
                        loras=([lora] if lora
                               else [l.model_dump() for l in spec.final.loras]),
                        tiled_diffusion=spec.tiled_diffusion,
                    )
                    seconds = round(time.monotonic() - t0, 1)
                    keep(out / name, png)
                    row = {"file": name, "model": key, "scene": scene, "seed": seed,
                           "steps": steps, "seconds": seconds, "size": size,
                           "loras": ([lora] if lora
                                     else [l.model_dump() for l in spec.final.loras]),
                           "control_ok": ctl_ok,
                           **({"lora": lora["file"], "lora_weight": lora["weight"],
                               "cfg": cfg, "tag": tag} if lora else {}),
                           **analyse(png, (size, size))}
                    results.append(row)
                    log.write_text(json.dumps(results, indent=1))
                    flag = "" if row["size_ok"] and not row["flat"] else "  <-- LOOK"
                    print(f"  {name}: {seconds}s  lap={row['laplacian']}"
                          f"  std={row['stddev']}{flag}")

                suffix = f"__{tag}" if tag else ""
                paths = [have(f"{scene}__seed{seed}__steps{s}{suffix}.png")
                         for s in ladder]
                if all(paths):
                    secs = {r["steps"]: r["seconds"] for r in (*kept_rows, *results)
                            if r["scene"] == scene and r["seed"] == seed}
                    labels = [f"{key}  {scene}  seed {seed}  ·  {s} steps"
                              f"{'  ' + tag if tag else ''}"
                              f"  ({secs.get(s, '?')}s)" for s in ladder]
                    sheet(paths, labels, out / f"SHEET__{scene}__seed{seed}{suffix}.png")
                    print(f"  -> SHEET__{scene}__seed{seed}{suffix}.png")
    return run_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="perfectdeliberate",
                    help="comma-separated keys, or 'all'")
    # One seed, two scenes. The second SCENE is the axis that has actually
    # caught something -- DRAFT-STEPS.md's wai-illustrious break shows up on the
    # portrait and not on the market at the same cell, and one scene would have
    # shipped it. A second seed doubles the bill for a weaker guarantee: it
    # separates "this cell is broken" from "this seed is unlucky", which matters
    # for a brightness delta and much less for a judgement made by eye on a
    # frame. Pass --seeds 1111,4242 when a cell looks marginal and the question
    # is which of the two it is.
    ap.add_argument("--seeds", default="1111")
    ap.add_argument("--scenes", default="market,portrait")
    ap.add_argument("--steps", default="",
                    help="comma-separated step counts, overriding the model's ladder")
    ap.add_argument("--lora", default="",
                    help="accelerator LoRA as FILE:WEIGHT, e.g. "
                         "dmd2_sdxl_4_step_lora_f16.ckpt:0.6")
    ap.add_argument("--cfg", type=float, default=None,
                    help="cfg override, which an accelerator LoRA usually needs")
    ap.add_argument("--tag", default="",
                    help="filename suffix so a tagged run sits beside the plain "
                         "cells instead of overwriting them (default: the LoRA's "
                         "first name part)")
    ap.add_argument("--run", help="resume a stopped run in this runs/ folder")
    ap.add_argument("--note", help="what this run is for; given, the run is archived "
                                   "into research/ when it completes")
    args = ap.parse_args()

    lora = None
    tag = args.tag
    if args.lora:
        file, _, weight = args.lora.partition(":")
        lora = {"file": file, "weight": float(weight or 1.0)}
        tag = tag or file.split("_")[0]
    keys = list(LADDERS) if args.models == "all" else args.models.split(",")
    unknown = [k for k in keys if k not in LADDERS]
    if unknown:
        raise SystemExit(f"no ladder for {unknown}; add one to LADDERS")
    run_dir = asyncio.run(run(keys, [int(s) for s in args.seeds.split(",")],
                              args.scenes.split(","),
                              [int(s) for s in args.steps.split(",")] if args.steps else None,
                              lora, args.cfg, tag,
                              run_dir=Path(args.run) if args.run else None))
    from scripts.research_frames import archive
    if args.note:
        print(f"archived into {archive(run_dir, args.note)}")
    else:
        print(f"archive when done: uv run python scripts/research_frames.py archive "
              f"{run_dir.relative_to(ROOT)} --note '...'")


if __name__ == "__main__":
    main()
