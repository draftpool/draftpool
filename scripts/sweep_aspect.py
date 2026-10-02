"""Render every shape the UI offers against a square control, keeping every frame.

    uv run python scripts/sweep_aspect.py --models wai-illustrious,qwen-image
    uv run python scripts/sweep_aspect.py --models all --seeds 1111,2222

What this checks is `aspect_size` in config/models.yaml: each non-square ratio
in app.dims.ASPECTS, at the size the app would really request for a final
(`stage_dims(final.size, ratio, aspect_size)`), beside a square at the final's
own size on the same prompt and seed. The square is the control SHAPES.md's
first wrong conclusion lacked -- a subject repeated in the tall frame means
nothing until the square at that seed is seen to be clean.

The default of every shape and two seeds is the exhaustive form, and it is
rarely what is wanted. The cheap form, which is what the test in
tests/test_registry.py asks for, is one cell of the widest shape at a few
steps:

    uv run python scripts/sweep_aspect.py --models bas-anima --shapes 16:9 --seeds 1111 --steps 4

If it doubles, step `--aspect-size` down one grid notch from the failing size
and probe again, stop at the first that holds, and ship that. Every doubling
on record is at the longest edge, so that cell is the whole question; the
figure count reads at a third of the steps even where the frame is a smear
(SHAPES.md, the third wrong turn). A full sweep of z-image-turbo at two
budgets, 45 minutes, only bracketed an answer that one 48-second cell then
gave.

It decides nothing: the question is whether the subject appears twice, or a
second horizon or head turns up down the frame, and that is read by eye on the
sheets. `detect_faces` is not used -- it has counted flowers as faces here.

The scene is one standing figure, full body, in open country, because that is
what a tall or wide frame duplicates first: the model fills the extra length
with a second person or a second horizon.

Each cell is preceded by a 256px render on a different model, so a broken
session stops the run instead of producing frames that look like findings
(docs/decisions/FALSE-SUCCESS.md). An empty `images` answer on the cell itself
is recorded as a row rather than raised: on qwen-image that is the step
ceiling, and it is part of what is being measured.

Resumable; output in research/aspect/<key>/ (PNGs gitignored, results.json
committed).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app import ROOT
from app.dims import ASPECTS, ratio, stage_dims
from app.dt_client import DrawThingsClient, DrawThingsError, DrawThingsStalled
from app.registry import load_registry, lora_tag
from research_frames import archive, archived, keep, start_run
from sweep_final_steps import analyse, sheet

TOPIC = "aspect"
OUT = ROOT / "research" / TOPIC   # archived frames; a run stages in runs/ (research_frames.py)

SCENE = {
    "prose": (
        "A young woman standing alone in a wide field of sunflowers, full body, "
        "facing the viewer: long dark hair, a white sundress, a straw hat held in "
        "one hand. Low horizon, a blue sky with a few white clouds, a single tree "
        "in the far distance. Photographic, 50mm, afternoon light."
    ),
    "tags": (
        "1girl, solo, full body, standing, looking at viewer, long hair, "
        "black hair, white sundress, holding hat, straw hat, sunflower field, "
        "horizon, blue sky, clouds, tree, outdoors, day"
    ),
}


def steps_for(spec, mode: str) -> int:
    if mode == "final":
        return spec.final.steps
    if mode == "draft":
        return spec.draft.steps
    return int(mode)


async def run(keys: list[str], seeds: list[int], shapes: list[str],
              steps_mode: str, lora_options: bool = False,
              aspect_size: int | None = None, run_dir: Path | None = None) -> Path:
    run_dir = run_dir or start_run(TOPIC)
    # the models as they ship, unless a tick is what is being measured -- in
    # which case its recipe (a bigger final, fewer steps) is what renders,
    # and the frames carry the tick's key so they sit beside the plain ones
    reg = load_registry(lora_options=lora_options)
    dt = DrawThingsClient("http://127.0.0.1:7860")
    if not await dt.is_up():
        raise SystemExit("Draw Things is not answering on 127.0.0.1:7860")
    control_key = next(k for k in ("z-image-turbo", "krea2-turbo", "bas-anima")
                       if reg.models[k].available)

    for key in keys:
        spec = reg.models[key]
        if not spec.available:
            print(f"{key}: not installed, skipping")
            continue
        out = run_dir / key
        out.mkdir(parents=True, exist_ok=True)
        log = out / "results.json"
        results: list[dict] = json.loads(log.read_text()) if log.is_file() else []
        kept_log = OUT / key / "results.json"
        kept_rows: list[dict] = json.loads(kept_log.read_text()) if kept_log.is_file() else []
        # recorded in this run or an archived one: a cell is never rendered twice
        done = {r["file"] for r in (*kept_rows, *results)}

        def have(name: str) -> Path | None:
            return (out / name) if (out / name).is_file() else archived(TOPIC, f"{key}/{name}")
        if aspect_size:
            # a candidate budget, not the shipped one: what this sweep exists
            # to check before it goes into models.yaml
            spec.aspect_size = aspect_size
        size = spec.final.size
        steps = steps_for(spec, steps_mode)
        tag = lora_tag(spec)
        ltag = f"__{tag}" if tag else ""
        print(f"\n=== {key}{' with ' + tag if tag else ''}: final {size}, "
              f"aspect_size {spec.aspect_size}, {steps} steps ===")

        for seed in seeds:
            for shape in shapes:
                w, h = stage_dims(size, ratio(shape), spec.aspect_size)
                name = f"seed{seed}__{shape.replace(':', 'x')}__{w}x{h}__s{steps}{ltag}.png"
                if have(name):
                    print(f"  {name}: already on disk ({have(name).parent.relative_to(ROOT)})")
                    continue
                if name in done:
                    # a recorded empty answer: keep it, do not re-spend the render
                    print(f"  {name}: recorded as empty")
                    continue
                ctl = reg.models[control_key]
                ctl_png = await dt.txt2img(
                    prompt="a red cube on a wooden table", negative_prompt="",
                    model_filename=ctl.dt_filename, width=256, height=256,
                    steps=4, cfg=ctl.cfg, seed=7, sampler=ctl.sampler,
                    clip_skip=ctl.clip_skip, loras=[],
                )
                if analyse(ctl_png, (256, 256))["flat"]:
                    raise SystemExit("control render came back flat: restart Draw Things")

                prompt = SCENE[spec.prompt_style]
                if spec.quality_opener:
                    prompt = f"{spec.quality_opener}, {prompt}"
                t0 = time.monotonic()
                row = {"file": name, "model": key, "seed": seed, "shape": shape,
                       "width": w, "height": h, "steps": steps,
                       "aspect_size": spec.aspect_size,
                       "cfg": spec.final.cfg if spec.final.cfg is not None else spec.cfg,
                       "loras": [l.model_dump() for l in spec.final.loras]}
                try:
                    png = await dt.txt2img(
                        timeout=1500,
                        prompt=prompt,
                        negative_prompt=spec.default_negative,
                        model_filename=spec.dt_filename,
                        width=w, height=h, steps=steps,
                        cfg=spec.final.cfg if spec.final.cfg is not None else spec.cfg,
                        seed=seed,
                        sampler=spec.final.sampler or spec.sampler,
                        clip_skip=spec.clip_skip,
                        loras=[l.model_dump() for l in spec.final.loras],
                        tiled_diffusion=spec.tiled_diffusion,
                    )
                except DrawThingsStalled:
                    row |= {"seconds": round(time.monotonic() - t0, 1), "empty": True}
                    results.append(row)
                    log.write_text(json.dumps(results, indent=1))
                    print(f"  {name}: EMPTY after {row['seconds']}s  <-- LOOK")
                    continue
                except DrawThingsError as exc:
                    raise SystemExit(f"{name}: {exc}")
                row["seconds"] = round(time.monotonic() - t0, 1)
                keep(out / name, png)
                row |= {"empty": False, **analyse(png, (w, h))}
                results.append(row)
                log.write_text(json.dumps(results, indent=1))
                flag = "" if row["size_ok"] and not row["flat"] else "  <-- LOOK"
                print(f"  {name}: {row['seconds']}s  {row['out_size']}{flag}")

            paths, labels = [], []
            for shape in shapes:
                w, h = stage_dims(size, ratio(shape), spec.aspect_size)
                p = have(f"seed{seed}__{shape.replace(':', 'x')}__{w}x{h}__s{steps}{ltag}.png")
                if p:
                    paths.append(p)
                    labels.append(f"{key} seed {seed} {shape} {w}x{h} {steps}st{' ' + tag if tag else ''}")
            if paths:
                sheet(paths, labels, out / f"SHEET__seed{seed}__s{steps}{ltag}.png", height=640)
                print(f"  -> SHEET__seed{seed}__s{steps}{ltag}.png")
    return run_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help="comma-separated keys, or 'all'")
    # two seeds: SHAPES.md's second wrong conclusion came from one
    ap.add_argument("--seeds", default="1111,2222")
    ap.add_argument("--shapes", default=",".join(ASPECTS),
                    help="ratios from app.dims.ASPECTS; 1:1 is the square control")
    # Composition -- where a second figure or horizon appears -- is settled in
    # the first few steps, so the draft's count shows a duplication as well as
    # the final's at a fraction of the time. Not on qwen-image: its failure is
    # an empty answer past a long-edge x steps ceiling, so fewer steps would
    # hide exactly what is being checked. Nor where a model duplicates only at
    # a higher count (rinflanime at 1536/24): lower steps cannot show that.
    ap.add_argument("--steps", default="draft",
                    help="'draft', 'final', or a number")
    ap.add_argument("--lora-options", action="store_true",
                    help="render with the model's optional LoRAs at their shipped "
                         "ticks (their recipes included); filenames get the keys")
    ap.add_argument("--aspect-size", type=int, default=None,
                    help="a candidate pixel budget to check instead of the shipped one")
    ap.add_argument("--run", help="resume a stopped run in this runs/ folder")
    ap.add_argument("--note", help="what this run is for; given, the run is archived "
                                   "into research/ when it completes")
    args = ap.parse_args()
    reg = load_registry(lora_options=False)
    keys = ([k for k, m in reg.models.items() if m.aspect_size]
            if args.models == "all" else args.models.split(","))
    run_dir = asyncio.run(run(keys, [int(s) for s in args.seeds.split(",")],
                              args.shapes.split(","), args.steps,
                              lora_options=args.lora_options, aspect_size=args.aspect_size,
                              run_dir=Path(args.run) if args.run else None))
    if args.note:
        print(f"archived into {archive(run_dir, args.note)}")
    else:
        print(f"archive when done: uv run python scripts/research_frames.py archive "
              f"{run_dir.relative_to(ROOT)} --note '...'")


if __name__ == "__main__":
    main()
