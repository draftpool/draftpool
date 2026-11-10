"""Check the setup: is Draw Things' API reachable, and which of the configured
models are on disk. Renders nothing unless asked: --render adds one tiny test
image (256px, 2 steps, a quick model) to prove a render comes back.

Nothing needs copying into config/models.yaml afterwards: the app matches every
configured model to the file Draw Things wrote, whatever variant was picked at
download, the same way the report at the end does.

    uv run python scripts/probe.py [--render [--model <key>]]
"""

from __future__ import annotations

import argparse
import base64
import sys
import time
from pathlib import Path

import httpx

# a script in scripts/ does not get the project root on sys.path, and probe
# output belongs wherever the rest of the data does (app/__init__ is stdlib
# only, so this import works even when the app itself cannot start)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import DATA_DIR  # noqa: E402
from app.dt_client import DrawThingsClient  # noqa: E402
from app.registry import Registry, load_registry  # noqa: E402
from app.schemas import ModelSpec  # noqa: E402

HOST = "http://127.0.0.1:7860"
OUT_DIR = DATA_DIR / "probe"
# a few seconds a draft each; the README's suggested first download leads
QUICK_MODELS = ["z-image-turbo", "flux2-klein"]


def get(client: httpx.Client, path: str):
    try:
        resp = client.get(HOST + path)
        return resp.status_code, resp
    except httpx.HTTPError as exc:
        return None, exc


def model_report(registry: Registry) -> list[str]:
    """What the app will find on disk, per configured model, after matching.

    Runs the same scan the server runs at startup, so a model shown missing
    here is one whose chip the page hides, and the remedy is downloading it in
    Draw Things -- never editing a filename.
    """
    wanted = {key: spec.dt_filename for key, spec in registry.models.items()}
    wanted_video = {key: v.dt_filename for key, v in registry.video_models.items()}
    registry.resolve_installed()
    if all(s.installed is None for s in registry.models.values()):
        dirs = ", ".join(registry.draw_things.models_dirs)
        return [f"No Draw Things model folder found (looked in {dirs})."]
    lines = []
    for title, specs, asked in (("Image models", registry.models, wanted),
                                ("Video models", registry.video_models, wanted_video)):
        if not specs:
            continue
        lines.append(f"{title}:")
        for key, spec in specs.items():
            if spec.installed:
                note = "" if spec.dt_filename == asked[key] else f"  (as {spec.dt_filename})"
                lines.append(f"  found    {spec.label}{note}")
            else:
                lines.append(f"  missing  {spec.label}  ({asked[key]}, or a LoRA it needs)")
    lines.append(f"Upscaler: {registry.upscaler or 'none found -- Upscale stays disabled'}")
    return lines


def pick_test_model(registry: Registry, key: str | None = None) -> ModelSpec | None:
    """The image model the generation test renders with.

    Always one from the registry and always sent by name: with no `model` in
    the request Draw Things renders with whatever its UI has selected, which
    after an Animate is the video model -- and the "quick test" became a clip.
    `key` picks one; otherwise the first installed of QUICK_MODELS, else the
    first installed in registry order. Not "fewest draft steps": an
    accelerator LoRA gives Qwen Image the fewest and it is still the slowest.
    Call after resolve_installed().
    """
    if key:
        return registry.models.get(key)
    installed = [s for s in registry.models.values() if s.installed]
    quick = [s for k in QUICK_MODELS for s in installed if s.key == k]
    return (quick or installed or [None])[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", action="store_true",
                    help="also render one 256px, 2-step test image")
    ap.add_argument("--model", help="with --render: registry key of the model (default: a quick one)")
    args = ap.parse_args()

    client = httpx.Client(timeout=600)

    print(f"== 1. Connectivity ({HOST}) ==")
    status, resp = get(client, "/")
    if status != 200:
        print(f"FAIL: Draw Things API not reachable ({resp}).")
        print("Enable it in Draw Things: Settings (gear) > API Server > HTTP, port 7860.")
        return 1
    print(f"OK. Selected in the app right now: {resp.json().get('model')}"
          " (the app and this probe always name their own model)")

    print("\n== 2. Models on disk ==")
    registry = load_registry()
    for line in model_report(registry):
        print(line)
    print("Missing models are hidden in the app; download them in Draw Things'"
          " model manager. Nothing in config/ needs editing.")

    if not args.render:
        return 0

    print("\n== 3. Test render ==")
    spec = pick_test_model(registry, args.model)
    if spec is None:
        print(f"FAIL: no model '{args.model}' in the registry" if args.model
              else "FAIL: no image model installed; download one in Draw Things first.")
        return 1
    if not spec.installed:
        print(f"FAIL: {spec.label} is not installed.")
        return 1
    print(f"Rendering with {spec.label} ({spec.dt_filename})")
    stage = spec.draft
    payload = DrawThingsClient(HOST)._build_payload(
        prompt="a red apple on a wooden table, studio lighting",
        negative_prompt="",
        model_filename=spec.dt_filename,
        width=256,
        height=256,
        steps=2,
        cfg=stage.cfg if stage.cfg is not None else 1.0,
        seed=42,
        sampler=stage.sampler,
        loras=[l.model_dump() for l in stage.loras],
    )
    t0 = time.time()
    resp = client.post(HOST + "/sdapi/v1/txt2img", json=payload)
    elapsed = time.time() - t0
    if resp.status_code != 200:
        print(f"FAIL: txt2img returned {resp.status_code}: {resp.text[:400]}")
        return 1
    images = resp.json().get("images") or []
    if not images:
        print(f"FAIL: no image came back: {resp.text[:400]}")
        return 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    b64 = images[0]
    if "," in b64 and b64.strip().startswith("data:"):
        b64 = b64.split(",", 1)[1]
    out = OUT_DIR / f"probe_{spec.key}_{int(elapsed)}s.png"
    out.write_bytes(base64.b64decode(b64))
    print(f"OK: 256px, 2 steps in {elapsed:.1f}s -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
