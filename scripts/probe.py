"""Probe the Draw Things HTTP API: connectivity, model list, per-request model
switching, and rough timings. Run this before first use and after adding models.

    uv run python scripts/probe.py [--model <dt_filename>] [--size 256] [--steps 4]
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path

import httpx

# a script in scripts/ does not get the project root on sys.path, and probe
# output belongs wherever the rest of the data does (app/__init__ is stdlib
# only, so this import works even when the app itself cannot start)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import DATA_DIR  # noqa: E402

HOST = "http://127.0.0.1:7860"
OUT_DIR = DATA_DIR / "probe"


def get(client: httpx.Client, path: str):
    try:
        resp = client.get(HOST + path)
        return resp.status_code, resp
    except httpx.HTTPError as exc:
        return None, exc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="Draw Things model filename to test explicitly")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--steps", type=int, default=4)
    args = ap.parse_args()

    client = httpx.Client(timeout=600)

    print(f"== 1. Connectivity ({HOST}) ==")
    status, resp = get(client, "/")
    if status != 200:
        print(f"FAIL: Draw Things API not reachable ({resp}).")
        print("Enable it in Draw Things: Settings (gear) > API Server > HTTP, port 7860.")
        return 1
    config = resp.json()
    print("OK. Current app config keys:", ", ".join(sorted(config)[:20]), "...")
    current_model = config.get("model")
    print(f"Currently selected model in app: {current_model}")

    print("\n== 2. Optional A1111 endpoints ==")
    for path in ("/sdapi/v1/sd-models", "/sdapi/v1/options", "/sdapi/v1/samplers"):
        status, resp = get(client, path)
        if status == 200:
            body = resp.json()
            preview = json.dumps(body)[:300]
            print(f"OK   {path}: {preview}")
        else:
            print(f"MISS {path}: {status}")

    print("\n== 3. Generation test ==")
    model = args.model or current_model
    payload = {
        "prompt": "a red apple on a wooden table, studio lighting",
        "negative_prompt": "",
        "width": args.size,
        "height": args.size,
        "steps": args.steps,
        "seed": 42,
        "batch_size": 1,
    }
    if args.model:
        payload["model"] = args.model
        print(f"Testing per-request model switch to: {args.model}")
    else:
        print(f"Testing with app-selected model: {model}")

    t0 = time.time()
    resp = client.post(HOST + "/sdapi/v1/txt2img", json=payload)
    elapsed = time.time() - t0
    if resp.status_code != 200:
        print(f"FAIL: txt2img returned {resp.status_code}: {resp.text[:400]}")
        return 1
    body = resp.json()
    images = body.get("images") or []
    if not images:
        print(f"FAIL: no images in response: {str(body)[:400]}")
        return 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    b64 = images[0]
    if "," in b64 and b64.strip().startswith("data:"):
        b64 = b64.split(",", 1)[1]
    out = OUT_DIR / f"probe_{args.size}px_{int(elapsed)}s.png"
    out.write_bytes(base64.b64decode(b64))
    print(f"OK: {args.size}px x {args.steps} steps in {elapsed:.1f}s -> {out}")
    info = body.get("info")
    if info:
        print("Server info:", str(info)[:300])
    print("\nNext: copy the exact model filenames into config/models.yaml (dt_filename).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
