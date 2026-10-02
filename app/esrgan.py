"""Non-generative 2x upscale via Real-ESRGAN (ncnn binary).

Runs entirely outside Draw Things: no queue, no diffusion, a few seconds.
Sharpens existing pixels without inventing anything, unlike the generative
Upscale pass.

The x4 models are used and the result is downscaled to 2x, which also acts as
mild anti-aliasing. That detour is not cosmetic: asking the binary for `-s 2`
directly is worse and barely quicker (Laplacian variance 359 against 754, the
two agreeing at only 0.71 correlation), so the oversized intermediate earns
its memory. Kept in preference to a Draw Things upscaler for this job because
it is both faster and sharper here: 2048 -> 4096 measured 17.0s at 754 against
30.8s at 525 for 4x UltraSharp. That is the opposite of the ranking for
*upscale init prep*, where the enlargement is 4x rather than 2x and UltraSharp
wins; see docs/decisions/UPSCALING.md.
"""

from __future__ import annotations

import asyncio
import io
import tempfile
from pathlib import Path

from PIL import Image

# The x4 pass produces an intermediate four times the input on each side, so a
# 4096px image (which Upscale can now produce) becomes 16384px = 268Mpx. Pillow
# refuses anything over 2x its 89Mpx default as a possible decompression bomb,
# which turned "2x fast" on a 4K image into a DecompressionBombError. These
# files are our own renders, written by the binary below, so the guard protects
# nothing here. Measured 4096 -> 8192: 64s for the pass, 4s to open and resize,
# 2.2GB peak RSS.
Image.MAX_IMAGE_PIXELS = None

TOOLS_DIR = Path(__file__).resolve().parent.parent / "tools" / "realesrgan"
BINARY = TOOLS_DIR / "realesrgan-ncnn-vulkan"
FLAVORS = {
    "photo": "realesrgan-x4plus",
    "anime": "realesrgan-x4plus-anime",
}


class EsrganError(RuntimeError):
    pass


def available() -> bool:
    return BINARY.is_file()


async def upscale2x(png: bytes, flavor: str) -> bytes:
    if flavor not in FLAVORS:
        raise EsrganError(f"unknown Real-ESRGAN flavor: {flavor}")
    if not available():
        raise EsrganError(
            "Real-ESRGAN binary not installed; run scripts/setup.sh "
            "(it downloads tools/realesrgan, ~50MB)"
        )
    # re-encode via Pillow: the binary's stb decoder rejects some PNG variants
    src = Image.open(io.BytesIO(png)).convert("RGB")
    with tempfile.TemporaryDirectory() as td:
        inp, outp = Path(td) / "in.png", Path(td) / "out.png"
        src.save(inp)
        proc = await asyncio.create_subprocess_exec(
            str(BINARY),
            "-i", str(inp),
            "-o", str(outp),
            "-n", FLAVORS[flavor],
            "-s", "4",
            "-m", str(TOOLS_DIR / "models"),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, err = await proc.communicate()
        if proc.returncode != 0 or not outp.is_file():
            raise EsrganError(
                f"realesrgan failed (exit {proc.returncode}): "
                f"{err.decode(errors='replace')[-300:]}"
            )
        out = Image.open(outp).convert("RGB")
    out = out.resize((src.width * 2, src.height * 2), Image.LANCZOS)
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return buf.getvalue()
