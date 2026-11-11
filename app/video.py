"""Assemble Draw Things' returned frames into a playable clip.

ffmpeg produces an h264 mp4, which every browser plays and which is far
smaller. It is not a hard requirement though: setup.sh does not install it, so
when it is missing we fall back to an animated WebP written with Pillow, which
is already a dependency. Callers get told which container was written because
the file extension differs.
"""

from __future__ import annotations

import io
import logging
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import NamedTuple, Optional

import numpy as np
from PIL import Image

# clip_size lives with the other grid geometry now that generation shapes
# stills the same way; re-exported here, where its callers look for it.
from .dims import clip_size  # noqa: F401

log = logging.getLogger(__name__)

# h264 needs even dimensions; Draw Things' buckets are multiples of 16, but a
# custom size could still arrive odd and yuv420p would fail on it.
# +faststart moves the moov atom to the front of the file. Without it the
# index sits after the frame data and a browser playing progressively has
# nothing to initialise from until the whole file has arrived, which shows up
# as a player that spins on the poster instead of starting.
_MP4_ARGS = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
             "-movflags", "+faststart",
             "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"]


class Soundtrack(NamedTuple):
    """A clip's sound: float [channels, samples] in -1..1, at `rate` per second."""

    samples: np.ndarray
    rate: int


def ffmpeg_available() -> bool:
    """Whether ffmpeg is here AND can actually run.

    `shutil.which` alone is not enough, and the difference is not academic: a
    Homebrew ffmpeg whose x265 was upgraded out from under it stays on PATH and
    aborts at dyld load, so `which` says yes and every render then pays a failed
    subprocess before falling back to WebP. Worse, it made the two mp4 tests
    fail rather than skip -- their guard is this function, and a guard that
    cannot tell "missing" from "broken" reports a broken toolchain as a broken
    app.

    One `-version` invocation, ~10ms, and no caching on purpose: this is called
    once per clip (a job measured in minutes) and a cached answer would keep
    writing WebP for the life of the process after the user had fixed ffmpeg.
    """
    if shutil.which("ffmpeg") is None:
        return False
    try:
        proc = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True, timeout=5, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0


def frame_size(frames: list[bytes]) -> tuple[int, int]:
    """Actual pixel size of the returned frames.

    Draw Things overrides the requested size (832x480 comes back 832x448), so
    the stored width/height must come from here, not from the request.
    """
    with Image.open(io.BytesIO(frames[0])) as im:
        return im.size


def assemble(
    frames: list[bytes], fps: int, out_dir: Path, stem: str,
    audio: Optional[Soundtrack] = None,
) -> Path:
    """Write frames as a clip; returns the path actually written.

    Extension is .mp4 when ffmpeg is present, .webp otherwise. The soundtrack
    goes into the mp4 only: WebP has no audio, and a soundtrack ffmpeg will not
    take costs the sound, not the clip.
    """
    if not frames:
        raise ValueError("no frames to assemble")
    out_dir.mkdir(parents=True, exist_ok=True)

    if ffmpeg_available():
        out = out_dir / f"{stem}.mp4"
        if audio is not None:
            try:
                return _write_mp4(frames, fps, out, audio)
            except Exception as exc:  # noqa: BLE001 - a silent clip beats none
                log.warning("ffmpeg could not add the soundtrack (%s); writing it silent", exc)
        try:
            return _write_mp4(frames, fps, out)
        except Exception as exc:  # noqa: BLE001 - fall back rather than fail the job
            log.warning("ffmpeg assembly failed (%s); falling back to webp", exc)
    return _write_webp(frames, fps, out_dir / f"{stem}.webp")


def _write_wav(audio: Soundtrack, out: Path) -> None:
    pcm = (np.clip(audio.samples, -1.0, 1.0).T * 32767).astype("<i2")
    with wave.open(str(out), "wb") as w:
        w.setnchannels(audio.samples.shape[0])
        w.setsampwidth(2)
        w.setframerate(audio.rate)
        w.writeframes(pcm.tobytes())


def _write_mp4(
    frames: list[bytes], fps: int, out: Path, audio: Optional[Soundtrack] = None
) -> Path:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for i, raw in enumerate(frames):
            (tmp / f"f{i:05d}.png").write_bytes(raw)
        sound: list[str] = []
        if audio is not None:
            _write_wav(audio, tmp / "audio.wav")
            sound = ["-i", str(tmp / "audio.wav"), "-c:a", "aac", "-b:a", "192k"]
        proc = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-framerate", str(fps), "-i", str(tmp / "f%05d.png"),
             *sound, *_MP4_ARGS, str(out)],
            capture_output=True, text=True, timeout=300,
        )
    if proc.returncode != 0 or not out.is_file():
        raise RuntimeError(f"ffmpeg exited {proc.returncode}: {proc.stderr[:300]}")
    return out


def _write_webp(frames: list[bytes], fps: int, out: Path) -> Path:
    images = [Image.open(io.BytesIO(f)).convert("RGB") for f in frames]
    images[0].save(
        out,
        save_all=True,
        append_images=images[1:],
        duration=max(1, round(1000 / fps)),
        loop=0,
        quality=80,
    )
    return out


def snap_frame_count(n: int, step: int = 4, offset: int = 1) -> int:
    """Snap to the nearest clip length on the model's grid: offset + step*k.

    Wan models take 4k+1 frames (the defaults); MiniMax H3 takes 17k+5 (its
    Turbo LoRA's card says so, and 22, 73 and 124 came back as asked). The
    queued length is then the one the LLM's motion plan is written for. One
    step is the shortest clip, so k is at least 1.
    """
    k = max(1, round((n - offset) / step))
    return offset + step * k


def clip_presets(fps: int, step: int = 4, offset: int = 1) -> list[int]:
    """Frame counts for the Animate dialog's 1, 2, 3 and 5 second choices.

    Each is the grid length nearest to that many seconds at the model's own
    fps: at 16 fps on Wan's grid that is 17/33/49/81, at 24 fps on H3's it is
    22/56/73/124. The dialog labels them with the seconds they really come to.
    """
    out: list[int] = []
    for seconds in (1, 2, 3, 5):
        n = snap_frame_count(seconds * fps, step, offset)
        if n not in out:
            out.append(n)
    return out
