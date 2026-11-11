"""The soundtrack Draw Things renders with a clip, read back from its history.

MiniMax H3 renders sound with the frames, but /sdapi/v1/img2img answers with
`images` only: the HTTP handler drops the audio it is handed
(`let (images, _) = ...generateImages` in draw-things-community's
HTTPAPIServer.swift) and no request field changes that. Its gRPC API does carry
it, but Draw Things serves one protocol at a time and everything else here
speaks HTTP. What the app does do is save every HTTP render to the open
project's history, sound included -- which is where the clip plays with sound in
Draw Things -- so the soundtrack is read back from there:

- a project is a SQLite file in the app's container, `Documents/*.sqlite3`;
- table `clip` holds one row per clip, a flatbuffer (`clip.fbs`: clip_id,
  count, frames_per_second, width, height, audio_id);
- table `tensors` holds the waveform as `audio_<audio_id>`: float32
  [channels, samples], fpzip-compressed by s4nnc, whose codec id 0xf7217 sits
  in the high half of `type`.

A clip is ours when it is newer than every clip the project held before the
request went out and has the frame count and size that came back. Anything that
does not fit -- Draw Things on another machine, the project switched or the
clip deleted mid-render, the format changed under us -- ends in a silent clip,
which is what every clip was before, and never in a failed job.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import struct
import time
from contextlib import closing
from pathlib import Path
from typing import Optional
from urllib.parse import quote, urlparse

import numpy as np

from .video import Soundtrack

log = logging.getLogger(__name__)

DOCUMENTS = Path.home() / "Library/Containers/com.liuliu.draw-things/Data/Documents"

_FPZIP = 0xF7217
_CCV_32F = 0x04000
# The rate is not stored with the waveform. It is read off the clip's own
# length instead of being configured per model: H3's 22-frame clip carries
# 29,600 samples (32,290/s) and its 73-frame one 97,600 (32,088/s), so the
# nearest common rate is unambiguous and a later model with a different one
# needs no entry.
_RATES = (16000, 22050, 24000, 32000, 44100, 48000)


def reachable(host: str) -> bool:
    """Whether Draw Things runs on this machine, so its history is on disk here."""
    return urlparse(host).hostname in ("127.0.0.1", "localhost", "::1")


def _connect(path: Path) -> sqlite3.Connection:
    # read-only: this is the app's own database, open in the app while we read
    return sqlite3.connect(f"file:{quote(str(path))}?mode=ro", uri=True, timeout=5)


def snapshot() -> dict[str, int]:
    """The newest clip id in each project, taken before the request goes out.

    A project with no clip table yet (never rendered a clip) counts from -1. A
    project that cannot be read is left out, and so is never searched: with no
    baseline, an older clip of the same length and size would pass for ours.
    """
    out: dict[str, int] = {}
    try:
        # macOS guards other apps' containers; a refusal costs the sound only
        projects = sorted(DOCUMENTS.glob("*.sqlite3"))
    except OSError as exc:
        log.warning("cannot list Draw Things' projects: %s", exc)
        return out
    for path in projects:
        try:
            with closing(_connect(path)) as db:
                if db.execute(
                    "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'clip'"
                ).fetchone() is None:
                    out[str(path)] = -1
                else:
                    out[str(path)] = db.execute(
                        "SELECT coalesce(max(__pk0), -1) FROM clip"
                    ).fetchone()[0]
        except sqlite3.Error as exc:
            log.debug("draw things project %s unreadable: %s", path.name, exc)
    return out


def _clip(blob: bytes) -> tuple[int, int, int, int]:
    """(count, width, height, audio_id) from a `clip` row's flatbuffer."""
    root = struct.unpack_from("<I", blob, 0)[0]
    vtable = root - struct.unpack_from("<i", blob, root)[0]
    fields = (struct.unpack_from("<H", blob, vtable)[0] - 4) // 2

    def field(i: int, fmt: str) -> int:
        # an absent field is the schema default, which is 0 for all of these
        offset = struct.unpack_from("<H", blob, vtable + 4 + 2 * i)[0] if i < fields else 0
        return struct.unpack_from(fmt, blob, root + offset)[0] if offset else 0

    return field(1, "<i"), field(3, "<i"), field(4, "<i"), field(5, "<q")


def _decode(type_: int, datatype: int, dim: bytes, data: bytes) -> np.ndarray:
    if type_ >> 32 != _FPZIP or datatype != _CCV_32F:
        raise ValueError(f"audio stored as codec {type_ >> 32:#x}, type {datatype:#x}")
    dims = [d for d in struct.unpack(f"<{len(dim) // 4}i", dim) if d]
    if len(dims) != 2 or dims[0] not in (1, 2):
        raise ValueError(f"audio of shape {dims} is not [channels, samples]")
    import fpzip

    return fpzip.decompress(data, order="C").reshape(dims)


def find(
    before: dict[str, int], frames: int, width: int, height: int, fps: float,
) -> Optional[Soundtrack]:
    """The soundtrack of the clip rendered since `before`, if it is saved yet."""
    for name, newest in before.items():
        path = Path(name)
        try:
            with closing(_connect(path)) as db:
                rows = db.execute(
                    "SELECT p FROM clip WHERE __pk0 > ? ORDER BY __pk0 DESC", (newest,)
                ).fetchall()
                for (blob,) in rows:
                    count, w, h, audio_id = _clip(blob)
                    if (count, w, h) != (frames, width, height) or not audio_id:
                        continue
                    row = db.execute(
                        "SELECT type, datatype, dim, data FROM tensors WHERE name = ?",
                        (f"audio_{audio_id}",),
                    ).fetchone()
                    if row is None:
                        continue
                    samples = _decode(*row)
                    seconds = frames / fps
                    rate = min(_RATES, key=lambda r: abs(samples.shape[1] / seconds - r))
                    return Soundtrack(samples, rate)
        except (sqlite3.Error, struct.error, ValueError) as exc:
            log.warning("could not read the soundtrack from %s: %s", path.name, exc)
    return None


async def soundtrack(
    before: dict[str, int], frames: int, width: int, height: int, fps: float,
    *, wait: float = 20.0,
) -> Optional[Soundtrack]:
    """`find`, retried for `wait` seconds: the app may still be saving the clip
    when the HTTP response arrives."""
    if not before:
        return None
    deadline = time.monotonic() + wait
    while True:
        found = await asyncio.to_thread(find, before, frames, width, height, fps)
        if found is not None or time.monotonic() >= deadline:
            return found
        await asyncio.sleep(1.0)
