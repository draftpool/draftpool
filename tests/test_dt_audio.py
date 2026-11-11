"""Reading a clip's soundtrack back out of Draw Things' project history."""

import asyncio
import io
import json
import sqlite3
import struct
import subprocess

import fpzip
import numpy as np
import pytest
from PIL import Image

from app import dt_audio, video

# A real `clip` row from Draw Things' project file: clip 20, MiniMax H3, 73
# frames at 24 fps, 640x640. Checks the flatbuffer reading against the app's
# own layout rather than one built here.
REAL_CLIP = bytes.fromhex(
    "1400000010002C0024002000140010000C0004001000000002636172000000008002000080020000"
    "000000000000384000000000490000001400000000000000"
)


def _clip_blob(clip_id: int, count: int, width: int, height: int, audio_id: int) -> bytes:
    # field order as clip.fbs: clip_id, count, frames_per_second, width, height, audio_id
    vtable = struct.pack("<8H", 16, 40, 8, 4, 16, 24, 28, 32 if audio_id else 0)
    table = struct.pack("<iiqdiiq", 4 + len(vtable) - 4, count, clip_id, 24.0, width,
                        height, audio_id)
    return struct.pack("<I", 4 + len(vtable)) + vtable + table


def _project(path, clips=(), audio=None):
    """A Draw Things project file holding `clips` [(clip_id, count, w, h, audio_id)]
    and `audio` {audio_id: float32 [channels, samples]}."""
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE clip (rowid INTEGER PRIMARY KEY AUTOINCREMENT, __pk0 INTEGER,"
               " p BLOB, UNIQUE(__pk0))")
    db.execute("CREATE TABLE tensors (name TEXT, type INTEGER, format INTEGER, datatype"
               " INTEGER, dim BLOB, data BLOB, PRIMARY KEY (name))")
    _add(db, clips, audio)
    db.close()


def _add(db, clips=(), audio=None):
    for c in clips:
        db.execute("INSERT INTO clip (__pk0, p) VALUES (?, ?)", (c[0], _clip_blob(*c)))
    for audio_id, samples in (audio or {}).items():
        dim = struct.pack("<12i", *samples.shape, *[0] * (12 - samples.ndim))
        db.execute("INSERT INTO tensors VALUES (?, ?, 1, 16384, ?, ?)",
                   (f"audio_{audio_id}", (0xF7217 << 32) | 1, dim,
                    fpzip.compress(samples, order="C")))
    db.commit()


def _sound(seconds: float, rate: int = 32000) -> np.ndarray:
    t = np.arange(int(seconds * rate)) / rate
    tone = 0.5 * np.sin(2 * np.pi * 440 * t)
    return np.stack([tone, tone]).astype(np.float32)


@pytest.fixture
def documents(tmp_path, monkeypatch):
    d = tmp_path / "Documents"
    d.mkdir()
    monkeypatch.setattr(dt_audio, "DOCUMENTS", d)
    return d


def test_reads_a_real_draw_things_clip_row():
    assert dt_audio._clip(REAL_CLIP) == (73, 640, 640, 1918984962)


def test_finds_the_soundtrack_of_the_clip_just_rendered(documents):
    _project(documents / "p.sqlite3", [(3, 22, 256, 256, 77)], {77: _sound(1.0)})
    before = dt_audio.snapshot()
    assert before == {str(documents / "p.sqlite3"): 3}
    db = sqlite3.connect(documents / "p.sqlite3")
    _add(db, [(4, 73, 640, 640, 99)], {99: _sound(97600 / 32000)})
    db.close()
    found = dt_audio.find(before, 73, 640, 640, 24)
    assert found is not None
    assert found.samples.shape == (2, 97600)
    # the rate is not stored; it is read off the clip's length
    assert found.rate == 32000
    assert np.allclose(found.samples, _sound(97600 / 32000), atol=1e-6)


def test_an_identical_older_clip_is_never_taken_for_the_new_one(documents):
    """A Retry renders the same length and size again. Until Draw Things has
    saved the new clip, the old one matches on everything but its age."""
    _project(documents / "p.sqlite3", [(4, 73, 640, 640, 99)], {99: _sound(3.05)})
    before = dt_audio.snapshot()
    assert dt_audio.find(before, 73, 640, 640, 24) is None


def test_a_new_clip_of_another_size_or_without_sound_is_not_ours(documents):
    _project(documents / "p.sqlite3")
    before = dt_audio.snapshot()
    db = sqlite3.connect(documents / "p.sqlite3")
    # rendered in the app meanwhile, and a clip with no audio stream (Wan)
    _add(db, [(1, 56, 640, 640, 5), (2, 73, 640, 640, 0)], {5: _sound(2.3)})
    db.close()
    assert dt_audio.find(before, 73, 640, 640, 24) is None


def test_a_project_that_never_rendered_a_clip_counts_from_the_start(documents):
    sqlite3.connect(documents / "fresh.sqlite3").execute("CREATE TABLE t (x)").connection.close()
    (documents / "junk.sqlite3").write_bytes(b"not a database at all, just bytes " * 4)
    assert dt_audio.snapshot() == {str(documents / "fresh.sqlite3"): -1}


def test_a_container_macos_will_not_open_costs_only_the_sound(documents, monkeypatch):
    def refused(self, pattern):
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(type(documents), "glob", refused)
    assert dt_audio.snapshot() == {}


async def test_nothing_to_search_does_not_wait(documents):
    t = asyncio.get_running_loop().time()
    assert await dt_audio.soundtrack({}, 22, 640, 640, 24, wait=30) is None
    assert asyncio.get_running_loop().time() - t < 1


def test_only_a_local_draw_things_has_its_history_here():
    assert dt_audio.reachable("http://127.0.0.1:7860")
    assert dt_audio.reachable("http://localhost:7860/")
    assert not dt_audio.reachable("http://192.168.1.20:7860")


def _png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (200, 40, 40)).save(buf, "PNG")
    return buf.getvalue()


def _streams(path) -> list[str]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "json",
         str(path)], capture_output=True, text=True, check=True).stdout
    return [s["codec_type"] for s in json.loads(out)["streams"]]


ffmpeg = pytest.mark.skipif(not video.ffmpeg_available(), reason="ffmpeg not installed")


@ffmpeg
def test_assemble_muxes_the_soundtrack(tmp_path):
    out = video.assemble([_png()] * 24, 24, tmp_path, "clip",
                         audio=video.Soundtrack(_sound(1.0), 32000))
    assert _streams(out) == ["video", "audio"]


@ffmpeg
def test_a_soundtrack_ffmpeg_refuses_costs_the_sound_not_the_clip(tmp_path):
    # a rate no wav can carry makes the audio leg fail
    out = video.assemble([_png()] * 24, 24, tmp_path, "clip",
                         audio=video.Soundtrack(_sound(1.0), 0))
    assert out.suffix == ".mp4"
    assert _streams(out) == ["video"]


@ffmpeg
async def test_an_h3_job_comes_out_with_its_sound(documents, tmp_path, monkeypatch):
    """The whole path: Draw Things returns frames only and saves the clip,
    sound included, to its history; the card's mp4 carries that sound."""
    from fastapi.testclient import TestClient

    from app import main
    from app.registry import CONFIG_PATH, load_registry
    from app.sessions import Store

    reg = load_registry(CONFIG_PATH)
    reg.video_models["minimax-h3"].installed = True
    reg.draw_things.host = "http://127.0.0.1:7860"
    monkeypatch.setattr(main, "registry", reg)
    monkeypatch.setattr(main, "store", Store(tmp_path / "data"))
    monkeypatch.setattr(type(reg), "resolve_installed", lambda self: None)
    _project(documents / "p.sqlite3", [(1, 22, 64, 64, 5)], {5: _sound(0.9)})

    async def img2video(init_png, **kw):
        db = sqlite3.connect(documents / "p.sqlite3")
        _add(db, [(2, kw["num_frames"], 64, 64, 6)], {6: _sound(29600 / 32000)})
        db.close()
        return [_png()] * kw["num_frames"]

    async def ready(gen):
        return None

    monkeypatch.setattr(main.dt, "img2video", img2video)
    monkeypatch.setattr(main, "wait_for_draw_things", ready)
    client = TestClient(main.app)  # no lifespan: no worker to race the job
    sid = client.post("/api/sessions", json={"instruction": "a cat"}).json()["id"]
    still = main.store.create_generation(
        session_id=sid, kind="final", model_key="juggernaut-ragnarok", prompt="a cat",
        negative_prompt="", seed=1, width=64, height=64, steps=20, cfg=4.0,
        sampler="DPM++ 2M Karras",
    )
    main.store.save_image(still, _png())
    main.store.set_status(still.id, "done", None)
    clip = client.post(f"/api/generations/{still.id}/animate",
                       json={"model_key": "minimax-h3", "num_frames": 22}).json()
    await main.run_video_job(main.store.get_generation(clip["id"]))
    gen = main.store.get_generation(clip["id"])
    assert gen.status == "done", gen.error
    path = main.store.data_dir / "sessions" / gen.session_id / f"{gen.id}.mp4"
    assert _streams(path) == ["video", "audio"]
