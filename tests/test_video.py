import io
import subprocess

import pytest
from PIL import Image

from app import video


def _frames(n: int, size=(64, 48)) -> list[bytes]:
    out = []
    for i in range(n):
        im = Image.new("RGB", size, (i * 8 % 256, 40, 200 - i * 4 % 256))
        buf = io.BytesIO()
        im.save(buf, "PNG")
        out.append(buf.getvalue())
    return out


def test_snap_frame_count_to_4n_plus_1():
    # Wan models expect 4n+1; anything else is snapped to the nearest valid value
    assert video.snap_frame_count(33) == 33
    assert video.snap_frame_count(81) == 81
    assert video.snap_frame_count(20) == 21
    assert video.snap_frame_count(50) == 49
    # below the minimum clamps up rather than producing a degenerate clip
    assert video.snap_frame_count(1) == 5
    assert video.snap_frame_count(0) == 5


def test_snap_frame_count_always_valid():
    for n in range(0, 100):
        snapped = video.snap_frame_count(n)
        assert snapped >= 5
        assert (snapped - 1) % 4 == 0


def test_frame_size_reads_actual_pixels():
    # Draw Things overrides requested sizes, so size must come from the frames
    assert video.frame_size(_frames(2, size=(832, 448))) == (832, 448)


def test_assemble_writes_webp_without_ffmpeg(tmp_path, monkeypatch):
    monkeypatch.setattr(video, "ffmpeg_available", lambda: False)
    out = video.assemble(_frames(5), fps=16, out_dir=tmp_path, stem="clip")
    assert out.name == "clip.webp"
    assert out.stat().st_size > 0
    with Image.open(out) as im:
        assert getattr(im, "n_frames", 1) == 5


def test_assemble_falls_back_to_webp_when_ffmpeg_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(video, "ffmpeg_available", lambda: True)

    def boom(*a, **k):
        raise RuntimeError("ffmpeg exploded")

    monkeypatch.setattr(video, "_write_mp4", boom)
    out = video.assemble(_frames(5), fps=16, out_dir=tmp_path, stem="clip")
    # a broken ffmpeg must not fail the job when Pillow can still write a clip
    assert out.name == "clip.webp"
    assert out.is_file()


def test_assemble_rejects_empty_frames(tmp_path):
    with pytest.raises(ValueError):
        video.assemble([], fps=16, out_dir=tmp_path, stem="clip")


def test_an_ffmpeg_that_cannot_run_counts_as_unavailable(monkeypatch):
    """Present on PATH is not the same as working.

    A Homebrew ffmpeg whose x265 was upgraded out from under it stays on PATH
    and aborts at dyld load. Reporting that as available makes every render pay
    a failed subprocess, and made the two mp4 tests below fail rather than skip.
    """
    monkeypatch.setattr(video.shutil, "which", lambda _: "/opt/homebrew/bin/ffmpeg")
    monkeypatch.setattr(
        video.subprocess, "run",
        lambda *a, **kw: subprocess.CompletedProcess(a[0], -6, b"", b"dyld: not loaded"),
    )
    assert video.ffmpeg_available() is False


def test_a_working_ffmpeg_counts_as_available(monkeypatch):
    monkeypatch.setattr(video.shutil, "which", lambda _: "/usr/bin/ffmpeg")
    monkeypatch.setattr(
        video.subprocess, "run",
        lambda *a, **kw: subprocess.CompletedProcess(a[0], 0, b"ffmpeg version 8.0", b""),
    )
    assert video.ffmpeg_available() is True


def test_a_missing_ffmpeg_is_not_probed(monkeypatch):
    """No subprocess at all when the binary is not there."""
    monkeypatch.setattr(video.shutil, "which", lambda _: None)
    monkeypatch.setattr(
        video.subprocess, "run",
        lambda *a, **kw: pytest.fail("should not probe a binary that is absent"),
    )
    assert video.ffmpeg_available() is False


@pytest.mark.skipif(not video.ffmpeg_available(), reason="ffmpeg not installed")
def test_assemble_writes_mp4_when_ffmpeg_present(tmp_path):
    out = video.assemble(_frames(5), fps=16, out_dir=tmp_path, stem="clip")
    assert out.name == "clip.mp4"
    assert out.stat().st_size > 0


@pytest.mark.skipif(not video.ffmpeg_available(), reason="ffmpeg not installed")
def test_assemble_handles_odd_dimensions(tmp_path):
    # yuv420p needs even dimensions; the pad filter must cover an odd size
    out = video.assemble(_frames(5, size=(65, 47)), fps=16, out_dir=tmp_path, stem="odd")
    assert out.suffix == ".mp4"
    assert out.stat().st_size > 0


def test_mp4_args_put_the_index_first():
    # without +faststart the moov atom lands after mdat and browsers stall on
    # the poster instead of playing
    args = video._MP4_ARGS
    assert "-movflags" in args
    assert "+faststart" in args[args.index("-movflags") + 1]


def test_clip_size_keeps_the_still_s_aspect():
    # the bug this exists for: a square still asked for 832x480 came back
    # flattened, because the init image is resized to the requested frame
    assert video.clip_size(512, 512, 832, 480) == (640, 640)
    assert video.clip_size(1024, 1024, 832, 480) == (640, 640)
    assert video.clip_size(1536, 1024, 832, 480) == (768, 512)
    assert video.clip_size(1024, 1536, 832, 480) == (512, 768)
    # rounding each side alone would give 704x576 here (1.22, not 4:3)
    assert video.clip_size(1024, 768, 832, 480) == (768, 576)
    assert video.clip_size(512, 1024, 832, 480) == (448, 896)
    # a still already at the model's shape is left where it is
    assert video.clip_size(832, 448, 832, 480) == (832, 448)


def test_clip_size_holds_area_and_bucket_alignment():
    for src_w, src_h in ((512, 512), (1024, 768), (768, 1024), (1920, 1080), (600, 900)):
        w, h = video.clip_size(src_w, src_h, 832, 480)
        # off-grid requests get floored by Draw Things, losing the aspect
        assert w % 64 == 0 and h % 64 == 0
        # aspect within a rounding step of the source
        assert abs((w / h) / (src_w / src_h) - 1) < 0.05
        # area stays in the model's league, so timing stays predictable
        assert 0.6 <= (w * h) / (832 * 480) <= 1.15


def test_clip_size_survives_extremes():
    # a panorama trades area rather than stretching out indefinitely
    w, h = video.clip_size(4096, 512, 832, 480)
    assert max(w, h) <= 1024 and min(w, h) >= 256
    # degenerate input must not produce a zero or negative frame
    assert video.clip_size(0, 0, 832, 480) == (640, 640)
