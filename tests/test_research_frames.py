"""A measurement frame is never replaced, and a missing one is reported."""

import json

from scripts.research_frames import keep


def test_keep_writes_a_new_name_and_never_overwrites(tmp_path):
    p = tmp_path / "market__seed1111__draft640.png"
    assert keep(p, b"first") == p
    again = keep(p, b"second")
    assert again != p and again.parent == p.parent
    assert again.name.startswith("market__seed1111__draft640__") and again.suffix == ".png"
    assert p.read_bytes() == b"first" and again.read_bytes() == b"second"


def test_the_check_names_missing_frames_and_skips_empty_renders(tmp_path, monkeypatch, capsys):
    import scripts.research_frames as rf
    folder = tmp_path / "research" / "x"
    folder.mkdir(parents=True)
    (folder / "a.png").write_bytes(b"x")
    (folder / "results.json").write_text(json.dumps([
        {"file": "a.png"}, {"file": "b.png"}, {"file": "c.png", "empty": True}]))
    monkeypatch.setattr(rf, "RESEARCH", tmp_path / "research")
    monkeypatch.setattr(rf, "ROOT", tmp_path)
    monkeypatch.setattr("sys.argv", ["research_frames.py"])
    assert rf.main() == 1
    out = capsys.readouterr().out
    assert "missing  b.png" in out and "c.png" not in out and "1 of 2" in out


def test_archive_moves_a_run_into_research_without_replacing_and_writes_it_up(tmp_path, monkeypatch):
    import scripts.research_frames as rf
    monkeypatch.setattr(rf, "ROOT", tmp_path)
    monkeypatch.setattr(rf, "RESEARCH", tmp_path / "research")
    monkeypatch.setattr(rf, "RUNS", tmp_path / "runs")
    monkeypatch.setattr("sys.argv", ["sweep_x.py", "--model", "m"])
    # an earlier run already archived one frame under the same name
    kept = tmp_path / "research" / "draft-size" / "m"
    kept.mkdir(parents=True)
    (kept / "a.png").write_bytes(b"old")
    (kept / "results.json").write_text(json.dumps([{"file": "a.png", "seconds": 1}]))

    run = rf.start_run("draft-size")
    assert run.parent == tmp_path / "runs" and json.loads((run / "run.json").read_text())["topic"] == "draft-size"
    assert rf.keep(run / "m" / "a.png", b"new") == run / "m" / "a.png"
    rf.keep(run / "m" / "b.png", b"b")
    (run / "m" / "results.json").write_text(json.dumps([{"file": "a.png", "seconds": 2},
                                                        {"file": "b.png", "seconds": 3}]))
    into = rf.archive(run, "why this was rendered")
    assert into == tmp_path / "research" / "draft-size"
    assert (kept / "a.png").read_bytes() == b"old"            # never replaced
    renamed = [p for p in kept.glob("a__*.png")]
    assert len(renamed) == 1 and renamed[0].read_bytes() == b"new"
    assert (kept / "b.png").read_bytes() == b"b"
    rows = json.loads((kept / "results.json").read_text())
    assert [r["file"] for r in rows] == ["a.png", renamed[0].name, "b.png"]
    text = (kept.parent / "RUNS.md").read_text()
    assert "why this was rendered" in text and "sweep_x.py --model m" in text
    assert f"a.png → {renamed[0].name}" in text and "b.png" in text
    assert not run.exists()                                    # staging folder gone
    # and an archived frame is what the sweeps consult before rendering
    assert rf.archived("draft-size", "m/b.png") == kept / "b.png"
    assert rf.archived("draft-size", "m/zzz.png") is None
