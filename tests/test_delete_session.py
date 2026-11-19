"""Deleting a session: its prompts, every card and every file it owns go.

What stays is what was never the session's: other sessions, and files in its
folder that this app did not write. A session with a render still queued or
running is refused, because that job would write into a session that is gone.
"""

import pytest
from fastapi.testclient import TestClient

from app import main
from app.sessions import SessionBusy, Store


@pytest.fixture
def store(tmp_path):
    return Store(data_dir=tmp_path)


@pytest.fixture
def client(store, monkeypatch):
    monkeypatch.setattr(main, "store", store)
    with TestClient(main.app) as c:
        yield c


def make(store, session_id, kind="draft", parent=None, status="done"):
    gen = store.create_generation(
        session_id=session_id, kind=kind, model_key="m", prompt=f"{kind} prompt",
        negative_prompt="", seed=7, width=256, height=256, steps=4,
        cfg=1.0, sampler=None, parent_id=parent.id if parent else None,
    )
    if status == "done":
        store.save_image(gen, b"\x89PNG")
    store.set_status(gen.id, status)
    return store.get_generation(gen.id)


def test_everything_in_the_session_goes(store):
    s = store.create_session("a cat")
    store.save_prompts(s.id, "a cat on a roof", {"groups": []})
    d = make(store, s.id)
    f = make(store, s.id, kind="final", parent=d)
    store.init_path(f).write_bytes(b"\x89PNG init")
    v = make(store, s.id, kind="video", parent=f)
    store.video_path(v).write_bytes(b"mp4")
    failed = make(store, s.id, status="error")

    done = store.delete_session(s.id)

    assert done["generations"] == 4
    assert sorted(done["files"]) == sorted(
        f"{s.id}/{name}" for name in (
            f"{d.id}.png", f"{f.id}.png", f"{f.id}_init.png",
            f"{v.id}.png", f"{v.id}.mp4",
        )
    )
    assert done["bytes"] > 0
    assert store.get_session(s.id) is None
    assert all(store.get_generation(g.id) is None for g in (d, f, v, failed))
    assert all(x.id != s.id for x in store.list_sessions())
    assert not (store.data_dir / "sessions" / s.id).exists()


def test_a_session_with_only_prompts_can_be_deleted(store):
    s = store.create_session("a cat")
    store.save_prompts(s.id, "a cat", {"groups": []})
    assert store.delete_session(s.id) == {"generations": 0, "files": [], "bytes": 0}
    assert store.get_session(s.id) is None


def test_other_sessions_are_never_touched(store):
    keep = store.create_session("keep")
    kept = make(store, keep.id)
    gone = store.create_session("gone")
    make(store, gone.id)
    store.delete_session(gone.id)
    assert store.get_session(keep.id) is not None
    assert store.get_generation(kept.id) is not None
    assert store.image_path(kept).is_file()


def test_a_file_this_app_did_not_write_is_left_with_its_folder(store):
    s = store.create_session("x")
    make(store, s.id)
    folder = store.data_dir / "sessions" / s.id
    (folder / "notes.txt").write_text("mine")
    store.delete_session(s.id)
    assert (folder / "notes.txt").read_text() == "mine"
    assert [p.name for p in folder.iterdir()] == ["notes.txt"]


@pytest.mark.parametrize("status", ["pending", "running"])
def test_a_session_still_rendering_is_refused_and_left_whole(store, status):
    s = store.create_session("x")
    d = make(store, s.id)
    make(store, s.id, kind="final", parent=d, status=status)
    with pytest.raises(SessionBusy, match="1 render in this session is still"):
        store.delete_session(s.id)
    assert store.get_session(s.id) is not None
    assert store.image_path(d).is_file()


def test_an_unknown_session_reports_none(store):
    assert store.delete_session("nope") is None


def test_the_endpoint_deletes_and_reports(client, store):
    s = store.create_session("x")
    d = make(store, s.id)
    resp = client.delete(f"/api/sessions/{s.id}")
    assert resp.status_code == 200
    assert resp.json()["files"] == [f"{s.id}/{d.id}.png"]
    assert client.get(f"/api/sessions/{s.id}").status_code == 404
    assert all(x["id"] != s.id for x in client.get("/api/sessions").json())


def test_the_endpoint_refuses_a_busy_session_with_a_reason(client, store):
    s = store.create_session("x")
    make(store, s.id, status="pending")
    resp = client.delete(f"/api/sessions/{s.id}")
    assert resp.status_code == 409
    assert "still queued or running" in resp.json()["detail"]


def test_the_endpoint_refuses_an_unknown_session(client):
    assert client.delete("/api/sessions/nope").status_code == 404
