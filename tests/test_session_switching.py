"""Switching sessions must not touch work already queued.

The job queue and its worker belong to the server process, not to whatever
the page is currently looking at: a draft queued from one session keeps
rendering while you start another, and its result is written to the session
it belongs to. The page only stops *watching* -- it closes its event stream
and opens one for the new session.
"""

import asyncio
import time

import pytest
from fastapi.testclient import TestClient

from app import main
from app.sessions import Store

MODEL = "bas-anima"


ran_ids: list[str] = []


@pytest.fixture
def ran(monkeypatch):
    """Every gen_id the worker actually picked up, in order."""
    ran_ids.clear()

    async def fake_run_job(gen_id: str) -> None:
        main.store.set_status(gen_id, "done")
        ran_ids.append(gen_id)  # after the write, so a waiter sees both

    monkeypatch.setattr(main, "run_job", fake_run_job)
    return ran_ids


@pytest.fixture
def client(tmp_path, monkeypatch, ran):
    monkeypatch.setattr(main, "store", Store(tmp_path))
    # A fresh queue per test. asyncio.Queue binds to the first event loop that
    # uses it, and every TestClient runs the app on a new one, so the module's
    # queue would carry the previous test's loop (and its undrained jobs) into
    # this one. The server has a single loop for its whole life, so this is a
    # harness concern only.
    monkeypatch.setattr(main, "job_queue", asyncio.Queue())
    monkeypatch.setattr(main, "subscribers", {})
    with TestClient(main.app) as c:
        yield c


def drafts(client, sid, prompt):
    resp = client.post(f"/api/sessions/{sid}/drafts", json={
        "variants": [{"prompt": prompt}], "model_keys": [MODEL], "seeds_per_combo": 1,
    })
    assert resp.status_code == 200, resp.text
    return [g["id"] for g in resp.json()]


def drain(client, ids, timeout=5.0):
    """Wait for the worker to finish these, over HTTP.

    Polled rather than awaited: TestClient runs the app on its own event
    loop in another thread, and touching job_queue from this one binds it to
    the wrong loop.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if all(gid in ran_ids for gid in ids):
            return
        time.sleep(0.02)
    raise AssertionError(f"worker never picked up {set(ids) - set(ran_ids)}")


def test_a_new_session_does_not_stop_the_old_one_rendering(client, ran):
    old = client.post("/api/sessions", json={"instruction": "a cat"}).json()["id"]
    queued = drafts(client, old, "1girl, park")
    # "new session" in the picker: the page reloads and posts a new session.
    # Nothing about that reaches the queue.
    new = client.post("/api/sessions", json={"instruction": "a dog"}).json()["id"]
    assert new != old
    drain(client, queued)
    assert queued[0] in ran, "the old session's job was dropped"
    got = client.get(f"/api/sessions/{old}").json()["generations"]
    assert [g["status"] for g in got] == ["done"]


def test_both_sessions_work_is_kept_and_kept_apart(client, ran):
    a = client.post("/api/sessions", json={"instruction": "a cat"}).json()["id"]
    b = client.post("/api/sessions", json={"instruction": "a dog"}).json()["id"]
    qa = drafts(client, a, "1girl, park")
    qb = drafts(client, b, "1boy, beach")
    drain(client, qa + qb)
    assert set(qa + qb) <= set(ran)
    assert [g["id"] for g in client.get(f"/api/sessions/{a}").json()["generations"]] == qa
    assert [g["id"] for g in client.get(f"/api/sessions/{b}").json()["generations"]] == qb


def test_nobody_listening_is_not_an_error(client, ran):
    """Results are written to the store, not delivered to a live page.

    notify() only reaches whoever is subscribed, so a session with its event
    stream closed loses the live update and nothing else -- reopening it
    reads the finished state back off disk.
    """
    sid = client.post("/api/sessions", json={"instruction": "a cat"}).json()["id"]
    assert main.subscribers.get(sid, set()) == set()  # no page watching
    queued = drafts(client, sid, "1girl, park")
    drain(client, queued)
    assert queued[0] in ran
    assert client.get(f"/api/sessions/{sid}").json()["generations"][0]["status"] == "done"
