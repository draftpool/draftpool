"""A job that cannot start fails its own card, never the queue.

The regression: `run_job` looked up `registry.models[gen.model_key]` above its
own try block, so a key removed from models.yaml raised KeyError there. The
worker had a `finally` but no `except`, so that escaped `while True` and killed
the worker for the life of the process -- every later job stayed "pending",
Draw Things went idle, and /api/health still reported everything up.
"""

import asyncio
from types import SimpleNamespace

import pytest

from app import main


@pytest.fixture
def spy(monkeypatch):
    seen = []
    monkeypatch.setattr(
        main.store, "set_status",
        lambda gid, status, *a, **k: seen.append((gid, status, a[0] if a else None)),
    )
    monkeypatch.setattr(main, "notify", lambda *a, **k: None)
    return seen


def gen(gen_id, model_key="gone", status="pending"):
    return SimpleNamespace(
        id=gen_id, session_id="s1", model_key=model_key, kind="draft",
        status=status, describe_first=False, parent_id=None,
    )


async def test_a_removed_model_key_errors_the_card_not_the_queue(spy, monkeypatch):
    monkeypatch.setattr(main.store, "get_generation", lambda gid: gen(gid))

    await main.run_job("g1")  # must return, not raise

    assert [s for _, s, _ in spy] == ["error"]
    assert "unknown model key: gone" in spy[0][2]


async def test_the_worker_keeps_draining_after_a_job_raises(monkeypatch):
    """The queue must outlive one job that blows up before run_job's handler."""
    done = []

    async def run_job(gen_id):
        if gen_id == "bad":
            raise KeyError("unknown model key: gone")
        done.append(gen_id)

    monkeypatch.setattr(main, "run_job", run_job)
    monkeypatch.setattr(main.store, "get_generation", lambda gid: gen(gid))
    monkeypatch.setattr(main.store, "set_status", lambda *a, **k: None)
    monkeypatch.setattr(main, "notify", lambda *a, **k: None)
    # a queue of this test's own: job_queue is a module global that other
    # tests leave rows in, and worker() reads it by name at call time
    queue: asyncio.Queue = asyncio.Queue()
    monkeypatch.setattr(main, "job_queue", queue)

    for gen_id in ("bad", "good1", "good2"):
        queue.put_nowait(gen_id)

    task = asyncio.create_task(main.worker())
    await asyncio.wait_for(queue.join(), timeout=5)
    task.cancel()

    # the two after the bad one still ran: the worker survived
    assert done == ["good1", "good2"]
    assert not task.done() or task.cancelled()
