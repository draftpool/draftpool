"""A stalled Draw Things session holds the job; a stalled request fails it."""

from types import SimpleNamespace

import pytest

from app import main
from app.dt_client import DrawThingsStalled


@pytest.fixture
def held(monkeypatch):
    seen = []
    monkeypatch.setattr(main.store, "set_status", lambda gid, status, *a, **k: seen.append(status))
    monkeypatch.setattr(main, "notify", lambda *a, **k: None)

    async def no_sleep(_):
        seen.append("slept")

    monkeypatch.setattr(main.asyncio, "sleep", no_sleep)
    return seen


def fake_dt(monkeypatch, renders):
    calls = {"probes": 0}

    async def probe(model_filename, **kw):
        calls["probes"] += 1
        return renders[min(calls["probes"], len(renders)) - 1]

    monkeypatch.setattr(main.dt, "renders", probe)
    return calls


GEN = SimpleNamespace(id="g1", session_id="s1")
SPEC = SimpleNamespace(dt_filename="m.ckpt")


async def test_a_job_that_renders_is_left_alone(held, monkeypatch):
    calls = fake_dt(monkeypatch, [True])

    async def render():
        return b"png"

    assert await main.render_or_hold(GEN, SPEC, render) == b"png"
    assert held == [] and calls["probes"] == 0  # nothing held, nothing probed


async def test_a_stall_waits_for_the_renderer_and_then_succeeds(held, monkeypatch):
    # the probe says no twice (the app has not been restarted yet), then yes
    calls = fake_dt(monkeypatch, [False, False, True])
    attempts = []

    async def render():
        attempts.append(1)
        if len(attempts) == 1:
            raise DrawThingsStalled("no images")
        return b"png"

    assert await main.render_or_hold(GEN, SPEC, render) == b"png"
    assert len(attempts) == 2
    # held, waited out two failed probes, then put back to running
    assert held == ["pending", "slept", "slept", "running"]
    assert calls["probes"] == 3


async def test_a_second_stall_on_a_working_session_fails_the_job(held, monkeypatch):
    fake_dt(monkeypatch, [True])

    async def render():
        raise DrawThingsStalled("no images")

    with pytest.raises(DrawThingsStalled) as exc:
        await main.render_or_hold(GEN, SPEC, render)
    assert "rendered a test image" in str(exc.value)


async def test_other_failures_are_not_held(held, monkeypatch):
    fake_dt(monkeypatch, [True])

    async def render():
        raise RuntimeError("parent generation not found")

    with pytest.raises(RuntimeError):
        await main.render_or_hold(GEN, SPEC, render)
    assert held == []
