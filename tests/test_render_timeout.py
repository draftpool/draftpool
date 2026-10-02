"""A render waits as long as the model has been taking, not a flat 900s."""

from types import SimpleNamespace

from app import main


def gen(width=1024, height=1024, steps=16, strength=None):
    return SimpleNamespace(width=width, height=height, steps=steps, strength=strength)


SPEC = SimpleNamespace(key="qwen-edit")


def rate_of(monkeypatch, rate):
    seen = {}

    def render_rate(model_key, long_edge):
        seen["asked"] = (model_key, long_edge)
        return rate

    monkeypatch.setattr(main.store, "render_rate", render_rate)
    monkeypatch.setattr(main.registry.draw_things, "request_timeout", 900)
    return seen


def test_no_history_leaves_the_configured_timeout_alone(monkeypatch):
    rate_of(monkeypatch, None)
    assert main.render_timeout(gen(), SPEC) is None


def test_the_edit_that_failed_would_now_have_finished(monkeypatch):
    # 20.2 s/step measured at 1024; a 50-step instruct edit is ~1010s of work
    # and used to be cut off at the flat 900
    rate_of(monkeypatch, 20.2)
    waited = main.render_timeout(gen(steps=50, strength=1.0), SPEC)
    assert waited > 1010
    assert waited == 20.2 * 50 * main.RENDER_TIMEOUT_SAFETY


def test_a_short_job_keeps_the_configured_floor(monkeypatch):
    rate_of(monkeypatch, 4.6)
    # 8 steps at 0.3 is 2.4 units, about 11s of work: nothing here shortens
    # the wait, it only ever lengthens it
    assert main.render_timeout(gen(steps=8, strength=0.3), SPEC) == 900


def test_a_rate_inflated_by_a_hold_cannot_hang_the_queue(monkeypatch):
    # a job held while Draw Things restarted carries the hold in its duration
    rate_of(monkeypatch, 4000.0)
    assert main.render_timeout(gen(steps=20), SPEC) == main.RENDER_TIMEOUT_CEILING


def test_the_rate_is_asked_for_this_job_s_size(monkeypatch):
    seen = rate_of(monkeypatch, 5.0)
    main.render_timeout(gen(width=1344, height=768), SPEC)
    assert seen["asked"] == ("qwen-edit", 1344)


def test_a_missing_strength_counts_as_a_full_pass(monkeypatch):
    rate_of(monkeypatch, 100.0)
    assert main.render_timeout(gen(steps=10, strength=None), SPEC) == 3000.0
