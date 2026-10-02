"""Measured render timings: what the UI shows in place of a guess.

The store tests pin what counts as a measurement and how it is grouped; the
node tests pin that the page looks a job up under the same key the server
filed it under, since a mismatch there does not fail -- it just shows nothing.
"""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import dims, main
from app.sessions import TIMING_MIN_SAMPLES, TIMING_WINDOW, Store, timing_size

ROOT = Path(__file__).resolve().parent.parent


def done(store, sid, secs, *, kind="draft", model="m", w=768, h=768, steps=10,
         status="done", describe_first=False):
    g = store.create_generation(
        session_id=sid, kind=kind, model_key=model, prompt="p",
        negative_prompt="", seed=1, width=w, height=h, steps=steps, cfg=1.0,
        sampler=None, describe_first=describe_first,
    )
    store.set_status(g.id, status, duration_s=secs)
    return g


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path)


@pytest.fixture
def sid(store):
    return store.create_session("x").id


def test_a_group_is_the_median_seconds_per_step(store, sid):
    # 18s is the job that loaded the checkpoint, 167s the one that sat through
    # a restart: neither is what a draft takes
    for secs in (20.0, 22.0, 167.0, 21.0, 18.0):
        done(store, sid, secs, steps=10)
    assert store.render_timings() == [
        {"model_key": "m", "kind": "draft", "size": 768, "sec_per_step": 2.1, "n": 5}
    ]


def test_steps_are_divided_out_so_a_changed_step_count_still_matches(store, sid):
    for steps in (10, 20, 40):
        done(store, sid, steps * 2.0, steps=steps)
    (t,) = store.render_timings()
    assert t["sec_per_step"] == 2.0


def test_a_ratio_is_filed_with_the_square_of_the_same_budget(store, sid):
    # 768 at 3:4 and 9:16, and 1536 at 3:4 -- which dims.budget reads as 1552
    for w, h in ((768, 768), (640, 896), (576, 1024)):
        done(store, sid, 20.0, w=w, h=h)
    for _ in range(3):
        done(store, sid, 50.0, kind="final", w=1344, h=1792, steps=10)
    got = {(t["kind"], t["size"]): t["n"] for t in store.render_timings()}
    assert got == {("draft", 768): 3, ("final", 1536): 3}


def test_kinds_models_and_sizes_are_kept_apart(store, sid):
    for _ in range(3):
        done(store, sid, 10.0)
        done(store, sid, 40.0, kind="final", w=1536, h=1536)
        done(store, sid, 30.0, kind="faithful", w=1536, h=1536)
        done(store, sid, 50.0, model="other")
        done(store, sid, 5.0, w=512, h=512)
    rates = {(t["model_key"], t["kind"], t["size"]): t["sec_per_step"]
             for t in store.render_timings()}
    assert rates == {
        ("m", "draft", 512): 0.5,
        ("m", "draft", 768): 1.0,
        ("m", "faithful", 1536): 3.0,
        ("m", "final", 1536): 4.0,
        ("other", "draft", 768): 5.0,
    }


def test_too_few_renders_are_not_reported(store, sid):
    for _ in range(TIMING_MIN_SAMPLES - 1):
        done(store, sid, 20.0)
    assert store.render_timings() == []
    done(store, sid, 20.0)
    assert len(store.render_timings()) == 1


def test_only_work_whose_cost_is_fixed_is_measured(store, sid):
    for _ in range(TIMING_MIN_SAMPLES):
        # an LLM round in the duration
        done(store, sid, 900.0, kind="final", describe_first=True)
        # depends on the rectangle or the faces, not the size
        done(store, sid, 900.0, kind="edit")
        done(store, sid, 900.0, kind="detail")
        done(store, sid, 900.0, kind="esrgan")
        # not a render that finished
        done(store, sid, 900.0, status="error")
        # recovered rather than timed
        done(store, sid, None)
    assert store.render_timings() == []


def test_only_the_most_recent_window_counts(store, sid):
    # a machine that got faster: the old pace must age out, not be averaged in
    for _ in range(TIMING_WINDOW):
        done(store, sid, 100.0)
    for _ in range(TIMING_WINDOW):
        done(store, sid, 10.0)
    (t,) = store.render_timings()
    assert t["n"] == TIMING_WINDOW and t["sec_per_step"] == 1.0


def test_pruned_drafts_still_count(store, sid):
    # Remove drafts takes the picture, not the fact of how long it took
    for _ in range(TIMING_MIN_SAMPLES):
        g = done(store, sid, 20.0)
        with store._connect() as conn:
            conn.execute("UPDATE generations SET pruned_at = 'x' WHERE id = ?", (g.id,))
    assert len(store.render_timings()) == 1


def test_the_endpoint_serves_the_store(tmp_path, monkeypatch):
    s = Store(tmp_path)
    monkeypatch.setattr(main, "store", s)
    sid = s.create_session("x").id
    for _ in range(TIMING_MIN_SAMPLES):
        done(s, sid, 30.0, model="bas-anima")
    with TestClient(main.app) as c:
        got = c.get("/api/timings").json()
    assert got == [{"model_key": "bas-anima", "kind": "draft", "size": 768,
                    "sec_per_step": 3.0, "n": TIMING_MIN_SAMPLES}]


# -- the page's side ---------------------------------------------------------


def js_function(html: str, signature: str) -> str:
    fn = re.search(rf"^function {re.escape(signature)} \{{.*?^\}}", html, re.S | re.M)
    assert fn, f"{signature} missing from static/index.html"
    return fn.group(0)


def run_node(script: str):
    out = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")


@needs_node
def test_every_stage_size_the_app_renders_is_looked_up_under_its_own_key():
    """What the server files a render under is what the page asks for.

    Every size a stage can have, at every ratio, through the page's own
    stageDims and draftDims: a key that disagreed by one grid step would leave
    that model and ratio showing no time at all, however many renders it had.
    """
    html = (ROOT / "static" / "index.html").read_text()
    aspects = [{"key": k, "ratio": dims.ratio(k)} for k in dims.ASPECTS]
    models = [
        {"key": m.key, "supports_aspect": m.supports_aspect,
         "aspect_size": m.aspect_size,
         "faithful_aspect_size": m.shaped_cap(img2img=True),
         "faithful_size": m.faithful_budget,
         "draft": {"size": m.draft.size}, "final": {"size": m.final.size}}
        for m in main.registry.models.values()
    ]
    script = "\n".join([
        js_function(html, "stageDims(size, r, aspectSize)"),
        js_function(html, "stageSize(width, height)"),
        js_function(html, "draftDims(m, stage = m.draft)"),
        f"const ASPECTS = {json.dumps(aspects)};",
        f"const models = {json.dumps(models)};",
        "let selectedAspect;",
        "const out = [];",
        "for (const m of models) for (const a of ASPECTS) {",
        "  selectedAspect = a.key;",
        "  const [w, h] = draftDims(m);",
        "  const [fw, fh] = stageDims(m.final.size, a.ratio, m.aspect_size);",
        "  const [xw, xh] = stageDims(m.faithful_size, a.ratio, m.faithful_aspect_size);",
        "  out.push([w, h, stageSize(w, h), fw, fh, stageSize(fw, fh),",
        "            xw, xh, stageSize(xw, xh)]);",
        "}",
        "console.log(JSON.stringify(out));",
    ])
    expected, budgets = [], []
    for m in main.registry.models.values():
        for key in dims.ASPECTS:
            w, h, _ = main.draft_dims(m, m.draft, key)
            fw, fh = dims.stage_dims(m.final.size, dims.ratio(key), m.aspect_size)
            # a Faithful is not held to a composition cap (ModelSpec.shaped_cap)
            # and may have a budget of its own (ModelSpec.faithful_size)
            xw, xh = dims.stage_dims(m.faithful_budget, dims.ratio(key), m.shaped_cap(img2img=True))
            expected.append([w, h, timing_size(w, h), fw, fh, timing_size(fw, fh),
                             xw, xh, timing_size(xw, xh)])
            budgets.append((m, key))
    assert run_node(script) == expected
    # and the key is the budget the stage rendered at, so every ratio shares
    # the square's measurements -- except where aspect_size caps a non-square
    # render below the stage size, which then costs less and is its own group.
    # To the nearest grid step: qwen-image's cap is 864, which files as 896.
    # A Faithful is only capped where the cap is a render ceiling, so on the
    # SDXL entries it files under the square's budget at every ratio.
    for (m, key), (_, _, size, _, _, fsize, _, _, xsize) in zip(budgets, expected):
        for got, stage, budget, cap in ((size, m.draft, m.draft.size, m.aspect_size),
                                        (fsize, m.final, m.final.size, m.aspect_size),
                                        (xsize, m.final, m.faithful_budget,
                                         m.shaped_cap(img2img=True))):
            square = key == "1:1" or not (m.supports_aspect or stage is m.final)
            want = budget if square or not cap else min(budget, cap)
            assert abs(got - want) <= 32, (m.key, key, budget)


@needs_node
def test_the_page_derives_faithful_steps_as_the_server_does():
    html = (ROOT / "static" / "index.html").read_text()
    models = main.registry.models.values()
    finals = [{"final": {"steps": m.final.steps, "strength": m.final.strength}}
              for m in models]
    script = (
        js_function(html, "faithfulSteps(m)")
        + f"\nconsole.log(JSON.stringify({json.dumps(finals)}.map(faithfulSteps)));"
    )
    assert run_node(script) == [m.faithful_steps for m in models]


@needs_node
def test_an_upscale_is_looked_up_under_its_target_size():
    """The frame _queue_upscale renders, keyed as the page keys it.

    Unlike a draft or a final there is no aspect_size cap on an upscale, so
    every ratio of a target is one group -- the one the size picker names.
    """
    html = (ROOT / "static" / "index.html").read_text()
    ratios = [dims.ratio(k) for k in dims.ASPECTS]
    sizes = sorted(main.registry.upscale_sizes)
    script = "\n".join([
        js_function(html, "stageDims(size, r, aspectSize)"),
        js_function(html, "stageSize(width, height)"),
        f"const ratios = {json.dumps(ratios)}, sizes = {json.dumps(sizes)};",
        "console.log(JSON.stringify(sizes.flatMap(z => ratios.map(r => {",
        "  const [w, h] = stageDims(z, r); return stageSize(w, h); }))));",
    ])
    expected = [timing_size(*dims.stage_dims(z, r)) for z in sizes for r in ratios]
    assert run_node(script) == expected
    assert expected == [z for z in sizes for _ in ratios]


def test_upscales_are_timed_per_target_size(store, sid):
    for _ in range(TIMING_MIN_SAMPLES):
        done(store, sid, 80.0, kind="upscale", w=2048, h=2048, steps=8)
        # 3:4 at 3K
        done(store, sid, 144.0, kind="upscale", w=2688, h=3520, steps=8)
    rates = {t["size"]: t["sec_per_step"] for t in store.render_timings()}
    assert rates == {2048: 10.0, 3072: 18.0}
