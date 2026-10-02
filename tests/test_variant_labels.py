"""Which composer prompt a draft came from, and which press of Generate.

Both used to be reconstructed in the page by string-matching prompts after
the fact, which is wrong in two ways that showed up together:

  * the original prompt stopped being recognised as soon as its box was
    edited -- the hygiene fixer rewriting one tag inside it was enough -- so
    it was counted as a further enriched variant and a round of three
    prompts put a "V4" on screen that the composer had no box for;
  * reopening a session folded every draft in it into one round, so a second
    round's prompts carried on numbering from the first.

So the label and the batch are recorded on the row when the prompt is queued.
These pin that contract end to end.
"""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import main
from app.sessions import Store

ROOT = Path(__file__).resolve().parents[1]

MODEL = "bas-anima"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "store", Store(tmp_path))
    with TestClient(main.app) as c:
        yield c


def make_session(client, instruction="a cat on a roof"):
    return client.post("/api/sessions", json={"instruction": instruction}).json()["id"]


def drafts(client, sid, variants, batch=None):
    body = {"variants": variants, "model_keys": [MODEL], "seeds_per_combo": 1}
    if batch:
        body["batch"] = batch
    resp = client.post(f"/api/sessions/{sid}/drafts", json=body)
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_the_label_the_composer_showed_is_stored_on_the_row(client):
    sid = make_session(client)
    made = drafts(client, sid, [
        {"prompt": "1girl, park", "label": "V1"},
        {"prompt": "1girl, rooftop", "label": "V2"},
    ], batch="b1")
    assert [g["variant_label"] for g in made] == ["V1", "V2"]
    assert {g["batch"] for g in made} == {"b1"}


def test_the_original_stays_the_original_after_its_box_is_edited(client):
    """The regression itself: a prompt that no longer echoes the instruction.

    The hygiene fixer rewrites tags inside the original box ("thigh-high
    stockings" -> "thighhighs"), and each model's quality opener goes on the
    front. Matching that text back to the instruction fails; the label does
    not care.
    """
    sid = make_session(client, "1girl, thigh-high stockings")
    made = drafts(client, sid, [
        {"prompt": "masterpiece, best quality, 1girl, thighhighs", "label": "orig"},
    ], batch="b1")
    assert made[0]["variant_label"] == "orig"
    assert "thigh-high stockings" not in made[0]["prompt"]


def test_two_presses_of_generate_are_two_batches(client):
    """The batch is the round; the label is the prompt, and they are separate.

    Pressing Generate twice on one set of prompts makes two rounds of the
    same V1, which is why the label cannot be derived from the batch.
    """
    sid = make_session(client)
    first = drafts(client, sid, [{"prompt": "1girl, park", "label": "V1"}], batch="b1")
    second = drafts(client, sid, [{"prompt": "1girl, park", "label": "V1"}], batch="b2")
    assert first[0]["batch"] != second[0]["batch"]
    assert first[0]["variant_label"] == second[0]["variant_label"] == "V1"


def test_label_and_batch_survive_reopening_the_session(client):
    """Deliberately order-independent: rows sort by timestamp then id.

    Two presses inside the same second come back interleaved, in whatever
    order their random ids fall. That is why the page groups the rows by
    batch rather than by where they sit next to each other.
    """
    sid = make_session(client)
    drafts(client, sid, [{"prompt": "1girl, park", "label": "T1"}], batch="b1")
    drafts(client, sid, [{"prompt": "1girl, beach", "label": "T1"}], batch="b2")
    gens = client.get(f"/api/sessions/{sid}").json()["generations"]
    # the stored prompt carries this model's own required opener (main.prompt_for),
    # so the written half is what identifies the row here
    assert {(g["batch"], g["variant_label"], g["prompt"].split(", 1girl, ")[-1])
            for g in gens} == {("b1", "T1", "park"), ("b2", "T1", "beach")}


def test_a_prompt_with_no_label_is_still_accepted(client):
    """More like this and Refine queue drafts of their own, without a box."""
    sid = make_session(client)
    made = drafts(client, sid, [{"prompt": "1girl, park"}])
    assert made[0]["variant_label"] is None
    assert made[0]["batch"] is None


def test_an_older_database_gains_the_columns(tmp_path):
    """The store opens a pre-existing db by migration, not by recreating it."""
    import sqlite3

    db = tmp_path / "draftpool.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        """CREATE TABLE sessions (id TEXT PRIMARY KEY, instruction TEXT NOT NULL,
                                  created_at TEXT NOT NULL);
           CREATE TABLE generations (
             id TEXT PRIMARY KEY, session_id TEXT NOT NULL, parent_id TEXT,
             kind TEXT NOT NULL, model_key TEXT NOT NULL, prompt TEXT NOT NULL,
             negative_prompt TEXT NOT NULL DEFAULT '', seed INTEGER NOT NULL,
             width INTEGER NOT NULL, height INTEGER NOT NULL, steps INTEGER NOT NULL,
             cfg REAL NOT NULL, sampler TEXT, status TEXT NOT NULL DEFAULT 'pending',
             error TEXT, created_at TEXT NOT NULL);
           INSERT INTO sessions VALUES ('s1', 'a cat', '2026-01-01T00:00:00+00:00');
           INSERT INTO generations VALUES
             ('g1','s1',NULL,'draft','bas-anima','1girl','',1,512,512,20,5.0,NULL,
              'done',NULL,'2026-01-01T00:00:00+00:00');"""
    )
    conn.commit()
    conn.close()

    gen = Store(tmp_path).get_generation("g1")
    assert gen is not None
    assert gen.variant_label is None and gen.batch is None


# -- the page's side of the same contract -------------------------------------


def _js(*names: str) -> str:
    """Pull named functions out of static/index.html, as test_dims.py does."""
    html = (ROOT / "static" / "index.html").read_text()
    out = []
    for name in names:
        m = re.search(rf"^function {name}\(.*?^\}}", html, re.S | re.M)
        assert m, f"{name} missing from static/index.html"
        out.append(m.group(0))
    return "\n".join(out)


def run_js(script: str):
    return json.loads(
        subprocess.run(["node", "-e", script], capture_output=True, text=True,
                       check=True).stdout
    )


needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")


@needs_node
def test_the_page_badges_a_round_by_its_stored_labels():
    """Three enriched prompts and the original: V1, V2, V3 and orig.

    The instruction deliberately does not match any of the prompts, which is
    what the old text-matching path needed and this one does not.
    """
    gens = [
        {"id": "a", "kind": "draft", "model_key": "m", "prompt": "x", "variant_label": "V1"},
        {"id": "b", "kind": "draft", "model_key": "m", "prompt": "y", "variant_label": "V2"},
        {"id": "c", "kind": "draft", "model_key": "m", "prompt": "z", "variant_label": "V3"},
        {"id": "d", "kind": "draft", "model_key": "m", "prompt": "q", "variant_label": "orig"},
    ]
    script = (
        _js("variantBadges", "legacyTagger", "variantTag") +
        f"\nconsole.log(JSON.stringify(variantBadges({json.dumps(gens)},"
        " 'something else entirely', () => 'tags')));"
    )
    assert run_js(script) == {"a": "V1", "b": "V2", "c": "V3", "d": "orig"}


@needs_node
def test_a_second_enrich_carries_on_from_where_the_first_stopped():
    """V1 must name one prompt for the life of the session.

    The composer only ever shows the latest set, so restarting at V1 each
    round would leave an old card badged V1 pointing at a box that has since
    been overwritten with different words.
    """
    script = (
        _js("variantTag", "nextVariantLabels", "resetVariantSeq") +
        "\nlet variantSeq = {prose: 0, tags: 0};" +
        "\nconst modelByKey = () => ({prompt_style: 'tags'});" +
        "\nconst first = nextVariantLabels('tags', 3, false);" +
        "\nconst second = nextVariantLabels('tags', 3, false);" +
        "\nconsole.log(JSON.stringify([first, second]));"
    )
    assert run_js(script) == [["V1", "V2", "V3"], ["V4", "V5", "V6"]]


@needs_node
def test_the_two_dialects_are_numbered_apart():
    """T and P separate them, so each dialect keeps its own count."""
    script = (
        _js("variantTag", "nextVariantLabels") +
        "\nlet variantSeq = {prose: 0, tags: 0};" +
        "\nconsole.log(JSON.stringify([nextVariantLabels('tags', 2, true),"
        " nextVariantLabels('prose', 2, true)]));"
    )
    assert run_js(script) == [["T1", "T2"], ["P1", "P2"]]


@needs_node
def test_v_and_t_share_one_counter_within_a_dialect():
    """A tags round alone is V1..; add prose later and it must not reuse 1.

    The letter only says whether a second dialect is on screen. V1 and T1
    would otherwise be two different tag prompts in one session.
    """
    script = (
        _js("variantTag", "nextVariantLabels") +
        "\nlet variantSeq = {prose: 0, tags: 0};" +
        "\nconsole.log(JSON.stringify([nextVariantLabels('tags', 2, false),"
        " nextVariantLabels('tags', 2, true)]));"
    )
    assert run_js(script) == [["V1", "V2"], ["T3", "T4"]]


@needs_node
def test_reopening_a_session_resumes_from_the_numbers_already_on_screen():
    """Otherwise the next Enrich hands out names the cards already wear."""
    gens = [
        {"model_key": "m", "variant_label": "V1"},
        {"model_key": "m", "variant_label": "V5"},
        {"model_key": "m", "variant_label": "orig"},
    ]
    groups = [{"style": "tags", "variants": [{"label": "V7"}]}]
    script = (
        _js("variantTag", "nextVariantLabels", "resetVariantSeq") +
        "\nlet variantSeq = {prose: 0, tags: 0};" +
        "\nconst modelByKey = () => ({prompt_style: 'tags'});" +
        f"\nresetVariantSeq({json.dumps(gens)}, {json.dumps(groups)});" +
        "\nconsole.log(JSON.stringify(nextVariantLabels('tags', 2, false)));"
    )
    assert run_js(script) == ["V8", "V9"]


@needs_node
def test_a_round_made_from_one_prompt_is_still_badged():
    """The number names a prompt in the session, not a slot in the round.

    It was suppressed while numbering restarted each round, where the same
    "V1" on every card of a round said nothing.
    """
    same = [
        {"id": "a", "kind": "draft", "model_key": "m", "prompt": "x", "variant_label": "V4"},
        {"id": "b", "kind": "draft", "model_key": "m", "prompt": "x", "variant_label": "V4"},
    ]
    script = (
        _js("variantBadges", "legacyTagger", "variantTag") +
        f"\nconsole.log(JSON.stringify(variantBadges({json.dumps(same)}, '', () => 'tags')));"
    )
    assert run_js(script) == {"a": "V4", "b": "V4"}


@needs_node
def test_identical_variant_texts_stay_separate_boxes():
    """Two boxes holding the same words are still V1 and V2.

    The old path deduplicated by prompt text, so the second one vanished into
    the first and every later variant shifted down a number.
    """
    gens = [
        {"id": "a", "kind": "draft", "model_key": "m", "prompt": "same", "variant_label": "V1"},
        {"id": "b", "kind": "draft", "model_key": "m", "prompt": "same", "variant_label": "V2"},
    ]
    script = (
        _js("variantBadges", "legacyTagger", "variantTag") +
        f"\nconsole.log(JSON.stringify(variantBadges({json.dumps(gens)}, '', () => 'tags')));"
    )
    assert run_js(script) == {"a": "V1", "b": "V2"}


def _group_lengths(rows):
    script = (
        _js("groupByKey") +
        f"\nconst rows = {json.dumps(rows)};" +
        "\nconsole.log(JSON.stringify(groupByKey(rows, g => g.batch || g.created_at)"
        ".map(r => r.length)));"
    )
    return run_js(script)


@needs_node
def test_rounds_are_split_by_batch_not_by_timestamp():
    """One press fans out into several requests; they are one round.

    Their timestamps are not the batch: the requests of a single press land
    seconds apart whenever the server is busy.
    """
    assert _group_lengths([
        {"batch": "b1", "created_at": "t1"},
        {"batch": "b1", "created_at": "t2"},
        {"batch": "b2", "created_at": "t2"},
    ]) == [2, 1]


@needs_node
def test_two_batches_landing_in_one_second_come_back_interleaved():
    """Rows arrive ordered by timestamp then id, so the ids shuffle them.

    Grouping only neighbours shattered a pair of rounds into a round per
    card, and a round of one card is a round with nothing to badge.
    """
    assert _group_lengths([
        {"batch": "b1", "created_at": "t1"},
        {"batch": "b2", "created_at": "t1"},
        {"batch": "b1", "created_at": "t1"},
        {"batch": "b2", "created_at": "t1"},
    ]) == [2, 2]


@needs_node
def test_rows_without_a_batch_fall_back_to_their_timestamp():
    assert _group_lengths([
        {"batch": None, "created_at": "t1"},
        {"batch": None, "created_at": "t1"},
        {"batch": None, "created_at": "t2"},
    ]) == [2, 1]


@needs_node
def test_the_original_prompt_reaches_every_model_with_its_own_opener():
    """It looked like tag models never got the user's own words.

    They do -- but as one request per quality opener, so the same instruction
    becomes several different prompt strings and none of them is the
    instruction verbatim. That is what made the original unrecognisable to the
    old badge code, and made it read on screen as an extra unexplained
    variant instead of "orig".
    """
    models = [
        {"key": "bas", "quality_prefix": "anime, masterpiece", "prompt_style": "tags"},
        {"key": "illu", "quality_prefix": "masterpiece, best quality", "prompt_style": "tags"},
        {"key": "prose", "quality_prefix": "", "prompt_style": "prose"},
    ]
    script = (
        _js("asIsRequests", "originalVariant", "alreadyOpened") +
        f"\nconst MODELS = {json.dumps(models)};" +
        "\nconsole.log(JSON.stringify(asIsRequests('a girl in a park',"
        " new Set(['bas', 'illu', 'prose']))));"
    )
    out = run_js(script)
    assert sorted(k for r in out for k in r["model_keys"]) == ["bas", "illu", "prose"]
    assert all(r["variants"][0]["label"] == "orig" for r in out)
    by_key = {r["model_keys"][0]: r["variants"][0]["prompt"] for r in out}
    assert by_key["bas"] == "anime, masterpiece, a girl in a park"
    assert by_key["illu"] == "masterpiece, best quality, a girl in a park"
    assert by_key["prose"] == "a girl in a park"


@needs_node
def test_the_length_label_bands_a_prompt_against_the_budget():
    """Three states, because a bare character count says nothing.

    Nothing truncates a prompt (docs/decisions/PROMPT-LENGTH.md), so the label
    reports which of the three bands the prompt is in rather than filling a bar
    towards a limit: inside the budget, into the fading band, or past the point
    where the end stops reaching the image.
    """
    script = _js("budgetState") + """
    const b = {full: 300, fading: 600};
    console.log(JSON.stringify([
      budgetState(1, b), budgetState(300, b),
      budgetState(301, b), budgetState(600, b),
      budgetState(601, b), budgetState(5000, b),
    ]));
    """
    assert run_js(script) == ["ok", "ok", "warn", "warn", "over", "over"]


@needs_node
def test_the_label_follows_the_tightest_selected_budget():
    """A prompt written for the roomiest model renders faintly on the rest."""
    script = _js("tightestBudget") + """
    globalThis.MODELS = [
      {key: "a", prompt_budget: {full: 600, fading: 900}},
      {key: "b", prompt_budget: {full: 240, fading: 450}},
      {key: "c", prompt_budget: {full: 300, fading: 600}},
    ];
    globalThis.selectedModels = new Set(["a", "c"]);
    const two = tightestBudget();
    globalThis.selectedModels = new Set(["a", "b", "c"]);
    const three = tightestBudget();
    globalThis.selectedModels = new Set();
    console.log(JSON.stringify([two.full, three.full, tightestBudget()]));
    """
    assert run_js(script) == [300, 240, None]
