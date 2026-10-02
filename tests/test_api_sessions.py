"""The HTTP contract for a session, including one that has only prompts.

The store tests cover what is written down; these cover what the app promises
over the wire -- that Enrich alone creates something the picker can list and
loadSession can restore.
"""

import pytest
from fastapi.testclient import TestClient

from app import main
from app.sessions import Store

COMPOSER = {
    "models": ["bas-anima"],
    "groups": [{
        "style": "tags",
        "usesNeg": False,
        "keys": ["bas-anima"],
        "variants": [{"prompt": "1girl, park", "negative_prompt": ""}],
    }],
}


@pytest.fixture
def client(tmp_path, monkeypatch):
    """The app against a throwaway store, so no test writes to data/."""
    monkeypatch.setattr(main, "store", Store(tmp_path))
    with TestClient(main.app) as c:
        yield c


def session_id(client, instruction="a cat"):
    return client.post("/api/sessions", json={"instruction": instruction}).json()["id"]


def test_prompts_round_trip_without_a_single_generation(client):
    sid = session_id(client)
    resp = client.put(f"/api/sessions/{sid}/prompts",
                      json={"instruction": "a cat on a roof", "prompts": COMPOSER})
    assert resp.status_code == 200
    body = client.get(f"/api/sessions/{sid}").json()
    assert body["prompts"] == COMPOSER
    assert body["generations"] == []


def test_saving_prompts_updates_the_name_the_picker_shows(client):
    sid = session_id(client, "a cat")
    client.put(f"/api/sessions/{sid}/prompts",
               json={"instruction": "a cat on a roof", "prompts": COMPOSER})
    listed = {s["id"]: s["instruction"] for s in client.get("/api/sessions").json()}
    assert listed[sid] == "a cat on a roof"


def test_a_prompts_only_session_is_listed_like_any_other(client):
    sid = session_id(client)
    client.put(f"/api/sessions/{sid}/prompts",
               json={"instruction": "a cat", "prompts": COMPOSER})
    assert any(s["id"] == sid for s in client.get("/api/sessions").json())


def test_saving_prompts_for_a_session_that_does_not_exist_is_a_404(client):
    resp = client.put("/api/sessions/nope/prompts",
                      json={"instruction": "x", "prompts": COMPOSER})
    assert resp.status_code == 404


def test_a_session_saved_before_the_feature_reads_back_without_prompts(client):
    sid = session_id(client)
    assert client.get(f"/api/sessions/{sid}").json()["prompts"] is None


def listed(client, sid):
    return next(x for x in client.get("/api/sessions").json() if x["id"] == sid)


def test_a_session_can_be_renamed(client):
    sid = session_id(client)
    resp = client.patch(f"/api/sessions/{sid}", json={"name": "  Park series  "})
    assert resp.status_code == 200 and resp.json()["name"] == "Park series"
    assert listed(client, sid)["name"] == "Park series"
    assert client.get(f"/api/sessions/{sid}").json()["name"] == "Park series"


def test_a_rename_leaves_the_prompt_alone(client):
    # the instruction is the composer's text, not a label: renaming must not
    # put the name in the prompt box
    sid = session_id(client, "a cat")
    client.patch(f"/api/sessions/{sid}", json={"name": "Cats"})
    assert client.get(f"/api/sessions/{sid}").json()["instruction"] == "a cat"


def test_a_name_survives_enrich_and_generate_saving_the_composer(client):
    # those saves rewrite the instruction, which used to be the only name
    sid = session_id(client)
    client.patch(f"/api/sessions/{sid}", json={"name": "Cats"})
    client.put(f"/api/sessions/{sid}/prompts",
               json={"instruction": "a dog instead", "prompts": COMPOSER})
    s = listed(client, sid)
    assert s["name"] == "Cats" and s["instruction"] == "a dog instead"


def test_an_empty_name_goes_back_to_the_prompt(client):
    sid = session_id(client)
    client.patch(f"/api/sessions/{sid}", json={"name": "Cats"})
    assert client.patch(f"/api/sessions/{sid}", json={"name": "   "}).json()["name"] is None
    assert listed(client, sid)["name"] is None


def test_a_new_session_has_no_name(client):
    assert listed(client, session_id(client))["name"] is None


def test_renaming_a_session_that_does_not_exist_is_a_404(client):
    assert client.patch("/api/sessions/nope", json={"name": "x"}).status_code == 404


def test_a_name_has_a_length_limit(client):
    sid = session_id(client)
    assert client.patch(f"/api/sessions/{sid}", json={"name": "x" * 201}).status_code == 422


def test_health_reports_a_stalled_renderer_separately(client, monkeypatch):
    """draw_things stays true: the server answers, it just cannot render."""
    async def up(): return True
    monkeypatch.setattr(main.dt, "is_up", up)
    monkeypatch.setattr(main.llm, "is_up", up)
    main.dt.stalled = True
    try:
        body = client.get("/api/health").json()
    finally:
        main.dt.stalled = False
    assert body["draw_things"] is True
    assert body["draw_things_stalled"] is True
