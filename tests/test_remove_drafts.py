"""Remove drafts: the drafts go, the lineage through them does not.

A child names its draft by id in parent_id, and the "from" link, the lineage
filter, the params chain and a Civitai post's settings all follow that id. So a
draft something was made from loses its image and keeps its row; only a draft
nothing surviving descends from is deleted outright.
"""

import pytest
from fastapi.testclient import TestClient

from app import main
from app.civitai import source_of
from app.sessions import Store


@pytest.fixture
def store(tmp_path):
    return Store(data_dir=tmp_path)


@pytest.fixture
def client(store, monkeypatch):
    monkeypatch.setattr(main, "store", store)
    with TestClient(main.app) as c:
        yield c


def make(store, session_id, kind="draft", parent=None, status="done", image=True):
    gen = store.create_generation(
        session_id=session_id, kind=kind, model_key="m", prompt=f"{kind} prompt",
        negative_prompt="", seed=7, width=256, height=256, steps=4,
        cfg=1.0, sampler=None, parent_id=parent.id if parent else None,
    )
    if image:
        store.save_image(gen, b"\x89PNG")
    store.set_status(gen.id, status)
    return store.get_generation(gen.id)


def test_a_draft_nothing_was_made_from_is_deleted(store):
    s = store.create_session("x")
    d = make(store, s.id)
    done = store.prune_drafts(s.id)
    assert done["deleted"] == [d.id] and done["pruned"] == []
    assert store.get_generation(d.id) is None
    assert not store.image_path(d).exists()
    assert done["files"] == [f"{s.id}/{d.id}.png"]


def test_a_finalised_draft_keeps_its_row_and_loses_its_image(store):
    s = store.create_session("x")
    d = make(store, s.id)
    f = make(store, s.id, kind="faithful", parent=d)
    done = store.prune_drafts(s.id)
    assert done["pruned"] == [d.id] and done["deleted"] == []
    stub = store.get_generation(d.id)
    assert stub.pruned_at and stub.seed == 7 and stub.prompt == "draft prompt"
    assert not store.image_path(d).exists()
    # the final is untouched and still resolves its parent
    assert store.image_path(f).is_file()
    assert store.get_generation(f.id).parent_id == d.id


def test_a_chain_of_drafts_under_a_final_is_kept_whole(store):
    # draft -> More like this -> Finalize: both drafts are on the way to the
    # final, and cutting either would end the chain there
    s = store.create_session("x")
    d1 = make(store, s.id)
    d2 = make(store, s.id, parent=d1)
    f = make(store, s.id, kind="final", parent=d2)
    done = store.prune_drafts(s.id)
    assert sorted(done["pruned"]) == sorted([d1.id, d2.id])
    session = store.get_session(s.id)
    by_id = {g.id: g for g in session.generations}
    chain, cur = [], by_id[f.id]
    while cur:
        chain.append(cur.id)
        cur = by_id.get(cur.parent_id)
    assert chain == [f.id, d2.id, d1.id]


def test_a_chain_of_drafts_with_nothing_else_below_goes_entirely(store):
    s = store.create_session("x")
    d1 = make(store, s.id)
    d2 = make(store, s.id, parent=d1)
    d3 = make(store, s.id, parent=d2)
    done = store.prune_drafts(s.id)
    assert done["deleted"] == [d1.id, d2.id, d3.id] and done["pruned"] == []
    assert store.get_session(s.id).generations == []


def test_a_sibling_branch_is_deleted_while_the_finalised_one_is_kept(store):
    s = store.create_session("x")
    d = make(store, s.id)
    dead_end = make(store, s.id, parent=d)
    kept_branch = make(store, s.id, parent=d)
    make(store, s.id, kind="final", parent=kept_branch)
    done = store.prune_drafts(s.id)
    assert done["deleted"] == [dead_end.id]
    assert sorted(done["pruned"]) == sorted([d.id, kept_branch.id])


def test_a_draft_a_queued_render_still_needs_is_left_alone(store):
    # a queued faithful reads its parent's PNG when it starts
    s = store.create_session("x")
    for status in ("pending", "running"):
        d = make(store, s.id)
        make(store, s.id, kind="faithful", parent=d, status=status, image=False)
        done = store.prune_drafts(s.id)
        assert d.id in [k["id"] for k in done["kept"]]
        assert store.get_generation(d.id).pruned_at is None
        assert store.image_path(d).is_file()


def test_a_draft_marked_published_is_removed_like_any_other(store):
    # drafts cannot be published; a row that says otherwise predates that rule
    # and is no reason to keep a preview
    s = store.create_session("x")
    d = make(store, s.id)
    store.mark_published(d.id, 123)
    done = store.prune_drafts(s.id)
    assert done["deleted"] == [d.id] and done["kept"] == []


def test_queued_drafts_and_other_kinds_are_not_candidates(store):
    s = store.create_session("x")
    queued = make(store, s.id, status="pending", image=False)
    final = make(store, s.id, kind="final")
    done = store.prune_drafts(s.id)
    assert done["deleted"] == [] and done["pruned"] == []
    assert store.get_generation(queued.id) and store.get_generation(final.id)


def test_a_failed_draft_is_removed_too(store):
    s = store.create_session("x")
    d = make(store, s.id, status="error", image=False)
    assert store.prune_drafts(s.id)["deleted"] == [d.id]


def test_other_sessions_are_never_touched(store):
    a, b = store.create_session("a"), store.create_session("b")
    da, db = make(store, a.id), make(store, b.id)
    assert store.prune_drafts(a.id)["deleted"] == [da.id]
    assert store.get_generation(db.id) is not None


def test_there_is_no_install_wide_removal(store):
    # drafts are removed one session at a time, never across the install
    with pytest.raises(TypeError):
        store.prune_drafts()
    with pytest.raises(TypeError):
        store.survey_drafts()


def test_the_survey_changes_nothing(store):
    s = store.create_session("x")
    d = make(store, s.id)
    f = make(store, s.id, kind="final", parent=d)
    leaf = make(store, s.id)
    found = store.survey_drafts(s.id)
    assert found["deleted"] == [leaf.id] and found["pruned"] == [d.id]
    assert found["bytes"] == 8
    assert store.get_generation(d.id).pruned_at is None
    assert store.image_path(d).is_file() and store.image_path(leaf).is_file()


def test_a_second_run_finds_nothing(store):
    s = store.create_session("x")
    d = make(store, s.id)
    make(store, s.id, kind="final", parent=d)
    store.prune_drafts(s.id)
    again = store.survey_drafts(s.id)
    assert again["deleted"] == [] and again["pruned"] == [] and again["files"] == []


def test_a_stub_goes_once_what_it_was_kept_for_is_deleted(store):
    s = store.create_session("x")
    d = make(store, s.id)
    f = make(store, s.id, kind="final", parent=d)
    up = make(store, s.id, kind="upscale", parent=f)
    store.prune_drafts(s.id)
    store.delete_generation(up.id)
    assert store.get_generation(d.id) is not None   # still the final's parent
    store.delete_generation(f.id)
    assert store.get_generation(d.id) is None


def test_a_stub_is_not_damage_to_the_storage_sweep(store):
    # It is a done row with no image, which is exactly what the sweep deletes.
    # Deleting it there would undo the whole point of keeping it.
    s = store.create_session("x")
    d = make(store, s.id)
    f = make(store, s.id, kind="final", parent=d)
    store.prune_drafts(s.id)
    found = store.survey_storage()
    assert found["missing"] == [] and found["blocked"] is None
    store.cleanup_storage()
    assert store.get_generation(d.id) is not None


def test_stubs_alone_do_not_prove_the_image_folder(store):
    # a done row with no image would otherwise count as a missing one, and an
    # install left with nothing but stubs would read as a misconfigured folder
    s = store.create_session("x")
    d = make(store, s.id)
    f = make(store, s.id, kind="final", parent=d)
    store.prune_drafts(s.id)
    store.image_path(f).unlink()
    assert store.survey_storage()["missing"][0]["id"] == f.id
    assert store.survey_storage()["blocked"] is not None


def test_stats_stop_counting_a_pruned_draft(store):
    s = store.create_session("x")
    d = make(store, s.id)
    make(store, s.id, kind="final", parent=d)
    assert store.session_stats(s.id)["kinds"] == {"draft": 1, "final": 1}
    store.prune_drafts(s.id)
    assert store.session_stats(s.id)["kinds"] == {"final": 1}


def test_a_post_from_an_upscale_still_finds_its_settings(store):
    s = store.create_session("x")
    d = make(store, s.id)
    f = make(store, s.id, kind="final", parent=d)
    up = make(store, s.id, kind="upscale", parent=f)
    store.prune_drafts(s.id)
    ancestors = {g.id: g for g in store.get_session(s.id).generations}
    assert source_of(ancestors[up.id], ancestors).id == f.id


# -- over the API ---------------------------------------------------------------


def test_the_endpoints_survey_then_remove(client, store):
    sid = client.post("/api/sessions", json={"instruction": "a cat"}).json()["id"]
    d = make(store, sid)
    f = make(store, sid, kind="final", parent=d)
    leaf = make(store, sid)

    found = client.get(f"/api/drafts/remove?session_id={sid}").json()
    assert found["deleted"] == [leaf.id] and found["pruned"] == [d.id]
    assert sorted(found["files"]) == sorted([f"{sid}/{d.id}.png", f"{sid}/{leaf.id}.png"])
    assert store.image_path(leaf).is_file()

    done = client.post(f"/api/drafts/remove?session_id={sid}").json()
    assert done["deleted"] == [leaf.id] and done["pruned"] == [d.id]

    gens = client.get(f"/api/sessions/{sid}").json()["generations"]
    stub = next(g for g in gens if g["id"] == d.id)
    assert stub["pruned_at"] and stub["seed"] == 7
    assert next(g for g in gens if g["id"] == f.id)["parent_id"] == d.id
    assert client.get(f"/images/{sid}/{d.id}.png").status_code == 404


@pytest.mark.parametrize("method", ["get", "post"])
def test_the_endpoints_need_a_session(client, store, method):
    sid = client.post("/api/sessions", json={"instruction": "a cat"}).json()["id"]
    d = make(store, sid)
    assert getattr(client, method)("/api/drafts/remove").status_code == 422
    assert getattr(client, method)("/api/drafts/remove?session_id=").status_code == 422
    assert store.image_path(d).is_file()


@pytest.mark.parametrize("method", ["get", "post"])
def test_the_endpoints_refuse_an_unknown_session(client, method):
    assert getattr(client, method)("/api/drafts/remove?session_id=nope").status_code == 404


def test_a_pruned_draft_cannot_be_acted_on(client, store):
    sid = client.post("/api/sessions", json={"instruction": "a cat"}).json()["id"]
    d = make(store, sid)
    make(store, sid, kind="final", parent=d)
    store.prune_drafts(sid)
    r = client.post(f"/api/generations/{d.id}/finalize", json={"mode": "faithful"})
    assert r.status_code == 404 and "removed" in r.json()["detail"]


def test_a_failed_child_that_needs_the_image_says_why_it_cannot_retry(client, store):
    sid = client.post("/api/sessions", json={"instruction": "a cat"}).json()["id"]
    d = make(store, sid)
    faithful = make(store, sid, kind="faithful", parent=d, status="error", image=False)
    store.prune_drafts(sid)
    r = client.post(f"/api/generations/{faithful.id}/retry")
    assert r.status_code == 400 and "removed" in r.json()["detail"]
    assert store.get_generation(faithful.id).status == "error"


def test_a_failed_plain_final_still_retries_from_a_pruned_draft(client, store, monkeypatch):
    # txt2img from the draft's prompt and seed, both of which the stub kept
    monkeypatch.setattr(main.job_queue, "put_nowait", lambda _id: None)
    sid = client.post("/api/sessions", json={"instruction": "a cat"}).json()["id"]
    d = make(store, sid)
    final = make(store, sid, kind="final", parent=d, status="error", image=False)
    store.prune_drafts(sid)
    assert client.post(f"/api/generations/{final.id}/retry").status_code == 200
