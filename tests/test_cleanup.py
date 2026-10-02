"""Deleting a card takes its files; cleanup takes what is left over.

Two directions of damage: a finished row with no image draws as a broken card,
and a file no row owns is invisible and only occupies the disk. A row counts as
finished when it says done or error -- neither will ever show a picture -- and
queued or running rows are never candidates, because a job writes its image
before it sets its status.
"""

import pytest
from fastapi.testclient import TestClient

from app import main
from app.sessions import StorageMisconfigured, Store


@pytest.fixture
def store(tmp_path):
    return Store(data_dir=tmp_path)


@pytest.fixture
def client(store, monkeypatch):
    monkeypatch.setattr(main, "store", store)
    with TestClient(main.app) as c:
        yield c


def make(store, session_id, status="done", kind="draft", image=True):
    gen = store.create_generation(
        session_id=session_id, kind=kind, model_key="m", prompt="p",
        negative_prompt="", seed=1, width=256, height=256, steps=4,
        cfg=1.0, sampler=None,
    )
    if image:
        store.save_image(gen, b"\x89PNG")
    store.set_status(gen.id, status)
    return store.get_generation(gen.id)


def orphan(store, session_id, name):
    folder = store.data_dir / "sessions" / session_id
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_bytes(b"\x89PNG")
    return path


# -- the cross on a card -------------------------------------------------------


def test_deleting_a_card_over_the_api_takes_its_files(client, store):
    sid = client.post("/api/sessions", json={"instruction": "a cat"}).json()["id"]
    gen = make(store, sid)
    store.init_path(gen).write_bytes(b"\x89PNG init")
    assert store.image_path(gen).is_file()

    assert client.delete(f"/api/generations/{gen.id}").status_code == 200

    assert store.get_generation(gen.id) is None
    assert not store.image_path(gen).exists()
    # the pre-composited init of a reference edit is the card's too
    assert not store.init_path(gen).exists()


def test_a_running_card_is_not_deleted_from_under_its_job(client, store):
    sid = client.post("/api/sessions", json={"instruction": "a cat"}).json()["id"]
    gen = make(store, sid, status="running")
    assert client.delete(f"/api/generations/{gen.id}").status_code == 400
    assert store.image_path(gen).is_file()


# -- which rows are candidates -------------------------------------------------


def test_a_done_row_whose_image_was_deleted(store):
    s = store.create_session("x")
    gone = make(store, s.id)
    store.image_path(gone).unlink()
    kept = make(store, s.id)

    found = store.survey_storage()
    assert [r["id"] for r in found["missing"]] == [gone.id]
    assert found["orphans"] == []
    assert store.get_generation(kept.id) is not None


def test_a_failed_row_counts_too(store):
    """The card that started this: a render that returned no image at all."""
    s = store.create_session("x")
    make(store, s.id)  # the healthy card that proves the folder is the right one
    failed = make(store, s.id, status="error", image=False)
    assert [r["id"] for r in store.survey_storage()["missing"]] == [failed.id]


def test_a_failed_row_survives_until_cleanup_is_actually_run(client, store):
    sid = client.post("/api/sessions", json={"instruction": "x"}).json()["id"]
    make(store, sid)
    failed = make(store, sid, status="error", image=False)
    # listing it, twice, changes nothing
    client.get("/api/cleanup")
    client.get("/api/cleanup")
    assert store.get_generation(failed.id) is not None
    client.post("/api/cleanup")
    assert store.get_generation(failed.id) is None


def test_queued_and_running_rows_are_never_candidates(store):
    s = store.create_session("x")
    make(store, s.id)
    # a held job stays pending for as long as Draw Things is away
    make(store, s.id, status="pending", image=False)
    make(store, s.id, status="running", image=False)
    assert store.survey_storage()["missing"] == []


# -- which files are candidates ------------------------------------------------


def test_files_no_card_owns(store):
    s = store.create_session("x")
    make(store, s.id)
    orphan(store, s.id, "aaaaaaaaaaaa.png")
    orphan(store, s.id, "bbbbbbbbbbbb_init.png")
    orphan(store, s.id, "cccccccccccc.mp4")

    names = sorted(p.name for p in store.survey_storage()["orphans"])
    assert names == ["aaaaaaaaaaaa.png", "bbbbbbbbbbbb_init.png", "cccccccccccc.mp4"]


def test_a_file_this_app_did_not_write_is_never_a_candidate(store):
    s = store.create_session("x")
    gen = make(store, s.id)
    note = store.image_path(gen).parent / "notes.txt"
    note.write_text("keep me")
    assert store.survey_storage()["orphans"] == []
    assert note.is_file()


def test_a_file_whose_row_lives_in_another_session_is_not_an_orphan(store):
    """Ownership is decided against the whole database, not the swept folder."""
    here = store.create_session("here")
    there = store.create_session("there")
    stray = make(store, there.id)
    moved = store.data_dir / "sessions" / here.id
    moved.mkdir(parents=True, exist_ok=True)
    (moved / f"{stray.id}.png").write_bytes(b"\x89PNG")
    assert store.survey_storage(here.id)["orphans"] == []


# -- the wrong folder is not damage --------------------------------------------


def test_a_misconfigured_image_folder_blocks_the_whole_sweep(client, store):
    """DRAFTPOOL_DATA_DIR pointed somewhere else must not empty the database."""
    sid = client.post("/api/sessions", json={"instruction": "x"}).json()["id"]
    kept = [make(store, sid).id for _ in range(3)]
    for gen_id in kept:
        store.image_path(store.get_generation(gen_id)).unlink()

    survey = client.get("/api/cleanup").json()
    assert "DRAFTPOOL_DATA_DIR" in survey["blocked"]

    resp = client.post("/api/cleanup")
    assert resp.status_code == 400
    assert "DRAFTPOOL_DATA_DIR" in resp.json()["detail"]
    for gen_id in kept:
        assert store.get_generation(gen_id) is not None

    with pytest.raises(StorageMisconfigured):
        store.cleanup_storage()


def test_one_surviving_image_is_enough_to_prove_the_folder(client, store):
    sid = client.post("/api/sessions", json={"instruction": "x"}).json()["id"]
    make(store, sid)
    gone = [make(store, sid).id for _ in range(3)]
    for gen_id in gone:
        store.image_path(store.get_generation(gen_id)).unlink()

    assert client.get("/api/cleanup").json()["blocked"] is None
    assert sorted(client.post("/api/cleanup").json()["rows"]) == sorted(gone)


def test_an_install_whose_renders_all_failed_is_not_blocked(client, store):
    """Nothing to prove the folder with, and nothing a sweep could destroy."""
    sid = client.post("/api/sessions", json={"instruction": "x"}).json()["id"]
    failed = [make(store, sid, status="error", image=False).id for _ in range(3)]
    assert client.get("/api/cleanup").json()["blocked"] is None
    assert sorted(client.post("/api/cleanup").json()["rows"]) == sorted(failed)


def test_an_empty_install_is_not_blocked(client, store):
    """No rows, nothing to prove, nothing to do."""
    body = client.get("/api/cleanup").json()
    assert body["blocked"] is None and body["missing"] == []
    assert client.post("/api/cleanup").json() == {"rows": [], "files": []}


# -- doing it ------------------------------------------------------------------


def test_cleanup_removes_both_and_reports_what_it_did(client, store):
    sid = client.post("/api/sessions", json={"instruction": "a cat"}).json()["id"]
    gone = make(store, sid)
    store.image_path(gone).unlink()
    store.init_path(gone).write_bytes(b"\x89PNG init")
    kept = make(store, sid)
    stray = orphan(store, sid, "dddddddddddd.png")

    preview = client.get("/api/cleanup").json()
    assert [r["id"] for r in preview["missing"]] == [gone.id]
    # the broken row names what it will take with it, the orphan names itself
    assert preview["missing"][0]["files"] == [f"{sid}/{gone.id}_init.png"]
    assert preview["files"] == [f"{sid}/dddddddddddd.png"]
    # looking changes nothing
    assert store.get_generation(gone.id) is not None

    done = client.post("/api/cleanup").json()
    assert done["rows"] == [gone.id]
    assert done["files"] == [f"{sid}/{gone.id}_init.png", f"{sid}/dddddddddddd.png"]
    assert store.get_generation(gone.id) is None
    # the row went through delete_generation, so its leftovers went too
    assert not store.init_path(gone).exists()
    assert not stray.exists()
    # and the intact card is untouched
    assert store.get_generation(kept.id) is not None
    assert store.image_path(kept).is_file()


def test_cleanup_can_be_held_to_one_session(client, store):
    mine = client.post("/api/sessions", json={"instruction": "mine"}).json()["id"]
    other = client.post("/api/sessions", json={"instruction": "other"}).json()["id"]
    broken = {}
    for sid in (mine, other):
        make(store, sid)
        gen = make(store, sid)
        store.image_path(gen).unlink()
        broken[sid] = gen
        orphan(store, sid, "eeeeeeeeeeee.png")

    scoped = client.get("/api/cleanup", params={"session_id": mine}).json()
    assert [r["id"] for r in scoped["missing"]] == [broken[mine].id]
    assert scoped["files"] == [f"{mine}/eeeeeeeeeeee.png"]

    client.post("/api/cleanup", params={"session_id": mine})
    assert store.get_generation(broken[mine].id) is None
    # the other session is left exactly as it was, both halves of it
    assert store.get_generation(broken[other].id) is not None
    assert (store.data_dir / "sessions" / other / "eeeeeeeeeeee.png").is_file()


def test_cleanup_on_a_healthy_install_does_nothing(client, store):
    sid = client.post("/api/sessions", json={"instruction": "a cat"}).json()["id"]
    make(store, sid)
    body = client.get("/api/cleanup").json()
    assert body["missing"] == [] and body["files"] == []
    assert client.post("/api/cleanup").json() == {"rows": [], "files": []}


def test_no_more_and_no_less(client, store):
    """One sweep over one of everything: exactly the damage, nothing else."""
    sid = client.post("/api/sessions", json={"instruction": "everything"}).json()["id"]

    doomed_rows, doomed_files, safe_rows, safe_files = [], [], [], []

    # damage: a finished card with no image, either way it finished
    for status in ("done", "error"):
        gen = make(store, sid, status=status)
        store.image_path(gen).unlink()
        doomed_rows.append(gen.id)
    # damage: a leftover init beside one of them, and a file with no row at all
    init = store.init_path(store.get_generation(doomed_rows[0]))
    init.write_bytes(b"\x89PNG init")
    doomed_files += [init, orphan(store, sid, "ffffffffffff.png")]

    # everything else in the folder has a reason to survive
    healthy = make(store, sid)
    safe_rows.append(healthy.id)
    safe_files.append(store.image_path(healthy))
    for status in ("pending", "running"):                 # still to come
        safe_rows.append(make(store, sid, status=status, image=False).id)
    note = store.data_dir / "sessions" / sid / "notes.txt"                # not ours
    note.write_text("keep me")
    safe_files.append(note)

    before = {p.name for p in (store.data_dir / "sessions" / sid).iterdir()}
    done = client.post("/api/cleanup").json()

    assert sorted(done["rows"]) == sorted(doomed_rows)
    assert sorted(done["files"]) == sorted(store.relative(p) for p in doomed_files)
    for gen_id in doomed_rows:
        assert store.get_generation(gen_id) is None
    for path in doomed_files:
        assert not path.exists()
    for gen_id in safe_rows:
        assert store.get_generation(gen_id) is not None
    for path in safe_files:
        assert path.is_file()
    # and nothing outside those two lists moved
    after = {p.name for p in (store.data_dir / "sessions" / sid).iterdir()}
    assert before - after == {p.name for p in doomed_files}


# -- where the images live -----------------------------------------------------


def test_storage_reports_the_path_and_what_is_in_it(client, store):
    sid = client.post("/api/sessions", json={"instruction": "a cat"}).json()["id"]
    gen = make(store, sid)
    (store.image_path(gen).parent / "notes.txt").write_text("not an image")

    body = client.get("/api/storage").json()
    assert body["data_dir"] == str(store.data_dir)
    assert body["images_dir"] == str(store.data_dir / "sessions")
    # what this app wrote, not everything in the folder
    assert body["files"] == 1
    assert body["bytes"] == store.image_path(gen).stat().st_size
