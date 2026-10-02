from pathlib import Path

from app.sessions import Store


def make_store(tmp_path: Path) -> Store:
    return Store(data_dir=tmp_path)


def test_session_and_generation_roundtrip(tmp_path):
    store = make_store(tmp_path)
    s = store.create_session("a cat")
    gen = store.create_generation(
        session_id=s.id, kind="draft", model_key="flux2-klein",
        prompt="a cat, detailed", negative_prompt="", seed=42,
        width=256, height=256, steps=6, cfg=1.0, sampler=None,
    )
    assert gen.status == "pending"
    store.set_status(gen.id, "done")
    detail = store.get_session(s.id)
    assert detail is not None
    assert len(detail.generations) == 1
    assert detail.generations[0].status == "done"


def test_lineage_parent_chain(tmp_path):
    store = make_store(tmp_path)
    s = store.create_session("x")
    draft = store.create_generation(
        session_id=s.id, kind="draft", model_key="m", prompt="p",
        negative_prompt="", seed=1, width=256, height=256, steps=4,
        cfg=1.0, sampler=None,
    )
    final = store.create_generation(
        session_id=s.id, kind="final", model_key="m", prompt="p",
        negative_prompt="", seed=1, width=1024, height=1024, steps=16,
        cfg=1.0, sampler=None, parent_id=draft.id,
    )
    up = store.create_generation(
        session_id=s.id, kind="upscale", model_key="m", prompt="p",
        negative_prompt="", seed=1, width=2048, height=2048, steps=12,
        cfg=1.0, sampler=None, parent_id=final.id,
    )
    assert store.get_generation(up.id).parent_id == final.id
    assert store.get_generation(final.id).parent_id == draft.id
    # final keeps the draft's seed and prompt for reproducibility
    assert store.get_generation(final.id).seed == draft.seed


def _video_gen(store, session_id, **kw):
    return store.create_generation(
        session_id=session_id, kind="video", model_key="fusionx",
        prompt="hair drifts", negative_prompt="", seed=7,
        width=832, height=480, steps=8, cfg=1.0, sampler=None,
        num_frames=33, fps=16, **kw,
    )


def test_video_generation_roundtrip(tmp_path):
    store = make_store(tmp_path)
    s = store.create_session("x")
    gen = _video_gen(store, s.id)
    got = store.get_generation(gen.id)
    assert got.kind == "video"
    assert got.num_frames == 33
    assert got.fps == 16
    # extension is unknown until the clip is actually assembled
    assert got.video_ext is None
    store.set_video_ext(gen.id, "mp4")
    assert store.get_generation(gen.id).video_ext == "mp4"


def test_session_stats_counts_by_kind_and_model(tmp_path):
    store = make_store(tmp_path)
    s = store.create_session("x")
    other = store.create_session("y")

    def draft(session_id, model, status="done", secs=10.0):
        g = store.create_generation(
            session_id=session_id, kind="draft", model_key=model,
            prompt="p", negative_prompt="", seed=1,
            width=256, height=256, steps=4, cfg=1.0, sampler=None,
        )
        store.set_status(g.id, status, duration_s=secs if status == "done" else None)
        return g

    draft(s.id, "flux2-klein"); draft(s.id, "flux2-klein")
    draft(s.id, "lustrousmix-anima")
    fin = store.create_generation(
        session_id=s.id, kind="final", model_key="flux2-klein",
        prompt="p", negative_prompt="", seed=2,
        width=1024, height=1024, steps=20, cfg=1.0, sampler=None,
    )
    store.set_status(fin.id, "done", duration_s=30.0)
    draft(s.id, "flux2-klein", status="error")
    draft(s.id, "flux2-klein", status="pending")
    draft(other.id, "flux2-klein")   # a different session must not leak in

    st = store.session_stats(s.id)
    assert st["done"] == 4
    assert st["failed"] == 1
    assert st["pending"] == 1
    assert st["kinds"] == {"draft": 3, "final": 1}
    # a queued job has contributed no time, and a failed one is not an image
    assert st["seconds"] == 60
    assert [m["key"] for m in st["models"]] == ["flux2-klein", "lustrousmix-anima"]
    assert st["models"][0]["kinds"] == {"draft": 2, "final": 1}
    assert st["models"][1]["total"] == 1


def test_stats_do_not_credit_a_checkpoint_for_a_2x_fast_pass(tmp_path):
    # 2x fast stores the parent's model_key as provenance, but Real-ESRGAN is a
    # separate binary: crediting the checkpoint with the image AND its seconds
    # would overstate what that model cost and produced
    store = make_store(tmp_path)
    s = store.create_session("x")
    for kind, secs in (("final", 30.0), ("esrgan", 4.0)):
        g = store.create_generation(
            session_id=s.id, kind=kind, model_key="krea2-turbo",
            prompt="p", negative_prompt="", seed=1,
            width=1024, height=1024, steps=20, cfg=1.0, sampler=None,
        )
        store.set_status(g.id, "done", duration_s=secs)

    st = store.session_stats(s.id)
    by_key = {m["key"]: m for m in st["models"]}
    assert by_key["krea2-turbo"]["total"] == 1
    assert by_key["krea2-turbo"]["seconds"] == 30
    assert by_key["real-esrgan"]["kinds"] == {"esrgan": 1}
    assert by_key["real-esrgan"]["seconds"] == 4
    # the split must still account for everything
    assert sum(m["total"] for m in st["models"]) == st["done"] == 2


def test_running_video_generations_finds_only_rendering_clips(tmp_path):
    # read at shutdown to warn about a render Draw Things cannot be told to
    # stop; a still or an already-finished clip is not that
    store = make_store(tmp_path)
    s = store.create_session("x")
    rendering = _video_gen(store, s.id)
    store.set_status(rendering.id, "running")
    done = _video_gen(store, s.id)
    store.set_status(done.id, "done")
    still = store.create_generation(
        session_id=s.id, kind="draft", model_key="flux2-klein",
        prompt="p", negative_prompt="", seed=1,
        width=256, height=256, steps=4, cfg=1.0, sampler=None,
    )
    store.set_status(still.id, "running")

    assert store.running_video_generations() == [rendering.id]


def test_set_dimensions_records_actual_output_size(tmp_path):
    # Draw Things snaps requests to its own buckets: 832x480 comes back 832x448,
    # so the requested size must not be left in the record as fact
    store = make_store(tmp_path)
    s = store.create_session("x")
    gen = _video_gen(store, s.id)
    assert gen.height == 480
    store.set_dimensions(gen.id, 832, 448)
    got = store.get_generation(gen.id)
    assert (got.width, got.height) == (832, 448)


def test_video_path_follows_extension(tmp_path):
    store = make_store(tmp_path)
    s = store.create_session("x")
    gen = _video_gen(store, s.id)
    # defaults to mp4 before the container is known
    assert store.video_path(gen).name == f"{gen.id}.mp4"
    assert store.video_path(gen, "webp").name == f"{gen.id}.webp"


def test_delete_generation_removes_clip_and_poster(tmp_path):
    store = make_store(tmp_path)
    s = store.create_session("x")
    gen = _video_gen(store, s.id)
    store.save_image(gen, b"\x89PNG poster")
    store.set_video_ext(gen.id, "webp")
    clip = store.video_path(store.get_generation(gen.id))
    clip.write_bytes(b"RIFF fake webp")
    assert clip.is_file()
    store.delete_generation(gen.id)
    assert store.get_generation(gen.id) is None
    # both the poster frame and the clip must go, whichever container was used
    assert not clip.exists()
    assert not store.image_path(gen).exists()


def test_image_save_path(tmp_path):
    store = make_store(tmp_path)
    s = store.create_session("x")
    gen = store.create_generation(
        session_id=s.id, kind="draft", model_key="m", prompt="p",
        negative_prompt="", seed=1, width=256, height=256, steps=4,
        cfg=1.0, sampler=None,
    )
    path = store.save_image(gen, b"\x89PNG fake")
    assert path.is_file()
    assert path == tmp_path / "sessions" / s.id / f"{gen.id}.png"


def test_prompts_survive_a_session_with_no_generations(tmp_path):
    store = Store(tmp_path)
    s = store.create_session("a cat")
    state = {"models": ["bas-anima"], "groups": [
        {"style": "tags", "usesNeg": False, "keys": ["bas-anima"],
         "variants": [{"prompt": "1girl", "negative_prompt": ""}]}
    ]}
    assert store.save_prompts(s.id, "a cat on a roof", state) is True
    got = store.get_session(s.id)
    assert got.instruction == "a cat on a roof"  # the picker's label follows the edit
    assert got.prompts == state
    assert got.generations == []


def test_saving_prompts_for_an_unknown_session_reports_it(tmp_path):
    assert Store(tmp_path).save_prompts("nope", "x", {}) is False


def test_a_session_saved_before_prompts_existed_reads_back_as_none(tmp_path):
    store = Store(tmp_path)
    s = store.create_session("older session")
    assert store.get_session(s.id).prompts is None


def test_render_rate_is_seconds_per_step_at_full_strength(tmp_path):
    store = make_store(tmp_path)
    s = store.create_session("x")
    # the real measurements that produced this: qwen-edit at 1024 runs about
    # 20 s/step whatever the step count it is given
    for steps, secs in ((8, 174.9), (16, 316.6), (16, 330.1), (24, 484.7)):
        gen = store.create_generation(
            session_id=s.id, kind="edit", model_key="qwen-edit", prompt="p",
            negative_prompt="", seed=1, width=1024, height=1024, steps=steps,
            cfg=1.0, sampler=None, strength=1.0,
        )
        store.set_status(gen.id, "done", duration_s=secs)
    rate = store.render_rate("qwen-edit", 1024)
    assert rate is not None and 19.0 < rate < 21.0
    # a 50-step edit is over 1000s of work: the flat 900s timeout it used to
    # get is what cut a finished render off
    assert rate * 50 > 900


def test_render_rate_halves_a_strength_pass(tmp_path):
    store = make_store(tmp_path)
    s = store.create_session("x")
    gen = store.create_generation(
        session_id=s.id, kind="upscale", model_key="m", prompt="p",
        negative_prompt="", seed=1, width=2048, height=2048, steps=8,
        cfg=1.0, sampler=None, strength=0.5,
    )
    store.set_status(gen.id, "done", duration_s=40.0)
    # 8 steps at 0.5 is 4 steps of work, so 40s is 10 s/unit, not 5
    assert store.render_rate("m", 2048) == 10.0


def test_render_rate_prefers_this_size_and_falls_back_to_the_slowest(tmp_path):
    store = make_store(tmp_path)
    s = store.create_session("x")
    for edge, secs in ((512, 10.0), (1024, 100.0)):
        gen = store.create_generation(
            session_id=s.id, kind="draft", model_key="m", prompt="p",
            negative_prompt="", seed=1, width=edge, height=edge, steps=10,
            cfg=1.0, sampler=None,
        )
        store.set_status(gen.id, "done", duration_s=secs)
    assert store.render_rate("m", 512) == 1.0
    assert store.render_rate("m", 1024) == 10.0
    # an unseen size gets the slowest one measured rather than the nearest
    assert store.render_rate("m", 2048) == 10.0


def test_render_rate_ignores_crop_passes_and_untimed_rows(tmp_path):
    store = make_store(tmp_path)
    s = store.create_session("x")
    detail = store.create_generation(
        session_id=s.id, kind="detail", model_key="m", prompt="p",
        negative_prompt="", seed=1, width=1024, height=1024, steps=10,
        cfg=1.0, sampler=None,
    )
    store.set_status(detail.id, "done", duration_s=600.0)
    untimed = store.create_generation(
        session_id=s.id, kind="draft", model_key="m", prompt="p",
        negative_prompt="", seed=1, width=1024, height=1024, steps=10,
        cfg=1.0, sampler=None,
    )
    store.set_status(untimed.id, "done")
    assert store.render_rate("m", 1024) is None
