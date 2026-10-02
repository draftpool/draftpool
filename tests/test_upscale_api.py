"""What the Upscale endpoint will and will not enlarge.

The rule under test is that *size* decides, not lineage. The card used to offer
Upscale on a final, a faithful and an edit only, which left a Fix faces result
-- the same picture, at the same size, one pass later -- with no way to reach
2K except by going back to its parent and redoing the face fix afterwards. The
endpoint never had that restriction; the button list was the whole of it.

What replaces it is the check that was actually missing: a target that is not
larger than the picture. Nothing stopped a 3072px image being "upscaled" to
2048 and coming back smaller than it went in.
"""

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import main
from app.registry import CONFIG_PATH, load_registry
from app.sessions import Store

# bas-anima tiles the denoiser, so it offers 3072 and 4096 as well as 2048 and
# can exercise the upper end; oneobsession does not, and stops at 2048.
TILED = "bas-anima"
UNTILED = "oneobsession"


@pytest.fixture
def client(tmp_path, monkeypatch):
    reg = load_registry(CONFIG_PATH)
    for spec in reg.models.values():
        spec.installed = True
    # the pass prepares its init with a Draw Things upscaler and refuses to run
    # without one, which has nothing to do with what these tests are about
    reg.upscaler = "4x_ultrasharp_f16.ckpt"
    monkeypatch.setattr(main, "registry", reg)
    monkeypatch.setattr(main, "store", Store(tmp_path))
    # no lifespan: the worker would pick these rows off the queue and try to
    # render them against a Draw Things that is not there
    return TestClient(main.app)


def an_image(client, kind="final", model_key=TILED, size=1536):
    """A finished generation of `kind`, square at `size`, with bytes on disk."""
    sid = client.post("/api/sessions", json={"instruction": "a cat"}).json()["id"]
    gen = main.store.create_generation(
        session_id=sid, kind=kind, model_key=model_key, prompt="1girl, a park",
        negative_prompt="blurry", seed=42, width=size, height=size, steps=20,
        cfg=4.0, sampler="DPM++ 2M Karras",
    )
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (30, 60, 90)).save(buf, format="PNG")
    main.store.save_image(gen, buf.getvalue())
    main.store.set_status(gen.id, "done", None)
    return main.store.get_generation(gen.id)


def upscale(client, gen, **body):
    return client.post(f"/api/generations/{gen.id}/upscale", json=body)


def test_a_face_fix_can_be_upscaled_like_the_final_under_it(client):
    """The case that started this: same picture, same size, no button."""
    g = an_image(client, kind="detail")
    resp = upscale(client, g, size=2048)
    assert resp.status_code == 200
    assert resp.json()["width"] == 2048


@pytest.mark.parametrize("kind", ["final", "faithful", "edit", "detail", "esrgan", "upscale"])
def test_how_an_image_was_made_does_not_decide_whether_it_can_grow(client, kind):
    """Every kind that leaves a picture behind, at a size with room above it.

    The pass reads its parent's PNG and enlarges it; by then a face fix, a
    refine, a 2x fast and an earlier upscale are all just pictures at a size.
    """
    g = an_image(client, kind=kind, size=512)
    assert upscale(client, g, size=2048).status_code == 200


def test_a_pass_that_would_not_add_pixels_is_refused(client):
    """A 2048 image "upscaled" to 2048 is a re-render, and it used to be allowed."""
    g = an_image(client, kind="upscale", size=2048)
    resp = upscale(client, g, size=2048)
    assert resp.status_code == 400
    assert "already 2048x2048" in resp.json()["detail"]


def test_an_image_larger_than_the_target_is_refused_rather_than_shrunk(client):
    """A 2x fast of a 1536 image is 3072; the 2048 pass would hand back less."""
    g = an_image(client, kind="esrgan", size=3072)
    resp = upscale(client, g, size=2048)
    assert resp.status_code == 400
    assert "smaller, not larger" in resp.json()["detail"]


def test_an_upscale_can_be_upscaled_again_where_the_model_tiles(client):
    """2048 -> 3072 is the offer tiled_diffusion exists to make."""
    g = an_image(client, kind="upscale", size=2048)
    assert upscale(client, g, size=3072).status_code == 200


def test_an_untiled_model_still_stops_at_its_own_ceiling(client):
    """The size rule is additional to the capability rule, not a way round it."""
    g = an_image(client, kind="upscale", model_key=UNTILED, size=2048)
    resp = upscale(client, g, size=3072)
    assert resp.status_code == 400
    assert "renders at most 2048px" in resp.json()["detail"]


def test_a_stand_in_lends_its_own_ceiling_to_another_models_image(client):
    """What the card's two pickers have to agree about.

    The size rule belongs to the model that runs the pass, not to the one that
    made the image: the endpoint resolves `via` before it checks anything about
    size, so an untiled model's 2048 output reaches 4K through a tiled stand-in
    even though that model cannot take its own output past 2048.
    """
    g = an_image(client, kind="upscale", model_key=UNTILED, size=2048)
    assert upscale(client, g, size=4096).status_code == 400          # itself: no
    assert upscale(client, g, size=4096, via=TILED).status_code == 200  # stand-in: yes


def test_a_stand_in_that_cannot_enlarge_it_either_is_still_refused(client):
    """The stand-in is not a way round the capability rule, only around a gap."""
    g = an_image(client, kind="upscale", model_key=UNTILED, size=2048)
    resp = upscale(client, g, size=3072, via="perfectdeliberate")
    assert resp.status_code == 400
    assert "renders at most 2048px" in resp.json()["detail"]


def test_a_tall_picture_is_judged_on_area_not_on_its_long_edge(client):
    """2304x4096: a 9:16 image at the 2048 budget, doubled by a 2x fast.

    Its long edge is already 4096, so judged on that the 4K pass looks like a
    downscale and would be refused -- wrongly, because the pass returns
    3072x5440, nearly twice the pixels. Judged as an area the picture is 3072,
    which makes 4096 a real enlargement and 3072 exactly not one.
    """
    sid = client.post("/api/sessions", json={"instruction": "a cat"}).json()["id"]
    gen = main.store.create_generation(
        session_id=sid, kind="esrgan", model_key=TILED, prompt="1girl",
        negative_prompt="", seed=1, width=2304, height=4096, aspect="9:16",
        steps=20, cfg=4.0, sampler="Euler a",
    )
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (30, 60, 90)).save(buf, format="PNG")
    main.store.save_image(gen, buf.getvalue())
    main.store.set_status(gen.id, "done", None)
    g = main.store.get_generation(gen.id)
    resp = upscale(client, g, size=3072)
    assert resp.status_code == 400
    assert "smaller, not larger" in resp.json()["detail"]
    assert upscale(client, g, size=4096).status_code == 200
