"""The publish endpoint: what it refuses, and what it writes down when it works.

Civitai is stubbed throughout -- these tests are about the app's half of the
bargain. The one that matters most is that a row is marked published only after
Civitai has answered with a post id, because a card claiming a post that does
not exist is worse than a publish that has to be retried.
"""

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import civitai, main
from app.civitai import CivitaiClient as RealCivitaiClient  # bound before patching
from app.sessions import Store


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "store", Store(tmp_path))
    monkeypatch.setattr(civitai, "token", lambda: "test-key")
    with TestClient(main.app) as c:
        yield c


async def _true():
    return True


async def _false():
    return False


def a_finished_image(client, kind="final", model_key="bas-anima", **kw):
    """A generation with real bytes on disk, as a publish needs."""
    sid = client.post("/api/sessions", json={"instruction": "a cat"}).json()["id"]
    gen = main.store.create_generation(
        session_id=sid, kind=kind, model_key=model_key, prompt="1girl, a park",
        negative_prompt="blurry", seed=42, width=64, height=64, steps=28,
        cfg=5.5, sampler="Euler a", **kw,
    )
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (30, 60, 90)).save(buf, format="PNG")
    main.store.save_image(gen, buf.getvalue())
    main.store.set_status(gen.id, "done", None)
    return main.store.get_generation(gen.id)


class FakeClient:
    """Records what it was asked to send, and answers like Civitai does."""

    sent: dict = {}

    def __init__(self, *a, **kw):
        pass

    async def upload_image(self, png):
        FakeClient.png = png
        return "uuid-1"

    async def create_post(self, **kw):
        FakeClient.sent.clear()
        FakeClient.sent.update(kw)
        return {"id": 4242, "published_at": None,
                "url": "https://civitai.com/posts/4242"}


@pytest.fixture
def fake(monkeypatch):
    FakeClient.sent = {}
    monkeypatch.setattr(civitai, "CivitaiClient", FakeClient)
    return FakeClient


def test_a_successful_publish_records_the_post_and_when(client, fake):
    g = a_finished_image(client)
    body = client.post(f"/api/generations/{g.id}/publish", json={}).json()
    assert body["published_post_id"] == 4242
    assert body["published_at"]
    # ...and it survives a reload, which is the whole point of the columns
    assert main.store.get_generation(g.id).published_post_id == 4242


def test_the_uploaded_png_carries_the_prompt_and_the_settings(client, fake):
    g = a_finished_image(client)
    client.post(f"/api/generations/{g.id}/publish", json={})
    text = Image.open(io.BytesIO(fake.png)).text["parameters"]
    assert text.startswith("1girl, a park")
    assert "Negative prompt: blurry" in text
    assert "Seed: 42" in text
    assert "Steps: 28" in text
    # bas-anima is a Civitai checkpoint, so the post can credit it
    assert '"modelVersionId":3004325' in text
    assert fake.sent["model_version_id"] == 3004325


@pytest.mark.parametrize("method, route", [
    ("post", "publish"),
    ("get", "publish-defaults"),
    ("post", "caption-image"),
    ("post", "describe-image"),
])
def test_a_draft_is_never_published(client, fake, method, route):
    # a draft is a low-resolution preview for picking a direction from; the
    # dialog's helpers refuse it too, so there is no half-way into posting one
    g = a_finished_image(client, kind="draft")
    resp = getattr(client, method)(f"/api/generations/{g.id}/{route}")
    assert resp.status_code == 400
    assert "draft" in resp.json()["detail"].lower()
    assert fake.sent == {}
    assert main.store.get_generation(g.id).published_at is None


def test_an_upscale_publishes_its_parents_prompt_and_seed(client, fake):
    parent = a_finished_image(client)
    up = main.store.create_generation(
        session_id=parent.session_id, kind="upscale", model_key="bas-anima",
        prompt=parent.prompt, negative_prompt=parent.negative_prompt,
        seed=parent.seed, width=128, height=128, steps=20, cfg=4.0,
        sampler="Euler a", parent_id=parent.id,
    )
    buf = io.BytesIO()
    Image.new("RGB", (128, 128), (1, 2, 3)).save(buf, format="PNG")
    main.store.save_image(up, buf.getvalue())
    main.store.set_status(up.id, "done", None)
    client.post(f"/api/generations/{up.id}/publish", json={})
    text = Image.open(io.BytesIO(fake.png)).text["parameters"]
    assert "Steps: 28" in text            # the final's, not the upscale pass's
    # ...but the size is the file's: nothing is downscaled on the way out, and
    # a post that reads 64x64 beside a 128px image is describing another picture
    assert "Size: 128x128" in text
    assert "Size: 64x64" not in text
    # and the image attached is that same big one
    assert fake.sent["width"] == 128


def test_a_qwen_edit_publishes_the_checkpoint_that_drew_it(client, fake):
    """Reported: an instruction edit went up with no generation data."""
    parent = a_finished_image(client)
    edit = main.store.create_generation(
        session_id=parent.session_id, kind="edit", model_key="qwen-edit",
        prompt="replace the head piece with jewelry. Keep everything else exactly unchanged.",
        negative_prompt="", seed=7, width=64, height=64, steps=16, cfg=1.0,
        sampler="Euler A Trailing", parent_id=parent.id, strength=1.0,
    )
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (1, 2, 3)).save(buf, format="PNG")
    main.store.save_image(edit, buf.getvalue())
    main.store.set_status(edit.id, "done", None)
    defaults = client.get(f"/api/generations/{edit.id}/publish-defaults").json()
    assert defaults["prompt"] == "1girl, a park"
    version = main.registry.models["bas-anima"].civitai_version_id
    assert f'"modelVersionId":{version}' in defaults["settings"]
    client.post(f"/api/generations/{edit.id}/publish", json={})
    text = Image.open(io.BytesIO(fake.png)).text["parameters"]
    assert "replace the head piece" not in text
    assert "Steps: 28" in text and "qwen" not in text.lower()


def an_upscale_of(client, parent, model_key="janku"):
    up = main.store.create_generation(
        session_id=parent.session_id, kind="upscale", model_key=model_key,
        prompt=parent.prompt, negative_prompt=parent.negative_prompt,
        seed=parent.seed, width=128, height=128, steps=20, cfg=4.0,
        sampler="Euler a", parent_id=parent.id,
    )
    buf = io.BytesIO()
    Image.new("RGB", (128, 128), (1, 2, 3)).save(buf, format="PNG")
    main.store.save_image(up, buf.getvalue())
    main.store.set_status(up.id, "done", None)
    return main.store.get_generation(up.id)


def test_an_upscale_credits_the_model_that_ran_it_as_well(client, fake):
    """Both checkpoints worked on the picture, so both are on the post."""
    parent = a_finished_image(client, model_key="krea2-turbo")
    up = an_upscale_of(client, parent)
    krea = main.registry.models["krea2-turbo"].civitai_version_id
    janku = main.registry.models["janku"].civitai_version_id
    defaults = client.get(f"/api/generations/{up.id}/publish-defaults").json()
    assert f'"modelVersionId":{krea}' in defaults["settings"]
    assert f'"modelVersionId":{janku}' in defaults["settings"]
    client.post(f"/api/generations/{up.id}/publish", json={})
    text = Image.open(io.BytesIO(fake.png)).text["parameters"]
    # the composer is first and keeps Model: to itself
    assert text.index(str(krea)) < text.index(str(janku))
    assert "krea" in text and "janku" not in text.split("Civitai resources")[0]
    # the picture is still krea2's, so its numbers stand
    assert "Steps: 28" in text
    # the post links the composer, which is the checkpoint whose seed is on it
    assert fake.sent["model_version_id"] == krea


def test_the_upscaler_is_the_credit_when_the_composer_has_no_civitai_page(client, fake, monkeypatch):
    """Reported: Civitai detected no resources at all.

    Every model ships with a version id now, so this is the case of one that
    cannot have one -- a checkpoint with no page. The picture's own numbers
    still go up; what changes is that the post links the model that ran the
    pass rather than nothing.
    """
    monkeypatch.setattr(main.registry.models["krea2-turbo"], "civitai_version_id", None)
    parent = a_finished_image(client, model_key="krea2-turbo")
    up = an_upscale_of(client, parent)
    janku = main.registry.models["janku"].civitai_version_id
    client.post(f"/api/generations/{up.id}/publish", json={})
    text = Image.open(io.BytesIO(fake.png)).text["parameters"]
    assert f'"modelVersionId":{janku}' in text
    assert "Steps: 28" in text and "krea" in text
    assert fake.sent["model_version_id"] == janku


def test_the_post_is_always_a_draft_and_nothing_can_ask_otherwise(client, fake):
    """Publishing is one click on Civitai's own page; the app never does it."""
    g = a_finished_image(client)
    client.post(f"/api/generations/{g.id}/publish", json={"publish": True})
    assert "publish" not in fake.sent          # the route does not pass it on
    # ...and the client hardcodes the flag, so no caller can turn it on either
    import inspect
    assert "publish" not in inspect.signature(RealCivitaiClient.create_post).parameters
    assert '"publish": False' in inspect.getsource(RealCivitaiClient.create_post)


def test_the_title_description_and_tags_are_passed_through(client, fake):
    g = a_finished_image(client)
    client.post(f"/api/generations/{g.id}/publish", json={
        "title": "  Park girl  ", "detail": "made locally",
        "tags": ["anime", "  ", "park"]})
    assert fake.sent["title"] == "Park girl"
    assert fake.sent["detail"] == "made locally"
    assert fake.sent["tags"] == ["anime", "park"]


def test_an_edited_prompt_is_what_gets_embedded(client, fake):
    """The dialog's text wins over the row's, without touching the row."""
    g = a_finished_image(client)
    client.post(f"/api/generations/{g.id}/publish", json={
        "prompt": "  a girl in a park  ", "negative_prompt": "watermark"})
    text = Image.open(io.BytesIO(fake.png)).text["parameters"]
    assert text.startswith("a girl in a park\n")
    assert "Negative prompt: watermark" in text
    # the settings are not the dialog's to rewrite and come through unchanged
    assert "Seed: 42" in text and "Steps: 28" in text
    # and the generation still knows what actually rendered it
    assert main.store.get_generation(g.id).prompt == "1girl, a park"


def test_an_emptied_prompt_publishes_without_one(client, fake):
    """"" is a choice; only a missing field falls back to the row."""
    g = a_finished_image(client)
    client.post(f"/api/generations/{g.id}/publish",
                json={"prompt": "", "negative_prompt": ""})
    text = Image.open(io.BytesIO(fake.png)).text["parameters"]
    assert "1girl" not in text
    assert "Negative prompt" not in text
    assert text.startswith("Steps: 28")


def test_a_simple_post_carries_the_settings_and_no_prompt(client, fake):
    """The other post: a title, a description, tags, and what the render did."""
    g = a_finished_image(client)
    client.post(f"/api/generations/{g.id}/publish", json={
        "simple": True, "title": "Green Hour",
        "detail": "A girl waits under the trees.", "tags": ["park", "anime"]})
    text = Image.open(io.BytesIO(fake.png)).text["parameters"]
    assert "1girl" not in text and "Negative prompt" not in text
    # the technicals are the half a simple post keeps
    assert text.startswith("Steps: 28")
    assert "Seed: 42" in text and '"modelVersionId":3004325' in text
    assert fake.sent["title"] == "Green Hour"
    assert fake.sent["detail"] == "A girl waits under the trees."
    assert fake.sent["tags"] == ["park", "anime"]


def test_a_simple_post_drops_a_prompt_the_client_sent_anyway(client, fake):
    """The flag is enforced here, not trusted to the dialog to blank the boxes."""
    g = a_finished_image(client)
    client.post(f"/api/generations/{g.id}/publish", json={
        "simple": True, "prompt": "1girl, a park", "negative_prompt": "blurry"})
    text = Image.open(io.BytesIO(fake.png)).text["parameters"]
    assert "1girl" not in text and "blurry" not in text


def test_a_simple_post_needs_no_fields_at_all(client, fake):
    """The minimum: a picture and the numbers behind it."""
    g = a_finished_image(client)
    resp = client.post(f"/api/generations/{g.id}/publish", json={"simple": True})
    assert resp.status_code == 200
    assert resp.json()["published_post_id"] == 4242
    # nothing was invented to fill the fields nobody filled in; the real
    # client leaves an empty one out of the JSON-RPC arguments entirely
    assert fake.sent["title"] == "" and fake.sent["detail"] == ""
    assert fake.sent["tags"] == []
    text = Image.open(io.BytesIO(fake.png)).text["parameters"]
    assert text.startswith("Steps: 28")


def test_an_unedited_publish_still_carries_the_rows_own_prompt(client, fake):
    g = a_finished_image(client)
    client.post(f"/api/generations/{g.id}/publish", json={"title": "t"})
    text = Image.open(io.BytesIO(fake.png)).text["parameters"]
    assert text.startswith("1girl, a park")
    assert "Negative prompt: blurry" in text


def test_reading_a_prompt_off_the_image_returns_tags_to_match(client, monkeypatch):
    """The dialog's "Read from image": the picture wins over the stored text."""
    g = a_finished_image(client)
    seen = {}

    async def fake_describe(image_b64, style="prose", uses_negative_prompt=False,
                            hint="", guidance="", opener="", budget=None):
        seen.update(image_b64=image_b64, style=style, hint=hint,
                    uses_negative_prompt=uses_negative_prompt)
        from app.schemas import PromptVariant
        return PromptVariant(prompt="1girl, forest, dappled sunlight")

    monkeypatch.setattr(main.llm, "has_vision", lambda *a: _true())
    monkeypatch.setattr(main.llm, "describe", fake_describe)
    body = client.post(f"/api/generations/{g.id}/describe-image").json()
    assert body["prompt"] == "1girl, forest, dappled sunlight"
    # tags are re-derived from the new text, not left over from the old
    assert "forest" in body["tags"]
    # the image really was sent, and the old prompt was not: it is the thing
    # being replaced, and feeding it back anchors the model to it
    assert seen["image_b64"] and "a park" not in seen["hint"]
    # nothing the picture cannot show is asked for
    assert seen["uses_negative_prompt"] is False
    # bas-anima is a tag-prompted checkpoint, so the reading is in tags
    assert seen["style"] == "tags"
    # and the row still holds what actually rendered it
    assert main.store.get_generation(g.id).prompt == "1girl, a park"


def test_captioning_reads_the_picture_and_nothing_else(client, monkeypatch):
    """The simple post's "Write from image": no prompt goes in or comes out."""
    g = a_finished_image(client)
    seen = {}

    async def fake_caption(image_b64):
        seen.update(image_b64=image_b64)
        from app.schemas import ArtworkCaption
        return ArtworkCaption(title="Green Hour",
                              description="A girl waits under the trees.",
                              tags=["park", "anime"])

    monkeypatch.setattr(main.llm, "has_vision", lambda *a: _true())
    monkeypatch.setattr(main.llm, "caption", fake_caption)
    body = client.post(f"/api/generations/{g.id}/caption-image").json()
    assert body["title"] == "Green Hour"
    # the tool is named on the end: Civitai's Tools taxonomy cannot be set
    # through the API, and the description is the only field that reaches the page
    assert body["description"] == (
        "A girl waits under the trees. Rendered locally in Draw Things."
    )
    # bas-anima is a tag-style checkpoint, so `anime` goes in front: its output
    # is anime whatever anyone wrote, and neither vision model volunteers it
    assert body["tags"] == ["anime", "park"]
    # the picture is all the model was given: no prompt to anchor it back to
    # words that may describe a picture that never arrived
    assert seen["image_b64"] and "prompt" not in seen
    assert "prompt" not in body


def test_a_prose_model_gets_no_anime_tag_put_in_front(client, monkeypatch):
    """The rule is the checkpoint's dialect, not a tag added to everything."""
    g = a_finished_image(client, model_key="flux2-klein")

    async def fake_caption(image_b64):
        from app.schemas import ArtworkCaption
        return ArtworkCaption(title="t", description="d", tags=["park"])

    monkeypatch.setattr(main.llm, "has_vision", lambda *a: _true())
    monkeypatch.setattr(main.llm, "caption", fake_caption)
    body = client.post(f"/api/generations/{g.id}/caption-image").json()
    assert body["tags"] == ["park"]


def test_captioning_needs_a_vision_model(client, monkeypatch):
    g = a_finished_image(client)
    monkeypatch.setattr(main.llm, "has_vision", lambda *a: _false())
    resp = client.post(f"/api/generations/{g.id}/caption-image")
    assert resp.status_code == 400
    assert "vision" in resp.json()["detail"].lower()


def test_a_clip_cannot_be_captioned(client):
    g = a_finished_image(client, kind="video", num_frames=33, fps=16)
    assert client.post(f"/api/generations/{g.id}/caption-image").status_code == 400


def test_reading_a_prompt_needs_a_vision_model(client, monkeypatch):
    g = a_finished_image(client)
    monkeypatch.setattr(main.llm, "has_vision", lambda *a: _false())
    resp = client.post(f"/api/generations/{g.id}/describe-image")
    assert resp.status_code == 400
    assert "vision" in resp.json()["detail"].lower()


def test_a_clip_has_no_prompt_to_read(client):
    g = a_finished_image(client, kind="video", num_frames=33, fps=16)
    assert client.post(f"/api/generations/{g.id}/describe-image").status_code == 400


def test_more_than_five_tags_is_refused_rather_than_silently_cut(client, fake):
    g = a_finished_image(client)
    resp = client.post(f"/api/generations/{g.id}/publish",
                       json={"tags": ["a", "b", "c", "d", "e", "f"]})
    assert resp.status_code == 422


def test_a_failure_at_civitai_leaves_the_row_unpublished(client, monkeypatch):
    g = a_finished_image(client)

    class Failing(FakeClient):
        async def create_post(self, **kw):
            raise civitai.CivitaiError("account is muted")

    monkeypatch.setattr(civitai, "CivitaiClient", Failing)
    resp = client.post(f"/api/generations/{g.id}/publish", json={})
    assert resp.status_code == 400
    assert "muted" in resp.json()["detail"]
    assert main.store.get_generation(g.id).published_at is None


def test_without_a_token_the_error_says_where_to_put_one(client, monkeypatch):
    g = a_finished_image(client)
    monkeypatch.setattr(civitai, "token", lambda: None)
    resp = client.post(f"/api/generations/{g.id}/publish", json={})
    assert resp.status_code == 400
    assert "CIVITAI_TOKEN" in resp.json()["detail"]


def test_an_unfinished_image_cannot_be_published(client, fake):
    sid = client.post("/api/sessions", json={"instruction": "a cat"}).json()["id"]
    gen = main.store.create_generation(
        session_id=sid, kind="final", model_key="bas-anima", prompt="p",
        negative_prompt="", seed=1, width=64, height=64, steps=28, cfg=5.5,
        sampler="Euler a",
    )
    assert client.post(f"/api/generations/{gen.id}/publish", json={}).status_code == 400


def test_a_clip_cannot_be_published(client, fake):
    g = a_finished_image(client, kind="video", num_frames=33, fps=16)
    resp = client.post(f"/api/generations/{g.id}/publish", json={})
    assert resp.status_code == 400
    assert "clips" in resp.json()["detail"]


def test_health_reports_whether_publishing_is_available_at_all(client, monkeypatch):
    assert client.get("/api/health").json()["civitai"] is True
    monkeypatch.setattr(civitai, "token", lambda: None)
    assert client.get("/api/health").json()["civitai"] is False


def test_the_dialog_shows_the_metadata_that_will_be_embedded(client, fake):
    """The preview must be the real chunk, not a description of it."""
    g = a_finished_image(client)
    shown = client.get(f"/api/generations/{g.id}/publish-defaults").json()["parameters"]
    client.post(f"/api/generations/{g.id}/publish", json={})
    embedded = Image.open(io.BytesIO(fake.png)).text["parameters"]
    assert shown == embedded


def test_the_dialog_gets_the_prompt_and_the_settings_as_separate_fields(client):
    """The editable halves apart from the one that is not."""
    g = a_finished_image(client)
    pre = client.get(f"/api/generations/{g.id}/publish-defaults").json()
    assert pre["prompt"] == "1girl, a park"
    assert pre["negative_prompt"] == "blurry"
    assert pre["settings"].startswith("Steps: 28")
    assert "1girl" not in pre["settings"]
    # the three still add up to the chunk, so an untouched dialog changes nothing
    assert pre["parameters"] == (
        f"{pre['prompt']}\nNegative prompt: {pre['negative_prompt']}\n{pre['settings']}"
    )


def test_the_preview_for_an_upscale_is_its_parents_settings(client, fake):
    parent = a_finished_image(client)
    up = main.store.create_generation(
        session_id=parent.session_id, kind="upscale", model_key="bas-anima",
        prompt=parent.prompt, negative_prompt="", seed=999, width=2048,
        height=2048, steps=20, cfg=4.0, sampler="Euler a", parent_id=parent.id,
    )
    shown = client.get(f"/api/generations/{up.id}/publish-defaults").json()["parameters"]
    assert "Seed: 42" in shown and "Steps: 28" in shown   # the final's, not the pass's
    assert "Seed: 999" not in shown
    assert "Size: 2048x2048" in shown                     # the upscale's own pixels


def test_the_dialog_is_prefilled_from_the_session_and_the_prompt(client, fake):
    sid = client.post("/api/sessions", json={
        "instruction": "a girl in a leather jacket on a neon street"}).json()["id"]
    g = main.store.create_generation(
        session_id=sid, kind="final", model_key="lustrousmix-anima",
        prompt="masterpiece, best quality, absurdres, 1girl, black leather jacket, "
               "city street at night, looking at viewer",
        negative_prompt="", seed=1, width=1536, height=1536, steps=28, cfg=4.5,
        sampler="Euler a",
    )
    body = client.get(f"/api/generations/{g.id}/publish-defaults").json()
    assert body["title"] == "A girl in a leather jacket on a neon street"
    assert "1536×1536" in body["detail"]
    assert body["tags"][0] == "anime" and "1girl" in body["tags"]
    assert "absurdres" not in body["tags"]


def test_the_defaults_for_an_upscale_describe_the_image_it_came_from(client, fake):
    parent = a_finished_image(client)
    up = main.store.create_generation(
        session_id=parent.session_id, kind="upscale", model_key="bas-anima",
        prompt=parent.prompt, negative_prompt="", seed=1, width=2048, height=2048,
        steps=20, cfg=4.0, sampler="Euler a", parent_id=parent.id,
    )
    body = client.get(f"/api/generations/{up.id}/publish-defaults").json()
    assert "64×64" in body["detail"] and "upscaled to 2048×2048" in body["detail"]


def test_defaults_for_an_unknown_generation_are_a_404(client, fake):
    assert client.get("/api/generations/nope/publish-defaults").status_code == 404


def test_the_loras_a_picture_was_rendered_with_are_credited(client, fake):
    """Recorded on the row when it was queued, so they are what actually ran."""
    turbo = next(o for o in main.registry.models["lustrousmix-anima"].lora_options
                 if o.key == "turbo")
    parent = a_finished_image(client, model_key="lustrousmix-anima",
                              loras=[{"file": turbo.file, "weight": 1.0}])
    defaults = client.get(f"/api/generations/{parent.id}/publish-defaults").json()
    # a row recorded without the id gets it from the option it came from
    assert f'"type":"lora","weight":1.0,"modelVersionId":{turbo.civitai_version_id}' \
        in defaults["settings"]
    client.post(f"/api/generations/{parent.id}/publish", json={})
    text = Image.open(io.BytesIO(fake.png)).text["parameters"]
    assert f'"modelVersionId":{turbo.civitai_version_id}' in text
    # the post still links the checkpoint, not the LoRA
    assert fake.sent["model_version_id"] == \
        main.registry.models["lustrousmix-anima"].civitai_version_id


def test_an_upscale_credits_the_loras_of_the_picture_it_enlarged(client, fake):
    turbo = next(o for o in main.registry.models["lustrousmix-anima"].lora_options
                 if o.key == "turbo")
    parent = a_finished_image(client, model_key="lustrousmix-anima",
                              loras=[{"file": turbo.file, "weight": 1.0,
                                      "civitai_version_id": turbo.civitai_version_id}])
    up = an_upscale_of(client, parent)
    defaults = client.get(f"/api/generations/{up.id}/publish-defaults").json()
    assert f'"modelVersionId":{turbo.civitai_version_id}' in defaults["settings"]


def test_a_row_from_before_loras_were_recorded_credits_none(client, fake):
    parent = a_finished_image(client, model_key="lustrousmix-anima")
    defaults = client.get(f"/api/generations/{parent.id}/publish-defaults").json()
    assert '"type":"lora"' not in defaults["settings"]
