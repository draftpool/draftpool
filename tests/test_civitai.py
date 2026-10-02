"""Publishing: the metadata that goes with an image, and the two calls that send it.

Nothing here touches the network. The parts worth pinning down are the ones a
Civitai-side change would not tell us about: which generation's numbers get
attached, the exact shape of the `parameters` chunk Civitai parses, and that a
failure from the far end arrives as a readable message rather than a KeyError.
"""

import io
import json

import pytest
from PIL import Image

from app import civitai
from app.schemas import Generation, LoraSpec


def gen(**kw) -> Generation:
    base = dict(
        id="aaa111", session_id="s1", parent_id=None, kind="final",
        model_key="bas-anima", prompt="1girl, a park", negative_prompt="blurry",
        seed=42, width=1024, height=1024, steps=28, cfg=5.5,
        sampler="Euler a", status="done", created_at="2026-08-30T10:00:00+00:00",
    )
    base.update(kw)
    return Generation(**base)


def png(size=(8, 8)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (10, 20, 30)).save(buf, format="PNG")
    return buf.getvalue()


# -- which generation's parameters get published -------------------------------


def test_an_upscale_publishes_the_settings_of_the_image_it_enlarged():
    final = gen(id="final1", kind="final", steps=28, seed=42)
    up = gen(id="up1", kind="upscale", parent_id="final1", steps=20, seed=999,
             cfg=4.0, prompt="1girl, a park")
    assert civitai.source_of(up, {"final1": final}).id == "final1"


def test_a_chain_of_enhancement_passes_walks_all_the_way_back():
    final = gen(id="f", kind="final")
    up = gen(id="u", kind="upscale", parent_id="f")
    esr = gen(id="e", kind="esrgan", parent_id="u")
    face = gen(id="d", kind="detail", parent_id="e")
    chain = {"f": final, "u": up, "e": esr}
    assert civitai.source_of(face, chain).id == "f"


def test_a_refine_is_its_own_source():
    """Refine changes the prompt, so its row is what made the picture."""
    final = gen(id="f", kind="final", prompt="1girl, a park")
    edit = gen(id="e", kind="edit", parent_id="f", prompt="1girl, a park, red coat")
    assert civitai.source_of(edit, {"f": final}).id == "e"


def test_an_instruction_edit_publishes_the_checkpoint_that_drew_it():
    """Reported: a Qwen Edit picture went up with no generation data.

    Its row names the editor, which has no Civitai version to attribute, and
    its "prompt" is the order it was given. Neither made the picture.
    """
    final = gen(id="f", kind="final", model_key="oneobsession", prompt="1girl, headdress")
    edit = gen(id="e", kind="edit", parent_id="f", model_key="qwen-edit",
               prompt="replace the head piece with jewelry.")
    again = gen(id="g", kind="edit", parent_id="e", model_key="qwen-edit", prompt="make it night")
    up = gen(id="u", kind="upscale", parent_id="g", model_key="oneobsession")
    chain = {"f": final, "e": edit, "g": again}
    editors = {"qwen-edit"}
    assert civitai.source_of(edit, chain, editors).id == "f"
    assert civitai.source_of(up, chain, editors).id == "f"
    # a checkpoint's repaint is still its own source, editors or not
    refine = gen(id="r", kind="edit", parent_id="f", prompt="1girl, headdress, red coat")
    assert civitai.source_of(refine, chain, editors).id == "r"


def test_an_instruction_edit_with_nothing_above_it_is_its_own_source():
    edit = gen(id="e", kind="edit", parent_id="gone", model_key="qwen-edit")
    assert civitai.source_of(edit, {}, {"qwen-edit"}).id == "e"


def test_a_broken_chain_falls_back_to_the_row_itself():
    """A deleted parent must not stop a publish."""
    orphan = gen(id="u", kind="upscale", parent_id="gone")
    assert civitai.source_of(orphan, {}).id == "u"


def test_a_cycle_cannot_hang_the_walk():
    a = gen(id="a", kind="upscale", parent_id="b")
    b = gen(id="b", kind="upscale", parent_id="a")
    assert civitai.source_of(a, {"a": a, "b": b}).id in {"a", "b"}


# -- the parameters chunk ------------------------------------------------------


def test_parameters_has_the_layout_civitai_parses():
    text = civitai.parameters(gen(), "basbetteranimestyleplusanima_basev10_f16", 3004325)
    lines = text.splitlines()
    assert lines[0] == "1girl, a park"
    assert lines[1] == "Negative prompt: blurry"
    assert "Steps: 28" in lines[2]
    assert "CFG scale: 5.5" in lines[2]
    assert "Seed: 42" in lines[2]
    assert "Size: 1024x1024" in lines[2]
    assert "Model: basbetteranimestyleplusanima_basev10_f16" in lines[2]


def test_the_size_can_be_the_published_file_rather_than_the_row():
    """An upscale posts its parent's numbers -- but its own pixels.

    Every other field here measures the render; this one measures the file that
    is being attached to the post, and the two are different for every
    passthrough pass.
    """
    text = civitai.parameters(gen(), "m", None, size=(3072, 3072))
    assert "Size: 3072x3072" in text
    assert "Steps: 28" in text and "Seed: 42" in text    # untouched
    assert "Size: 1024x1024" not in text


def test_the_row_describes_itself_when_no_size_is_given():
    assert "Size: 1024x1024" in civitai.settings_line(gen(), "m")


def test_the_checkpoint_is_attributed_by_version_id_not_hash():
    text = civitai.parameters(gen(), "whatever", 3004325)
    payload = text.split("Civitai resources: ")[1]
    assert json.loads(payload) == [{"type": "checkpoint", "modelVersionId": 3004325}]


def test_every_model_that_took_part_is_credited_composer_first():
    """An upscale through another checkpoint: both worked on the picture."""
    text = civitai.parameters(gen(), "krea_2_turbo", None, also=[2786084])
    assert json.loads(text.split("Civitai resources: ")[1]) == [
        {"type": "checkpoint", "modelVersionId": 2786084}]
    text = civitai.parameters(gen(), "waiIllustrious", 2883731, also=[2786084])
    assert json.loads(text.split("Civitai resources: ")[1]) == [
        {"type": "checkpoint", "modelVersionId": 2883731},
        {"type": "checkpoint", "modelVersionId": 2786084},
    ]
    # the same model twice is one resource
    text = civitai.parameters(gen(), "waiIllustrious", 2883731, also=[2883731])
    assert len(json.loads(text.split("Civitai resources: ")[1])) == 1


def test_a_model_that_did_not_come_from_civitai_claims_no_resource():
    assert "Civitai resources" not in civitai.parameters(gen(), "flux_1_schnell_q8p", None)


def test_no_negative_prompt_means_no_negative_line():
    text = civitai.parameters(gen(negative_prompt=""), "m", None)
    assert "Negative prompt:" not in text
    assert text.splitlines()[1].startswith("Steps: ")


def test_a_post_with_no_prompt_is_settings_and_nothing_else():
    """A simple post's chunk: no prompt line, not an empty one."""
    text = civitai.parameters(gen(), "m", None, prompt="", negative_prompt="")
    assert text.splitlines() == [text]  # one line
    assert text.startswith("Steps: 28")
    assert "Seed: 42" in text and "Size: 1024x1024" in text


def test_draw_things_sampler_names_are_translated_to_the_ones_a1111_uses():
    assert "Sampler: Euler a" in civitai.parameters(gen(sampler="Euler Ancestral"), "m", None)
    assert "Sampler: DDIM" in civitai.parameters(gen(sampler="DDIM Trailing"), "m", None)
    # unrecognised ones go through as written rather than being dropped
    assert "Sampler: ER SDE" in civitai.parameters(gen(sampler="ER SDE"), "m", None)


def test_a_whole_cfg_does_not_publish_as_a_float():
    assert "CFG scale: 4" in civitai.parameters(gen(cfg=4.0), "m", None)


# -- naming the tool -----------------------------------------------------------


def test_the_tool_is_named_at_the_end_of_a_description():
    assert civitai.with_provenance("A girl waits under the trees") == (
        "A girl waits under the trees. Rendered locally in Draw Things."
    )
    # already punctuated, and whitespace normalised
    assert civitai.with_provenance("  Quiet\n wonder.  ") == (
        "Quiet wonder. Rendered locally in Draw Things."
    )
    assert civitai.with_provenance("Is it a dream?").endswith(
        "Is it a dream? Rendered locally in Draw Things."
    )


def test_a_description_that_already_names_the_tool_is_left_alone():
    """The technical description says it, and so may the user."""
    text = "BAS+ Anima, rendered locally in Draw Things at 512x512."
    assert civitai.with_provenance(text) == text


def test_an_empty_description_becomes_the_tool_alone():
    assert civitai.with_provenance("") == "Rendered locally in Draw Things."


# -- embedding it --------------------------------------------------------------


def test_the_chunk_is_readable_back_and_the_picture_is_unchanged():
    original = png((16, 12))
    out = civitai.embed_parameters(original, "1girl\nSteps: 28")
    img = Image.open(io.BytesIO(out))
    assert img.text["parameters"] == "1girl\nSteps: 28"
    assert img.size == (16, 12)
    assert img.convert("RGB").tobytes() == \
        Image.open(io.BytesIO(original)).convert("RGB").tobytes()


# -- talking to the far end ----------------------------------------------------


def unwrap(payload):
    return civitai.CivitaiClient._unwrap("create_post", json.dumps(payload))


def test_a_tool_error_becomes_a_readable_failure():
    """Civitai answers HTTP 200 with isError when the account cannot post."""
    with pytest.raises(civitai.CivitaiError, match="muted"):
        unwrap({"result": {"isError": True,
                           "content": [{"type": "text", "text": "account is muted"}]}})


def test_a_protocol_error_becomes_a_readable_failure():
    with pytest.raises(civitai.CivitaiError, match="no such tool"):
        unwrap({"error": {"code": -32601, "message": "no such tool"}})


def test_a_json_object_in_the_text_content_is_parsed():
    assert unwrap({"result": {"content": [{"type": "text", "text": '{"id": 77}'}]}}) == {"id": 77}


def test_structured_content_wins_when_the_server_sends_it():
    assert unwrap({"result": {"structuredContent": {"id": 5},
                              "content": [{"type": "text", "text": "ignored"}]}}) == {"id": 5}


def test_an_sse_framed_answer_is_read_the_same_way():
    body = 'event: message\ndata: {"result": {"content": [{"type": "text", "text": "{\\"id\\": 9}"}]}}\n\n'
    assert civitai.CivitaiClient._unwrap("create_post", body) == {"id": 9}


def test_the_post_id_is_found_even_when_it_is_nested():
    """The server has wrapped this once already; one level down costs nothing."""
    assert civitai._first({"post": {"id": 1234}}, "id") == 1234


def test_a_missing_key_is_reported_rather_than_guessed():
    assert civitai._first({"post": {"title": "x"}}, "id") is None


# The two envelopes below are verbatim from a live round trip on 30 Aug 2026
# (upload, draft post, delete). They are here so a change in the shape Civitai
# answers with is a test failure rather than a publish that half works.


def test_the_real_upload_image_envelope_yields_the_uuid():
    body = ('{"result":{"content":[{"type":"text","text":"d4d72cb0-0fcf-4d0f-a760-a0cf2543f15c'
            '\\nUploaded image. UUID: d4d72cb0-0fcf-4d0f-a760-a0cf2543f15c (64x64)"}],'
            '"structuredContent":{"uuid":"d4d72cb0-0fcf-4d0f-a760-a0cf2543f15c","width":64,'
            '"height":64,"contentType":"image/png"}},"jsonrpc":"2.0","id":1}')
    result = civitai.CivitaiClient._unwrap("upload_image", body)
    assert civitai._first(result, "uuid", "id") == "d4d72cb0-0fcf-4d0f-a760-a0cf2543f15c"


def test_the_real_create_post_envelope_yields_the_id_and_the_url():
    body = ('{"result":{"content":[{"type":"text","text":"Post created (draft).\\nID: 30719198"}],'
            '"structuredContent":{"ok":true,"id":30719198,"imageIds":[141300877],'
            '"published":false,"publishedAt":null,"nsfwLevel":0,'
            '"url":"https://civitai.com/posts/30719198"}},"jsonrpc":"2.0","id":1}')
    result = civitai.CivitaiClient._unwrap("create_post", body)
    assert civitai._first(result, "id") == 30719198
    assert civitai._first(result, "url") == "https://civitai.com/posts/30719198"
    # a draft has no publish time of its own, which is why the app stamps one
    assert civitai._first(result, "publishedAt") is None


def test_the_url_we_would_build_matches_the_one_civitai_sends():
    assert civitai.post_url(30719198) == "https://civitai.com/posts/30719198"


def test_a_missing_post_reads_as_an_error_not_an_empty_result():
    """What a deleted post answers: isError, at HTTP 200."""
    body = ('{"result":{"content":[{"type":"text","text":"Error: post.get: Could not find entity"}],'
            '"structuredContent":{"ok":false,"error":"post.get: Could not find entity"},'
            '"isError":true},"jsonrpc":"2.0","id":1}')
    with pytest.raises(civitai.CivitaiError, match="Could not find entity"):
        civitai.CivitaiClient._unwrap("get_post", body)


@pytest.mark.asyncio
async def test_an_oversized_image_is_refused_before_it_is_sent():
    client = civitai.CivitaiClient("t")
    with pytest.raises(civitai.CivitaiError, match="upload limit"):
        await client.upload_image(b"x" * (civitai.MAX_UPLOAD_BYTES + 1))


# -- where the token comes from ------------------------------------------------


def test_the_environment_wins_over_the_key_file(tmp_path, monkeypatch):
    key = tmp_path / "civitai.key"
    key.write_text("from-file\n")
    monkeypatch.setattr(civitai, "KEY_FILE", key)
    monkeypatch.setenv("CIVITAI_TOKEN", "from-env")
    assert civitai.token() == "from-env"
    monkeypatch.delenv("CIVITAI_TOKEN")
    assert civitai.token() == "from-file"


def test_no_token_anywhere_is_not_an_error(tmp_path, monkeypatch):
    monkeypatch.setattr(civitai, "KEY_FILE", tmp_path / "absent.key")
    monkeypatch.setenv("CIVITAI_TOKEN", "   ")
    assert civitai.token() is None
    assert civitai.available() is False


# -- what the dialog is prefilled with -----------------------------------------
#
# These are suggestions in editable fields, so the bar is "not embarrassing"
# rather than "correct". What they must not do is offer the boilerplate that is
# on every prompt this app sends.

from app.registry import load_registry  # noqa: E402

ANIME_PROMPT = (
    "masterpiece, best quality, amazing quality, very aesthetic, absurdres, "
    "1girl, silver hair, black leather jacket, city street at night, neon signs, "
    "rain, looking at viewer, cowboy shot"
)


@pytest.fixture(scope="module")
def registry():
    return load_registry()


def test_tags_come_out_of_the_prompt_ranked_by_how_broad_they_are(registry):
    tags = civitai.suggest_tags(ANIME_PROMPT, registry.models["lustrousmix-anima"])
    assert tags[0] == "anime"          # an anime checkpoint's output always is
    assert "1girl" in tags
    assert "looking at viewer" in tags
    assert len(tags) <= 5              # Civitai's own cap


def test_the_quality_boilerplate_never_becomes_a_tag(registry):
    tags = civitai.suggest_tags(ANIME_PROMPT, registry.models["lustrousmix-anima"])
    for boilerplate in ("masterpiece", "best quality", "very aesthetic", "absurdres"):
        assert boilerplate not in tags


def test_a_models_own_opener_is_excluded_too(registry):
    """bas-anima leads every prompt with anime_screenshot; that says nothing."""
    spec = registry.models["bas-anima"]
    tags = civitai.suggest_tags(
        "anime, anime_coloring, anime_screenshot, masterpiece, best quality, "
        "1girl, twin braids, school uniform", spec)
    assert "anime coloring" not in tags and "anime screenshot" not in tags
    assert "1girl" in tags


def test_a_fragment_of_boilerplate_is_not_a_tag_either(registry):
    """"quality" is a real tag with a real post count, and says nothing."""
    tags = civitai.suggest_tags(
        "masterpiece, best quality, worst detail, 1girl, red scarf",
        registry.models["lustrousmix-anima"])
    assert "quality" not in tags and "detail" not in tags
    assert "1girl" in tags


def test_a_word_already_inside_a_better_phrase_does_not_take_a_slot(registry):
    tags = civitai.suggest_tags("1girl, long hair, blue eyes",
                                registry.models["lustrousmix-anima"])
    assert "long hair" in tags and "hair" not in tags


def test_prose_offers_phrases_and_not_the_nouns_it_tripped_over(registry):
    """A word plucked from mid-clause is incidental, not what the prompt meant."""
    spec = registry.models["juggernaut-ragnarok"]
    fisherman = civitai.suggest_tags(
        "A weathered fisherman mends his nets on a stone harbour wall at dawn, "
        "gulls overhead, cold north light on his hands.", spec)
    assert "wall" not in fisherman and "cold" not in fisherman
    blossoms = civitai.suggest_tags(
        "A young woman with long red hair sits by a window in a coffee shop, "
        "cherry blossoms outside, looking at viewer.", spec)
    assert "cherry blossoms" in blossoms and "red hair" in blossoms


def test_a_prose_model_is_not_labelled_anime(registry):
    tags = civitai.suggest_tags("cherry blossoms, red hair",
                                registry.models["juggernaut-ragnarok"])
    assert "anime" not in tags


def test_the_title_is_what_the_user_asked_for_not_what_the_llm_wrote():
    title = civitai.suggest_title("a girl holding a ginger cat in a forest", ANIME_PROMPT)
    assert title == "A girl holding a ginger cat in a forest"


def test_a_long_instruction_is_cut_at_a_word():
    title = civitai.suggest_title("word " * 40, "", cap=30)
    assert len(title) <= 31 and title.endswith("…") and not title.endswith("wor…")


def test_without_an_instruction_the_title_steps_over_the_quality_tags():
    assert civitai.suggest_title("", ANIME_PROMPT) == (
        "1girl, silver hair, black leather jacket, city street at night, neon signs, rain,"
        " looking…")


def test_an_instruction_that_is_itself_a_prompt_loses_its_boilerplate():
    """Store.save_prompts writes the composer's text over `instruction`, so on
    any session that has been through the composer the instruction IS a prompt.
    Found on real data, where a title came out "Masterpiece, best quality, …".
    """
    title = civitai.suggest_title(ANIME_PROMPT, "")
    assert title.startswith("1girl, silver hair")


def test_a_stray_tag_with_no_comma_after_it_is_peeled_too():
    """Hand-typed prompts drop the comma: "very aesthetic, absurdres 1girl"."""
    title = civitai.suggest_title(
        "masterpiece, best quality, very aesthetic, absurdres 1girl full body, "
        "sitting on sofa", "")
    assert title == "1girl full body, sitting on sofa"


def test_stripping_never_rebuilds_the_punctuation_it_found():
    """Prose keeps its full stops; only the front is touched."""
    text = "A fisherman mends his nets. Gulls overhead. Cold light."
    assert civitai.suggest_title(text, "") == text.rstrip(".")


def test_a_title_that_is_all_boilerplate_comes_back_empty_not_wrong():
    """The final segment has no separator after it, so it needs its own check."""
    assert civitai.suggest_title("masterpiece, best quality", "") == ""


def test_an_all_boilerplate_instruction_falls_through_to_the_prompt():
    title = civitai.suggest_title("masterpiece, best quality", ANIME_PROMPT)
    assert title.startswith("1girl, silver hair")


def test_the_description_says_what_rendered_it_and_at_what_size(registry):
    spec = registry.models["bas-anima"]
    source = gen(width=1536, height=1536)
    text = civitai.suggest_detail(source, source, spec)
    assert spec.label in text and "1536×1536" in text
    assert "upscaled" not in text


def test_an_upscale_says_so_in_the_description(registry):
    spec = registry.models["bas-anima"]
    source = gen(width=1536, height=1536)
    published = gen(id="up", kind="upscale", width=2048, height=2048)
    text = civitai.suggest_detail(published, source, spec)
    assert "1536×1536" in text and "upscaled to 2048×2048" in text


def test_no_vocabulary_snapshot_means_no_tags_rather_than_a_crash(monkeypatch):
    monkeypatch.setattr(civitai.vocab, "get", lambda: civitai.vocab.Vocabulary({}, {}))
    assert civitai.suggest_tags(ANIME_PROMPT, None) == []


def test_every_civitai_checkpoint_can_attribute_itself(registry):
    """A model imported from Civitai should credit its page when published."""
    expected = {
        "lustrousmix-anima": 2982979, "bas-anima": 3004325,
        "cyberrealistic-anima": 3136380, "wai-illustrious": 2883731,
        "cyberrealistic-illustrious": 2886849, "oneobsession": 3218603,
        "juggernaut-ragnarok": 1759168, "perfectdeliberate": 2925672,
        "janku": 2786084,
    }
    for key, version_id in expected.items():
        assert registry.models[key].civitai_version_id == version_id, key


def test_perfectdeliberate_credits_the_anime_line_not_its_near_namesake(registry):
    """v50a is PerfectDeliberate-Anime v5.0_A (111274), not PerfectDeliberate
    v5.0 (24350). One letter, same author, same version number, different
    weights -- and crediting the wrong one would have looked perfectly normal.
    """
    assert registry.models["perfectdeliberate"].civitai_version_id == 2925672


def test_a_model_with_no_version_id_publishes_no_resource_claim():
    """Blank stays the safe state: no attribution beats a wrong one."""
    text = civitai.parameters(gen(), "flux_2_klein_9b_q8p", None)
    assert "Civitai resources" not in text


def test_a_lora_with_a_civitai_page_is_credited_after_the_checkpoints():
    """The form Civitai's own generator writes: type, weight, version."""
    loras = [LoraSpec(file="anima_turbo_lora_v0.2_lora_f16.ckpt", weight=0.8,
                      civitai_version_id=2979642)]
    text = civitai.parameters(gen(), "m", 3004325, also=[2786084], loras=loras)
    assert json.loads(text.split("Civitai resources: ")[1]) == [
        {"type": "checkpoint", "modelVersionId": 3004325},
        {"type": "checkpoint", "modelVersionId": 2786084},
        {"type": "lora", "weight": 0.8, "modelVersionId": 2979642},
    ]


def test_a_lora_without_a_page_claims_nothing_and_the_same_one_is_listed_once():
    loras = [LoraSpec(file="dmd2_sdxl_4_step_lora_f16.ckpt", weight=0.6),
             LoraSpec(file="a.ckpt", civitai_version_id=11),
             LoraSpec(file="a.ckpt", civitai_version_id=11)]
    resources = json.loads(civitai.settings_line(gen(), "m", None, loras=loras)
                           .split("Civitai resources: ")[1])
    assert resources == [{"type": "lora", "weight": 1.0, "modelVersionId": 11}]
    # nothing at all to credit is no field at all, as before
    assert "Civitai resources" not in civitai.settings_line(
        gen(), "m", None, loras=[LoraSpec(file="x.ckpt")])
