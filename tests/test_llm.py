import asyncio

import pytest

from app.llm import EDIT_INSTRUCTION_SYSTEM, LlmError, OllamaClient


def test_parse_variants_trims_and_limits():
    data = {
        "variants": [
            {"prompt": "  a  ", "negative_prompt": " b "},
            {"prompt": "c", "negative_prompt": ""},
            {"prompt": "", "negative_prompt": "ignored, empty prompt"},
            {"prompt": "d"},
        ]
    }
    variants = OllamaClient._parse_variants(data, k=2)
    assert len(variants) == 2
    assert variants[0].prompt == "a"
    assert variants[0].negative_prompt == "b"


def test_tags_style_dedupes_repeated_tags():
    data = {"variants": [{
        "prompt": "masterpiece, best quality, 1girl, masterpiece, red hair, Best Quality",
        "negative_prompt": "lowres, lowres, blurry",
    }]}
    v = OllamaClient._parse_variants(data, 1, "tags")[0]
    assert v.prompt == "masterpiece, best quality, 1girl, red hair"
    assert v.negative_prompt == "lowres, blurry"


def test_tags_style_strips_stable_diffusion_garnish():
    """The tail of a real enriched prompt that flattened Illustrious XL."""
    data = {"variants": [{
        "prompt": (
            "masterpiece, best quality, 1girl, castle ruins, golden hour, "
            "intricate details, concept art, highly detailed, digital painting, "
            "artstation, trending, 8k, cinematic"
        ),
        "negative_prompt": "",
    }]}
    v = OllamaClient._parse_variants(data, 1, "tags")[0]
    # the blocklist takes the garnish. "castle ruins" is not a Danbooru tag
    # ("castle" and "ruins" are, separately) but it is kept whole now: folding
    # it to the bare noun took the castle out of the picture, and the pass no
    # longer edits meaning it was not asked to (docs/decisions/PROMPT-STYLE.md)
    assert v.prompt == "masterpiece, best quality, 1girl, castle ruins, golden hour"


def test_meta_tags_the_user_asked_for_are_kept():
    """The filter stops the LLM garnishing prompts; it does not overrule anyone."""
    data = {"variants": [{"prompt": "1girl, concept art, 8k", "negative_prompt": ""}]}
    asked = "concept art of a girl"
    assert OllamaClient._parse_variants(data, 1, "tags", asked)[0].prompt == "1girl, concept art"
    assert OllamaClient._parse_variants(data, 1, "tags")[0].prompt == "1girl"


def test_real_danbooru_resolution_tags_survive():
    data = {"variants": [{"prompt": "absurdres, highres, 1girl, 8k, uhd", "negative_prompt": ""}]}
    v = OllamaClient._parse_variants(data, 1, "tags")[0]
    assert v.prompt == "absurdres, highres, 1girl"


def test_prose_style_is_never_tag_filtered():
    data = {"variants": [{
        "prompt": "a highly detailed, cinematic portrait, 8k", "negative_prompt": ""}]}
    v = OllamaClient._parse_variants(data, 1, "prose")[0]
    assert v.prompt == "a highly detailed, cinematic portrait, 8k"


def test_prose_style_keeps_prompt_untouched():
    data = {"variants": [{"prompt": "a cat, a very big cat, sitting", "negative_prompt": ""}]}
    v = OllamaClient._parse_variants(data, 1, "prose")[0]
    assert v.prompt == "a cat, a very big cat, sitting"


def test_parse_variants_empty_raises():
    with pytest.raises(LlmError):
        OllamaClient._parse_variants({"variants": []}, k=3)
    with pytest.raises(LlmError):
        OllamaClient._parse_variants({}, k=3)


def _stub_chat(monkeypatch, seen):
    async def fake(self, system, user, images=None):
        seen.update(system=system, user=user, images=images)
        return {"variants": [{"prompt": "  grass bends left  ", "negative_prompt": ""}]}
    monkeypatch.setattr(OllamaClient, "_chat_json", fake)


def test_motion_sends_the_image_only_when_there_is_one(monkeypatch):
    # a vision model gets the first frame; a text-only one must not be sent
    # images at all, or Ollama errors instead of falling back
    seen: dict = {}
    _stub_chat(monkeypatch, seen)
    client = OllamaClient("http://x", "m")

    variant = asyncio.run(client.motion("wind", "a field at dusk", "B64"))
    assert variant.prompt == "grass bends left"
    assert seen["images"] == ["B64"]
    assert "a field at dusk" in seen["user"] and "wind" in seen["user"]

    asyncio.run(client.motion("wind", "a field at dusk", None))
    assert seen["images"] is None


def test_motion_tells_the_llm_when_no_wish_was_given(monkeypatch):
    # an empty box means "propose the ambient motion", not an empty instruction
    seen: dict = {}
    _stub_chat(monkeypatch, seen)
    asyncio.run(OllamaClient("http://x", "m").motion("   ", "a field at dusk"))
    assert "(not specified)" in seen["user"]


def test_edit_instruction_sends_the_image_only_when_there_is_one(monkeypatch):
    seen: dict = {}
    _stub_chat(monkeypatch, seen)
    client = OllamaClient("http://x", "m")

    asyncio.run(client.edit_instruction("blue mug", "a red mug on a table", "B64"))
    assert seen["images"] == ["B64"]
    assert "a red mug on a table" in seen["user"] and "blue mug" in seen["user"]
    assert seen["system"] == EDIT_INSTRUCTION_SYSTEM

    asyncio.run(client.edit_instruction("blue mug", "a red mug on a table"))
    assert seen["images"] is None


def test_edit_instruction_is_never_written_in_a_checkpoints_dialect(monkeypatch):
    """An order to an edit model, not a prompt: no opener, no tag cleaning.

    The quality opener is the trap here -- "masterpiece, best quality" welded
    onto the front of "make her smile" is an instruction to repaint the whole
    picture, and every other single-variant call in this class takes one.
    """
    seen: dict = {}

    async def fake(self, system, user, images=None):
        seen.update(system=system, user=user, images=images)
        return {"variants": [{
            "prompt": "  Change her dress to deep burgundy velvet.  ",
            "negative_prompt": ""}]}

    monkeypatch.setattr(OllamaClient, "_chat_json", fake)
    variant = asyncio.run(
        OllamaClient("http://x", "m").edit_instruction("red dress", "1girl, a park")
    )
    assert variant.prompt == "Change her dress to deep burgundy velvet."


def test_the_cleaning_pipeline_takes_the_garnish_and_nothing_else():
    """Blocklist, vocabulary and dedupe together, on the reported failure.

    "artstation" and "8k" are named folklore and still go. "1man" and "long
    grey hair" no longer move: the first is a rescue ("1boy") and the second a
    compound split, and both are changes of meaning rather than spelling. They
    are offered by Vocabulary.check instead. The pathology that made the old
    rewrite necessary -- "1boy, 1man" collapsing into a single figure -- was a
    tag-list artifact, and 108 rendered prose prompts got the subject count
    right 108 times (docs/decisions/PROMPT-STYLE.md).
    """
    out = OllamaClient._clean_tags("1girl, 1man, long grey hair, artstation, 8k", "")
    assert out == "1girl, 1man, long grey hair"


def test_a_compound_keeps_its_words_and_its_spelling_is_still_fixed():
    """Nothing here is an exact alias, so nothing here moves.

    "long grey hair" used to be broken into "long hair, grey hair" and "thigh
    highs" folded to "thighhighs". Both are reached by rescue rather than by
    Danbooru's alias table, which makes them rewrites of what was asked for
    rather than spelling fixes; check() offers them and nobody applies them.
    A one-word alias is also held back here on purpose: "thigh" IS an alias
    for "thighs", and firing it inside "thigh highs" gave "thighs highs".
    """
    out = OllamaClient._clean_tags("1girl, long grey hair, long hair, thigh highs", "")
    assert out == "1girl, long grey hair, long hair, thigh highs"
    # the exact alias in the same shape still moves
    assert OllamaClient._clean_tags("1girl, mini skirt", "") == "1girl, miniskirt"


def test_the_garnish_still_goes_even_when_the_rest_is_kept():
    """The garnish is named in META_TAGS, not dropped for being unrecognised.

    "volumetric lighting" has zero Danbooru posts and used to disappear only
    because repair dropped everything it did not recognise. That pass now
    keeps what it does not know, so the folklore is on the blocklist by name.
    """
    out = OllamaClient._clean_tags(
        "2boys, looking at each other, castle ruins, volumetric lighting", ""
    )
    assert out == "2boys, looking at each other, castle ruins"


def test_an_inlined_negative_prompt_is_cut_off_the_positive():
    """The floor must never be sent as things to PUT IN the picture.

    gemma3 answers the JSON field and then writes the negative into the
    positive one as a labelled tail as well -- seen on both variants of one
    request. Left in, the prompt asks the model for "bad hands, watermark,
    censored".
    """
    data = {"variants": [{
        "prompt": "1girl, on a rooftop at night.\nnegative_prompt: worst quality, bad hands",
        "negative_prompt": "worst quality, bad hands",
    }]}
    v = OllamaClient._parse_variants(data, 1, "tags")[0]
    assert v.prompt == "1girl, on a rooftop at night"
    assert v.negative_prompt == "worst quality, bad hands"


def test_the_mandated_opener_is_enforced_not_merely_requested():
    """BAS+'s card calls its trigger tags required; asking is not getting."""
    opener = "masterpiece, best quality, highres, score_7, score_8, realistic"
    # the observed failure: the opener's last word ran into the first content
    # word, leaving neither the opener nor the count tag intact
    data = {"variants": [{"prompt": f"{opener} 3girls, from below", "negative_prompt": ""}]}
    v = OllamaClient._parse_variants(data, 1, "tags", "", opener)[0]
    assert v.prompt == f"{opener}, 3girls, from below"
    # missing entirely, and already correct, both end up the same way
    for written in ("3girls, from below", f"{opener}, 3girls, from below"):
        data = {"variants": [{"prompt": written, "negative_prompt": ""}]}
        out = OllamaClient._parse_variants(data, 1, "tags", "", opener)[0]
        assert out.prompt == f"{opener}, 3girls, from below"
    # and a model with no opener of its own is left alone
    data = {"variants": [{"prompt": "a quiet street", "negative_prompt": ""}]}
    assert OllamaClient._parse_variants(data, 1, "prose")[0].prompt == "a quiet street"


def test_an_artist_the_user_did_not_ask_for_is_stripped():
    """The LLM signs nearly every tag prompt with somebody: 11 of 12, measured.

    Asking it not to is half the fix; this is the other half, and it is the
    same bargain as the rest of `_clean_tags` -- the LLM's garnish goes, the
    user's own words never do.
    """
    clean = OllamaClient._strip_attributions

    assert clean("1girl, a park, soft lighting, art by Ilya Kuvshinov.",
                 "a girl in a park") == "1girl, a park, soft lighting"
    # a whole sentence, which is how it arrives in prose-shaped tag prompts
    assert clean("A tall boy. The style is reminiscent of Artgerm.",
                 "a tall boy") == "A tall boy."
    for phrase in ("in the style of WLOP", "inspired by Alphonse Mucha",
                   "illustration by Kuvshinov", "evokes the work of Mucha",
                   "reminiscent of Ilya Kuvshinov", "in the style of Studio Ghibli"):
        assert clean(f"1girl, a castle, {phrase}, wide shot", "a castle") == \
            "1girl, a castle, wide shot", phrase


def test_only_a_named_someone_counts_as_an_attribution():
    """A style is not a signature, and the trigger words are ordinary English.

    "reminiscent of" and "inspired by" are how anyone describes a picture, so
    what makes them an attribution is the NAME after them -- which is why the
    second half of the pattern is case-sensitive. Styles stay either way:
    "Alphonse Mucha style" reads as a style rather than a signature, and the
    rule that would catch it also catches "realistic style".
    """
    clean = OllamaClient._strip_attributions
    for kept in (
        "1girl, a castle, reminiscent of a fairytale, soft lighting",
        "1girl, a dress inspired by the 1920s, wide shot",
        "1girl, in the style of a woodblock print, wide shot",
        "1girl, Alphonse Mucha style, realistic style, wide shot",
        "A tall boy. The style is loose and painterly.",
    ):
        assert clean(kept, "a girl") == kept


def test_an_artist_the_user_asked_for_survives():
    """`asked_for` is the escape hatch, exactly as it is for the meta tags."""
    clean = OllamaClient._strip_attributions
    text = "1girl, a castle, in the style of Alphonse Mucha, wide shot"
    assert clean(text, "a castle in the style of Alphonse Mucha") == text
    assert clean(text, "a castle") == "1girl, a castle, wide shot"
    # and the form the LLM writes back when the user does ask for one, which
    # was never a candidate for stripping in the first place
    bare = "1girl, a castle, Alphonse Mucha style, wide shot"
    assert clean(bare, "a castle") == bare


def test_a_prompt_with_no_attribution_is_returned_untouched():
    """The rejoin normalises spacing, so it must not run on prompts it has
    nothing to do with -- which is most of them."""
    text = "masterpiece, best quality, 1girl with red hair.  Sitting;  a park"
    assert OllamaClient._strip_attributions(text, "a girl") == text


def test_the_multi_subject_rules_reach_the_tag_writer(monkeypatch):
    """Two people in one prompt, and which of them has what.

    The rules are unenforceable in code -- nothing here can tell a well-bound
    prompt from a mixed-up one -- so this only guards the wiring: the tag
    template is what a tag model gets, and it carries the rules that keep each
    person's count word next to their own description. What the LLM does with
    them is measured, in docs/decisions/PROMPT-STYLE.md.
    """
    seen: dict = {}
    _stub_chat(monkeypatch, seen)
    client = OllamaClient("http://x", "m")

    asyncio.run(client.enrich("a girl and a boy", k=1, uses_negative_prompt=True,
                              style="tags"))
    system = seen["system"]
    assert '1boy and 1girl' in system          # the shape it is told not to write
    assert 'pronoun' in system
    assert 'as much description as the first' in system
    assert 'art by <name>' in system          # and no artist nobody asked for

    # and the prose writer is a different template that was never given them
    asyncio.run(client.enrich("a girl and a boy", k=1, uses_negative_prompt=True,
                              style="prose"))
    assert 'pronoun' not in seen["system"]


def test_the_length_budget_reaches_the_system_prompt():
    """The model is told what it has room for, and to front-load.

    Advisory everywhere: nothing in the app truncates a prompt, because past
    the budget the tail is not cut, it is merely too faint to matter (see
    docs/decisions/PROMPT-LENGTH.md). Telling the writer is the whole fix.
    """
    from app.schemas import PromptBudget

    text = OllamaClient._budget_instruction(PromptBudget(full=300, fading=700))
    assert "300 characters" in text
    assert "700 characters" in text
    assert "FIRST" in text            # the front-loading half
    assert OllamaClient._budget_instruction(None) == ""


def test_a_mixed_selection_writes_to_the_tightest_budget():
    """Unlike guidance and openers, a budget has a safe direction to fall back on."""
    from app.main import _budget_for, _opener_for

    reg_keys = ["bas-anima", "lustrousmix-anima"]
    tightest = min(_budget_for([k]).full for k in reg_keys)
    assert _budget_for(reg_keys).full == tightest
    # and the existing rule is unchanged: openers must agree or none is sent
    assert _opener_for(reg_keys) == ""
    assert _budget_for([]) is None


# -- failure reporting --------------------------------------------------------


def _timing_out_client(monkeypatch, exc, timeout=300):
    """An OllamaClient whose POST always raises `exc`; returns the call count."""
    calls = []

    class FakeAsyncClient:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, *a, **kw):
            calls.append(1)
            raise exc

    monkeypatch.setattr("app.llm.httpx.AsyncClient", FakeAsyncClient)
    return OllamaClient("http://x", "m", request_timeout=timeout), calls


def test_a_timeout_says_so_instead_of_stringifying_to_nothing(monkeypatch):
    """str(ReadTimeout()) is "", which is how this used to reach the UI."""
    import httpx

    client, _ = _timing_out_client(monkeypatch, httpx.ReadTimeout(""), timeout=42)
    with pytest.raises(LlmError) as err:
        asyncio.run(client._chat_json("sys", "user"))
    assert "42s" in str(err.value)
    assert str(err.value).strip().endswith(".")


def test_a_timeout_is_not_retried_into_a_second_full_wait(monkeypatch):
    import httpx

    client, calls = _timing_out_client(monkeypatch, httpx.ReadTimeout(""))
    with pytest.raises(LlmError):
        asyncio.run(client._chat_json("sys", "user"))
    assert len(calls) == 1


def test_a_refused_connection_names_the_thing_to_start(monkeypatch):
    import httpx

    client, calls = _timing_out_client(monkeypatch, httpx.ConnectError("nope"))
    with pytest.raises(LlmError) as err:
        asyncio.run(client._chat_json("sys", "user"))
    assert "ollama serve" in str(err.value)
    assert len(calls) == 2  # a refusal can be transient; a timeout cannot


def test_the_model_name_is_in_the_message(monkeypatch):
    import httpx

    client, _ = _timing_out_client(monkeypatch, httpx.ConnectError("nope"))
    with pytest.raises(LlmError) as err:
        asyncio.run(client._chat_json("sys", "user"))
    assert "m" in str(err.value)


def test_an_image_for_the_vision_model_is_capped_on_its_long_edge():
    """A 4K render costs several times the reading time and adds nothing."""
    import base64
    import io

    from PIL import Image

    from app.llm import VISION_MAX_EDGE, image_for_vision

    buf = io.BytesIO()
    Image.new("RGBA", (2048, 1024), (10, 20, 30, 255)).save(buf, format="PNG")
    out = Image.open(io.BytesIO(base64.b64decode(image_for_vision(buf.getvalue()))))
    assert out.size == (VISION_MAX_EDGE, VISION_MAX_EDGE // 2)  # aspect kept
    assert out.format == "JPEG"           # RGBA would not have survived as PNG


def test_a_small_image_is_not_enlarged_to_the_cap():
    import base64
    import io

    from PIL import Image

    from app.llm import image_for_vision

    buf = io.BytesIO()
    Image.new("RGB", (512, 384), (1, 2, 3)).save(buf, format="PNG")
    out = Image.open(io.BytesIO(base64.b64decode(image_for_vision(buf.getvalue()))))
    assert out.size == (512, 384)


# -- the simple post's caption -------------------------------------------------


def _stub_caption(monkeypatch, answer, seen):
    async def fake(self, system, user, images=None, schema=None):
        seen.update(system=system, user=user, images=images, schema=schema)
        return answer
    monkeypatch.setattr(OllamaClient, "_chat_json", fake)


def test_caption_asks_for_its_own_shape_and_sends_only_the_picture(monkeypatch):
    from app.llm import ARTWORK_SCHEMA
    seen: dict = {}
    _stub_caption(monkeypatch, {
        "title": "Green Hour", "description": " A girl waits\n under the trees. ",
        "tags": ["park", "anime"]}, seen)
    got = asyncio.run(OllamaClient("http://x", "m").caption("B64"))
    assert got.title == "Green Hour"
    assert got.description == "A girl waits under the trees."
    assert got.tags == ["park", "anime"]
    # the variants schema would have got a prompt back whatever the system said
    assert seen["schema"] is ARTWORK_SCHEMA
    assert seen["images"] == ["B64"]
    # nothing about the request reaches it: the prompt is what is being left out
    assert "1girl" not in seen["user"]


def test_the_caption_is_asked_for_an_admiring_voice_and_no_invention():
    """Reported: captions read "A person in ornate red robes sits ..." -- dry."""
    from app.llm import ARTWORK_SYSTEM
    text = ARTWORK_SYSTEM.lower()
    assert "stunningly beautiful" in text and "fairytale" in text
    assert "never add" in text and "visible in the picture" in text
    # the dry openings are named as what not to write
    assert '"a person"' in text and '"a figure"' in text


def test_a_title_dressed_as_one_is_undressed(monkeypatch):
    seen: dict = {}
    _stub_caption(monkeypatch, {
        "title": ' Title: "The Green Hour." ', "description": "x", "tags": []}, seen)
    assert asyncio.run(OllamaClient("http://x", "m").caption("B64")).title == "The Green Hour"


def test_caption_tags_are_lowercased_deduped_and_capped(monkeypatch):
    seen: dict = {}
    _stub_caption(monkeypatch, {
        "title": "t", "description": "d",
        "tags": ["Park.", "park", "a", "Anime!", "forest", "dusk", "trees", "rain"]}, seen)
    got = asyncio.run(OllamaClient("http://x", "m").caption("B64"))
    # "a" is too short to be a tag anyone browses by, "park" arrives twice
    assert got.tags == ["park", "anime", "forest", "dusk", "trees"]


def test_a_caption_the_model_botched_does_not_take_the_dialog_down(monkeypatch):
    seen: dict = {}
    _stub_caption(monkeypatch, {"title": "", "tags": "not a list"}, seen)
    got = asyncio.run(OllamaClient("http://x", "m").caption("B64"))
    assert (got.title, got.description, got.tags) == ("", "", [])
