import json

import pytest

from app import vocab
from app.vocab import Vocabulary, load


@pytest.fixture
def v():
    """A small stand-in for the shipped snapshot, in Danbooru's own format."""
    return Vocabulary(
        tags={
            "1girl": 8322035,
            "looking_at_viewer": 4855937,
            "cowboy_shot": 842881,
            "from_above": 144494,
            "fog": 7640,
            "ruins": 14896,
            "sheeta": 239,
            # real meta tags, and the ones the opener leans on
            "absurdres": 2944149,
            "highres": 8043469,
        },
        aliases={"mist": "fog", "aerial_view": "from_above"},
    )


@pytest.fixture
def people():
    """The part of the vocabulary that says who is in the picture."""
    return Vocabulary(
        tags={
            "1girl": 8322035, "2girls": 1422016, "1boy": 2104300,
            "2boys": 461417, "3boys": 104574, "multiple_boys": 688523,
            "2others": 14608, "looking_at_another": 430578,
            "leaning_on_person": 10878, "holding_hands": 144698,
            "park": 100, "sword": 100,
        },
        aliases={},
    )


# -- who is in the picture, and what they are doing with each other -----------


def test_a_count_the_model_never_saw_is_rewritten_not_dropped(people):
    # "1man" is not a Danbooru tag; dropping it costs a person, not an adjective
    assert people.rescue("1man") == "1boy"
    assert people.rescue("2 boys") == "2boys"
    assert people.rescue("three men") == "3boys"
    assert people.rescue("two people") == "2others"


def test_counts_past_the_end_of_the_scale_become_multiple(people):
    assert people.rescue("7 men") == "multiple boys"


def test_rescue_rewrites_but_never_invents(people):
    # 5boys is a real tag, but not in this snapshot: no snapshot, no rewrite
    assert people.rescue("five men") is None
    assert people.rescue("laputa princess") is None
    assert people.rescue("1girl") is None  # already the trained spelling


def test_two_count_words_are_left_alone_and_offered_instead(people):
    """Counts are no longer summed, because prompts are prose.

    The bug this replaces: "a knight and his squire" enriched to the tag list
    "1boy, 1man", the rescue folded "1man" to "1boy", and two identical tags
    collapsed into one figure -- so the merge summed them into "2boys". Prose
    writes "a knight and his squire" and the pathology has no way in. It is
    also not silently rewritten any more: 108 rendered prompts carrying count
    tags beside prose describing those subjects came back with the right
    number of people 108 times (docs/decisions/PROMPT-STYLE.md).
    """
    assert people.repair("1boy, 1man, sword") == "1boy, 1man, sword"
    assert people.repair("1boy, 2boys, sword") == "1boy, 2boys, sword"


def test_a_rewritable_count_is_offered_as_a_fix_not_reported_unknown(people):
    """Still surfaced -- as a choice, since "1man" -> "1boy" is a rescue."""
    r = people.check("1girl, 1man, park")
    assert r["suggestions"] == [["1man", ["1boy"]]]
    assert r["aliases"] == []


def test_prompts_use_spaces_and_danbooru_uses_underscores(v):
    assert v.count("cowboy shot") == 842881
    assert v.count("cowboy_shot") == 842881
    assert v.count("Cowboy Shot") == 842881


def test_weighting_syntax_is_not_part_of_the_name(v):
    assert v.count("(cowboy shot:1.2)") == 842881
    assert v.count("(1girl)") == 8322035


def test_a_word_the_snapshot_does_not_have_is_not_a_complaint(v):
    """Absence alone stopped being worth saying when prompts became prose.

    "laputa princess" is an invented character tag and "highly detailed" is
    Stable Diffusion folklore, and neither is answerable: the snapshot has no
    tag that means them. Saying "not in the vocabulary, remove it" about an
    LLM-written prose prompt fills the composer with yellow that the
    measurements say to ignore (Vocabulary.check). The folklore is still
    caught -- by the garnish list, which is measured and is a different claim.
    """
    r = v.check("1girl, laputa_princess, highly detailed, ruins")
    assert r == {"aliases": [], "suggestions": []}


def test_aliases_point_at_the_spelling_the_model_saw(v):
    r = v.check("1girl, mist, aerial view")
    assert r["aliases"] == [["mist", "fog"], ["aerial view", "from above"]]
    assert r["suggestions"] == []


def test_quality_tokens_are_not_danbooru_tags_and_not_mistakes(v):
    """"masterpiece" has zero Danbooru posts but opens Illustrious's own prompt."""
    r = v.check("masterpiece, best quality, very aesthetic, absurdres, 1girl")
    assert r == {"aliases": [], "suggestions": []}
    assert v.repair("masterpiece, best quality, 1girl") == "masterpiece, best quality, 1girl"


def test_repair_swaps_exact_aliases_and_keeps_everything_else(v):
    """The reversal: dropping unrecognised words was the old "whole point".

    It cost meaning the vocabulary cannot carry -- disciplining "green
    raincoat" to the only real tag, "raincoat", lost the colour in 6 renders
    of 6 (docs/decisions/PROMPT-STYLE.md). "mist" is an exact alias for "fog",
    so Danbooru itself says they are one tag and the swap is safe. Nothing
    else here is, so nothing else moves -- including "artstation", which is
    real folklore but is llm._strip_meta_tags' job, not this pass's.
    """
    out = v.repair("1girl, mist, laputa_princess, cowboy shot, artstation")
    assert out == "1girl, fog, laputa_princess, cowboy shot, artstation"


def test_repair_keeps_anything_the_user_asked_for(v):
    """The vocabulary stops the LLM inventing tags; it does not overrule anyone."""
    assert v.repair("1girl, laputa_princess", keep="laputa_princess please") == \
        "1girl, laputa_princess"


def test_a_short_sentence_survives_repair(v):
    """Several models' guidance allows one plain sentence after the tags."""
    prose = "she is holding the lantern above her head"
    assert prose in v.repair(f"1girl, cowboy shot, {prose}")
    assert v.check(f"1girl, {prose}") == {"aliases": [], "suggestions": []}


def test_no_vocabulary_means_no_opinions():
    """A fresh clone has no snapshot; that must cost the checks, not the app."""
    empty = Vocabulary({}, {})
    assert not empty
    assert empty.check("1girl, laputa_princess") == {"aliases": [], "suggestions": []}
    assert empty.repair("1girl, laputa_princess") == "1girl, laputa_princess"


def test_load_survives_a_missing_or_broken_snapshot(tmp_path):
    assert not load(tmp_path / "nope.json")
    broken = tmp_path / "broken.json"
    broken.write_text("{not json")
    assert not load(broken)


def test_shipped_snapshot_knows_the_tags_the_guidance_recommends():
    """config/models.yaml tells the LLM to use these; they had better exist."""
    shipped = vocab.load()
    if not shipped:  # snapshot not fetched in this checkout
        pytest.skip("no config/danbooru_tags.json; run scripts/fetch_tags.py")
    for tag in ("1girl", "cowboy shot", "portrait", "upper body", "full body",
                "looking at viewer", "from above", "from behind"):
        assert shipped.count(tag), tag
    # and the tags rescue() rewrites into, which are useless if absent
    for tag in ("1boy", "2boys", "2girls", "multiple boys", "2others",
                "looking at another", "face-to-face", "leaning on person",
                "holding hands", "hug", "arm around shoulder", "side-by-side"):
        assert shipped.count(tag), tag
    # and the folklore the measurements pinned as the cause of flat output
    for tag in ("highly detailed", "intricate details", "artstation",
                "detailed background", "volumetric lighting"):
        assert not shipped.count(tag), tag


def test_snapshot_file_is_the_shape_the_fetch_script_writes():
    if not vocab.VOCAB_PATH.is_file():
        pytest.skip("no config/danbooru_tags.json; run scripts/fetch_tags.py")
    data = json.loads(vocab.VOCAB_PATH.read_text())
    assert data["tags"] and data["aliases"] and data["fetched_at"]
    assert all(isinstance(c, int) for c in list(data["tags"].values())[:50])


def test_a_plural_danbooru_keeps_singular_is_folded_not_dropped(v):
    # "ruins" is in the fixture as itself; "sheeta" has no plural to fold
    small = Vocabulary(tags={"shadow": 100, "hair_ribbon": 100, "boots": 100,
                             "butterfly": 100, "wolf": 100}, aliases={})
    assert small.rescue("shadows") == "shadow"
    assert small.rescue("butterflies") == "butterfly"
    assert small.rescue("wolves") == "wolf"
    # modifiers are kept: the head noun alone would be a different picture
    assert small.rescue("hair ribbons") == "hair ribbon"
    # a plural Danbooru keeps is left exactly as it is
    assert small.rescue("boots") is None


def test_folding_never_drops_words_to_find_a_hit(v):
    """A modifier may be lost to a fold; the subject of the phrase may not."""
    small = Vocabulary(tags={"field": 100, "dress": 100, "clothes": 100}, aliases={})
    # "field" is a real tag, and this phrase is not about a field
    assert small.rescue("shallow depth of field") is None
    # these do fold: a garment without its adjective beats no garment
    assert small.rescue("gothic dress") == "dress"
    assert small.rescue("casual clothes") == "clothes"


@pytest.fixture
def compounds():
    return Vocabulary(
        tags={"jacket": 1474676, "skirt": 2158389, "grey_hair": 1016062,
              "sky": 613144, "leaf": 100, "hat": 100, "field": 100,
              "dress": 1930805, "cape": 300491},
        aliases={},
    )


def test_a_compound_folds_to_the_noun_the_model_knows(compounds):
    assert compounds.rescue("rain jacket") == "jacket"
    assert compounds.rescue("tight skirt") == "skirt"
    assert compounds.rescue("overcast sky") == "sky"


def test_folding_keeps_the_longest_suffix_that_is_a_tag(compounds):
    # "hair" alone is not a tag here, and "grey hair" says more than it anyway
    assert compounds.rescue("long grey hair") == "grey hair"


def test_folding_reaches_through_a_plural(compounds):
    assert compounds.rescue("wet leaves") == "leaf"


def test_a_preposition_means_the_tail_is_not_the_subject(compounds):
    # the trap: "field" is a real tag and this phrase is not about a field
    assert compounds.rescue("shallow depth of field") is None


def test_a_negation_is_never_folded_into_its_opposite(compounds):
    assert compounds.rescue("no hat") is None


def test_folklore_stays_deleted_because_its_head_is_not_a_tag(compounds):
    # "lighting" and "background" are not Danbooru tags, so there is nothing
    # to fold to -- which is what keeps the measured-harmful tags out
    assert compounds.rescue("volumetric lighting") is None
    assert compounds.rescue("detailed background") is None


@pytest.fixture
def wardrobe():
    return Vocabulary(
        tags={"miniskirt": 381550, "skirt": 2158389, "black_skirt": 260905,
              "long_hair": 4000000, "grey_hair": 1016062, "hair": 0,
              "black_jacket": 500000, "leather_jacket": 100000, "jacket": 1474676,
              "thighhighs": 1520819, "serious": 39473, "toned": 60033,
              "blurry": 308765, "blurry_background": 300000},
        aliases={"mini_skirt": "miniskirt"},
    )


def test_the_longest_suffix_wins_over_the_bare_noun(wardrobe):
    # "skirt" is right there and would lose the mini
    assert wardrobe.rescue("dark navy mini skirt") == "miniskirt"


def test_danbooru_closed_up_spellings_are_found(wardrobe):
    assert wardrobe.rescue("thigh highs") == "thighhighs"


def test_every_modifier_gets_its_own_tag(wardrobe):
    assert wardrobe.rescue("long grey hair") == "long hair, grey hair"
    assert wardrobe.rescue("black leather jacket") == "black jacket, leather jacket"


def test_a_word_alone_when_nothing_at_the_end_is_a_tag(wardrobe):
    assert wardrobe.rescue("serious expression") == "serious"


def test_a_word_danbooru_spells_differently_lands_the_phrase(wardrobe):
    assert wardrobe.rescue("blurred background") == "blurry background"


def test_a_word_danbooru_does_not_have_at_all_becomes_the_one_it_does(wardrobe):
    # "athletic" is not a tag in any spelling; "toned" is what it means here
    assert wardrobe.rescue("athletic build") == "toned"


def test_a_counted_animal_is_offered_the_plain_noun():
    """Danbooru counts people and nothing else, so "1cat" is not a word.

    An LLM told to write 1girl/2boys reaches for the same shape when it gets
    to the animals; observed on gemma3 the first time the prose guidance ran.
    Offered by check(), never applied -- the counts for people are untouched.
    """
    v = vocab.get()
    assert v.rescue("1cat") == "cat"
    assert v.rescue("2cats") == "cat"
    assert v.rescue("1girl") is None
    assert v.rescue("1boy") is None
    assert v.check("1girl, 1cat")["suggestions"] == [["1cat", ["cat"]]]


def test_a_stack_of_modifiers_keeps_its_modifiers(wardrobe):
    """The old pass answered this with a bare "miniskirt".

    That threw away the colour and the shadow -- the same failure mode as
    "green raincoat". Nothing is rewritten inside the segment either: an alias
    is only applied when it is the whole of one, because "sitting on" is an
    alias for "sitting on person" and firing it inside a clause gave "sitting
    on person a rooftop railing".
    """
    assert wardrobe.repair("shadow dark navy mini skirt") == "shadow dark navy mini skirt"
    # on its own, the same phrase is the tag and is respelled
    assert wardrobe.repair("1girl, mini skirt") == "1girl, miniskirt"
    # and the sentence the guidance allows is still untouchable
    prose = "she is holding the lantern above her head"
    assert prose in wardrobe.repair(f"1girl, {prose}")


# -- against the shipped snapshot, not a fixture --------------------------------
#
# The fixtures above pin the rules; these pin the answers a user actually gets,
# and would catch a snapshot refetch that moved an alias out from under one.


def shipped_or_skip():
    shipped = vocab.load()
    if not shipped:
        pytest.skip("no config/danbooru_tags.json; run scripts/fetch_tags.py")
    return shipped


@pytest.mark.parametrize("written,expected", [
    ("shadow dark navy mini skirt", "miniskirt"),   # alias, and past the prose rule
    ("mini skirt", "miniskirt"),                    # Danbooru's own redirect
    ("thigh highs", "thighhighs"),                  # closed-up, no alias for it
    ("long grey hair", "long hair, grey hair"),     # every modifier keeps a tag
    ("black leather jacket", "black jacket, leather jacket"),
    ("athletic build", "toned"),                    # no spelling of it exists
    ("blurred background", "blurry background"),    # synonym lands the phrase
    ("wet leaves", "leaf"),                         # fold reaching through a plural
    ("serious expression", "serious"),              # nothing at the end is a tag
    ("1man", "1boy"),                               # a person, not an adjective
    ("two people", "2others"),
    ("looking at each other", "looking at another"),
    ("no hat", "missing headwear"),                 # the negation is kept, not inverted
])
def test_shipped_snapshot_rewrites_what_it_should(written, expected):
    assert shipped_or_skip().rescue(written) == expected


@pytest.mark.parametrize("written", [
    "volumetric lighting", "dramatic lighting", "rim lighting", "ambient lighting",
    "detailed background", "sharp focus", "highly detailed", "crisp lines",
    "shallow depth of field",   # the preposition guard, on the real vocabulary
])
def test_shipped_snapshot_still_deletes_the_measured_folklore(written):
    """The 2048-correlation work removed these; no rewrite may bring them back."""
    assert shipped_or_skip().rescue(written) is None


def test_shipped_snapshot_never_answers_with_a_tag_it_does_not_have():
    shipped = shipped_or_skip()
    written = [
        "1man", "two people", "long grey hair", "athletic build", "thigh highs",
        "laputa princess", "shadow dark navy mini skirt", "wet leaves",
        "looking at each other", "blurred background", "gothic dress",
    ]
    for tag in written:
        for piece in (shipped.rescue(tag) or "").split(","):
            if piece.strip():
                assert shipped.known(piece.strip()), f"{tag} -> {piece}"


def test_an_empty_snapshot_rescues_nothing():
    """A fresh clone has no vocabulary; every path must fall silent, not guess."""
    empty = Vocabulary({}, {})
    for tag in ("1man", "long grey hair", "athletic build", "thigh highs"):
        assert empty.rescue(tag) is None


# -- the user's own words: respelled, never rewritten ---------------------------


def test_the_users_own_spelling_is_corrected_to_the_trained_one(wardrobe):
    """The reported case: "mini skirt" in the instruction shielded the LLM's
    "mini skirt" from every rewrite, so the model was shown a tag it has never
    seen. Spelling is not a change of meaning, so it goes through."""
    assert wardrobe.repair("mini skirt", "large girl in mini skirt") == "miniskirt"


def test_a_word_the_user_invented_is_still_never_touched(v):
    """Shielding is about not overruling anyone; that part does not change."""
    assert v.repair("1girl, laputa princess", "a laputa princess") == "1girl, laputa princess"


def test_a_synonym_is_a_different_word_so_it_is_never_applied(wardrobe):
    """"toned" is what "athletic build" means, and meaning is not spelling.

    This used to be swapped in silently whenever the LLM was the author. It is
    now offered by check() and applied by nobody -- the case that made the
    rule: the shipped snapshot answers "silver hair" with six hair tags and
    neither of the two right ones (grey hair, white hair), so an eager
    replacer makes that prompt worse rather than better.
    """
    assert wardrobe.repair("athletic build", "girl with athletic build") == "athletic build"
    assert wardrobe.repair("athletic build", "") == "athletic build"
    assert wardrobe.check("athletic build")["suggestions"] == [["athletic build", ["toned"]]]


def test_shipped_snapshot_respells_the_reported_instruction():
    shipped = shipped_or_skip()
    instruction = "large girl in mini skirt, see-through clothes, crouch over small boy"
    written = "masterpiece, best quality, 1girl, long hair, mini skirt, see-through clothes"
    assert shipped.repair(written, instruction) == (
        "masterpiece, best quality, 1girl, long hair, miniskirt, see-through clothes"
    )


# -- suggestions ---------------------------------------------------------------
#
# A word the models have no spelling of, but do have a tag for. Separate from
# an alias because the tag on the right is a different word, so applying it is
# a decision about meaning and belongs to the user.


def test_a_bare_gender_word_offers_the_count_tags():
    """The case this was built for: Danbooru counts people, it does not name
    their sex, so "male" has to become "1boy" and nobody can guess how many."""
    v = vocab.get()
    assert v.suggest("male")[:2] == ["1boy", "male focus"]
    assert v.suggest("female")[0] == "1girl"
    assert "multiple boys" in v.suggest("guy")


def test_the_plurals_and_the_other_words_for_it_all_land():
    v = vocab.get()
    for word in ("male", "males", "guy", "guys", "dude", "boy"):
        assert v.suggest(word)[0] == "1boy", word
    for word in ("woman", "women", "lady", "ladies", "female", "females", "girl"):
        assert v.suggest(word)[0] == "1girl", word


def test_a_word_danbooru_already_redirects_is_left_to_the_alias():
    """"man" is an alias of "male focus" on Danbooru itself.

    check() reports it as a spelling to fix, which is both more certain and
    applied with one click, so a suggestion would only be a second, worse
    answer to the same question.
    """
    v = vocab.get()
    assert v.aliases["man"] == "male focus"
    assert v.suggest("man") == []
    assert ["man", "male focus"] in v.check("1girl, man")["aliases"]


def test_a_word_the_vocabulary_has_no_answer_for_gets_none():
    """Offering a near-miss is worse than saying there is nothing.

    Danbooru has no tag meaning "gorgeous" or "elegant", and the lexical
    near-miss search that would have invented one is documented in
    docs/decisions/VOCABULARY.md as measured and rejected.
    """
    v = vocab.get()
    assert v.suggest("gorgeous") == []
    assert v.suggest("elegant") == []
    assert v.suggest("beautiful") == []


def test_a_tag_that_is_already_a_tag_is_never_suggested_over():
    v = vocab.get()
    assert v.suggest("1girl") == []
    assert v.suggest("sunset") == []  # a listed word that IS the tag


def test_every_suggestion_exists_in_the_snapshot():
    """The tables can go stale; the snapshot is the authority.

    A suggestion pointing at a tag the models never saw would be worse than
    the unknown word it replaced.
    """
    v = vocab.get()
    for word, tags in v.suggestion_map().items():
        for tag in tags:
            assert tag in v.counts, f"{word} -> {tag} is not in the snapshot"


def test_check_separates_a_suggestion_from_a_spelling_and_from_nothing():
    v = vocab.get()
    res = v.check("1girl, male, thigh high stockings, gorgeous")
    # a rescue is offered, not presented as a spelling: only an exact alias is
    # one, and "thigh high stockings" is reached by breaking the phrase up
    assert ["thigh high stockings", ["thighhighs"]] in res["suggestions"]
    assert "male" in [row[0] for row in res["suggestions"]]
    assert dict(res["suggestions"])["male"][0] == "1boy"
    # "gorgeous" has no answer in the snapshot, so there is nothing to say
    # about it here; it is the garnish list's business, not the vocabulary's
    assert "gorgeous" not in [row[0] for row in res["suggestions"]]


def test_a_rescue_that_drops_a_word_is_not_offered():
    """The green-raincoat rule, applied to what check() is allowed to say.

    Disciplining "green raincoat" to the only real tag, `raincoat`, lost the
    colour in 6 renders of 6 (docs/decisions/PROMPT-STYLE.md), which is what
    made `vtags` the worst of the five arms measured. A rewrite that reaches
    inside a phrase and comes back with less than it went in with is that
    failure, so it is no longer offered -- and on an LLM-written prose prompt
    that is most of what the composer used to say.
    """
    v = vocab.get()
    for phrase in ("green raincoat", "gentle smile", "green field", "plain dress",
                   "dust motes", "rolling hills", "bright sunlight", "simple clothes"):
        # the rescue still exists; check() is what stopped volunteering it
        assert v.rescue(phrase), f"{phrase} is expected to be rescuable"
        assert v.check(phrase) == {"aliases": [], "suggestions": []}, phrase


def test_a_rescue_that_keeps_every_word_is_still_offered():
    """The other half: losing nothing is what makes a rewrite safe to show.

    Two shapes qualify. One replaces the whole phrase, so nothing of it was
    kept to be dropped ("two girls", "thigh high stockings"); the other uses
    every word it was given ("long grey hair" -> `long hair, grey hair`).
    """
    v = vocab.get()
    assert ["two girls", ["2girls"]] in v.check("two girls")["suggestions"]
    assert ["long grey hair", ["long hair, grey hair"]] in \
        v.check("long grey hair")["suggestions"]
    assert ["thigh high stockings", ["thighhighs"]] in \
        v.check("thigh high stockings")["suggestions"]


def test_the_word_overlap_search_does_not_volunteer_itself():
    """It stays in the type-ahead, where somebody asked; not in the report.

    docs/decisions/VOCABULARY.md argues for a generous search on the grounds
    that "a row you can see is wrong costs a glance". That holds for a picker
    a person opened. It does not hold for a line that appears under a prompt
    nobody asked about, where the same search answers "silver hair" with six
    hair tags and neither right one, and "wildflowers scattered around" with
    an unrelated clothing tag.
    """
    v = vocab.get()
    assert v.searched_alternatives("silver hair"), "the search itself still works"
    assert v.check("silver hair") == {"aliases": [], "suggestions": []}
    assert v.check("wildflowers scattered around") == {"aliases": [], "suggestions": []}
    # and it is still reachable where it was asked for



def test_keeps_every_word_folds_a_plural():
    """Danbooru keeps the singular, so the fold has to run on both sides.

    Without it "rolling hills" -> `hill` shares no word with what it replaced
    and would read as a whole-phrase respelling, which is exactly the reading
    that lets a dropped word through.
    """
    keeps = Vocabulary.keeps_every_word
    assert keeps("shadows", "shadow")
    assert not keeps("rolling hills", "hill")
    assert keeps("long grey hair", "long hair, grey hair")
    assert not keeps("green raincoat", "raincoat")
    assert keeps("two girls", "2girls")


def test_a_suggestion_is_reported_once_however_often_it_is_typed():
    v = vocab.get()
    res = v.check("male, 1girl, male, male")
    assert len(res["suggestions"]) == 1


def test_the_suggestion_map_is_proportionate_to_what_the_page_already_loads():
    """It ships whole on page load, which is what makes the type-ahead work
    without a round trip per keystroke.

    The budget is the tag list beside it: /api/vocab/tags is ~310KB of the
    same JSON, over localhost, so this is allowed to be of that order and no
    more. It grew from 2KB to ~290KB when the wiki's other names landed,
    which is the whole point of pinning it -- the next thing that lands
    should have to argue for its bytes too.
    """
    import json

    v = vocab.get()
    ranked = sorted(v.counts.items(), key=lambda kv: -kv[1])[:12000]
    tags_payload = len(json.dumps([[n, c] for n, c in ranked]))
    assert len(json.dumps(v.suggestion_map())) < tags_payload * 1.5


def test_no_vocabulary_means_no_suggestions_either():
    assert Vocabulary({}, {}).suggest("male") == []
    assert Vocabulary({}, {}).suggestion_map() == {}


# -- other names, from the tag wiki --------------------------------------------


def _with_other_names():
    """A small vocabulary carrying the wiki's other names, as load() does."""
    return Vocabulary(
        {"1girl": 8322055, "1boy": 2104300, "male focus": 1115213,
         "multiple boys": 688523, "multiple girls": 2090716,
         "dirt road": 90000, "glasses": 500000},
        {"dirt track": "dirt road"},
        {"1girl": ["girl", "woman", "female"], "1boy": ["male"],
         "dirt_road": ["dirt path", "country road"],
         "glasses": ["eyeglasses", "spectacles"]},
    )


def test_the_wiki_answers_the_word_the_alias_table_never_had():
    """Danbooru has no alias for "female"; its 1girl wiki page has it.

    The alias table is spellings of a tag. The wiki's other_names is what
    people call it, which is a different and more useful thing here.
    """
    v = _with_other_names()
    assert v.suggest("female")[0] == "1girl"
    assert v.suggest("woman")[0] == "1girl"
    assert v.suggest("dirt path") == ["dirt road"]
    assert v.suggest("spectacles") == ["glasses"]


def test_the_wiki_and_the_count_tags_are_merged_not_chosen_between():
    """The wiki gives "1boy" for "male" but neither the plural nor the focus
    tag, and it has nothing at all for "guy"."""
    v = _with_other_names()
    assert v.suggest("male") == ["1boy", "male focus", "multiple boys"]
    assert v.suggest("guy")[0] == "1boy"


def test_an_other_name_pointing_at_a_tag_we_do_not_ship_is_dropped():
    v = Vocabulary({"1girl": 10}, {}, {"laputa_princess": ["sheeta"], "1girl": ["girl"]})
    assert v.suggest("sheeta") == []
    assert v.suggest("girl")[0] == "1girl"


def test_a_name_belonging_to_several_tags_offers_the_best_learned_first():
    v = Vocabulary(
        {"1girl": 8322055, "solo": 6990312},
        {},
        {"solo": ["single"], "1girl": ["single"]},
    )
    assert v.suggest("single") == ["1girl", "solo"]


def test_a_snapshot_without_other_names_still_loads():
    """The committed snapshot predates the wiki pass; it must not break.

    The count tables still answer, which is the point of keeping them: the
    wiki pass adds coverage, it is not load-bearing.
    """
    v = Vocabulary({"1girl": 10}, {})
    assert v.called == {}
    assert v.suggest("female") == ["1girl"]


# -- the bare-modifier search --------------------------------------------------


def test_a_bare_modifier_offers_the_tags_that_start_with_it():
    """"small" is not a tag: Danbooru names what is small, not smallness."""
    v = vocab.get()
    assert "small horns" in v.suggest("small")
    assert "long hair" in v.suggest("long")
    assert "open mouth" in v.suggest("open")
    assert "bare shoulders" in v.suggest("bare")


def test_the_bare_modifier_search_cannot_reach_the_old_failures():
    """The whole-word-first rule is what keeps this out of the rejects.

    A head-noun or substring search answered "stockings" with "king" and
    "pony tail" with "cat tail"; see docs/decisions/VOCABULARY.md.
    """
    v = vocab.get()
    assert "king" not in v.suggest("stockings")
    assert v.suggest("pony tail") != ["cat tail"]
    assert v.suggest("gorgeous") == []


def test_a_closed_up_form_is_still_reachable():
    """No tag starts with the word "pony"; "ponytail" starts with the letters.

    Tested on the search itself rather than through suggest(), because the
    wiki has its own answer for "pony" ("pony (animal)") and a recorded name
    outranks a search -- which is the precedence this file pins elsewhere.
    """
    v = vocab.get()
    assert "ponytail" in v._closed_up("pony", 5)
    assert v.named_alternatives("pony") == ["pony (animal)"]


def test_a_phrase_gets_its_head_noun_searched_too():
    """Loose on purpose: a person reads these and picks.

    rescue() gets first refusal on a compound and usually answers it, so the
    search only sees phrases nothing else could place -- and there, tags
    sharing the head noun are the only lead there is.
    """
    v = vocab.get()
    assert "shaded face" in v.suggest("angry face")
    assert v.suggest("small fluffy thing") == []  # no tag is built from "thing"


# -- the aesthetic words -------------------------------------------------------


def test_the_quality_words_are_a_subset_of_the_blocklist():
    """They are still garnish; they just have an answer the rest do not.

    The composer needs both: the blocklist to strip them, and this subset to
    say what to use instead, which for these is nothing -- the checkpoint's
    own quality opener already carries it.
    """
    from app.llm import META_TAGS
    from app.vocab import QUALITY_FOLKLORE

    assert QUALITY_FOLKLORE <= META_TAGS
    assert {"beautiful", "gorgeous", "stunning"} <= QUALITY_FOLKLORE
    # a technique or a marketplace is not an aesthetic judgement and has no
    # such answer, so it must not be in here
    assert not QUALITY_FOLKLORE & {"artstation", "unreal engine", "8k", "cinematic"}


def test_an_aesthetic_word_is_not_specially_silenced():
    """These used to be excluded from the search on the grounds that a
    near-miss is worse than nothing. It is not, in a list you pick from:
    "pretty" turning up the Pretty Cure franchise tag is obviously not what
    you meant, and you can see that it is obviously not what you meant.

    Most of them still come back empty simply because no tag is built from
    the word, which is the honest answer arriving on its own rather than
    being enforced. What actually answers them is the quality opener, which
    the composer says in the garnish line.
    """
    v = vocab.get()
    for word in ("beautiful", "gorgeous", "stunning"):
        assert v.suggest(word) == [], word
    assert v.suggest("pretty") == ["pretty series"]


def test_a_recorded_name_beats_splitting_the_compound_up():
    """The bug this precedence exists for.

    rescue()'s last resort is breaking a compound into the tags it can find
    inside it, which for "dirt path" found a bare "path" and dropped what
    kind. The wiki records the whole phrase as another name for "dirt road",
    and that is plainly the better answer, so a recorded name is tried before
    rescue.
    """
    v = vocab.get()
    assert v.named_alternatives("dirt path") == ["dirt road"]
    res = v.check("dirt path")
    assert res["suggestions"] == [["dirt path", ["dirt road"]]]
    assert res["aliases"] == []


def test_the_search_does_not_beat_rescue():
    """The other half of the same rule, and why the two are separate.

    A plural is better folded to its singular than answered with whatever
    happens to begin with it, so the search is the last thing tried, after
    rescue has had its go.
    """
    v = vocab.get()
    # "shadows" is in Danbooru's own alias table, so it stays a spelling fix;
    # the two below are only reachable by rescue, so they are offered instead
    assert ["shadows", "shadow"] in v.check("shadows")["aliases"]
    assert ["two girls", ["2girls"]] in v.check("two girls")["suggestions"]
    assert ["long grey hair", ["long hair, grey hair"]] in \
        v.check("long grey hair")["suggestions"]
