"""The tag vocabulary a Danbooru-trained model actually knows.

config/danbooru_tags.json is a snapshot of Danbooru's tag list with post counts
and aliases (scripts/fetch_tags.py builds it). Post count is how many images
the model saw with a tag, which makes it the difference between a tag that
steers the image and one that does nothing at all:

    cowboy_shot         842,881 posts   strongly learned
    ruins                14,896 posts   solid
    artstation                1 post    noise
    highly_detailed           0 posts   the model has never seen it
    laputa_princess       absent        not a tag in the first place

Prompts are written with spaces, Danbooru stores underscores, and prompts
weight tags with parentheses -- so everything here normalises before it looks
anything up.

Two jobs, one vocabulary, and prompts are prose now
(docs/decisions/PROMPT-STYLE.md), which decides how much either may do.

`repair` cleans prompts the LLM wrote, silently, because the LLM is not the
author -- but silently is a licence to fix spelling, not to edit meaning. It
applies an exact alias when that alias is the whole of a segment, and changes
nothing else. It used to drop every segment it did not recognise, which under
prose deletes ordinary English: "green raincoat" became "raincoat" and the
colour went with it.

`check` is the advisory layer and reports without touching anything, so it can
afford to say more: spellings to swap, tags offered for a word that is not one,
and words nothing can be offered for. What it reports is never applied without
a click. A rescue -- reaching a tag by breaking a phrase up -- is offered here
rather than applied there, because "laputa princess" -> "princess" is a
different request, not a spelling.

Both fall back to doing nothing when the snapshot is missing: a fresh clone has
no vocabulary until the script is run, and that must cost the checks rather
than the app.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

VOCAB_PATH = Path(__file__).resolve().parent.parent / "config" / "danbooru_tags.json"

# The subset of META_TAGS that is an aesthetic judgement rather than a
# technique or a marketplace. These have an answer, which is why they are named
# separately: the checkpoints carry their own aesthetic buckets ("masterpiece",
# "best quality", "very aesthetic") and every tag prompt this app sends already
# opens with the ones its model was fine-tuned on (quality_prefix in
# config/models.yaml). So "beautiful" is not a word the models cannot hear --
# it is one they have already been told, and repeating it in Danbooru's own
# vocabulary only dilutes the tags that name something.
QUALITY_FOLKLORE = frozenset(
    {
        "masterful",
        "stunning",
        "beautiful",
        "gorgeous",
        "perfect",
        "flawless",
        "pretty",
        "elegant",
        "exquisite",
        "breathtaking",
    }
)


# Quality and rating tokens the checkpoints add during training, which are NOT
# Danbooru post tags and would otherwise be reported as unknown -- "masterpiece"
# has zero posts on Danbooru and is deprecated there, yet it opens Illustrious's
# own recommended prompt. These come from the aesthetic buckets a checkpoint is
# fine-tuned with, so the vocabulary cannot see them and they are listed here.
# Kept deliberately short: it is an exception list, not a second vocabulary.
QUALITY_TOKENS = frozenset(
    {
        "masterpiece",
        "best quality",
        "amazing quality",
        "great quality",
        "good quality",
        "normal quality",
        "very aesthetic",
        "aesthetic",
        "high quality",
        "worst quality",
        "low quality",
        "bad quality",
        "worst detail",
        "newest",
        "recent",
        "late",
        "mid",
        "early",
        "oldest",
        "general",
        "sensitive",
        "questionable",
        "explicit",
        "safe",
        "nsfw",
        "sfw",
        "rating_safe",
        "rating_explicit",
        "score_1",
        "score_2",
        "score_3",
        "score_4",
        "score_5",
        "score_6",
        "score_7",
        "score_8",
        "score_9",
        "source_anime",
        "source_cartoon",
        "source_pony",
        "source_furry",
        # BAS+ requires "anime, anime_coloring, anime_screenshot" as trigger
        # tags. The last two are real Danbooru tags; a bare "anime" is not, and
        # was being offered them as replacements for a word its own card says
        # must be there.
        "anime",
    }
)
# Written above with underscores, the way the checkpoints' own cards write
# them, and compared against normalise() output, which uses spaces. Without
# this every underscored entry silently never matched: "score_7" is part of
# CyberRealistic's required opener and was being reported as an unknown word,
# and "score_1" in its negative was answered with "overwatch 1".
QUALITY_TOKENS = frozenset(t.replace("_", " ") for t in QUALITY_TOKENS)

# A comma segment carrying a function word is a clause, not a tag, and length
# is not the signal. "she is holding the lantern above her head" is prose;
# "shadow dark navy mini skirt" is five words of tag that happen to be stacked,
# and treating that as prose left it in the prompt as text the model cannot
# read. A clause needs at least one of these; a stack of modifiers has none.
# Several of these do appear inside real tags ("arm behind back", "looking at
# viewer"), which is why check() looks a segment up as a tag before it asks
# whether the segment is prose.
_FUNCTION_WORDS = frozenset(
    {"a", "an", "the", "is", "are", "was", "were", "be", "being", "been",
     "her", "his", "its", "their", "our", "your", "my", "she", "he", "they",
     "it", "who", "that", "which", "while", "as", "than", "then", "there",
     "has", "have", "had", "does", "do", "did", "can", "will", "would",
     "of", "with", "in", "on", "at", "from", "to", "by", "and", "or",
     "over", "under", "behind", "against", "near", "for", "between"}
)

# "(detailed face:1.2)" and "\(cosplay\)" -- prompt weighting and escaped
# parens are syntax, not part of the name being looked up.
_WEIGHT = re.compile(r":\s*-?\d+(?:\.\d+)?\s*$")

# One word, for the in-place repair pass. Punctuation is deliberately
# outside every span: a prose prompt's commas and full stops are grammar,
# and the pass must be able to put the text back exactly as it found it.
_WORD = re.compile(r"[A-Za-z0-9_'\u2019]+")

# Where one segment of a prompt ends. Commas and semicolons always, and a full
# stop only when whitespace or the end follows it -- so the decimal in a
# "(tag:1.2)" weight is not mistaken for the end of a sentence.
_SEGMENT = re.compile(r"[,;]|\.(?=\s|$)")

# How many people are in the picture is not garnish. Danbooru counts them per
# gender and in exactly one spelling -- 1girl, 2boys, multiple girls -- while
# an LLM asked for a scene with two people writes "1man", "1woman", "2 boys"
# or "two people", none of which is a tag. Dropping those the way an invented
# adjective is dropped costs a person: "a knight and his squire" enriches to
# "1boy, 1man, ..." and renders as one figure, which is the bug this exists
# for. Counts are rewritten to the trained spelling instead.
_COUNT = re.compile(
    r"^(\d+|one|two|three|four|five|a|an)\s*[- ]?\s*"
    r"(girls?|women|woman|females?|ladies|lady"
    r"|boys?|men|man|males?|guys?|dudes?"
    r"|others?|people|persons?)$"
)
_WORD_NUMBERS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
# a count tag in the spelling the model was trained on, which is what several
# of them have to be merged back into
# Words that make the tail of a compound mean something other than itself.
# "shallow depth OF field" is not a field and "NO hat" is not a hat, so a tag
# containing either is left alone rather than folded to its last noun.
_PREPOSITIONS = frozenset(
    {"of", "with", "in", "on", "at", "from", "to", "by", "and", "or",
     "over", "under", "behind", "against", "near", "for", "between"}
)
_NEGATIONS = frozenset({"no", "not", "without", "missing", "non", "never", "anti"})

# Words the LLM reaches for that Danbooru has no spelling of -- not a
# misspelling or a plural, a different word -- mapped to the one it trained on.
# Danbooru already redirects "slender" to "skinny" and "chubby" to "plump"; the
# entries here are the ones it has no redirect for. Kept to build and blur,
# where the LLM's vocabulary and Danbooru's diverge hardest and the tag would
# otherwise be lost outright ("athletic build" has no salvageable word in it).
# Every target is checked against the snapshot before use, like every other
# rewrite here.
_SYNONYMS = {
    "athletic": "toned",
    "fit": "toned",
    "lean": "toned",
    "slim": "skinny",
    "buff": "muscular",
    "ripped": "muscular",
    "stocky": "muscular",
    "blurred": "blurry",
}


def _count_tag(n: int, base: str) -> str:
    """"1boy", "2boys", "multiple boys" -- Danbooru stops counting at six."""
    if n > 5:
        return f"multiple {base}s"
    return f"{n}{base}" if n == 1 else f"{n}{base}s"


_GENDER_NOUNS = {
    "girl": "girl", "girls": "girl", "woman": "girl", "women": "girl",
    "female": "girl", "females": "girl", "lady": "girl", "ladies": "girl",
    "boy": "boy", "boys": "boy", "man": "boy", "men": "boy",
    "male": "boy", "males": "boy", "guy": "boy", "guys": "boy",
    "dude": "boy", "dudes": "boy",
    "other": "other", "others": "other",
    "people": "other", "person": "other", "persons": "other",
}

# What two subjects are doing with each other, in the phrasings an LLM reaches
# for against the one Danbooru was trained on. Same reasoning as the counts:
# these are what hold a two-person scene together, and without them the second
# subject is a tag with nothing to do. Every target is checked against the
# snapshot before it is used, so a stale table can only fail to help.
_RELATIONS = {
    "looking at each other": "looking at another",
    "look at each other": "looking at another",
    "looking at one another": "looking at another",
    "gazing at each other": "looking at another",
    "facing each other": "face-to-face",
    "face to face": "face-to-face",
    "leaning against each other": "leaning on person",
    "leaning on each other": "leaning on person",
    "hugging each other": "hug",
    "embracing each other": "hug",
    "embrace": "hug",
    "arm around each other": "arm around shoulder",
    "arms around each other": "arm around shoulder",
    "arm around the other": "arm around shoulder",
    "side by side": "side-by-side",
    "standing side by side": "side-by-side",
    "walking side by side": "side-by-side",
    "holding each other's hand": "holding hands",
    "hand in hand": "holding hands",
}


class Vocabulary:
    def __init__(
        self,
        tags: dict[str, int],
        aliases: dict[str, str],
        other_names: Optional[dict[str, list[str]]] = None,
    ) -> None:
        # stored with underscores, looked up with spaces: normalise once here
        # so every lookup is a plain dict hit
        self.counts = {k.replace("_", " "): v for k, v in tags.items()}
        self.aliases = {
            k.replace("_", " "): v.replace("_", " ") for k, v in aliases.items()
        }
        # Longest alias worth trying as a phrase. Derived rather than fixed:
        # the table has entries up to 17 words, but the mass is 1-5 and the
        # window is the inner loop of every repair pass. Capped so one freak
        # entry cannot make the pass quadratic in the prompt length.
        self._max_alias_words = min(
            max((len(a.split()) for a in self.aliases), default=1), 6
        )
        # tags grouped by their first word, built on first use; see _opens_with
        self._by_first: Optional[dict[str, list[str]]] = None
        # and by every word in them, for the looser pass; see _uses_word
        self._by_word: Optional[dict[str, list[str]]] = None
        # The wiki's other names, inverted: what somebody types -> the tags it
        # is another word for, best-learned first. This is where "male" finds
        # "1boy" and "female" finds "1girl" without anybody writing it down.
        # A name can belong to several tags, so the value is a list.
        self.called: dict[str, list[str]] = {}
        for tag, names in (other_names or {}).items():
            canon = tag.replace("_", " ")
            if canon not in self.counts:
                continue
            for name in names:
                n = name.replace("_", " ").lower().strip()
                if n and n != canon:
                    self.called.setdefault(n, []).append(canon)
        for names in self.called.values():
            names.sort(key=lambda t: -self.counts[t])

    def __bool__(self) -> bool:
        return bool(self.counts)

    @staticmethod
    def normalise(tag: str) -> str:
        """The name to look up for one comma-separated piece of a prompt."""
        t = tag.strip().lower().replace("_", " ")
        t = t.strip("()")           # (emphasis) with no weight
        t = _WEIGHT.sub("", t)      # (tag:1.2)
        t = t.replace("\\(", "(").replace("\\)", ")")
        # Prose ends its clauses with punctuation and tags do not, so a tag at
        # the end of a sentence used to match nothing: "from above." is the
        # framing instruction, not an unknown word.
        return " ".join(t.split()).strip("() ").strip(".!?;:\"'")

    def is_prose(self, tag: str) -> bool:
        """Is this segment a clause rather than a tag?

        A function word is the signal, not length. The old bar also demanded
        more than four words, which was right when a tag prompt was
        allowed one trailing sentence and wrong now that the whole prompt is
        prose: "wears a green raincoat" is four words, so it failed the bar,
        was treated as a tag, and got reported as unknown on every prompt that
        described a coat. Two words is the floor because a lone function word
        is not a clause. A known tag is looked up before this is consulted, so
        real tags carrying a function word ("looking at viewer", "from above")
        never reach it.
        """
        words = tag.split()
        return len(words) >= 2 and bool(_FUNCTION_WORDS & set(words))

    def count(self, tag: str) -> Optional[int]:
        """Posts the model saw for this tag, or None if it is not a tag."""
        return self.counts.get(self.normalise(tag))

    def canonical(self, tag: str) -> Optional[str]:
        """The spelling the model was trained on, when this is not it."""
        return self.aliases.get(self.normalise(tag))

    def known(self, tag: str) -> bool:
        t = self.normalise(tag)
        return t in self.counts or t in self.aliases or t in QUALITY_TOKENS

    def _spelling(self, cand: str) -> Optional[str]:
        """The same tag as the model was trained to spell it, or None.

        Three spellings count as one tag: the candidate itself, an alias
        pointing at one (Danbooru redirects "mini skirt" to "miniskirt"), and
        the closed-up form, because Danbooru writes a good many compounds as
        one word and has no alias for the spaced version at all -- "thigh
        highs" is nothing, "thighhighs" has 1.5M posts.

        Nothing here changes which tag is meant, which is why this much is
        applied even to words the user typed themselves; see repair().
        """
        for form in (cand, cand.replace(" ", "")):
            if form in self.counts:
                return form
            if (canon := self.aliases.get(form)) is not None:
                return canon
        return None

    def _known_form(self, cand: str) -> Optional[str]:
        """A spelling of this tag, or the different word the model does have.

        The synonym half is a real substitution -- "athletic" is not "toned",
        it is the nearest thing Danbooru has -- so this is for prompts the LLM
        wrote, and _spelling is what the user's own words get.
        """
        if (spelled := self._spelling(cand)) is not None:
            return spelled
        syn = _SYNONYMS.get(cand)
        return syn if syn in self.counts else None

    def _singular(self, tag: str) -> Optional[str]:
        """The singular of a plural the snapshot does not have.

        Danbooru names most things in the singular ("shadow", "puddle",
        "bookshelf") and an LLM writing prose-shaped tags pluralises them, so
        this is the commonest near miss there is. Only the last word is folded
        and only when the plural itself is unknown, so "boots" and "freckles",
        which Danbooru does keep plural, never move. Modifiers are preserved:
        "hair ribbons" becomes "hair ribbon", never "ribbon" -- dropping words
        to find a hit is how "shallow depth of field" would become "field".
        """
        head, sep, last = tag.rpartition(" ")
        for form in (
            last[:-3] + "y" if last.endswith("ies") else None,
            last[:-3] + "f" if last.endswith("ves") else None,
            last[:-3] + "fe" if last.endswith("ves") else None,
            last[:-2] if last.endswith("es") else None,
            last[:-1] if last.endswith("s") else None,
        ):
            if form and (known := self._spelling(head + sep + form)):
                return known
        return None

    def _split(self, tag: str) -> Optional[str]:
        """A compound the snapshot has no phrase for, as the tags it does have.

        The LLM writes prose-shaped compounds -- "long grey hair", "black
        leather jacket", "dark navy mini skirt" -- where Danbooru has the
        pieces and not the phrase. Two moves, in order:

        1. The longest suffix that is a real tag, so "dark navy mini skirt"
           lands on "miniskirt" and not on the bare "skirt".
        2. Each word that move 1 dropped, retried against the head noun on its
           own. "long grey hair" is "long hair, grey hair" -- both real tags,
           and between them they say what the phrase said. This is what stops
           the fold quietly throwing away half the description.

        If no suffix is a tag at all, the words are tried alone, which is what
        rescues "serious expression" as "serious" and "athletic build" as
        "toned". A modifier can only come through this way if the model was
        trained on it, so the folklore stays out: "dramatic", "volumetric",
        "ambient", "rim", "crisp" and "detailed" are not tags any more than
        "lighting" and "background" are.

        Fenced the same way throughout: a preposition or a leading negation
        means the phrase is not about its own words ("shallow depth of field",
        "no hat"), and nothing is returned that the snapshot does not have.
        """
        words = tag.split()
        if len(words) < 2 or _PREPOSITIONS & set(words) or words[0] in _NEGATIONS:
            return None
        # one word spelled Danbooru's way can land the whole phrase, and the
        # whole phrase says more than any piece of it: "blurred background" is
        # nothing, "blurry background" is a tag with 300k posts
        swapped = " ".join(_SYNONYMS.get(w, w) for w in words)
        if swapped != tag and (whole := self._known_form(swapped)):
            return whole
        for i in range(1, len(words)):
            suffix = " ".join(words[i:])
            known = self._known_form(suffix) or self._singular(suffix)
            if not known:
                continue
            out = []
            for modifier in words[:i]:
                pair = self._known_form(f"{modifier} {words[-1]}")
                if pair and pair not in out and pair != known:
                    out.append(pair)
            out.append(known)
            return ", ".join(out)
        # nothing at the end of it is a tag: keep whichever single words are
        alone = [
            found for word in words
            if (found := self._known_form(word)) is not None
        ]
        return ", ".join(dict.fromkeys(alone)) or None

    def rescue(self, tag: str) -> Optional[str]:
        """A tag the model never saw, rewritten to one it did -- or None.

        The answer may be more than one tag: a compound Danbooru has no phrase
        for comes back as the tags it does have ("long grey hair" as "long
        hair, grey hair"), which is the point -- one folded noun would have
        dropped half of what was written.

        Deliberately narrow: how many people are in the picture, what they are
        doing with each other, a plural where Danbooru keeps the singular, and
        the pieces of a compound it has no phrase for. The first two change
        what is in the frame rather than how it is dressed; the last two are
        the commonest near misses and each costs a real noun.
        Everything else the LLM invents is still dropped -- including the
        folklore this project measured as harmful, which must stay dropped.
        The rewrite is only returned if the snapshot has it, so this can
        rewrite but never invent.
        """
        t = self.normalise(tag)
        canon = _RELATIONS.get(t)
        if canon is not None:
            canon = canon if canon in self.counts else None
        elif (m := _COUNT.match(t)) is not None:
            n = _WORD_NUMBERS.get(m.group(1)) or int(m.group(1))
            canon = _count_tag(n, _GENDER_NOUNS[m.group(2)])
            canon = canon if canon in self.counts else None
        else:
            # the whole tag under another spelling first, then a plural, and
            # only then the split, which is the one that can lose a word
            canon = (
                self._known_form(t)
                or self._counted_thing(t)
                or self._singular(t)
                or self._split(t)
            )
        # a tag that is already the trained spelling is not a rewrite
        return canon if canon and canon != t else None

    def _counted_thing(self, tag: str) -> Optional[str]:
        """"1cat" -> "cat". Danbooru counts people, and nothing else.

        The count words (1girl, 2boys, multiple girls) are the vocabulary's
        only numbered tags, and an LLM told to use them reaches for the same
        shape when it gets to the animals -- "1cat" for a picture with a cat in
        it. It is not a word the model knows, and unlike the gender counts
        there is nothing to fold it into: the tag is simply "cat". Reported by
        check(), applied by nobody, like every other rescue.
        """
        m = re.match(r"^(\d+)\s*(.+)$", tag)
        if not m or _COUNT.match(tag):
            return None
        thing = m.group(2).strip()
        # "2cats" pluralises the noun as well as counting it, and Danbooru
        # keeps the singular, so the plural fold has to run here too
        return thing if thing in self.counts else self._singular(thing)

    # A bare modifier is only worth suggesting for if the model has enough
    # pictures of it to have learned the pairing; below this the tail is
    # tags nobody means.
    _OPENS_FLOOR = 2000

    def _opens_with(self, word: str, limit: int = 4) -> list[str]:
        """Tags that begin with this word, most-used first.

        The one search over the snapshot that is worth running. A bare
        modifier -- "small", "long", "open", "bare" -- is not a tag on its own
        because Danbooru names what is small, not smallness; the tags that
        answer it all *start* with the word. Matching the first word, and only
        for a word that is not a tag itself, is what keeps this out of the
        failures in docs/decisions/VOCABULARY.md: "stockings" cannot reach
        "king" and "pony tail" cannot reach "cat tail" from here.

        A closed-up hit is allowed after the whole-word ones, which is how
        "pony" reaches "ponytail".
        """
        if self._by_first is None:
            self._by_first = {}
            for name, count in self.counts.items():
                if count >= self._OPENS_FLOOR and "(" not in name:
                    self._by_first.setdefault(name.split()[0], []).append(name)
                    if " " not in name:  # for the closed-up pass below
                        self._by_first.setdefault("", []).append(name)
            for names in self._by_first.values():
                names.sort(key=lambda n: -self.counts[n])
        return [n for n in self._by_first.get(word, []) if n != word][:limit]

    def _closed_up(self, word: str, limit: int) -> list[str]:
        """One-word tags starting with these letters: "pony" -> "ponytail"."""
        if self._by_first is None:
            self._opens_with(word, 1)  # builds the index
        return [
            n for n in self._by_first.get("", [])
            if n != word and n.startswith(word)
        ][:limit]

    def suggest(self, tag: str) -> list[str]:
        """Tags to offer for a word the vocabulary has no rewrite for.

        Unlike `rescue`, these are not the same word under another spelling:
        "male" is not "1boy", it is the tag a person means when they type
        "male". So they are returned as a list for the composer to offer and
        the user to pick from, and nothing here is ever applied on its own.

        Three sources, strongest first: what Danbooru's own tag wiki records
        as another name for a tag, then a small table for what it misses, then
        a search over the vocabulary. The search is generous -- a weak row
        costs a glance, a missing row costs the answer -- so the ranking does
        the work rather than exclusion. What would generalise past all three
        is semantic search over tag embeddings; see
        docs/decisions/VOCABULARY.md for that and for the measurements.
        """
        named = self.named_alternatives(tag)
        return named or self.searched_alternatives(tag)

    def named_alternatives(self, tag: str) -> list[str]:
        """Tags something is *recorded* as another word for: wiki, or table.

        Kept apart from the search below because these outrank `rescue`: the
        wiki saying "dirt path" is another name for "dirt road" beats rescue
        splitting it down to a bare "path", which loses the word that carried
        half the meaning. The search does not outrank rescue --
        a plural is better folded to its singular than answered with whatever
        happens to start with it.
        """
        t = self.normalise(tag)
        # A word that is already a tag, or an alias of one, needs no standing
        # in for: check() has better things to say about both, and the
        # type-ahead already completes them from the vocabulary itself.
        if self.known(t):
            return []
        # Danbooru's own answer first: the tags whose wiki lists this as
        # another name for them. Real data beats a table, and it is the same
        # source that already supplies the aliases.
        out = list(self.called.get(t, ()))
        # A bare gender word: Danbooru counts people rather than naming their
        # sex, so the counts are added to whatever the wiki said. The wiki has
        # "1boy" for "male" but nothing for "guy", and neither offers the
        # plural, which is a real thing somebody might have meant.
        if (base := _GENDER_NOUNS.get(t)) is not None:
            out.append(_count_tag(1, base))
            if base == "boy":
                out.append("male focus")  # the tag Danbooru redirects "man" to
            out.append(f"multiple {base}s")  # last: the least likely reading
        if not out and (syn := _SYNONYMS.get(t)) is not None:
            out = [syn]
        return self._offerable(out, t)[:6]

    def searched_alternatives(self, tag: str, limit: int = 6) -> list[str]:
        """Anything in the vocabulary built from this word, best guess first.

        Generous on purpose. These reach a person as a list to look at and
        pick from, not as a claim that the word means any of them, so a weak
        row costs a glance and a good row that was filtered out costs the
        answer. That is why "pretty" is allowed to turn up the Pretty Cure
        franchise tag: it is obviously not what you meant, and you can see
        that it is obviously not what you meant.

        Ordered strongest first, because the ranking is what does the work
        once nothing is being excluded:

          1. tags that open with the word   small -> small horns
          2. tags that use it anywhere      male  -> muscular male
          3. tags sharing its head noun     pony tail -> cat tail, horse tail
          4. the closed-up spelling         pony  -> ponytail

        Whole words throughout. A raw substring is not a weak match but a
        non-match -- it answered "stockings" with "king" -- and nobody can
        pick their way out of that.
        """
        t = self.normalise(tag)
        if self.known(t):
            return []
        words = t.split()
        head = words[-1]
        opens = self._opens_with(t, limit)
        uses = [n for n in self._uses_word(head, limit * 2) if n not in opens]
        # only for a phrase: for a single word, "uses" already covered it
        kin = [] if len(words) == 1 else [
            n for n in self._uses_word(head, limit * 2)
            if n not in opens and n not in uses
        ]
        closed = [n for n in self._closed_up(t, limit) if n not in opens]
        return self._offerable(opens + uses + kin + closed, t)[:limit]

    def _uses_word(self, word: str, limit: int) -> list[str]:
        """Tags carrying this word anywhere in them, most-used first."""
        if self._by_word is None:
            self._by_word = {}
            for name, count in self.counts.items():
                if count >= self._OPENS_FLOOR and "(" not in name:
                    for w in name.split():
                        self._by_word.setdefault(w, []).append(name)
            for names in self._by_word.values():
                names.sort(key=lambda n: -self.counts[n])
        return [n for n in self._by_word.get(word, []) if n != word][:limit]

    def _offerable(self, cands: list[str], typed: str) -> list[str]:
        """Never a tag the snapshot lacks, never the word just typed, no
        duplicates, and never so many that the list stops being a suggestion."""
        seen: list[str] = []
        for cand in cands:
            if cand in self.counts and cand != typed and cand not in seen:
                seen.append(cand)
        return seen

    def suggestion_map(self) -> dict[str, list[str]]:
        """Every word `suggest` has an answer for, for the type-ahead.

        Small enough to hand the page once at load -- a few dozen words --
        which is what lets a suggestion appear while somebody is still typing
        instead of after a round trip.
        """
        words = set(_GENDER_NOUNS) | set(_SYNONYMS) | set(self.called)
        out = {w: self.suggest(w) for w in sorted(words)}
        return {w: tags for w, tags in out.items() if tags}

    def check(self, prompt: str) -> dict[str, list]:
        """What is worth saying about a prompt, without changing it.

        This is the advisory layer, and the only one that reaches the user:
        `repair` fixes spellings in what the LLM wrote, this reports and waits
        for a click.

        `aliases` are spellings the model never saw but which map to one it
        did, so they can be fixed outright. `suggestions` are words the model
        has no spelling of at all but which it has a tag *for* -- a different
        word, offered as a choice because picking one is a decision about
        meaning. Prose segments and the checkpoints' own quality tokens are
        not reported: neither is a Danbooru tag and neither is a mistake.

        Two things it must get right now that prompts are prose. Segments end
        at a sentence as well as at a comma, or "best quality. 1girl" arrives
        as one segment that matches nothing. And a segment is looked up as a
        tag BEFORE it is asked whether it is prose, because plenty of real
        tags carry a function word -- "looking at viewer", "arm behind back",
        "from above" -- and the prose test alone would wave them through
        unexamined.

        ## Why this reports so much less than it could

        It used to answer a prose prompt the way it would answer a tag list,
        and on an LLM-written prompt that is a screenful of yellow saying that
        "gentle smile" is not a tag, that "green field" is not a tag, and that
        "silver hair" should perhaps be `long hair`. Two measurements in
        docs/decisions/PROMPT-STYLE.md say all three of those are wrong to
        say. Disciplining "green raincoat" to the only real tag, `raincoat`,
        lost the colour in 6 renders of 6 -- so a rewrite that drops a word is
        not a fix, it is the failure that made `vtags` the worst arm of five.
        And verifying a prose prompt's vocabulary bought nothing measurable
        at all: `prose` 93%, `vprose` 93%.

        So what is left is what the snapshot is *certain* of, and nothing
        else:

        - an exact alias, where Danbooru itself says the two spellings are one
          tag;
        - a name the wiki or the synonym table *records* for a tag
          ("daytime" -> `day`, "dirt path" -> `dirt road`);
        - a rescue that accounts for every word it was given -- "two girls" ->
          `2girls`, "long grey hair" -> `long hair, grey hair`. A rescue that
          drops one is the green-raincoat case and is now silent.

        Two tiers were dropped outright. The word-overlap search stays in the
        type-ahead, where somebody has asked a question and a weak match costs
        a glance (docs/decisions/VOCABULARY.md), but it does not get to
        volunteer a claim about a prompt nobody asked it about: it answers
        "silver hair" with six hair tags and neither right one, and
        "wildflowers scattered around" with an unrelated clothing tag. And the
        bare "not in the vocabulary, nothing to offer" line went with it --
        under prose that is a complaint about ordinary English. The garnish
        the checkpoints genuinely render as haze is a separate, measured list
        and is unaffected (META_TAGS, reported by the endpoint).
        """
        if not self.counts:
            return {"aliases": [], "suggestions": []}
        aliases: list[list[str]] = []
        suggestions: list[list] = []
        for raw in _SEGMENT.split(prompt):
            tag = self.normalise(raw)
            if not tag or tag in QUALITY_TOKENS:
                continue
            if tag in self.counts or self.is_prose(tag):
                continue
            # Precedence, and the reason for it: an exact spelling is
            # certain, so it wins. What the wiki or a table *records* as
            # another word for a tag comes next, because rescue's last resort
            # is splitting a compound into its parts and that loses words
            # ("dirt path" -> "path", dropping what kind). Rescue then takes
            # what is left, and only a bare word nothing has an entry for
            # falls through to the search.
            offer: list[str] = []
            canon = self.aliases.get(tag)
            if not canon:
                offer = self.named_alternatives(tag)
                if not offer:
                    # A rescue is OFFERED, never presented as a spelling fix.
                    # Only an exact alias is one: Danbooru itself says the two
                    # names are the same tag, so swapping cannot change what
                    # was asked for. And only a rescue that keeps every word
                    # is offered at all -- see the note above.
                    rescued = self.rescue(tag)
                    offer = [rescued] if rescued and self.keeps_every_word(tag, rescued) else []
            if canon:
                if [tag, canon] not in aliases:
                    aliases.append([tag, canon])
            elif offer:
                if not any(row[0] == tag for row in suggestions):
                    suggestions.append([tag, offer])
        return {"aliases": aliases, "suggestions": suggestions}

    @staticmethod
    def keeps_every_word(tag: str, rewrite: str) -> bool:
        """Whether a rewrite says everything the segment it replaces said.

        The question the green-raincoat measurement asks of every rescue: is
        this the same request in the vocabulary's own words, or is it a
        shorter request? Two shapes pass, and they are the two the rescue
        chain actually produces.

        A rewrite that shares no word with the segment has replaced the whole
        of it rather than reached inside it -- "two girls" -> `2girls`,
        "thigh high stockings" -> `thighhighs`, "athletic build" -> `toned`.
        Nothing was dropped because nothing was kept.

        A rewrite that does share words has to account for all of them:
        "long grey hair" -> `long hair, grey hair` uses every word and passes,
        while "green raincoat" -> `raincoat` and "gentle smile" -> `smile`
        leave a word behind and do not.

        A trailing "s" is folded on both sides, because Danbooru keeps the
        singular and the fold is the whole point of half these rewrites:
        without it "shadows" -> `shadow` reads as sharing nothing (fine, it
        passes either way) but "rolling hills" -> `hill` does too, and that
        one drops a word.
        """
        fold = lambda ws: {w[:-1] if len(w) > 3 and w.endswith("s") else w for w in ws}
        before = fold(tag.split())
        after = fold(rewrite.replace(",", " ").split())
        if not (before & after):
            return True  # a whole-phrase respelling: nothing of it was kept
        return before <= after

    def repair(self, prompt: str, keep: str = "") -> str:
        """Fix spellings the model was never trained on. Change nothing else.

        Prompts are prose now (docs/decisions/PROMPT-STYLE.md), so this edits
        segments in place rather than splitting the prompt on commas and
        rejoining it as a tag list. The old pass turned "standing in an alley.
        She has red hair" into "standing in an alley, She has red hair" and
        dropped "best quality. 1girl" whole, because a sentence boundary had
        landed inside a comma segment.

        Only an exact alias is applied, and only when it is the ENTIRE segment.
        Both halves of that rule were bought with a corruption:

            "thigh highs"                -> "thighs highs"
            "sitting on a rooftop rail"  -> "sitting on person a rooftop rail"

        because "thigh" is an alias for "thighs" and "sitting on" for "sitting
        on person". Inside a clause those are words, not tags, and a pass that
        rewrites words inside a clause is editing meaning -- the one thing this
        must not do. A whole segment that IS an alias is unambiguous: Danbooru
        itself records the two names as one tag, so the swap cannot change what
        was asked for.

        Everything else is left exactly as written, including phrases the
        vocabulary has no entry for. That reverses this pass's old policy of
        dropping them, and the measurement is in the decision doc: disciplining
        every phrase to a real tag turned "green raincoat" into "raincoat" and
        lost the colour in 6 renders of 6, while the same request in prose kept
        it 6 of 6. Danbooru has no tag for a green raincoat, and a prompt is
        allowed to say things the vocabulary cannot.

        Folklore that genuinely has to go is named in llm.META_TAGS. It used to
        disappear here by accident, as a side effect of dropping the
        unrecognised.

        `keep` is the user's own wording. It does not block a spelling fix --
        showing the model a spelling it never saw is not respecting a request,
        it is failing it -- but nothing is dropped or swapped for a different
        word, whoever wrote it.
        """
        # No snapshot means no opinion: a fresh clone has no vocabulary, and
        # that must cost the fixes rather than the prompt.
        if not self.aliases:
            return prompt
        out: list[str] = []
        at = 0
        for cut in list(_SEGMENT.finditer(prompt)) + [None]:
            end = cut.start() if cut else len(prompt)
            seg = prompt[at:end]
            canon = self.aliases.get(self.normalise(seg))
            if canon and self.normalise(seg) not in self.counts:
                # keep the segment's own padding so the prompt reads the same
                lead = seg[:len(seg) - len(seg.lstrip())]
                tail = seg[len(seg.rstrip()):]
                out.append(f"{lead}{canon}{tail}")
            else:
                out.append(seg)
            if cut:
                out.append(cut.group())
            at = end + (len(cut.group()) if cut else 0)
        return "".join(out)


_cached: Optional[Vocabulary] = None


def get() -> Vocabulary:
    """The shared vocabulary, loaded once on first use."""
    global _cached
    if _cached is None:
        _cached = load()
    return _cached


def load(path: Path = VOCAB_PATH) -> Vocabulary:
    """The shipped snapshot, or an empty vocabulary that checks nothing."""
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        log.info("no tag vocabulary at %s (%s); tag checks are off", path, exc)
        return Vocabulary({}, {})
    vocab = Vocabulary(
        data.get("tags", {}), data.get("aliases", {}), data.get("other_names", {})
    )
    log.info(
        "tag vocabulary: %s tags, %s aliases, %s other names (fetched %s)",
        len(vocab.counts), len(vocab.aliases), len(vocab.called),
        data.get("fetched_at", "?"),
    )
    return vocab
