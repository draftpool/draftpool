"""Prompt enrichment and refinement via Ollama structured outputs."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import random
import re

import httpx
from PIL import Image

from . import vocab
from .schemas import DEFAULT_TAG_NEGATIVE, ArtworkCaption, PromptVariant

log = logging.getLogger(__name__)

# Long edge of an image sent to the vision model. The composer's "From image…"
# has downscaled to this in the browser since it shipped, for the same two
# reasons: a vision model tiles what it is given, so a 4K render costs several
# times the time to read and adds nothing -- the model is looking for framing,
# subject, light and style, none of which survive only at full resolution.
VISION_MAX_EDGE = 1024


def image_for_vision(data: bytes, max_edge: int = VISION_MAX_EDGE) -> str:
    """`data` as base64 JPEG, no longer than `max_edge` on its long side.

    Blocking (PIL): call it in a thread. Returns bare base64, not a data URL,
    which is what Ollama's `images` field wants.
    """
    img = Image.open(io.BytesIO(data))
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    scale = min(1.0, max_edge / max(img.width, img.height))
    if scale < 1.0:
        img = img.resize(
            (max(1, round(img.width * scale)), max(1, round(img.height * scale))),
            Image.LANCZOS,
        )
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=85)
    return base64.b64encode(out.getvalue()).decode()


VARIANTS_SCHEMA = {
    "type": "object",
    "properties": {
        "variants": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "negative_prompt": {"type": "string"},
                },
                "required": ["prompt", "negative_prompt"],
            },
        }
    },
    "required": ["variants"],
}

# The publish dialog's "simple post" asks for two things and neither of them is
# a prompt, so it gets its own schema rather than a variant with the fields
# renamed: a model handed the variants schema writes a prompt into `prompt`
# whatever the system message says the field is for.
ARTWORK_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "description": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["title", "description", "tags"],
}

# How much the {k} variants are allowed to differ. "explore" is the original
# behaviour: one instruction, k different pictures. "hold" is for a prompt that
# already IS a picture -- typically one written by describe() from a seed image
# -- where re-rolling the framing throws away the thing the user is starting
# from, and the useful variation is depth of detail instead.
VARY_PROSE = """Diversity must come only from dimensions the user left open, exploring different
ways to REALISE the stated intent: composition, camera angle, framing, pose,
expression, wardrobe and styling, lighting, colour palette, time of day, setting
details, atmosphere, and secondary elements. Never dilute or drop the intent in
favour of a generally different image. One variant may push further within the
open dimensions (an unusual angle, dramatic lighting, an unexpected setting)
while still strictly matching the user's subject and style."""

HOLD_PROSE = """The instruction already describes one specific picture: treat it as the
composition, not as a theme. Every variant keeps its framing, crop, camera
angle, subject placement, pose, gaze and the arrangement of everything else in
the frame EXACTLY as described -- the {k} results should read as the same shot
{k} times. Carry every concrete detail the instruction states through into each
variant rather than paraphrasing it away. Vary only what cannot move the
picture: how precisely materials, textures, surfaces and small objects are
named, how deep the detail goes, and the fine grain of the light already
described. Never introduce a different camera angle, a different crop, another
pose, or a setting element that would re-block the scene."""

# The opening line of each template: hold mode must not ask for DIVERSE prompts
# in its first sentence and then forbid divergence three lines later.
VARY_LEAD = "DIVERSE, detailed "
HOLD_LEAD = "closely related, detailed "

VARY_TAGS = """Diversity must come only from dimensions the user left open, exploring
different ways to REALISE the stated intent: composition, framing, pose,
expression, wardrobe and styling, lighting, palette, setting details, and
atmosphere. Add striking, specific tags in those open dimensions; one variant
may push further (dutch angle, dramatic backlighting, an unexpected but fitting
location) while still strictly matching the user's subject, style and intent."""

HOLD_TAGS = """The instruction already describes one specific picture: treat it as the
composition, not as a theme. Every variant repeats its framing tag, camera angle,
pose, expression and gaze tags unchanged and keeps the same subject count tags,
so the {k} lists describe the same shot. Vary only tags that cannot re-block the
scene: clothing and accessory detail, small props, objects the setting already
implies, and the fine grain of the lighting already named. Never swap the
framing tag, add an angle tag that was not there, or change the pose."""


# Composition principles, one per variant. Asking for "diverse" prompts and
# leaving composition as a bare word in a list gets the same picture k times:
# subject centred, filling the frame, nothing else decided. Each entry pairs the
# principle with how to WRITE it, because these models read described geometry
# and not the term -- "rule of thirds" is a phrase from photography books and
# means nothing to a diffusion model, while "her eye on the upper-left third"
# places something. The names here are for the LLM's benefit only and are
# explicitly banned from the prompt text itself.
COMPOSITION_PRINCIPLES: tuple[tuple[str, str], ...] = (
    (
        "rule of thirds",
        "place the subject or its focal point off-centre, a third of the way "
        "in, and say what occupies the rest of the frame",
    ),
    (
        "leading lines",
        "name a line in the scene -- a path, railing, shoreline, shadow, row "
        "of lights -- and the direction it runs toward the subject",
    ),
    (
        "frame within a frame",
        "look through something in the foreground: a doorway, window, arch, "
        "branches, a gap in a crowd, with its edges dark and close",
    ),
    (
        "negative space",
        "give the subject a large empty field -- sky, water, wall, fog -- and "
        "keep it small against it",
    ),
    (
        "foreground-midground-background layering",
        "put something specific in each of the three depths so the eye travels "
        "back through the picture",
    ),
    (
        "diagonal or triangular structure",
        "build the arrangement on a slant or a three-point group rather than "
        "on horizontals, and say what forms each corner",
    ),
    (
        "symmetry",
        "centre the subject deliberately on a mirrored setting -- corridor, "
        "facade, reflection in still water -- with both halves described",
    ),
    (
        "shallow depth of field",
        "one plane sharp and named, everything nearer and further dissolved, "
        "with the out-of-focus shapes described as shapes",
    ),
    (
        "scale contrast",
        "set the subject against something enormous or swarming so its size "
        "reads, and say what that thing is",
    ),
    (
        "chiaroscuro",
        "one light source, most of the frame in shadow, and name exactly which "
        "surfaces the light lands on",
    ),
    (
        "lead room",
        "the subject looks or moves toward the open side of the frame; say "
        "where the gaze goes and what waits in that space",
    ),
    (
        "repetition with a break",
        "a pattern of repeated elements the subject interrupts -- name the "
        "pattern and name the interruption",
    ),
    (
        "radial convergence",
        "the setting's lines converge on a single vanishing point; say where "
        "it sits relative to the subject",
    ),
    (
        "colour accent against a muted field",
        "hold the palette to a narrow desaturated range and let one saturated "
        "colour, named and placed, carry the eye",
    ),
    (
        "high or low camera",
        "commit to looking steeply down or up, and describe what that angle "
        "does to the ground plane or the sky behind",
    ),
    (
        "silhouette",
        "read the subject as a dark shape against a bright ground; the outline "
        "does the describing, not the surface detail",
    ),
)

COMPOSITION_HEADER = """Composition: each variant is assigned a different organising principle below.
Build that variant's frame on its principle -- it decides where the subject
sits, what else is in the picture and where the eye goes. Express it as
described geometry in the prompt's own visual language; NEVER write the
principle's name or any photography-textbook term in the prompt itself. If the
request has no single subject (a landscape, a pattern, an abstract), apply the
principle to the dominant shapes or the horizon instead. The principle governs
only the framing: it never overrides the user's stated subject, style or intent.
"""


ENRICH_SYSTEM = """You are an expert prompt writer for text-to-image diffusion models.
Given a short instruction, write {k} {variation_lead}image prompts.
Hard constraints: everything the user states or clearly implies is fixed in EVERY
variant, i.e. the subject and any style, medium, genre, era, mood or intent the
instruction names or implies. An anime request stays anime in all variants, a
photo request stays photographic, an alluring or sensual request stays alluring
at the requested level in all variants, a cosy request stays cosy, a menacing
request stays menacing.
{variation_instruction}
If the request is suggestive or sensual, depicted people must be adults. Invent specific, evocative details wherever
the variation rule above leaves room, so each image stands out; prefer concrete visual specifics over
generic quality words like "beautiful", "stunning" or "highly detailed".
Each prompt should be one paragraph of concrete visual language (subject, setting,
lighting, style, detail level), no meta commentary, no numbered lists inside the
prompt.
{budget_instruction}
{composition_instruction}
{neg_instruction}
Respond only with JSON."""

NEG_ON = (
    "Also write a short negative_prompt per variant listing artefacts to avoid "
    "(e.g. deformed hands, blurry, watermark, text)."
)
NEG_OFF = "Set negative_prompt to an empty string for every variant."

ENRICH_SYSTEM_TAGS = """You are an expert prompt writer for anime diffusion models
trained on Danbooru tags (the Anima/Cosmos anime checkpoints). Given a short
instruction, write {k} {variation_lead}prompts.
Write each prompt as PROSE -- ordinary sentences describing the picture -- but
say each thing using the exact wording Danbooru uses, because that wording is
what these models were trained on. Write "twin braids", not "two plaits";
"from above", not "a high angle"; "looking at viewer", not "she meets your
gaze"; "1girl" and "1boy" for how many people there are. Where Danbooru has no
word for what you mean, say it in plain English: prose is allowed to describe
things the tag vocabulary cannot name, and that is the point of writing it
this way.
Open with "masterpiece, best quality, amazing quality, very aesthetic,
absurdres" and then start the description.
Prose, not a tag list, and the reason is measured: a comma-separated tag list
cannot bind an attribute to a subject or place one thing relative to another.
Asked for a girl and a boy, tag lists gave the boy her hair colour; asked for a
cat on top of a fence, tag lists put the cat on the ground in 7 of 8 renders
while prose put it on the fence in 8 of 8. Say who has what and what is where
in sentences, and the model follows.
BANNED, and the single most common way to ruin one of these prompts: the
Stable Diffusion 1.5 / Midjourney garnish. Never write "8k", "4k", "uhd",
"highly detailed", "intricate details", "ultra detailed", "concept art",
"digital painting", "digital art", "matte painting", "cinematic", "epic",
"artstation", "trending", "octane render", "unreal engine", "volumetric
lighting", "god rays", "sharp focus", "photorealistic", "beautiful",
"stunning", or anything like them. This model was trained on Danbooru tags and
knows none of these; what it does with them is soften everything into a hazy
matte painting and lose the line art and the face. The words that ask for
detail are the ones that destroy it. Detail comes from naming things ("frilled
dress", "hair ribbon", "stained glass window"), never from adjectives about
quality.
ALSO BANNED: signing the picture over to an artist nobody asked for. Never
write "art by <name>", "in the style of <name>", "reminiscent of <name>" or
"inspired by <name>" unless the instruction itself names them. An artist the
user asked for is theirs and stays; one you chose is a name they did not order,
and it is stripped before the prompt is rendered anyway.
{budget_instruction}
Make the subject dominate the prompt: at most one sentence on background and
atmosphere. Name the framing exactly once
(portrait, upper body, cowboy shot, full body, wide shot) and pick one that
leaves the subject large enough to render -- a character in a "wide shot" or
"aerial view" comes back a few pixels tall with no face. Use at most one
atmosphere word (mist, fog, haze, backlighting): stacking them fogs the whole
image. Only name a character if it is a real Danbooru tag you are sure of;
otherwise describe how they look.
Count every person the instruction mentions using Danbooru's own words: 1girl,
1boy, 2girls, 2boys, 3girls, multiple girls, multiple boys. There is no "1man",
"1woman" or "1male". The counted words are for PEOPLE only -- an animal or an
object is named plainly ("a cat", not "1cat", which is not a word the model
knows).
With more than one person in the picture, keep them apart on the page, because
this is where these models mix people up:
- Do NOT collect the counts at the front ("1boy and 1girl, ..."). Open each
  person's description with that person's own count word -- "1girl with ...",
  then "1boy with ..." -- and keep everything that belongs to them (hair,
  clothes, expression, what they are doing) right there next to it.
- Never let a pronoun carry an attribute or an action while more than one
  person is in the picture. "She wears a red coat", "he hands it to her" and
  "they stand close together" are exactly where it comes apart: write "the girl
  wears a red coat", "the boy hands the cone to the girl".
- If they are doing something with each other, say it once at the end, naming
  both ("the boy hands the girl an ice cream cone"). If the instruction has
  them doing nothing with each other, do not invent an interaction.
- An attribute the instruction gives a person is theirs, in the instruction's
  own words: red hair stays red hair, not auburn and not strawberry blonde.
- Give the second person as much description as the first -- without that the
  model draws one of them and drops the rest.
Hard constraints: the user's subject and any style, mood, intent, era or
character the instruction names or implies stay fixed in EVERY variant; an
alluring or sensual request stays alluring at the requested level in every
variant.
{variation_instruction}
{neg_instruction}
Respond only with JSON."""

REFINE_SYSTEM_TAGS = """You are an expert prompt writer for anime diffusion models
trained on Danbooru tags. The user liked an image generated from the prompt
below and wants a change applied. Write {k} updated prompts in the same style
as the original: prose, using Danbooru's exact wording wherever it has a word
for what you mean, plain English where it does not.
The requested change WINS every conflict: first remove or replace everything in
the original that contradicts it (framing and crop -- close-up, portrait, upper
body, cowboy shot -- camera angle, pose, expression, or anything else the change
supersedes), then say what the change requires, then keep the rest of the
original as it was. Keep how many people there are (1girl, 1boy, 2girls,
multiple boys) unless the change itself is about who is in the picture:
dropping one takes a person out of it. With more than one person, keep each
person's count word next to that person's own description the way the original
has it ("1girl with ...", then "1boy with ..."), and never let a pronoun carry
an attribute or an action -- "she wears the red coat" and "he hands it to her"
are where these models lose track of who is who.
Never ADD Stable Diffusion garnish that is not already there ("8k", "highly detailed", "intricate details", "concept
art", "digital painting", "cinematic", "epic", "artstation", "volumetric
lighting", "photorealistic"): this model knows none of it and it softens the
image into a hazy matte painting. Never ADD an artist either -- no "art by
<name>", no "in the style of <name>" the original did not already have.
The result must be internally consistent:
never say two things that cannot both be true (a close-up that is also full
body, facing the viewer while seen from behind). Rework whatever depends on the
change: a view from behind means facial detail no longer applies.
Make the {k} variants differ slightly in how they realise the change.
{budget_instruction}
{neg_instruction}
Respond only with JSON."""

# Same baseline the app falls back to when no negative was written at all, so
# an enriched prompt and a hand-written one start from the same floor.
NEG_TAGS = (
    f'Every negative_prompt must start with "{DEFAULT_TAG_NEGATIVE}" and may '
    "append scene-specific tags to avoid."
)

# Prompt folklore from the Stable Diffusion 1.5 and Midjourney era, which fills
# the training data of every general-purpose LLM and which they therefore reach
# for unprompted. None of it is Danbooru vocabulary, so on an anime checkpoint
# these name nothing the model was taught; the nearest thing it has for
# "concept art, digital painting, artstation, 8k" is the soft matte-painting
# look, and the subject dissolves into atmosphere along with the line art.
#
# Measured on Illustrious XL 2.0, same seed, same 28 steps / cfg 6 / Euler a,
# only these tags removed: the hazy version puts the character at a few percent
# of frame with a smudge for a face; without them the same seed returns clean
# line art, rendered eyes and readable cloth folds. The words that cost the
# most detail are the ones that ask for it -- "highly detailed", "intricate
# details", "8k".
#
# Stripped only from prompts the LLM wrote, and only when the user did not ask
# for them: a request for concept art still gets concept art (see
# _strip_meta_tags), and anything typed by hand goes to the model untouched.
META_TAGS = frozenset(
    {
        # marketplace and gallery bait
        "artstation",
        "trending",
        "trending on artstation",
        "deviantart",
        "pixiv",
        "behance",
        "cgsociety",
        "award-winning",
        "award winning",
        "featured on artstation",
        # renderers, which an anime checkpoint has never seen
        "unreal engine",
        "unreal engine 5",
        "octane render",
        "octane",
        "vray",
        "v-ray",
        "redshift",
        "ray tracing",
        "ray traced",
        "3d render",
        "cgi",
        "photorealistic",
        "hyperrealistic",
        "photoreal",
        # resolution incantations ("absurdres" and "highres" are real Danbooru
        # tags and stay)
        "8k",
        "4k",
        "16k",
        "uhd",
        "ultra hd",
        "hdr",
        "high resolution",
        "ultra high resolution",
        # generic quality words: the model has the opener for this, and these
        # only dilute the tags that name something
        "highly detailed",
        "high detail",
        "extremely detailed",
        "ultra detailed",
        "ultra-detailed",
        "hyperdetailed",
        "hyper detailed",
        "intricate details",
        "intricate detail",
        "intricate design",
        "insane details",
        "ornate details",
        "fine details",
        "fine detail",
        # bare "detailed" only. NB "detailed background", "detailed face" and
        # "detailed eyes" are not spared for being real tags -- checked, they
        # have zero Danbooru posts each and the vocabulary drops them too; they
        # are simply not this list's business.
        "detailed",
        "very detailed",
        "super detailed",
        *vocab.QUALITY_FOLKLORE,
        # painterly media that fight the line art
        "concept art",
        "digital painting",
        "digital art",
        "matte painting",
        "speedpaint",
        "artstation trending",
        "cinematic",
        "cinematic lighting",
        "epic",
        "epic scale",
        "epic composition",
        # Renderer-and-lighting folklore. These used to be caught by accident:
        # vocab.repair dropped every comma segment that was not a tag, and each
        # of these has zero Danbooru posts. That pass now keeps what it does
        # not recognise, because under prose an unrecognised phrase is usually
        # ordinary English (see docs/decisions/PROMPT-STYLE.md) -- so the words
        # that genuinely have to go have to be named here, which is what this
        # list is for. Same failure mode as the rest of it: they name nothing
        # the model was taught and the nearest thing it has is the soft
        # matte-painting look that eats the line art.
        "volumetric lighting",
        "volumetric",
        "god rays",
        "dramatic lighting",
        "rim lighting",
        "studio lighting",
        "sharp focus",
        "octane render",
        "unreal engine",
        "vray",
        "raytracing",
    }
)

# What comes out of here is not a caption: it is the prompt the user then edits
# and enriches, and in "hold composition" mode it is the ONLY record of how the
# picture was framed. A vision model left to its own devices writes a summary --
# mood, impression, a hedge or two -- and the framing, the one thing a caption
# never needs and a prompt cannot reconstruct, is what goes missing. Hence the
# checklist, and hence framing sitting at the top of it.
DESCRIBE_SYSTEM = """You are an expert prompt writer for text-to-image diffusion models.
Look at the attached image and write ONE prompt that would recreate it as closely
as possible.
Describe only what is actually visible. Never guess at what lies outside the
frame, never name a real person, character, artist or place you think you
recognise, never hedge ("appears to be", "seems", "likely"), and never talk
about the picture instead of writing it ("this image shows", "a photograph
capturing"). No mood words standing in for visual facts: name what makes the
mood, not the mood.
Cover every item below that the image answers -- the style rule underneath
decides what order they are written in and how:
1. FRAMING. The shot size (extreme close-up, close-up, head and shoulders,
   waist up, three-quarter, full body, wide shot), the camera height and angle
   (eye level, low angle looking up, high angle looking down, overhead, tilted),
   and how near the camera is. This is the item most often lost and the one
   that cannot be recovered from anything else, so it never gets skipped.
2. PLACEMENT. Which part of the frame the subject occupies (centred, left
   third, low in the frame, filling it edge to edge), which way it faces, where
   it looks, and what fills the space it does not.
3. SUBJECT. Count the people or animals exactly. For each: build, apparent age
   range, hair, face, expression, pose, and every garment with its colour,
   material and cut. A second subject gets as much description as the first.
4. SETTING. Where this is, and what stands in the foreground, the middle
   distance and the background -- including which of those are sharp and which
   are blurred.
5. LIGHT. Where it comes from, how hard or soft, what colour, and which
   surfaces it lands on. Then the palette, as named colours.
6. MEDIUM AND STYLE. Photograph, oil painting, anime cel, 3d render, ink
   drawing, and its texture: lens character, film grain, brush work, line
   weight.
7. DETAILS that vanish if unnamed: props, jewellery, tattoos, text on signs,
   damage, weather, particles in the air.
{style_instruction}
{neg_instruction}
Respond only with JSON containing exactly one variant."""

DESCRIBE_PROSE = (
    "Write the prompt as ONE flowing paragraph of concrete visual language. The "
    "checklist is the order to think in, never a list to write out: no "
    "numbering, no headings, no labels like \"Framing:\". Open with the shot "
    "and the angle, then the subject, then the rest. Aim for 80 to 150 words. "
    "Name things rather than rate them -- \"chipped enamel mug\" not "
    "\"beautifully detailed mug\", \"hard sunlight from a window on the left, "
    "one cheek lit and the other in shadow\" not \"dramatic lighting\". No "
    "meta commentary."
)
DESCRIBE_TAGS = (
    "Write the prompt as PROSE -- ordinary sentences describing the picture -- "
    "but say each thing in the exact wording Danbooru uses, because that is "
    "what the model was trained on: \"twin braids\" not \"two plaits\", "
    '"from above" not "a high angle", "looking at viewer" not "meeting your '
    'gaze". Where Danbooru has no word for what you can see, say it in plain '
    "English. Open with \"masterpiece, best quality, amazing quality, very "
    'aesthetic, absurdres" and then start describing. The subject must '
    "dominate it; at most one sentence on background and effects. "
    "Count the people in Danbooru's own spelling (1girl, 1boy, 2girls, 2boys, "
    'multiple girls -- never "1man", "1woman" or "2 people", which are not '
    'tags, and never "1cat" or the like: the counted words are for people, and '
    "an animal is just named), describe each of them, and when there is more than one say what "
    "they are doing with each other and which of them has which feature -- a "
    "tag list cannot bind an attribute to a person and prose can. Name the "
    "framing exactly once, using the word for the actual crop (portrait, upper "
    "body, cowboy shot, full body, wide shot) and, whenever the camera is not "
    "at eye level or the subject is not facing the viewer, the word that says "
    "so (from above, from below, from side, from behind, dutch angle, looking "
    "at viewer, looking away, looking to the side, profile). Never describe "
    "two things that cannot both be true, never name an artist, and never use "
    "Stable Diffusion garnish -- no \"8k\", \"4k\", \"highly detailed\", "
    '"intricate details", "concept art", "digital painting", "cinematic", '
    '"epic", "artstation", "trending", "unreal engine", "octane render", '
    '"volumetric lighting" or "photorealistic". The model was trained on '
    "Danbooru tags and knows none of them; they soften the whole image into a "
    "hazy matte painting. Detail comes from naming things, never from "
    "adjectives about quality."
)


# The other way of publishing: no prompt at all, just what a gallery caption
# would say. It exists because a prompt is what was ASKED for and a post is read
# by people looking at what came out -- on any render that took a liberty the
# two are not the same picture, and a prompt that describes the wrong one is
# worse than no prompt. So this asks for the two things a viewer can check
# against the image in front of them: a name for it, a sentence about it, and
# the words they would have gone looking for it under.
#
# Everything a prompt writer is told to do is wrong here, which is why this is
# its own system message rather than a mode of DESCRIBE_SYSTEM: no checklist,
# no framing, no Danbooru vocabulary, no quality opener, and above all no
# comma-separated pile of attributes -- that reads as a prompt however it is
# labelled, and a prompt is exactly what this post is not publishing.
# The publish caption's voice. It used to describe the picture the way a friend
# beside it would, and read as an inventory ("A figure in elaborate red robes
# sits amongst a field of yellow flowers") -- reported as dry, and it was. It
# now admires: the subject and the setting are raised to their enchanted
# versions, and the one rule that keeps that honest is that nothing may be
# mentioned that is not in the picture. One call, measured against a
# look-then-write pair: the same voice at a third of the time (see
# PUBLISHING.md, "The caption's voice").
ARTWORK_SYSTEM = """You write the gallery caption for the attached picture. These pictures are made to be breathtaking and
otherworldly, and the caption has to live up to them: someone scrolling past
should stop, and want to step into the picture. Write like a poet who adores
the piece, never like someone taking an inventory.

TITLE: what this piece would be called if it hung in a gallery of wonders. Two
to six words, evocative, memorable, a little mysterious, and particular to this
picture -- built from something in it, never a stock phrase like "Sunlit
Reverie" or "Golden Dreams" that could title anything; no ending full stop, no
quotation marks. Never the word "AI", "prompt", "render", "artwork" or
"digital art".

DESCRIPTION: one or two flowing sentences, 25 to 55 words, present tense.
- Elevate the subject: a woman is "a stunningly beautiful woman", "an ethereal
  maiden", "a breathtaking beauty" -- never "a person", "a figure" or just "a
  young woman".
- Elevate the setting to the enchanted version of itself: birch trees are "a
  fairytale birch forest", a cobbled street is "a sun-drenched old-world lane",
  a sitting room is "a cosy sunlit parlour".
- Let the light, the colour and the mood carry feeling: glowing, drifting,
  shimmering, hushed, golden.
Example of the voice: "A stunningly beautiful woman in ornate crimson robes
kneels in a fairytale birch forest, golden wildflowers glowing around her as
the last light of evening drifts through the trees."
Every thing, person and animal you mention must be visible in the picture,
with the colours it has. Make what is there sound wonderful; never add an
object, a person, an animal, an event or a story that is not there.

TAGS: three to five words or short phrases someone would browse a gallery by --
what it is of, where it is, what medium it is in, what mood it is in.
Lowercase, no punctuation, broad and plain ("forest", "fantasy", "portrait"),
never repeating each other.

Never name a real person, character, artist or place. Never talk about the
picture as a file ("this image shows", "a digital painting of"). No
comma-separated attribute lists, no quality words ("masterpiece", "8k",
"highly detailed").

Respond only with JSON containing the title, the description and the tags."""


REFINE_SYSTEM = """You are an expert prompt writer for text-to-image diffusion models.
The user liked an image generated from the prompt below and wants a change applied.
Write {k} updated prompts, each a single paragraph of concrete visual language.
The requested change WINS every conflict: first remove or rewrite every part of
the original prompt that contradicts it (framing, crop, camera angle, pose, or
anything else the change supersedes), then work the change in, then keep all
remaining qualities of the original. The result must be internally consistent:
never describe mutually exclusive things (e.g. a close-up of a face together
with a full-body view from behind). Rework dependent details to fit the change.
Make the {k} variants differ slightly in how they realise the change.
{budget_instruction}
{neg_instruction}
Respond only with JSON."""


MOTION_SYSTEM = """You are an expert prompt writer for image-to-video diffusion
models (Wan-family I2V). The attached or described image is the FIRST FRAME of
the clip; the model animates that exact frame.
Read what the user wants to happen, work out the intent behind it, and write ONE
prompt that achieves that same intent in the form these models respond to.
How to write it:
- Describe MOTION, not appearance. The image already carries the subject, the
  setting and the style; restating them wastes the prompt. Name a static detail
  only to anchor what moves ("the hem of her coat", "the grass in the
  foreground").
- Be physical and specific about how things move: direction, speed, and what
  drives it. "long grass bends in slow gusts from the left" beats "wind blows".
- Order it as it happens in the clip, present tense.
- Add plausible secondary motion the user's intent implies (cloth settling,
  hair lifting, breath, ripples, drifting cloud, light shifting) but nothing
  that changes the scene.
- NEVER write the word camera unless the user asked for a camera move, not even
  to say it stays still. These models do not read negation: "the camera holds
  still" was measured to produce a camera push-in, while the same scene with no
  mention of a camera at all stayed put. To keep the framing fixed, say nothing
  about the camera.
- Mention camera movement only if the user asked for it. If they did, keep it to
  ONE slow move (push in, pan left, slight orbit) and hold everything else
  still: a clip has a single motion budget, and a camera move competing with
  swaying hair and shifting light comes out as all three barely happening. Say
  in the prompt itself that the move is continuous and lasts the whole clip
  ("pushes in slowly and continuously for the entire clip"), otherwise the model
  performs a small nudge and then holds. Do not attach motion to things a camera
  move cannot move: panning does not shift shadows or sunlight.
- The clip's exact length is given below. Ask for one continuous action that
  can comfortably finish in that time, never a sequence of events or a story.
  Two seconds is a gesture, not a journey.
- The subject's pose is a constraint as real as the scenery. A figure caught
  mid-stride keeps walking whatever the prompt says, so write motion that
  continues from the pose in the frame rather than fighting it: what the pose
  implies is what the model will do.
- Work only with what the first frame already shows. These models animate the
  pixels in front of them and cannot introduce anything: an object that is not
  in the frame arrives as warped geometry growing out of whatever was there, not
  as an object. Never add, remove or replace subjects or objects, and never
  change setting, style, time of day or wardrobe, or cut to another shot.
- The wish still wins, so serve it with what IS visible rather than dropping it:
  if it needs a surface or a support, use one you can see in the frame (a step,
  a kerb, a low wall, a stall edge, the ground) instead of inventing furniture.
  "Sit down" in a frame with no chair is "she lowers herself onto the stone step",
  never "she walks to a chair and sits". If nothing visible can support the wish,
  write the closest motion the frame does support and keep it to one action.
- No camera or lens jargon, no quality words, no meta commentary.
Keep it under 60 words, one or two sentences.
If the user's wish is empty, invent the subtle ambient motion this image most
naturally supports.
Respond only with JSON containing exactly one variant, with negative_prompt set
to an empty string."""


# The Instruct action's rewrite. It sits next to MOTION_SYSTEM and is its
# opposite in spirit: an instruction editor is not handed a scene to describe,
# it is handed an order, and the failure modes run the other way. A video
# prompt loses by saying too little about the motion; an edit instruction loses
# by saying too much about the picture -- a paragraph describing the finished
# image reads to the model as "repaint all of this", and the untouched two
# thirds of the frame come back subtly redrawn. So this asks for one imperative
# sentence, a findable target and a concrete result, and explicitly forbids the
# generation-prompt habits every other system message here encourages.
EDIT_INSTRUCTION_SYSTEM = """You are an expert at writing edit instructions for
instruction-following image editors (the Qwen-Image-Edit family). The attached
or described image is the picture being edited: the model keeps those exact
pixels and changes only what the instruction tells it to.
Read what the user wants changed and write ONE instruction that gets it, in the
form these models follow most reliably.
How to write it:
- Give an ORDER, not a description. Imperative verb first -- change, replace,
  remove, add, make, turn, open, close. A sentence describing the finished
  picture ("a woman in a red dress in a sunlit cafe") is a generation prompt
  and edits nothing; "change her dress to red" is the instruction.
- Name the target so it can only be one thing in THIS picture: what it is,
  where it is in the frame, and whatever tells it apart from its neighbours
  ("the man on the left", "the tallest bottle on the shelf", "the red car
  behind her"). A vague target ("it", "the thing at the back") is the usual
  reason the wrong object comes back repainted.
- Say what the result should look like, concretely: colour, material, shape,
  direction, expression. "Change the dress to deep burgundy velvet" lands where
  "make the dress nicer" does not.
- Never phrase anything as a negation. These models do not read "don't", "no"
  or "without" and act on the thing named instead: "remove the sunglasses",
  never "she should not be wearing sunglasses".
- Do NOT list what must stay the same. Everything the instruction does not
  mention is already held unchanged, and the app appends that sentence itself.
  Naming the background, the pose or the style only to preserve them puts them
  back in play and widens what the model repaints.
- Keep it to the one change the user asked for. If they asked for two, write
  both in one sentence rather than inventing a third.
- A change that covers the whole picture (black and white, a different medium,
  a season, a time of day) is one transformation stated once -- "convert to
  black and white", "restyle as a loose watercolour painting" -- with no
  description of the composition attached.
- New text in the image goes in quotation marks, spelled exactly as it should
  appear.
- Something being added has to be told where it goes and how big it is,
  anchored to something already visible ("add a tabby cat sitting on the wall
  beside her").
- No camera or lens jargon, no quality words ("masterpiece", "8k", "highly
  detailed"), no style garnish, no meta commentary, no explanation of your
  choices.
Keep it under 40 words, one or two sentences. Longer instructions do not edit
more precisely; they repaint more of the frame.
Respond only with JSON containing exactly one variant, with negative_prompt set
to an empty string."""


# A caption model hands back a title dressed as one -- in quotes, with a full
# stop, sometimes prefixed with "Title:" -- and Civitai shows it verbatim.
_TITLE_LABEL = re.compile(r"^\s*(?:title|caption)\s*[:\-]\s*", re.IGNORECASE)
_TAG_NOISE = re.compile(r"[^a-z0-9 '\u2019-]")


def _clean_title(raw: str, cap: int = 90) -> str:
    """The model's title as a post title: unlabelled, unquoted, unpunctuated."""
    text = " ".join(str(raw).split())
    text = _TITLE_LABEL.sub("", text)
    text = text.strip("\"'\u201c\u201d\u2018\u2019").strip()
    text = text.rstrip(".,;:")
    if len(text) > cap:
        text = text[:cap].rsplit(" ", 1)[0].rstrip(",;") + "\u2026"
    return text


def _clean_caption_tags(raw, limit: int = 5) -> list[str]:
    """The model's tags as post tags: lowercase, unpunctuated, deduped.

    Civitai takes at most five and a tag with a full stop in it is a tag
    nobody else has, so this trims rather than passing the model's list on.
    Order is the model's own: it was asked for the broad ones first.
    """
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for item in raw:
        tag = _TAG_NOISE.sub(" ", str(item).lower())
        tag = " ".join(tag.split()).strip("-' \u2019")
        if len(tag) < 2 or tag in out:
            continue
        out.append(tag)
        if len(out) == limit:
            break
    return out


class LlmError(RuntimeError):
    """An Ollama call that could not be completed, phrased for the user.

    The message reaches the page verbatim (main.py turns it into a 503), so it
    says what went wrong and what to do about it -- never a bare exception
    repr. `str(httpx.ReadTimeout())` is the empty string, which is how a
    timeout used to show up in the log as "ollama attempt 1 failed:" and in
    the UI as nothing at all.
    """


def _reason(exc: Exception, timeout: float) -> str:
    """One actionable sentence for an Ollama failure."""
    if isinstance(exc, httpx.TimeoutException):
        return (
            f"no reply within {timeout:.0f}s. A large model on a busy machine "
            "can exceed that on its first call, while it loads; a second try "
            "usually lands. Otherwise pick a smaller LLM."
        )
    if isinstance(exc, httpx.ConnectError):
        return "cannot reach it. Is `ollama serve` running?"
    if isinstance(exc, httpx.HTTPStatusError):
        body = exc.response.text.strip().replace("\n", " ")[:200]
        return f"it answered HTTP {exc.response.status_code}" + (f": {body}" if body else "")
    if isinstance(exc, (json.JSONDecodeError, KeyError)):
        return (
            "its reply was not the JSON that was asked for. Models without "
            "structured-output support do this on every call."
        )
    return f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__


class OllamaClient:
    def __init__(
        self,
        host: str,
        model: str,
        temperature: float = 0.9,
        request_timeout: float = 300,
    ) -> None:
        self._host = host.rstrip("/")
        self._model = model
        self._temperature = temperature
        self._timeout = request_timeout
        # Capabilities per model tag, as reported by /api/show. Fixed for a
        # given tag, so cache: /api/llms is hit on every page load.
        self._caps: dict[str, set[str]] = {}
        # Serialise chat requests: concurrent calls pile up inside Ollama and
        # the later ones blow the HTTP timeout while just waiting their turn.
        # With the lock, the timeout only covers the actively running request.
        self._lock = asyncio.Lock()
        # Undealt composition principles; see _next_principles.
        self._deck: list[tuple[str, str]] = []

    @property
    def model(self) -> str:
        return self._model

    def set_model(self, name: str) -> None:
        self._model = name

    async def _capabilities(self, name: str) -> set[str] | None:
        """What Ollama says a model can do; None if it won't say."""
        if name in self._caps:
            return self._caps[name]
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                # "model" is the current field name, "name" the older one;
                # the unused key is ignored either way.
                resp = await client.post(
                    self._host + "/api/show", json={"model": name, "name": name}
                )
            resp.raise_for_status()
            caps = resp.json().get("capabilities")
        except (httpx.HTTPError, KeyError, json.JSONDecodeError) as exc:
            log.warning("ollama /api/show failed for %s: %s", name, exc)
            return None  # not cached: a transient failure must not stick
        if not isinstance(caps, list):
            return None  # Ollama too old to report capabilities
        self._caps[name] = {str(c) for c in caps}
        return self._caps[name]

    async def has_vision(self, name: str | None = None) -> bool | None:
        """Whether a model accepts images. None when Ollama doesn't report it.

        Callers must treat None as "allow": on an Ollama build without
        capability reporting, guessing False would disable image input
        against every model, including ones that handle it fine.
        """
        caps = await self._capabilities(name or self._model)
        return None if caps is None else "vision" in caps

    async def list_models(self) -> list[dict]:
        """Installed models, each flagged with whether it can read images.

        Text-only models stay in the list: they are perfectly good for
        enrich/refine, and only the image-reading flows need vision.
        """
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(self._host + "/api/tags")
            resp.raise_for_status()
            names = sorted(m["name"] for m in resp.json().get("models", []))
        except (httpx.HTTPError, KeyError, json.JSONDecodeError) as exc:
            # /api/llms swallows this to keep the page loading, but the model
            # picker POSTs here too, and there a raw httpx error reached the
            # user as "Internal Server Error"
            raise LlmError(f"Ollama: {_reason(exc, 10)}") from exc
        flags = await asyncio.gather(*(self.has_vision(n) for n in names))
        return [{"name": n, "vision": v} for n, v in zip(names, flags)]

    async def is_up(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(self._host + "/api/tags")
                return resp.status_code == 200
        except httpx.HTTPError:
            return False

    async def _chat_json(
        self,
        system: str,
        user: str,
        images: list[str] | None = None,
        schema: dict | None = None,
    ) -> dict:
        """One structured answer from Ollama, retried once.

        `schema` is the JSON shape demanded of the model; the variants shape
        every prompt-writing call wants is the default, and the publish
        dialog's caption call passes its own.
        """
        user_msg: dict = {"role": "user", "content": user}
        if images:
            user_msg["images"] = images
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                user_msg,
            ],
            "format": schema or VARIANTS_SCHEMA,
            "stream": False,
            "options": {"temperature": self._temperature},
        }
        last_error: Exception | None = None
        async with self._lock:
            for attempt in range(2):
                try:
                    async with httpx.AsyncClient(timeout=self._timeout) as client:
                        resp = await client.post(self._host + "/api/chat", json=payload)
                    resp.raise_for_status()
                    content = resp.json()["message"]["content"]
                    return json.loads(content)
                except (httpx.HTTPError, KeyError, json.JSONDecodeError) as exc:
                    last_error = exc
                    log.warning(
                        "ollama attempt %s failed (%s): %s",
                        attempt + 1, type(exc).__name__, _reason(exc, self._timeout),
                    )
                    # A timeout has already cost the caller the full window;
                    # retrying spends it a second time before anyone is told
                    # anything, so a two-attempt call sat silent for 10 minutes
                    # and then 500'd. Hand the timeout back now -- the page
                    # offers a Retry button, which is the same second attempt
                    # with the user deciding to wait for it.
                    if isinstance(exc, httpx.TimeoutException):
                        break
        raise LlmError(f"Ollama ({self._model}): {_reason(last_error, self._timeout)}")

    @staticmethod
    def _strip_meta_tags(text: str, asked_for: str = "") -> str:
        """Drop SD-folklore tags the LLM added of its own accord.

        `asked_for` is the user's own words. A tag they used themselves is kept
        however meta it looks -- "concept art of a castle" is a real request,
        and this filter exists to stop the LLM garnishing prompts, not to
        overrule anybody. Non-tag text is left alone by the callers, so a prose
        prompt never reaches here.
        """
        wanted = asked_for.lower()
        kept = [
            part.strip()
            for part in text.split(",")
            if part.strip()
            and (part.strip().lower() not in META_TAGS or part.strip().lower() in wanted)
        ]
        return ", ".join(kept)

    # Style attributions the LLM adds of its own accord: "art by <name>", "in
    # the style of <name>", "the style is reminiscent of <name>". Measured at
    # 11 of 12 enriched prompts (docs/decisions/PROMPT-STYLE.md), and none of
    # the twelve instructions named an artist. A pattern rather than a list
    # because the names are the open half -- the trigger phrase is the closed
    # one -- and the whole segment goes, since "art by Ilya Kuvshinov" is not
    # improved by dropping the name and keeping "art by".
    #
    # Two halves, and the case-sensitivity is the point of the second one.
    # "<something> by <name>" is authorship however it is written, but
    # "reminiscent of" and "inspired by" are ordinary English -- "reminiscent
    # of a fairytale" describes the picture and stays. What makes those an
    # attribution is a NAME after them, so that half requires a capital letter.
    #
    # A style is not a signature and is left alone throughout: "Alphonse Mucha
    # style", "realistic style" and "in the style of a woodblock print" all
    # stay. Only handing the picture to a named someone goes.
    _ATTRIBUTION = re.compile(
        r"(?i:\b(?:(?:digital\s+|concept\s+)?art(?:work)?|illustration|painting|"
        r"drawn|painted|rendered|render)\s+by\b)"
        r"|(?i:\bin\s+the\s+(?:style|manner)\s+of|\bwork\s+of"
        r"|\breminiscent\s+of|\binspired\s+by|\bevokes?\s+the\s+work\s+of)"
        r"\s+(?:the\s+)?[A-Z]"
    )
    # Words too common to prove the user meant this segment when they turn up
    # in both it and the instruction.
    _COMMON = frozenset(
        "the a an of in on at by is are and or with for style art artwork "
        "illustration painting drawn painted rendered render work scene "
        "picture image this that its it his her their our my".split()
    )

    @classmethod
    def _strip_attributions(cls, text: str, asked_for: str = "") -> str:
        """Drop segments that hand the picture to an artist nobody asked for.

        Same principle as `_strip_meta_tags` and the same escape hatch: a
        segment survives if it shares a real word with the user's own
        instruction, so "in the style of Alphonse Mucha" typed by the user
        stays and the same phrase invented by the LLM goes. Segments here are
        sentences as well as comma parts, because the attribution arrives both
        ways -- ", art by WLOP" and "The style is reminiscent of Ilya
        Kuvshinov." were both in one sample of twelve.
        """
        # most prompts have none, and a prompt this does not act on must come
        # out byte for byte as it went in: the rejoin below normalises spacing
        if not cls._ATTRIBUTION.search(text):
            return text
        wanted = {w for w in re.findall(r"[\w'\u2019-]+", asked_for.lower())
                  if w not in cls._COMMON}
        parts = re.split(r"([,.;])", text)
        out: list[str] = []
        for seg, delim in zip(parts[::2], parts[1::2] + [""]):
            if cls._ATTRIBUTION.search(seg):
                words = {w for w in re.findall(r"[\w'\u2019-]+", seg.lower())
                         if w not in cls._COMMON}
                if not (words & wanted):
                    continue
            out.append(seg.strip() + delim)
        # a dropped segment leaves the one before it holding a comma that now
        # ends the prompt, and two survivors can end up glued without a space
        joined = " ".join(p for p in out if p.strip(" ,.;"))
        return re.sub(r"\s+([,.;])", r"\1", joined).strip(" ,;")

    @staticmethod
    def _dedupe_tags(text: str) -> str:
        """Drop repeated comma-separated tags, keeping first occurrence."""
        seen: set[str] = set()
        out = []
        for part in text.split(","):
            key = part.strip().lower()
            if key and key not in seen:
                seen.add(key)
                out.append(part.strip())
        return ", ".join(out)

    @classmethod
    def _clean_tags(cls, text: str, asked_for: str) -> str:
        """Everything that has to be true of a prompt the LLM wrote.

        Style attributions go first because they are the only pass that reads
        sentences rather than comma segments, and it is cheaper to find "art by
        WLOP" before the text has been cut up. The blocklist then does the rest
        of the removing: it
        catches the folklore, including words Danbooru does have but these
        checkpoints render as haze ("concept art" is a real tag with 8k posts)
        and words it does not have at all ("volumetric lighting"), which used
        to disappear further down by accident. The vocabulary then fixes
        spellings the model was never trained on, and nothing else -- see
        Vocabulary.repair for why it stopped dropping what it does not
        recognise. Dedupe comes last because a respelling can collapse two
        segments into one.
        """
        cleaned = cls._strip_attributions(text, asked_for)
        cleaned = cls._strip_meta_tags(cleaned, asked_for)
        cleaned = vocab.get().repair(cleaned, asked_for)
        return cls._dedupe_tags(cleaned)

    # The model is asked for the negative in its own JSON field, and sometimes
    # writes it into the positive one as well, as a labelled tail. Left alone
    # that is not cosmetic: the whole floor ("bad hands, watermark, censored")
    # would be sent to Draw Things as things to PUT IN the picture. Seen on
    # gemma3 for both variants of one request, so it is not a rare slip.
    _INLINE_NEG = re.compile(
        r"[\s,.;]*\bnegative[ _-]?prompts?\s*[:\-]", re.IGNORECASE
    )

    @classmethod
    def _strip_inlined_negative(cls, text: str) -> str:
        m = cls._INLINE_NEG.search(text)
        return text[:m.start()].rstrip() if m else text

    @staticmethod
    def _ensure_opener(text: str, opener: str) -> str:
        """Guarantee the model's mandated opener, punctuated.

        Several checkpoints do not merely prefer their opener -- BAS+'s card
        calls its trigger tags required -- and the guidance asks for it, but
        asking is not getting. Observed: CyberRealistic's opener ends in
        "realistic" and the model ran it straight into the first word of the
        description, giving "realistic 3girls", which is neither the opener nor
        the count tag.
        """
        if not opener:
            return text
        body = text.lstrip()
        if body[:len(opener)].lower() == opener.lower():
            rest = body[len(opener):]
            # already punctuated, or nothing follows
            if not rest.strip() or rest[0] in ",.;":
                return body
            return f"{opener}, {rest.lstrip()}"
        return f"{opener}, {body}" if body else opener

    @classmethod
    def _parse_variants(
        cls, data: dict, k: int, style: str = "prose", asked_for: str = "",
        opener: str = "",
    ) -> list[PromptVariant]:
        def positive(raw: str) -> str:
            text = cls._strip_inlined_negative(raw)
            text = cls._clean_tags(text, asked_for) if style == "tags" else text.strip()
            return cls._ensure_opener(text, opener)

        variants = [
            PromptVariant(
                prompt=positive(v["prompt"]),
                negative_prompt=cls._dedupe_tags(v.get("negative_prompt", "")),
            )
            for v in data.get("variants", [])
            if v.get("prompt", "").strip()
        ]
        if not variants:
            raise LlmError("LLM returned no usable prompt variants")
        return variants[:k]

    @staticmethod
    def _budget_instruction(budget) -> str:
        """What to tell the LLM about how much prompt this model reads.

        Advisory on purpose. Nothing truncates a prompt anywhere in this app,
        because nothing is being cut off: past a model's budget the tail is
        still encoded, it just arrives too faint to change the picture, and
        everything else in the prompt is diluted along with it (measured in
        docs/decisions/PROMPT-LENGTH.md). Telling the writer is the fix;
        cutting the writing off in the middle is not.
        """
        if not budget:
            return ""
        return (
            f"LENGTH. Keep the whole prompt under {budget.full} characters. Up "
            f"to there every word still affects the picture. Past about "
            f"{budget.fading} characters the end of the prompt is unlikely to "
            "reach the image at all, and a long prompt does not merely lose "
            "its tail -- it renders EVERYTHING it asks for more faintly, "
            "including the subject, which comes back small and far away.\n"
            "ORDER. Put the most important things FIRST: who or what the "
            "picture is of and what they are doing, then their appearance, "
            "then the setting, then lighting and style. Whatever matters least "
            "goes last, because that is what fades first. If a draft runs long, "
            "cut from the end and cut description of the background before "
            "description of the subject."
        )

    @staticmethod
    def _neg_instruction(uses_negative_prompt: bool, style: str) -> str:
        if not uses_negative_prompt:
            return NEG_OFF
        return NEG_TAGS if style == "tags" else NEG_ON

    def _next_principles(self, k: int) -> list[tuple[str, str]]:
        """`k` distinct composition principles, dealt from a shuffled deck.

        A deck rather than an independent random sample per call: sampling
        afresh repeats a principle on consecutive Enrich runs often enough to
        notice, and the point of the rotation is that the next round of drafts
        is framed unlike this one. The deck is reshuffled only once it empties,
        so every principle is used once before any is used twice; a reshuffle
        mid-deal drops the ones already dealt in this call, which would
        otherwise hand two variants the same principle.
        """
        k = min(k, len(COMPOSITION_PRINCIPLES))
        picked: list[tuple[str, str]] = []
        while len(picked) < k:
            if not self._deck:
                deck = random.sample(
                    COMPOSITION_PRINCIPLES, len(COMPOSITION_PRINCIPLES)
                )
                self._deck = [p for p in deck if p not in picked]
            picked.append(self._deck.pop())
        return picked

    @staticmethod
    def _composition_instruction(picked: list[tuple[str, str]]) -> str:
        if not picked:
            return ""
        lines = "\n".join(
            f"- Variant {i}: {name} -- {how}"
            for i, (name, how) in enumerate(picked, 1)
        )
        return COMPOSITION_HEADER + lines

    @staticmethod
    def _with_guidance(system: str, guidance: str) -> str:
        if not guidance.strip():
            return system
        return (
            system
            + "\n\nTarget-model prompt format requirements (these override any "
            + "conflicting general guidance above):\n"
            + guidance.strip()
        )

    async def enrich(
        self,
        instruction: str,
        k: int,
        uses_negative_prompt: bool,
        style: str = "prose",
        guidance: str = "",
        mode: str = "explore",
        opener: str = "",
        budget=None,
    ) -> list[PromptVariant]:
        tags = style == "tags"
        hold = mode == "hold"
        template = ENRICH_SYSTEM_TAGS if tags else ENRICH_SYSTEM
        variation = (HOLD_TAGS if hold else VARY_TAGS) if tags else (
            HOLD_PROSE if hold else VARY_PROSE
        )
        # Tag prompts already carry their own framing rules (exactly one framing
        # tag, one atmosphere tag, subject dominates) and their vocabulary is a
        # closed tag list, so the prose principles are not dealt into them. In
        # hold mode nobody gets one: the picture's composition is the thing
        # being preserved, and an assigned principle would re-frame it.
        composition = (
            ""
            if tags or hold
            else self._composition_instruction(self._next_principles(k))
        )
        system = template.format(
            k=k,
            neg_instruction=self._neg_instruction(uses_negative_prompt, style),
            budget_instruction=self._budget_instruction(budget),
            composition_instruction=composition,
            variation_instruction=variation.format(k=k),
            variation_lead=HOLD_LEAD if hold else VARY_LEAD,
        )
        data = await self._chat_json(self._with_guidance(system, guidance), instruction)
        return self._parse_variants(data, k, style, instruction, opener)

    async def describe(
        self,
        image_b64: str,
        style: str = "prose",
        uses_negative_prompt: bool = False,
        hint: str = "",
        guidance: str = "",
        opener: str = "",
        budget=None,
    ) -> PromptVariant:
        system = DESCRIBE_SYSTEM.format(
            style_instruction=DESCRIBE_TAGS if style == "tags" else DESCRIBE_PROSE,
            neg_instruction=self._neg_instruction(uses_negative_prompt, style),
        )
        if budget:
            system = f"{system}\n{self._budget_instruction(budget)}"
        user = hint.strip() or "Describe this image as a generation prompt."
        data = await self._chat_json(
            self._with_guidance(system, guidance), user, images=[image_b64]
        )
        return self._parse_variants(data, 1, style, hint, opener)[0]

    async def caption(self, image_b64: str) -> ArtworkCaption:
        """A title, a description and browse-by tags, read off the picture.

        The publish dialog's "simple post": what the image IS, rather than what
        was asked of the model that made it. Nothing about the request reaches
        this call -- not the prompt, not the instruction, not the model's
        dialect -- because the whole reason a simple post exists is that those
        describe a picture that may never have arrived.

        Cleaned rather than trusted: a caption model reaches for a quoted title
        with a full stop on the end and tags that repeat the title, none of
        which a Civitai post wants.
        """
        data = await self._chat_json(
            ARTWORK_SYSTEM,
            "Title and describe this picture.",
            images=[image_b64],
            schema=ARTWORK_SCHEMA,
        )
        return ArtworkCaption(
            title=_clean_title(data.get("title", "")),
            description=" ".join(str(data.get("description", "")).split())[:400],
            tags=_clean_caption_tags(data.get("tags"), limit=5),
        )

    async def motion(
        self,
        wish: str,
        still_prompt: str,
        image_b64: str | None = None,
        seconds: float = 2.0,
    ) -> PromptVariant:
        """Rewrite a motion wish into an image-to-video prompt.

        `image_b64` is optional: with a vision model the clip's first frame
        grounds what can plausibly move, without one the still's own prompt
        stands in for it.
        """
        user = (
            f"Prompt the first frame was generated from:\n{still_prompt}\n\n"
            f"Clip length: {seconds:.1f} seconds.\n\n"
            f"What the user wants to happen:\n{wish.strip() or '(not specified)'}"
        )
        data = await self._chat_json(
            MOTION_SYSTEM, user, images=[image_b64] if image_b64 else None
        )
        return self._parse_variants(data, 1)[0]

    async def edit_instruction(
        self,
        wish: str,
        source_prompt: str,
        image_b64: str | None = None,
    ) -> PromptVariant:
        """Rewrite a plain-words edit wish into an instruction-editor order.

        `image_b64` is optional in the same way it is for `motion`: a vision
        model can name the target as it actually appears in the picture ("the
        woman on the left", "the taller of the two mugs"), which is exactly
        what an instruction editor needs to repaint the right object; without
        one the source image's own prompt is all there is to point at.
        """
        # labelled loosely because it is not always a prompt: editing an edit
        # makes the parent's "prompt" the previous instruction, and calling
        # that a generation prompt invites the LLM to treat it as a
        # description of what is in the picture
        user = (
            "How the image was made (its prompt, or the instruction of an "
            f"earlier edit):\n{source_prompt}\n\n"
            f"What the user wants changed:\n{wish.strip()}"
        )
        data = await self._chat_json(
            EDIT_INSTRUCTION_SYSTEM, user, images=[image_b64] if image_b64 else None
        )
        # prose parsing on purpose: this is an order to an edit model, not a
        # prompt in a checkpoint's dialect, so no tag cleaning and no quality
        # opener -- "masterpiece, best quality" on the front of "make her
        # smile" is an instruction to repaint the picture
        return self._parse_variants(data, 1)[0]

    async def refine(
        self,
        liked_prompt: str,
        tweak: str,
        k: int,
        uses_negative_prompt: bool,
        style: str = "prose",
        guidance: str = "",
        opener: str = "",
        budget=None,
    ) -> list[PromptVariant]:
        """Merge `tweak` into `liked_prompt`: the picture's prompt, changed.

        There was briefly a second mode here that wrote a prompt for a
        rectangle's contents instead, for adding something by repaint. That
        route is gone and so is the mode (EDITING.md decision 12); an addition
        is an instruction now, and instructions are written by
        `edit_instruction`.
        """
        template = REFINE_SYSTEM_TAGS if style == "tags" else REFINE_SYSTEM
        system = template.format(
            k=k, neg_instruction=self._neg_instruction(uses_negative_prompt, style),
            budget_instruction=self._budget_instruction(budget),
        )
        user = f"Original prompt:\n{liked_prompt}\n\nRequested change:\n{tweak}"
        data = await self._chat_json(self._with_guidance(system, guidance), user)
        # the liked prompt counts as asked-for: it is already on screen and its
        # tags were the user's to edit before they pressed Edit
        return self._parse_variants(data, k, style, f"{liked_prompt} {tweak}", opener)
