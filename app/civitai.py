"""Publishing a finished image to Civitai, over the official MCP endpoint.

Why this is two calls to a JSON-RPC endpoint and not a REST POST: until some
time in 2026 there was no way to do it at all. Civitai's public REST API was
read-only -- browse models, download weights, nothing that writes -- and the
documented answer to "can I upload through the API" was no, use the web UI.
The workaround of the period was `civitai.com/intent/post`, a Twitter-style web
intent that hands the browser a pre-filled post form; it needs a publicly
reachable `mediaUrl`, which an app on localhost does not have, so it was never
an option here.

What replaced it is `https://mcp.civitai.com/mcp`: MCP tools over JSON-RPC,
authenticated with an ordinary personal API key rather than OAuth. `upload_image`
takes base64 bytes -- so nothing has to be publicly reachable -- and returns a
UUID; `create_post` attaches that UUID to a new post and optionally publishes
it, in one call that cleans up after itself if any part fails. Writing needs the
MediaWrite scope, which a Full API key carries.

Only two tools are used and the payloads are three fields each, so this speaks
JSON-RPC directly rather than pulling in an MCP client library. The endpoint is
stateless for `tools/call`: no `initialize` handshake, no session id.

The part the API will not do for us is generation metadata. `create_post` has no
prompt/seed/sampler fields -- Civitai reads those off the image file when it
ingests it, from the A1111 `parameters` text chunk. Draw Things does not write
one (it writes the same information as XMP, which Civitai's parser does not
read), so `embed_parameters` below builds the chunk from the database row. That
is why a published image carries its prompt and a bare upload does not.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
from pathlib import Path
from typing import Optional, Sequence

import httpx
from PIL import Image, PngImagePlugin

from . import ROOT
from . import vocab

log = logging.getLogger(__name__)

ENDPOINT = "https://mcp.civitai.com/mcp"

# The key file scripts/setup.sh already knows about, as a fallback for installs
# that keep the token in a file rather than in .env. Both are gitignored.
KEY_FILE = ROOT / "civitai.key"

# Civitai's uploader caps a single image; the number is the documented default
# and is checked here so an oversized 4K PNG fails with something actionable
# instead of a JSON-RPC error from the far end.
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

# Enhancement passes: they sharpen or repair an image without composing it, and
# their own row holds that pass's numbers (strength 0.3, 20 steps) rather than
# what made the picture. Publishing those as "the parameters" would be a lie, so
# source_of() walks through them to the ancestor that did compose. `edit` is not
# here on purpose -- a Refine changes the prompt, so it is its own source. An
# instruction editor's edit is walked through, but by model rather than by kind
# (see source_of): its row holds an order and a model nobody drew with.
PASSTHROUGH_KINDS = {"upscale", "esrgan", "detail"}

# Draw Things' sampler names against the ones A1111 metadata uses, which is what
# Civitai's parser matches on. Anything unlisted goes through as written: an
# unrecognised sampler is shown verbatim, which is better than dropping it.
SAMPLER_ALIASES = {
    "Euler Ancestral": "Euler a",
    "Euler A Trailing": "Euler a",
    "DDIM Trailing": "DDIM",
    "UniPC Trailing": "UniPC",
}


def token() -> Optional[str]:
    """The Civitai API key, or None when this install has not set one up.

    CIVITAI_TOKEN first (app/__init__ has already applied .env to the
    environment by the time this runs), then the civitai.key file that
    scripts/setup.sh uses. A blank value counts as unset.
    """
    env = (os.environ.get("CIVITAI_TOKEN") or "").strip()
    if env:
        return env
    try:
        return (KEY_FILE.read_text().strip() or None)
    except OSError:
        return None


def available() -> bool:
    """Whether publishing can be offered at all. Cheap: no network."""
    return token() is not None


def source_of(gen, ancestors: dict, editors=frozenset()):
    """The generation whose parameters describe this picture.

    `ancestors` maps id -> Generation for everything in the session. An upscale
    of a faithful of a draft publishes the faithful's prompt, seed and steps,
    because those are what produced what you are looking at. Falls back to the
    row itself when the chain is broken (a deleted parent) rather than failing.

    `editors` are the instruction editors' model keys. Their edits are walked
    through as well: the row holds the order it was given ("replace the head
    piece with jewelry") and a model with no Civitai version, so publishing it
    attached no generation data at all. The checkpoint that drew the picture
    being edited is the one to credit, with its prompt and numbers.
    """
    seen = set()
    cur = gen
    while (
        (cur.kind in PASSTHROUGH_KINDS or cur.model_key in editors)
        and cur.parent_id and cur.parent_id not in seen
    ):
        seen.add(cur.id)
        nxt = ancestors.get(cur.parent_id)
        if nxt is None:
            break
        cur = nxt
    return cur


def settings_line(
    gen,
    model_name: str,
    version_id: Optional[int] = None,
    size: Optional[tuple[int, int]] = None,
    also: Sequence[Optional[int]] = (),
    loras: Sequence = (),
) -> str:
    """The comma-separated settings half of the `parameters` chunk.

    Split out from `parameters` because the publish dialog lets the prompt be
    rewritten but not these: they are measurements of a render that already
    happened, and a post that claims the wrong seed or step count is worse than
    one that claims nothing.

    `size` is the one field that describes the file rather than the render.
    Every other number here comes from the generation that composed the picture
    -- for an upscale that is an ancestor, see `source_of` -- but the pixels
    being uploaded are the upscale's, and a 3K post that reads `Size: 1536x1536`
    because its parent rendered at 1536 is telling Civitai the wrong thing about
    the file it is attached to. Callers publishing one row's bytes with another
    row's parameters pass the bytes' dimensions; left out, the row describes
    itself, which is right for everything that is not a passthrough pass.

    `Civitai resources` is what actually attributes the checkpoint. Matching by
    `Model hash` is not open to us -- Draw Things renders from a converted
    .ckpt, which does not hash to the safetensors on the model page -- so the
    version id from models.yaml goes in instead, and the model name rides along
    as the human-readable half.

    `also` credits the other models that worked on this picture -- the
    checkpoint that ran an upscale or a face pass, when it is not the one that
    composed it. Both took part, and either may be the only one Civitai knows:
    a picture composed by a Draw Things catalogue model (FLUX, Krea, Z-Image,
    Qwen -- no Civitai page, no version id) and upscaled through an Illustrious
    checkpoint used to be published claiming no resource at all. The composer
    stays first and keeps `Model:` to itself.

    `loras` are the LoRAs the picture was rendered with (LoraSpec, as recorded
    on the rows). One with a Civitai version is credited after the
    checkpoints, in the form Civitai's own generator writes -- type, weight,
    version; one without is left out rather than named by a filename Civitai
    cannot resolve.
    """
    sampler = SAMPLER_ALIASES.get(gen.sampler or "", gen.sampler or "")
    fields = [
        f"Steps: {gen.steps}",
        f"Sampler: {sampler}" if sampler else None,
        f"CFG scale: {gen.cfg:g}",
        f"Seed: {gen.seed}",
        "Size: {}x{}".format(*(size or (gen.width, gen.height))),
        f"Model: {model_name}",
    ]
    ids = list(dict.fromkeys(int(v) for v in (version_id, *also) if v))
    resources = [{"type": "checkpoint", "modelVersionId": v} for v in ids]
    credited = set()
    for lora in loras:
        v = lora.civitai_version_id
        if v and v not in credited:
            credited.add(v)
            resources.append({"type": "lora", "weight": lora.weight, "modelVersionId": int(v)})
    if resources:
        fields.append("Civitai resources: " + json.dumps(resources, separators=(",", ":")))
    return ", ".join(f for f in fields if f)


def parameters(
    gen,
    model_name: str,
    version_id: Optional[int] = None,
    prompt: Optional[str] = None,
    negative_prompt: Optional[str] = None,
    size: Optional[tuple[int, int]] = None,
    also: Sequence[Optional[int]] = (),
    loras: Sequence = (),
) -> str:
    """The A1111 `parameters` chunk for one generation.

    The layout is fixed by convention rather than by a spec: prompt on the
    first lines, `Negative prompt:` on its own line if there is one, then a
    single comma-separated line of settings. Civitai's parser keys on exactly
    this shape.

    `prompt` and `negative_prompt` override what the row holds, which is what
    the publish dialog sends when the user has edited them. Either may be an
    empty string, meaning "publish this picture with no prompt at all"; None
    means untouched, so a caller that does not care gets the row's own text.

    A chunk with no prompt is the whole shape of a simple post, so it is a
    settings line and nothing else -- not a blank first line, which is a prompt
    that happens to be empty and reads to a parser as one.

    `size` passes through to `settings_line`: the dimensions of the image being
    published, when they are not this row's own. So does `also`: the other
    models that worked on the picture, and `loras`: the LoRAs they ran with.
    """
    text = gen.prompt if prompt is None else prompt
    negative = gen.negative_prompt if negative_prompt is None else negative_prompt
    lines = [text] if text else []
    if negative:
        lines.append(f"Negative prompt: {negative}")
    lines.append(settings_line(gen, model_name, version_id, size, also, loras))
    return "\n".join(lines)


def embed_parameters(png: bytes, params: str) -> bytes:
    """The same PNG with an A1111 `parameters` text chunk added.

    Re-encoded rather than spliced: Pillow is already here for every other
    image pass, and hand-writing a chunk into the byte stream is a lot of care
    for one string. Draw Things' own XMP block is dropped in the round trip,
    which loses nothing -- its content is the same facts in a form Civitai does
    not read, and the database row is the record that matters.
    """
    img = Image.open(io.BytesIO(png))
    meta = PngImagePlugin.PngInfo()
    meta.add_text("parameters", params)
    out = io.BytesIO()
    img.save(out, format="PNG", pnginfo=meta)
    return out.getvalue()


# Words that are never a useful post tag: the checkpoints' own quality dials,
# the folklore the prompt guard already strips, and the gallery bait. All three
# sets exist for other reasons; this is the fourth place they earn their keep.
def _skip_tags() -> frozenset[str]:
    from .llm import META_TAGS  # local: llm is a big module about another job
    from .schemas import DEFAULT_TAG_PREFIX
    opener = {w.strip().lower() for w in DEFAULT_TAG_PREFIX.split(",")}
    return frozenset(
        vocab.QUALITY_TOKENS | vocab.QUALITY_FOLKLORE | META_TAGS | opener | _RESOLUTION_TAGS
    )


# Danbooru meta tags about the file rather than the picture. They are real tags
# with real post counts, so the vocabulary happily returns them, and they are
# never something anyone browses a gallery by.
_RESOLUTION_TAGS = frozenset(
    {"absurdres", "incredibly absurdres", "highres", "lowres", "hires",
     "official art", "scan", "artist request", "commentary", "translated"}
)


# Longest phrase looked up as a single tag. Danbooru's mass is 1-3 words
# ("long hair", "looking at viewer"), and every extra width costs a pass over
# the prompt for tags that would not be good post tags anyway.
_MAX_TAG_WORDS = 3

_TAG_WORD = re.compile(r"[a-z0-9'\u2019]+")
_TAG_SEGMENT = re.compile(r"[,;.]")


def suggest_tags(prompt: str, spec=None, limit: int = 5) -> list[str]:
    """Post tags for this prompt, best first, at most `limit`.

    Ranked by Danbooru post count, descending, which sounds like the wrong way
    round and is not: a Civitai post tag wants to be broad and recognisable
    ("1girl", "cat", "forest"), and post count is exactly a measure of how
    broad a tag is. The fine-grained tags that make a prompt good -- "scottish
    fold", "dappled sunlight" -- are the ones nobody browses by.

    Works on prose and tag prompts alike, because it looks up every 1-3 word
    window rather than trusting the commas. A window that is not in the
    vocabulary simply does not match.
    """
    v = vocab.get()
    if not v:
        return []
    skip = set(_skip_tags())
    # The model's own opener is on every prompt it ever renders, so it says
    # nothing about this picture. Taking it from the spec rather than listing
    # words here means a model with its own dial ("score_7", "anime_coloring",
    # "absurdres") is handled by the entry that already declares it.
    if spec is not None:
        skip.update(
            part.strip().lower()
            for part in getattr(spec, "quality_opener", "").split(",")
            if part.strip()
        )
    found: dict[str, int] = {}
    for segment in _TAG_SEGMENT.split(prompt.lower()):
        words = _TAG_WORD.findall(segment)
        # A segment that IS boilerplate is not looked inside. Checking only the
        # windows let "quality" through the middle of "best quality" -- a real
        # Danbooru tag, a real post count, and nothing whatever about the
        # picture. Skipping the whole segment is both the fix and the honest
        # rule: if the phrase says nothing, no part of it does either.
        if " ".join(words) in skip:
            continue
        # A tag prompt's segments ARE tags, so any window in one is meant. A
        # prose clause is a sentence, and a single word plucked out of the
        # middle of it is incidental -- "stone harbour wall at dawn" is about a
        # harbour, not about "stone", "wall" and "cold". Requiring a phrase
        # there keeps the accidental nouns out, at the cost of prose prompts
        # offering fewer tags, which is the right way round to be wrong.
        floor = 2 if v.is_prose(" ".join(words)) else 1
        for width in range(floor, _MAX_TAG_WORDS + 1):
            for i in range(len(words) - width + 1):
                phrase = " ".join(words[i:i + width])
                if phrase in skip or len(phrase) < 3:
                    continue
                # the spelling the model was trained on, when the prompt used
                # another: "blonde hair" is stored as "blonde hair" but plenty
                # of the vocabulary is reached through an alias
                canon = v.canonical(phrase) or phrase
                if canon in skip:
                    continue
                count = v.count(canon)
                if count:
                    found[canon] = count
    if not found:
        return []
    ranked = sorted(found, key=lambda t: -found[t])
    # A one-word tag already inside a better-ranked phrase says nothing extra:
    # "hair" beside "long hair" spends one of five slots on nothing.
    kept: list[str] = []
    for tag in ranked:
        words = set(tag.split())
        if any(words < set(k.split()) for k in kept):
            continue
        kept.append(tag)
    # An anime checkpoint's output is anime whatever the prompt said, and it is
    # the tag people actually browse by. Prose models here are photographic or
    # painterly and have no single equivalent, so they get nothing.
    if spec is not None and getattr(spec, "prompt_style", "") == "tags":
        kept = ["anime"] + [t for t in kept if t != "anime"]
    return kept[:limit]


_LEADING_SEGMENT = re.compile(r"^\s*([^,;.]+?)\s*[,;.]\s*")
_LEADING_WORD = re.compile(r"^\s*([A-Za-z0-9_'\u2019]+)\s*[,;.]?\s+")


def _strip_boilerplate(text: str, skip: frozenset[str]) -> str:
    """`text` with any leading quality tags peeled off, the rest untouched.

    Peels a whole segment when the segment is boilerplate, and failing that a
    single leading word -- a hand-typed prompt drops the comma often enough
    ("very aesthetic, absurdres 1girl full body") that segments alone leave one
    stray tag at the front of every title. Only ever removes from the front,
    and never rebuilds what is left, so the remainder keeps its own commas and
    full stops instead of being re-joined with punctuation it never had.
    """
    while True:
        # the last segment carries no separator, so it is only ever caught here
        if text.strip().rstrip(".,;").lower() in skip:
            return ""
        seg = _LEADING_SEGMENT.match(text)
        if seg and seg.group(1).strip().lower() in skip:
            text = text[seg.end():]
            continue
        word = _LEADING_WORD.match(text)
        if word and word.group(1).strip().lower() in skip:
            text = text[word.end():]
            continue
        return text


def suggest_title(instruction: str, prompt: str, cap: int = 90) -> str:
    """A title for the post: the user's own words for what they asked for.

    The session instruction first, the prompt only when there is none.

    The instruction is not trustworthy on its own, which cost a wrong title
    before it was noticed. `Store.save_prompts` writes the composer's text back
    over `sessions.instruction` -- by design, it is what names the session in
    the picker -- so on any session that has been through the composer the
    "instruction" IS a prompt, quality opener and all. Both sources therefore
    get the same treatment: peel the boilerplate off the front, keep the rest
    verbatim.
    """
    skip = _skip_tags()
    # an instruction that is nothing but boilerplate is no better than none:
    # fall through to the prompt rather than titling the post "Best quality"
    text = _strip_boilerplate(" ".join((instruction or "").split()), skip).strip()
    if not text:
        text = _strip_boilerplate(" ".join(prompt.split()), skip).strip()
    text = text.rstrip(".,;")
    if len(text) > cap:
        # cut at a word so the title does not end mid-word
        text = text[:cap].rsplit(" ", 1)[0].rstrip(",;") + "…"
    return text[:1].upper() + text[1:]


# The tool that made the picture, as a sentence the description can carry.
#
# Civitai has a Tools taxonomy and "Draw Things" is in it, but nothing we are
# allowed to call can set it: `create_post` takes title, detail, images, tags,
# modelVersionId and collectionId and nothing else, no tool on the MCP endpoint
# touches the taxonomy, and the metadata route is closed too -- their parser
# derives the tool by sniffing the format, and the only ones it knows are
# automatic1111, comfyui, swarmui and ruinedfooocus. An A1111 chunk is always
# read as automatic1111 however it is written. Their tRPC layer does expose the
# taxonomy, but the publish path is not going to depend on an unversioned
# endpoint that can change without notice.
#
# So the tool goes in the one field that does reach the page. The technical
# description says it already; this is for the simple post, whose description
# is written by the vision model and would otherwise drop it.
PROVENANCE = "Rendered locally in Draw Things."


def with_provenance(description: str) -> str:
    """`description` with the tool named at the end of it.

    Left alone when it already says so -- the caption model is told not to name
    software and does not, but a description the user has typed themselves goes
    through here too on the way out of a re-read, and repeating the sentence
    under their own would read as a bug.
    """
    text = " ".join(description.split())
    if "draw things" in text.lower():
        return text
    if not text:
        return PROVENANCE
    if text[-1] not in ".!?":
        text += "."
    return f"{text} {PROVENANCE}"


def suggest_detail(gen, source, spec) -> str:
    """A one-line description: what made it, and at what size.

    Deliberately not a settings dump -- Civitai renders the prompt, seed,
    sampler and steps from the metadata already embedded in the file, and
    repeating them under it is noise. What that panel does not say is which
    app drove the render and whether the picture was enlarged after it.
    """
    label = spec.label if spec is not None else gen.model_key
    text = f"{label}, rendered locally in Draw Things at {source.width}×{source.height}"
    if (gen.width, gen.height) != (source.width, source.height):
        text += f" and upscaled to {gen.width}×{gen.height}"
    return text + "."


class CivitaiError(RuntimeError):
    """A publish that did not happen, with the reason in the message."""


class CivitaiClient:
    """The two write tools, over JSON-RPC.

    One client per call site; there is no session to keep alive.
    """

    def __init__(self, api_token: str, endpoint: str = ENDPOINT, timeout: float = 180):
        self.token = api_token
        self.endpoint = endpoint
        self.timeout = timeout

    async def _call(self, tool: str, arguments: dict) -> dict | str:
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": tool, "arguments": arguments},
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                resp = await client.post(
                    self.endpoint,
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {self.token}",
                        "Content-Type": "application/json",
                        # the endpoint may answer either way; both are handled
                        # in _unwrap below
                        "Accept": "application/json, text/event-stream",
                    },
                )
            except httpx.HTTPError as exc:
                raise CivitaiError(f"could not reach Civitai: {exc}") from exc
        if resp.status_code == 401:
            raise CivitaiError(
                "Civitai rejected the API key. Check CIVITAI_TOKEN in .env, or "
                "civitai.key, against civitai.com -> Account settings -> API Keys."
            )
        if resp.status_code >= 400:
            raise CivitaiError(f"Civitai returned HTTP {resp.status_code}")
        return self._unwrap(tool, resp.text)

    @staticmethod
    def _unwrap(tool: str, body: str) -> dict | str:
        """The tool's own result, or CivitaiError with what the far end said.

        Two envelopes to get through: JSON-RPC, which reports protocol errors in
        `error`, and MCP, which reports tool errors as `isError` with the message
        in the text content. A tool that fails upstream (a muted account, an
        un-onboarded one) arrives as the second, HTTP 200 and all.
        """
        # an SSE-framed answer carries the same JSON on a `data:` line
        if body.lstrip().startswith("event:") or body.lstrip().startswith("data:"):
            body = "\n".join(
                line[5:].strip()
                for line in body.splitlines()
                if line.startswith("data:")
            )
        try:
            envelope = json.loads(body)
        except ValueError as exc:
            raise CivitaiError(f"{tool}: unreadable response from Civitai") from exc
        if isinstance(envelope.get("error"), dict):
            raise CivitaiError(f"{tool}: {envelope['error'].get('message', 'failed')}")
        result = envelope.get("result") or {}
        texts = [
            c.get("text", "")
            for c in result.get("content", [])
            if isinstance(c, dict) and c.get("type") == "text"
        ]
        if result.get("isError"):
            raise CivitaiError(f"{tool}: {' '.join(texts).strip() or 'failed'}")
        # newer servers put the parsed object here; older ones only have text
        if isinstance(result.get("structuredContent"), dict):
            return result["structuredContent"]
        joined = "\n".join(texts).strip()
        try:
            parsed = json.loads(joined)
        except ValueError:
            return joined
        return parsed if isinstance(parsed, dict) else {"result": parsed}

    async def upload_image(self, png: bytes) -> str:
        """Stage a PNG and return its UUID. Not visible to anyone yet.

        Verified against the live endpoint: the answer carries
        `structuredContent: {uuid, width, height, contentType}` alongside a
        human-readable text rendering of the same. _unwrap prefers the former.
        """
        if len(png) > MAX_UPLOAD_BYTES:
            raise CivitaiError(
                f"image is {len(png) / 1e6:.1f}MB, over Civitai's "
                f"{MAX_UPLOAD_BYTES // 1024 // 1024}MB upload limit. Publish the "
                "image it was upscaled from, which is smaller."
            )
        result = await self._call(
            "upload_image",
            {
                "data": base64.b64encode(png).decode(),
                "contentType": "image/png",
            },
        )
        uuid = _first(result, "uuid", "id", "imageId", "key")
        if not uuid:
            raise CivitaiError(f"Civitai accepted the upload but named no image: {result}")
        return str(uuid)

    async def create_post(
        self,
        *,
        uuid: str,
        width: int,
        height: int,
        title: str = "",
        detail: str = "",
        tags: Optional[list[str]] = None,
        model_version_id: Optional[int] = None,
    ) -> dict:
        """Create the post as a draft. Returns {"id", "published_at", "url"}.

        `create_post` can publish in the same call and this deliberately does
        not use it. Publishing a draft is one click on the post page you are
        sent to anyway, so a second route to it here saves nothing and is the
        only irreversible thing the app could do (see docs/decisions/PUBLISHING.md).

        Verified against the live endpoint: `structuredContent` is
        `{ok, id, imageIds, published, publishedAt, nsfwLevel, url}`. A draft
        comes back with `published: false` and `publishedAt: null`, which is why
        the timestamp stored here is our own and not Civitai's.
        """
        args: dict = {
            "images": [{"uuid": uuid, "width": width, "height": height, "type": "image"}],
            "publish": False,
        }
        if title:
            args["title"] = title
        if detail:
            args["detail"] = detail
        if tags:
            args["tags"] = tags[:5]  # Civitai takes at most five
        if model_version_id:
            args["modelVersionId"] = int(model_version_id)
        result = await self._call("create_post", args)
        post_id = _first(result, "id", "postId", "post_id")
        if post_id is None:
            raise CivitaiError(f"Civitai created no post: {result}")
        return {
            "id": int(post_id),
            "published_at": _first(result, "publishedAt", "published_at"),
            # the server sends the canonical link; post_url() is the fallback
            "url": _first(result, "url") or post_url(int(post_id)),
        }


def _first(result, *keys):
    """The first of `keys` present in a tool result, at the top level or nested.

    The tools answer with a small object, but whether it is bare or wrapped in
    `post`/`image`/`data` is the server's business and has already changed once.
    Looking one level down costs nothing and survives that.
    """
    if not isinstance(result, dict):
        return None
    for key in keys:
        if result.get(key) not in (None, ""):
            return result[key]
    for value in result.values():
        if isinstance(value, dict):
            found = _first(value, *keys)
            if found is not None:
                return found
    return None


def post_url(post_id: int) -> str:
    return f"https://civitai.com/posts/{post_id}"
