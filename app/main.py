"""FastAPI app: orchestrates Ollama enrichment + Draw Things generation.

Run with:  uv run uvicorn app.main:app --port 8500
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import math
import random
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from .detailer import (
    DEFAULT_FACE_STRENGTH,
    add_in_region,
    build_collage_init,
    build_combine_frame,
    combine,
    combine_prompt,
    combine_words,
    eligible_faces,
    refine_faces,
    repaint_region,
)
from . import civitai, dims, esrgan, video
from .dt_client import DrawThingsClient, DrawThingsStalled
from . import vocab
from .llm import META_TAGS, LlmError, OllamaClient, image_for_vision
from .registry import load_registry
from .schemas import (
    AnimateRequest,
    CombineRequest,
    CreateSessionRequest,
    RenameSessionRequest,
    DescribeRequest,
    DetailRequest,
    DraftRequest,
    EditPromptRequest,
    EditRequest,
    EnrichRequest,
    EnrichResponse,
    EsrganRequest,
    FinalizeRequest,
    Generation,
    MoreLikeThisRequest,
    MotionPromptRequest,
    PromptVariant,
    PublishRequest,
    RefineRequest,
    HideUninstalledRequest,
    LoraChoiceRequest,
    LoraSpec,
    ModelEnabledRequest,
    SavePromptsRequest,
    StepsRequest,
    UpscaleRequest,
)
from .sessions import StorageMisconfigured, Store
from .settings import STAGES, SettingsStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
log = logging.getLogger("draftpool")

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

VISION_REQUIRED = (
    "The selected Ollama model ({model}) is text-only and cannot read images. "
    "Pick a vision model in the LLM dropdown (gemma3 and qwen3-vl both work) "
    "for this action; text-only models are fine for Enrich, Amend and Refine."
)

registry = load_registry()
registry.resolve_installed()
store = Store()
settings_store = SettingsStore()
# whatever this install has changed the shipped step counts to, laid over the
# registry before anything can render with the defaults
registry.apply_lora_choices(settings_store.lora_choices())
registry.apply_step_overrides(settings_store.steps())
# ...and which models this install has hidden from the chips and pickers. Only
# the UI honours it: every endpoint still renders a hidden model asked for by
# key, so a card made before the model was hidden keeps all of its buttons.
registry.apply_model_visibility(
    settings_store.disabled_models(), settings_store.hide_uninstalled()
)
dt = DrawThingsClient(registry.draw_things.host, registry.draw_things.request_timeout)
llm = OllamaClient(
    registry.llm.host, registry.llm.model, registry.llm.temperature,
    registry.llm.request_timeout,
)

job_queue: asyncio.Queue[str] = asyncio.Queue()
# session_id -> set of subscriber queues for SSE
subscribers: dict[str, set[asyncio.Queue]] = {}


def notify(session_id: str, event: dict) -> None:
    for q in subscribers.get(session_id, set()):
        q.put_nowait(event)


def random_seed() -> int:
    return random.randint(0, 2**31 - 1)


# Standing guard appended to instruct edits: told to change one thing, the
# model will happily drift the rest of the frame unless told not to, so the
# request carries the "hold everything else" sentence by default instead of
# the user retyping it.
EDIT_GUARD = "Keep everything else exactly unchanged."

# A rectangle is the whole frame the editor is given, and these models compose
# to fill a frame: asked for a creature "two steps tall" in a strip of stairs,
# it drew one whose head is above the top edge and the render itself came back
# cropped. Nothing downstream can recover that, so it is asked for up front.
REGION_FRAME_GUARD = (
    "Show the whole subject inside the frame, complete and not cropped by any edge."
)

# Image-to-video used to get the same treatment (ANIMATE_GUARD, mechanically
# appended to every clip), but "no objects added or removed" fights any wish
# that needs one -- "she sits down" with no chair in frame, say -- and the
# model tries to honour both, which is where the melted-geometry clips came
# from. It is a suggestion now, not a constraint: MOTION_SYSTEM leans the LLM
# rewrite toward holding the scene steady by default, and a user who wants the
# hard guarantee can still type it into the motion prompt themselves.
ANIMATE_GUARD_SUGGESTION = (
    "Nothing else changes: same subject, same scene, same style, "
    "no objects added or removed, no cut to another shot."
)


def stage_loras(stage) -> list[dict]:
    """The LoRAs a job queued on this stage is recorded with, fixed and ticked.

    Taken when the job is queued, not when it renders, and kept on the row:
    a tick changed while the job waits, or before a Retry, must not change
    what the card was made with (see Generation.loras).
    """
    return [l.model_dump() for l in stage.loras]


def negative_for(spec, text: str) -> str:
    """The negative prompt to store for a generation on this model.

    A model that uses a negative prompt should never be sent an empty one: the
    paths that skip the LLM (your prompt used as written, Generate with no
    enrichment, and anything inheriting from a generation created before this
    existed) had nothing to put there, and an SDXL anime checkpoint renders
    visibly softer without it. Whatever was actually written still wins; this
    only fills a blank, and is stored on the row so the record says what was
    generated rather than what was asked for.
    """
    if not spec.uses_negative_prompt:
        return ""
    return text.strip() or spec.default_negative


def prompt_for(spec, text: str) -> str:
    """The prompt to store for a generation on this model, opener and all.

    The counterpart to `negative_for`, and it exists for a bug the enrichment
    path cannot fix by itself. Several checkpoints treat their opener as
    required -- BAS+'s card says so outright -- and `_opener_for` only hands
    one to the LLM when every model in the enrich call agrees on it. Two tag
    models with different openers are enriched together (they share a dialect,
    which is what the grouping is for), so they agree on nothing and the LLM
    is given no opener to honour. What came back then was whatever it wrote
    freehand: an opener it half-remembered, run into the first tag without a
    comma -- "very aesthetic, absurdres 1girl full body" -- and BAS+'s
    required trigger tags nowhere in the prompt at all.

    Doing it here instead means one enriched prompt can serve every model of
    its dialect and each still gets its own opener, which is what the
    composer's grouping already assumes. A leading opener belonging to some
    *other* model is stripped first, since after enrichment that is exactly
    what a shared prompt carries; only a string that is precisely one of the
    registry's openers is ever removed, longest first so the short openers
    that prefix the long ones cannot claim them.

    Applied at queue time rather than at enrich time so the row records what
    was actually rendered, the same reason `negative_for` fills its blank here.
    """
    if not spec.quality_opener:
        return text
    body = text.lstrip()
    # the shipped openers too: while a LoRA adds tags to one, the plain form
    # is what the LLM was told to write, and left unrecognised it would be
    # kept as body and the opener written twice
    openers = sorted(
        {o for m in registry.models.values()
         for o in (m.quality_opener, m.shipped_opener) if o},
        key=len, reverse=True,
    )
    for opener in openers:
        if body[: len(opener)].lower() == opener.lower():
            body = body[len(opener):].lstrip(" ,")
            break
    return OllamaClient._ensure_opener(body, spec.quality_opener)


def with_guard(prompt: str, guard: str, *already_said: str) -> str:
    """`prompt` plus `guard`, unless the prompt already says as much."""
    low = prompt.lower()
    if any(phrase in low for phrase in already_said):
        return prompt
    return prompt.rstrip(". ") + ". " + guard


def draft_dims(spec, stage, aspect: str) -> tuple[int, int, str]:
    """Frame size for a from-scratch render by `spec` at the requested ratio.

    Returns the ratio actually used alongside the size, because it is not
    always the one asked for: a model without supports_aspect renders square
    whatever the buttons said, and the row records what it got rather than what
    was requested, so a later Finalize of that draft reproduces the square it
    can see. Only text-to-image goes through here -- an img2img stage takes its
    shape from the picture it is given (see inherited_aspect), whatever the
    model doing the work was trained on.
    """
    used = (
        aspect
        if spec.supports_aspect and aspect in dims.ASPECTS
        else dims.DEFAULT_ASPECT
    )
    w, h = dims.stage_dims(stage.size, dims.ratio(used), spec.aspect_size)
    return w, h, used


def drawn_by(gen: Generation) -> Generation:
    """The picture whose model and prompt describe `gen`'s pixels.

    Usually `gen` itself. Not when an instruction editor made it: its row
    carries the editor's key and the order it was given ("replace the head
    piece with jewelry"), and neither is a way to draw the picture again. A
    pass over such a picture -- Upscale, Fix faces, More like this -- belongs
    to the checkpoint that drew what was edited, so this walks up through
    instruction edits (and through anything that copied their model key, like
    a 2x fast) to the first picture a generator made. A removed draft still
    answers: its row keeps the prompt and the model. Returns the last row
    reached when the chain stops before one, so callers check the model.
    """
    seen: set[str] = set()
    cur = gen
    while cur.id not in seen:
        seen.add(cur.id)
        spec = registry.models.get(cur.model_key)
        if spec is None or not spec.edit_only or not cur.parent_id:
            return cur
        up = store.get_generation(cur.parent_id)
        if up is None:
            return cur
        cur = up
    return cur


def _drawing_spec(parent: Generation, drawn: Generation):
    """The checkpoint that drew `drawn`, or a 400 saying why there is none."""
    try:
        spec = registry.model_or_raise(drawn.model_key)
    except (KeyError, ValueError) as exc:
        raise HTTPException(400, f"the model that drew this picture is not available: {exc}")
    if spec.edit_only:
        raise HTTPException(
            400, "this picture was made by an instruction editor and nothing it "
            "was edited from is left to say which model drew it",
        )
    return spec


def edit_orders(gen: Generation) -> list[str]:
    """The instructions that turned drawn_by(gen) into `gen`, oldest first.

    In the user's words: the guards the app appends to every order are taken
    off, and a Combine contributes what the user typed plus what its subject
    picture was drawn from, since the scene's prompt knows nothing of it.
    """
    orders: list[str] = []
    seen: set[str] = set()
    cur = gen
    while cur.id not in seen:
        seen.add(cur.id)
        spec = registry.models.get(cur.model_key)
        if spec is None or not spec.edit_only:
            break
        if cur.kind == "edit":
            if cur.source_id:
                words = combine_words(cur.prompt)
                subject = store.get_generation(cur.source_id)
                if subject is not None:
                    words += f" (the subject: {drawn_by(subject).prompt})"
            else:
                words = cur.prompt
                for guard in (EDIT_GUARD, REGION_FRAME_GUARD):
                    words = words.replace(guard, "")
            if words.strip():
                orders.append(words.strip())
        up = store.get_generation(cur.parent_id) if cur.parent_id else None
        if up is None:
            break
        cur = up
    return orders[::-1]


async def edited_prompt(gen: Generation, parent: Generation, spec) -> Optional[str]:
    """`gen`'s prompt with the edits behind `parent` worked in, or None.

    For an upscale of an instruction edit. The pass is a detail pass, but at
    its strength the words still move details -- measured: with no prompt the
    eyes changed colour, with the pre-edit prompt the removed headdress grew
    back (docs/decisions/UPSCALING.md, decision 9). So the picture's own
    prompt, changed the way the picture was, by the same merge Refine uses.
    None when there is nothing to merge or the LLM cannot do it; the job then
    renders with the prompt it was queued with rather than waiting for Ollama.
    """
    orders = edit_orders(parent)
    if not orders:
        return None
    try:
        # from the drawn picture's prompt every time, not gen.prompt: a retry
        # would otherwise merge the same edits into an already merged prompt
        variants = await llm.refine(
            drawn_by(parent).prompt, " Then: ".join(orders), 1, spec.uses_negative_prompt,
            spec.prompt_style, spec.prompt_guidance, spec.quality_opener,
            spec.prompt_budget,
        )
    except Exception as exc:  # noqa: BLE001 - the pre-edit prompt still works
        log.warning("could not merge the edits into %s's prompt: %s", gen.id, exc)
        return None
    return variants[0].prompt if variants else None


def inherited_aspect(parent: Generation) -> str:
    """The ratio a child of `parent` should render at.

    The stored ratio when there is one, so a 1536 final of a 768 draft is the
    shape that was picked and not the shape 768 could round to. Rows made
    before ratios existed, and anything whose pixels came from elsewhere (a
    Real-ESRGAN 2x, a region repaint), fall back to whichever preset their
    pixels are closest to.
    """
    if parent.aspect in dims.ASPECTS:
        return parent.aspect
    return dims.nearest_aspect(parent.width, parent.height)


def edit_ceiling(spec, sizes) -> int:
    """The longest edge `spec` can render: 2048 unaided, or the largest offered
    size when it tiles the denoiser. Above its ceiling a model returns a flat
    colour field and reports success, which is the failure this cap exists to
    prevent (UPSCALING.md Decision 1)."""
    return max(sizes) if spec.tiled_diffusion else 2048


def edit_target(spec, width: int, height: int, full_size: bool) -> int:
    """The pixel budget a full-image edit of this picture should render at.

    Normally the model's own working size, which is what it was tuned for and
    keeps the pass to seconds. `full_size` asks for the picture's own budget
    instead, so an edit of an upscaled image keeps it; `edit_dims` then holds
    the long edge under what the model can render. The max() means the flag
    can never render *smaller* than the working size, which is what an
    instruct model needs from a small draft.
    """
    target = spec.final.size
    if full_size:
        target = max(target, dims.stage_size(width, height))
    return target


def edit_dims(width: int, height: int, size: int, ceiling: int) -> tuple[int, int]:
    """Frame for a full-image edit of a width x height picture at the `size` budget.

    A stage size is an area here as everywhere else (dims.stage_dims): the
    picture keeps its shape and the sides are the equal-area solution on the
    64px grid, so a 16:9 picture edits at 1344x768 on a 1024 model -- the
    frame that model draws 16:9 at -- and not at 1024x576. Scaling the long
    edge down to the size instead, which is what this did first, handed an
    instruction editor 56% of the area it works at (EDITING.md Decision 17).
    `ceiling` bounds the long edge (edit_ceiling): a strip past the offered
    shapes would carry it past what the model can render, so the budget gives
    way until the long edge fits. No 512px floor here, unlike region crops, or
    a panorama distorts. static/index.html mirrors this to say what an edit
    will return before it runs; tests/test_edit_dims.py fails if the two drift
    apart.
    """
    r = width / height
    size = min(size, int(ceiling / math.sqrt(max(r, 1 / r))))
    return dims.stage_dims(size, r)


def edit_frame(spec, width: int, height: int, full_size: bool, sizes) -> tuple[int, int]:
    """The frame a full-image edit of a width x height picture by `spec` runs in.

    The picture's own size unless it is larger than the model's working
    budget: editing an upscaled 2048px image at 2048px is minutes of wasted
    compute. An instruction editor barely functions below ~1MP, so for one a
    smaller picture is scaled UP to the working budget too; a generator's
    img2img pass over a small draft just runs at the draft's size, as it
    always did. Compared as stage sizes, not long edges, so a picture the app
    drew at this budget in any shape passes through untouched.
    """
    current = dims.stage_size(width, height)
    if current > spec.final.size or (spec.edit_only and current < spec.final.size):
        target = edit_target(spec, width, height, full_size)
        return edit_dims(width, height, target, edit_ceiling(spec, sizes))
    return width, height


async def run_esrgan_job(gen) -> None:
    store.set_status(gen.id, "running")
    notify(gen.session_id, {"id": gen.id, "status": "running"})
    started = time.monotonic()
    try:
        parent = store.get_generation(gen.parent_id) if gen.parent_id else None
        if parent is None:
            raise RuntimeError("parent generation not found")
        flavor = "anime" if "anime" in (gen.sampler or "") else "photo"
        png = await esrgan.upscale2x(store.image_path(parent).read_bytes(), flavor)
        store.save_image(gen, png)
        duration = round(time.monotonic() - started, 1)
        store.set_status(gen.id, "done", duration_s=duration)
        notify(gen.session_id, {"id": gen.id, "status": "done", "duration_s": duration})
    except Exception as exc:  # noqa: BLE001 - surfaced to the UI
        log.exception("esrgan %s failed", gen.id)
        store.set_status(gen.id, "error", str(exc)[:500])
        notify(gen.session_id, {"id": gen.id, "status": "error", "error": str(exc)[:200]})


async def wait_for_draw_things(gen) -> None:
    """Hold a job until Draw Things answers, instead of failing it.

    The app may simply be closed, or its API Server toggle reset; holding lets
    a queue drain by itself once it is back, with no manual retries.
    """
    waited = False
    while not await dt.is_up():
        if not waited:
            log.warning("Draw Things unreachable, holding job %s", gen.id)
            waited = True
        await asyncio.sleep(10)


async def render_or_hold(gen, spec, render):
    """Run a Draw Things render, holding the job if the renderer has stalled.

    A stalled session answers every request with 200 and an empty image list,
    so failing here would fail every job queued behind this one too, and the
    Retry button fails just as fast -- which is what the user sees. Hold
    instead, the same way an unreachable Draw Things is held, until a cheap
    probe render proves the renderer works again (normally once the app has
    been restarted), then take one more attempt.

    A second stall on a session that has just provably rendered is this
    request's own problem rather than the session's -- an unsupported size for
    the model, say -- so that one is allowed to fail, with the distinction
    spelled out in the message.
    """
    for attempt in (1, 2):
        try:
            return await render()
        except DrawThingsStalled as exc:
            if attempt == 2:
                raise DrawThingsStalled(
                    f"{exc} This happened again right after Draw Things "
                    "rendered a test image, so the session is working and this "
                    "size or model is what it will not render."
                ) from exc
            log.warning("draw things stalled on job %s, holding it", gen.id)
            store.set_status(gen.id, "pending")
            notify(gen.session_id, {"id": gen.id, "status": "pending"})
            while not await dt.renders(spec.dt_filename):
                await asyncio.sleep(30)
            store.set_status(gen.id, "running")
            notify(gen.session_id, {"id": gen.id, "status": "running"})


async def run_video_job(gen) -> None:
    """Animate the parent still into a clip. Minutes to hours per job."""
    try:
        spec = registry.video_model_or_raise(gen.model_key)
        parent = store.get_generation(gen.parent_id) if gen.parent_id else None
        if parent is None:
            raise RuntimeError("parent image not found for video job")
        init_png = store.image_path(parent).read_bytes()
    except Exception as exc:  # noqa: BLE001 - surfaced to the UI
        store.set_status(gen.id, "error", str(exc)[:500])
        notify(gen.session_id, {"id": gen.id, "status": "error", "error": str(exc)[:200]})
        return

    await wait_for_draw_things(gen)
    store.set_status(gen.id, "running")
    notify(gen.session_id, {"id": gen.id, "status": "running"})
    started = time.monotonic()
    num_frames = gen.num_frames or 33
    fps = gen.fps or 16
    try:
        frames = await dt.img2video(
            init_png,
            prompt=gen.prompt,
            model_filename=spec.dt_filename,
            # the size decided when the job was queued: it carries the still's
            # aspect, which spec.width/height do not
            width=gen.width or spec.width,
            height=gen.height or spec.height,
            steps=gen.steps,
            cfg=gen.cfg,
            seed=gen.seed,
            num_frames=num_frames,
            fps=fps,
            motion_scale=spec.motion_scale,
            start_frame_guidance=spec.start_frame_guidance,
            guiding_frame_noise=spec.guiding_frame_noise,
            shift=spec.shift,
            # the default request timeout is minutes; these jobs run for
            # tens of minutes, so allow generous headroom over the estimate
            timeout=max(1800.0, spec.estimate_seconds(num_frames) * 3.0),
        )
        # the poster frame doubles as the card thumbnail, and lets the existing
        # image plumbing (paths, deletion) treat a video like any generation
        store.save_image(gen, frames[0])
        width, height = video.frame_size(frames)
        store.set_dimensions(gen.id, width, height)
        path = video.assemble(
            frames, fps, store.data_dir / "sessions" / gen.session_id, gen.id
        )
        store.set_video_ext(gen.id, path.suffix.lstrip("."))
        duration = round(time.monotonic() - started, 1)
        store.set_status(gen.id, "done", duration_s=duration)
        notify(
            gen.session_id,
            {"id": gen.id, "status": "done", "duration_s": duration,
             "frames": len(frames), "video_ext": path.suffix.lstrip(".")},
        )
    except Exception as exc:  # noqa: BLE001 - surfaced to the UI
        log.exception("video %s failed", gen.id)
        store.set_status(gen.id, "error", str(exc)[:500])
        notify(gen.session_id, {"id": gen.id, "status": "error", "error": str(exc)[:200]})


# Wait this many times the model's own measured pace before giving up on a
# render. The video path uses the same factor against its own estimate.
RENDER_TIMEOUT_SAFETY = 3.0
# A rate read off history can be wrong in both directions -- a job held while
# Draw Things restarted carries the hold in its duration -- so the computed
# wait is bounded. Four hours is far past any single still measured here and
# still short enough that a renderer that has died is reported the same day.
RENDER_TIMEOUT_CEILING = 4 * 3600.0

# The floor for a rectangle rendered by an instruction editor: one request at
# about a megapixel on a very large model. 703s measured at 36 steps, so the
# flat 900 is not the margin it looks like.
REGION_INSTRUCT_TIMEOUT = 2400.0


def render_timeout(gen, spec) -> Optional[float]:
    """How long to wait for this one render, or None for the flat default.

    The flat 900s default fits every model here at its shipped settings and
    none of them at a raised step count: qwen-edit runs about 20 s/step at
    1024, so the 16-step Instruct default lands at 320s and a 50-step edit at
    over 1000 -- past the timeout, at which point httpx gives up while Draw
    Things carries on and finishes a render nothing can collect any more (see
    DrawThingsTimeout). The step count is the dial the UI hands the user, so
    the timeout has to move with it.

    Measured per model rather than configured per model: the store already
    records how long every finished job took, and one number in models.yaml
    could not follow a per-install step override anyway. A model with no
    history at this size falls back to the configured timeout, which is what
    every job ran on before.
    """
    rate = store.render_rate(spec.key, max(gen.width, gen.height))
    if rate is None:
        return None
    work = gen.steps * (gen.strength if gen.strength is not None else 1.0)
    wanted = rate * work * RENDER_TIMEOUT_SAFETY
    return min(RENDER_TIMEOUT_CEILING, max(registry.draw_things.request_timeout, wanted))


async def run_job(gen_id: str) -> None:
    gen = store.get_generation(gen_id)
    if gen is None:
        return
    if gen.kind == "esrgan":
        # no Draw Things involved: runs even while DT jobs are held
        await run_esrgan_job(gen)
        return
    if gen.kind == "video":
        # model_key refers to video_models, so this must branch before the
        # image registry lookup below
        await run_video_job(gen)
        return
    try:
        # A key that has since been removed or renamed in models.yaml reaches
        # here on any row queued before the edit -- including rows the lifespan
        # resumes at startup. That is a fault of this card, not of the queue.
        spec = registry.model_or_raise(gen.model_key)
    except (KeyError, ValueError) as exc:
        log.warning("generation %s cannot run: %s", gen.id, exc)
        store.set_status(gen.id, "error", str(exc)[:500])
        notify(gen.session_id, {"id": gen.id, "status": "error", "error": str(exc)[:200]})
        return
    stage = {"draft": spec.draft, "upscale": spec.upscale}.get(gen.kind, spec.final)
    # Draw Things may be closed, or Ollama stopped (Finalize+ needs it for
    # the describe step): hold the job instead of failing it, so a queue of
    # cards completes without manual retries once the service is back
    needs_llm = bool(gen.describe_first and gen.parent_id)
    waited = False
    while not await dt.is_up() or (needs_llm and not await llm.is_up()):
        if not waited:
            log.warning("Draw Things or Ollama unreachable, holding job %s", gen.id)
            waited = True
        await asyncio.sleep(10)
    store.set_status(gen.id, "running")
    notify(gen.session_id, {"id": gen.id, "status": "running"})
    started = time.monotonic()
    try:
        if gen.describe_first and gen.parent_id:
            # Finalize+: extract a detailed recreation prompt from the parent
            # image before generating, so the hi-res pass is anchored to what
            # the user actually saw
            parent = store.get_generation(gen.parent_id)
            if parent is None:
                raise RuntimeError("parent generation not found for describe step")
            if await llm.has_vision() is False:
                # the LLM was switched to a text-only model after queueing
                raise RuntimeError(VISION_REQUIRED.format(model=llm.model))
            image_b64 = base64.b64encode(store.image_path(parent).read_bytes()).decode()
            hint = (
                f"The image was generated from this prompt: {parent.prompt}\n"
                "Preserve that intent, and capture every visual detail of the "
                "image so it can be recreated precisely."
            )
            variant = await llm.describe(
                image_b64, spec.prompt_style, spec.uses_negative_prompt, hint,
                spec.prompt_guidance, spec.quality_opener, spec.prompt_budget,
            )
            store.update_prompts(gen.id, variant.prompt, variant.negative_prompt)
            gen = store.get_generation(gen.id)
        if gen.kind == "upscale" and gen.parent_id:
            parent = store.get_generation(gen.parent_id)
            editor = registry.models.get(parent.model_key) if parent else None
            if editor is not None and editor.edit_only:
                merged = await edited_prompt(gen, parent, spec)
                if merged:
                    store.update_prompts(gen.id, merged, gen.negative_prompt)
                    gen = store.get_generation(gen.id)
        base_common = dict(
            prompt=gen.prompt,
            negative_prompt=gen.negative_prompt,
            model_filename=spec.dt_filename,
            width=gen.width,
            height=gen.height,
            steps=gen.steps,
            cfg=gen.cfg,
            seed=gen.seed,
            sampler=gen.sampler,
            clip_skip=spec.clip_skip,
            # as recorded when it was queued; a row from before that was
            # recorded renders with the stage's own, as it always did
            loras=[l.model_dump() for l in (gen.loras if gen.loras is not None else stage.loras)],
            # only consulted at >=2048, so the face and region passes below
            # (which work at 1024) are unaffected
            tiled_diffusion=spec.tiled_diffusion,
        )

        # Only the whole-image passes get it. A detail or region pass is a
        # handful of small crops, one request each, and none of them is
        # anywhere near the flat timeout however many steps it is given.
        #
        # Except one: a rectangle handed to an instruction editor is a single
        # request at roughly the model's full working size, and a 20GB editor
        # at 36 steps took 703s of the flat 900 in a measured run. Past the
        # timeout httpx gives up while Draw Things finishes a render nobody can
        # collect, so that path gets the same measured allowance as a
        # whole-image pass, and never less than the flat one.
        timeout = render_timeout(gen, spec)
        # a Combine is one request on a frame larger than the picture it
        # returns, so the rate for that picture's size undercounts it
        if gen.kind == "edit" and (gen.region or gen.source_id) and spec.edit_only:
            timeout = max(timeout or 0, REGION_INSTRUCT_TIMEOUT)

        async def render() -> bytes:
            # a fresh copy per attempt: the detail and edit paths drop width
            # and height from it, and a held job that resumes needs them back
            common = dict(base_common)
            if gen.kind in ("faithful", "upscale", "detail", "edit"):
                if not spec.supports_img2img:
                    raise RuntimeError(f"{spec.label} does not support img2img")
                parent = store.get_generation(gen.parent_id) if gen.parent_id else None
                if parent is None:
                    raise RuntimeError("parent generation not found for img2img job")
                init_override = store.init_path(gen)
                init_png = (
                    init_override.read_bytes()
                    if init_override.is_file()
                    else store.image_path(parent).read_bytes()
                )
                if gen.kind == "detail":
                    del common["width"], common["height"]
                    # a region on a detail job is a hand-picked face: stored as
                    # [l, t, r, b] like every other region, converted here to the
                    # (x, y, w, h) the detector would have produced
                    manual = json.loads(gen.region) if gen.region else None
                    png = await refine_faces(
                        dt,
                        init_png,
                        anime=spec.prompt_style == "tags",
                        strength=gen.strength or DEFAULT_FACE_STRENGTH,
                        faces=(
                            [(manual[0], manual[1],
                              manual[2] - manual[0], manual[3] - manual[1])]
                            if manual else None
                        ),
                        **common,
                    )
                elif gen.kind == "edit" and gen.source_id:
                    # the init is the two-panel frame the request built; the
                    # scene panel of what comes back is the picture
                    if not init_override.is_file():
                        raise RuntimeError("the combine frame for this job is missing")
                    del common["width"], common["height"]
                    png = await combine(
                        dt, init_png, scene_w=gen.width, timeout=timeout, **common
                    )
                elif gen.kind == "edit" and gen.region and spec.edit_only:
                    # an instruction editor pointed at a rectangle: it renders
                    # the window with the new thing in it and only the pixels
                    # it moved are kept, so the picture's own light, ground and
                    # resolution survive (EDITING.md 10)
                    del common["width"], common["height"]
                    png = await add_in_region(
                        dt, init_png,
                        box=json.loads(gen.region),
                        work_size=spec.final.size,
                        # one big request, unlike every other region pass: it
                        # needs the allowance computed above or a slow render
                        # is finished by Draw Things and lost by the client
                        timeout=timeout,
                        **common,
                    )
                elif gen.kind == "edit" and gen.region:
                    del common["width"], common["height"]
                    png = await repaint_region(
                        dt, init_png,
                        box=json.loads(gen.region),
                        # 0.7 is where a change lands without the window being
                        # re-invented; an addition arrives from the dialog at
                        # 0.85, which is a different edit rather than a bolder
                        # one. See repaint_region and EDITING.md 8 and 9.
                        strength=gen.strength or 0.7,
                        **common,
                    )
                else:
                    strength = gen.strength or (
                        spec.upscale.strength if gen.kind == "upscale" else spec.final.strength
                    )
                    if gen.kind == "upscale":
                        if registry.upscaler is None:
                            raise RuntimeError(
                                "no Draw Things upscaler installed; the upscale "
                                "pass needs one to prepare its init"
                            )
                        # a sharp init is worth more to this pass than anything the
                        # low strength could add on its own; see dt.enlarge
                        init_png = await dt.enlarge(
                            init_png,
                            target=max(gen.width, gen.height),
                            upscaler=registry.upscaler,
                            model_filename=spec.dt_filename,
                            sampler=gen.sampler,
                            cfg=gen.cfg,
                            seed=gen.seed,
                        )
                    png = await dt.img2img(
                        init_png, strength, timeout=timeout, **common
                    )
            else:
                png = await dt.txt2img(timeout=timeout, **common)
            return png

        # Draw Things can answer 200 with no image at all; that is the session
        # failing, not this job, so it is held rather than failed
        png = await render_or_hold(gen, spec, render)
        store.save_image(gen, png)
        duration = round(time.monotonic() - started, 1)
        store.set_status(gen.id, "done", duration_s=duration)
        notify(gen.session_id, {"id": gen.id, "status": "done", "duration_s": duration})
    except Exception as exc:  # noqa: BLE001 - job errors are surfaced to the UI
        log.exception("generation %s failed", gen.id)
        store.set_status(gen.id, "error", str(exc)[:500])
        notify(gen.session_id, {"id": gen.id, "status": "error", "error": str(exc)[:200]})


async def worker() -> None:
    while True:
        gen_id = await job_queue.get()
        try:
            await run_job(gen_id)
        except Exception:  # noqa: BLE001
            # run_job guards its own body, but anything raised before that
            # guard -- an unknown model key from a config edit, a row the
            # store cannot map -- used to escape this loop and kill the
            # worker for the life of the process. Nothing restarts it and
            # /api/health does not watch it, so every later job sat at
            # "pending" while Draw Things went idle and every service still
            # reported up. One bad row must not take the queue with it.
            log.exception("job %s raised outside run_job's handler", gen_id)
            try:
                gen = store.get_generation(gen_id)
                if gen is not None and gen.status not in ("done", "error"):
                    store.set_status(gen_id, "error", "job could not be started")
                    notify(gen.session_id, {
                        "id": gen_id, "status": "error",
                        "error": "job could not be started",
                    })
            except Exception:  # noqa: BLE001
                log.exception("could not mark job %s as failed", gen_id)
        finally:
            job_queue.task_done()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Printed every start because the failure it catches is silent: a typo in
    # DRAFTPOOL_DATA_DIR, or an update run from a folder that never had it set,
    # points the store at an empty directory, which it creates and then reports
    # as an install with no history at all rather than as an error.
    log.info(
        "data directory: %s (%d session(s)); override with DRAFTPOOL_DATA_DIR",
        store.data_dir, len(store.list_sessions()),
    )
    # a clip left rendering by the previous process is still rendering inside
    # Draw Things, and these jobs will queue behind it (see the shutdown
    # warning below and docs/decisions/VIDEO.md)
    stranded = store.running_video_generations()
    # resume jobs that a previous server process left unfinished
    resumed = store.unfinished_generations()
    for gen_id in resumed:
        job_queue.put_nowait(gen_id)
    if resumed:
        log.info("resumed %d unfinished generation(s)", len(resumed))
    if stranded:
        log.warning(
            "%d video job(s) were rendering when this server last stopped. Draw "
            "Things cannot be told to cancel, so it may still be working on that "
            "clip and delivering it to nobody -- queued jobs wait behind it, for "
            "up to the clip's full render time. Restart Draw Things to clear it.",
            len(stranded),
        )
    task = asyncio.create_task(worker())
    yield
    # Draw Things exposes no cancel: /sdapi/v1/interrupt, skip, cancel and stop
    # all answer 400, and dropping the HTTP connection does not stop the render.
    # So a clip in flight keeps rendering after this process exits, its result
    # goes nowhere, and the job resumed on the next start queues behind it --
    # paying the full cost twice. Nothing in the UI would explain the wait, so
    # say it here.
    in_flight = store.running_video_generations()
    if in_flight:
        log.warning(
            "shutting down with %d video render(s) in flight: Draw Things will "
            "keep rendering %s and the result will be lost. The job resumes on "
            "next start but waits behind that render; restart Draw Things too if "
            "you want the queue clear.",
            len(in_flight),
            ", ".join(in_flight),
        )
    task.cancel()


app = FastAPI(title="draftpool", lifespan=lifespan)


@app.exception_handler(LlmError)
async def llm_failed(request, exc: LlmError) -> JSONResponse:
    """Give the page the reason instead of a bare 500.

    Every LLM route (enrich, describe, refine, animate-prompt) raised this
    uncaught, so an Ollama outage reached the browser as "Internal Server
    Error" with the actual cause only in the server log. 503, because the
    request was fine and trying it again is the right move.
    """
    log.warning("LLM call failed: %s", exc)
    return JSONResponse({"detail": str(exc), "retry": True}, status_code=503)


# -- static / images -------------------------------------------------------


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/images/{session_id}/{gen_id}.png")
async def image(session_id: str, gen_id: str) -> FileResponse:
    path = store.data_dir / "sessions" / session_id / f"{gen_id}.png"
    if not path.is_file():
        raise HTTPException(404)
    return FileResponse(path)


# -- meta -------------------------------------------------------------------


@app.get("/api/health")
async def health() -> dict:
    dt_up, llm_up = await asyncio.gather(dt.is_up(), llm.is_up())
    return {
        "draw_things": dt_up,
        # answering but not rendering: the failure mode is_up() cannot see, and
        # the one that needs the user to restart the app rather than wait
        "draw_things_stalled": dt_up and dt.stalled,
        "ollama": llm_up,
        "esrgan": esrgan.available(),
        "civitai": civitai.available(),
    }


async def _require_vision() -> None:
    """Reject image-reading actions on a text-only LLM, with the reason why."""
    # `is False` on purpose: None means this Ollama build doesn't report
    # capabilities, in which case let the request through rather than
    # blocking a model that may well be fine.
    if await llm.has_vision() is False:
        raise HTTPException(400, VISION_REQUIRED.format(model=llm.model))


@app.get("/api/llms")
async def list_llms() -> dict:
    try:
        available = await llm.list_models()
    except Exception:
        available = []
    current_vision = next(
        (m["vision"] for m in available if m["name"] == llm.model), None
    )
    return {
        "current": llm.model,
        "current_vision": current_vision,
        "available": available,
    }


@app.post("/api/llms")
async def select_llm(body: dict) -> dict:
    name = (body.get("model") or "").strip()
    available = await llm.list_models()
    chosen = next((m for m in available if m["name"] == name), None)
    if chosen is None:
        raise HTTPException(400, f"unknown Ollama model: {name}")
    # text-only models are allowed here: they only lose the image-reading
    # flows, which say so when used (see _require_vision)
    llm.set_model(name)
    log.info("LLM switched to %s (vision=%s)", name, chosen["vision"])
    return {"current": name, "vision": chosen["vision"]}


@app.get("/api/video-models")
async def video_models() -> list[dict]:
    registry.resolve_installed()
    return [
        {
            "key": v.key,
            "label": v.label,
            "available": v.available,
            "installed": v.installed,
            # the settings tick and what it works out to; see /api/models
            "enabled": v.enabled,
            "shown": v.shown,
            "width": v.width,
            "height": v.height,
            "steps": v.steps,
            # minutes for the two presets, so the UI can warn before a job that
            # runs for an hour: cost grows faster than frame count
            "est_minutes": {
                str(n): max(1, v.estimate_seconds(n) // 60) for n in (17, 33, 49, 81)
            },
        }
        for v in registry.video_models.values()
    ]


@app.post("/api/generations/{gen_id}/animate")
async def animate(gen_id: str, req: AnimateRequest) -> Generation:
    """Queue an image-to-video job from a finished still."""
    parent = _get_gen_or_404(gen_id)
    if parent.status != "done":
        raise HTTPException(400, "parent image not available yet")
    if parent.kind == "video":
        raise HTTPException(400, "that is already a clip; animate a still instead")
    try:
        spec = registry.video_model_or_raise(req.model_key)
    except (KeyError, ValueError) as exc:
        raise HTTPException(400, str(exc))
    frames = video.snap_frame_count(req.num_frames)
    # the model's configured size is a pixel budget, not a fixed shape: sending
    # a square still to 832x480 flattens it, so take the aspect from the still
    width, height = video.clip_size(
        parent.width, parent.height, spec.width, spec.height
    )
    gen = store.create_generation(
        session_id=parent.session_id,
        kind="video",
        model_key=spec.key,
        # an empty motion prompt reuses the still's own prompt: better than
        # nothing, though describing the motion explicitly works far better.
        # No mechanical guard here any more (see ANIMATE_GUARD_SUGGESTION):
        # sent exactly as written, so a wish that needs a new object -- "she
        # sits down" -- is free to ask for one instead of fighting a
        # server-appended "no objects added".
        prompt=req.prompt.strip() or parent.prompt,
        negative_prompt="",
        seed=random_seed(),
        width=width,
        height=height,
        aspect=dims.nearest_aspect(width, height),
        steps=spec.steps,
        cfg=spec.cfg,
        sampler=None,
        parent_id=parent.id,
        num_frames=frames,
        fps=req.fps,
    )
    job_queue.put_nowait(gen.id)
    return gen


@app.post("/api/generations/{gen_id}/animate-prompt")
async def animate_prompt(gen_id: str, req: MotionPromptRequest) -> dict:
    """Rewrite a motion wish into a prompt an I2V model reads well.

    Returned for the user to edit, not queued: the clip costs tens of minutes,
    so the rewritten prompt is worth reading before spending that.
    """
    parent = _get_gen_or_404(gen_id)
    if parent.status != "done":
        raise HTTPException(400, "parent image not available yet")
    # the first frame is what actually gets animated, so show it to the LLM
    # when the selected model can read images; text-only models still work
    # from the still's own prompt
    image_b64 = None
    if await llm.has_vision() is not False:
        image_b64 = base64.b64encode(store.image_path(parent).read_bytes()).decode()
    # the real clip length, not a guess: a motion planned for 5s and rendered
    # in 2 is performed as a nudge and a hold (docs/decisions/VIDEO.md)
    seconds = video.snap_frame_count(req.num_frames) / max(1, req.fps)
    variant = await llm.motion(req.prompt, parent.prompt, image_b64, seconds=seconds)
    return {"prompt": variant.prompt, "used_image": image_b64 is not None}


@app.post("/api/generations/{gen_id}/edit-prompt")
async def edit_prompt(gen_id: str, req: EditPromptRequest) -> dict:
    """Rewrite a plain-words edit wish the way an instruction editor reads it.

    The Instruct dialog's optional step, and optional on purpose: the words
    typed there go to the edit model verbatim, which is the whole point of the
    action, so the rewrite is offered for reading and editing rather than
    applied on the way to the queue.
    """
    parent = _get_gen_or_404(gen_id)
    if parent.status != "done":
        raise HTTPException(400, "parent image not available yet")
    wish = req.prompt.strip()
    if not wish:
        raise HTTPException(400, "type what you want changed first")
    # the picture being edited is what the instruction has to point into, so
    # show it to the LLM when the selected model can read images; a text-only
    # one has only the prompt the image was generated from to name things by
    image_b64 = None
    if await llm.has_vision() is not False:
        image_b64 = base64.b64encode(store.image_path(parent).read_bytes()).decode()
    variant = await llm.edit_instruction(wish, parent.prompt, image_b64)
    return {"prompt": variant.prompt, "used_image": image_b64 is not None}


@app.get("/api/video/{gen_id}")
async def video_file(gen_id: str) -> FileResponse:
    gen = _get_gen_or_404(gen_id)
    path = store.video_path(gen)
    if not path.is_file():
        raise HTTPException(404, "clip not written yet")
    media = "video/mp4" if path.suffix == ".mp4" else "image/webp"
    return FileResponse(path, media_type=media)


@app.get("/api/aspects")
async def aspects() -> list[dict]:
    """The aspect ratios the composer offers, in button order.

    Served rather than hardcoded in the page so the list has one definition;
    the size each ratio produces is computed in the page from the same rule
    (dims.stage_dims), and tests/test_dims.py fails if the two drift apart.
    """
    return [
        {
            "key": key,
            "label": dims.ASPECT_LABELS[key],
            "w": w,
            "h": h,
            "ratio": w / h,
        }
        for key, (w, h) in dims.ASPECTS.items()
    ]


@app.get("/api/models")
async def models() -> list[dict]:
    # re-scan the model folders so a model imported into Draw Things after
    # startup appears on the next page refresh, without a server restart
    registry.resolve_installed()
    return [
        {
            "key": m.key,
            "label": m.label,
            "available": m.available,
            "installed": m.installed,
            # `enabled` is the tick in the settings dialog, `shown` that tick
            # and the hide-not-installed rule together. The page leaves a model
            # that is not `shown` out of the chips and the pickers, and keeps
            # every model in the list so the settings dialog can offer it back
            "enabled": m.enabled,
            "shown": m.shown,
            "edit_only": m.edit_only,
            # whether the aspect-ratio buttons apply to this chip. Models
            # without it stay selectable and simply come back square; the
            # composer says which ones those are before you generate.
            "supports_aspect": m.supports_aspect,
            # the stage sizes and the cap, so the composer can show what each
            # ratio button will actually produce for the selected models
            "draft_size": m.draft.size,
            "final_size": m.final.size,
            "aspect_size": m.aspect_size,
            # what a Faithful honours: None on the SDXL entries, whose cap is
            # a composition budget a Faithful does not need (ModelSpec.shaped_cap)
            "faithful_aspect_size": m.shaped_cap(img2img=True),
            # the budget a Faithful renders at: the final's, unless the model
            # names its own (ModelSpec.faithful_size)
            "faithful_size": m.faithful_budget,
            "uses_negative_prompt": m.uses_negative_prompt,
            "prompt_style": m.prompt_style,
            # the opener the UI puts in front of a hand-written prompt for this
            # model; empty for prose models
            "quality_prefix": m.quality_opener,
            "prompt_budget": m.prompt_budget.model_dump(),
            "supports_img2img": m.supports_img2img,
            # the pixel budget a full-image edit by this model runs at, in the
            # picture's own shape (edit_frame): the UI warns before resizing a
            # picture larger than this, instead of silently shrinking it
            "edit_size": m.final.size,
            # what a full-size edit costs and how far it can go: the UI offers
            # it as an opt-in with an estimate scaled off the measured 2048
            # upscale figure
            "tiled_diffusion": m.tiled_diffusion,
            # ...but not for instruction editors. They run at strength 1.0, so
            # every step is real work, and the image is also an edit reference
            # whose conditioning cost grows with its area: Qwen Edit at 2048
            # measured ~11 min for the FIRST of 8 steps, ~90 min for the pass,
            # against ~15 min for edit-at-1024-then-upscale. The endpoint still
            # honours full_size if asked directly; the UI does not offer a
            # trade that bad.
            "allow_full_edit": not m.edit_only,
            "upscale": m.upscale.model_dump(),
            "upscale_seconds": m.upscale_seconds,
            # the reason travels with the flag: the UI shows a disabled button
            # explaining which model is missing, rather than hiding the action
            "can_upscale": registry.upscale_block(m.key) is None,
            "upscale_block": registry.upscale_block(m.key),
            # sizes above 2048 need tiled_diffusion; the others would come
            # back blank, so they are not offered for those models
            "upscale_size_options": [
                z
                for z in registry.upscale_sizes
                if (z <= max(2048, m.upscale.size) or m.tiled_diffusion)
                and (m.max_upscale_size is None or z <= m.max_upscale_size)
            ],
            "upscale_default_size": m.upscale.size,
            # every model that could run the 2048 pass for this one, default
            # first, so the card can offer the fast/faithful trade-off
            "upscale_options": [
                {
                    "key": c,
                    "label": registry.models[c].label,
                    "seconds": registry.models[c].upscale_seconds,
                }
                for c in registry.upscale_candidates(m.key)
            ],
            "draft": m.draft.model_dump(),
            "final": m.final.model_dump(),
            "lora_options": [_lora_option_reply(o) for o in m.lora_options],
        }
        for m in registry.models.values()
    ]


def _lora_option_reply(opt) -> dict:
    return {
        "key": opt.key,
        "label": opt.label,
        "on": opt.on,
        "default": opt.default,
        "available": opt.available,
        "stages": sorted(opt.stages),
    }


def _steps_reply(spec) -> dict:
    """What a model's step settings are now, and what they ship as."""
    return {
        "key": spec.key,
        **{
            stage: {
                "steps": getattr(spec, stage).steps,
                "default": getattr(spec, stage).default_steps,
            }
            for stage in STAGES
        },
    }


@app.put("/api/models/{model_key}/steps")
async def set_model_steps(model_key: str, req: StepsRequest) -> dict:
    """Set this model's draft and final step counts for every future render.

    A stage sent as null goes back to the value in config/models.yaml, which
    is how Reset is expressed: the request always states both stages, so there
    is nothing to remember about what was set last.
    """
    spec = registry.models.get(model_key)
    if spec is None:
        raise HTTPException(404, f"unknown model key: {model_key}")
    try:
        settings_store.set_steps(model_key, req.as_dict())
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    registry.apply_step_overrides(settings_store.steps())
    log.info(
        "steps for %s: draft=%s final=%s", model_key, spec.draft.steps, spec.final.steps
    )
    return _steps_reply(spec)


@app.put("/api/models/{model_key}/loras/{option_key}")
async def set_lora_choice(model_key: str, option_key: str, req: LoraChoiceRequest) -> dict:
    """Tick or untick one of this model's optional LoRAs for every future render.

    What changes with it -- steps, cfg, sampler, whether the negative prompt
    is used -- comes back in the reply, because the page's cost estimates and
    the negative box read those. Cards already queued keep the LoRAs they were
    queued with.
    """
    spec = registry.models.get(model_key)
    if spec is None:
        raise HTTPException(404, f"unknown model key: {model_key}")
    if not any(o.key == option_key for o in spec.lora_options):
        raise HTTPException(404, f"{spec.label} has no LoRA option {option_key!r}")
    settings_store.set_lora_choice(model_key, option_key, req.on)
    registry.apply_lora_choices(settings_store.lora_choices())
    log.info("LoRA %s on %s: %s", option_key, model_key,
             next(o.on for o in spec.lora_options if o.key == option_key))
    return {
        **_steps_reply(spec),
        "uses_negative_prompt": spec.uses_negative_prompt,
        "draft_stage": spec.draft.model_dump(),
        "final_stage": spec.final.model_dump(),
        "lora_options": [_lora_option_reply(o) for o in spec.lora_options],
    }


@app.delete("/api/settings/steps")
async def reset_all_steps() -> list[dict]:
    """Every model back to the step counts it ships with."""
    settings_store.clear_steps()
    registry.apply_step_overrides({})
    return [_steps_reply(spec) for spec in registry.models.values()]


def _visibility_reply() -> dict:
    """Which models the UI should offer, for image and video models alike.

    The whole picture rather than the one model that changed: `shown` folds in
    the hide-not-installed rule, so ticking that box moves rows the caller did
    not name, and the page redraws its chips from this in one go.
    """
    return {
        "hide_uninstalled": registry.hide_uninstalled,
        "models": [
            {"key": spec.key, "enabled": spec.enabled, "shown": spec.shown}
            for spec in (*registry.models.values(), *registry.video_models.values())
        ],
    }


@app.get("/api/settings/models")
async def model_visibility() -> dict:
    """Which models are ticked, which are shown, and whether the rule is on.

    /api/models carries each model's own answer, but not the rule that
    produced it -- the dialog's checkbox needs that, and inferring it from the
    rows is wrong on an install where everything happens to be installed.
    """
    return _visibility_reply()


@app.put("/api/models/{model_key}/enabled")
async def set_model_enabled(model_key: str, req: ModelEnabledRequest) -> dict:
    """Show or hide one model in the composer chips and the pickers.

    Hiding is a tidying of the UI, not a restriction: the render endpoints do
    not consult it, so Upscale, Refine and Retry on an existing card still run
    the model that made it. Video models are addressed here too -- the keys
    share one namespace, and one hidden list is easier to reason about.
    """
    if model_key not in registry.models and model_key not in registry.video_models:
        raise HTTPException(404, f"unknown model key: {model_key}")
    settings_store.set_model_enabled(model_key, req.enabled)
    registry.apply_model_visibility(
        settings_store.disabled_models(), settings_store.hide_uninstalled()
    )
    log.info("model %s %s in the UI", model_key, "shown" if req.enabled else "hidden")
    return _visibility_reply()


@app.put("/api/settings/hide-uninstalled")
async def set_hide_uninstalled(req: HideUninstalledRequest) -> dict:
    """Leave every model that is not installed in Draw Things out of the UI.

    A standing rule, not a bulk untick: a model imported later reappears on
    its own, and one removed goes away without the box being hit again. It is
    kept apart from the per-model ticks so that turning it off restores
    exactly what the user had chosen by hand.
    """
    settings_store.set_hide_uninstalled(req.hide_uninstalled)
    registry.apply_model_visibility(
        settings_store.disabled_models(), settings_store.hide_uninstalled()
    )
    return _visibility_reply()


# -- enrichment ---------------------------------------------------------------


def _guidance_for(model_keys: list[str]) -> str:
    """Model-specific prompting rules, when the selected models agree on them."""
    texts = {
        registry.models[k].prompt_guidance.strip()
        for k in model_keys
        if k in registry.models and registry.models[k].prompt_guidance.strip()
    }
    return texts.pop() if len(texts) == 1 else ""


def _budget_for(model_keys: list[str]):
    """The prompt budget to write to, when the selected models agree on one.

    Where they do not, the tightest wins rather than nothing: a prompt written
    for the roomiest model in the set renders faintly on the others, and unlike
    guidance or an opener a budget has a safe direction to fall back on.
    """
    budgets = [
        registry.models[k].prompt_budget
        for k in model_keys if k in registry.models
    ]
    return min(budgets, key=lambda b: b.full) if budgets else None


def _opener_for(model_keys: list[str]) -> str:
    """The quality opener, when the selected models agree on one.

    Enforced on the way out rather than merely asked for in the guidance: a
    checkpoint that documents its opener as required means it, and the LLM has
    been seen running the last word of one straight into the first word of the
    description ("...score_8, realistic 3girls"), which leaves neither the
    opener nor the count tag intact. Models are grouped by opener before they
    are enriched together, so a mixed set has nothing to agree on and gets
    nothing -- the same rule _guidance_for uses.
    """
    openers = {
        registry.models[k].quality_opener
        for k in model_keys
        if k in registry.models and registry.models[k].quality_opener
    }
    return openers.pop() if len(openers) == 1 else ""


@app.get("/api/vocab/tags")
async def vocab_tags(limit: int = 12000) -> list[list]:
    """The vocabulary as [name, post_count], most-used first, for autocomplete.

    Capped because the whole snapshot is a megabyte and the composer only needs
    enough to rank what somebody is part-way through typing; the tail is tags
    the model barely saw anyway. Empty when no snapshot is installed, which
    turns autocomplete off rather than breaking the page.
    """
    v = vocab.get()
    ranked = sorted(v.counts.items(), key=lambda kv: -kv[1])[:limit]
    return [[name, count] for name, count in ranked]


@app.get("/api/vocab/suggestions")
async def vocab_suggestions() -> dict:
    """Words the vocabulary has no spelling of but does have a tag for.

    Handed to the page whole because it is a few dozen entries: the type-ahead
    can then offer "1boy" the moment somebody finishes typing "male", which a
    round trip per keystroke could not. Empty without a snapshot, which turns
    the offers off rather than breaking the page.
    """
    return vocab.get().suggestion_map()


@app.post("/api/vocab/check")
async def vocab_check(body: dict) -> dict:
    """What is wrong with a tag prompt, without changing a character of it.

    The composer calls this for prompts the USER wrote. It only reports; the
    page offers the fix and the user clicks it. Prompts the LLM wrote are
    repaired outright instead (llm._clean_tags), because those are not theirs.
    """
    prompt = str(body.get("prompt", ""))
    result = vocab.get().check(prompt)
    # tags the vocabulary cannot object to -- they are real Danbooru tags --
    # but which these checkpoints render as haze all the same
    garnish = [
        t.strip() for t in prompt.split(",")
        if t.strip() and t.strip().lower() in META_TAGS
    ]
    # The garnish list is the one thing here that is not a vocabulary question:
    # these ARE real Danbooru tags, and they are reported because these
    # checkpoints render them as haze (measured; see llm.META_TAGS). check()
    # no longer reports a word merely for being absent from the snapshot, so
    # there is nothing left for this to be double-counted against.
    result["garnish"] = garnish
    return result


@app.get("/api/quality-words")
async def quality_words() -> list[str]:
    """The garnish that is an aesthetic judgement, not a technique.

    Served apart from the rest of META_TAGS because the composer has something
    better to say about these than "remove it": the checkpoint's own quality
    opener already says them, so there is an answer to "then how do I ask for
    a beautiful picture?".
    """
    return sorted(vocab.QUALITY_FOLKLORE)


@app.get("/api/meta-tags")
async def meta_tags() -> list[str]:
    """Tags that flatten a tag-prompted model, for the composer to warn about.

    The same list llm.py strips from prompts the LLM writes. It cannot strip
    them from prompts the user wrote -- those are the user's -- so the composer
    points them out instead and offers to remove them. Served rather than
    duplicated in the page so there is one definition.
    """
    return sorted(META_TAGS)


@app.post("/api/describe")
async def describe(req: DescribeRequest):
    await _require_vision()
    b64 = req.image_b64
    if b64.startswith("data:"):
        b64 = b64.split(",", 1)[1]
    return await llm.describe(
        b64, req.style, req.uses_negative_prompt, req.hint,
        _guidance_for(req.model_keys), _opener_for(req.model_keys),
        _budget_for(req.model_keys),
    )


@app.post("/api/enrich")
async def enrich(req: EnrichRequest) -> EnrichResponse:
    variants = await llm.enrich(
        req.instruction, req.k, req.uses_negative_prompt, req.style,
        _guidance_for(req.model_keys), req.mode, _opener_for(req.model_keys),
        _budget_for(req.model_keys),
    )
    return EnrichResponse(variants=variants)


# -- sessions -----------------------------------------------------------------


@app.get("/api/sessions")
async def list_sessions() -> list:
    return store.list_sessions()


@app.post("/api/sessions")
async def create_session(req: CreateSessionRequest) -> dict:
    session = store.create_session(req.instruction)
    return {"id": session.id}


@app.put("/api/sessions/{session_id}/prompts")
async def save_prompts(session_id: str, req: SavePromptsRequest) -> dict:
    """Keep the composer for a session that has no images yet.

    Enrich costs an LLM round per dialect, and until now none of it was
    written down: a session was only created when the first draft was queued,
    so prompts written and not generated from died with the tab.
    """
    if not store.save_prompts(session_id, req.instruction, req.prompts):
        raise HTTPException(404, "session not found")
    return {"ok": True}


@app.patch("/api/sessions/{session_id}")
async def rename_session(session_id: str, req: RenameSessionRequest) -> dict:
    if not store.rename_session(session_id, req.name):
        raise HTTPException(404, "session not found")
    return {"id": session_id, "name": store.get_session(session_id).name}


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    session = store.get_session(session_id)
    if session is None:
        raise HTTPException(404)
    return session


# 2x fast rows store the parent's model_key as provenance, but no checkpoint
# rendered them (app/esrgan.py runs a separate binary). Stats attribute them
# here instead; static/index.html mirrors the same rule for the card badge and
# the model filter, so all three agree on who made what.
ESRGAN_MODEL_KEY = "real-esrgan"


def _label_models(stats: dict) -> dict:
    """Attach display names: the registry owns them, the store only has keys.

    A key with no spec is shown as the key rather than dropped -- a model
    removed from models.yaml still made those images, and hiding them would
    make the per-model rows disagree with the total.
    """
    for m in stats["models"]:
        if m["key"] == ESRGAN_MODEL_KEY:
            m["label"] = "Real-ESRGAN (2x fast)"
            continue
        spec = registry.models.get(m["key"]) or registry.video_models.get(m["key"])
        m["label"] = spec.label if spec else m["key"]
    return stats


@app.get("/api/sessions/{session_id}/stats")
async def session_stats(session_id: str) -> dict:
    """What this session has actually produced, by kind and by model."""
    if store.get_session(session_id) is None:
        raise HTTPException(404)
    return _label_models(store.session_stats(session_id))


@app.get("/api/timings")
async def timings() -> list[dict]:
    """How long each model has actually taken here, per step, by kind and size.

    Install-wide rather than per session: it is this machine's pace, and a new
    session should start with everything the last one learned. See
    Store.render_timings for what is measured and what is left out.
    """
    return store.render_timings()


def _queue_draft(
    session_id: str,
    variant: PromptVariant,
    model_key: str,
    seed: int,
    parent_id: Optional[str] = None,
    aspect: str = dims.DEFAULT_ASPECT,
    batch: Optional[str] = None,
) -> Generation:
    spec = registry.model_or_raise(model_key)
    width, height, used_aspect = draft_dims(spec, spec.draft, aspect)
    gen = store.create_generation(
        session_id=session_id,
        kind="draft",
        model_key=model_key,
        prompt=prompt_for(spec, variant.prompt),
        negative_prompt=negative_for(spec, variant.negative_prompt),
        seed=seed,
        width=width,
        height=height,
        aspect=used_aspect,
        steps=spec.draft.steps,
        cfg=spec.draft.cfg if spec.draft.cfg is not None else spec.cfg,
        sampler=spec.draft.sampler or spec.sampler,
        loras=stage_loras(spec.draft),
        parent_id=parent_id,
        variant_label=variant.label,
        batch=batch,
    )
    job_queue.put_nowait(gen.id)
    return gen


@app.post("/api/sessions/{session_id}/drafts")
async def create_drafts(session_id: str, req: DraftRequest) -> list[Generation]:
    if store.get_session(session_id) is None:
        raise HTTPException(404)
    # an omitted ratio inherits the parent's, so Amend on a portrait stays a
    # portrait without the dialog having to pass the shape back
    aspect = req.aspect
    if aspect is None:
        parent = store.get_generation(req.parent_id) if req.parent_id else None
        aspect = inherited_aspect(parent) if parent else dims.DEFAULT_ASPECT
    created = []
    try:
        for variant in req.variants:
            for model_key in req.model_keys:
                for _ in range(req.seeds_per_combo):
                    created.append(
                        _queue_draft(
                            session_id, variant, model_key, random_seed(),
                            req.parent_id, aspect, req.batch,
                        )
                    )
    except (KeyError, ValueError) as exc:
        raise HTTPException(400, str(exc))
    return created


# -- actions on a generation ---------------------------------------------------


def _get_gen_or_404(gen_id: str) -> Generation:
    gen = store.get_generation(gen_id)
    if gen is None:
        raise HTTPException(404, "generation not found")
    if gen.pruned_at:
        # kept for its lineage only: there is no image to act on
        raise HTTPException(404, "this draft was removed; only its settings are kept")
    return gen


def _publishable(gen_id: str) -> Generation:
    """The generation, if it is something that may go to Civitai.

    A draft never is: it is a low-resolution preview for choosing a direction,
    not a picture to show anyone. Refused by the publish endpoint and by the
    ones that only exist to fill its dialog, so no route leads half-way there.
    """
    gen = _get_gen_or_404(gen_id)
    if gen.kind == "draft":
        raise HTTPException(
            400, "Drafts are not published: finalize it and publish the final"
        )
    return gen


# What a job reads its parent's image for. A plain Finalize is txt2img and
# needs only the parent's prompt and seed, which a pruned draft still has.
NEEDS_PARENT_IMAGE = {"faithful", "upscale", "detail", "edit", "esrgan", "video"}


@app.post("/api/generations/{gen_id}/finalize")
async def finalize(gen_id: str, req: FinalizeRequest) -> Generation:
    parent = _get_gen_or_404(gen_id)
    # the recipe the draft was made under, not the chip's current tick: a
    # tick flipped between a draft and its final must not change the chain
    # (Registry.as_made)
    spec = registry.as_made(registry.model_or_raise(parent.model_key), parent.loras)
    kind = "faithful" if req.mode == "faithful" else "final"
    if kind == "faithful" and not spec.supports_img2img:
        raise HTTPException(
            400, f"{spec.label} does not support img2img (crashes Draw Things); use Finalize instead"
        )
    if req.mode in ("faithful", "describe") and parent.status != "done":
        raise HTTPException(400, "parent image not available yet")
    if req.mode == "describe":
        # fail here rather than deep inside the worker, minutes later
        await _require_vision()
    # the parent's ratio, not the model's flag: Faithful is img2img and would
    # squash a portrait draft into a square frame, and a Finalize that changed
    # shape would move the composition the draft was picked for
    aspect = inherited_aspect(parent)
    # A Faithful renders a shaped frame at the final's own budget, where a
    # Finalize keeps to aspect_size: the cap is the area the model composes an
    # off-square frame at from scratch, and a Faithful composes nothing -- it
    # is img2img at 0.5 over a draft that already holds the figure. Capping it
    # cost the SDXL models 57% of a 3:2 Faithful's pixels, and on a full-length
    # figure that was the face (docs/decisions/DRAFT-STEPS.md). qwen-image's
    # cap is a render ceiling and stays on both (ModelSpec.shaped_cap). The
    # budget itself is the final's unless the model names a Faithful one
    # (ModelSpec.faithful_size: rinflanime's final is pinned at 1024 by what
    # its txt2img does at 1536, which a Faithful does not do).
    width, height = dims.stage_dims(
        spec.faithful_budget if kind == "faithful" else spec.final.size,
        dims.ratio(aspect), spec.shaped_cap(img2img=kind == "faithful"),
    )
    gen = store.create_generation(
        session_id=parent.session_id,
        kind=kind,
        model_key=parent.model_key,
        prompt=parent.prompt,
        describe_first=req.mode == "describe",
        negative_prompt=negative_for(spec, parent.negative_prompt),
        seed=parent.seed,
        width=width,
        height=height,
        aspect=aspect,
        # A Faithful asks for more, and gets the same amount of work: an
        # img2img pass does `steps x strength`, so the model's measured final
        # number delivers half of itself unless it is doubled first (see
        # ModelSpec.faithful_steps). Finalize is txt2img and runs every step of
        # what it is given, so it takes the number as measured.
        #
        # req.steps still wins when a caller names one -- the UI no longer
        # offers a box, but the endpoint is the same one a script would use.
        steps=req.steps or (
            spec.faithful_steps if kind == "faithful" else spec.final.steps
        ),
        cfg=spec.final.cfg if spec.final.cfg is not None else spec.cfg,
        sampler=spec.final.sampler or spec.sampler,
        loras=stage_loras(spec.final),
        parent_id=parent.id,
    )
    job_queue.put_nowait(gen.id)
    return gen


@app.post("/api/generations/{gen_id}/more")
async def more_like_this(gen_id: str, req: MoreLikeThisRequest) -> list[Generation]:
    parent = _get_gen_or_404(gen_id)
    # "the same prompt" of an instruction edit is the prompt of the picture
    # it was edited from; its order would be drafted as a scene description
    drawn = drawn_by(parent)
    model_keys = req.model_keys or [_drawing_spec(parent, drawn).key]
    variant = PromptVariant(
        prompt=drawn.prompt, negative_prompt=drawn.negative_prompt
    )  # _queue_draft applies each target model's own negative floor
    created = []
    try:
        for model_key in model_keys:
            for _ in range(req.count):
                created.append(
                    _queue_draft(
                        parent.session_id, variant, model_key, random_seed(),
                        parent.id, inherited_aspect(parent),
                    )
                )
    except (KeyError, ValueError) as exc:
        raise HTTPException(400, str(exc))
    return created


@app.post("/api/generations/{gen_id}/refine")
async def refine(gen_id: str, req: RefineRequest):
    parent = _get_gen_or_404(gen_id)
    spec = registry.model_or_raise(req.model_key or parent.model_key)
    variants = await llm.refine(
        parent.prompt, req.tweak, req.k, spec.uses_negative_prompt,
        spec.prompt_style, spec.prompt_guidance, spec.quality_opener,
        spec.prompt_budget,
    )
    if req.preview:
        return EnrichResponse(variants=variants)
    created = [
        _queue_draft(
            parent.session_id, v, spec.key, random_seed(), parent.id,
            inherited_aspect(parent),
        )
        for v in variants
    ]
    return created


@app.post("/api/generations/{gen_id}/upscale")
async def upscale(gen_id: str, req: Optional[UpscaleRequest] = None) -> Generation:
    parent = _get_gen_or_404(gen_id)
    if parent.status != "done":
        raise HTTPException(400, "parent image not available yet")
    # An instruction edit is finished by the time it is upscaled: the pass
    # belongs to the checkpoint that drew the picture, with a prompt that
    # describes it, and nothing of the edit's order or its editor. See drawn_by
    # and docs/decisions/UPSCALING.md, decision 9.
    image_spec = registry.model_or_raise(parent.model_key)
    edited = image_spec.edit_only
    drawn = drawn_by(parent) if edited else parent
    via = req.via if req else None
    if edited and not via:
        via = _drawing_spec(parent, drawn).key
    if not edited:
        block = registry.upscale_block(parent.model_key)
        if block:
            # required, not optional: the pass is fed an upscaled init and a
            # plain resize is visibly softer, so refuse rather than quietly
            # degrade
            raise HTTPException(400, block)
    # The model that made the image upscales it by default. A different one
    # can be asked for, but it is validated against the candidate list rather
    # than trusted: an arbitrary key here could queue a job that renders a
    # flat colour field and reports success.
    #
    # Its own model finishes the picture under the tick state the picture
    # records (Registry.as_made); a stand-in has no such state to inherit.
    spec = registry.as_made(image_spec, parent.loras)
    if via:
        allowed = registry.upscale_candidates(parent.model_key)
        if via not in allowed:
            raise HTTPException(
                400,
                f"'{via}' cannot upscale a {image_spec.label} image; "
                f"choose one of: {', '.join(allowed) or 'none available'}",
            )
        spec = registry.model_or_raise(via)
    size = req.size if req and req.size else spec.upscale.size
    if size not in registry.upscale_sizes:
        raise HTTPException(
            400,
            f"{size} is not an offered upscale size; "
            f"choose one of: {', '.join(str(x) for x in registry.upscale_sizes)}",
        )
    # The only thing that ever made an image ineligible: a target that is not
    # bigger than the picture. Compared as budgets rather than long edges,
    # because a stage size is an area (see dims.budget). Kind is deliberately
    # not checked -- a face fix, a refine, a 2x fast and an earlier upscale are
    # all just pictures at a size by the time they get here, and the pass reads
    # the parent's PNG without caring which of them made it.
    current = dims.budget(parent.width, parent.height)
    if size <= current:
        raise HTTPException(
            400,
            f"this image is already {parent.width}x{parent.height}: a {size}px "
            "pass would render it smaller, not larger",
        )
    if spec.max_upscale_size is not None and size > spec.max_upscale_size:
        raise HTTPException(
            400,
            f"{spec.label} is capped at {spec.max_upscale_size}px: larger "
            "passes were measured returning no image at all",
        )
    if size > 2048 and not spec.tiled_diffusion and size > spec.upscale.size:
        # above 2048 only tiled models are safe: an untiled one would return a
        # flat colour field and report success, which is the failure this
        # whole path exists to avoid
        raise HTTPException(
            400,
            f"{spec.label} renders at most {spec.upscale.size}px: it does not "
            "use tiled_diffusion, so a larger pass would come back blank",
        )
    # queued with the prompt the picture was drawn from; the job folds the
    # edit orders into it before rendering (see edited_prompt)
    return _queue_upscale(parent, spec, size, source=drawn if edited else None)


def _queue_upscale(
    parent: Generation,
    spec,
    size: int,
    source: Optional[Generation] = None,
) -> Generation:
    """Queue the 2048 pass for `parent`, rendered by `spec`.

    Every stage value comes from the model doing the work, not the one that
    made the image: when the two differ the stand-in has its own step count,
    cfg and sampler, and borrowing the source's would push it outside what it
    was tuned for.

    The shape comes from the image rather than from `spec`: this is img2img fed
    an enlarged init, so a square frame would squash a portrait however the
    stand-in model was trained. spec.aspect_size is deliberately not applied
    either. It exists because a model asked to *compose* an off-square frame
    can repeat the subject across it; at strength 0.3 over an init that already
    holds the composition there is nothing left to compose, and honouring it
    here would cap a 3:4 upscale below the final it came from.
    """
    aspect = inherited_aspect(parent)
    width, height = dims.stage_dims(size, dims.ratio(aspect))
    # `source` is where the words come from when they are not the parent's
    # own: the picture an instruction edit was made from (see drawn_by)
    words = source or parent
    gen = store.create_generation(
        session_id=parent.session_id,
        kind="upscale",
        model_key=spec.key,
        prompt=words.prompt,
        negative_prompt=negative_for(spec, words.negative_prompt),
        seed=parent.seed,
        width=width,
        height=height,
        aspect=aspect,
        steps=spec.upscale.steps,
        cfg=spec.upscale.cfg if spec.upscale.cfg is not None else spec.cfg,
        sampler=spec.upscale.sampler or spec.sampler,
        loras=stage_loras(spec.upscale),
        parent_id=parent.id,
    )
    job_queue.put_nowait(gen.id)
    return gen


@app.post("/api/generations/{gen_id}/edit")
async def edit(gen_id: str, req: EditRequest) -> Generation:
    """img2img edit of this exact image toward a changed prompt."""
    parent = _get_gen_or_404(gen_id)
    if parent.status != "done":
        raise HTTPException(400, "parent image not available yet")
    spec = registry.model_or_raise(req.model_key or parent.model_key)
    if not spec.supports_img2img:
        raise HTTPException(
            400, f"{spec.label} does not support img2img (crashes Draw Things)"
        )
    if req.ref_image_b64 and not req.box:
        raise HTTPException(400, "select a target region when using a reference image")
    prompt = req.prompt
    if spec.edit_only:
        # instruct models sometimes drift nearby colours/objects; this
        # standing guard suppresses it without the user having to type it
        prompt = with_guard(prompt, EDIT_GUARD, "unchanged")
        if req.box:
            # and the rectangle is the frame, so what does not fit it is lost
            prompt = with_guard(
                prompt, REGION_FRAME_GUARD, "inside the frame", "not cropped"
            )
    # a full-image edit runs at the model's working budget, not the parent's
    # size (edit_frame); a rectangle is a crop of the parent and keeps it
    width, height = parent.width, parent.height
    if not req.box:
        width, height = edit_frame(
            spec, width, height, req.full_size, registry.upscale_sizes
        )
    gen = store.create_generation(
        session_id=parent.session_id,
        kind="edit",
        model_key=spec.key,
        prompt=prompt,
        negative_prompt=negative_for(spec, parent.negative_prompt),
        seed=random_seed(),
        width=width,
        height=height,
        aspect=inherited_aspect(parent),
        steps=req.steps or spec.final.steps,
        cfg=spec.final.cfg if spec.final.cfg is not None else spec.cfg,
        sampler=spec.final.sampler or spec.sampler,
        loras=stage_loras(spec.final),
        parent_id=parent.id,
        strength=req.strength,
        region=json.dumps(req.box) if req.box else None,
    )
    if req.ref_image_b64:
        # paste the reference crop into the target region now; the job then
        # repaints from those actual pixels instead of a text description
        b64 = req.ref_image_b64
        if b64.startswith("data:"):
            b64 = b64.split(",", 1)[1]
        try:
            init = build_collage_init(
                store.image_path(parent).read_bytes(),
                base64.b64decode(b64),
                tuple(req.box),
                tuple(req.ref_box) if req.ref_box else None,
            )
        except Exception as exc:
            store.delete_generation(gen.id)
            raise HTTPException(400, f"reference compositing failed: {exc}")
        store.init_path(gen).write_bytes(init)
    job_queue.put_nowait(gen.id)
    return gen


@app.post("/api/combine")
async def combine_pictures(req: CombineRequest) -> Generation:
    """Put the subject of one picture into another picture's scene.

    The result is an edit of the scene: parent_id is the scene, source_id the
    picture the subject came from. The two-panel frame the instruction editor
    works on is built now, from both images as they are, and kept as the job's
    init -- so the job does not depend on either picture still being there
    when it starts, and a held job resumes on exactly what was asked for.
    """
    if req.scene_id == req.subject_id:
        raise HTTPException(400, "pick two different pictures to combine")
    scene = _get_gen_or_404(req.scene_id)
    subject = _get_gen_or_404(req.subject_id)
    for g in (scene, subject):
        if g.kind == "video":
            raise HTTPException(400, "a clip cannot be combined; pick still images")
        if g.status != "done" or not store.image_path(g).is_file():
            raise HTTPException(400, f"picture {g.id[:6]} is not available yet")
    if scene.session_id != subject.session_id:
        raise HTTPException(400, "both pictures have to be in the same session")
    if req.model_key:
        spec = registry.model_or_raise(req.model_key)
    else:
        spec = next((m for m in registry.models.values()
                     if m.edit_only and m.available and m.shown), None)
        if spec is None:
            raise HTTPException(400, "Combine needs an instruction editor (Qwen Edit) installed in Draw Things")
    if not spec.edit_only:
        raise HTTPException(400, f"{spec.label} is not an instruction editor; Combine needs one")
    try:
        frame, scene_w, height = build_combine_frame(
            store.image_path(scene).read_bytes(), store.image_path(subject).read_bytes()
        )
    except Exception as exc:  # noqa: BLE001 - a bad image is the request's fault
        raise HTTPException(400, f"could not build the combine frame: {exc}")
    gen = store.create_generation(
        session_id=scene.session_id,
        kind="edit",
        model_key=spec.key,
        prompt=combine_prompt(req.prompt),
        negative_prompt=negative_for(spec, ""),
        seed=random_seed(),
        width=scene_w,
        height=height,
        aspect=inherited_aspect(scene),
        steps=req.steps or spec.final.steps,
        cfg=spec.final.cfg if spec.final.cfg is not None else spec.cfg,
        sampler=spec.final.sampler or spec.sampler,
        loras=stage_loras(spec.final),
        parent_id=scene.id,
        source_id=subject.id,
        strength=1.0,
    )
    store.init_path(gen).write_bytes(frame)
    job_queue.put_nowait(gen.id)
    return gen


@app.post("/api/generations/{gen_id}/reveal")
async def reveal(gen_id: str) -> dict:
    """Select this generation's file in Finder.

    The browser cannot open Finder, and the server runs on the same Mac, so
    the reveal happens here. The path comes from the store and the arguments
    are literals, so a crafted gen_id can only 404: there is no shell and
    nothing user-supplied on the command line.
    """
    gen = _get_gen_or_404(gen_id)
    if sys.platform != "darwin":
        raise HTTPException(400, "Reveal in Finder is macOS only")
    path = store.video_path(gen) if gen.kind == "video" else store.image_path(gen)
    if not path.is_file():
        raise HTTPException(404, "nothing on disk for this generation yet")
    proc = await asyncio.create_subprocess_exec("open", "-R", str(path))
    if await proc.wait() != 0:
        raise HTTPException(500, "Finder did not open the file")
    return {"revealed": path.name}


def _loras_credited(gen: Generation, source: Generation) -> list[LoraSpec]:
    """The LoRAs to credit on a post of `gen`: the composer's, then the pass's.

    Both from the rows, which recorded what actually ran. A LoRA recorded
    before its Civitai version was known -- or before the record carried one
    -- is given the id the config has for that file now; the file names the
    weights, so the id is the same fact, looked up late.
    """
    known: dict[str, int] = {}
    for spec in registry.models.values():
        stages = (spec.draft, spec.final, spec.upscale)
        for lora in (*spec.lora_options, *(l for s in stages for l in s.loras)):
            if lora.civitai_version_id:
                known.setdefault(lora.file, lora.civitai_version_id)
    rows = [source] if gen.id == source.id else [source, gen]
    out = []
    for row in rows:
        for lora in row.loras or []:
            if not lora.civitai_version_id and lora.file in known:
                lora = lora.model_copy(update={"civitai_version_id": known[lora.file]})
            out.append(lora)
    return out


def _also_credited(gen: Generation, source: Generation) -> list[Optional[int]]:
    """Civitai versions of the other models that worked on this picture.

    An upscale or a face pass through a different checkpoint than composed the
    picture: it is on the post too. It is also the only credit there is when
    the composer came from Draw Things' catalogue and has no Civitai page --
    which published a post claiming no resources at all.
    """
    if gen.model_key == source.model_key:
        return []
    spec = registry.models.get(gen.model_key)
    return [spec.civitai_version_id] if spec else []


def _publish_source(gen: Generation, ancestors: dict) -> Generation:
    """civitai.source_of, told which models are instruction editors."""
    editors = {k for k, m in registry.models.items() if m.edit_only}
    return civitai.source_of(gen, ancestors, editors)


@app.get("/api/generations/{gen_id}/publish-defaults")
async def publish_defaults(gen_id: str) -> dict:
    """What to put in the publish dialog before the user touches it.

    Everything here is a suggestion the dialog shows in editable fields, so it
    is allowed to be imperfect; what it must not do is fail and leave the
    dialog empty, which is why a missing session or model still returns a
    usable shape.
    """
    gen = _publishable(gen_id)
    detail = store.get_session(gen.session_id)
    ancestors = {g.id: g for g in (detail.generations if detail else [])}
    # the same row the metadata comes from: an upscale is titled and tagged
    # after the picture it enlarged, not after the enlarging
    source = _publish_source(gen, ancestors)
    spec = registry.models.get(source.model_key)
    version_id = spec.civitai_version_id if spec else None
    return {
        "title": civitai.suggest_title(
            detail.instruction if detail else "", source.prompt
        ),
        "detail": civitai.suggest_detail(gen, source, spec),
        "tags": civitai.suggest_tags(source.prompt, spec),
        # the two halves of the chunk that will be written into the uploaded
        # PNG: the dialog offers these for editing, so it shows the text itself
        # rather than a description of it -- and an upscale visibly offers its
        # parent's prompt, which is what composed the picture
        "prompt": source.prompt,
        "negative_prompt": source.negative_prompt,
        # the half that is not editable, shown verbatim beneath them. The size
        # is this image's, not the source's: everything else here describes the
        # render, but that field describes the file being posted
        "settings": civitai.settings_line(
            source,
            spec.dt_filename.removesuffix(".ckpt") if spec else source.model_key,
            version_id,
            (gen.width, gen.height),
            _also_credited(gen, source),
            _loras_credited(gen, source),
        ),
        # the whole chunk as it stands before any edit, which is what a client
        # that only wants to show the preview needs
        "parameters": civitai.parameters(
            source,
            spec.dt_filename.removesuffix(".ckpt") if spec else source.model_key,
            version_id,
            size=(gen.width, gen.height),
            also=_also_credited(gen, source),
            loras=_loras_credited(gen, source),
        ),
    }


@app.post("/api/generations/{gen_id}/caption-image")
async def caption_image(gen_id: str) -> dict:
    """A title, a description and tags for a simple post, read off the image.

    The other half of the publish dialog's answer to the same problem
    describe-image solves: the stored prompt describes what was asked for, and
    on a render that took a liberty that is not the picture anyone is looking
    at. describe-image replaces the prompt with a better one; this replaces the
    idea that a post needs a prompt at all -- a name for the piece, a sentence
    about it, and the words someone would browse for it under, all of which a
    reader can check against the image in front of them.

    Nothing about the request is passed in, on purpose and for the same reason
    as describe-image: the prompt is the thing being left out, and handing it to
    the model anchors the caption back to words that may describe a picture that
    never arrived. The tags are the model's own for the same reason -- the
    suggested tags elsewhere in the dialog are a reading of the prompt, so they
    inherit exactly the mismatch a simple post exists to avoid.

    Two things are added afterwards, both facts about the file rather than
    readings of the prompt. `anime` on a tag-style checkpoint, by the same rule
    `suggest_tags` uses: an anime checkpoint's output is anime whatever anyone
    wrote, and it is the tag people actually browse by -- measured on qwen3-vl
    and gemma3, neither volunteers it. And the tool, on the end of the
    description, because Civitai's Tools taxonomy cannot be set through any API
    we are allowed to call and the description is the only field that reaches
    the page (see civitai.with_provenance). The caption model is told not to
    name software, so this is the only place it comes from.

    Nothing is written down: the answer goes to the dialog, and only a publish
    puts it anywhere.
    """
    gen = _publishable(gen_id)
    if gen.kind == "video":
        raise HTTPException(400, "Captioning a clip is not supported")
    path = store.image_path(gen)
    if not path.is_file():
        raise HTTPException(404, "nothing on disk for this generation yet")
    await _require_vision()
    image_b64 = await asyncio.to_thread(
        lambda: image_for_vision(path.read_bytes())
    )
    caption = await llm.caption(image_b64)
    caption.description = civitai.with_provenance(caption.description)
    # the checkpoint that composed the picture, which for an upscale is the
    # parent's -- the same walk the rest of the dialog does
    detail = store.get_session(gen.session_id)
    ancestors = {g.id: g for g in (detail.generations if detail else [])}
    spec = registry.models.get(_publish_source(gen, ancestors).model_key)
    if spec is not None and spec.prompt_style == "tags":
        caption.tags = (
            ["anime"] + [t for t in caption.tags if t != "anime"]
        )[:5]
    return caption.model_dump()


@app.post("/api/generations/{gen_id}/describe-image")
async def describe_image(gen_id: str) -> dict:
    """A prompt read off this image, for the publish dialog to offer in place
    of the stored one.

    The stored prompt is what was *asked for*, and on a draft that took a
    liberty -- or an edit, or a face pass -- it can have very little to do with
    the picture that came out. A post is read by people who want the prompt to
    describe what they are looking at, so the dialog can ask the vision model
    to write one from the image itself.

    Deliberately not given the original prompt: the whole point is that the
    picture wins, and handing the model the text it is meant to be replacing
    anchors it straight back to the words that were already wrong. (Finalize+
    does pass it, because there the prompt is the intent being preserved -- see
    the describe step in run_generation.)

    Only the positive prompt comes back. What a render was told to avoid leaves
    no trace in the result, so a negative prompt read off the image would be
    invention, and the dialog leaves that box alone.

    The post tags come back with it, re-derived from the new text: they are a
    reading of the prompt, so a prompt that has just been replaced for not
    matching the picture leaves tags that do not match it either.

    Nothing is written down: the answer goes to the dialog, and only a publish
    puts it anywhere.
    """
    gen = _publishable(gen_id)
    if gen.kind == "video":
        raise HTTPException(400, "Reading a prompt from a clip is not supported")
    path = store.image_path(gen)
    if not path.is_file():
        raise HTTPException(404, "nothing on disk for this generation yet")
    await _require_vision()
    # the dialect belongs to the model that composed the picture, so an upscale
    # is described the way its parent's checkpoint is prompted
    detail = store.get_session(gen.session_id)
    ancestors = {g.id: g for g in (detail.generations if detail else [])}
    spec = registry.models.get(_publish_source(gen, ancestors).model_key)
    image_b64 = await asyncio.to_thread(
        lambda: image_for_vision(path.read_bytes())
    )
    variant = await llm.describe(
        image_b64,
        spec.prompt_style if spec else "prose",
        uses_negative_prompt=False,
        guidance=spec.prompt_guidance if spec else "",
        opener=spec.quality_opener if spec else "",
        budget=spec.prompt_budget if spec else None,
    )
    return {
        "prompt": variant.prompt,
        "tags": civitai.suggest_tags(variant.prompt, spec),
    }


@app.post("/api/generations/{gen_id}/publish")
async def publish(gen_id: str, req: PublishRequest = PublishRequest()) -> Generation:
    """Send this image to Civitai as a draft post, and record that it went.

    Two calls to Civitai (stage the bytes, create the post) and one write here,
    in that order: the row is only marked published once Civitai has answered
    with a post id, so the card cannot claim a post that does not exist. A
    failure anywhere leaves the row untouched and the button ready to try again.

    What is published is this image, but the metadata written into it belongs to
    the generation that composed it -- see civitai.source_of. The exception is
    `Size:`, which is the file's own: nothing is resized on the way out, and an
    upscale that reported its parent's dimensions was describing a picture
    other than the one attached. Nothing about the picture is changed: the
    parameters chunk is added to a copy in memory.

    The dialog may have rewritten the prompt and negative prompt on the way out
    -- a Danbooru tag list is rarely what someone wants a gallery to read. That
    only changes what is written into the copy being uploaded; the stored
    generation keeps the text that actually rendered it, so the card, a re-run
    and a later publish all still show the real thing.

    `simple` takes that to its end: no prompt at all, only the title, the
    description, the tags and the settings the render measured. It is enforced
    here rather than left to the client to send empty strings, so a post that
    was meant to carry no prompt cannot carry one because a box was populated
    behind the dialog's back.
    """
    gen = _publishable(gen_id)
    if gen.kind == "video":
        raise HTTPException(400, "Publishing is for images; clips are not supported")
    if gen.status != "done":
        raise HTTPException(400, "image is not finished yet")
    api_token = civitai.token()
    if api_token is None:
        raise HTTPException(
            400,
            "No Civitai API key. Put CIVITAI_TOKEN in .env (or civitai.key beside "
            "it) from civitai.com -> Account settings -> API Keys, then restart.",
        )
    path = store.image_path(gen)
    if not path.is_file():
        raise HTTPException(404, "nothing on disk for this generation yet")

    # the prompt, seed and settings that made the picture, which for an upscale
    # or a face pass are its parent's rather than its own
    detail = store.get_session(gen.session_id)
    ancestors = {g.id: g for g in (detail.generations if detail else [])}
    source = _publish_source(gen, ancestors)
    spec = registry.models.get(source.model_key)
    version_id = spec.civitai_version_id if spec else None
    also = _also_credited(gen, source)
    params = civitai.parameters(
        source,
        spec.dt_filename.removesuffix(".ckpt") if spec else source.model_key,
        version_id,
        prompt="" if req.simple else (
            None if req.prompt is None else req.prompt.strip()
        ),
        negative_prompt="" if req.simple else (
            None if req.negative_prompt is None else req.negative_prompt.strip()
        ),
        # the picture's own pixels, which for an upscale are not the source's
        size=(gen.width, gen.height),
        also=also,
        loras=_loras_credited(gen, source),
    )
    # the post links one version, and it is whichever of them Civitai knows:
    # the composer's when it has a page, else the model that ran the pass
    post_version_id = version_id or next((v for v in also if v), None)
    # PIL re-encode of what may be a 4K PNG: off the event loop, or the whole
    # server stalls for a second on every publish
    png = await asyncio.to_thread(civitai.embed_parameters, path.read_bytes(), params)

    client = civitai.CivitaiClient(api_token)
    try:
        uuid = await client.upload_image(png)
        post = await client.create_post(
            uuid=uuid,
            width=gen.width,
            height=gen.height,
            title=req.title.strip(),
            detail=req.detail.strip(),
            tags=[t.strip() for t in req.tags if t.strip()],
            model_version_id=post_version_id,
        )
    except civitai.CivitaiError as exc:
        raise HTTPException(400, str(exc)) from exc
    store.mark_published(gen.id, post["id"])
    log.info("published %s to %s (draft)", gen.id, post["url"])
    return store.get_generation(gen.id)


@app.delete("/api/generations/{gen_id}")
async def delete_generation(gen_id: str) -> dict:
    gen = _get_gen_or_404(gen_id)
    if gen.status == "running":
        raise HTTPException(400, "generation is in progress; wait for it to finish")
    store.delete_generation(gen.id)
    return {"deleted": gen.id}


@app.get("/api/storage")
async def storage() -> dict:
    """Where the images are kept, and what they are using."""
    return store.storage_summary()


@app.get("/api/cleanup")
async def cleanup_survey(session_id: Optional[str] = None) -> dict:
    """What the database and the disk disagree about, without changing either.

    Every file that would be deleted is named, because that is the only thing
    a user can check this against before pressing the second button.
    """
    found = store.survey_storage(session_id)
    return {
        "missing": [
            {**r, "files": [store.relative(p) for p in r["files"]]}
            for r in found["missing"]
        ],
        "files": [store.relative(p) for p in found["orphans"]],
        # set when the images on disk don't look like this database's images;
        # the check refuses to sweep rather than deleting every row
        "blocked": found["blocked"],
    }


@app.post("/api/cleanup")
async def cleanup_run(session_id: Optional[str] = None) -> dict:
    """Delete the finished cards with no image, and the files no card owns.

    Finished means done or error: both are cards that will never show a
    picture. Anything queued or running is left alone, so a render in flight
    cannot be swept up by this.
    """
    try:
        return store.cleanup_storage(session_id)
    except StorageMisconfigured as exc:
        raise HTTPException(400, str(exc)) from exc


# One session at a time, by design: there is no install-wide form of these.
DraftsSession = Annotated[str, Query(min_length=1)]


def _session_or_404(session_id: str) -> str:
    if store.get_session(session_id) is None:
        raise HTTPException(404, "session not found")
    return session_id


@app.get("/api/drafts/remove")
async def drafts_survey(session_id: DraftsSession) -> dict:
    """What Remove drafts would take from one session, without taking it.

    See Store.survey_drafts for why some drafts lose only their image and
    some are not touched at all.
    """
    found = store.survey_drafts(_session_or_404(session_id))
    return {**found, "files": [store.relative(p) for p in found["files"]]}


@app.post("/api/drafts/remove")
async def drafts_remove(session_id: DraftsSession) -> dict:
    """Remove one session's finished drafts, keeping the rows others descend from."""
    return store.prune_drafts(_session_or_404(session_id))


@app.post("/api/generations/{gen_id}/esrgan")
async def esrgan_upscale(gen_id: str, req: EsrganRequest) -> Generation:
    """Non-generative Real-ESRGAN 2x: runs outside Draw Things, in seconds."""
    parent = _get_gen_or_404(gen_id)
    if parent.status != "done":
        raise HTTPException(400, "parent image not available yet")
    if not esrgan.available():
        raise HTTPException(
            400, "Real-ESRGAN is not installed: run scripts/setup.sh to fetch it"
        )
    gen = store.create_generation(
        session_id=parent.session_id,
        kind="esrgan",
        model_key=parent.model_key,
        prompt=parent.prompt,
        negative_prompt=parent.negative_prompt,
        seed=parent.seed,
        width=parent.width * 2,
        height=parent.height * 2,
        aspect=inherited_aspect(parent),
        steps=0,
        cfg=0,
        sampler=f"Real-ESRGAN {esrgan.FLAVORS[req.flavor]}",
        parent_id=parent.id,
    )
    # inline, not queued: seconds of CPU/GPU, independent of Draw Things
    await run_esrgan_job(gen)
    return store.get_generation(gen.id)


@app.post("/api/generations/{gen_id}/retry")
async def retry(gen_id: str) -> Generation:
    gen = _get_gen_or_404(gen_id)
    if gen.status != "error":
        raise HTTPException(400, "only failed generations can be retried")
    parent = store.get_generation(gen.parent_id) if gen.parent_id else None
    if parent and parent.pruned_at and (
        gen.kind in NEEDS_PARENT_IMAGE or gen.describe_first
    ):
        raise HTTPException(
            400, "the draft this was made from has been removed, so there is "
            "no image left to render it from",
        )
    store.set_status(gen.id, "pending", None)
    job_queue.put_nowait(gen.id)
    return store.get_generation(gen.id)


@app.post("/api/generations/{gen_id}/detail")
async def detail(gen_id: str, req: DetailRequest = DetailRequest()) -> Generation:
    """ADetailer-style face refinement pass on a finished image."""
    parent = _get_gen_or_404(gen_id)
    if parent.status != "done":
        raise HTTPException(400, "parent image not available yet")
    # an instruction edit's faces are redrawn by the checkpoint that drew
    # them, from its prompt: the editor and its order have no part in it
    drawn = drawn_by(parent)
    spec = registry.as_made(_drawing_spec(parent, drawn), drawn.loras)
    if not spec.supports_img2img:
        raise HTTPException(
            400, f"{spec.label} does not support img2img (crashes Draw Things)"
        )
    box = None
    if req.box:
        # pointing at the face by hand: the detector keys on eyes, so a
        # blindfold, visor or mask hides one that is plainly there
        if len(req.box) != 4:
            raise HTTPException(400, "box must be [left, top, right, bottom]")
        l, t, r, b = (int(v) for v in req.box)
        if r - l < 32 or b - t < 32:
            raise HTTPException(400, "selected face region is too small")
        box = [l, t, r, b]
    if box is None:
        faces, skipped = eligible_faces(
            store.image_path(parent).read_bytes(), spec.prompt_style == "tags"
        )
    else:
        faces, skipped = [box], False
    if not faces:
        raise HTTPException(
            400,
            "face already fills most of the frame; nothing to gain from a detail pass"
            if skipped
            # Not a dead end, and worth saying why: lbpcascade_animeface keys
            # on eyes, so a blindfold, visor, mask or heavy fringe defeats it
            # even when the face is obvious to a person -- measured, covering
            # only the eye band of a detected face drops it from 1 to 0, and
            # no cascade setting recovers it (looser ones return specks
            # elsewhere in the frame instead). Refine with a box runs
            # repaint_region, which is the same pass this endpoint applies per
            # face, so it is a real equivalent rather than a consolation.
            else (
                "no face detected. The anime face detector keys on eyes, so a "
                "blindfold, visor, mask or heavy fringe defeats it even when "
                "the face is obvious. Drag a box around the face in this "
                "dialog and run it again: that skips detection and gives the "
                "region the identical pass."
            ),
        )
    gen = store.create_generation(
        session_id=parent.session_id,
        kind="detail",
        model_key=spec.key,
        prompt=drawn.prompt,
        negative_prompt=negative_for(spec, drawn.negative_prompt),
        seed=parent.seed,
        width=parent.width,
        height=parent.height,
        aspect=inherited_aspect(parent),
        steps=spec.final.steps,
        cfg=spec.final.cfg if spec.final.cfg is not None else spec.cfg,
        sampler=spec.final.sampler or spec.sampler,
        loras=stage_loras(spec.final),
        parent_id=parent.id,
        # recorded rather than left None so the params dialog shows what this
        # pass actually ran at, and Retry reproduces it
        strength=req.strength or DEFAULT_FACE_STRENGTH,
        region=json.dumps(box) if box else None,
    )
    job_queue.put_nowait(gen.id)
    return gen


# -- SSE ------------------------------------------------------------------------


@app.get("/api/events/{session_id}")
async def events(session_id: str) -> StreamingResponse:
    queue: asyncio.Queue = asyncio.Queue()
    subscribers.setdefault(session_id, set()).add(queue)

    async def stream():
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=20)
                    yield f"data: {json.dumps(event)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            subscribers.get(session_id, set()).discard(queue)

    return StreamingResponse(stream(), media_type="text/event-stream")
