"""Pydantic models shared across the app."""

from __future__ import annotations

import json
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, PrivateAttr, field_validator

from .dims import ASPECTS
from .settings import STAGES, STEP_MAX, STEP_MIN


class LoraSpec(BaseModel):
    file: str
    weight: float = 1.0
    # which Civitai version the LoRA is, for crediting it when a picture made
    # with it is published; see ModelSpec.civitai_version_id for why a hash
    # cannot stand in for it
    civitai_version_id: Optional[int] = None


class StageRecipe(BaseModel):
    """What an optional LoRA changes about one stage while it is ticked.

    A field left out keeps the stage's own value. Steps set here become the
    stage's default -- what the settings table shows greyed and what Reset
    returns to -- so a step override the user typed still wins over them.
    """

    # A few-step recipe can need a bigger draft: at 8 steps and CFG 1 the
    # Faithful over a small draft cannot put back detail the draft lacked.
    size: Optional[int] = None
    steps: Optional[int] = None
    cfg: Optional[float] = None
    sampler: Optional[str] = None


class LoraOption(BaseModel):
    """A LoRA a model can run with or without: one tick in the settings table.

    The fixed kind lives on the stage (`StageSpec.loras`) and is always sent;
    this kind is the user's call per install. Several can be offered on one
    model and several can be on at once -- each one's LoRA is added to the
    stages it names, and its recipe applied in the order they are listed.

    An accelerator is the reason this exists: it only works with its own
    steps and cfg, so the recipe travels with the tick rather than being a
    second thing to remember to change.
    """

    key: str
    label: str
    file: str
    weight: float = 1.0
    # ticked unless this install has said otherwise
    default: bool = True
    # The stages it applies to and what it changes there. A stage not named
    # here renders without it. Name the upscale too when the LoRA is what
    # the tick means by "this model": otherwise a picture made with it is
    # finished by a pass without it.
    stages: dict[Literal["draft", "final", "upscale"], StageRecipe]
    # False for a LoRA whose recipe runs at CFG 1, where a negative prompt
    # has no second pass to act in: the composer then hides the box and the
    # LLM is not asked to write one.
    negative_prompt: bool = True
    # Tags added to the model's opener while this is on, for a LoRA whose
    # page asks for them. They land where the opener does -- every prompt,
    # hand-written or enriched, gets them at queue time (main.prompt_for).
    opener: Optional[str] = None
    # recorded on every picture made with it, so a publish can credit it
    civitai_version_id: Optional[int] = None
    # set by Registry.resolve_installed(); None means no scan has run
    installed: Optional[bool] = None
    # set by Registry: ticked and installed, i.e. what renders
    on: bool = False

    @property
    def available(self) -> bool:
        return self.installed is not False


class StageSpec(BaseModel):
    size: int
    steps: int
    # What `steps` says in config/models.yaml, captured by load_registry()
    # before any per-install override from app.settings is laid over it.
    # `steps` is what renders; this is what Reset puts back, and what the UI
    # shows beside a changed value so a number that was measured is
    # distinguishable from one that was typed. None until a registry load
    # sets it.
    default_steps: Optional[int] = None
    strength: float = 0.3  # only used by img2img stages
    # optional per-stage overrides of the model-level settings
    sampler: Optional[str] = None
    cfg: Optional[float] = None
    loras: list[LoraSpec] = []


# What a tag model wants in front of an unenriched prompt. Every tag family has
# its own opener and they are not interchangeable ("very aesthetic" and
# "absurdres" are Illustrious vocabulary, "score_7" is Pony-derived), so this is
# only the default for models that do not name their own in models.yaml.
DEFAULT_TAG_PREFIX = "masterpiece, best quality, amazing quality, very aesthetic, absurdres"

# What a tag model gets as its negative prompt when nothing else supplied one.
# An SDXL anime checkpoint leans on its negative: sent an empty one it renders
# softer, muddier and with worse hands than the same prompt and seed with this
# baseline in place. The LLM writes a negative during Enrich (llm.NEG_TAGS asks
# for exactly this list plus scene-specific tags), but the paths that skip the
# LLM -- your prompt used as written, Generate with no enrichment -- had nothing
# to put there, so a third of the Illustrious generations on record went out
# with the field empty. This is that floor.
DEFAULT_TAG_NEGATIVE = (
    "bad quality, worst quality, worst detail, sketch, censor, lowres, "
    "bad anatomy, bad hands, jpeg artifacts, signature, watermark, username, blurry"
)


class PromptBudget(BaseModel):
    """How much prompt this model actually acts on, in characters.

    Measured, not guessed, and per model because it differs sharply between
    them (docs/decisions/PROMPT-LENGTH.md). Nothing is ever truncated: the
    failure is not a cliff but a dilution -- every extra word takes attention
    share from the rest, so a long prompt does not lose its tail so much as
    render everything in it faintly. So these are advisory: they tell the LLM
    what to write to, and the composer what to warn about.

    full    every word still lands. A prompt this long or shorter renders what
            it asks for.
    fading  past this the tail is unlikely to reach the image at all. Between
            the two, detail late in the prompt arrives progressively weaker.
    """

    full: int
    fading: int


class ModelSpec(BaseModel):
    key: str
    label: str
    dt_filename: str
    sampler: Optional[str] = None
    cfg: float = 1.0
    uses_negative_prompt: bool = False
    prompt_style: Literal["prose", "tags"] = "prose"
    prompt_guidance: str = ""
    # tags-style openers only; None resolves to DEFAULT_TAG_PREFIX for tag
    # models and to nothing for prose models (see quality_opener)
    quality_prefix: Optional[str] = None
    # The negative prompt to fall back on when a generation is queued without
    # one. None resolves to DEFAULT_TAG_NEGATIVE for tag models and to nothing
    # for prose models; set it per model where the checkpoint documents its own
    # (score tags are Pony-derived vocabulary and mean nothing to Illustrious).
    # Only consulted when uses_negative_prompt is true.
    negative_prefix: Optional[str] = None
    clip_skip: int = 1
    # How much prompt this model acts on. Defaults are the conservative end of
    # what was measured, for a model nobody has probed yet.
    prompt_budget: PromptBudget = PromptBudget(full=240, fading=600)
    supports_img2img: bool = True
    # Tile the denoising loop when rendering at 2048 or above. Required by
    # models whose working resolution stops short of it -- without it they
    # return a flat colour field and report success. Off for models that
    # reach 2048 unaided, which gain nothing from it.
    tiled_diffusion: bool = False
    # measured seconds for one 2048px upscale pass. Indicative only (M4 Pro);
    # it lets the Upscale picker show what each choice costs.
    upscale_seconds: Optional[int] = None
    # Largest size the Upscale picker may offer, when tiled_diffusion alone
    # would offer more. tiled_diffusion means "this model needs tiling to
    # reach 2048", and that was being read as "so it can reach 4096 too" --
    # qwen-image needs the flag at 2048 and still returns an empty response at
    # 4096, so the picker was offering an eight-minute render that produces
    # nothing. None means the flag decides, as before.
    max_upscale_size: Optional[int] = None
    # instruction-editing models (Qwen Image Edit): no generation chip; used
    # by the Instruct action, where the user's words go through verbatim
    edit_only: bool = False
    # The pixel budget to render this model at when a non-square shape is
    # asked for, as a square-equivalent side, or None when nothing has been
    # measured. One number does both jobs -- it is the opt-in for the ratio
    # buttons and the budget dims.stage_dims derives the sides from -- because
    # they are the same fact: what a model composes correctly at is an area,
    # and a stage size is only evidence about the area it was measured at.
    # Below final.size on the SDXL entries (1024 against 1536) and qwen-image
    # (864 against 1024), so a shaped Finalize on those is smaller than the
    # square; the field is an int rather than a flag because absent means
    # "never measured, render square". A model with None still appears on
    # every chip; it just renders square whatever is picked.
    aspect_size: Optional[int] = None
    # What the cap above is a cap ON. False (every SDXL entry): a composition
    # budget -- the area the model was seen to compose one figure at from
    # scratch -- which a Faithful does not need, because img2img at 0.5 takes
    # its composition from the draft and only paints detail; there it costs a
    # 3:2 Faithful 57% of its pixels and the face with them
    # (docs/decisions/DRAFT-STEPS.md, "Shaped Faithfuls"). True (qwen-image): a
    # render ceiling -- frames above it come back empty whatever stage asks --
    # so every stage honours it. See shaped_cap.
    aspect_size_is_ceiling: bool = False
    # The budget a Faithful renders at when it is not the final's. Absent on
    # every model but one: a Faithful is img2img at 0.5 over a draft and
    # normally takes `final.size`. rinflanime's final is held at 1024 because
    # its txt2img at 1536 draws the figure twice, but a Faithful composes
    # nothing -- four of four 1536 Faithfuls on it were one figure, with the
    # face formed where the 1024 one's was smeared (DRAFT-STEPS.md, "A Faithful
    # size of its own"). A number rather than a flag so a model can name a
    # size that is neither its final nor a ceiling. See faithful_budget.
    faithful_size: Optional[int] = None
    # Which Civitai model version these weights are, used only when publishing:
    # it is what attributes the checkpoint on the post. Not derivable from the
    # file -- Draw Things renders from a converted .ckpt that does not hash to
    # the safetensors on the model page -- so it is transcribed from the model
    # page URL, and stays None for anything that did not come from Civitai.
    civitai_version_id: Optional[int] = None
    draft: StageSpec
    final: StageSpec
    upscale: StageSpec
    # LoRAs this install can tick on or off; see LoraOption
    lora_options: list[LoraOption] = []

    # What models.yaml says, before any LoRA option or step override is laid
    # over it: set by load_registry(), read by Registry._recompute_stages(),
    # which rebuilds `draft` and `final` from these every time rather than
    # patching whatever the last call left.
    _shipped: dict[str, StageSpec] = PrivateAttr(default_factory=dict)
    _shipped_negative: bool = PrivateAttr(default=False)
    _shipped_prefix: Optional[str] = PrivateAttr(default=None)

    # set by Registry.resolve_installed(); None means no scan has run
    installed: Optional[bool] = None

    # Both set by Registry.apply_model_visibility(). `enabled` is the tick in
    # the settings dialog; `shown` is that and the hide-not-installed rule
    # together, i.e. the one answer the UI asks for. Neither gates rendering:
    # a hidden model still renders when a card asks for it by key.
    enabled: bool = True
    shown: bool = True

    @property
    def available(self) -> bool:
        return bool(self.dt_filename) and self.installed is not False

    @property
    def supports_aspect(self) -> bool:
        """Whether the aspect-ratio buttons apply to this model."""
        return self.aspect_size is not None

    def shaped_cap(self, img2img: bool) -> Optional[int]:
        """The `aspect_size` budget a final stage has to honour.

        A txt2img final always does: the cap is what the model composes an
        off-square frame correctly at. An img2img final (Faithful) is exempt
        unless the cap is a render ceiling, since the frame it is given already
        holds the composition. Upscale never honours it, for the same reason
        (see _queue_upscale in app/main.py).
        """
        if img2img and not self.aspect_size_is_ceiling:
            return None
        return self.aspect_size

    @property
    def faithful_budget(self) -> int:
        """The stage size a Faithful renders at: `faithful_size`, else the final's."""
        return self.faithful_size or self.final.size

    @property
    def faithful_steps(self) -> int:
        """Steps to ask for so a Faithful actually runs `final.steps` of them.

        An img2img pass skips the early part of the schedule in proportion to
        strength, so it does `steps x strength` real work: at 0.5 this model's
        own measured final number delivers half of itself. Finalize (txt2img)
        does not, so one figure in models.yaml cannot mean the same thing to
        both, and this is the one the img2img path needs to reach parity.

        Derived rather than configured, and it is not a claim that more steps
        are better: DRAFT-STEPS.md records measurements that say the gain at
        this stage is small to invisible, and every draft value in that
        document was judged on a Faithful rendered the old way. What it is, is
        one figure in models.yaml meaning the same amount of work whichever
        button is pressed. Anything above parity is unmeasured, and would need
        the same two-scene two-seed sweep as everything else.

        Capped at STEP_MAX, which one model (cyberrealistic-anima, 44) exceeds;
        it gets 64 and lands short of its own final rather than being allowed
        past a limit nothing here has tested.

        Used by `finalize()` for every Faithful. There is no per-card step box
        and no opt-out: the UI stopped offering per-image step tweaks, and the
        gear dialog is where a model's numbers are changed.
        """
        strength = self.final.strength
        if strength <= 0 or strength >= 1:
            return self.final.steps
        return min(STEP_MAX, round(self.final.steps / strength))

    @property
    def quality_opener(self) -> str:
        """The tag opener to put in front of a prompt written by hand."""
        if self.quality_prefix is not None:
            return self.quality_prefix
        return DEFAULT_TAG_PREFIX if self.prompt_style == "tags" else ""

    @property
    def shipped_opener(self) -> str:
        """The opener models.yaml gives, without any ticked LoRA's tags.

        What the model's prompt_guidance names, so what the LLM writes; the
        tags a LoRA adds are put on at queue time (main.prompt_for).
        """
        if self._shipped_prefix is not None or self._shipped:
            prefix = self._shipped_prefix
        else:
            prefix = self.quality_prefix
        if prefix is not None:
            return prefix
        return DEFAULT_TAG_PREFIX if self.prompt_style == "tags" else ""

    @property
    def default_negative(self) -> str:
        """The negative prompt for a generation that was given none."""
        if not self.uses_negative_prompt:
            return ""
        if self.negative_prefix is not None:
            return self.negative_prefix
        return DEFAULT_TAG_NEGATIVE if self.prompt_style == "tags" else ""


class VideoModelSpec(BaseModel):
    """An image-to-video model for the Animate action."""

    key: str
    label: str
    dt_filename: str
    steps: int = 8
    cfg: float = 1.0
    width: int = 832
    height: int = 480
    # Draw Things' own defaults for these; changing them was not an
    # improvement in testing, and raising start_frame_guidance actively hurt.
    motion_scale: int = 127
    start_frame_guidance: float = 1.0
    guiding_frame_noise: float = 0.02
    # Sent explicitly, unlike the above, because it must be: omitting it let
    # Draw Things fall back to whatever its own slider held, so two identical
    # requests could render differently after somebody touched the app. Measured
    # not to matter much for motion (5.0 vs the app's 3.158 moved background
    # flow 1.302 -> 1.392 px/frame, divergence unchanged), so the value is the
    # Wan reference one and the point of setting it is reproducibility.
    # See docs/decisions/VIDEO.md, Finding 5.
    shift: float = 5.0
    # measured seconds per output frame, for the UI's wait estimate
    sec_per_frame: float = 20.0

    installed: Optional[bool] = None

    # as on ModelSpec: the settings tick, and the tick plus the
    # hide-not-installed rule
    enabled: bool = True
    shown: bool = True

    @property
    def available(self) -> bool:
        return bool(self.dt_filename) and self.installed is not False

    def estimate_seconds(self, num_frames: int) -> int:
        """Rough wall-clock estimate. Cost grows faster than frame count.

        Measured 17.1 s/frame at 33 frames and 22.8 at 81, i.e. about
        frames^1.32. sec_per_frame is calibrated at 81, so scale from there
        rather than treating cost as linear.
        """
        return int(self.sec_per_frame * num_frames * (num_frames / 81) ** 0.32)


class AnimateRequest(BaseModel):
    model_key: str
    # motion prompt: what should move. Empty falls back to the parent's prompt.
    prompt: str = ""
    # Wan expects 4n+1 frames; validated and snapped server-side
    num_frames: int = Field(default=33, ge=5, le=81)
    fps: int = Field(default=16, ge=8, le=24)


class PromptVariant(BaseModel):
    prompt: str
    negative_prompt: str = ""
    # What the composer called this box -- "orig", "V1", "T2". Carried through
    # so a card can be badged with the prompt it actually came from instead of
    # the badge being reconstructed by string-matching prompts after the fact,
    # which lost track the moment a box was edited. None from anything that is
    # not the composer (the LLM's own output, More like this).
    label: Optional[str] = None


class ArtworkCaption(BaseModel):
    """What the vision model writes for a simple post.

    Not a prompt and deliberately not shaped like one: a name for the picture,
    a sentence or two about it, and the words someone would have browsed for it
    under. See llm.ARTWORK_SYSTEM for why a post can want these instead.
    """

    title: str = ""
    description: str = ""
    tags: list[str] = Field(default_factory=list)


class EnrichRequest(BaseModel):
    instruction: str
    k: int = Field(default=4, ge=1, le=8)
    uses_negative_prompt: bool = False
    style: Literal["prose", "tags"] = "prose"
    model_keys: list[str] = []
    # "explore" gives every variant its own composition; "hold" keeps the
    # composition the instruction already describes and varies only detail,
    # which is what a prompt read off a seed image wants.
    mode: Literal["explore", "hold"] = "explore"


class EnrichResponse(BaseModel):
    variants: list[PromptVariant]


class DescribeRequest(BaseModel):
    image_b64: str
    style: Literal["prose", "tags"] = "prose"
    uses_negative_prompt: bool = False
    hint: str = ""
    model_keys: list[str] = []


class CreateSessionRequest(BaseModel):
    instruction: str


class RenameSessionRequest(BaseModel):
    # blank clears the name, and the picker goes back to showing the prompt
    name: Optional[str] = Field(default=None, max_length=200)


class UpscaleRequest(BaseModel):
    # which model renders the pass; None means the model that made the image,
    # which is the default and keeps its look
    via: Optional[str] = None
    # target long edge; None uses the model's own upscale stage size. Must be
    # one of registry.upscale_sizes -- anything above 2048 relies on
    # tiled_diffusion and is only offered for models that have it.
    size: Optional[int] = None


class DraftRequest(BaseModel):
    variants: list[PromptVariant]
    model_keys: list[str]
    seeds_per_combo: int = Field(default=1, ge=1, le=8)
    parent_id: Optional[str] = None
    # one of dims.ASPECTS. None is not the same as "1:1": it means "inherit",
    # so a caller that queues drafts from an existing image (Amend) keeps that
    # image's shape without having to know what it was. Models without a
    # measured aspect_size ignore it and render square either way, which is
    # recorded on the row rather than assumed.
    aspect: Optional[Literal[tuple(ASPECTS)]] = None  # type: ignore[valid-type]
    # One press of Generate, which the page sends the same for every request
    # the press fans out into. Only the page can know this: one press becomes
    # one request per dialect and per quality opener, and they do not reliably
    # land in the same second. Reopening the session groups the rows back into
    # the rounds they were made in by this.
    batch: Optional[str] = None


class StepsRequest(BaseModel):
    """A model's global step counts, as set from the UI.

    Every stage is stated on every call and None means "back to the shipped
    default", so changing a value and resetting it are the same request with
    a different body -- there is no second code path to keep in step.
    """

    draft: Optional[int] = Field(default=None, ge=STEP_MIN, le=STEP_MAX)
    final: Optional[int] = Field(default=None, ge=STEP_MIN, le=STEP_MAX)

    def as_dict(self) -> dict[str, Optional[int]]:
        return {stage: getattr(self, stage) for stage in STAGES}


class LoraChoiceRequest(BaseModel):
    """Tick or untick one of a model's optional LoRAs; None is its default."""

    on: Optional[bool] = None


class ModelEnabledRequest(BaseModel):
    """Whether a model appears in the composer chips and the pickers."""

    enabled: bool


class HideUninstalledRequest(BaseModel):
    """Whether models missing from Draw Things are left out of the UI."""

    hide_uninstalled: bool


class FinalizeRequest(BaseModel):
    mode: Literal["regen", "faithful", "describe"] = "regen"
    # Steps for this one final only, leaving the model's own setting alone:
    # what to reach for when a particular draft is worth more time than the
    # rest of the batch. None uses the model's final steps as configured.
    steps: Optional[int] = Field(default=None, ge=STEP_MIN, le=STEP_MAX)


class MoreLikeThisRequest(BaseModel):
    count: int = Field(default=4, ge=1, le=8)
    model_keys: Optional[list[str]] = None  # default: same model as parent


class EditRequest(BaseModel):
    prompt: str
    # 1.0 is the instruct-edit setting: the init image acts as the edit
    # reference rather than a noised starting point
    strength: float = Field(default=0.5, ge=0.2, le=1.0)
    box: Optional[list[int]] = None  # [l, t, r, b] in image pixels: repaint only this region
    # reference image whose pixels are pasted into `box` before repainting
    ref_image_b64: Optional[str] = None
    ref_box: Optional[list[int]] = None  # crop of the reference to use
    model_key: Optional[str] = None  # repaint with a different model than the parent's
    # override the model's edit step count: instruct models wash out flat/
    # repainted areas at their default 8, while colour swaps are fine there
    steps: Optional[int] = Field(default=None, ge=2, le=64)
    # repaint at the picture's own resolution instead of the model's working
    # size, so an edit of an upscaled image keeps its resolution. Minutes
    # rather than seconds, and capped at what the model can render (2048, or
    # registry.upscale_sizes for a tiled_diffusion model)
    full_size: bool = False


class CombineRequest(BaseModel):
    """Put what one picture shows into another picture's scene."""

    # the picture that comes back, with the subject in it; the new image's parent
    scene_id: str
    # the picture the subject is taken from, whole
    subject_id: str
    # how to combine them, in the user's words: which subject, where, in what pose
    prompt: str = Field(min_length=1)
    model_key: Optional[str] = None  # an instruction editor; the installed one by default
    # None uses the editor's own final steps (see qwen-edit in models.yaml)
    steps: Optional[int] = Field(default=None, ge=2, le=64)


class MotionPromptRequest(BaseModel):
    # the user's motion wish as typed; empty asks the LLM to propose the
    # ambient motion the still naturally supports
    prompt: str = ""
    # The clip this prompt is for. Without it the LLM was told a static "1 to 5
    # seconds" and planned motion for up to 2.4x the clip it actually got, which
    # a model performs by starting the move and then holding.
    num_frames: int = Field(default=33, ge=5, le=81)
    fps: int = Field(default=16, ge=8, le=24)


class EditPromptRequest(BaseModel):
    # the user's edit wish as typed. Required, unlike MotionPromptRequest's:
    # an empty motion box means "propose the ambient motion this still
    # supports", but there is no equivalent guess for an edit -- an LLM
    # inventing a change to a picture the user is happy with is not a feature.
    prompt: str = Field(min_length=1)


class DetailRequest(BaseModel):
    # how much of each face crop the pass may rewrite; None keeps the server
    # default (detailer.DEFAULT_FACE_STRENGTH). Low values preserve the
    # expression, high ones let the model re-imagine the face
    strength: Optional[float] = Field(default=None, ge=0.1, le=1.0)
    # [l, t, r, b] in image pixels: treat this region as the face and skip
    # detection. lbpcascade_animeface keys on eyes, so a blindfold, visor, mask
    # or heavy fringe hides a face that is obvious to a person; pointing at it
    # by hand is the whole fix. Padded and squared exactly as a detected box
    # is, so the result matches the automatic pass.
    box: Optional[list[int]] = None


class EsrganRequest(BaseModel):
    # which Real-ESRGAN weights: photo (x4plus) or anime (x4plus-anime)
    flavor: Literal["photo", "anime"] = "photo"


class SavePromptsRequest(BaseModel):
    instruction: str = ""
    # opaque composer state: the client owns its shape, the server only keeps it
    prompts: dict = Field(default_factory=dict)


class RefineRequest(BaseModel):
    tweak: str
    k: int = Field(default=3, ge=1, le=6)
    preview: bool = False  # return the merged prompts instead of queueing drafts
    model_key: Optional[str] = None  # write for this model's dialect instead of the parent's


class PublishRequest(BaseModel):
    """What the publish dialog collects. Everything here is optional.

    There is no `publish` flag: a post is always created as a draft, because
    publishing it is one click on the Civitai page the user is sent to anyway
    (see docs/decisions/PUBLISHING.md).

    `prompt` and `negative_prompt` are the dialog's edits to the text written
    into the uploaded PNG; they change what the post says, never the stored
    generation. None means "leave the row's own text alone", which is what an
    older client or a bare POST sends -- an empty string is a deliberate
    "publish it with none", so the two cannot be collapsed. The settings that
    ride with them (seed, steps, sampler, size) are not editable: they are
    measurements of a render that already happened.

    `simple` is that second choice made once, for the whole post rather than a
    box at a time: no prompt and no negative prompt, only the title, the
    description, the tags and the technical settings. It overrides whatever
    `prompt` and `negative_prompt` hold, so a client that sends both cannot end
    up publishing a prompt it meant to leave out. Everything else stays
    optional -- a simple post with all three boxes empty is a picture and its
    settings, which is a legitimate thing to post.
    """

    title: str = ""
    detail: str = ""
    tags: list[str] = Field(default_factory=list, max_length=5)
    prompt: Optional[str] = None
    negative_prompt: Optional[str] = None
    simple: bool = False


class Generation(BaseModel):
    id: str
    session_id: str
    parent_id: Optional[str]
    # a Combine's subject picture; parent_id is the scene it was put into.
    # None for everything else.
    source_id: Optional[str] = None
    kind: Literal[
        "draft", "final", "faithful", "upscale", "detail", "edit", "esrgan", "video"
    ]
    model_key: str
    prompt: str
    negative_prompt: str
    seed: int
    width: int
    height: int
    # the ratio this image was shaped to, carried down the chain so a final or
    # an upscale reproduces the draft's shape exactly instead of re-deriving it
    # from pixels that have already been rounded to the grid. None on rows made
    # before the feature; dims.nearest_aspect covers those.
    aspect: Optional[str] = None
    steps: int
    cfg: float
    sampler: Optional[str]
    # Every LoRA this was rendered with, fixed or ticked, as sent to Draw
    # Things. Recorded when the job is queued, so a Retry repeats the recipe
    # the card was made with even after the tick has changed. None on rows
    # made before it was recorded: those render with the stage's LoRAs as
    # configured at the time, which is what they always did.
    loras: Optional[list[LoraSpec]] = None
    status: Literal["pending", "running", "done", "error"]
    error: Optional[str] = None
    strength: Optional[float] = None
    describe_first: bool = False
    region: Optional[str] = None
    duration_s: Optional[float] = None
    # the composer's name for the prompt behind this image, and the press of
    # Generate it belongs to; see PromptVariant.label and DraftRequest.batch.
    # None on rows made before either existed, and on everything that does not
    # come from the composer.
    variant_label: Optional[str] = None
    batch: Optional[str] = None
    # video generations only
    num_frames: Optional[int] = None
    fps: Optional[int] = None
    # container actually written ("mp4" or "webp"); the fallback path differs
    video_ext: Optional[str] = None
    # Set once this image has been sent to Civitai. `published_at` is the
    # moment the post was created here, not the moment it went public: a draft
    # post has a timestamp and no audience until it is published on Civitai.
    # Both stay None for everything never sent.
    published_at: Optional[str] = None
    published_post_id: Optional[int] = None
    # Set on a draft whose image Remove drafts deleted while keeping the row,
    # because something made from it still names it as parent. Such a row has
    # no card; it is there so the lineage can still say what it came from.
    pruned_at: Optional[str] = None
    created_at: str

    @field_validator("loras", mode="before")
    @classmethod
    def _loras_from_db(cls, v: Any) -> Any:
        # stored as JSON text in SQLite
        return json.loads(v) if isinstance(v, str) else v


class SessionSummary(BaseModel):
    id: str
    instruction: str
    # set by Rename; None means the session is known by its instruction
    name: Optional[str] = None
    created_at: str


class SessionDetail(SessionSummary):
    # the composer state saved by the last Enrich or Generate, if any: what
    # restores a session that has prompts but no images yet
    prompts: Optional[dict] = None
    generations: list[Generation]
