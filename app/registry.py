"""Loads and validates config/models.yaml."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel

from .schemas import LoraSpec, ModelSpec, StageSpec, VideoModelSpec
from .settings import STAGES

# The stages a LoRA option can name and the rebuild covers. Wider than STAGES,
# which is what the settings table lets the user override: the upscale's step
# count is half of a tuned pair and stays the config's, but a tick that says
# "this model runs with the LoRA" has to reach the upscale too, or the
# finishing pass renders a different model than the picture was made with.
LORA_STAGES = ("draft", "final", "upscale")

log = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "models.yaml"

DEFAULT_MODELS_DIRS = [
    "~/Library/Containers/com.liuliu.draw-things/Data/Documents/Models",
    "~/Pictures/models",
]

# Draw Things appends a quantisation/precision suffix to the file it writes;
# which one depends on the variant the user picked at download time.
_QUANT_SUFFIX = re.compile(r"_(f16|f32|q8p|q6p|q5p|q4p|i8x)$")
# preferred variant when several quantisations of the same model are installed
_QUANT_RANK = ["f16", "f32", "q8p", "i8x", "q6p", "q5p", "q4p"]
# packaging noise that varies between an official download and a custom import
# of the same weights: precision markers, format markers, the word "lora"
_TRAILING_TOKEN = re.compile(r"_(f16|f32|fp16|fp32|bf16|q8p|q6p|q5p|q4p|i8x|lora)$")


def _stem(filename: str) -> str:
    """Model identity with packaging noise stripped.

    'krea_2_turbo_i8x.ckpt' and 'krea_2_turbo_q8p.ckpt' are the same model in
    different quantisations; 'dmd2_sdxl_4_step_lora_f16.ckpt' (official) and
    'dmd2_sdxl_4step_lora_fp16_lora_f16.ckpt' (custom import) are the same
    LoRA under different packaging. Both pairs share a stem.
    """
    s = filename.removesuffix(".ckpt").lower()
    while (m := _TRAILING_TOKEN.search(s)) is not None:
        s = s[: m.start()]
    return s.replace("_", "")


def installed_files(dirs: list[str]) -> set[str]:
    found: set[str] = set()
    for d in dirs:
        p = Path(d).expanduser()
        if p.is_dir():
            found.update(f.name for f in p.glob("*.ckpt"))
    return found


def match_filename(wanted: str, installed: set[str]) -> str | None:
    """The installed file that is `wanted`, allowing the quantisation to differ."""
    if wanted in installed:
        return wanted
    stem = _stem(wanted)
    candidates = sorted(f for f in installed if _stem(f) == stem)
    if not candidates:
        return None
    def rank(f: str) -> int:
        m = _QUANT_SUFFIX.search(f.removesuffix(".ckpt").lower())
        return _QUANT_RANK.index(m.group(1)) if m else len(_QUANT_RANK)
    return min(candidates, key=rank)


class LlmConfig(BaseModel):
    model: str = "gemma3:12b"
    host: str = "http://127.0.0.1:11434"
    temperature: float = 0.9
    request_timeout: float = 300


class DrawThingsConfig(BaseModel):
    host: str = "http://127.0.0.1:7860"
    request_timeout: float = 900
    models_dirs: list[str] = DEFAULT_MODELS_DIRS
    upscalers: list[str] = []


class Registry(BaseModel):
    llm: LlmConfig
    draw_things: DrawThingsConfig
    models: dict[str, ModelSpec]
    video_models: dict[str, VideoModelSpec] = {}
    upscale_sizes: list[int] = [2048]
    # the first configured upscaler actually on disk; set by
    # resolve_installed(). None means Upscale cannot run at all.
    upscaler: Optional[str] = None
    # what the settings dialog has hidden, held here so resolve_installed()
    # can recompute `shown` from a fresh scan without the caller restating it
    disabled_models: set[str] = set()
    hide_uninstalled: bool = False
    # the per-install step overrides and LoRA ticks, held for the same reason:
    # a rescan can change which LoRAs are installed, and the stages have to be
    # rebuilt from all three inputs when it does
    step_overrides: dict[str, dict[str, int]] = {}
    lora_choices: dict[str, dict[str, bool]] = {}

    def model_or_raise(self, key: str) -> ModelSpec:
        spec = self.models.get(key)
        if spec is None:
            raise KeyError(f"unknown model key: {key}")
        if not spec.available:
            raise ValueError(
                f"model '{key}' is not installed in Draw Things"
                if spec.installed is False
                else f"model '{key}' has no dt_filename configured in models.yaml"
            )
        return spec

    def upscale_block(self, key: str) -> str | None:
        """Why Upscale cannot run for this model, or None when it can.

        Every model in the registry renders its own 2048 pass -- the ones
        whose working resolution stops short of it get there with
        tiled_diffusion -- so what is left to check is that the model is
        installed, that it does img2img, and that a Draw Things upscaler is
        available to enlarge the init: the pass is fed an upscaled init rather
        than a soft Lanczos one, and that is not optional polish. The reason
        travels with the answer because the UI disables the button and says
        what is missing rather than failing on click; a card can outlive the
        model that made it.
        """
        spec = self.models.get(key)
        if spec is None:
            return "unknown model"
        if not spec.supports_img2img:
            return f"{spec.label} does not support img2img"
        if not spec.available:
            return f"{spec.label} is not installed in Draw Things"
        if self.upscaler is None:
            return (
                "no Draw Things upscaler is installed, and the upscale pass "
                "needs one to prepare a sharp init. Download Remacri or "
                "4x UltraSharp in Draw Things, under Official Upscalers"
            )
        return None

    def upscale_candidates(self, key: str) -> list[str]:
        """Models that could run Upscale 2K for an image made by `key`.

        The image's own model comes first and is the default: it keeps the
        look it was generated with. The rest are offered because upscaling
        through a different model is a real (if blunt) creative tool -- at
        strength 0.3 the source still dominates the composition while the
        stand-in supplies the texture. Returns [] when nothing can do it.
        """
        if key not in self.models:
            return []
        others = [
            k
            for k in self.models
            if k != key
            # a model hidden in settings is not offered as a stand-in, the
            # same as it is not offered a chip. The image's own model is
            # exempt: it is checked before this list is built, so hiding a
            # model does not take the faithful upscale off its own cards
            and self.models[k].shown
            and self.upscale_block(k) is None
            # an instruction editor is not offered as a stand-in: it is not a
            # generator, and it is the slowest thing here
            and not self.models[k].edit_only
        ]
        # An instruction editor does not upscale even its own output: that
        # picture was drawn by a checkpoint and only edited, so the checkpoint
        # is the one to keep its look (the endpoint puts it first), and the
        # editor's one measured pass came back unchanged after 12 minutes.
        if self.upscale_block(key) is not None or self.models[key].edit_only:
            return others
        return [key] + others

    def apply_step_overrides(self, overrides: dict[str, dict[str, int]]) -> None:
        """Lay the per-install step settings over the shipped ones.

        Every stage is set from either the override or its `default_steps`,
        never left at what a previous call put there, so this is the whole
        answer rather than a patch on the last one: applying an emptied set of
        overrides is what Reset does. Overrides naming a model that is not in
        the registry are ignored here and kept on disk (see app/settings.py).

        Mutating `steps` in place, rather than resolving it at each of the
        dozen call sites, is what keeps this one feature from touching all of
        them -- including the stage dicts /api/models hands the page, which is
        where the UI's cost estimates and the Instruct dialog's default come
        from. What renders is `steps`, everywhere, as before.
        """
        self.step_overrides = dict(overrides)
        self._recompute_stages()

    def apply_lora_choices(self, choices: dict[str, dict[str, bool]]) -> None:
        """Lay the per-install LoRA ticks over the shipped stages.

        `choices` is {model_key: {option_key: on}}, only for what this install
        has set; an option it does not mention takes its `default`. The whole
        answer, like apply_step_overrides, and mutating in place for the same
        reason: the draft and final stages every call site already reads are
        the ones that carry the LoRA, its steps, cfg and sampler.
        """
        self.lora_choices = {k: dict(v) for k, v in choices.items()}
        self._recompute_stages()

    def _recompute_stages(self) -> None:
        """Rebuild each model's draft and final from what ships plus both settings.

        Order matters and is fixed: the shipped stage, then each ticked LoRA's
        recipe in the order the options are listed, then the step override.
        So a recipe's steps are the default the settings table shows, and a
        number the user typed still wins over it. An option whose file is not
        installed counts as unticked: the model renders as it ships rather
        than being hidden for want of an optional file.
        """
        for spec in self.models.values():
            _capture_shipped(spec)
            chosen = self.lora_choices.get(spec.key) or {}
            for opt in spec.lora_options:
                opt.on = opt.available and chosen.get(opt.key, opt.default)
            _lay_over(spec, self.step_overrides.get(spec.key) or {})

    def as_made(self, spec: ModelSpec, loras: Optional[list[LoraSpec]]) -> ModelSpec:
        """`spec` with its stages as they were when a picture was queued with `loras`.

        A tick decides a model's recipe -- size, steps, cfg, sampler, opener,
        negative -- for every stage at once, and a picture made under one
        state must be finished under the same one: a Turbo Faithful (8 real
        steps at CFG 1) over a draft made without it, at the untick 640
        budget, is a soft picture (docs/decisions/DRAFT-STEPS.md). So the
        finishing passes ask for the spec as the draft recorded it rather
        than as the chip is set now, and flipping the tick mid-chain changes
        nothing the chain does. A row from before LoRAs were recorded
        (`loras` None) has no state to inherit and takes the live one, as
        it always did.
        """
        if loras is None or not spec.lora_options:
            return spec
        files = {l.file for l in loras}
        if all(opt.on == (opt.file in files) for opt in spec.lora_options):
            return spec
        made = spec.model_copy(deep=True)
        for opt in made.lora_options:
            opt.on = opt.file in files
        _lay_over(made, self.step_overrides.get(spec.key) or {})
        return made

    def apply_model_visibility(
        self, disabled: set[str], hide_uninstalled: bool
    ) -> None:
        """Lay the per-install visibility settings over the registry.

        Like apply_step_overrides this is the whole answer rather than a patch
        on the last one, so a model dropped from `disabled` comes back. The
        two inputs are kept because `shown` also depends on `installed`, which
        a rescan can change under us -- resolve_installed() recomputes from
        them rather than leaving a model hidden after it was imported.
        """
        self.disabled_models = set(disabled)
        self.hide_uninstalled = bool(hide_uninstalled)
        self._recompute_shown()

    def _recompute_shown(self) -> None:
        for spec in (*self.models.values(), *self.video_models.values()):
            spec.enabled = spec.key not in self.disabled_models
            spec.shown = spec.enabled and (
                spec.available or not self.hide_uninstalled
            )

    def video_model_or_raise(self, key: str) -> VideoModelSpec:
        spec = self.video_models.get(key)
        if spec is None:
            raise KeyError(f"unknown video model key: {key}")
        if not spec.available:
            raise ValueError(f"video model '{key}' is not installed in Draw Things")
        return spec

    def resolve_installed(self) -> None:
        """Match every registry filename against the models actually on disk.

        The exact filename depends on choices made at download/import time
        (quantisation variant, mostly), so registry entries are treated as
        'this model, any variant': a stem match rewrites dt_filename to the
        installed file, no match marks the model unavailable so the UI hides
        its chip. Re-run any time; the scan is a few directory listings.
        """
        files = installed_files(self.draw_things.models_dirs)
        if not files:
            # Nothing to scan: an unknown folder layout, or the Draw Things
            # container momentarily unreadable while the app restarts. Leave
            # every previous answer standing rather than overwriting it with
            # the results of a scan that saw nothing. Resolving the upscaler
            # above this guard was a bug: it set self.upscaler to None, and
            # upscale_block() then reported "no upscaler installed" for every
            # model, so Upscale rendered disabled on every card. /api/models
            # rescans on each page load, so one page load at the wrong moment
            # was enough to lose the button until the next refresh.
            log.warning("no Draw Things model folders found; skipping matching")
            self._recompute_shown()
            return
        # upscalers live in the same folders and are matched the same way, so a
        # different quantisation of the same one still counts
        self.upscaler = next(
            (m for u in self.draw_things.upscalers if (m := match_filename(u, files))),
            None,
        )
        for spec in self.models.values():
            found = match_filename(spec.dt_filename, files)
            spec.installed = found is not None
            if found and found != spec.dt_filename:
                log.info("model %s: using %s for %s", spec.key, found, spec.dt_filename)
                spec.dt_filename = found
            elif not found:
                log.info("model %s (%s) not installed; chip hidden", spec.key, spec.dt_filename)
            _capture_shipped(spec)
            for stage in spec._shipped.values():
                for lora in stage.loras:
                    lfound = match_filename(lora.file, files)
                    if lfound and lfound != lora.file:
                        lora.file = lfound
                    elif not lfound:
                        # a missing fixed LoRA breaks generations even when
                        # the model itself is present
                        spec.installed = False
                        log.info("model %s: LoRA %s not installed; chip hidden", spec.key, lora.file)
            for opt in spec.lora_options:
                lfound = match_filename(opt.file, files)
                opt.installed = lfound is not None
                if lfound:
                    opt.file = lfound
                else:
                    # optional: the tick is unavailable, the model is not
                    log.info("model %s: optional LoRA %s not installed", spec.key, opt.file)
        for vspec in self.video_models.values():
            found = match_filename(vspec.dt_filename, files)
            vspec.installed = found is not None
            if found and found != vspec.dt_filename:
                vspec.dt_filename = found
            elif not found:
                log.info("video model %s (%s) not installed", vspec.key, vspec.dt_filename)
        # `shown` folds in `available`, which the scan above has just decided,
        # and a LoRA option is only on while its file is there
        self._recompute_shown()
        self._recompute_stages()


def _lay_over(spec: ModelSpec, wanted: dict[str, int]) -> None:
    """Rebuild `spec`'s stages from what ships, its `on` options, then `wanted`.

    The body of Registry._recompute_stages, taken out so as_made can run it
    on a copy for a tick state other than the live one. Expects
    _capture_shipped to have run and each option's `on` to be set.
    """
    negative = spec._shipped_negative
    spec.quality_prefix = spec._shipped_prefix
    openers: list[str] = []
    stages = {
        name: spec._shipped[name].model_copy(
            deep=True, update={"default_steps": spec._shipped[name].steps}
        )
        for name in LORA_STAGES
    }
    for opt in spec.lora_options:
        if not opt.on:
            continue
        if not opt.negative_prompt:
            negative = False
        if opt.opener:
            openers.append(opt.opener)
        for name, recipe in opt.stages.items():
            stage = stages[name]
            stage.loras = [*stage.loras, LoraSpec(
                file=opt.file, weight=opt.weight,
                civitai_version_id=opt.civitai_version_id,
            )]
            if recipe.size is not None:
                stage.size = recipe.size
            if recipe.steps is not None:
                stage.default_steps = recipe.steps
            if recipe.cfg is not None:
                stage.cfg = recipe.cfg
            if recipe.sampler is not None:
                stage.sampler = recipe.sampler
    for name, stage in stages.items():
        # only draft and final take a typed override (STAGES); the
        # upscale's count is half of a tuned pair and stays put
        stage.steps = (wanted.get(name, stage.default_steps)
                       if name in STAGES else stage.default_steps)
        setattr(spec, name, stage)
    spec.uses_negative_prompt = negative
    if openers:
        spec.quality_prefix = ", ".join(
            p for p in (spec.quality_opener, *openers) if p)


def _capture_shipped(spec: ModelSpec) -> None:
    """Keep the draft and final stages as models.yaml has them, once.

    load_registry() calls this before anything can change them; the other
    callers only reach it for a spec built by hand (tests), where the stages
    are still untouched the first time.
    """
    if not spec._shipped:
        spec._shipped = {
            name: StageSpec(**getattr(spec, name).model_dump()) for name in LORA_STAGES
        }
        spec._shipped_negative = spec.uses_negative_prompt
        spec._shipped_prefix = spec.quality_prefix


def load_registry(path: Path = CONFIG_PATH, *, lora_options: bool = True) -> Registry:
    """The registry as models.yaml ships it.

    `lora_options` False unticks every optional LoRA, which is what a
    measurement script wants unless it is measuring one: with them at their
    defaults, a sweep of a model whose accelerator ships ticked renders the
    accelerator's recipe while recording the numbers as the model's own.
    The app leaves them as shipped and lays the install's ticks over them.
    """
    raw = yaml.safe_load(path.read_text())
    models = {
        key: ModelSpec(key=key, **spec) for key, spec in (raw.get("models") or {}).items()
    }
    video_models = {
        key: VideoModelSpec(key=key, **spec)
        for key, spec in (raw.get("video_models") or {}).items()
    }
    for spec in models.values():
        # the shipped value, captured before anything can overwrite it: a
        # per-install override is applied on top of this and Reset returns to it
        for stage in (spec.draft, spec.final, spec.upscale):
            stage.default_steps = stage.steps
        _capture_shipped(spec)
    registry = Registry(
        llm=LlmConfig(**(raw.get("llm") or {})),
        draw_things=DrawThingsConfig(**(raw.get("draw_things") or {})),
        models=models,
        video_models=video_models,
        upscale_sizes=sorted(raw.get("upscale_sizes") or [2048]),
    )
    # options take their defaults until the settings are laid over them
    if lora_options:
        registry._recompute_stages()
    else:
        registry.apply_lora_choices({
            key: {o.key: False for o in spec.lora_options}
            for key, spec in models.items() if spec.lora_options
        })
    return registry


def lora_tag(spec: ModelSpec) -> str:
    """The ticked options' keys, for a measurement's filenames: "" when none.

    A frame rendered with an accelerator is a different experiment on the same
    cell and has to sit beside the plain one, not be mistaken for it.
    """
    return "+".join(o.key for o in spec.lora_options if o.on)
