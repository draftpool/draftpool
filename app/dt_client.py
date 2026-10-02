"""Async client for the Draw Things HTTP API (A1111-compatible).

Draw Things processes one generation at a time; an asyncio.Lock serialises
requests so queued jobs never hit the server concurrently.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
from typing import Any, Optional

import httpx
from PIL import Image

log = logging.getLogger(__name__)


class DrawThingsError(RuntimeError):
    pass


STALL_HINT = (
    "Draw Things answered but returned no image: its render session has "
    "stopped working. The HTTP server still replies, so the app looks up, and "
    "every job queued behind this one would fail the same way. Restarting "
    "Draw Things is the only known fix; queued jobs are held, not lost."
)


class DrawThingsStalled(DrawThingsError):
    """200 OK with an empty `images` list.

    Not a bad request. A Draw Things session that has stopped rendering
    answers this way for every job, whatever the model or size, while `GET /`
    keeps returning 200 -- see docs/decisions/SHAPES.md, where a whole
    measurement batch was taken on a session in exactly this state. Callers
    treat it as "the renderer is out", not "this generation failed".

    It is no longer the only cause of an empty list, which matters because
    this class states a diagnosis rather than a symptom: a render cancelled by
    hand in the Draw Things window answers identically, as does qwen-image
    above a threshold of work per request. Both are recoverable and neither
    needs a restart, so a caller acting on this should be ready to be wrong.
    docs/decisions/FALSE-SUCCESS.md is the list and how to tell them apart.
    """


class DrawThingsTimeout(DrawThingsError):
    """The client gave up waiting; Draw Things may still be rendering.

    Draw Things hands an image back in the response to the POST that asked
    for it and nowhere else: its HTTP API has no /sdapi/v1/progress, no
    history and no gallery endpoint (all 404 -- only `/`, `options`, `txt2img`
    and `img2img` answer). So a render that finishes after this is raised
    cannot be collected afterwards, by a retry or by anything else, and the
    work is spent. That is why the timeout is sized from what the model has
    actually been taking rather than left at one flat number.
    """


class DrawThingsClient:
    def __init__(self, host: str, request_timeout: float = 900) -> None:
        self._host = host.rstrip("/")
        self._timeout = request_timeout
        self._lock = asyncio.Lock()
        # set by the first empty response, cleared by the next real image (or
        # by the app going away, which ends that session); read by /api/health
        self.stalled = False

    async def is_up(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(self._host + "/")
                return resp.status_code == 200
        except httpx.HTTPError:
            # the app is gone, so whatever state its renderer was in is gone
            # with it; the next one to answer here is a fresh session
            self.stalled = False
            return False

    async def current_config(self) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(self._host + "/")
            resp.raise_for_status()
            return resp.json()

    def _build_payload(
        self,
        *,
        prompt: str,
        negative_prompt: str,
        model_filename: str,
        width: int,
        height: int,
        steps: int,
        cfg: float,
        seed: int,
        sampler: Optional[str],
        clip_skip: int = 1,
        loras: Optional[list[dict[str, Any]]] = None,
        tiled_diffusion: bool = False,
        tile: int = 1024,
        upscaler: Optional[str] = None,
        upscaler_scale: int = 0,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "model": model_filename,
            "width": width,
            "height": height,
            # steps < 2 crashes Draw Things' sampler discretization
            "steps": max(2, steps),
            "cfg_scale": cfg,
            "seed": seed,
            "batch_size": 1,
        }
        # Same trap as `loras` and `upscaler` below, and it has already cost a
        # model: omitting `sampler_name` does not select the checkpoint's own
        # default, it inherits whatever sampler is selected in the Draw Things
        # UI -- which the last render through this client is what set. So a
        # spec with no sampler renders differently depending on which model ran
        # before it, and krea2-turbo returned pure noise for a day because
        # three unrelated models had moved to DPM++ 2M Karras. Every spec now
        # names one (Draw Things' catalogue has a recommendation for each; see
        # SAMPLERS in scripts/add_model.py), so this branch should not be
        # reachable -- keep it that way rather than relying on it.
        if sampler:
            payload["sampler_name"] = sampler
        # Sent, and as far as anything measurable goes, ignored -- the same
        # trap as `shift` below. The same prompt and seed on an SDXL checkpoint
        # with clip_skip sent and with it omitted came back at mean difference
        # 0.011 (max 8) on two models, i.e. the same image. Draw Things appears
        # to use its own per-model value whatever this says. Left in place
        # because it costs nothing and the day it starts working the entries
        # already carry the right number; do not spend renders tuning it.
        if clip_skip > 1:
            payload["clip_skip"] = clip_skip
        # Always send loras explicitly: omitting the field makes Draw Things
        # fall back to whatever LoRA is selected in the app UI. Only the two
        # fields its API takes: a LoRA as recorded here also carries its
        # Civitai version, which is ours, not the renderer's.
        payload["loras"] = [
            {"file": l["file"], "weight": l.get("weight", 1.0)} for l in loras or []
        ]
        # Same trap, worse consequence. Draw Things applies whatever upscaler
        # is selected in the app UI when the field is absent, and it multiplies
        # the *rendered* size: with one selected, a 512 request comes back 2048
        # and every stage silently renders 4x too big. Downloading an upscaler
        # in the app is enough to select it, so this is not a hypothetical.
        # Hence always explicit, and off unless a caller asks for it (only
        # enlarge() does).
        payload["upscaler"] = upscaler
        payload["upscaler_scale"] = upscaler_scale if upscaler else 0
        # `shift` is deliberately not sent here, and not because nobody thought
        # of it: Draw Things recommends a different value per model (3 for
        # Z-Image Turbo, 2 for Qwen Image, 1 for Ideogram 4), so a per-model
        # field looked worth adding. Measured first, on txt2img: the same
        # prompt and seed at shift 1, 2 and 3 came back PIXEL-IDENTICAL, mean
        # and max difference both exactly 0. The endpoint accepts the field and
        # ignores it, exactly as it does the separate-encoder fields in
        # docs/decisions/PROMPT-LENGTH.md. Whether Draw Things' own
        # resolution-dependent shift setting is what overrides it was not
        # tested -- only that nothing this client can send changes the image.
        # img2video is the exception and does send it; see VideoModelSpec.shift
        # and docs/decisions/VIDEO.md, Finding 5.
        if max(width, height) >= 2048:
            payload["tiled_decoding"] = True
            # Two different tilings, and only this one lifts a model's
            # resolution ceiling. tiled_decoding above tiles the VAE decode,
            # which bounds decode memory but does nothing for the denoiser:
            # several models (Cosmos 2B, Krea 2, Qwen Edit) return a flat
            # colour field above ~2.4MP with it on and still report success.
            # tiled_diffusion tiles the denoising loop itself, so each tile
            # sits inside the model's working resolution even when the whole
            # canvas does not, which takes those models from unusable to
            # correlating 0.996+ with their own source image.
            #
            # It is opt-in per model rather than automatic at this size: the
            # models that already render 2048 natively gain nothing from it
            # and are left on the path they were measured on.
            if tiled_diffusion:
                payload["tiled_diffusion"] = True
                payload["diffusion_tile_width"] = tile
                payload["diffusion_tile_height"] = tile
                payload["diffusion_tile_overlap"] = 128
        return payload

    async def _post_all(
        self, endpoint: str, payload: dict[str, Any], timeout: Optional[float] = None
    ) -> list[bytes]:
        """Every image the endpoint returned, in order.

        For a still that is a one-element list; with num_frames set, Draw
        Things returns the whole clip here, one PNG per frame.
        """
        async with self._lock:
            log.info(
                "draw-things %s model=%s %sx%s steps=%s seed=%s frames=%s",
                endpoint,
                payload.get("model"),
                payload.get("width"),
                payload.get("height"),
                payload.get("steps"),
                payload.get("seed"),
                payload.get("num_frames", 1),
            )
            waited = timeout or self._timeout
            async with httpx.AsyncClient(timeout=waited) as client:
                try:
                    resp = await client.post(self._host + endpoint, json=payload)
                except httpx.TimeoutException as exc:
                    # httpx raises these with no message at all, which reached
                    # the card as a failure with nothing written on it
                    raise DrawThingsTimeout(
                        f"Draw Things did not answer within {round(waited)}s. The "
                        "render may still be finishing in Draw Things, but its API "
                        "cannot hand back an image once this request has gone: if "
                        "the picture appears there, it has to be saved by hand. "
                        "Retry re-renders it from the start."
                    ) from exc
        if resp.status_code != 200:
            raise DrawThingsError(
                f"{endpoint} returned {resp.status_code}: {resp.text[:500]}"
            )
        body = resp.json()
        images = body.get("images") or []
        if not images:
            self.stalled = True
            raise DrawThingsStalled(f"{STALL_HINT} ({endpoint} returned {str(body)[:200]})")
        self.stalled = False
        out = []
        for b64 in images:
            # Some A1111-compatible servers prefix a data URI
            if "," in b64 and b64.strip().startswith("data:"):
                b64 = b64.split(",", 1)[1]
            out.append(base64.b64decode(b64))
        return out

    async def _post(self, endpoint: str, payload: dict[str, Any]) -> bytes:
        return (await self._post_all(endpoint, payload))[0]

    async def txt2img(self, *, timeout: Optional[float] = None, **kwargs: Any) -> bytes:
        return (await self._post_all(
            "/sdapi/v1/txt2img", self._build_payload(**kwargs), timeout=timeout
        ))[0]

    async def renders(self, model_filename: str, *, timeout: float = 120) -> bool:
        """Does the renderer actually work, as opposed to merely answering?

        `is_up()` only proves the HTTP server is alive; a stalled session
        passes it. This asks for the cheapest real image there is (256px, the
        size the FLUX drafts already use, at the 2-step floor) and reports
        whether one came back. Used to decide when a held job may resume, so
        it must be allowed to hang: a stalled session often does, and a wrong
        "healthy" costs a full-size render to find out.
        """
        payload = self._build_payload(
            prompt="a red square",
            negative_prompt="",
            model_filename=model_filename,
            width=256,
            height=256,
            steps=2,
            cfg=1.0,
            seed=1,
            sampler=None,
        )
        try:
            await self._post_all("/sdapi/v1/txt2img", payload, timeout=timeout)
            return True
        except (DrawThingsError, httpx.HTTPError):
            return False

    async def enlarge(
        self,
        init_image_png: bytes,
        *,
        target: int,
        upscaler: str,
        model_filename: str,
        sampler: Optional[str] = None,
        cfg: float = 1.0,
        seed: int = 0,
    ) -> bytes:
        """Enlarge an image through Draw Things' own upscaler, no diffusion.

        Used to prepare the init for an upscale pass. `denoising_strength: 0`
        turns the denoiser off entirely while the upscaler still runs, so this
        is a pure enlargement: measured 0.9990 correlation with the source,
        against 0.9886 at strength 0.05, and roughly twice as quick.

        upscaler_scale multiplies the *rendered* size rather than setting it,
        and the models are 4x, so the only useful scales are 2 and 4. This
        overshoots to whichever of those clears `target` and leaves the caller
        to come down to the exact size, because downscaling is clean and
        upscaling is not.

        A model still has to be named even with the denoiser off, so callers
        pass the one already loaded for the pass that follows; it costs
        nothing extra and avoids a second model load.
        """
        img = Image.open(io.BytesIO(init_image_png))
        if max(img.size) >= target:
            return init_image_png
        scale = 2 if max(img.size) * 2 >= target else 4
        payload = self._build_payload(
            prompt="",
            negative_prompt="",
            model_filename=model_filename,
            width=img.width,
            height=img.height,
            # 2 is the floor _build_payload enforces; with strength 0 no step
            # actually runs, so this is a formality
            steps=2,
            cfg=cfg,
            seed=seed,
            sampler=sampler,
            upscaler=upscaler,
            upscaler_scale=scale,
        )
        payload["init_images"] = [base64.b64encode(init_image_png).decode()]
        payload["denoising_strength"] = 0.0
        return await self._post("/sdapi/v1/img2img", payload)

    async def img2img(
        self,
        init_image_png: bytes,
        strength: float,
        *,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> bytes:
        payload = self._build_payload(**kwargs)
        init = fit_init_image(init_image_png, kwargs["width"], kwargs["height"])
        payload["init_images"] = [base64.b64encode(init).decode()]
        payload["denoising_strength"] = strength
        return (await self._post_all("/sdapi/v1/img2img", payload, timeout=timeout))[0]

    async def img2video(
        self,
        init_image_png: bytes,
        *,
        prompt: str,
        model_filename: str,
        width: int,
        height: int,
        steps: int,
        cfg: float,
        seed: int,
        num_frames: int,
        fps: int,
        motion_scale: int = 127,
        start_frame_guidance: float = 1.0,
        guiding_frame_noise: float = 0.02,
        shift: float = 5.0,
        timeout: Optional[float] = None,
    ) -> list[bytes]:
        """Animate a still into `num_frames` PNG frames, in order.

        The start frame goes in as init_images: Wan I2V models condition on it
        through a dedicated path, so denoising_strength stays at 1.0. Lowering
        it is not what makes the start frame stick for these models.

        Callers must read the real frame size off the returned PNGs rather than
        trusting the request: Draw Things floors each side to a multiple of 64,
        so 832x480 comes back as 832x448 and 416x832 as 384x832. Sizes from
        video.clip_size() are already on that grid and come back unchanged, but
        a hand-picked size may not.
        """
        payload: dict[str, Any] = {
            "prompt": prompt,
            "negative_prompt": "",
            "model": model_filename,
            "width": width,
            "height": height,
            "steps": max(2, steps),
            "cfg_scale": cfg,
            "seed": seed,
            "batch_size": 1,
            "loras": [],
            # see _build_payload: an upscaler selected in the app UI would
            # otherwise be applied to every frame
            "upscaler": None,
            "upscaler_scale": 0,
            "init_images": [
                base64.b64encode(fit_init_image(init_image_png, width, height)).decode()
            ],
            "denoising_strength": 1.0,
            "num_frames": num_frames,
            "fps": fps,
            "motion_scale": motion_scale,
            "start_frame_guidance": start_frame_guidance,
            "guiding_frame_noise": guiding_frame_noise,
            # like `upscaler` above: left out, Draw Things uses whatever its own
            # slider holds, so an identical request renders differently after
            # somebody touches the app. See docs/decisions/VIDEO.md, Finding 5.
            "shift": shift,
        }
        frames = await self._post_all("/sdapi/v1/img2img", payload, timeout=timeout)
        if len(frames) < 2:
            raise DrawThingsError(
                f"expected {num_frames} frames but got {len(frames)}: the selected "
                "model is probably not an image-to-video model"
            )
        return frames


def fit_init_image(png: bytes, width: int, height: int) -> bytes:
    """Draw Things requires init_images to match the requested output size
    exactly, so resize (Lanczos) before sending.

    This is a plain resize, which distorts the image if the requested size is
    not the source's aspect. Nothing is cropped to avoid that: callers pass a
    size derived from the source instead (see video.clip_size).
    """
    img = Image.open(io.BytesIO(png))
    if img.size == (width, height):
        return png
    img = img.convert("RGB").resize((width, height), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
