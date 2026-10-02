# Adding a model

How to add a model that doesn't come with the app.

## 1. Import it into Draw Things

Download the checkpoint and import it in Draw Things: **Models list → Manage →
Import**. The quantisation doesn't matter. The app matches any variant of the
filename it finds (q8p, i8x, f16), and hides the chip when none is installed.

## 2. Run the harness

```
uv run python scripts/add_model.py --model <dt_filename> --key <key>
```

Options:

| flag | use |
|---|---|
| `--tags` | the checkpoint is Danbooru or booru-trained |
| `--level quick` | about 18 renders instead of 30, for a slow model |
| `--label` | UI label; defaults to the name Draw Things has |
| `--final` | candidate final size, default 1024 |

It renders about 30 images and prints a draft `models.yaml` entry with every
number it could establish. A run is resumable: re-running skips cells that
finished, so an interrupted run costs nothing. It also refuses to start while
another batch is running, because two batches sharing Draw Things distort each
other's timings.

A full run takes about 20 minutes on a fast model and hours on a slow one. If
the new model has the same architecture as one already in the app, there is a
shorter route, below.

## 2b. The short path, when it is the same machine as a model you have

If the new checkpoint has **exactly the same architecture** as a model already
in `config/models.yaml` (same base version, autoencoder and text encoders),
most settings can be copied from that model. The sampler and CFG can't: they
differ even between closely related models, so the short path tests just
those:

```
uv run python scripts/add_model.py --model <dt_filename> --key <key> \
  --like <measured-sibling-key> --level confirm \
  --card-sampler "Euler a" --card-cfg 5 --card-steps 30
```

This takes about a quarter of an hour instead of an hour.

| flag | use |
|---|---|
| `--like` | the sibling to inherit geometry from. Must be a **measured** key and must match on architecture; the script checks both and refuses otherwise |
| `--card-sampler`, `--card-cfg`, `--card-steps` | what the model card recommends. A Civitai import is not in the Draw Things catalogue, so the card is the only other source there is |

It prints every setting it copied, and writes three comparison sheets under
`research/confirm/<key>/`:

- `sheet_sampler_cfg.png` — the card's sampler against the sibling's, at the
  card's cfg and about a third lower, two seeds. Look for rainbow speckle on
  skin, hair edges and metal, and for a waxy face with an unformed mouth. A
  speckled high-cfg column and a clean low one is CFG burn; both cfgs speckled
  under one sampler and clean under the other is the sampler.
- `sheet_step_direction.png` — the same cells at half again the steps. Worse
  means leftover ancestral noise, better means it was only under-converged.
- `sheet_final_size.png` — one subject, or two. That is all this sheet is for.

Then fill in the sampler and CFG and go to step 5. The script also lists what
it didn't test.

If the architecture differs at all, or the model you compare against was never
fully measured, the script refuses. Run a full batch instead.

## 3. Look at the images it names

The script prints the folder. Four things need your eyes, because it cannot
judge them:

- `grid*.png` — does the subject appear twice at any size? That decides the
  final size.
- `draft*.png` — does the small render look like the big one? That decides the
  draft size.
- `portrait_*` and `wide_4242`, against the `*_lenL0_*` squares — did the
  non-square shapes compose correctly? A first look; step 7 checks every ratio.
- `steps*.png` — does the recommended step count actually beat half of it?

## 4. Fill in the TODOs

The printed entry has two, plus one guess to check:

- `draft: { size: TODO }` — from the draft sheet above.
- `aspect_size` — already filled in with a guess: the final size, or 1024 for
  an SDXL model. Keep it unless the portrait or wide sheet duplicated the
  subject, in which case lower it.
- `prompt_guidance` — a paragraph telling the LLM how this model wants to be
  prompted. Write it from the model's card; it is the model's dialect, not
  something to measure. Copy the shape of an existing entry in
  `config/models.yaml`.

## 5. Add `civitai_version_id`

This credits the model when you publish an image made with it. The script
doesn't fetch it. Without it, Civitai shows *"We weren't able to detect any
resources used in the creation of this image"*.

Find it with `GET /api/v1/models/{id}` and match your `dt_filename` against the
`.safetensors` each version ships. Hashes don't work, because Draw Things
renders from its own conversion. `uv run python scripts/check_civitai_ids.py`
prints what every id in `models.yaml` currently resolves to, which is the
quickest way to check your new one.

**Look for it even when the model came from Draw Things' own catalogue.** Most
well-known models have an official page on Civitai.

Three warnings:

- If the filenames don't line up exactly, leave it blank. Crediting the wrong
  checkpoint is worse than crediting none, and nothing will tell you.
- Don't trust the first search result. Re-uploads often rank above the
  official page. Check the creator (`CivitaiOfficial` for a catalogue model)
  and the base model.
- Check the author's other pages first. `perfectdeliberate` is one letter from
  a different model by the same author with the same version number
  (PerfectDeliberate v5.0 vs PerfectDeliberate-Anime v5.0_A), and searching the
  base name doesn't show the variant.

## 6. Paste the entry in and restart the server

Put the entry in `config/models.yaml`, then **restart the server**.
`config/models.yaml` is read once, at startup, so nothing you add appears until
you do.

If you skip the restart, nothing looks broken. Draw Things is up, Ollama is up,
`/api/health` is all green, and the Draw Things queue stays empty because the
request fails before a job is queued:

| what you changed | what happens |
|---|---|
| added a key | 400 `unknown model key: <key>` |
| renamed or removed a key | the old chip is still on screen; 400 `model '<old-key>' is not installed in Draw Things` |

The second one looks like a Draw Things problem and isn't.

## 7. Check every shape

    uv run python scripts/sweep_aspect.py --models <key>

Renders every ratio the shape row offers, plus a square, at two seeds, and
writes one sheet per seed to `research/aspect/<key>/`. It uses the draft step
count, so it takes minutes rather than hours. Look for a second person, a second
head, or a second horizon further down the frame. If a shape shows one and the
square at the same seed doesn't, lower `aspect_size` and run it again. If the
square shows one too, restart Draw Things before believing either.

## What the harness decides, and what it doesn't

| field | where it comes from |
|---|---|
| `sampler`, `cfg`, `steps`, `clip_skip` | Draw Things' own catalogue |
| `prompt_budget` | measured, 10 renders |
| `tiled_diffusion`, `upscale_seconds` | measured |
| `final.size`, `draft.size`, `aspect_size` | measured, but you judge the sheets |
| `uses_negative_prompt` | `cfg > 1` |
| `prompt_style` | prose, or tags with `--tags` |
| `max_upscale_size` | the largest size measured good |
| `upscale.steps` | half the final's |
| `prompt_guidance` | you write it |
| `civitai_version_id` | you look it up |

On the short path (`--level confirm`) only `sampler` and `cfg` come from
renders; every other row above is inherited from `--like` and printed as it is
inherited.

A model imported from Civitai isn't in the Draw Things catalogue, so take the
sampler and CFG from its model card, not from a similar model already in
`config/models.yaml`.

## If the results look wrong

- **Rainbow speckle on skin, hair edges and metal.** Try cfg about a third
  lower, then try `DPM++ 2M Karras`. Raise the step count to tell the two
  apart: too few steps gets better with more, ancestral noise gets worse.
- **Soft or blobby Faithful results.** Check the draft size before touching the
  final's settings. Faithful is img2img over the draft, so the draft is all the
  structure it has.
- **A model at cfg 1 with a single-digit step count** carries a distilled
  few-step recipe. Leave its sampler alone; swapping in a deterministic one can
  produce pure noise.
- **Empty responses above a certain size or step count.** Some models do this
  instead of failing. Record what worked and cap the entry there.
