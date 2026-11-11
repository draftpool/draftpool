# Adding a model

There are two ways to add a model the app doesn't include:

- **[Plug it in from Civitai](#plugging-in-a-model-from-civitai)**: any
  Illustrious, Anima or SDXL checkpoint, in a minute, with its base model's
  settings. Start here.
- **[Measure it on your machine](#measuring-a-model-on-your-machine)**: renders
  that settle this checkpoint's own numbers. An hour or more, for when the base
  model's settings aren't good enough or the base model has no profile.

The video models need neither. **MiniMax H3** and **Wan 2.1 FusionX** are
already defined in the app: install them in Draw Things
([setup](SETUP.md#the-video-models)) and they appear in **Animate**.

## Plugging in a model from Civitai

**What can be plugged in.** A full checkpoint, the kind Civitai lists as
*Checkpoint*, of one of these base models:

| Civitai's "Base Model" | what it is | settings it gets from |
|---|---|---|
| **Illustrious** | SDXL-architecture anime models trained on Danbooru tags | six Illustrious finetunes measured here |
| **Anima** | Cosmos 2.5 2B anime models | the Anima models that ship with the app |
| **SDXL 1.0** | SDXL photographic and general models | Juggernaut XL Ragnarok |

What it doesn't take:

- **Other base models**, among them SD 1.5, FLUX and the newer
  architectures. No model of those bases was measured here, so there's no
  profile to start from, and the server leaves them out. Pony and NoobAI are
  SDXL underneath, so the server does add them, with its SDXL guess (below);
  the add script refuses them unless you name a profile with `--base`. For
  FLUX, Z-Image, Krea 2 and Qwen Image, the app
  already includes a model from Draw Things' catalogue. Anything else Draw
  Things can import can be [measured](#measuring-a-model-on-your-machine)
  instead.
- **LoRAs**, which add a style or character to a checkpoint rather than being
  one. The speed LoRAs the app uses are set up already
  ([setup](SETUP.md#loras)).
- **Video models**. Both are already defined, as above.

1. **Download the checkpoint** from its Civitai page.
2. **Import it into Draw Things**: **Models list → Manage → Import Model**.
   Any name will do.
3. **Restart the server.** The model appears in the composer's model row, and
   the server's log names what it added.

What the entry gets:

- **Settings measured on that exact version**, when it is one of the
  versions measured here individually (`config/tuned_models.yaml`), and its
  Civitai version for the credit when you publish. The app recognises the
  weights themselves, so the file's name doesn't matter.
- **Otherwise its kind's settings**, from `config/base_models.yaml`: what the
  models of that base measured here agree on. A sound start, not a measurement
  of your checkpoint. Draw Things says whether it is an Anima or an SDXL
  model; an SDXL one whose name says *Illustrious* or *Noob* gets the
  Illustrious settings, and any other the plain SDXL ones.

A model the app already has, imported again under another name, is the same
model: it is not added a second time.

The server looks at each import once, on the first start after it, and from
then on uses the entry in `config/models.private.yaml`. To set one up
yourself instead -- when the guess is wrong (an Illustrious model whose name
doesn't say so), or to credit the checkpoint when you publish -- delete the
entry the server added, then name the version the way its Civitai page shows
it:

```bash
uv run python scripts/add_civitai_model.py                     # what the app doesn't use
uv run python scripts/add_civitai_model.py <file> --air "urn:air:sdxl:checkpoint:civitai:1307857@3313575"
uv run python scripts/add_civitai_model.py <file> --hash <SHA256 from the page>
uv run python scripts/add_civitai_model.py <file> --civitai <version id>
```

The AIR is under the version's details, with a copy button; the SHA256 is the
**Hash** row, and it is the hash of the file you downloaded, not of Draw
Things' converted copy. The version id is the number after `modelVersionId=`
in the page's address. Restart the server afterwards.

| option | use |
|---|---|
| `--base illustrious`, `--base anima`, `--base sdxl` | name the base yourself instead of asking Civitai. An Anima model needs neither, because Draw Things already knows it is one |
| `--key`, `--label` | the app's internal name for it, and the name it shows |
| `--dry-run` | print the entry without adding it |

The script refuses, rather than guesses, when:

- Civitai lists the checkpoint as **Pony**, **NoobAI** or another base with no
  profile. No model of those bases was measured here; `--base` overrides it if
  you know which profile fits.
- The base it's told doesn't match the architecture Draw Things recorded for
  the import, which usually means the wrong file was picked.
- The import is already in the app's config.

The entry goes in `config/models.private.yaml`, which is yours: updates never
touch it and it isn't part of the repository. To change a setting, edit the
entry there and restart. Sampler and CFG vary most between finetunes, so if
the model's Civitai page recommends others, those are the two to change. To
remove the model, delete its entry.

## Measuring a model on your machine

Renders that settle the checkpoint's own sampler, steps and sizes. Use it when
the profile's settings don't suit a checkpoint, or for a base model the add
script has no profile for.

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
- Check the author's other pages first. A model can be one letter from a
  different model by the same author with the same version number (an "-Anime"
  variant against the plain one, say), and searching the base name doesn't
  show the variant.

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
