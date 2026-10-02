# Setup in full

The short version is in the [README](../README.md#setup). This page covers the
rest: what the setup script does, installing by hand, the Civitai models, where
your data lives, updating, moving to another Mac, and troubleshooting.

## What `scripts/setup.sh` does

With no options, it installs the Homebrew packages, starts Ollama, sets up the
Python environment and downloads the Real-ESRGAN upscaler used by **2x fast**.
It then lists the models to install in Draw Things and where to get each one.
It stops early with a clear message if you're not on an Apple Silicon Mac.

Model downloads are optional:

| option | downloads |
|---|---|
| `--llms` | the two Ollama models, gemma3:12b and qwen3-vl:8b (~14GB) |
| `--anime-checkpoints` | three anime models from Civitai (LustrousMix Anima, BAS+ and CyberRealistic Anima, ~16GB), plus the Anima Turbo and Qwen Image Turbo LoRAs |

```bash
./scripts/setup.sh                            # tools + upscaler
./scripts/setup.sh --llms                     # + the two Ollama models
./scripts/setup.sh --llms --anime-checkpoints # + the Civitai models
```

`--anime-checkpoints` needs a Civitai API key (see below). Neither option is
required: the models in Draw Things' own catalogue are enough to use the app.

The video model is never downloaded by the script. Draw Things downloads and
converts it itself; the script just tells you what to look for.

Before downloading anything, the script checks you have enough free disk space
and stops if you don't. Importing a model into Draw Things temporarily needs
space for two copies, so leave some room to spare.

## Getting the Civitai models

Some of the models are community models hosted on Civitai rather than in Draw
Things' catalogue. There are two ways to get them:

- **With the script** (`--anime-checkpoints`, for the three below). Put a
  Civitai API key in `CIVITAI_TOKEN`, either in `.env` or exported in your
  shell. Create one at civitai.com → Account settings → API Keys. The same key
  is used by **Publish…**, so it's worth setting anyway.
- **By hand.** Open the model's page in your browser while logged in to
  Civitai, download the `.safetensors` file, then in Draw Things go to **Models
  list → Manage → Import Model**. Do this if the scripted download fails.

| Model | Civitai page |
|---|---|
| LustrousMix Anima v1.0 | https://civitai.com/models/941345?modelVersionId=2982979 |
| BAS+ Better Anime Style Plus (Anima base v1.0) | https://civitai.com/models/2675648?modelVersionId=3004325 |
| CyberRealistic Anima v6.0 | https://civitai.com/models/2668799?modelVersionId=3136380 |

These are imported by hand only:

| Model | Civitai page |
|---|---|
| WAI-illustrious-SDXL v17.0 | https://civitai.com/models/827184?modelVersionId=2883731 |
| CyberRealistic CyberIllustrious Anime v5.0 | https://civitai.com/models/2081804?modelVersionId=2886849 |
| One obsession v24 | https://civitai.com/models/1318945?modelVersionId=3218603 |
| JANKU Illustrious v7.77 | https://civitai.com/models/1277670?modelVersionId=2786084 |
| RIN Flanime Illustrious v5.0 | https://civitai.com/models/1544647?modelVersionId=3262295 |
| PerfectDeliberate-**Anime** v5.0_A | https://civitai.com/models/111274?modelVersionId=2925672 |
| Juggernaut XL Ragnarok | https://civitai.com/models/133005?modelVersionId=1759168 |

⚠️ **PerfectDeliberate-Anime is not the same as PerfectDeliberate.** The same
author publishes both with the same version numbers. You want
`perfectdeliberate_v50A.safetensors` from model **111274**, not
`perfectdeliberate_v50.safetensors` from model 24350, which a search shows
first.

The app's settings were tested with these exact versions. A newer version will
probably work too. If you install a different one, update `civitai_version_id`
for that model in `config/models.yaml`, or a published post will credit the
wrong version.

Everything else (FLUX.2 klein, FLUX.1 schnell, Z-Image Turbo, Krea 2 Turbo,
Qwen Image and Qwen Image Edit) comes from Draw Things' own catalogue and needs
no Civitai account.

To add a model the app doesn't include, see [Adding a model](ADDING-A-MODEL.md).

## The video model

Optional, for **Animate**. Search Draw Things' model manager for **Wan 2.1 14B
I2V FusionX** (17GB); it downloads and converts it for you. What it can and
can't do is in the [guide](GUIDE.md#animate).

## Manual install

1.  **Homebrew packages**:

    ```bash
    brew install ollama uv
    ```

2.  **Ollama and an LLM** (~8GB download):

    ```bash
    brew services start ollama
    ollama pull gemma3:12b
    ```

3.  **Python environment**, from the project folder:

    ```bash
    uv sync
    ```

4.  **Draw Things**: install it from the
    [Mac App Store](https://apps.apple.com/app/id6444050820) (free), then:

    - In Settings, turn on **API Server**, protocol **HTTP**, port **7860**.
      Check this again after restarting the app, as it sometimes turns off.
    - Download models in the model manager. Start with **Z-Image Turbo**
      (~6GB, and the fastest here), then add **FLUX.2 klein 9B** for a second
      look to compare. Add **Qwen Image Edit 2511** for editing by instruction
      and Combine, and the [video model](#the-video-model) for Animate.
    - **FLUX.1 schnell** works but is mostly redundant with FLUX.2 klein and
      Z-Image Turbo.
    - **Qwen Image 2512** is much slower than the rest (minutes per image).
      Use it for a single image you care about, not for trying things out.
    - Download an upscaler under **Official Upscalers**: **4x UltraSharp**
      (~34MB). **Upscale** is disabled until one is installed.
    - Import the Civitai models you want (see
      [above](#getting-the-civitai-models)).
    - Add the [LoRAs](#loras) for the models you installed.

5.  **Check the connection to Draw Things**:

    ```bash
    uv run python scripts/probe.py
    ```

    It lists the model files Draw Things has. If a filename differs from
    `dt_filename` in `config/models.yaml`, update it there. Restart the server
    after any change to that file; it's only read at startup.

### LoRAs

Small add-on files that make certain models several times faster. All are
optional, and you only need the ones for models you have.

| LoRA | size | for | what it does | where to get it |
|---|---|---|---|---|
| **Qwen Image Edit 2511 Lightning 4-Step v1.0** | 0.8GB | Qwen Edit | Edits in about 1.5 minutes instead of 5 | Draw Things' LoRA list · [source](https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning) |
| **Wuli-art Qwen-Image-2512 Turbo (4 steps)** | 1.1GB | Qwen Image 2512 | Finished images in under a minute instead of 3.5 | [Civitai](https://civitai.com/models/2267786?modelVersionId=2552570), then **LoRA list → Manage → Import LoRA**. Not the "Qwen Image 2512 Lightning" in Draw Things' list, which doesn't work here |
| **Anima Turbo LoRA v0.2** | 0.14GB | LustrousMix Anima, CyberRealistic Anima | Finished images in about a minute instead of six; changes the look | [Civitai](https://civitai.com/models/2560840?modelVersionId=2979642), then **LoRA list → Manage → Import LoRA** |

`setup.sh --anime-checkpoints` downloads the two Civitai LoRAs for you. Once
installed they're switched on; you can turn each one off in the ⚙ dialog.
[The guide](GUIDE.md#speed-loras) explains the trade-offs.

## Where your data lives

`data/` holds the only things an install can't rebuild: `draftpool.db` (every
session, prompt, seed and setting) and `sessions/`, which holds every image and
clip. Everything else comes back from the code and `scripts/setup.sh`.

It defaults to `data/` inside the project folder. **Move it somewhere else** so
an update can never touch it:

```bash
mkdir -p ~/draftpool-data
mv data/* ~/draftpool-data/
```

Then set `DRAFTPOOL_DATA_DIR` in `.env` to the new location. `.env` is created
by `setup.sh` on its first run and is never touched by updates:

```
DRAFTPOOL_DATA_DIR=/Users/you/draftpool-data
#CIVITAI_TOKEN=
```

Use a full path (`~` works). A shell variable overrides `.env`, which is handy
for a one-off run:

```bash
DRAFTPOOL_DATA_DIR=~/other-pool uv run uvicorn app.main:app --port 8500
```

When the server starts, it prints which data folder it opened:

```
data directory: /Users/you/draftpool-data (37 session(s)); override with DRAFTPOOL_DATA_DIR
```

Check this after an update or a move. If `DRAFTPOOL_DATA_DIR` is wrong, the
app simply starts with an empty history. Nothing is lost: set the path
correctly and restart.

## Updating

The database upgrades itself on the first start of a new version, so there's
no separate migration step.

| | replaced by an update? | what to do |
|---|---|---|
| `app/`, `static/`, `scripts/`, `tests/` | yes | nothing |
| `.venv/` | yes | `setup.sh` rebuilds it |
| `tools/realesrgan` | yes | keep it, or let `setup.sh` download it again |
| `data/` | **no** | keep it, see [Where your data lives](#where-your-data-lives) |
| `config/models.yaml` | yes, but you may have edited it | keep a copy of your changes, see below |
| `.env` | no | keep it |

**If you cloned the repository:**

```bash
git diff config/models.yaml     # see your own changes first
git pull
./scripts/setup.sh
uv run uvicorn app.main:app --port 8500
```

`data/` and `tools/` are ignored by git, so a pull never touches them.

**If you downloaded a zip:** unpack the new version into a *new* folder, never
over the old one. Copy `.env` across (and `data/` if it's still inside the old
folder), run `./scripts/setup.sh`, start the server and check the data folder
line. Then delete the old folder.

**`config/models.yaml` is the one file where an update can lose your changes.**
If you've edited it, keep a copy before updating and compare afterwards:

```bash
cp config/models.yaml ~/models.yaml.mine
# ...update...
diff ~/models.yaml.mine config/models.yaml
```

There's no way to downgrade the database. To go back to an older version, go
back to a copy of `data/` from before the update too.

## Moving to another Mac

1. Copy or clone the project folder, and copy your
   [data folder](#where-your-data-lives) if you want your history. Don't copy
   `.venv/`; `setup.sh` rebuilds it.
2. Run `scripts/setup.sh` and follow the Draw Things steps it prints.
3. **Models**: downloading the catalogue models again in Draw Things is usually
   easiest. Copy the ones you imported by hand (the Civitai models and LoRAs)
   from Draw Things' model folder:

   ```
   ~/Library/Containers/com.liuliu.draw-things/Data/Documents/Models
   ```

   If you use a custom model folder, make sure it's listed under
   `draw_things.models_dirs` in `config/models.yaml`.

The app matches model files automatically, even if they're a different
quantisation (`q8p`, `i8x`, `f16`). A model it can't find shows as a greyed-out
chip.

## Troubleshooting

- **`setup.sh` stops with an architecture error**: the app needs an Apple
  Silicon Mac.
- **Every `uv` command fails with `Abort trap: 6` or `Library not loaded`**:
  the `.venv` folder was copied from elsewhere. Run `rm -rf .venv && uv sync`
  (or just `scripts/setup.sh`).
- **Port already in use** (8500 for this app, 7860 for Draw Things, 11434 for
  Ollama): something else is using it, or an old copy is still running.
  `lsof -iTCP:8500 -sTCP:LISTEN` shows what. Or use a different `--port`.
- **Generate fails and Draw Things never gets a job**, with a `400 unknown
  model key` or `not installed in Draw Things` error: you changed
  `config/models.yaml` since starting the server. Restart it.
- **All model chips are greyed out**: the app can't find Draw Things' model
  folder. Check `draw_things.models_dirs` in `config/models.yaml`, then refresh
  the page.
- **`--anime-checkpoints` says `CIVITAI_TOKEN` is not set** although it's in
  `.env`: check the line isn't commented out and is a plain `KEY=value`.
- **A Civitai download fails or times out**: download from the
  [model pages](#getting-the-civitai-models) in your browser instead.
- **Your sessions disappeared after an update or move**: the app opened a
  different data folder. Check the `data directory:` line printed at startup
  and set `DRAFTPOOL_DATA_DIR` correctly. See
  [Where your data lives](#where-your-data-lives).
- **Draw Things connection refused**: the API Server setting in Draw Things
  switched itself off, often after an app update. Turn it back on.
- **"no reply within 300s" from the LLM**: the model took too long, often
  because it was still loading. Press **Retry**. If it keeps happening, choose
  a smaller model, or raise `llm.request_timeout` in `config/models.yaml`.
- **Running out of disk during downloads**: the script checks space first. The
  Civitai models need about 16GB, the LLMs about 14GB, and Draw Things' own
  models typically about 40GB more.
- **Upscale is disabled with "no upscaler installed"**: download **4x
  UltraSharp** in Draw Things under **Official Upscalers**.
