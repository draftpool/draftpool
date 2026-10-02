#!/bin/bash
# One-time setup for draftpool on a new Mac (Apple Silicon).
#
# Installs the tooling (Ollama, uv, Real-ESRGAN) unconditionally; model
# downloads are opt-in and additive:  ./scripts/setup.sh [--llms] [--anime-checkpoints]
# See --help. Draw Things itself, and every model in its own catalogue, is a
# manual step: the script ends by printing exactly what to install and where
# each piece comes from.
#
# The Civitai checkpoints behind --anime-checkpoints need a login, either way round:
#   export CIVITAI_TOKEN=...   # civitai.com -> Account settings -> API Keys
# or skip the token and open the model page links this script prints (at the
# end, and inline on any failure) -- a browser session is logged in already.
set -euo pipefail

# Model names are what you type into Draw Things' search box, so they are
# emphasised in the output. Bold only, and only on a real terminal: piping to
# a file or a dumb TERM gets clean text, and NO_COLOR is honoured.
if [ -t 1 ] && [ -z "${NO_COLOR:-}" ] && [ -n "${TERM:-}" ] && [ "$TERM" != "dumb" ]; then
  B=$'\033[1m'; N=$'\033[0m'
else
  B=""; N=""
fi

DOWNLOAD_LLMS=0
DOWNLOAD_CHECKPOINTS=0

# The project's own settings file (gitignored; written further down on a
# first run, so it is normally present from the second run onward).
# Parsed rather than sourced -- it is read by app/__init__.py too, which
# understands KEY=value and nothing else, and sourcing would run whatever is in
# it. An already-exported variable wins, same as there.
ENV_FILE="$(dirname "$0")/../.env"
if [ -f "$ENV_FILE" ]; then
  while IFS='=' read -r key value; do
    key="${key#export }"
    key="$(echo "$key" | tr -d '[:space:]')"
    case "$key" in
      ''|\#*|*[!A-Za-z0-9_]*) continue ;;
    esac
    value="${value%%  #*}"
    value="$(echo "$value" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' -e 's/^"\(.*\)"$/\1/' -e "s/^'\(.*\)'$/\1/")"
    [ -z "${!key:-}" ] && export "$key=$value"
  done < "$ENV_FILE"
fi

usage() {
  cat <<'USAGE'
Usage: ./scripts/setup.sh [--llms] [--anime-checkpoints]

Always run, no flags needed: Homebrew packages (Ollama, uv), the Ollama
service, the Python environment, and the Real-ESRGAN upscaler. With no flags
you still get all of that, plus a printed list of every model to install and
where each one comes from.

The flags are additive and only decide what gets downloaded for you:

  --llms               The two Ollama models, ~14GB: gemma3:12b (prompt
                        writing) and qwen3-vl:8b (reads images; needed by
                        From image..., Finalize+ and Animate's Improve with
                        LLM).
  --anime-checkpoints   Three Danbooru-tagged checkpoints from Civitai,
                        ~16GB: the LustrousMix Anima, BAS+ and
                        CyberRealistic Anima mixes. Its own flag
                        because these live on Civitai rather than Draw
                        Things' own catalogue, so they need CIVITAI_TOKEN
                        (civitai.com -> Account settings -> API Keys); none
                        of them is needed to run the app.

Models in Draw Things' own catalogue (FLUX, Krea 2, Juggernaut, Qwen Edit,
both video models) are never downloaded here -- the app fetches and
converts those itself. Setup prints their exact names instead.

Free space is checked against the total before anything downloads, so the run
stops rather than filling the disk part way through.

Examples:
  ./scripts/setup.sh                            # tooling + upscaler + what to install
  ./scripts/setup.sh --llms                     # + the two Ollama models
  ./scripts/setup.sh --llms --anime-checkpoints # + the three anime checkpoints
USAGE
}

for arg in "$@"; do
  case "$arg" in
    --llms) DOWNLOAD_LLMS=1 ;;
    --anime-checkpoints) DOWNLOAD_CHECKPOINTS=1 ;;
    # dropped rather than aliased: it used to pull checkpoints too, and doing
    # that silently under a flag that says nothing about them would be worse
    # than an error that names the replacement
    --download-models)
      echo "--download-models is gone. Use: --llms (Ollama models) and/or"
      echo "--anime-checkpoints (the Civitai anime checkpoints)."
      exit 1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $arg"; echo; usage; exit 1 ;;
  esac
done

# Civitai checkpoints Draw Things cannot fetch itself. Version IDs pinned to
# the releases config/models.yaml is tuned for. Each row also carries the
# model's ordinary web page: that page is downloadable straight from a
# browser (your logged-in Civitai session covers any creator-side
# login-required flag) if you'd rather skip CIVITAI_TOKEN entirely, and it's
# the fallback whenever the scripted download below fails.
# Format: <download name> <civitai version id> <download GB> <url>
# Every model here is a Danbooru-tagged community mix on the Anima/Cosmos 2B
# base (none of them is SDXL, so SDXL LoRAs do not apply). Sizes are
# the rough download size, used for the disk check below; importing into Draw
# Things converts the file and keeps both copies for a while, so budget
# roughly double while an import runs.
CIVITAI_MODELS="
  hosekiLustrousmixAnima_animaV10 2982979 5 https://civitai.com/models/941345?modelVersionId=2982979
  BASBetterAnimeStylePlusAnima_baseV10 3004325 5 https://civitai.com/models/2675648?modelVersionId=3004325
  cyberrealisticAnima_v60AestheticV11 3136380 5 https://civitai.com/models/2668799?modelVersionId=3136380
"
# LoRAs from Civitai, fetched with the checkpoints they belong to. Kept apart
# from CIVITAI_MODELS because the disk check sums whole gigabytes and these are
# a fraction of one. Optional: without one, its models render at their own
# (slower) settings -- see docs/GUIDE.md, Speed LoRAs.
# Format: <download name> <civitai version id> <download MB> <url>
CIVITAI_LORAS="
  anima-turbo-lora-v0.2 2979642 140 https://civitai.com/models/2560840?modelVersionId=2979642
  wuli-qwen-image-2512-turbo-4steps 2552570 1125 https://civitai.com/models/2267786?modelVersionId=2552570
"
# Ollama pulls, measured. Format: <model tag> <download GB>
OLLAMA_MODELS="
  gemma3:12b 8
  qwen3-vl:8b 6
"
LLM_GB=14        # both of the above, for messages about --llms in general
LLM_NEED_GB=14   # only what is actually missing; recomputed by the disk check

# A model that is already pulled costs no disk space to pull again, so it
# should not count toward the estimate. `ollama list` prints a header row then
# one row per model with the tag in the first column; an unreachable daemon
# prints nothing, in which case everything counts and the estimate is merely
# pessimistic.
ollama_has() {
  ollama list 2>/dev/null | awk 'NR>1 {print $1}' | grep -qx "$1"
}
# What Draw Things itself will pull for the video model, printed at the end so
# the number is visible before starting the download in the app.
VIDEO_GB=17

# GB free on the volume holding a given path
free_gb() { df -Pk "$1" | awk 'NR==2 {printf "%d", $4/1024/1024}'; }

echo "== 1/6 Platform check"
if [ "$(uname -s)" != "Darwin" ]; then
  echo "!! This project only runs on macOS (Draw Things is Mac-only)."; exit 1
fi
if [ "$(uname -m)" != "arm64" ]; then
  echo "!! Draw Things requires Apple Silicon (arm64); this Mac reports $(uname -m)."
  echo "   An Intel Mac cannot run the local models this project depends on."
  exit 1
fi

echo "== 2/6 Homebrew packages"
command -v brew >/dev/null || { echo "!! Install Homebrew first: https://brew.sh"; exit 1; }
brew list ollama >/dev/null 2>&1 || brew install ollama
brew list uv >/dev/null 2>&1 || brew install uv

echo "== 3/6 Ollama service"
brew services start ollama

# Ollama 0.6.4 (2 April 2025) was the first release whose /api/show reports a
# model's `capabilities`. That is the only way this app can tell a vision model
# from a text-only one, and it decides whether the image-reading flows (From
# image..., Finalize+, and Improve with LLM inside Animate) actually get shown
# the picture. On an older build every model looks equally capable, so those
# flows quietly run blind and the LLM invents what it cannot see.
MIN_OLLAMA="0.6.4"
# guarded so a missing binary yields an empty string instead of tripping set -e
ollama_version() {
  command -v ollama >/dev/null 2>&1 || return 0
  ollama --version 2>/dev/null | awk '{print $NF}'
}
# true when $1 is an older version than $2
older_than() {
  [ "$1" != "$2" ] && [ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | head -1)" = "$1" ]
}

have="$(ollama_version || true)"
if [ -z "$have" ]; then
  echo "-- could not read 'ollama --version'; skipping the version check"
elif older_than "$have" "$MIN_OLLAMA"; then
  echo "-- Ollama $have is too old (need $MIN_OLLAMA+ so /api/show reports model"
  echo "   capabilities; without it vision models cannot be told apart)."
  echo "-- upgrading via Homebrew"
  brew upgrade ollama
  brew services restart ollama
  have="$(ollama_version || true)"
  if [ -n "$have" ] && older_than "$have" "$MIN_OLLAMA"; then
    echo "!! still on $have after the upgrade. Install a build >= $MIN_OLLAMA from"
    echo "   https://ollama.com/download, or the image-reading actions will run blind."
  else
    echo "-- now on $have"
  fi
else
  echo "-- Ollama $have (>= $MIN_OLLAMA, reports model capabilities)"
  # a newer build is worth having but nothing here depends on it, so say so
  # rather than upgrading something that already works
  if brew outdated --quiet ollama 2>/dev/null | grep -q .; then
    echo "-- a newer Ollama is available: brew upgrade ollama"
  fi
fi

echo "== 4/6 Python environment"
cd "$(dirname "$0")/.."
# A .venv does not reliably survive being copied. .venv/bin/python is a symlink
# to the interpreter uv manages outside the repo, and any copy that follows
# symlinks instead of preserving them (zip -r without -y, cp -RL, rsync without
# -l) replaces it with a real copy of the launcher. That copy still resolves its
# runtime as @executable_path/../lib/libpython3.x.dylib, which now points inside
# the copied venv, where no libpython exists. Every uv command in that folder
# then aborts with SIGABRT and a dyld trace before it can say anything useful.
# Deleting the interpreter a venv was built from (a uv Python upgrade, a
# Homebrew uninstall) fails the same way. Neither is repairable and a venv is
# disposable, so rebuild from uv.lock rather than leave that to be decoded.
# The trailing 2>/dev/null is on the group, not the command: bash reports a
# child killed by a signal on its own stderr, so without it a raw "Abort trap:
# 6" line lands on top of the explanation below.
if [ -e .venv ] && ! { .venv/bin/python -c "" >/dev/null 2>&1; } 2>/dev/null; then
  echo "-- .venv has a broken interpreter, rebuilding it from uv.lock"
  echo "   (usual cause: this folder was copied from another one -- a venv does"
  echo "   not travel, everything else in the project does)"
  rm -rf .venv
fi
uv sync

# Where the half of an install that cannot be rebuilt lives: the database and
# every image ever generated. Asked of the app rather than recomputed here so
# there is one definition of it (app/__init__.py).
DATA_IN_USE="$(uv run python -c 'from app import DATA_DIR; print(DATA_DIR)' 2>/dev/null || true)"

# The settings file is written here rather than shipped as a template to copy:
# a file of commented-out lines is not a config, and the one setting that
# matters has a real value already -- the directory in use. Moving the data
# later is then editing a line that is there and correct, not discovering that
# an environment variable exists.
if [ ! -f "$ENV_FILE" ]; then
  cat > "$ENV_FILE" <<ENVFILE
# draftpool settings for this install. Gitignored, so no update touches it.
# A variable exported in your shell overrides anything set here.

# Where the database and every generated image live -- the one part of this
# install that cannot be rebuilt from the package. To move it somewhere an
# update cannot reach, move the data first, then change this line:
#
#     mkdir -p \$HOME/draftpool-data
#     mv "${DATA_IN_USE:-$PWD/data}"/* \$HOME/draftpool-data/
#
DRAFTPOOL_DATA_DIR=${DATA_IN_USE:-$PWD/data}

# Only read by ./scripts/setup.sh --anime-checkpoints, to download the Civitai
# checkpoints. Make a key at civitai.com -> Account settings -> API Keys.
#CIVITAI_TOKEN=
ENVFILE
  echo "-- wrote .env (gitignored): data directory and Civitai token live there"
fi

# An absolute path in .env survives being wrong: the app would create the
# missing directory and report an install with no history at all. Cheaper to
# say so here, where the old path is still on screen.
if [ -n "${DRAFTPOOL_DATA_DIR:-}" ] && [ ! -d "$DRAFTPOOL_DATA_DIR" ]; then
  echo "!! DRAFTPOOL_DATA_DIR points at a directory that does not exist:"
  echo "   $DRAFTPOOL_DATA_DIR"
  echo "   It will be created empty. If your sessions and images are somewhere"
  echo "   else -- a folder you moved, or the previous install -- fix the path"
  echo "   in .env before starting the server."
fi

# Printed on every run because this script is also what you run after an
# update, and this is the moment to notice the directory is not the one you
# meant -- a wrong one is not an error, it is an empty install alongside your
# real history.
if [ -n "$DATA_IN_USE" ]; then
  echo "-- data directory: $DATA_IN_USE"
  # tested by location, not by whether the variable is set: after the file
  # above is written it is always set, and pointing inside the project folder
  # is the thing worth saying something about
  case "$DATA_IN_USE" in
    "$PWD"/*)
      echo "   (inside the project folder. Change DRAFTPOOL_DATA_DIR in .env to"
      echo "    move it somewhere an update cannot touch -- see docs/SETUP.md,"
      echo "    'Where your data lives')" ;;
  esac
fi

echo "== 5/6 Real-ESRGAN (the '2x fast' button, ~50MB)"
# Installed inside the repo at tools/realesrgan, not system-wide: app/esrgan.py
# runs it by that path, so nothing needs to be on PATH. The binary ships
# universal (x86_64 + arm64) and ad-hoc signed, so it runs on Apple Silicon as
# unpacked; only the executable bit is missing (the zip stores it 644).
ESRGAN_URL="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-macos.zip"
ESRGAN_DIR="tools/realesrgan"
# the two weight sets app/esrgan.py actually asks for; an interrupted unzip can
# leave the binary in place without them, which fails only at upscale time
esrgan_complete() {
  [ -x "$ESRGAN_DIR/realesrgan-ncnn-vulkan" ] || return 1
  for m in realesrgan-x4plus realesrgan-x4plus-anime; do
    [ -f "$ESRGAN_DIR/models/$m.bin" ] && [ -f "$ESRGAN_DIR/models/$m.param" ] || return 1
  done
}

if esrgan_complete; then
  echo "-- already present, skipping"
else
  mkdir -p "$ESRGAN_DIR"   # unzip -d will not create missing parent dirs
  zip="$(mktemp -t realesrgan)" || zip="/tmp/realesrgan.$$.zip"
  # deliberately not fatal: a missing binary costs only the "2x fast" button
  # (Upscale uses a Draw Things upscaler instead), and aborting here would skip
  # the model list printed at the end, which is the part a first-time run is
  # actually after. `set -e` would do exactly that, hence the explicit if.
  # not -s: a silent failure here is what makes a missing upscaler mysterious
  if curl -L --fail --retry 2 --progress-bar -o "$zip" "$ESRGAN_URL" \
     && unzip -o -q "$zip" -d "$ESRGAN_DIR"; then
    chmod +x "$ESRGAN_DIR/realesrgan-ncnn-vulkan" 2>/dev/null || true
    # -h exits 255 in this build, so the smoke test looks for its output rather
    # than its status: what matters is that the binary executes at all. The
    # `|| true` keeps that 255 from becoming the pipeline status under pipefail.
    if esrgan_complete \
       && { "$ESRGAN_DIR/realesrgan-ncnn-vulkan" -h 2>&1 || true; } | grep -q "Usage:"; then
      echo "-- installed and runs: $ESRGAN_DIR/realesrgan-ncnn-vulkan"
    else
      echo "!! unpacked to $ESRGAN_DIR but it does not run. Check it by hand:"
      echo "   $ESRGAN_DIR/realesrgan-ncnn-vulkan -h"
      echo "   The '2x fast' button stays disabled until it does."
    fi
  else
    echo "!! Real-ESRGAN download or unpack failed. Continuing without it; the"
    echo "   '2x fast' button stays disabled and everything else works."
    echo "   Retry by re-running this script, or install by hand:"
    echo "   $ESRGAN_URL -> unzip into $ESRGAN_DIR (chmod +x the binary)"
  fi
  rm -f "$zip"
fi

echo "== 6/6 Draw Things API server"
# The HTTP API is off until switched on in the app's own settings. There is no
# launch flag for it (the app binary takes no server options) and the setting
# lives in the app's private config store, so this only reports the state.
if curl -s --max-time 3 -o /dev/null "http://127.0.0.1:7860/sdapi/v1/options"; then
  echo "-- reachable on port 7860, nothing to do"
else
  echo "-- not reachable on port 7860, expected until step 1 below is done"
  echo "   (the toggle persists between launches, so it is a one-time step)"
fi

if [ "$DOWNLOAD_LLMS" = "1" ] || [ "$DOWNLOAD_CHECKPOINTS" = "1" ]; then
  # Add up what this run will actually pull and refuse before starting rather
  # than dying half way through a 20GB download with a full disk.
  echo "== Disk check"
  NEED_GB=0
  if [ "$DOWNLOAD_LLMS" = "1" ]; then
    LLM_NEED_GB=0
    while read -r name gb; do
      [ -z "$name" ] && continue
      if ollama_has "$name"; then
        echo "-- $name is already pulled, not counted"
      else
        LLM_NEED_GB=$((LLM_NEED_GB + gb))
      fi
    done <<EOF
$OLLAMA_MODELS
EOF
    NEED_GB=$((NEED_GB + LLM_NEED_GB))
  fi
  CKPT_NEED_GB=0
  if [ "$DOWNLOAD_CHECKPOINTS" = "1" ]; then
    # not checked the same way: a checkpoint in $DEST may be a partial download
    # that curl will resume, so its presence says nothing about what is left
    while read -r name version gb url; do
      [ -z "$name" ] && continue
      CKPT_NEED_GB=$((CKPT_NEED_GB + gb))
    done <<EOF
$CIVITAI_MODELS
EOF
    NEED_GB=$((NEED_GB + CKPT_NEED_GB))
  fi
  HAVE_GB="$(free_gb "$HOME")"
  echo "-- this run downloads about ${NEED_GB}GB; ${HAVE_GB}GB free on $HOME"
  if [ "$DOWNLOAD_CHECKPOINTS" = "1" ]; then
    echo "-- importing a checkpoint converts it and keeps both copies while that"
    echo "   runs, so leave a few GB of headroom beyond that figure"
  fi
  if [ "$HAVE_GB" -lt "$NEED_GB" ]; then
    echo "!! Not enough free space. Nothing has been downloaded."
    if [ "$DOWNLOAD_CHECKPOINTS" = "1" ]; then
      echo "   Dropping --anime-checkpoints would cut this by ${CKPT_NEED_GB}GB."
    fi
    echo "   Free up space and re-run, or download by hand from the Civitai"
    echo "   pages printed at the end of this run."
    exit 1
  elif [ "$((HAVE_GB - NEED_GB))" -lt 10 ]; then
    echo "!! Only $((HAVE_GB - NEED_GB))GB would be left over. Draw Things needs room to"
    echo "   convert each checkpoint on import, so this may still run out."
  fi

fi

if [ "$DOWNLOAD_LLMS" = "1" ]; then
  if [ "$LLM_NEED_GB" -gt 0 ]; then
    echo "== Ollama LLMs (~${LLM_NEED_GB}GB still to pull)"
  else
    echo "== Ollama LLMs (both already pulled, just verifying)"
  fi
  sleep 3
  ollama pull gemma3:12b     # default: prompt enrichment + image describe
  ollama pull qwen3-vl:8b    # stronger vision: the most accurate reader of an
                             # image, so prefer it for From image..., Finalize+
                             # and Animate's Improve with LLM. Pick in the UI.

fi

if [ "$DOWNLOAD_CHECKPOINTS" = "1" ]; then
  echo "== Civitai anime checkpoints (~22GB, for import into Draw Things)"
  DEST="$HOME/Downloads/draftpool-models"
  mkdir -p "$DEST"
  TOKEN_PARAM=""
  if [ -n "${CIVITAI_TOKEN:-}" ]; then
    TOKEN_PARAM="&token=$CIVITAI_TOKEN"
  else
    echo "!! CIVITAI_TOKEN is not set, so downloads needing a login will 401."
    echo "   Make a key at civitai.com -> Account settings -> API Keys and put it"
    echo "   in .env, where the line is waiting commented out. Or skip the token"
    echo "   and use the page links printed below to download in a browser, where"
    echo "   your Civitai login already covers it."
  fi
  rm -f "$DEST/.failed"
  echo "$CIVITAI_MODELS" | while read -r name version gb url; do
    [ -z "$name" ] && continue
    out="$DEST/$name.safetensors"
    echo "-- $name  ~${gb}GB  ($url)"
    # curl -C - resumes a partial file and no-ops a already-complete one, so
    # there's no need to pre-check the file: safe to just always attempt it,
    # whether this is a fresh run or a retry after an interrupted download.
    if curl -L --fail --retry 3 -C - -o "$out" \
        "https://civitai.com/api/download/models/$version?type=Model&format=SafeTensor$TOKEN_PARAM"; then
      echo "   done"
    else
      echo "   FAILED -- download it by hand instead: $url"
      echo "$name" >> "$DEST/.failed"
    fi
  done
  echo "$CIVITAI_LORAS" | while read -r name version mb url; do
    [ -z "$name" ] && continue
    out="$DEST/$name.safetensors"
    echo "-- $name (LoRA)  ~${mb}MB  ($url)"
    if curl -L --fail --retry 3 -C - -o "$out" \
        "https://civitai.com/api/download/models/$version?type=Model&format=SafeTensor$TOKEN_PARAM"; then
      echo "   done"
    else
      echo "   FAILED -- download it by hand instead: $url"
      echo "$name" >> "$DEST/.failed"
    fi
  done
  if [ -f "$DEST/.failed" ]; then
    echo "Some downloads failed; see the manual links above for those, or re-run with CIVITAI_TOKEN set."
    rm -f "$DEST/.failed"
  fi
  echo "Downloaded to $DEST. Import each file in Draw Things:"
  echo "  checkpoints: Models list -> Manage -> Import Model (conversion runs once)"
  echo "  the LoRA:    LoRA list -> Manage -> Import LoRA"
fi

if [ "$DOWNLOAD_LLMS" != "1" ] && [ "$DOWNLOAD_CHECKPOINTS" != "1" ]; then
  echo
  echo "== No models downloaded (no --llms / --anime-checkpoints given)"
  echo "-- everything below is the full list of what to install, and where each"
  echo "   piece comes from. Re-run with --llms and/or --anime-checkpoints to have this"
  echo "   script fetch what it can."
fi

echo
echo "== What to install by hand"
# unquoted heredocs throughout: ${B}/${N} must expand, and the text carries no
# other $ or backticks
cat <<EOF

Draw Things plus one image model is the minimum. The rest is optional and
additive: a model you skip only costs the action that uses it, which the UI
then hides rather than failing on.

1. Draw Things, free on the Mac App Store, needs macOS 12.4+
     https://apps.apple.com/app/id6444050820
   Then in its settings: API Server on, protocol HTTP, port 7860.

2. Download the models you like inside the app. Supported list, with
   rough on-disk sizes (they vary with the quantisation you pick, and
   the app pulls a few GB of shared text encoders and VAEs alongside):
     ${B}FLUX.1 [schnell]${N}         ~11GB
     ${B}FLUX.2 [klein] 9B${N}         ~9GB
     ${B}Krea 2 Turbo${N}             ~12GB
     ${B}Juggernaut XL Ragnarok${N}    ~5GB
   Any subset works: every model upscales its own images, so none of
   these is required by another.

3. Instruct action: ${B}Qwen Image Edit 2511${N} (~19GB), and its
   ${B}Qwen Image Edit 2511 Lightning 4-Step${N} LoRA (~0.8GB), which makes
   every edit about three times faster. Optional: without it edits run
   at 16 steps.

4. ${B}Upscale needs an upscaler${N} (~34MB): get ${B}4x UltraSharp${N} from
   the app's ${B}Official Upscalers${N} list -- it measured best by a wide
   margin over the alternatives. It enlarges the image before the detail
   pass, and a plain resize is visibly softer, so the button stays disabled
   until one is installed.

5. The three anime checkpoints below are not in that catalogue (optional
   only if you like the style), so they have to be fetched separately
   (~22GB), all Danbooru-tagged SDXL models. Take any subset: each generates
   its own images. Two ways to get them:
     - ./scripts/setup.sh --anime-checkpoints, which needs CIVITAI_TOKEN
     - open the pages below and download in a browser: your Civitai
       login there is enough, no token needed

EOF
echo "$CIVITAI_MODELS" | while read -r name version gb url; do
  [ -z "$name" ] && continue
  echo "     ${B}${name}${N}  ~${gb}GB"
  echo "         $url"
done
cat <<EOF

   Either way that only downloads the files. Import each .safetensors in
   Draw Things via Models list -> Manage -> Import Model; conversion runs
   once per file.

   With them, optionally, two speed LoRAs: ${B}Anima Turbo${N} (~140MB), which
   makes LustrousMix and CyberRealistic Anima about five times faster (a
   darker look, and no negative prompt), and ${B}Wuli Qwen-Image-2512 Turbo${N}
   (~1.1GB), four times faster Qwen Image drafts and finals. Each has a tick
   in the app's settings. --anime-checkpoints fetches both; by hand, from:
EOF
echo "$CIVITAI_LORAS" | while read -r name version mb url; do
  [ -z "$name" ] && continue
  echo "         $url"
done
cat <<EOF
   and import them in Draw Things via LoRA list -> Manage -> Import LoRA.

EOF
# unquoted so the free-space figure expands; the text below has no other
# $ or backticks to worry about
cat <<EOF
6. Animate action: the video model, same catalogue as step 2.
   ${VIDEO_GB}GB, $(free_gb "$HOME")GB free here.
       ${B}Wan 2.1 14B I2V FusionX${N}   17GB, 8 steps, ~12 min a clip
   Camera moves and ambient motion work well; pose changes do not.
   See docs/GUIDE.md for detail.
EOF
cat <<EOF

7. Ollama models, unless --llms already pulled them:
     ollama pull ${B}gemma3:12b${N}     # prompt writing
     ollama pull ${B}qwen3-vl:8b${N}    # reads images: From image..., Finalize+,
                                # and Animate's Improve with LLM

8. Verify:  uv run python scripts/probe.py
10. Run:    uv run uvicorn app.main:app --port 8500
EOF
