"""Writing entries into config/models.private.yaml, and the measured settings
an entry can take from config/tuned_models.yaml.

Shared by scripts/add_civitai_model.py (a model the user imported) and
app/migrate.py (a model an earlier release shipped and this one does not).
The file itself is the user's: setup and updates never overwrite it, so
everything here appends.
"""

from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from pathlib import Path

import yaml

from .registry import CONFIG_PATH

TUNED_PATH = CONFIG_PATH.parent / "tuned_models.yaml"
PROFILES_PATH = CONFIG_PATH.parent / "base_models.yaml"

HEADER = """\
# Models this checkout adds to the ones that ship.
#
# Same format as the `models` section of models.yaml, loaded after it
# (app/registry.py, private_path). scripts/add_civitai_model.py appends to it,
# and so does the startup migration for a model an earlier release shipped;
# entries can also be written by hand. It is never published and never
# overwritten by an update, so your models survive one.

models:
"""


# what the comment above an entry says when a script or startup wrote it from
# a profile or the tuned file: settings nobody measured on that file here
ADDED_MARKS = ("by scripts/add_civitai_model.py", "on startup: installed in Draw Things",
               "given in the settings dialog")


def added_keys(path: Path) -> set[str]:
    """The entries of a models.private.yaml that were written for the user, not measured."""
    if not path.is_file():
        return set()
    added, pending = set(), ""
    for line in path.read_text().splitlines():
        m = re.match(r"^  ([a-z0-9][a-z0-9-]*):\s*$", line)
        if m:
            if any(mark in pending for mark in ADDED_MARKS):
                added.add(m.group(1))
            pending = ""
        elif re.match(r"^  #", line):
            pending += line
    return added


def load_tuned(path: Path = TUNED_PATH) -> dict[int, dict]:
    """Settings measured on exact checkpoints, by Civitai version id."""
    if not path.is_file():
        return {}
    return {int(k): v for k, v in ((yaml.safe_load(path.read_text()) or {}).get("tuned") or {}).items()}


# precision and packaging words a download or an import adds to the name
_NAME_NOISE = re.compile(r"[._\- ](f16|f32|fp16|fp32|bf16|q8p|q6p|q5p|q4p|i8x|pruned)$")


def file_digest(filename: str) -> str:
    """What config/tuned_models.yaml records of a measured checkpoint's file.

    A hash of the name with case, extension, precision words and every
    separator stripped, so the shipped file recognises an import without
    naming the model: Draw Things keeps no Civitai id and converts the file,
    so its name is all an install has to go on. The stripping is broad
    because Civitai's names ("Some Mix v7.777.safetensors") reach Draw Things
    as an import name it rewrites itself.
    """
    name = filename.lower()
    for ext in (".ckpt", ".safetensors"):
        name = name.removesuffix(ext)
    while (m := _NAME_NOISE.search(name)) is not None:
        name = name[: m.start()]
    return hashlib.sha256(re.sub(r"[^a-z0-9]", "", name).encode()).hexdigest()


def fingerprint(path: Path) -> str | None:
    """The weights' identity in a Draw Things .ckpt, whatever the file is called.

    A hash of the small tensors (norms and biases, a few hundred of them) by
    name. Draw Things converts a download the same way every time: two imports
    of one Civitai file months apart came out byte-identical, on an Anima and
    an SDXL checkpoint, while every other checkpoint on the machine hashed
    differently. Reading only the small ones takes a tenth of a second where
    hashing the whole file takes twenty, and survives Draw Things changing
    its file layout as long as the weights stay the same. A quantised import
    has other weights, and so another fingerprint. None when the file is not
    a Draw Things model.
    """
    import sqlite3

    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            rows = con.execute("SELECT name, data FROM tensors WHERE length(data) <= 65536 "
                               "ORDER BY name").fetchall()
        finally:
            con.close()
    except sqlite3.Error:
        return None
    if not rows:
        return None
    h = hashlib.sha256()
    for name, data in rows:
        h.update(name.encode())
        h.update(data)
    return h.hexdigest()


def imported_models(models_dirs: list[str]) -> list[dict]:
    """What Draw Things' own list of imported models says, from the first dir that has one."""
    for d in models_dirs:
        f = Path(d).expanduser() / "custom.json"
        if f.is_file():
            return [e for e in json.loads(f.read_text()) if isinstance(e, dict) and e.get("file")]
    return []


def civitai_version(version_id: int, timeout: float = 30) -> dict:
    """The version's base model, names and creator, from Civitai's public API."""
    def get(url: str) -> dict:
        req = urllib.request.Request(url, headers={"User-Agent": "draftpool-add-model"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)

    v = get(f"https://civitai.com/api/v1/model-versions/{version_id}")
    creator = None
    try:
        creator = get(f"https://civitai.com/api/v1/models/{v['modelId']}")["creator"]["username"]
    except Exception:  # noqa: BLE001 - the credit line is a nicety
        pass
    return {"id": version_id, "base_model": v.get("baseModel"),
            "model": v.get("model", {}).get("name"), "version": v.get("name"),
            "creator": creator}


def version_from_air(air: str) -> int:
    """The version id in a Civitai AIR.

    "urn:air:sdxl:checkpoint:civitai:1307857@3313575" is model 1307857,
    version 3313575; the page's copy button may add "+FILE_ID" after it.
    Raises ValueError for anything else.
    """
    m = re.search(r"civitai:(\d+)@(\d+)", air.strip())
    if not m:
        raise ValueError(f"not a Civitai AIR (civitai:MODEL@VERSION): {air!r}")
    return int(m.group(2))


def civitai_version_by_hash(sha256: str, timeout: float = 30) -> int:
    """The version a downloaded file is, from its SHA256 (Civitai's by-hash lookup)."""
    url = f"https://civitai.com/api/v1/model-versions/by-hash/{sha256.strip()}"
    req = urllib.request.Request(url, headers={"User-Agent": "draftpool-add-model"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return int(json.load(r)["id"])


def version_from_ref(ref: str, by_hash=None) -> int:
    """A Civitai version from what its page shows, in any of its forms.

    A version id ("3313575"), the page's address ("...?modelVersionId=3313575"),
    the AIR ("urn:air:sdxl:checkpoint:civitai:1307857@3313575+3198988") or the
    SHA256 of the downloaded file, which Civitai is asked about (`by_hash`,
    civitai_version_by_hash by default). Raises ValueError for anything else.
    """
    ref = ref.strip()
    if re.fullmatch(r"\d+", ref):
        return int(ref)
    if m := re.search(r"modelVersionId=(\d+)", ref):
        return int(m.group(1))
    if "civitai:" in ref:
        return version_from_air(ref)
    if re.fullmatch(r"[0-9a-fA-F]{64}", ref):
        try:
            return (by_hash or civitai_version_by_hash)(ref)
        except Exception as exc:  # noqa: BLE001 - not found and offline read the same to the user
            raise ValueError(f"Civitai found no version with that SHA256 ({exc})") from exc
    raise ValueError("expected a Civitai version id, AIR, page address or SHA256")


def replace_entry(path: Path, key: str, text: str) -> None:
    """Put `text` (an entry_block) where entry `key` and its comment stand.

    The entry runs from the comment lines straight above its key to the next
    line at the entries' indent; the next entry's comments belong to it.
    Raises KeyError when the file has no such entry.
    """
    lines = path.read_text().split("\n")
    at = next((i for i, l in enumerate(lines) if re.fullmatch(rf"  {re.escape(key)}:\s*", l)), None)
    if at is None:
        raise KeyError(key)
    start = at
    while start > 0 and re.match(r"^  #", lines[start - 1]):
        start -= 1
    end = at + 1
    while end < len(lines) and not re.match(r"^  \S", lines[end]) and not re.match(r"^\S", lines[end]):
        end += 1
    # blank lines before the next entry stay between the two
    while end > at + 1 and not lines[end - 1].strip():
        end -= 1
    path.write_text("\n".join(lines[:start] + text.rstrip("\n").split("\n") + lines[end:]))


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "model"


def load_profiles(path: Path = PROFILES_PATH) -> dict:
    """The model classes, by name: config/base_models.yaml's profiles."""
    if not path.is_file():
        return {}
    return (yaml.safe_load(path.read_text()) or {}).get("profiles") or {}


def class_of(imported: dict, profiles: dict) -> str | None:
    """The profile an import fits by what Draw Things says of it alone, or None.

    The architecture picks the class; where two classes share one
    (Illustrious and plain SDXL), the import's name decides by the profile's
    `name_hints`, and the class without hints takes the rest -- a plain SDXL
    prose setup works on any SDXL, an Illustrious one does not.
    """
    hits = [k for k, p in profiles.items() if p["draw_things_version"] == imported.get("version")]
    if len(hits) <= 1:
        return hits[0] if hits else None
    name = f"{imported.get('file', '')} {imported.get('name', '')}".lower()
    hinted = [k for k in hits if any(h in name for h in profiles[k].get("name_hints") or ())]
    if hinted:
        return hinted[0] if len(hinted) == 1 else None
    plain = [k for k in hits if not profiles[k].get("name_hints")]
    return plain[0] if len(plain) == 1 else None


class _Block(str):
    pass


def dump_entry(entry: dict) -> str:
    """An entry as YAML, multi-line text as | blocks the way the shipped file has it."""
    class D(yaml.SafeDumper):
        pass

    D.add_representer(_Block, lambda d, s: d.represent_scalar("tag:yaml.org,2002:str", s, style="|"))

    def blocks(v):
        if isinstance(v, dict):
            return {k: blocks(x) for k, x in v.items()}
        if isinstance(v, list):
            return [blocks(x) for x in v]
        return _Block(v) if isinstance(v, str) and "\n" in v else v

    return yaml.dump(blocks(entry), Dumper=D, sort_keys=False, allow_unicode=True, width=88)


def entry_block(key: str, entry: dict, comment: list[str]) -> str:
    """`key: entry` indented under `models:`, after its comment lines."""
    body = "".join("    " + l + "\n" if l else "\n" for l in dump_entry(entry).splitlines())
    return "\n".join(f"  # {c}" if c else "  #" for c in comment) + f"\n  {key}:\n" + body


def append_entry(path: Path, key: str, text: str) -> None:
    """Add one entry's text under `models:`, creating the file if need be.

    Raises ValueError when the file already has `key`.
    """
    current = path.read_text() if path.is_file() else HEADER
    if re.search(rf"(?m)^  {re.escape(key)}:\s*$", current):
        raise ValueError(f"{path.name} already has a model {key!r}; choose another with --key")
    current = re.sub(r"(?m)^models:[ \t]*\{\}[ \t]*$", "models:", current)
    if not re.search(r"(?m)^models:[ \t]*$", current):
        current = current.rstrip("\n") + "\n\nmodels:\n"
    path.write_text(current.rstrip("\n") + "\n\n" + text)
