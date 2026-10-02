"""Resolve every `civitai_version_id` in models.yaml against Civitai.

    uv run python scripts/check_civitai_ids.py

Prints one line per model: the Draw Things filename beside the model, version
and base model the id actually points at, so a wrong id is visible instead of
silently crediting somebody else's checkpoint. Models with no id are listed at
the end -- a picture made by one of those is published with no resource at all,
which is what a viewer sees as "We weren't able to detect any resources".

Needs a Civitai API key (the publish one; CIVITAI_TOKEN or civitai.key).
"""
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import civitai  # noqa: E402
from app.registry import load_registry  # noqa: E402


def main() -> int:
    token = civitai.token()
    if token is None:
        print("no Civitai API key: put CIVITAI_TOKEN in .env or civitai.key beside it")
        return 2
    headers = {"Authorization": f"Bearer {token}"}
    registry = load_registry()
    missing = []
    for key, spec in registry.models.items():
        if not spec.civitai_version_id:
            missing.append(key)
            continue
        url = f"https://civitai.com/api/v1/model-versions/{spec.civitai_version_id}"
        resp = httpx.get(url, headers=headers, timeout=30)
        if resp.status_code != 200:
            print(f"{key:26} {spec.civitai_version_id} -> HTTP {resp.status_code}")
            continue
        version = resp.json()
        files = ", ".join(f["name"] for f in version.get("files", [])[:2])
        print(f"{key:26} {spec.dt_filename:44} -> "
              f"{version.get('model', {}).get('name', '?')} / {version.get('name', '?')} "
              f"({version.get('baseModel', '?')})  [{files}]")
    if missing:
        print("\nno civitai_version_id (published with no resource credited): "
              + ", ".join(missing))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
