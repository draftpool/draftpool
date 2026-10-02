"""Build the Danbooru tag vocabulary the tag-prompted models were trained on.

    uv run python scripts/fetch_tags.py [--out config/danbooru_tags.json]

LustrousMix, BAS+ and CyberRealistic are all Danbooru-trained, so
the tags they know are the tags Danbooru uses and `post_count` is how many
images the model actually saw with each one. That makes the tag list a
vocabulary the app can check a prompt against -- which catches far more than a
hand-written blocklist ever will: invented character names, typos, and the
Stable Diffusion folklore that reads as haze to these checkpoints.

Two things this records beyond the names:

  post_count  how strongly the tag is learned. "cowboy_shot" has 842k posts,
              "artstation" has 1 and "highly_detailed" has 0. A tag with no
              posts is a tag the model never saw.
  aliases     the canonical spelling a non-canonical one redirects to. Danbooru
              stores "mist" as an alias of "fog" and "aerial_view" as an alias
              of "from_above", so a prompt using either spends a tag on
              something the model was never trained on. Swapping them is free
              accuracy, which is why they are shipped alongside the names.
  other_names the words people call a tag that are not spellings of it. These
              come from the tag's wiki page, not the alias table, and they are
              a different thing: the alias table has "mist" pointing at
              "fog", but only the wiki knows that "1boy" is what
              somebody means by "male" and "1girl" is what they mean by
              "female" or "woman". Measured coverage is about a quarter of the
              common general tags, so this supplements the vocabulary rather
              than replacing anything (docs/decisions/VOCABULARY.md).

The snapshot lives in config/ rather than data/ because it ships with the app:
data/ is gitignored, and without the file the vocabulary checks simply switch
off (see app/vocab.py) rather than breaking the composer.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import httpx

API = "https://danbooru.donmai.us"
OUT_DEFAULT = Path(__file__).resolve().parent.parent / "config" / "danbooru_tags.json"

# Category -> the post count below which a tag is not worth shipping. General
# tags are the working vocabulary and get the lowest bar; characters are
# numerous and individually rare (the real Castle in the Sky tag, "sheeta", has
# 239 posts) so they get a lower one still; meta tags are few and mostly
# uninteresting, so only the common ones are kept.
CATEGORIES = {
    0: ("general", 200),
    4: ("character", 100),
    3: ("copyright", 200),
    5: ("meta", 500),
}

# Artist tags are deliberately absent: every model's prompt_guidance in
# config/models.yaml tells the LLM not to emit artist names, so shipping 100k+
# of them would only add weight and invite the thing the guidance forbids.

PAGE = 1000
PAUSE = 0.4  # be a polite API citizen; this runs once, not per request

# other_names is mostly Japanese, Korean and Chinese, which nobody is typing
# into an English prompt box; keeping only the latin ones is what stops the
# snapshot doubling in size for entries that can never match.
LATIN_NAME = re.compile(r"^[a-z0-9][a-z0-9 '&.-]*$")


def get(client: httpx.Client, path: str, **params) -> list[dict]:
    resp = client.get(f"{API}/{path}", params=params, timeout=60)
    resp.raise_for_status()
    return resp.json()


def fetch_category(client: httpx.Client, cat: int, floor: int) -> dict[str, int]:
    """Every tag in one category with more than `floor` posts."""
    out: dict[str, int] = {}
    page = 1
    while True:
        rows = get(
            client,
            "tags.json",
            **{
                "search[category]": cat,
                "search[post_count_gt]": floor,
                "search[order]": "count",
                "limit": PAGE,
                "page": page,
            },
        )
        if not rows:
            break
        for r in rows:
            # deprecated tags still exist in the training data, but Danbooru
            # has stopped using them, so they are recorded as known-but-weak
            # rather than dropped: the model did see them.
            out[r["name"]] = r["post_count"]
        print(f"  {CATEGORIES[cat][0]}: page {page}, {len(out)} tags", flush=True)
        page += 1
        time.sleep(PAUSE)
    return out


def fetch_aliases(client: httpx.Client, known: set[str]) -> dict[str, str]:
    """Active alias -> canonical name, for canonical names we shipped.

    An alias pointing at a tag too rare to ship is not worth the bytes: the
    suggestion would swap one tag the model barely knows for another.
    """
    out: dict[str, str] = {}
    page = 1
    while True:
        rows = get(
            client,
            "tag_aliases.json",
            **{"search[status]": "active", "limit": PAGE, "page": page},
        )
        if not rows:
            break
        for r in rows:
            if r["consequent_name"] in known and r["antecedent_name"] not in known:
                out[r["antecedent_name"]] = r["consequent_name"]
        print(f"  aliases: page {page}, {len(out)} kept", flush=True)
        page += 1
        time.sleep(PAUSE)
    return out


def fetch_other_names(client: httpx.Client, known: set[str]) -> dict[str, list[str]]:
    """Tag -> the other words people call it, from the tag wiki pages.

    `search[other_names_present]` narrows this to the pages that carry any,
    which is the difference between walking the whole wiki and walking a
    fraction of it. There is no way to ask for a list of titles --
    search[title_in] is accepted and then ignored, which is worse than an
    error -- so the filtering to tags we ship is done here.

    Only names that are latin, are not already a tag or an alias, and belong to
    a tag we ship are kept -- the wiki is full of romanised character-name
    variants ("undernightinbirth", "azkiart") that would bloat the file and
    can never be the answer to a prompt word.
    """
    out: dict[str, list[str]] = {}
    page = 1
    while True:
        rows = get(
            client, "wiki_pages.json",
            limit=PAGE, page=page, only="title,other_names",
            **{"search[other_names_present]": "true"},
        )
        if not rows:
            break
        for r in rows:
            title = r.get("title") or ""
            if title not in known:
                continue
            names = [
                n.lower() for raw in (r.get("other_names") or [])
                if LATIN_NAME.match(n := raw.replace("_", " ").lower().strip())
                and n.replace(" ", "_") != title
            ]
            if names:
                out[title] = sorted(dict.fromkeys(names))
        print(f"  wiki: page {page}, {len(out)} tags with other names", flush=True)
        page += 1
        time.sleep(PAUSE)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument(
        "--skip-aliases",
        action="store_true",
        help="names only; leaves any existing alias map in place",
    )
    ap.add_argument(
        "--wiki-only",
        action="store_true",
        help="refresh only other_names, merging into the existing snapshot. "
             "The tag list and the aliases move slowly; the wiki pass is the "
             "slow half, so it is worth being able to run it alone.",
    )
    args = ap.parse_args()

    if args.wiki_only:
        if not args.out.is_file():
            print(f"{args.out} does not exist; run without --wiki-only first", file=sys.stderr)
            return 1
        payload = json.loads(args.out.read_text())
        with httpx.Client(headers={"User-Agent": "draftpool-tag-vocab/1.0"}) as client:
            print("fetching tag wiki other names...", flush=True)
            payload["other_names"] = fetch_other_names(client, set(payload["tags"]))
        payload["fetched_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        # the same compact form the full run writes, so refreshing one half
        # does not rewrite every line of the file
        args.out.write_text(json.dumps(payload, separators=(",", ":")))
        print(f"wrote {args.out}: {len(payload['other_names'])} tags with other names")
        return 0

    tags: dict[str, int] = {}
    with httpx.Client(headers={"User-Agent": "draftpool-tag-vocab/1.0"}) as client:
        for cat, (name, floor) in CATEGORIES.items():
            print(f"fetching {name} tags (>{floor} posts)...", flush=True)
            tags.update(fetch_category(client, cat, floor))
        aliases: dict[str, str] = {}
        if not args.skip_aliases:
            print("fetching aliases...", flush=True)
            aliases = fetch_aliases(client, set(tags))
        print("fetching tag wiki other names...", flush=True)
        other_names = fetch_other_names(client, set(tags))

    if args.skip_aliases and args.out.is_file():
        aliases = json.loads(args.out.read_text()).get("aliases", {})

    payload = {
        "source": API,
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "thresholds": {name: floor for _, (name, floor) in CATEGORIES.items()},
        "tags": dict(sorted(tags.items(), key=lambda kv: -kv[1])),
        "aliases": dict(sorted(aliases.items())),
        "other_names": dict(sorted(other_names.items())),
    }
    args.out.write_text(json.dumps(payload, separators=(",", ":")))
    print(
        f"\nwrote {args.out} -- {len(tags)} tags, {len(aliases)} aliases, "
        f"{args.out.stat().st_size / 1024:.0f}KB"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
