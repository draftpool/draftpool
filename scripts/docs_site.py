"""MkDocs hooks for the documentation site (see mkdocs.yml).

The repo's markdown is the source; this only adapts it to the site:

- README.md becomes the home page, minus its "Documentation" section (the
  site's navigation replaces it), with its docs/ links made site-relative.
- LICENSE links go to the file in the GitHub repository.
- Links to what the site does not publish -- docs/decisions/, TODO.md --
  are reduced to their text rather than left broken; a link whose
  text is a path becomes that path in the clone, e.g. `docs/decisions/X.md`.
- Each page gets its own search-result description. They live here rather
  than as front matter because GitHub shows front matter as a table at the
  top of the file, and the files are read there too. The home page keeps
  `site_description`.
"""

import re
from pathlib import Path

from mkdocs.structure.files import File

ROOT = Path(__file__).resolve().parent.parent

# [text](target) where target leaves the published docs
_UNPUBLISHED = re.compile(
    r"\[([^\]]+)\]\((?:\.\./)?(docs/)?(decisions/[^)]*|TODO\.md|LICENSE)\)"
)


def _unlink(m):
    text, in_docs, target = m.groups()
    if "/" not in text and not text.endswith(".md"):
        return text
    return f"`{'docs/' if in_docs or target.startswith('decisions/') else ''}{target}`"
_DOCS_SECTION = re.compile(r"^## Documentation\n.*?(?=^## )", re.M | re.S)

# what a search engine shows under the page title; keep each under ~160 chars
DESCRIPTIONS = {
    "SETUP.md": "Installing draftpool on a Mac: the setup script, Draw Things and "
                "Ollama, the Civitai models, where your images live, updating "
                "and troubleshooting.",
    "GUIDE.md": "Using draftpool: drafts and prompt variants, finalize, upscale, "
                "edit, combine and animate, LoRAs, anime-model prompting, and "
                "how long each step takes.",
    "DEEP-DIVE.md": "The fine print of draftpool: what Edit can and cannot "
                    "change, upscaling edge cases, the tag vocabulary, and what "
                    "to do when Draw Things or the LLM fails.",
    "ADDING-A-MODEL.md": "Adding a Draw Things model to draftpool: import the "
                         "checkpoint, run the measuring harness, and check its "
                         "steps, sizes and shapes.",
    "CREDITS.md": "The apps, image and video models, LoRAs, upscalers and LLMs "
                  "draftpool is built on, and who made them.",
}


def on_files(files, config):
    readme = (ROOT / "README.md").read_text()
    readme = _DOCS_SECTION.sub("", readme)
    readme = readme.replace("](docs/", "](")
    files.append(File.generated(config, "index.md", content=readme))
    return files


def on_page_markdown(markdown, page, config, files):
    if page and page.file.src_uri in DESCRIPTIONS:
        page.meta.setdefault("description", DESCRIPTIONS[page.file.src_uri])
    # the license is in the repository, not on the site: link to it there
    if config and config.get("repo_url"):
        markdown = markdown.replace(
            "](LICENSE)", f"]({config['repo_url'].rstrip('/')}/blob/main/LICENSE)")
    markdown = _UNPUBLISHED.sub(_unlink, markdown)
    return markdown.replace("](../README.md", "](index.md")
