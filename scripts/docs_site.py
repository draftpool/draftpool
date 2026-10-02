"""MkDocs hooks for the documentation site (see mkdocs.yml).

The repo's markdown is the source; this only adapts it to the site:

- README.md becomes the home page, minus its "Documentation" section (the
  site's navigation replaces it), with its docs/ links made site-relative.
- LICENSE links go to the file in the GitHub repository.
- Links to what the site does not publish -- docs/decisions/, TODO.md --
  are reduced to their text rather than left broken; a link whose
  text is a path becomes that path in the clone, e.g. `docs/decisions/X.md`.
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


def on_files(files, config):
    readme = (ROOT / "README.md").read_text()
    readme = _DOCS_SECTION.sub("", readme)
    readme = readme.replace("](docs/", "](")
    files.append(File.generated(config, "index.md", content=readme))
    return files


def on_page_markdown(markdown, page, config, files):
    # the license is in the repository, not on the site: link to it there
    if config and config.get("repo_url"):
        markdown = markdown.replace(
            "](LICENSE)", f"]({config['repo_url'].rstrip('/')}/blob/main/LICENSE)")
    markdown = _UNPUBLISHED.sub(_unlink, markdown)
    return markdown.replace("](../README.md", "](index.md")
