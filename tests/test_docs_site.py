"""The docs-site hook's link rewriting (scripts/docs_site.py)."""

import pytest

pytest.importorskip("mkdocs")
from scripts.docs_site import on_page_markdown  # noqa: E402


def rewrite(md):
    return on_page_markdown(md, None, None, None)


def test_path_link_to_decisions_becomes_the_clone_path():
    assert rewrite("see [decisions/X.md](decisions/X.md).") == "see `docs/decisions/X.md`."
    assert rewrite("([X.md](decisions/X.md))") == "(`docs/decisions/X.md`)"


def test_prose_link_to_unpublished_keeps_only_its_text():
    assert rewrite("[planned work](TODO.md)") == "planned work"


def test_readme_links_point_at_the_home_page():
    assert rewrite("[README](../README.md#setup)") == "[README](index.md#setup)"


def test_published_links_are_untouched():
    md = "[guide](GUIDE.md#shape)"
    assert rewrite(md) == md


def test_license_links_to_the_repository():
    config = {"repo_url": "https://github.com/homiak/draftpool"}
    assert (on_page_markdown("see [LICENSE](LICENSE).", None, config, None)
            == "see [LICENSE](https://github.com/homiak/draftpool/blob/main/LICENSE).")
    assert rewrite("see [LICENSE](LICENSE).") == "see LICENSE."
