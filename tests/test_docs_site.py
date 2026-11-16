"""The docs-site hook (scripts/docs_site.py): link rewriting, and the search
titles and descriptions it gives each page."""

from pathlib import Path

import pytest

pytest.importorskip("mkdocs")
from scripts.docs_site import DESCRIPTIONS, TITLES, on_page_markdown  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


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
    config = {"repo_url": "https://github.com/draftpool/draftpool"}
    assert (on_page_markdown("see [LICENSE](LICENSE).", None, config, None)
            == "see [LICENSE](https://github.com/draftpool/draftpool/blob/main/LICENSE).")
    assert rewrite("see [LICENSE](LICENSE).") == "see LICENSE."


def _nav_pages():
    import yaml
    nav = yaml.load((ROOT / "mkdocs.yml").read_text(), Loader=_IgnoreTags)["nav"]
    return [next(iter(item.values())) for item in nav]


class _IgnoreTags(__import__("yaml").SafeLoader):
    """mkdocs.yml carries a !!python tag the safe loader refuses; skip it."""


_IgnoreTags.add_multi_constructor("tag:yaml.org,2002:python/", lambda *a: None)


def test_every_published_page_has_a_search_title_and_description():
    for page in _nav_pages():
        assert page in TITLES, page
        if page != "index.md":  # the home page's is site_description
            assert page in DESCRIPTIONS, page


def test_search_titles_fit_what_a_result_shows():
    # Google cuts a result's title at about 60 characters; every page but the
    # home page gets " - draftpool" added (overrides/main.html)
    for page, title in TITLES.items():
        full = title if page == "index.md" else f"{title} - draftpool"
        assert len(full) <= 65, (page, len(full))


def test_descriptions_fit_what_a_result_shows():
    for page, text in DESCRIPTIONS.items():
        assert len(text) <= 165, (page, len(text))
