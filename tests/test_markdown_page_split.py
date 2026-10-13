# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""Markdown pages obey PAGE_SPLIT_CHARS, by descending the heading hierarchy.

Markdown has six heading levels. `_split_markdown` honoured exactly one (`## `),
so any document organising itself with `#` or `###` collapsed into one page per
`##` section.

Measured on a live corpus (2026-08-16), before this change:

    within policy (<=2000 chars)   1056  56.6%
    2k-10k                          598  32.1%
    10k-50k                         121   6.5%
    >50k                             90   4.8%

43% of pages violated the size policy the plain-text splitter has always
enforced. The worst was a single 51,619-char page holding an entire meeting
transcript beneath eleven `###` timestamp headings.

Why that breaks retrieval rather than merely looking untidy: the embedding
backend truncates at the gateway's embed num_ctx (1024 tokens; a 512-token
model, ~1500 chars, when this was measured). A 51k page's embedding therefore
represents only its FIRST few thousand characters — for that transcript, a
header and a few minutes of small talk. The meeting's content was absent from
page-level search, not out-ranked. Retrieval is page-first, so the transcript's
chunks — all embedded, all present — were unreachable.

Measured on the same corpus: 51% of pages have embeddings that cannot
represent most of their own text, 8% are over 20k chars.

The fix restates no policy: PAGE_SPLIT_CHARS stays the single definition of how
big a page should be, and oversized sections with no headings left fall through
to `_split_plain`, which already enforces it. Size decides WHETHER to descend;
the document's own headings decide WHERE to cut.
"""
from __future__ import annotations

from trellion.live.pageindex_ingestion import (
    PAGE_SPLIT_CHARS,
    _split_markdown,
)


def _para(n: int, tag: str = "x") -> str:
    return f"{tag * 80}\n\n" * n


class TestSizePolicyIsObeyed:
    def test_h2_section_with_h3_children_is_descended(self):
        """The shape that produced the 51k page: one `##` holding many `###`."""
        doc = "## Transcript\n\n" + "".join(
            f"### 00:0{i}:00\n\n{_para(6)}" for i in range(9)
        )
        pages = _split_markdown(doc)
        assert len(pages) > 1
        assert all(len(p) <= PAGE_SPLIT_CHARS for p in pages), [len(p) for p in pages]

    def test_descends_past_h3_when_h4_is_where_the_bulk_is(self):
        doc = "## A\n\n### B\n\n" + "".join(f"#### C{i}\n\n{_para(6)}" for i in range(8))
        assert all(len(p) <= PAGE_SPLIT_CHARS for p in _split_markdown(doc))

    def test_oversized_with_no_headings_left_falls_through_to_plain(self):
        """`_split_plain` is the existing answer for headingless text and
        enforces the same constant — this must not reimplement it."""
        doc = "## Only heading\n\n" + _para(40)
        pages = _split_markdown(doc)
        assert len(pages) > 1
        assert all(len(p) <= PAGE_SPLIT_CHARS for p in pages)


class TestSmallDocumentsAreUntouched:
    """Size decides whether to descend. A document that already fits must come
    out exactly as it did before, or this change rewrites every corpus it
    touches for no reason."""

    def test_small_h2_sections_are_not_descended(self):
        doc = "## One\n\nshort\n\n### Sub\n\nalso short\n\n## Two\n\nshort too\n"
        pages = _split_markdown(doc)
        assert len(pages) == 2
        assert pages[0].startswith("## One")
        assert "### Sub" in pages[0], "a small section keeps its children inline"
        assert pages[1].startswith("## Two")

    def test_text_is_preserved_exactly(self):
        """Splitting must lose nothing — no trimming, no dropped headings."""
        doc = "## A\n\n" + _para(30) + "### B\n\n" + _para(30)
        joined = "".join(_split_markdown(doc))
        assert joined.replace("\n", "") == doc.replace("\n", "")

    def test_document_with_no_headings_is_unchanged_in_content(self):
        doc = "just a paragraph, no headings at all"
        assert _split_markdown(doc) == [doc]


class TestHeadingsAreTheCutPoints:
    def test_pages_start_at_heading_boundaries(self):
        doc = "## Meeting\n\n" + "".join(
            f"### 00:0{i}:00\n\n{_para(6)}" for i in range(6)
        )
        headed = [p for p in _split_markdown(doc) if p.lstrip().startswith("#")]
        assert headed, "descending should cut at headings, not mid-section"

    def test_h1_preamble_survives_as_its_own_page(self):
        """Real Google Docs exports open with an `#` title before any `##`."""
        doc = "# Notes Feb 1\n\n## Meeting\n\nbody\n"
        pages = _split_markdown(doc)
        assert pages[0].startswith("# Notes Feb 1")
