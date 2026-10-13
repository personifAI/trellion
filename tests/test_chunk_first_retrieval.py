# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""Chunk-level winners reach the candidate set, not only chunks inside winners.

Retrieval scored PAGES, then searched chunks only WITHIN the pages that won.

A page embedding cannot represent a page larger than the embedding model's
input window: the embedder sends the whole page, the backend takes only the
gateway's embed num_ctx (1024 tokens) and truncates the rest silently. A long page is therefore not
ranked poorly — most of it was never encoded — and its chunks then become
unreachable no matter how well they match.

Chunks are sized well inside that window (CHUNK_SIZE_CHARS), so they
are the one representation that is always faithful.

Observed live (2026-08-16): a 51,619-char meeting transcript lost to the
1,024-char "a summary wasn't produced" page beside it. Asked what was
discussed, the system answered "no summaries were produced" while 4,525
embedded transcript chunks sat unread.

`_merge_chunk_first_pages` searches chunks across the corpus and folds their
pages in, scored by their best chunk. It is ADDITIVE: a page that won on its
own keeps its score and its place, so this can only add reachability.
"""
from __future__ import annotations

import pytest

from trellion.live.pageindex_retriever import PageIndexRetriever


class FakeStore:
    """Minimal store double. Records what it was asked for so scoping and
    role propagation can be asserted, not assumed."""

    def __init__(self, chunks=None, pages_by_id=None, fail=False):
        self._chunks = chunks or []
        self._pages_by_id = pages_by_id or {}
        self._fail = fail
        self.chunk_calls: list[dict] = []

    async def search_chunks(self, embedding, top_k=10, page_ids=None, roles=None):
        self.chunk_calls.append({"top_k": top_k, "page_ids": page_ids, "roles": roles})
        if self._fail:
            raise RuntimeError("chunk search exploded")
        return list(self._chunks)


def _retriever(store):
    r = PageIndexRetriever.__new__(PageIndexRetriever)
    r.store = store
    return r


async def _fetch_pages_stub(store):
    async def _inner(embedding, page_ids, top_k=0, roles=None, doc_id_filter=None):
        return [dict(store._pages_by_id[p]) for p in page_ids if p in store._pages_by_id]
    return _inner


class TestChunkWinnersBecomeReachable:
    @pytest.mark.asyncio
    async def test_page_found_only_by_chunk_is_added(self):
        """The transcript case: the page loses at page level, one of its
        chunks is an excellent match, and it must still be retrievable."""
        store = FakeStore(
            chunks=[{"chunk_id": "c1", "page_id": "transcript", "text": "…", "score": 0.91}],
            pages_by_id={"transcript": {"page_id": "transcript", "doc_id": "d1", "page_no": 3, "text": "…", "score": 0.20}},
        )
        r = _retriever(store)
        r._retrieve_pages_by_ids = await _fetch_pages_stub(store)

        pages = await r._merge_chunk_first_pages(
            [0.1], [{"page_id": "boilerplate", "doc_id": "d1", "page_no": 2, "score": 0.55}], top_k=5
        )

        ids = [p["page_id"] for p in pages]
        assert "transcript" in ids, "a page found only by its chunks must be reachable"
        assert ids[0] == "transcript", "and its best-chunk score should rank it"

    @pytest.mark.asyncio
    async def test_existing_page_is_rescored_never_demoted(self):
        store = FakeStore(chunks=[{"page_id": "p1", "score": 0.9}])
        r = _retriever(store)
        pages = await r._merge_chunk_first_pages([0.1], [{"page_id": "p1", "score": 0.4}], top_k=5)
        assert pages[0]["score"] == pytest.approx(0.9)

    @pytest.mark.asyncio
    async def test_a_weaker_chunk_does_not_lower_a_strong_page(self):
        """MAX, not replace — being found twice must never cost a page."""
        store = FakeStore(chunks=[{"page_id": "p1", "score": 0.2}])
        r = _retriever(store)
        pages = await r._merge_chunk_first_pages([0.1], [{"page_id": "p1", "score": 0.8}], top_k=5)
        assert pages[0]["score"] == pytest.approx(0.8)


class TestScopingIsRespected:
    @pytest.mark.asyncio
    async def test_graph_scope_confines_chunk_search(self):
        """When the graph narrowed to candidate pages, this widens recall
        WITHIN that scope — it must not escape it."""
        store = FakeStore(chunks=[])
        r = _retriever(store)
        await r._merge_chunk_first_pages(
            [0.1], [], top_k=5, candidate_page_ids=["a", "b"]
        )
        assert store.chunk_calls[0]["page_ids"] == ["a", "b"]

    @pytest.mark.asyncio
    async def test_roles_are_passed_through(self):
        """Audience filtering is not optional on this path."""
        store = FakeStore(chunks=[])
        r = _retriever(store)
        await r._merge_chunk_first_pages([0.1], [], top_k=5, roles=["company"])
        assert store.chunk_calls[0]["roles"] == ["company"]

    @pytest.mark.asyncio
    async def test_unscoped_search_stays_unscoped(self):
        store = FakeStore(chunks=[])
        r = _retriever(store)
        await r._merge_chunk_first_pages([0.1], [], top_k=5)
        assert store.chunk_calls[0]["page_ids"] is None


class TestFailureIsNonFatal:
    @pytest.mark.asyncio
    async def test_chunk_search_error_leaves_page_results_intact(self):
        """A recall improvement must not become a new way for queries to
        fail."""
        store = FakeStore(fail=True)
        r = _retriever(store)
        original = [{"page_id": "p1", "score": 0.5}]
        assert await r._merge_chunk_first_pages([0.1], list(original), top_k=5) == original

    @pytest.mark.asyncio
    async def test_no_chunks_returns_pages_unchanged(self):
        store = FakeStore(chunks=[])
        r = _retriever(store)
        original = [{"page_id": "p1", "score": 0.5}]
        assert await r._merge_chunk_first_pages([0.1], list(original), top_k=5) == original

    @pytest.mark.asyncio
    async def test_chunks_without_a_page_id_are_skipped(self):
        store = FakeStore(chunks=[{"page_id": "", "score": 0.99}, {"score": 0.98}])
        r = _retriever(store)
        original = [{"page_id": "p1", "score": 0.5}]
        assert await r._merge_chunk_first_pages([0.1], list(original), top_k=5) == original
