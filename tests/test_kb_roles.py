# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""KB role segmentation — retrieval must never widen the caller's scope.

Offline: the store is mocked; these pin the *plumbing* (roles reaching
every store call), not the SQL. The graph-backfill case pins that when
graph filtering leaves fewer than top_k pages, the backfill still applies the
roles filter.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from trellion.live.pageindex_retriever import PageIndexRetriever

ROLES = ["company", "system", "user:demo-reader"]


@pytest.fixture
def retriever():
    store = MagicMock()
    store.search_pages = AsyncMock(return_value=[])
    store.search_chunks = AsyncMock(return_value=[])
    store.get_page = AsyncMock(return_value={"doc_id": "doc-1"})
    embedder = MagicMock()
    return PageIndexRetriever(store, embedder, graph_ranking=False)


async def test_backfill_stays_role_scoped(retriever):
    # Graph candidates resolve to doc-1 but yield no filtered hits →
    # the under-top_k backfill fires. It must carry the caller's roles.
    await retriever._retrieve_pages_by_ids(
        query_embedding=[0.0] * 3,
        candidate_page_ids=["page-1"],
        top_k=5,
        roles=ROLES,
    )
    calls = retriever.store.search_pages.await_args_list
    assert calls, "expected search_pages to be called"
    for call in calls:
        assert call.kwargs.get("roles") == ROLES, (
            f"search_pages called without the caller's roles: {call}"
        )


async def test_retrieve_pages_passes_roles(retriever):
    await retriever.retrieve_pages([0.0] * 3, top_k=5, roles=ROLES)
    call = retriever.store.search_pages.await_args
    assert call.kwargs.get("roles") == ROLES


async def test_empty_candidates_fall_back_role_scoped(retriever):
    retriever.store.get_page = AsyncMock(return_value=None)  # no doc resolution
    await retriever._retrieve_pages_by_ids(
        query_embedding=[0.0] * 3,
        candidate_page_ids=["page-x"],
        top_k=5,
        roles=ROLES,
    )
    for call in retriever.store.search_pages.await_args_list:
        assert call.kwargs.get("roles") == ROLES
