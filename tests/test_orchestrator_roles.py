# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""RAGOrchestrator.query must forward roles into PRIMARY retrieval.

query() accepts `roles` and must pass it to every execute_* strategy call,
not only to _verify_and_retry, so the primary vector search is never
unrestricted for a read-scope user.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from trellion.live.rag_orchestrator import RAGOrchestrator, RAGResponse

ROLES = ["company", "system", "user:demo-reader"]


def _empty_response() -> RAGResponse:
    return RAGResponse(
        question="q", answer="", evidence=[], graph_context=None, citations=[]
    )


@pytest.fixture
def orch():
    o = RAGOrchestrator.__new__(RAGOrchestrator)
    # Stub every collaborator query() touches; execute_* are real so the
    # test exercises the actual call-site plumbing, but each records roles.
    o.retriever = MagicMock()
    o.verifier = None
    o.video_memory = None
    o.narrower_orchestrator = None
    o.navigator = MagicMock()
    o.navigator.navigate = AsyncMock(return_value=MagicMock(
        candidate_page_ids=[], seed_entities=[]))
    o.retriever.retrieve = AsyncMock(return_value=[])
    o.generator = None  # no evidence → generator not reached anyway
    o._publish_query_event = AsyncMock()
    o._search_video_memory = AsyncMock(return_value=[])
    return o


@pytest.mark.parametrize("strategy", ["hybrid", "lookup", "graph_first"])
async def test_query_forwards_roles_to_primary_retrieval(orch, strategy):
    await orch.query("what is X", strategy=strategy, roles=ROLES)
    assert orch.retriever.retrieve.await_args_list, "retriever was never called"
    for call in orch.retriever.retrieve.await_args_list:
        assert call.kwargs.get("roles") == ROLES, (
            f"[{strategy}] primary retrieval dropped roles: {call}"
        )
