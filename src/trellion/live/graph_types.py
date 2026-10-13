# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""Shared graph result types.

These dataclasses are the *interface* between the graph narrower and the
retrieval/generation layer. They live here, not beside any narrower
implementation, so that consumers which merely handle graph results
(``rag_generator``, ``rag_orchestrator``) don't import the implementation:
the Basic tier ships these types and degrades gracefully when no graph
narrower is provided (optional add-on).

The classes themselves are defined in ``trellion_protocols.types`` (the
Apache-2.0 contract package an add-on imports instead of trellion) and
re-exported here, so there is one class object for each.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from trellion_protocols.types import GraphResult, Hypothesis

__all__ = ["GraphNarrower", "GraphResult", "Hypothesis"]


@runtime_checkable
class GraphNarrower(Protocol):
    """What trellion needs from a graph narrower (the ``graph_narrower`` provider role).

    Entity-graph traversal for a question: the result's seed entities and
    hypotheses become the graph context and the candidate pages that scope
    vector retrieval.
    """

    async def navigate(self, query: str, **kwargs: Any) -> GraphResult: ...
