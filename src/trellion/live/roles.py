# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""What trellion needs from each provider role an add-on may fill.

Every optional capability is reached by a role name (the constants in
:mod:`trellion.live._intel`) and used through one of these small Protocols.
Trellion calls only the methods named here; whoever fills the role may do
more. With no provider for a role, trellion runs without that capability.
"""

from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable


@runtime_checkable
class BackgroundWorker(Protocol):
    """A started/stopped background loop (relation classifier and extractor, entity embedder, graph metrics)."""

    async def start(self, *args: Any, **kwargs: Any) -> None: ...

    async def stop(self) -> None: ...


@runtime_checkable
class ClusterBuilder(Protocol):
    """Builds entity clusters on demand; ``close`` releases what it holds."""

    async def run_clustering(self, *args: Any, **kwargs: Any) -> Any: ...

    async def close(self) -> None: ...


@runtime_checkable
class GraphRanker(Protocol):
    """Expands seed entities along the graph and boosts page scores with them."""

    async def get_graph_boosted_entities(self, *, query: str, seed_entity_ids: list, store: Any, top_k: int = 20) -> list: ...

    def classify_query_intent(self, query: str) -> str: ...

    async def boost_page_scores(self, *, base_results: list, graph_entities: list, store: Any) -> list: ...


@runtime_checkable
class QueryRouter(Protocol):
    """Strategy-based retrieval over the memory system, per user."""

    async def init_nats(self) -> Any: ...

    async def retrieve_iterative(self, *, query: str, session: Optional[Any], max_iterations: int, confidence_threshold: float) -> list: ...

    async def retrieve_progressive(self, *, query: str, session: Optional[Any], callback: Any = None) -> dict: ...

    async def retrieve_hybrid(self, *, query: str, session: Optional[Any], callback: Any = None, enable_iteration: bool = True, confidence_threshold: float = 0.6) -> tuple: ...

    def get_performance_stats(self) -> dict: ...
