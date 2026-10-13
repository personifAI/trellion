# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""Persist a batch build's output as a snapshot.

The snapshot tables are the ONE graph storage (gateway proposal 27's
consolidation): the live stack writes them via ``SnapshotOrchestrator``
(slice of the live graph → build stages 4-7 → store), and this module
gives the offline batch pipeline (``trellion build``: docs → stages 1-8)
the same destination. The CSV the pipeline also emits is a demoted
debug/fixture artifact, not a storage format.

The write sequence mirrors ``SnapshotOrchestrator.consolidate`` exactly —
``SnapshotStore.write_concepts`` is documented as consuming precisely
what ``trellion.build.canonicalize`` produces, so there is no adapter
layer here on purpose.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog
from sqlalchemy.ext.asyncio import AsyncEngine

from .store import SnapshotStore, describe_failure

logger = structlog.get_logger(__name__)

SLICE_KIND_BATCH = "batch_build"


async def emit_build_snapshot(
    engine: AsyncEngine,
    canonical: list[Any],
    dag_edges: list[Any],
    communities: list[Any],
    *,
    input_dir: Path | str | None = None,
    label: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Write a batch build's stages-4-7 output as a complete snapshot.

    Returns ``(snapshot_id, stats)``. On any write failure the snapshot row
    is marked failed (same contract as the orchestrator) and the exception
    re-raised — a half-written snapshot must never look queryable.
    """
    store = SnapshotStore(engine)
    snapshot_id = await store.create_snapshot(
        slice_kind=SLICE_KIND_BATCH,
        slice_spec={"input_dir": str(input_dir) if input_dir else None},
        label=label,
    )
    try:
        concept_id_map = await store.write_concepts(snapshot_id, canonical)
        edge_count = await store.write_dag_edges(snapshot_id, dag_edges, concept_id_map)
        community_count = await store.write_communities(
            snapshot_id, communities, concept_id_map
        )
        stats = {
            "source": SLICE_KIND_BATCH,
            "concepts": len(canonical),
            "dag_edges": edge_count,
            "communities": community_count,
        }
        await store.mark_complete(snapshot_id, stats)
        logger.info("snapshot.batch_emitted", snapshot_id=snapshot_id, **stats)
        return snapshot_id, stats
    except Exception as exc:
        logger.exception("snapshot.batch_emit_failed", snapshot_id=snapshot_id)
        await store.mark_failed(snapshot_id, reason=describe_failure(exc))
        raise
