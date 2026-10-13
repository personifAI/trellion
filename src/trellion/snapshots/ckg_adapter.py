# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""Materialise a snapshot as a CKG for the T1-T5 query router.

The query side (``trellion.query.answer.answer_query``) operates on a
plain in-memory ``CKG`` dataclass with ``{concepts, label_to_id,
forward, backward, taxonomy_members}`` — originally built from the CSV
the build pipeline emits.

A snapshot in Postgres has the same shape, just keyed by UUID:
- ``trellion_concepts`` rows → CKG.concepts
- ``trellion_dag_edges`` rows → forward/backward + concept.dependencies
- concept ``type`` (or community membership) → taxonomy

This adapter reads the snapshot and builds a CKG so the existing T1–T5
router runs unchanged. Snapshots are immutable, so the resulting CKG is
safe to cache by snapshot_id.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

if TYPE_CHECKING:
    from trellion.query.graph import CKG

logger = structlog.get_logger(__name__)


def _norm(label: str) -> str:
    """Match trellion.query.graph._norm — single-pass lowercase + collapse-ws."""
    return " ".join(str(label).lower().split())


async def load_ckg_from_snapshot(engine: AsyncEngine, snapshot_id: str) -> "CKG":
    """Build a CKG from a snapshot's concepts + dag_edges + communities.

    Args:
        engine: trellion's shared async engine
        snapshot_id: UUID of the snapshot row in trellion_snapshots

    Returns:
        Populated CKG ready for ``trellion.query.answer.answer_query``.
    """
    from trellion.query.graph import CKG, Concept

    g = CKG()

    # Map UUID concept_id (str) → sequential int cid (1-indexed) so the
    # CKG int-cid contract holds. Stored on the concept's `extra` for
    # provenance back to the DB row if a caller wants it.
    uuid_to_cid: dict[str, int] = {}

    async with engine.connect() as conn:
        # 1. Concepts
        concept_rows = (await conn.execute(
            text(
                "SELECT id::text, canonical_name, type "
                "FROM trellion_concepts WHERE snapshot_id = CAST(:id AS uuid) "
                "ORDER BY canonical_name"
            ),
            {"id": snapshot_id},
        )).fetchall()

        for i, (uuid_str, name, ctype) in enumerate(concept_rows):
            cid = i + 1
            uuid_to_cid[uuid_str] = cid
            concept = Concept(
                cid=cid,
                label=name,
                dependencies=[],
                taxonomy=ctype or "UNASSIGNED",
                relation_type="PREREQ",
                source_chunks=[],
            )
            g.concepts[cid] = concept
            g.label_to_id[_norm(name)] = cid
            g.taxonomy_members[concept.taxonomy].append(cid)

        # 2. DAG edges → forward/backward + concept.dependencies
        edge_rows = (await conn.execute(
            text(
                "SELECT src_concept_id::text, dst_concept_id::text, rel_type "
                "FROM trellion_dag_edges WHERE snapshot_id = CAST(:id AS uuid)"
            ),
            {"id": snapshot_id},
        )).fetchall()

        for src_uuid, dst_uuid, rel_type in edge_rows:
            src_cid = uuid_to_cid.get(src_uuid)
            dst_cid = uuid_to_cid.get(dst_uuid)
            if src_cid is None or dst_cid is None:
                continue
            g.concepts[src_cid].dependencies.append(dst_cid)
            g.forward[src_cid].append(dst_cid)
            g.backward[dst_cid].append(src_cid)
            # Capture rel_type on the source concept when it's not the default
            if rel_type and rel_type != "PREREQ":
                g.concepts[src_cid].relation_type = rel_type

        # 3. Communities — override taxonomy when level-0 community exists.
        # A concept can belong to multiple communities at different levels;
        # we use the lowest-level (most-specific) community's name as the
        # taxonomy, falling back to the concept type when no community.
        community_rows = (await conn.execute(
            text(
                "SELECT name, level, member_concept_ids "
                "FROM trellion_communities "
                "WHERE snapshot_id = CAST(:id AS uuid) "
                "ORDER BY level"
            ),
            {"id": snapshot_id},
        )).fetchall()

        cid_to_community: dict[int, tuple[int, str]] = {}  # cid -> (level, name)
        for name, level, member_ids in community_rows:
            for member_uuid in (member_ids or []):
                cid = uuid_to_cid.get(member_uuid)
                if cid is None:
                    continue
                # Keep the lowest-level (most-specific) community per concept
                if cid not in cid_to_community or level < cid_to_community[cid][0]:
                    cid_to_community[cid] = (level, name)

        if cid_to_community:
            # Re-bucket taxonomy_members based on community names
            g.taxonomy_members.clear()
            for cid, concept in g.concepts.items():
                if cid in cid_to_community:
                    concept.taxonomy = cid_to_community[cid][1]
                g.taxonomy_members[concept.taxonomy].append(cid)

    logger.debug(
        "snapshot.ckg_loaded",
        snapshot_id=snapshot_id,
        concepts=len(g.concepts),
        edges=sum(len(v) for v in g.forward.values()),
        taxonomies=len(g.taxonomy_members),
    )
    return g


# ----------------------------------------------------------------------
# Cache — snapshots are immutable, so loaded CKGs can live forever.
# Keyed by snapshot_id; small LRU bound to keep memory predictable in
# long-running gateway processes.
# ----------------------------------------------------------------------

_CKG_CACHE: dict[str, "CKG"] = {}
_CKG_CACHE_MAX = 32


async def get_cached_ckg(engine: AsyncEngine, snapshot_id: str) -> "CKG":
    """Load-or-fetch a CKG by snapshot_id (LRU cache size 32)."""
    if snapshot_id in _CKG_CACHE:
        # Cheap LRU touch: reinsert
        ckg = _CKG_CACHE.pop(snapshot_id)
        _CKG_CACHE[snapshot_id] = ckg
        return ckg

    ckg = await load_ckg_from_snapshot(engine, snapshot_id)
    _CKG_CACHE[snapshot_id] = ckg
    while len(_CKG_CACHE) > _CKG_CACHE_MAX:
        _CKG_CACHE.pop(next(iter(_CKG_CACHE)))
    return ckg


def invalidate_cache(snapshot_id: str | None = None) -> None:
    """Drop one snapshot or the whole cache (for tests / admin actions)."""
    if snapshot_id is None:
        _CKG_CACHE.clear()
    else:
        _CKG_CACHE.pop(snapshot_id, None)
