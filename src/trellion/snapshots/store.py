# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""SnapshotStore — persistence for trellion_snapshots and family.

Writes pipeline output (concepts + dag_edges + communities + provenance)
into the snapshot tables. All writes for one consolidation cycle happen
under the same snapshot_id, so a partial failure leaves a 'failed'-status
snapshot rather than a silent half-write.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

logger = structlog.get_logger(__name__)


@dataclass
class SnapshotSummary:
    """Lightweight view of one snapshot row."""
    id: str
    generation: int
    slice_kind: str
    slice_spec: dict[str, Any]
    status: str
    created_at: datetime
    completed_at: datetime | None
    stats: dict[str, Any]
    label: str | None = None


@dataclass
class ConceptIdMap:
    """Maps build-pipeline canonical_id (str from canonicalize) → DB UUID."""
    by_canonical_id: dict[str, str] = field(default_factory=dict)


def describe_failure(exc: BaseException) -> str:
    """The reason stored on a failed snapshot: ``Type: message``.

    ``str(KeyError('canonical_name'))`` is just ``'canonical_name'``; the
    exception type is what makes a stored reason diagnosable (followup 1311).
    """
    return f"{type(exc).__name__}: {exc}"[:500]


def _field(obj, name, default=None):
    """Attr-or-key access that survives falsy values.

    The previous ``getattr(x, n, None) or x.get(n)`` idiom broke on exactly
    the documented input type: a pydantic ``CanonicalEntity`` with
    ``aliases=[]`` (falsy → fell through to ``.get`` which models don't
    have), and a ``CommunityAssignment`` with ``level=0`` or
    ``parent_cluster=None``. Caught by the batch-emit roundtrip DB test.
    """
    if isinstance(obj, dict):
        return obj.get(name, default)
    val = getattr(obj, name, None)
    return default if val is None else val


class SnapshotStore:
    """Async writer for the trellion_snapshots family of tables."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    # ------------------------------------------------------------------
    # Snapshot lifecycle
    # ------------------------------------------------------------------

    async def create_snapshot(
        self,
        slice_kind: str,
        slice_spec: dict[str, Any],
        label: str | None = None,
    ) -> str:
        """Create a snapshot row in 'building' status. Returns its id (str-uuid)."""
        snapshot_id = str(uuid.uuid4())
        async with self._engine.begin() as conn:
            generation = await self._next_generation(conn, slice_kind, slice_spec)
            await conn.execute(
                text(
                    "INSERT INTO trellion_snapshots "
                    "(id, generation, slice_kind, slice_spec, status, label) "
                    "VALUES (CAST(:id AS uuid), :gen, :kind, CAST(:spec AS jsonb), 'building', :label)"
                ),
                {
                    "id": snapshot_id,
                    "gen": generation,
                    "kind": slice_kind,
                    "spec": json.dumps(slice_spec),
                    "label": label,
                },
            )
        logger.info("snapshot.created", id=snapshot_id, kind=slice_kind, generation=generation)
        return snapshot_id

    async def mark_complete(self, snapshot_id: str, stats: dict[str, Any]) -> None:
        async with self._engine.begin() as conn:
            await conn.execute(
                text(
                    "UPDATE trellion_snapshots "
                    "SET status = 'complete', completed_at = NOW(), "
                    "    stats = CAST(:stats AS jsonb) "
                    "WHERE id = CAST(:id AS uuid)"
                ),
                {"id": snapshot_id, "stats": json.dumps(stats)},
            )
        logger.info("snapshot.complete", id=snapshot_id, **stats)

    async def mark_failed(self, snapshot_id: str, reason: str) -> None:
        async with self._engine.begin() as conn:
            await conn.execute(
                text(
                    "UPDATE trellion_snapshots "
                    "SET status = 'failed', completed_at = NOW(), "
                    "    stats = jsonb_build_object('error', CAST(:reason AS text)) "
                    "WHERE id = CAST(:id AS uuid)"
                ),
                {"id": snapshot_id, "reason": reason},
            )
        logger.warning("snapshot.failed", id=snapshot_id, reason=reason)

    # ------------------------------------------------------------------
    # Concepts / edges / communities — bulk writes per snapshot
    # ------------------------------------------------------------------

    async def write_concepts(
        self,
        snapshot_id: str,
        canonical_entities: list[Any],
    ) -> ConceptIdMap:
        """Insert concepts and return a build_id -> db_id map.

        ``canonical_entities`` is whatever ``trellion.build.canonicalize``
        produced — typically pydantic ``CanonicalEntity`` instances with
        ``canonical_id``, ``label``, ``entity_type``, ``aliases``.
        """
        if not canonical_entities:
            return ConceptIdMap()

        rows = []
        id_map: dict[str, str] = {}
        for ent in canonical_entities:
            db_id = str(uuid.uuid4())
            id_map[_field(ent, "canonical_id")] = db_id
            rows.append({
                "id": db_id,
                "snapshot_id": snapshot_id,
                "canonical_name": _field(ent, "label"),
                "type": _field(ent, "entity_type"),
                "aliases": json.dumps(list(_field(ent, "aliases", []))),
                "extra": json.dumps({}),
            })

        async with self._engine.begin() as conn:
            await conn.execute(
                text(
                    "INSERT INTO trellion_concepts "
                    "(id, snapshot_id, canonical_name, type, aliases, metadata) "
                    "VALUES (CAST(:id AS uuid), CAST(:snapshot_id AS uuid), "
                    "        :canonical_name, :type, "
                    "        CAST(:aliases AS jsonb), CAST(:extra AS jsonb))"
                ),
                rows,
            )
        logger.info("snapshot.concepts_written", snapshot_id=snapshot_id, count=len(rows))
        return ConceptIdMap(by_canonical_id=id_map)

    async def write_dag_edges(
        self,
        snapshot_id: str,
        dag_edges: list[Any],
        concept_id_map: ConceptIdMap,
    ) -> int:
        """Insert DAG edges. Skips edges whose endpoints aren't in the concept map."""
        rows = []
        skipped = 0
        for e in dag_edges:
            src_id = _field(e, "source_id")
            tgt_id = _field(e, "target_id")
            db_src = concept_id_map.by_canonical_id.get(src_id)
            db_tgt = concept_id_map.by_canonical_id.get(tgt_id)
            if not db_src or not db_tgt:
                skipped += 1
                continue
            rows.append({
                "id": str(uuid.uuid4()),
                "snapshot_id": snapshot_id,
                "src": db_src,
                "dst": db_tgt,
                "rel_type": _field(e, "relation_type", "related"),
                "weight": float(_field(e, "weight", 1.0)),
                "extra": json.dumps({}),
            })

        if rows:
            async with self._engine.begin() as conn:
                await conn.execute(
                    text(
                        "INSERT INTO trellion_dag_edges "
                        "(id, snapshot_id, src_concept_id, dst_concept_id, rel_type, weight, metadata) "
                        "VALUES (CAST(:id AS uuid), CAST(:snapshot_id AS uuid), "
                        "        CAST(:src AS uuid), CAST(:dst AS uuid), "
                        "        :rel_type, :weight, CAST(:extra AS jsonb))"
                    ),
                    rows,
                )
        logger.info(
            "snapshot.dag_edges_written",
            snapshot_id=snapshot_id, written=len(rows), skipped=skipped,
        )
        return len(rows)

    async def write_communities(
        self,
        snapshot_id: str,
        communities: list[Any],
        concept_id_map: ConceptIdMap,
    ) -> int:
        """Insert hierarchical communities.

        ``communities`` is whatever ``trellion.build.cluster`` produced —
        ``CommunityAssignment(entity_id, level, cluster_id, parent_cluster)``.
        We group assignments by (level, cluster_id) into a single
        trellion_communities row holding the member concept ids.
        """
        if not communities:
            return 0

        # Group: (level, cluster_id) -> list[member_concept_id_db]
        from collections import defaultdict
        groups: dict[tuple[int, int], dict[str, Any]] = defaultdict(
            lambda: {"members": [], "parent": None}
        )
        for a in communities:
            level = _field(a, "level")
            cluster_id = _field(a, "cluster_id")
            entity_id = _field(a, "entity_id")
            parent = _field(a, "parent_cluster")
            db_id = concept_id_map.by_canonical_id.get(entity_id)
            if db_id is None:
                continue
            groups[(level, cluster_id)]["members"].append(db_id)
            if parent is not None:
                groups[(level, cluster_id)]["parent"] = parent

        # Two-pass to resolve self-referential parent_community_id
        cluster_db_ids: dict[tuple[int, int], str] = {}
        for key in groups:
            cluster_db_ids[key] = str(uuid.uuid4())

        rows = []
        for (level, cluster_id), info in groups.items():
            parent_db_id = None
            if info["parent"] is not None:
                # Parent is at level - 1
                parent_db_id = cluster_db_ids.get((level - 1, info["parent"]))
            rows.append({
                "id": cluster_db_ids[(level, cluster_id)],
                "snapshot_id": snapshot_id,
                "level": level,
                "parent_id": parent_db_id,
                "name": f"L{level}.C{cluster_id}",
                "summary": None,
                "members": json.dumps(info["members"]),
                "member_count": len(info["members"]),
            })

        async with self._engine.begin() as conn:
            await conn.execute(
                text(
                    "INSERT INTO trellion_communities "
                    "(id, snapshot_id, level, parent_community_id, name, summary, "
                    " member_concept_ids, member_count) "
                    "VALUES (CAST(:id AS uuid), CAST(:snapshot_id AS uuid), "
                    "        :level, "
                    "        CAST(:parent_id AS uuid), "
                    "        :name, :summary, "
                    "        CAST(:members AS jsonb), :member_count)"
                ),
                rows,
            )
        logger.info("snapshot.communities_written", snapshot_id=snapshot_id, count=len(rows))
        return len(rows)

    async def write_provenance(
        self,
        snapshot_id: str,
        concept_id_map: ConceptIdMap,
        live_entity_to_canonical: dict[str, str],
    ) -> int:
        """Link concept rows back to the live KG entities they came from.

        ``live_entity_to_canonical`` maps a live ``noo_entities_typed.id``
        (string) to the build-pipeline canonical_id (string).
        """
        rows = []
        for live_id, canonical_id in live_entity_to_canonical.items():
            db_concept_id = concept_id_map.by_canonical_id.get(canonical_id)
            if db_concept_id is None:
                continue
            rows.append({
                "id": str(uuid.uuid4()),
                "snapshot_id": snapshot_id,
                "concept_id": db_concept_id,
                "source_kind": "temporal_edge",
                "source_id": live_id,
            })

        if rows:
            async with self._engine.begin() as conn:
                await conn.execute(
                    text(
                        "INSERT INTO trellion_provenance "
                        "(id, snapshot_id, concept_id, source_kind, source_id) "
                        "VALUES (CAST(:id AS uuid), CAST(:snapshot_id AS uuid), "
                        "        CAST(:concept_id AS uuid), :source_kind, "
                        "        CAST(:source_id AS uuid))"
                    ),
                    rows,
                )
        logger.info("snapshot.provenance_written", snapshot_id=snapshot_id, count=len(rows))
        return len(rows)

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    async def list_snapshots(
        self,
        slice_kind: str | None = None,
        status: str | None = None,
        limit: int = 50,
    ) -> list[SnapshotSummary]:
        sql = (
            "SELECT id::text, generation, slice_kind, slice_spec, status, "
            "       created_at, completed_at, stats, label "
            "FROM trellion_snapshots WHERE 1=1 "
        )
        params: dict[str, Any] = {"limit": limit}
        if slice_kind:
            sql += "AND slice_kind = :kind "
            params["kind"] = slice_kind
        if status:
            sql += "AND status = :status "
            params["status"] = status
        sql += "ORDER BY created_at DESC LIMIT :limit"

        async with self._engine.connect() as conn:
            rows = (await conn.execute(text(sql), params)).fetchall()
        return [
            SnapshotSummary(
                id=r[0],
                generation=r[1],
                slice_kind=r[2],
                slice_spec=r[3] or {},
                status=r[4],
                created_at=r[5],
                completed_at=r[6],
                stats=r[7] or {},
                label=r[8],
            )
            for r in rows
        ]

    async def get_snapshot(self, snapshot_id: str) -> SnapshotSummary | None:
        async with self._engine.connect() as conn:
            row = (await conn.execute(
                text(
                    "SELECT id::text, generation, slice_kind, slice_spec, status, "
                    "       created_at, completed_at, stats, label "
                    "FROM trellion_snapshots WHERE id = CAST(:id AS uuid)"
                ),
                {"id": snapshot_id},
            )).first()
        if row is None:
            return None
        return SnapshotSummary(
            id=row[0],
            generation=row[1],
            slice_kind=row[2],
            slice_spec=row[3] or {},
            status=row[4],
            created_at=row[5],
            completed_at=row[6],
            stats=row[7] or {},
            label=row[8],
        )

    async def diff(self, snapshot_a: str, snapshot_b: str) -> dict[str, Any]:
        """Return added/removed concepts and dag-edges between two snapshots."""
        async with self._engine.connect() as conn:
            a_concepts = {
                (r[0], r[1])
                for r in (await conn.execute(
                    text(
                        "SELECT canonical_name, type FROM trellion_concepts "
                        "WHERE snapshot_id = CAST(:id AS uuid)"
                    ),
                    {"id": snapshot_a},
                )).fetchall()
            }
            b_concepts = {
                (r[0], r[1])
                for r in (await conn.execute(
                    text(
                        "SELECT canonical_name, type FROM trellion_concepts "
                        "WHERE snapshot_id = CAST(:id AS uuid)"
                    ),
                    {"id": snapshot_b},
                )).fetchall()
            }

        return {
            "concepts_added": sorted(b_concepts - a_concepts),
            "concepts_removed": sorted(a_concepts - b_concepts),
            "concepts_in_both": len(a_concepts & b_concepts),
            "snapshot_a": snapshot_a,
            "snapshot_b": snapshot_b,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _next_generation(self, conn, slice_kind: str, slice_spec: dict[str, Any]) -> int:
        """Compute next generation for this (kind, spec) — just an incrementing counter per kind."""
        row = (await conn.execute(
            text(
                "SELECT COALESCE(MAX(generation), 0) + 1 FROM trellion_snapshots "
                "WHERE slice_kind = :kind"
            ),
            {"kind": slice_kind},
        )).first()
        return int(row[0])
