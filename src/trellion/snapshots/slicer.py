# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""GraphSlicer — extract a topic / time-window / user-attention subgraph
from the live bi-temporal graph.

The slicer feeds the snapshot orchestrator. It returns plain rows
(entities + edges) so the orchestrator can map them to the build
pipeline's ``ExtractedEntity`` / ``ExtractedRelation`` schema without
the slicer needing to know about it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

logger = structlog.get_logger(__name__)


@dataclass
class SliceSpec:
    """Specifies which portion of the live graph to consolidate.

    kind:
      - ``topic``: N-hop neighborhood around a typed entity. params:
          ``{"entity_type": "topic", "canonical_name": "X", "hops": 2}``
      - ``time_window``: all edges with valid_from in [from, to]. params:
          ``{"from": iso8601_str, "to": iso8601_str}``
      - ``user_attention``: entities the user has touched recently. params:
          ``{"user_id": "alice", "days": 30}``
    """
    kind: str
    params: dict[str, Any] = field(default_factory=dict)
    label: str | None = None

    def serialise(self) -> dict[str, Any]:
        return {"kind": self.kind, "params": self.params, "label": self.label}

    def validate(self) -> None:
        """Raise ``ValueError`` naming the missing params for this slice kind.

        A bare ``params["canonical_name"]`` on a topic slice with no params
        surfaced as ``KeyError: 'canonical_name'`` -- two words that read as
        a schema problem when the real fault is the caller's request shape
        (followup 1311). Say what is required and what arrived instead.
        """
        required = _REQUIRED_PARAMS.get(self.kind)
        if required is None:
            raise ValueError(
                f"Unknown slice kind: {self.kind!r} "
                f"(expected one of {sorted(_REQUIRED_PARAMS)})"
            )
        missing = [k for k in required if self.params.get(k) in (None, "")]
        if missing:
            raise ValueError(
                f"{self.kind} slice needs params {missing}; "
                f"got params keys {sorted(self.params)}"
            )


#: Params a slice kind cannot run without. ``topic`` also takes
#: ``entity_type`` (default "topic") and ``hops`` (default 2); ``user_attention``
#: takes ``days`` (default 30).
_REQUIRED_PARAMS: dict[str, tuple[str, ...]] = {
    "topic": ("canonical_name",),
    "time_window": ("from", "to"),
    "user_attention": ("user_id",),
}


@dataclass
class SliceEntity:
    """A live-KG entity included in a slice."""
    entity_id: str  # str(uuid)
    canonical_name: str
    type: str
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class SliceEdge:
    """A live-KG edge included in a slice (from noo_temporal_edges)."""
    edge_id: str
    subject_id: str
    predicate: str
    object_entity_id: str | None
    object_value: str | None
    confidence: float
    valid_from: datetime
    valid_to: datetime | None


class GraphSlicer:
    """Reads slices from trellion.live's bi-temporal graph tables."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def slice(self, spec: SliceSpec) -> tuple[list[SliceEntity], list[SliceEdge]]:
        """Dispatch to the slicer matching ``spec.kind``."""
        spec.validate()
        if spec.kind == "topic":
            return await self._topic_slice(spec.params)
        if spec.kind == "time_window":
            return await self._time_window_slice(spec.params)
        if spec.kind == "user_attention":
            return await self._user_attention_slice(spec.params)
        raise ValueError(f"Unknown slice kind: {spec.kind!r}")

    # ------------------------------------------------------------------
    # Topic slice — N hops out from a seed entity
    # ------------------------------------------------------------------

    async def _topic_slice(self, params: dict[str, Any]) -> tuple[list[SliceEntity], list[SliceEdge]]:
        seed_type = params.get("entity_type", "topic")
        seed_name = params["canonical_name"].strip().lower()
        hops = max(1, int(params.get("hops", 2)))

        async with self._engine.connect() as conn:
            seed_row = (await conn.execute(
                text(
                    "SELECT id::text FROM noo_entities_typed "
                    "WHERE type = :t AND lower(canonical_name) = :n LIMIT 1"
                ),
                {"t": seed_type, "n": seed_name},
            )).first()
            if seed_row is None:
                logger.info("slicer.topic_seed_missing", type=seed_type, name=seed_name)
                return [], []
            seed_id = seed_row[0]

            # BFS via valid (currently-active) edges
            visited: set[str] = {seed_id}
            frontier: set[str] = {seed_id}
            for _ in range(hops):
                if not frontier:
                    break
                rows = (await conn.execute(
                    text(
                        "SELECT subject_id::text, object_entity_id::text "
                        "FROM noo_temporal_edges "
                        "WHERE valid_to IS NULL "
                        "  AND object_entity_id IS NOT NULL "
                        "  AND (subject_id::text = ANY(:ids) OR object_entity_id::text = ANY(:ids))"
                    ),
                    {"ids": list(frontier)},
                )).fetchall()
                next_frontier: set[str] = set()
                for src, dst in rows:
                    if src and src not in visited:
                        next_frontier.add(src)
                    if dst and dst not in visited:
                        next_frontier.add(dst)
                visited.update(next_frontier)
                frontier = next_frontier

            return await self._materialise(conn, visited)

    # ------------------------------------------------------------------
    # Time-window slice — all edges with valid_from in [from, to]
    # ------------------------------------------------------------------

    async def _time_window_slice(self, params: dict[str, Any]) -> tuple[list[SliceEntity], list[SliceEdge]]:
        from_dt = _parse_dt(params["from"])
        to_dt = _parse_dt(params["to"])

        async with self._engine.connect() as conn:
            rows = (await conn.execute(
                text(
                    "SELECT subject_id::text, object_entity_id::text "
                    "FROM noo_temporal_edges "
                    "WHERE valid_from >= :f AND valid_from <= :t "
                    "  AND object_entity_id IS NOT NULL"
                ),
                {"f": from_dt, "t": to_dt},
            )).fetchall()
            entity_ids: set[str] = set()
            for src, dst in rows:
                if src:
                    entity_ids.add(src)
                if dst:
                    entity_ids.add(dst)
            return await self._materialise(conn, entity_ids, time_window=(from_dt, to_dt))

    # ------------------------------------------------------------------
    # User-attention slice — entities the user has touched in N days
    # ------------------------------------------------------------------

    async def _user_attention_slice(self, params: dict[str, Any]) -> tuple[list[SliceEntity], list[SliceEdge]]:
        user_id = params["user_id"].strip().lower()
        days = int(params.get("days", 30))
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)

        async with self._engine.connect() as conn:
            # Find the user's noo_entities_typed.id
            user_row = (await conn.execute(
                text(
                    "SELECT id::text FROM noo_entities_typed "
                    "WHERE type = 'user' AND lower(canonical_name) = :u LIMIT 1"
                ),
                {"u": user_id},
            )).first()
            if user_row is None:
                logger.info("slicer.user_missing", user_id=user_id)
                return [], []

            user_entity_id = user_row[0]

            # Collect entities touched by the user via temporal edges in window
            rows = (await conn.execute(
                text(
                    "SELECT object_entity_id::text "
                    "FROM noo_temporal_edges "
                    "WHERE subject_id::text = :u "
                    "  AND object_entity_id IS NOT NULL "
                    "  AND valid_from >= :cutoff"
                ),
                {"u": user_entity_id, "cutoff": cutoff},
            )).fetchall()
            entity_ids: set[str] = {user_entity_id}
            entity_ids.update(r[0] for r in rows if r[0])
            return await self._materialise(conn, entity_ids)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _materialise(
        self,
        conn,
        entity_ids: set[str],
        time_window: tuple[datetime, datetime] | None = None,
    ) -> tuple[list[SliceEntity], list[SliceEdge]]:
        """Fetch full entity + edge rows for the given entity ids."""
        if not entity_ids:
            return [], []

        ent_rows = (await conn.execute(
            text(
                "SELECT id::text, type, canonical_name, metadata "
                "FROM noo_entities_typed WHERE id::text = ANY(:ids)"
            ),
            {"ids": list(entity_ids)},
        )).fetchall()
        entities = [
            SliceEntity(
                entity_id=r[0],
                canonical_name=r[2],
                type=r[1],
                extra=r[3] or {},
            )
            for r in ent_rows
        ]

        # Edges: anything with both endpoints in the slice
        edge_query = (
            "SELECT id::text, subject_id::text, predicate, object_entity_id::text, "
            "       object_value, confidence, valid_from, valid_to "
            "FROM noo_temporal_edges "
            "WHERE subject_id::text = ANY(:ids) "
            "  AND (object_entity_id IS NULL OR object_entity_id::text = ANY(:ids))"
        )
        params = {"ids": list(entity_ids)}
        if time_window is not None:
            edge_query += " AND valid_from >= :f AND valid_from <= :t"
            params["f"] = time_window[0]
            params["t"] = time_window[1]

        edge_rows = (await conn.execute(text(edge_query), params)).fetchall()
        edges = [
            SliceEdge(
                edge_id=r[0],
                subject_id=r[1],
                predicate=r[2],
                object_entity_id=r[3],
                object_value=r[4],
                confidence=float(r[5]),
                valid_from=r[6],
                valid_to=r[7],
            )
            for r in edge_rows
        ]
        logger.info("slicer.materialised", entities=len(entities), edges=len(edges))
        return entities, edges


def _parse_dt(v: Any) -> datetime:
    if isinstance(v, datetime):
        return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
    return datetime.fromisoformat(str(v).replace("Z", "+00:00"))
