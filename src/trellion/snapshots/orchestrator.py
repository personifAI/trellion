# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""SnapshotOrchestrator — drives a consolidation cycle end-to-end.

Pipeline:
  1. Slice the live bi-temporal graph (GraphSlicer)
  2. Map slice rows to ExtractedEntity / ExtractedRelation
  3. Run the trellion build pipeline's stages 4-7:
       canonicalize → cluster → DAG project
  4. Persist concepts, DAG edges, communities, provenance (SnapshotStore)

Stages 1-3 of the build pipeline (ingest → classical NLP → LLM extract)
are skipped — the live KG already contains extracted entities + edges.

Each consolidation cycle is independent and can be triggered manually
(``trellion consolidate``), via the autonomy engine
(``consolidation_due`` location), or by an MCP tool (``noo_consolidate_now``).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import structlog
from sqlalchemy.ext.asyncio import AsyncEngine

from .slicer import GraphSlicer, SliceEdge, SliceEntity, SliceSpec
from .store import SnapshotStore, describe_failure

logger = structlog.get_logger(__name__)


@dataclass
class ConsolidationResult:
    snapshot_id: str
    status: str  # 'complete' | 'failed' | 'empty'
    stats: dict[str, Any]
    duration_ms: float


class SnapshotOrchestrator:
    """End-to-end driver for one consolidation cycle.

    The build dependencies (LLMClient, EmbedClient, BuildConfig) are
    optional — when absent (e.g., test environment without LLM access),
    the orchestrator runs a degraded path: identity canonicalization
    (no embed-clustering), no LLM-summarised communities, just
    structural deduplication on canonical_name+type.
    """

    def __init__(
        self,
        engine: AsyncEngine,
        build_config: Any | None = None,
    ) -> None:
        self._engine = engine
        self._slicer = GraphSlicer(engine)
        self._store = SnapshotStore(engine)
        self._build_config = build_config

    async def consolidate(self, spec: SliceSpec) -> ConsolidationResult:
        t0 = time.monotonic()
        # A malformed spec is the caller's mistake: refuse it before a
        # 'building' row exists, instead of leaving a failed one behind.
        spec.validate()
        snapshot_id = await self._store.create_snapshot(
            slice_kind=spec.kind,
            slice_spec=spec.params,
            label=spec.label,
        )

        try:
            entities, edges = await self._slicer.slice(spec)
            if not entities:
                stats = {"reason": "empty_slice", "entities": 0, "edges": 0}
                await self._store.mark_complete(snapshot_id, stats)
                return ConsolidationResult(
                    snapshot_id=snapshot_id,
                    status="empty",
                    stats=stats,
                    duration_ms=(time.monotonic() - t0) * 1000,
                )

            canonical, mention_map, dag_edges, communities = self._run_build(entities, edges)

            concept_id_map = await self._store.write_concepts(snapshot_id, canonical)
            edge_count = await self._store.write_dag_edges(snapshot_id, dag_edges, concept_id_map)
            community_count = await self._store.write_communities(snapshot_id, communities, concept_id_map)

            # Provenance: live entity_id -> canonical_id (from mention_map values)
            live_to_canonical: dict[str, str] = {}
            for ent in entities:
                # mention_map keys are (name, type) tuples
                key = (ent.canonical_name.lower().strip(), ent.type)
                if key in mention_map:
                    live_to_canonical[ent.entity_id] = mention_map[key]
            await self._store.write_provenance(snapshot_id, concept_id_map, live_to_canonical)

            stats = {
                "live_entities": len(entities),
                "live_edges": len(edges),
                "concepts": len(canonical),
                "dag_edges": edge_count,
                "communities": community_count,
            }
            await self._store.mark_complete(snapshot_id, stats)
            return ConsolidationResult(
                snapshot_id=snapshot_id,
                status="complete",
                stats=stats,
                duration_ms=(time.monotonic() - t0) * 1000,
            )

        except Exception as exc:
            logger.exception("snapshot.consolidate_failed", snapshot_id=snapshot_id)
            reason = describe_failure(exc)
            await self._store.mark_failed(snapshot_id, reason=reason)
            return ConsolidationResult(
                snapshot_id=snapshot_id,
                status="failed",
                stats={"error": reason},
                duration_ms=(time.monotonic() - t0) * 1000,
            )

    # ------------------------------------------------------------------
    # Build-pipeline integration (stages 4-7)
    # ------------------------------------------------------------------

    def _run_build(
        self,
        entities: list[SliceEntity],
        edges: list[SliceEdge],
    ) -> tuple[list[Any], dict[tuple[str, str], str], list[Any], list[Any]]:
        """Map slice rows → build schema and run stages 4-7.

        Returns:
            (canonical_entities, mention_map, dag_edges, communities)
        """
        # 1. Map slice → ExtractedEntity / ExtractedRelation
        try:
            from trellion.common.schema import ExtractedEntity, ExtractedRelation
        except ImportError:
            ExtractedEntity = ExtractedRelation = None  # type: ignore

        if ExtractedEntity is None:
            return self._degraded_build(entities, edges)

        # Each slice entity becomes one ExtractedEntity (no chunk-level dedup needed
        # since the live KG already merged duplicates per (type, canonical_name)).
        extracted_entities = [
            ExtractedEntity(
                name=e.canonical_name,
                entity_type=e.type,
                description=str(e.extra.get("description", "")),
                source_chunk_id=e.entity_id,  # use live entity_id as the "chunk" for traceability
            )
            for e in entities
        ]

        # Build subject_id -> name map for resolving edge endpoints
        ent_by_id = {e.entity_id: e for e in entities}
        extracted_relations: list[Any] = []
        for edge in edges:
            src_ent = ent_by_id.get(edge.subject_id)
            if src_ent is None:
                continue
            if edge.object_entity_id:
                tgt_ent = ent_by_id.get(edge.object_entity_id)
                if tgt_ent is None:
                    continue
                extracted_relations.append(ExtractedRelation(
                    source=src_ent.canonical_name,
                    target=tgt_ent.canonical_name,
                    relation_type=edge.predicate,
                    description="",
                    strength=edge.confidence,
                    source_chunk_id=edge.edge_id,
                ))
            # Skip literal-object edges (source -> string) — no graph edge to project.

        # 2. Run stages 4 (canonicalize), 6 (cluster), 7 (DAG project)
        try:
            from trellion.build import canonicalize as canon_mod
            from trellion.build import cluster as cluster_mod
            from trellion.build import dag_project as dag_mod
            from trellion.common.embed_client import EmbedClient
        except ImportError:
            return self._degraded_build(entities, edges)

        cfg = self._build_config
        if cfg is None:
            try:
                from trellion.common.config import BuildConfig
                cfg = BuildConfig()
            except Exception:
                return self._degraded_build(entities, edges)

        try:
            embed = EmbedClient(cfg.embed)
        except Exception:
            logger.warning("snapshot.embed_unavailable_using_degraded")
            return self._degraded_build(entities, edges)

        # Canonicalize is async — call it synchronously via a helper
        import asyncio
        try:
            canonical, mention_map = asyncio.get_event_loop().run_until_complete(
                canon_mod.canonicalize(extracted_entities, embed, cfg.canonicalization)
            )
        except Exception:
            # Fall back if loop is already running (caller is async)
            canonical, mention_map = self._degraded_canonicalize(entities)
            extracted_relations = self._degraded_relations(edges, ent_by_id)

        # Map relations to canonical-id edges
        canonical_edges = self._relations_to_canonical_edges(extracted_relations, mention_map)

        try:
            communities = cluster_mod.cluster(canonical_edges, cfg.cluster)
        except Exception:
            communities = []
        try:
            dag_edges = dag_mod.project_to_dag(canonical_edges, cfg.dag)
        except Exception:
            dag_edges = canonical_edges  # fall back to raw edges

        return canonical, mention_map, dag_edges, communities

    def _degraded_build(
        self,
        entities: list[SliceEntity],
        edges: list[SliceEdge],
    ) -> tuple[list[Any], dict[tuple[str, str], str], list[Any], list[Any]]:
        """Identity canonicalization — no LLM, no embedding, no clustering.

        Each live entity becomes one canonical concept; each edge becomes
        one DAG edge (no cycle removal). Used when build deps aren't
        available (LLM/embed clients missing or test environment).
        """
        canonical, mention_map = self._degraded_canonicalize(entities)
        ent_by_id = {e.entity_id: e for e in entities}
        canonical_edges = []
        for edge in edges:
            if not edge.object_entity_id:
                continue
            src = ent_by_id.get(edge.subject_id)
            tgt = ent_by_id.get(edge.object_entity_id)
            if not src or not tgt:
                continue
            src_canon = mention_map.get((src.canonical_name.lower().strip(), src.type))
            tgt_canon = mention_map.get((tgt.canonical_name.lower().strip(), tgt.type))
            if not src_canon or not tgt_canon or src_canon == tgt_canon:
                continue
            canonical_edges.append(_DegradedEdge(
                source_id=src_canon,
                target_id=tgt_canon,
                relation_type=edge.predicate,
                weight=edge.confidence,
            ))
        return canonical, mention_map, canonical_edges, []

    def _degraded_canonicalize(
        self,
        entities: list[SliceEntity],
    ) -> tuple[list[Any], dict[tuple[str, str], str]]:
        """Build CanonicalEntity-shaped objects without embedding."""
        canonical: list[Any] = []
        mention_map: dict[tuple[str, str], str] = {}
        seen: dict[tuple[str, str], str] = {}
        for ent in entities:
            key = (ent.canonical_name.lower().strip(), ent.type)
            if key in seen:
                mention_map[key] = seen[key]
                continue
            canonical_id = f"c_{len(canonical):06d}"
            seen[key] = canonical_id
            mention_map[key] = canonical_id
            canonical.append(_DegradedConcept(
                canonical_id=canonical_id,
                label=ent.canonical_name,
                entity_type=ent.type,
                aliases=[],
            ))
        return canonical, mention_map

    def _degraded_relations(self, edges, ent_by_id):
        """Build ExtractedRelation-shaped objects without LLM context."""
        rels = []
        for edge in edges:
            if not edge.object_entity_id:
                continue
            src = ent_by_id.get(edge.subject_id)
            tgt = ent_by_id.get(edge.object_entity_id)
            if not src or not tgt:
                continue
            rels.append(_DegradedRelation(
                source=src.canonical_name,
                target=tgt.canonical_name,
                relation_type=edge.predicate,
                strength=edge.confidence,
            ))
        return rels

    @staticmethod
    def _relations_to_canonical_edges(relations, mention_map):
        """Reuse trellion.build.pipeline's _relations_to_canonical_edges semantics."""
        try:
            from trellion.common.schema import CanonicalEdge
        except ImportError:
            CanonicalEdge = _DegradedEdge

        name_to_canon: dict[str, str] = {}
        for (name, _t), cid in mention_map.items():
            name_to_canon.setdefault(name, cid)

        out = []
        for r in relations:
            src_name = (getattr(r, "source", None) or r["source"]).lower().strip()
            tgt_name = (getattr(r, "target", None) or r["target"]).lower().strip()
            src = name_to_canon.get(src_name)
            tgt = name_to_canon.get(tgt_name)
            if not src or not tgt or src == tgt:
                continue
            out.append(CanonicalEdge(
                source_id=src,
                target_id=tgt,
                relation_type=getattr(r, "relation_type", None) or r["relation_type"],
                weight=max(1.0, float(getattr(r, "strength", None) or r.get("strength", 1.0))),
            ))
        return out


# Plain-data shim classes when trellion.common.schema isn't available

@dataclass
class _DegradedConcept:
    canonical_id: str
    label: str
    entity_type: str
    aliases: list


@dataclass
class _DegradedEdge:
    source_id: str
    target_id: str
    relation_type: str
    weight: float


@dataclass
class _DegradedRelation:
    source: str
    target: str
    relation_type: str
    strength: float
