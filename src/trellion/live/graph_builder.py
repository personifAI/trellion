# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""Automatic graph edge builder for the PageIndex knowledge graph.

Creates RELATED_TO edges for entities co-occurring on the same page and
REFERENCES edges for entities mentioned across multiple pages within the
same document.  No LLM calls are required -- relationships are inferred
from co-occurrence in the PageIndex.
"""

from __future__ import annotations

import asyncio
from itertools import combinations
from typing import Any, Optional

import structlog

from trellion.live.entity_quality import DropTally, check_entity
from trellion.live.evidence import Evidence
from trellion.live.pageindex_store import PageIndexStore

log = structlog.get_logger(__name__)


class GraphBuilder:
    """Builds relationship edges from entity co-occurrence patterns."""

    def __init__(
        self,
        store: PageIndexStore,
        messaging: Optional[Any] = None,
    ) -> None:
        self.store = store
        self.messaging = messaging  # Optional ClusterMessaging
        self._running = False
        self._task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # Per-page edges (RELATED_TO)
    # ------------------------------------------------------------------

    async def build_edges_for_page(self, page_id: str) -> int:
        """Create RELATED_TO edges between all entities on *page_id*.

        Returns the number of new edges created.
        """
        page = await self.store.get_page(page_id)
        if not page:
            log.warning("graph_builder.page_not_found", page_id=page_id)
            return 0

        entities = await self._get_page_entities(page_id)
        if len(entities) < 2:
            return 0

        snippet = (page["text"] or "")[:200]
        ev = Evidence(
            source_type="graph_edge",
            doc_id=page["doc_id"],
            page_no=page["page_no"],
            page_id=page_id,
            snippet=snippet,
            span=(0, len(snippet)),
            confidence=1.0,
        )
        evidence = ev.to_dict()

        created = 0
        for ent_a, ent_b in combinations(entities, 2):
            # Deterministic ordering so we don't create duplicate A->B / B->A
            src, dst = sorted([ent_a, ent_b])
            if await self._edge_exists(src, "RELATED_TO", dst):
                continue
            await self.store.insert_edge(src, "RELATED_TO", dst, evidence)
            created += 1

        if created:
            log.info(
                "graph_builder.page_edges",
                page_id=page_id,
                edges=created,
            )
        return created

    # ------------------------------------------------------------------
    # Per-doc edges (REFERENCES)
    # ------------------------------------------------------------------

    async def build_edges_for_doc(self, doc_id: str) -> int:
        """Create RELATED_TO + REFERENCES edges across all pages in a doc.

        - RELATED_TO: entities sharing the same page.
        - REFERENCES:  entity mentioned on multiple pages in the same doc.

        Returns total new edges created.
        """
        pages = await self.store.get_doc_pages(doc_id)
        if not pages:
            return 0

        total_created = 0

        # 1) RELATED_TO edges per page
        for page in pages:
            total_created += await self.build_edges_for_page(page["page_id"])

        # 2) REFERENCES edges: entities appearing on >1 page in this doc
        entity_pages: dict[str, list[dict]] = {}  # entity_id -> [page dicts]
        for page in pages:
            ents = await self._get_page_entities(page["page_id"])
            for eid in ents:
                entity_pages.setdefault(eid, []).append(page)

        for entity_id, pg_list in entity_pages.items():
            if len(pg_list) < 2:
                continue
            # Create REFERENCES edges between pairs of pages' entity sets
            # to capture cross-page relationships for this entity.
            for i, page_a in enumerate(pg_list):
                for page_b in pg_list[i + 1 :]:
                    ents_a = set(await self._get_page_entities(page_a["page_id"]))
                    ents_b = set(await self._get_page_entities(page_b["page_id"]))
                    # The cross-page entities that share *entity_id* in common
                    shared = ents_a & ents_b
                    for other_id in shared:
                        if other_id == entity_id:
                            continue
                        src, dst = sorted([entity_id, other_id])
                        if await self._edge_exists(src, "REFERENCES", dst):
                            continue
                        ref_snippet = (page_a["text"] or "")[:200]
                        ev_ref = Evidence(
                            source_type="graph_edge",
                            doc_id=doc_id,
                            page_no=page_a["page_no"],
                            page_id=page_a["page_id"],
                            snippet=ref_snippet,
                            span=(0, len(ref_snippet)),
                            confidence=1.0,
                        )
                        evidence = ev_ref.to_dict()
                        await self.store.insert_edge(
                            src, "REFERENCES", dst, evidence
                        )
                        total_created += 1

        if total_created:
            log.info(
                "graph_builder.doc_edges",
                doc_id=doc_id,
                edges=total_created,
            )

        # Publish summary events
        if self.messaging and self.messaging.is_connected:
            await self.messaging.publish_event(
                "graph.updated",
                {"doc_id": doc_id, "edge_count": total_created},
            )
            await self.messaging.publish_event(
                "graph.edges.built",
                {"doc_id": doc_id, "edge_count": total_created},
            )

        return total_created

    # ------------------------------------------------------------------
    # Background loop
    # ------------------------------------------------------------------

    async def start(self, interval: float = 15.0) -> None:
        """Start background loop listening for ``graph.entity.extracted``."""
        if self._running:
            return
        self._running = True

        if self.messaging and self.messaging.is_connected:
            await self.messaging.subscribe_events(
                "graph.entity.extracted",
                self._on_entity_extracted,
                durable_name="graph_builder",
            )
            log.info("graph_builder.subscribed", nats_subject="graph.entity.extracted")

        self._task = asyncio.create_task(self._loop(interval))
        log.info("graph_builder.started", interval=interval)

    async def stop(self) -> None:
        """Gracefully stop the background loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        log.info("graph_builder.stopped")

    async def _on_entity_extracted(self, event: dict) -> None:
        """React to a single entity-extracted event by building edges for its page."""
        page_id = event.get("page_id")
        if page_id:
            await self.build_edges_for_page(page_id)

    async def _loop(self, interval: float) -> None:
        """Polling fallback: find pages with entities but no edges yet."""
        while self._running:
            try:
                await asyncio.sleep(interval)
                pages = await self._get_pages_needing_edges()
                for page in pages:
                    if not self._running:
                        break
                    await self.build_edges_for_page(page["page_id"])
            except asyncio.CancelledError:
                break
            except Exception:
                log.exception("graph_builder.loop_error")
                await asyncio.sleep(interval)

    # ------------------------------------------------------------------
    # DB helpers
    # ------------------------------------------------------------------

    async def _get_page_entities(self, page_id: str) -> list[str]:
        """Return distinct entity_ids mentioned on a page."""
        from sqlalchemy import select
        from ..db.models import Entity, Mention
        _mention = Mention.__table__
        _entity = Entity.__table__

        async with self.store._engine.connect() as conn:
            result = await conn.execute(
                select(_entity.c.entity_id, _entity.c.canonical_name, _entity.c.type)
                .where(
                    _entity.c.entity_id.in_(
                        select(_mention.c.entity_id).where(_mention.c.page_id == page_id)
                    )
                )
            )
            rows = result.fetchall()
        # Junk that is already stored must not get edges either: an edge
        # endpoint goes through the same gate as an entity does.
        tally = DropTally()
        ids: list[str] = []
        for eid, name, etype in rows:
            reason = check_entity(name, etype)
            if reason:
                tally.add(name, reason)
            else:
                ids.append(str(eid))
        tally.log("graph_builder.junk_endpoints_skipped", page_id=page_id)
        return ids

    async def _edge_exists(
        self, src: str, rel_type: str, dst: str
    ) -> bool:
        """Check whether an edge already exists (avoids duplicates)."""
        from sqlalchemy import select, exists
        from ..db.models import Edge
        _edge = Edge.__table__

        async with self.store._engine.connect() as conn:
            result = await conn.execute(
                select(
                    exists(
                        select(_edge.c.edge_id)
                        .where(
                            _edge.c.src_entity_id == src,
                            _edge.c.rel_type == rel_type,
                            _edge.c.dst_entity_id == dst,
                        )
                    )
                )
            )
            return result.scalar() or False

    async def _get_pages_needing_edges(self, limit: int = 50) -> list[dict]:
        """Pages that have mentions but whose entities lack edges."""
        from sqlalchemy import distinct, select
        from ..db.models import Page, Mention, Edge
        _page = Page.__table__
        _mention = Mention.__table__
        _edge = Edge.__table__

        # Correlated subquery: pages where at least one mention's entity
        # has been connected via an edge
        m2 = _mention.alias("m2")
        edge_exists = (
            select(1)
            .select_from(_edge.join(m2, m2.c.entity_id == _edge.c.src_entity_id))
            .where(m2.c.page_id == _page.c.page_id)
            .exists()
        )

        async with self.store._engine.connect() as conn:
            result = await conn.execute(
                select(
                    distinct(_page.c.page_id),
                    _page.c.doc_id,
                    _page.c.page_no,
                    _page.c.text,
                )
                .select_from(_page.join(_mention, _mention.c.page_id == _page.c.page_id))
                .where(~edge_exists)
                .order_by(_page.c.page_id)
                .limit(limit)
            )
            rows = result.fetchall()
            return [
                {
                    "page_id": str(r[0]),
                    "doc_id": str(r[1]),
                    "page_no": r[2],
                    "text": r[3],
                }
                for r in rows
            ]
