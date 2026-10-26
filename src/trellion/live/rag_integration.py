# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""PageIndex RAG facade -- wires all components into a single entry point.

Usage:
    rag = PageIndexRAG(dsn="postgresql://...", embedding_url="https://gateway:8080")
    await rag.startup()
    response = await rag.query("What causes widget failures?")
    await rag.shutdown()
"""

from __future__ import annotations

import os
from typing import Optional

import structlog

# Live graph construction + navigation are optional add-on modules.
# Basic installs wire None and the facade degrades
# to ingest + vector retrieval — same pattern as the enrichment
# components below.
try:
    from .graph_builder import GraphBuilder
    _GRAPH_BUILDER_AVAILABLE = True
except ImportError:
    GraphBuilder = None  # type: ignore[assignment, misc]
    _GRAPH_BUILDER_AVAILABLE = False

try:
    from .graph_extractor import EntityExtractor
    _ENTITY_EXTRACTOR_AVAILABLE = True
except ImportError:
    EntityExtractor = None  # type: ignore[assignment, misc]
    _ENTITY_EXTRACTOR_AVAILABLE = False

# The graph narrower (entity-graph traversal that scopes page retrieval) is
# a provider role an add-on may fill; see trellion.live._intel.addon_provider.
try:
    from trellion.live._intel import (
        CLUSTER_BUILDER,
        ENTITY_EMBEDDER,
        GRAPH_METRICS,
        GRAPH_NARROWER,
        RELATION_CLASSIFIER,
        RELATION_EXTRACTOR,
        addon_provider,
    )
    _graph_narrower_factory = addon_provider(GRAPH_NARROWER)
    _GRAPH_NARROWER_AVAILABLE = True
except ImportError:
    _graph_narrower_factory = None
    _GRAPH_NARROWER_AVAILABLE = False

from .graph_types import GraphNarrower
from .roles import BackgroundWorker, ClusterBuilder
from .pageindex_embedder import PageIndexEmbedder
from .pageindex_ingestion import DocumentIngestor
from .pageindex_retriever import PageIndexRetriever
from .pageindex_schema import PageIndexSchema
from .pageindex_store import PageIndexStore
from .persona import PersonaAdapter, PersonaLens
from .rag_orchestrator import RAGOrchestrator, RAGResponse
from .rag_generator import RAGAnswerGenerator
from .rag_verifier import RAGVerifier

def _provider(role: str):
    """The add-on's object for ``role``, or None when no add-on fills it."""
    from trellion.live._intel import addon_provider
    try:
        return addon_provider(role)
    except ImportError:
        return None


# Graph enrichment components: each is a provider role an optional add-on may fill.
_relation_classifier_factory = _provider(RELATION_CLASSIFIER)
_relation_extractor_factory = _provider(RELATION_EXTRACTOR)
_entity_embedder_factory = _provider(ENTITY_EMBEDDER)
_cluster_builder_factory = _provider(CLUSTER_BUILDER)
_CLASSIFIER_AVAILABLE = _relation_classifier_factory is not None
_RELATION_EXTRACTOR_AVAILABLE = _relation_extractor_factory is not None
_GRAPH_EMBEDDER_AVAILABLE = _entity_embedder_factory is not None
_GRAPH_CLUSTERER_AVAILABLE = _cluster_builder_factory is not None

log = structlog.get_logger(__name__)

#: Set to 0 on a process that must not run the backlog loops because another
#: process on the same database runs them. Every loop claims its rows, so two
#: runners stay correct, but each one polls Postgres on its own (pop-os,
#: 2026-10-26: an add-on container built on this image ran a second copy of
#: every loop). The process still serves queries and ingests.
BACKGROUND_WORKERS_ENV = "TRELLION_BACKGROUND_WORKERS"


def background_workers_enabled() -> bool:
    """False only for an explicit 0/false/off/no; unset or anything else is on."""
    raw = os.environ.get(BACKGROUND_WORKERS_ENV, "").strip().lower()
    return raw not in {"0", "false", "off", "no"}


class PageIndexRAG:
    """Facade that initializes and wires all PageIndex + Graph components."""

    def __init__(
        self,
        dsn: str,
        embedding_url: str = "https://gateway:8080",
        llm_url: str = "https://gateway:8080",
        llm_model: str = "llama3",
        messaging=None,
        embedding_backend: str = "tei",
        embedding_model: Optional[str] = None,
        ocr_config=None,
        capability_router=None,
    ) -> None:
        self.dsn = dsn
        self.embedding_url = embedding_url
        self.llm_url = llm_url
        self.llm_model = llm_model
        self.messaging = messaging
        self.embedding_backend = embedding_backend
        self.embedding_model = embedding_model
        self.ocr_config = ocr_config
        self._capability_router = capability_router

        # Components (created in startup)
        self.schema: Optional[PageIndexSchema] = None
        self.store: Optional[PageIndexStore] = None
        self.ingestor: Optional[DocumentIngestor] = None
        self.embedder: Optional[PageIndexEmbedder] = None
        self.extractor: Optional[EntityExtractor] = None
        self.graph_builder: Optional[GraphBuilder] = None
        self.navigator: Optional[GraphNarrower] = None
        self.classifier: Optional[BackgroundWorker] = None
        self.relation_extractor: Optional[BackgroundWorker] = None
        self.entity_embedder: Optional[BackgroundWorker] = None
        self.clusterer: Optional[ClusterBuilder] = None
        self.retriever: Optional[PageIndexRetriever] = None
        self.orchestrator: Optional[RAGOrchestrator] = None
        self.verifier: Optional[RAGVerifier] = None
        self.generator: Optional[RAGAnswerGenerator] = None
        self.persona_adapter: Optional[PersonaAdapter] = None

        self._started = False
        self._background = False

    async def startup(self) -> None:
        """Initialize schema, connect store, create all components, start background workers."""
        if self._started:
            log.warning("pageindex_rag_already_started")
            return

        # 1. Ensure database schema
        self.schema = PageIndexSchema(self.dsn)
        await self.schema.ensure_tables()

        # 2. Connect store
        self.store = PageIndexStore(dsn=self.dsn)
        await self.store.connect()

        # 3. Wire components
        self.ingestor = DocumentIngestor(
            self.store,
            messaging=self.messaging,
            llm_url=self.llm_url,
            llm_model=self.llm_model,
            ocr_config=self.ocr_config,
        )

        self.embedder = PageIndexEmbedder(
            store=self.store,
            embedding_url=self.embedding_url,
            messaging=self.messaging,
            backend=self.embedding_backend,
            model=self.embedding_model,
            capability_router=self._capability_router,
        )

        if _ENTITY_EXTRACTOR_AVAILABLE:
            self.extractor = EntityExtractor(
                store=self.store,
                llm_url=self.llm_url,
                model=self.llm_model,
                messaging=self.messaging,
            )

        if _GRAPH_BUILDER_AVAILABLE:
            self.graph_builder = GraphBuilder(
                store=self.store,
                messaging=self.messaging,
            )

        if _GRAPH_NARROWER_AVAILABLE:
            self.navigator = _graph_narrower_factory(store=self.store)

        # Graph enrichment components (optional)
        if _CLASSIFIER_AVAILABLE:
            self.classifier = _relation_classifier_factory(
                store=self.store,
                llm_url=self.llm_url,
                model=self.llm_model,
                messaging=self.messaging,
                embedding_url=self.embedding_url,
            )
        if _RELATION_EXTRACTOR_AVAILABLE:
            self.relation_extractor = _relation_extractor_factory(
                store=self.store,
                llm_url=self.llm_url,
                model=self.llm_model,
                messaging=self.messaging,
            )
        if _GRAPH_EMBEDDER_AVAILABLE:
            self.entity_embedder = _entity_embedder_factory(
                store=self.store,
                embedder=self.embedder,
                messaging=self.messaging,
            )
        if _GRAPH_CLUSTERER_AVAILABLE:
            self.clusterer = _cluster_builder_factory(
                store=self.store,
                llm_url=self.llm_url,
                model=self.llm_model,
                messaging=self.messaging,
            )

        self.retriever = PageIndexRetriever(
            store=self.store,
            embedder=self.embedder,
        )

        self.verifier = RAGVerifier()

        self.generator = RAGAnswerGenerator(
            llm_url=self.llm_url,
            model=self.llm_model,
        )

        self.persona_adapter = PersonaAdapter()

        # Narrower stack — peer doc-level filters that complement the graph
        # navigator's page-level scoping. MetadataNarrower ships first;
        # TemporalNarrower + ConversationNarrower will be added to this list
        # as they land. Orchestrator no-ops when its narrower list is empty,
        # so this construction is safe even before all narrowers exist.
        from .narrowers import MetadataNarrower, NarrowerOrchestrator
        narrower_orchestrator = NarrowerOrchestrator(
            narrowers=[MetadataNarrower(store=self.store)],
        )

        self.orchestrator = RAGOrchestrator(
            navigator=self.navigator,
            retriever=self.retriever,
            store=self.store,
            messaging=self.messaging,
            verifier=self.verifier,
            generator=self.generator,
            narrower_orchestrator=narrower_orchestrator,
        )

        # 4. Start background workers, unless another process on the same
        # database runs them (background_workers_enabled).
        self._background = background_workers_enabled()
        if not self._background:
            log.info(
                "pageindex_rag.background_workers_off",
                reason=f"{BACKGROUND_WORKERS_ENV}=0: another process runs the backlog loops",
            )
        else:
            await self._start_background_workers()

        # No worker-pool start here. The host process owns its NATS worker
        # pool (noomesh's worker_pool entrypoint starts it on its connected
        # handle after this method returns). Calling the host's pool from a
        # library inverted the dependency and, with this object's unset
        # messaging handle, made every trellion start log a false
        # `worker_pool.no_nats` warning (live controller, every start since 10-03).

        self._started = True
        log.info(
            "pageindex_rag_started",
            dsn=self.dsn[:30] + "...",
            embedding_url=self.embedding_url,
            llm_url=self.llm_url,
        )

    async def _start_background_workers(self) -> None:
        """Start the backlog loops (embed, extract, graph, enrichment, metrics)."""
        await self.embedder.start()
        if self.extractor:
            await self.extractor.start()
        if self.graph_builder:
            await self.graph_builder.start()

        # 5. Start graph enrichment workers (classifier -> embedder; clusterer is heartbeat-triggered)
        if self.classifier:
            try:
                await self.classifier.start()
            except Exception:
                log.warning("pageindex_rag.classifier_start_failed", exc_info=True)
        if self.relation_extractor:
            try:
                await self.relation_extractor.start()
            except Exception:
                log.warning("pageindex_rag.relation_extractor_start_failed", exc_info=True)
        if self.entity_embedder:
            try:
                await self.entity_embedder.start()
            except Exception:
                log.warning("pageindex_rag.entity_embedder_start_failed", exc_info=True)
        # Note: clusterer is not auto-started; it runs on-demand via heartbeat task

        # 6. Start the graph metrics provider (add-on)
        await self._start_signal_computer()

    async def _start_signal_computer(self) -> None:
        """Start the graph metrics provider, or skip quietly in the open tier."""
        signal_factory = _provider(GRAPH_METRICS)
        if signal_factory is None:
            # Tier decided once: the open tier has no graph metrics provider. Not a
            # failure, so no traceback.
            log.info("pageindex_rag.signal_computer_skipped", reason="no graph metrics provider (add-on not installed)")
            self.signal_computer = None  # type: ignore[assignment]
        else:
            try:
                self.signal_computer = signal_factory(
                    store=self.store,
                    messaging=getattr(self, "messaging", None),
                )
                await self.signal_computer.start()
            except Exception:
                log.warning("pageindex_rag.signal_computer_start_failed", exc_info=True)
                self.signal_computer = None  # type: ignore[assignment]

    async def shutdown(self) -> None:
        """Stop background workers, close connections."""
        if not self._started:
            return
        # The clusterer is never started (heartbeat runs it on demand), but it
        # holds a client either way.
        if self.clusterer:
            try:
                await self.clusterer.close()
            except Exception:
                log.warning("pageindex_rag.clusterer_close_failed", exc_info=True)
        if self._background:
            await self._stop_background_workers()

        if self.generator:
            await self.generator.close()

        # Close store connection
        if self.store:
            await self.store.close()

        self._started = False
        log.info("pageindex_rag_stopped")

    async def _stop_background_workers(self) -> None:
        # Stop graph enrichment workers (reverse order: embedder, classifier)
        if self.entity_embedder:
            try:
                await self.entity_embedder.stop()
            except Exception:
                log.warning("pageindex_rag.entity_embedder_stop_failed", exc_info=True)
        if self.relation_extractor:
            try:
                await self.relation_extractor.stop()
            except Exception:
                log.warning("pageindex_rag.relation_extractor_stop_failed", exc_info=True)
        if self.classifier:
            try:
                await self.classifier.stop()
            except Exception:
                log.warning("pageindex_rag.classifier_stop_failed", exc_info=True)

        # Stop background workers (order: graph_builder, extractor, embedder)
        if self.graph_builder:
            await self.graph_builder.stop()
        if self.extractor:
            await self.extractor.stop()
        if self.embedder:
            await self.embedder.stop()

    async def ingest(
        self,
        file_path_or_text: str,
        title: str | None = None,
        source_uri: str | None = None,
        metadata: dict | None = None,
    ) -> str:
        """Ingest a document (file path or raw text). Returns doc_id.

        *source_uri* should be the **original** file path when the file
        has been copied to a working directory, so that watched-folder
        dedup can match against it later.

        *metadata* is an optional structured dict (folder_path, drive_id,
        tags, etc.) stored on the docs row. Merged with any existing
        metadata on dedup hits.

        If the content hash already exists the existing doc_id is
        returned and no pages/chunks are recreated (deduplication).
        """
        if not self._started:
            raise RuntimeError("PageIndexRAG not started. Call startup() first.")

        from pathlib import Path

        # Only treat path-plausible strings as file paths: Path.exists()
        # raises ENAMETOOLONG on raw document text (any input > ~255 bytes),
        # which made API text ingestion fail for every real document while
        # returning what looked like success upstream (2026-07-09).
        if len(file_path_or_text) < 4096 and "\n" not in file_path_or_text:
            try:
                path = Path(file_path_or_text)
                if path.exists() and path.is_file():
                    return await self.ingestor.ingest_file(
                        path, title=title, source_uri=source_uri, metadata=metadata,
                    )
            except OSError:
                pass  # not a usable path — fall through to raw text

        # Treat as raw text. source_uri passes through (it was hardcoded ""
        # here, which broke dedup-by-source for text ingests).
        return await self.ingestor.ingest_text(
            text=file_path_or_text,
            title=title or "Untitled",
            source_uri=source_uri or "",
            metadata=metadata,
        )

    async def query(
        self,
        question: str,
        strategy: str = "hybrid",
        persona_id: Optional[str] = None,
        roles: Optional[list[str]] = None,
    ) -> RAGResponse:
        """Run the full RAG pipeline.

        Args:
            question: Natural-language question.
            strategy: Retrieval strategy override.
            persona_id: Optional persona ID for biased retrieval.
        """
        if not self._started:
            raise RuntimeError("PageIndexRAG not started. Call startup() first.")

        # Load persona lens if requested
        persona_lens: Optional[PersonaLens] = None
        if persona_id and self.persona_adapter:
            persona_lens = await PersonaAdapter.load_from_db(persona_id)
            if persona_lens:
                log.info(
                    "pageindex_rag.persona_loaded",
                    persona_id=persona_id,
                    name=persona_lens.name,
                    strategy=persona_lens.preferred_strategy,
                )

        # roles: KB audience segmentation — None is unrestricted; a list
        # limits retrieval to docs whose metadata.roles overlaps it (docs
        # without tags count as ["user"]). Resolution of requester → roles
        # happens app-side (gateway worker); trellion only enforces.
        return await self.orchestrator.query(
            question, strategy=strategy, persona_lens=persona_lens, roles=roles
        )

    async def status(self) -> dict:
        """Return system status: doc count, page count, entity count, edge count, embedding queue."""
        if not self._started:
            return {"started": False}

        from sqlalchemy import func, select
        from ..db.models import (
            Doc, Page, Entity, Edge, PageEmbedding, PageChunk, ChunkEmbedding,
        )

        table_map = [
            (Doc.__table__, "doc_count"),
            (Page.__table__, "page_count"),
            (Entity.__table__, "entity_count"),
            (Edge.__table__, "edge_count"),
            (PageEmbedding.__table__, "embedded_pages"),
            (PageChunk.__table__, "chunk_count"),
            (ChunkEmbedding.__table__, "embedded_chunks"),
        ]

        counts: dict = {}
        async with self.store._engine.connect() as conn:
            for tbl, key in table_map:
                val = (await conn.execute(
                    select(func.count()).select_from(tbl)
                )).scalar()
                counts[key] = val if val is not None else 0

        # Compute unembedded counts
        counts["unembedded_pages"] = counts["page_count"] - counts["embedded_pages"]
        counts["unembedded_chunks"] = counts["chunk_count"] - counts["embedded_chunks"]

        # Duplicate page count (for autonomy dedup check)
        counts["duplicate_pages"] = await self.store.count_duplicate_pages()

        counts["embedder_state"] = self.embedder.get_status() if self.embedder else None

        # Graph enrichment layer stats
        graph_layers: dict = {}
        try:
            from sqlalchemy import text as _sa_text
            async with self.store._engine.connect() as conn:
                # Classification progress
                classified = (await conn.execute(
                    _sa_text("SELECT COUNT(*) FROM edges WHERE classified_at IS NOT NULL")
                )).scalar() or 0
                unclassified = (await conn.execute(
                    _sa_text("SELECT COUNT(*) FROM edges WHERE classified_at IS NULL")
                )).scalar() or 0
                total_edges = classified + unclassified

                # Edge type distribution
                edge_types_raw = (await conn.execute(
                    _sa_text(
                        "SELECT rel_type, COUNT(*) as cnt FROM edges "
                        "GROUP BY rel_type ORDER BY cnt DESC LIMIT 10"
                    )
                )).fetchall()
                edge_types = {str(r[0]): int(r[1]) for r in edge_types_raw}

                # Entity embedding coverage
                embedded_entities = (await conn.execute(
                    _sa_text("SELECT COUNT(*) FROM entity_embeddings")
                )).scalar() or 0

                # Cluster stats
                cluster_count = (await conn.execute(
                    _sa_text("SELECT COUNT(*) FROM entity_clusters")
                )).scalar() or 0
                clustered_entities = (await conn.execute(
                    _sa_text("SELECT COUNT(DISTINCT entity_id) FROM entity_cluster_members")
                )).scalar() or 0

            entity_count = counts.get("entity_count", 0)
            graph_layers = {
                "classification": {
                    "classified": classified,
                    "unclassified": unclassified,
                    "total": total_edges,
                    "progress_pct": round(classified / total_edges * 100, 1) if total_edges else 0,
                    "edge_types": edge_types,
                },
                "embeddings": {
                    "embedded_entities": embedded_entities,
                    "total_entities": entity_count,
                    "progress_pct": round(embedded_entities / entity_count * 100, 1) if entity_count else 0,
                },
                "clusters": {
                    "cluster_count": cluster_count,
                    "clustered_entities": clustered_entities,
                    "unclustered": max(0, entity_count - clustered_entities),
                },
            }
        except Exception:
            pass  # Tables may not exist yet (pre-migration)

        counts["graph_layers"] = graph_layers
        counts["started"] = True
        return counts
