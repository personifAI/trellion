# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""VideoMemoryManager — public facade backed by the v2 Postgres+FAISS path.

Replaces the original 877-line MP4+QR manager. Same class name, same
public method signatures + attributes that the gateway/cli/heartbeat
depend on; bodies delegate to ``trellion.live.recall`` plus a thin
DB join to materialize hits as the legacy ``SearchResult`` shape.

Conversation archiving is a no-op now — conversations already live in
``noo_entries`` with embeddings (managed by the embedding worker). The
v1 separate MP4 conversation archive was duplicating that storage.

Traversal capture still happens on every proven graph walk via
``capture_traversal(snapshot)``. The traversal_archiver attribute
exposes the same ``.capture(snapshot)`` API the retrieval feedback
loop expects (see ``trellion.live.video.traversal_archiver``).
"""
from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import httpx
import structlog

from trellion import embedding_space

from .config import VideoMemoryConfig
from .traversal_archiver import TraversalArchiver, TraversalSnapshot

logger = structlog.get_logger(__name__)


@dataclass
class SearchResult:
    """v1-compatible result shape — what the gateway expects from search().

    Accepts both v2 fields (``frame_id``) and legacy v1 fields
    (``frame_number``, ``video_path``) for backward-compat with code +
    tests written against the MP4 era. The legacy fields are stored but
    not interpreted.
    """
    record_id: str = ""
    text: str = ""
    record_type: str = "conversation"
    score: float = 0.0
    prominence_score: float = 0.5
    persona_id: str = ""
    metadata: dict = None  # type: ignore[assignment]
    source_index: str = ""
    frame_id: str = ""
    # Legacy v1 fields — kept so tests + callers built against the MP4
    # era still construct cleanly. Inert in v2.
    frame_number: int = 0
    video_path: str = ""

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class _NoOpConversationArchiver:
    """Placeholder — v1 archived conversations to MP4; v2 doesn't need to.

    Conversations live in ``noo_entries`` with their own embedding column,
    indexed via pgvector. The embedding worker keeps that current.
    """
    async def get_unarchived_count(self) -> int:
        return 0

    async def archive(self, *args, **kwargs):
        return {"archived": False, "reason": "v2 — conversations live in noo_entries"}


class VideoMemoryManager:
    """v2-backed facade — Postgres + FAISS, no MP4.

    Constructor accepts the v1 signature so existing wiring works
    unchanged. Some args (storage_backend, gateway_url) are kept for
    compatibility but unused in v2.
    """

    def __init__(
        self,
        config: VideoMemoryConfig,
        embedding_url: str = "https://gateway:8080",
        gateway_url: str = "http://127.0.0.1:8080",
        conversation_store: Optional[Any] = None,
        storage_backend: Optional[Any] = None,
    ) -> None:
        self.config = config
        self.embedding_url = embedding_url.rstrip("/")
        embedding_space.configure_url(self.embedding_url)
        self.gateway_url = gateway_url.rstrip("/")
        self._conversation_store = conversation_store

        self.storage_dir = Path(os.path.expanduser(config.storage_dir))
        # Don't mkdir at construction — caller may use a system path that
        # only resolves under Docker. mkdir lazily in startup() instead.

        # v1 compat attrs (used by callers that introspect the manager)
        self.archiver: Any = _NoOpConversationArchiver()
        self.traversal_archiver: TraversalArchiver = TraversalArchiver(
            embedding_url=self.embedding_url,
            faiss_base_path=str(self.storage_dir),
        )
        self.replay_engine: Any = None  # Built in startup once engine is wired
        self.lifecycle: Any = None
        self.indices: dict = {}  # v1 had {path -> FAISSIndex}; kept empty in v2

        self._rag_instance: Optional[Any] = None
        self._started = False
        self._engine = None
        self._v2_manager = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def startup(self) -> None:
        """Wire up v2 components against trellion's engine."""
        if self._started:
            return

        try:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            logger.warning("video_memory.storage_dir_unavailable",
                           path=str(self.storage_dir), error=str(exc)[:120])
        else:
            # Fold the pre-unification hyphen directory in, once (safe with
            # several containers sharing the volume). Never fatal.
            try:
                from .paths import migrate_legacy_dir
                await asyncio.to_thread(migrate_legacy_dir, self.storage_dir)
            except Exception as exc:
                logger.warning("video_memory.legacy_migration_failed",
                               error=str(exc)[:200])

        self._started = True
        if self._ensure_engine():
            logger.info("video_memory.started", backend="postgres+faiss",
                        storage=str(self.storage_dir))
        else:
            # The gateway starts this manager before the shared engine is
            # created. Not a fault: the first call that needs the engine
            # wires it (see _ensure_engine).
            logger.info("video_memory.engine_deferred")

    def _ensure_engine(self) -> bool:
        """Wire the v2 components once trellion's engine exists.

        Idempotent and cheap; called at startup and again by every method that
        needs the engine, so a manager started before the engine exists comes
        alive on first use instead of staying a no-op until restart.
        """
        if self._engine is not None:
            return True
        try:
            from trellion.db.engine import get_async_engine
            engine = get_async_engine()
        except Exception:
            return False

        from trellion.live.recall import (
            TraversalReplayEngine, RecallLifecycleManager, EpisodicRecallManager,
        )

        self._engine = engine
        self.replay_engine = TraversalReplayEngine(str(self.storage_dir), engine)
        self.lifecycle = RecallLifecycleManager(str(self.storage_dir), engine)
        self._v2_manager = EpisodicRecallManager(
            engine,
            faiss_base_path=str(self.storage_dir),
            min_outcome_score=self.config.promotion_confidence_threshold,
        )
        logger.info("video_memory.engine_wired")
        return True

    async def shutdown(self) -> None:
        if not self._started:
            return
        self._started = False
        logger.info("video_memory.stopped")

    # ------------------------------------------------------------------
    # Search — v1 takes a TEXT query; we embed then FAISS-search
    # ------------------------------------------------------------------

    async def search(
        self,
        query: str,
        top_k: int = 5,
        persona_id: Optional[str] = None,
    ) -> list[SearchResult]:
        """Search the persona's conversations FAISS index.

        Returns SearchResult objects matching v1's shape so callers
        (debug_router, persona_coordinator) keep working.
        """
        if not self._started or not self._ensure_engine():
            return []

        embedding = await self._embed_text(query)
        if not embedding:
            return []

        from trellion.live.recall import PersonaVectorIndex
        idx = PersonaVectorIndex(self._engine, persona_id or "default")
        try:
            hits = await idx.search(
                # None = all indexes. The old hardcoded "conversations"
                # pointed at a never-populated index, so this search
                # returned [] forever (found 2026-08-29).
                index_name=None,
                query_embedding=embedding,
                k=top_k,
            )
        except Exception:
            logger.warning("video_memory.search_failed", exc_info=True)
            return []

        return [
            SearchResult(
                record_id=h.frame_id,
                frame_id=h.frame_id,
                record_type=h.record_type,
                score=h.similarity,
                prominence_score=h.prominence,
                persona_id=persona_id or "default",
                source_index="conversations",
            )
            for h in hits
        ]

    async def search_traversals(
        self,
        query: str,
        top_k: int = 5,
        persona_id: Optional[str] = None,
    ) -> list[SearchResult]:
        """Search the traversals FAISS index — proven past graph walks."""
        if not self._started or not self._ensure_engine():
            return []

        embedding = await self._embed_text(query)
        if not embedding:
            return []

        matches = await self.replay_engine.find_similar(
            persona_id=persona_id or "default",
            query_embedding=embedding,
            k=top_k,
        )
        return [
            SearchResult(
                record_id=m.traversal_id,
                text=m.query,
                record_type="traversal",
                score=m.text_similarity,
                prominence_score=m.outcome_score,
                persona_id=persona_id or "default",
                source_index="traversals",
                metadata={"intent": m.intent, "outcome": m.outcome_score,
                          "path": m.traversal_path},
            )
            for m in matches
        ]

    async def search_with_evidence(
        self,
        query: str,
        top_k: int = 5,
        persona_id: Optional[str] = None,
    ) -> list[Any]:
        """Backward-compat alias — v1 returned EvidencePack-shaped results.

        For v2 we return SearchResult — callers that want richer evidence
        should query noo_entries / docs directly with the matched record_id.
        """
        return await self.search(query, top_k=top_k, persona_id=persona_id)

    async def find_traversal_replays(
        self,
        query: str,
        top_k: int = 5,
        persona_id: Optional[str] = None,
    ) -> list[Any]:
        return await self.search_traversals(query, top_k=top_k, persona_id=persona_id)

    # ------------------------------------------------------------------
    # Capture (write side)
    # ------------------------------------------------------------------

    def capture_traversal(self, snapshot: TraversalSnapshot) -> None:
        """Sync hand-off — schedules an async DB+FAISS write."""
        try:
            self.traversal_archiver.capture(snapshot)
        except Exception:
            logger.warning("video_memory.capture_failed", exc_info=True)

    async def archive_conversations(
        self,
        persona_id: Optional[str] = None,
        *args,
        **kwargs,
    ) -> dict[str, Any]:
        """No-op — v2 keeps conversations in noo_entries (embedding worker).

        The dedicated archiver path was the v1 MP4 design; v2 reuses the
        existing pgvector-indexed conversation column. Signature keeps
        ``persona_id`` so persona-scoping callers (and persona_scoping
        tests) compile against the same shape.
        """
        return {"archived": 0, "persona_id": persona_id,
                "skipped_reason": "v2 uses noo_entries"}

    async def archive_all(
        self,
        persona_id: Optional[str] = None,
        *args,
        **kwargs,
    ) -> dict[str, Any]:
        return await self.archive_conversations(persona_id, *args, **kwargs)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def run_lifecycle(self, rag_instance: Optional[Any] = None) -> dict[str, Any]:
        """Run prominence recompute + tier promotion across all personas."""
        if not self._started or not self._ensure_engine():
            return {"frames_scored": 0, "frames_elevated": 0, "frames_pruned": 0}

        # Discover all personas with frames, run lifecycle per persona
        from sqlalchemy import text
        async with self._engine.connect() as conn:
            personas = [r[0] for r in (await conn.execute(
                text("SELECT DISTINCT persona_id FROM trellion_episodes")
            )).fetchall()]

        agg = {"frames_scored": 0, "frames_elevated": 0, "frames_pruned": 0,
               "avg_prominence": 0.0, "personas": len(personas)}
        for p in personas:
            r = await self.lifecycle.run_lifecycle(p)
            agg["frames_scored"] += r.get("frames_scored", 0)
            agg["frames_elevated"] += r.get("frames_elevated", 0)
            agg["frames_pruned"] += r.get("frames_pruned", 0)
        return agg

    async def refresh(self, rag_instance: Optional[Any] = None) -> dict[str, Any]:
        """Alias for run_lifecycle."""
        return await self.run_lifecycle(rag_instance)

    # ------------------------------------------------------------------
    # Stats / diagnostics
    # ------------------------------------------------------------------

    def get_traversal_stats(self) -> dict[str, Any]:
        """Sync stats from trellion_traversal_records (cached snapshot)."""
        return {"backend": "postgres", "version": 2}

    def get_lifecycle_stats(self) -> dict[str, Any]:
        return {"backend": "postgres", "version": 2}

    def status(self) -> dict[str, Any]:
        """Return per-persona frame summary."""
        if not self._started:
            return {"started": False, "version": 2, "backend": "postgres"}
        return {
            "started": True,
            "version": 2,
            "backend": "postgres",
            "storage_dir": str(self.storage_dir),
            "engine_wired": self._ensure_engine(),
        }

    # ------------------------------------------------------------------
    # Setters (kept for v1 API compat)
    # ------------------------------------------------------------------

    def set_rag_instance(self, rag: Any) -> None:
        self._rag_instance = rag

    def set_conversation_store(self, store: Any) -> None:
        self._conversation_store = store

    def save_indices(self) -> None:
        """No-op — v2 PersonaFAISSIndex saves on every add."""
        return

    async def _save_indices(self) -> None:
        return

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _embed_text(self, text: str) -> Optional[list[float]]:
        """Embed via the gateway's embeddings endpoint.

        Same contract as ``pageindex_embedder._embed_gateway`` — the
        gateway is the only embedding transport since the harrier seam.
        This used to post TEI-style to ``{url}/embed``, which the gateway
        404s, so every video-memory embed failed silently at debug level.
        """
        if not self.embedding_url:
            return None
        try:
            from trellion.live._tls import client_cert, ssl_verify
        except ImportError:
            def ssl_verify() -> bool:  # type: ignore[misc]
                return True

            def client_cert() -> None:  # type: ignore[misc]
                return None

        try:
            from trellion.live.gateway_auth import GatewayHmacAuth

            # The shared auth hook (re-sign, key fallback, refusal circuit —
            # noomesh followup 436a), not a hand-built header.
            async with httpx.AsyncClient(
                timeout=10.0, verify=ssl_verify(), cert=client_cert(), auth=GatewayHmacAuth()
            ) as client:
                resp = await client.post(
                    f"{self.embedding_url}/api/external/embeddings",
                    json={"input": [text], "model": embedding_space.space_name()},
                )
                resp.raise_for_status()
                # A Vector: the list plus the model the response named, so a
                # similarity search over episodes compares like with like.
                out = embedding_space.from_response(resp.json())
                return out[0] if out else None
        except Exception:
            logger.warning("video_memory.embed_failed", exc_info=True)
            return None
