# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""Memory system facade.

Provides a single entry point for all memory operations:
- Session memory (in-process context window)
- Persistent storage (ConversationStore / PGVectorStore)
- Semantic search (embeddings via TEI/Ollama)
- Smart routing (add-on router, optional)
- Cross-node memory (NATS cluster context)
- Performance learning (per-user router stats)

The single write path ``record()`` writes to DB + session + NATS in one
call, eliminating dual-writes by design.
"""

from __future__ import annotations

import asyncio
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import structlog

from .pgvector_store import PGVectorConfig, PGVectorStore
from .store import FileMemoryStore, InMemoryStore, MemoryRecord

logger = structlog.get_logger()


@dataclass
class MemoryConfig:
    enabled: bool = False
    include_session: bool = True
    include_global: bool = True
    session_top_k: int = 4
    global_top_k: int = 4
    min_score: float = 0.1
    retrieval_scoring: str = "jaccard"
    max_chars: int = 2000
    max_session_items: int = 200
    max_global_items: int = 2000
    global_path: Optional[Path] = None
    embedding_model: Optional[str] = None
    embedding_base_url: Optional[str] = None
    pgvector: Optional[PGVectorConfig] = None


class SessionMemory:
    def __init__(self, store: InMemoryStore) -> None:
        self.store = store


class MemorySystem:
    """Facade for all memory operations.

    Provides session memory, semantic search, smart routing,
    NATS cross-node context, and embedding management.
    """

    def __init__(
        self,
        config: MemoryConfig,
        store: Optional[Any] = None,  # ConversationStore
        embedder: Optional[Callable[[str], List[float]]] = None,
        vector_store: Optional[PGVectorStore] = None,
    ) -> None:
        self.config = config
        self.store = store
        self.embedder = embedder
        # Legacy: PGVectorStore for CLI path (factory.py)
        self.vector_store = vector_store
        self._router_cache: Dict[str, Any] = {}
        self._nats_initialized = False

        # File-based fallback for CLI mode without DB
        self.global_store: Optional[FileMemoryStore] = None
        if config.include_global and config.global_path and not store and not vector_store:
            self.global_store = FileMemoryStore(
                path=config.global_path,
                scope="global",
                max_items=config.max_global_items,
            )

    # ==================== SESSION MANAGEMENT ====================

    def new_session(self) -> SessionMemory:
        store = InMemoryStore(scope="session", max_items=self.config.max_session_items)
        return SessionMemory(store)

    # ==================== SINGLE WRITE PATH ====================

    async def record_async(
        self,
        conversation_id: Optional[str],
        role: str,
        content: str,
        session: Optional[SessionMemory] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
        persona_id: Optional[str] = None,
    ) -> Optional[str]:
        """Record a message to all memory tiers (async, includes NATS).

        Returns entry_id from DB write (or None if no store).
        """
        entry_id = None

        # 1. DB (persistent, single source of truth)
        if self.store:
            conversation_id = await self._ensure_store_conversation_async(
                conversation_id=conversation_id,
                metadata=metadata,
            )
            entry_id = await self.store.add_message(conversation_id, role, content, metadata)

        # 2. Session (in-memory context window)
        if session and self.config.include_session:
            session.store.add(role=role, content=content)

        # 3. NATS (cross-node cluster memory)
        if user_id:
            await self._record_to_nats(user_id, role, content)

        return entry_id

    def record_sync(
        self,
        session: Optional[SessionMemory],
        role: str,
        content: str,
        conversation_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> None:
        """Sync variant for non-async callers.

        Writes to ConversationStore when configured, otherwise PGVectorStore or
        file-backed global memory. Gateway code can still use record_async(),
        but synchronous agent loops can use the full Postgres-backed store.

        Signature is compatible with the old MemoryManager.record() for
        callers that pass (session, role, content).
        """
        if not self.config.enabled:
            return

        # Session write (in-memory, no IO)
        if self.config.include_session and session:
            session.store.add(role=role, content=content)

        # PGVectorStore write (CLI path — sync store)
        if self.config.include_global:
            if self.store:
                try:
                    self._run_sync(
                        self._record_to_store_async(
                            role=role,
                            content=content,
                            conversation_id=conversation_id,
                            metadata=metadata,
                        )
                    )
                except Exception:
                    pass
            elif self.vector_store:
                try:
                    embedding = None
                    if self.embedder:
                        try:
                            embedding = self.embedder(content)
                        except Exception:
                            pass
                    self.vector_store.add(role=role, content=content, embedding=embedding)
                except Exception:
                    pass
            elif self.global_store:
                self.global_store.add(role=role, content=content)

    # ==================== SYNC RETRIEVAL (coordinator, agent) ====================

    def retrieve(self, query: str, session: Optional[SessionMemory]) -> str:
        """Search session + vector store. Used by coordinator and agent loop."""
        if not self.config.enabled:
            return ""

        lines: List[str] = []

        # Session hits
        if self.config.include_session and session:
            session_hits = session.store.search(
                query=query,
                top_k=self.config.session_top_k,
                min_score=self.config.min_score,
                scoring=self.config.retrieval_scoring,
            )
            lines.extend(self._format_hits("session", session_hits))

        # Global hits (ConversationStore or PGVectorStore or file)
        if self.config.include_global:
            global_hits = self._search_global(query)
            lines.extend(self._format_hits("global", global_hits))

        if not lines:
            return ""

        output = "Relevant memory:\n" + "\n".join(lines)
        return output[: self.config.max_chars]

    def _search_global(self, query: str) -> List[tuple]:
        """Search global memory (sync stores only).

        ConversationStore (async) is handled by _search_global_async().
        This method only searches PGVectorStore and FileMemoryStore.
        """
        # PGVectorStore (CLI path — sync)
        if self.store:
            try:
                return self._run_sync(self._search_store_async(query))
            except Exception:
                return []

        if self.vector_store and self.embedder:
            try:
                embedding = self.embedder(query)
                hits = self.vector_store.search(
                    embedding=embedding,
                    top_k=self.config.global_top_k,
                )
                return [
                    (record, score)
                    for record, score in hits
                    if score >= self.config.min_score
                ]
            except Exception:
                return []

        # File-based fallback
        if self.global_store:
            return self.global_store.search(
                query=query,
                top_k=self.config.global_top_k,
                min_score=self.config.min_score,
                scoring=self.config.retrieval_scoring,
            )

        return []

    # ==================== ASYNC RETRIEVAL (coordinator, agent) ====================

    async def retrieve_async(
        self,
        query: str,
        session: Optional[SessionMemory],
        persona_id: Optional[str] = None,
    ) -> str:
        """Async search session + conversation store. For gateway callers."""
        if not self.config.enabled:
            return ""

        lines: List[str] = []

        # Session hits (in-memory, no IO)
        if self.config.include_session and session:
            session_hits = session.store.search(
                query=query,
                top_k=self.config.session_top_k,
                min_score=self.config.min_score,
                scoring=self.config.retrieval_scoring,
            )
            lines.extend(self._format_hits("session", session_hits))

        # Global hits (ConversationStore async or PGVectorStore/file sync)
        if self.config.include_global:
            global_hits = await self._search_global_async(query, persona_id=persona_id)
            lines.extend(self._format_hits("global", global_hits))

        if not lines:
            return ""

        output = "Relevant memory:\n" + "\n".join(lines)
        return output[: self.config.max_chars]

    async def _search_global_async(
        self,
        query: str,
        persona_id: Optional[str] = None,
    ) -> List[tuple]:
        """Async search global memory (ConversationStore)."""
        # Prefer ConversationStore. Use semantic search when an embedder is
        # configured, otherwise fall back to pg_trgm text retrieval.
        if self.store:
            try:
                if self.embedder:
                    try:
                        embedding = self.embedder(query)
                        raw = await self.store.search_semantic(
                            embedding,
                            top_k=self.config.global_top_k,
                            persona_id=persona_id,
                        )
                        if not raw:
                            raw = await self.store.search_text(
                                query,
                                top_k=self.config.global_top_k,
                                persona_id=persona_id,
                            )
                    except Exception as exc:
                        logger.warning("conversation_store_semantic_search_failed_fallback_text", error=str(exc))
                        raw = await self.store.search_text(
                            query,
                            top_k=self.config.global_top_k,
                            persona_id=persona_id,
                        )
                else:
                    raw = await self.store.search_text(
                        query,
                        top_k=self.config.global_top_k,
                        persona_id=persona_id,
                    )
                return self._entries_to_records(raw)
            except Exception:
                return []

        # Legacy: PGVectorStore (CLI path, sync)
        if self.vector_store and self.embedder:
            try:
                embedding = self.embedder(query)
                hits = self.vector_store.search(
                    embedding=embedding,
                    top_k=self.config.global_top_k,
                )
                return [
                    (record, score)
                    for record, score in hits
                    if score >= self.config.min_score
                ]
            except Exception:
                return []

        # File-based fallback
        if self.global_store:
            return self.global_store.search(
                query=query,
                top_k=self.config.global_top_k,
                min_score=self.config.min_score,
                scoring=self.config.retrieval_scoring,
            )

        return []

    # ==================== ASYNC SMART RETRIEVAL ====================

    async def retrieve_smart(
        self,
        query: str,
        session: Optional[SessionMemory],
        user_id: str,
        strategy: str = "hybrid",
        **kwargs,
    ) -> str:
        """Smart async retrieval with intelligent routing.

        On Basic-tier installs (no add-on router module) this degrades to the plain synchronous retrieval
        path, same as an unknown strategy does.
        """

        try:
            router = await self._get_router(user_id)
        except ImportError:
            logger.info("retrieve_smart_router_unavailable", strategy=strategy)
            return self.retrieve(query, session)

        if strategy == "iterative":
            results = await router.retrieve_iterative(
                query=query,
                session=session,
                max_iterations=kwargs.get("max_iterations", 3),
                confidence_threshold=kwargs.get("confidence_threshold", 0.6),
            )
            return self._format_retrieval_results(results)

        elif strategy == "progressive":
            data = await router.retrieve_progressive(
                query=query,
                session=session,
                callback=kwargs.get("callback"),
            )
            immediate = data["immediate"]
            return "\n".join(immediate.items) if immediate.items else ""

        elif strategy == "hybrid":
            context, _ = await router.retrieve_hybrid(
                query=query,
                session=session,
                callback=kwargs.get("callback"),
                enable_iteration=kwargs.get("enable_iteration", True),
                confidence_threshold=kwargs.get("confidence_threshold", 0.6),
            )
            return context

        else:
            logger.warning("unknown_strategy", strategy=strategy)
            return self.retrieve(query, session)

    async def retrieve_with_fallback(
        self,
        query: str,
        session: Optional[SessionMemory],
        user_id: str,
        use_smart_routing: bool = True,
    ) -> str:
        """Retrieve with automatic fallback to sync if async fails."""
        if not use_smart_routing or not self.config.enabled:
            return self.retrieve(query, session)

        try:
            return await self.retrieve_smart(
                query=query,
                session=session,
                user_id=user_id,
                strategy="hybrid",
            )
        except Exception as e:
            logger.warning("smart_routing_failed_fallback", error=str(e), query=query[:50])
            return self.retrieve(query, session)

    # ==================== STATS ====================

    def get_performance_stats(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get learned performance statistics for a user."""
        if user_id in self._router_cache:
            return self._router_cache[user_id].get_performance_stats()
        return None

    async def get_storage_stats(self) -> Dict[str, Any]:
        """Consolidated storage stats for dashboard."""
        if self.store:
            return await self.store.get_stats()
        if self.vector_store:
            try:
                return self.vector_store.get_stats()
            except Exception:
                return {}
        return {}

    # ==================== INTERNAL ====================

    async def _get_router(self, user_id: str) -> Any:
        """Get or create the add-on memory router for user."""
        from trellion.live._intel import QUERY_ROUTER, addon_provider
        router_factory = addon_provider(QUERY_ROUTER)

        if user_id not in self._router_cache:
            router = router_factory(self, user_id)

            if not self._nats_initialized:
                try:
                    await router.init_nats()
                    self._nats_initialized = True
                    logger.info("nats_initialized_for_routing")
                except Exception as e:
                    logger.warning("nats_init_failed", error=str(e))

            self._router_cache[user_id] = router

        return self._router_cache[user_id]

    async def _record_to_nats(self, user_id: str, role: str, content: str) -> None:
        """Record message to NATS shared memory."""
        try:
            if user_id not in self._router_cache:
                return
            router = self._router_cache[user_id]
            if router.nats_conversation:
                await router.nats_conversation.add_message(role, content)
                logger.debug("message_recorded_nats", user_id=user_id, role=role)
        except Exception as e:
            logger.warning("nats_record_failed", user_id=user_id, error=str(e))

    def clear_router_cache(self) -> None:
        """Clear router cache (useful for testing)."""
        self._router_cache.clear()
        self._nats_initialized = False

    async def _record_to_store_async(
        self,
        *,
        role: str,
        content: str,
        conversation_id: Optional[str],
        metadata: Optional[Dict],
    ) -> Optional[str]:
        if not self.store:
            return None
        resolved_id = await self._ensure_store_conversation_async(
            conversation_id=conversation_id,
            metadata=metadata,
        )
        entry_id = await self.store.add_message(resolved_id, role, content, metadata)
        if self.embedder and entry_id:
            try:
                embedding = self.embedder(content)
                await self.store.update_embedding(entry_id, embedding)
            except Exception as exc:
                logger.warning("conversation_store_embedding_failed", error=str(exc), entry_id=entry_id)
        return entry_id

    async def _ensure_store_conversation_async(
        self,
        *,
        conversation_id: Optional[str],
        metadata: Optional[Dict],
    ) -> str:
        if not self.store:
            raise RuntimeError("ConversationStore is not configured")
        store_metadata = dict(metadata or {})
        if conversation_id and _is_uuid(conversation_id):
            await self.store.ensure_conversation(
                conversation_id,
                backend="trellion",
                model="memory",
                metadata=store_metadata,
            )
            return conversation_id
        if conversation_id:
            store_metadata.setdefault("external_conversation_id", conversation_id)
        return await self.store.create_conversation(
            backend="trellion",
            model="memory",
            metadata=store_metadata,
        )

    async def _search_store_async(self, query: str) -> List[tuple[MemoryRecord, float]]:
        if not self.store:
            return []
        if self.embedder:
            try:
                embedding = self.embedder(query)
                raw = await self.store.search_semantic(embedding, top_k=self.config.global_top_k)
                if not raw:
                    raw = await self.store.search_text(query, top_k=self.config.global_top_k)
            except Exception as exc:
                logger.warning("conversation_store_semantic_search_failed_fallback_text", error=str(exc))
                raw = await self.store.search_text(query, top_k=self.config.global_top_k)
        else:
            raw = await self.store.search_text(query, top_k=self.config.global_top_k)
        return self._entries_to_records(raw)

    def _entries_to_records(self, raw: List[tuple[dict, float]]) -> List[tuple[MemoryRecord, float]]:
        return [
            (
                MemoryRecord(
                    id=entry["id"],
                    scope="global",
                    role=entry.get("type", "assistant"),
                    content=entry["content"],
                    created_at=float(entry.get("created_at", 0)),
                ),
                score,
            )
            for entry, score in raw
            if score >= self.config.min_score
        ]

    @staticmethod
    def _run_sync(coro: Any) -> Any:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        if not loop.is_running():
            return loop.run_until_complete(coro)

        result: Dict[str, Any] = {}

        def runner() -> None:
            try:
                result["value"] = asyncio.run(coro)
            except BaseException as exc:
                result["error"] = exc

        thread = threading.Thread(target=runner, daemon=True)
        thread.start()
        thread.join()
        if "error" in result:
            raise result["error"]
        return result.get("value")

    @staticmethod
    def _format_hits(
        scope: str, hits: List[tuple[MemoryRecord, float]]
    ) -> List[str]:
        lines = []
        for record, score in hits:
            text = record.content.replace("\n", " ").strip()
            lines.append(f"- ({scope}, {record.role}, {score:.2f}) {text}")
        return lines

    @staticmethod
    def _format_retrieval_results(results: list) -> str:
        """Format list of RetrievalResults into context string."""
        if not results:
            return ""

        all_items = []
        for result in results:
            if isinstance(result, list):
                all_items.extend(result)
            else:
                # a router's result object carries its context lines as ``items``
                items = getattr(result, "items", None)
                if isinstance(items, list):
                    all_items.extend(items)

        if not all_items:
            return ""

        return "Relevant memory:\n" + "\n".join(all_items)


    def record(self, session: Optional[SessionMemory], role: str, content: str) -> None:
        """Sync record to session + global store.

        Backward-compatible with old MemoryManager.record(session, role, content).
        """
        self.record_sync(session, role, content)


# Backward-compatible alias so CLI code using MemoryManager still works
MemoryManager = MemorySystem


def _is_uuid(value: str) -> bool:
    try:
        uuid.UUID(str(value))
        return True
    except ValueError:
        return False
