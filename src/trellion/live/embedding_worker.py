# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""Background embedding worker for off-hours semantic indexing.

Features:
- Idempotent: only embeds records that don't have embeddings
- Batched: processes in configurable batch sizes
- Relationship building: computes similarity links between records
- Resumable: tracks progress, can be interrupted and resumed
- Throttled: configurable delays to manage GPU load

Usage:
    # CLI
    noo memory embed --batch-size 100 --build-graph

    # API
    POST /api/memory/embed/start
    GET /api/memory/embed/status
    POST /api/memory/embed/stop

    # Scheduled (cron)
    0 2 * * * noo memory embed --batch-size 500
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import structlog

from trellion import embedding_space

logger = structlog.get_logger()


class WorkerState(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"


@dataclass
class EmbeddingStats:
    """Track embedding job statistics."""
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    total_processed: int = 0
    total_embedded: int = 0
    total_failed: int = 0
    total_skipped: int = 0  # Already had embeddings
    relationships_created: int = 0
    current_batch: int = 0
    total_batches: int = 0
    last_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "total_processed": self.total_processed,
            "total_embedded": self.total_embedded,
            "total_failed": self.total_failed,
            "total_skipped": self.total_skipped,
            "relationships_created": self.relationships_created,
            "current_batch": self.current_batch,
            "total_batches": self.total_batches,
            "last_error": self.last_error,
            "duration_seconds": (
                (self.completed_at or datetime.now(timezone.utc)) - self.started_at
            ).total_seconds() if self.started_at else 0
        }


@dataclass
class EmbeddingConfig:
    """Configuration for embedding worker."""
    batch_size: int = 100
    delay_between_batches: float = 1.0  # seconds
    max_batches: Optional[int] = None  # None = unlimited
    build_relationships: bool = False
    similarity_threshold: float = 0.7  # For relationship building
    max_relationships_per_record: int = 10
    embedding_model: Optional[str] = None
    embedding_base_url: Optional[str] = None
    # Gateway is the only transport; other values are warned about and
    # ignored (kept as a field so stored configs keep validating).
    embedding_backend: str = "gateway"


class EmbeddingWorker:
    """Background worker for semantic embedding and relationship building."""

    def __init__(
        self,
        config: EmbeddingConfig,
        store: Any,  # ConversationStore (primary) or PGVectorStore (legacy)
        embedder: Optional[Callable[[str], List[float]]] = None,
        # Deprecated aliases for backward compat
        vector_store: Optional[Any] = None,
        conversation_store: Optional[Any] = None,
    ):
        self.config = config
        # Accept store from any of the three params
        self.store = store or conversation_store or vector_store
        self.embedder = embedder
        self.state = WorkerState.IDLE
        self.stats = EmbeddingStats()
        self._stop_requested = False
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> bool:
        """Start the embedding worker."""
        if self.state == WorkerState.RUNNING:
            logger.warning("embedding_worker_already_running")
            return False

        if not self.embedder:
            # Try to create embedder from config
            self.embedder = self._create_embedder()

        if not self.embedder:
            logger.error("no_embedder_available")
            return False

        self._stop_requested = False
        self.state = WorkerState.RUNNING
        self.stats = EmbeddingStats(started_at=datetime.now(timezone.utc))

        # Run in background
        self._task = asyncio.create_task(self._run())
        logger.info("embedding_worker_started", config=self.config)
        return True

    async def batch_index(self, max_batches: int = 5) -> Dict[str, Any]:
        """Embed every unembedded entry, once, and report what happened.

        `worker.memory.batch_index` has always called this method — and it
        did not exist, so the handler's `hasattr` check fell through and
        returned {"status": "not_supported"}. A silent no-op: the caller got
        a 200 and nothing was embedded. That is why noo_entries sat at 0/6106
        after the 768 -> 1024 migration blanked it, while pages and chunks
        recovered.

        Unlike `start()` this awaits the run and returns counts, so a caller
        can tell "embedded 250" from "did nothing".

        BOUNDED by `max_batches` on purpose. This is reached over a NATS
        request/reply subject, and an unbounded drain does not fit inside a
        request deadline: awaiting all 6106 entries at observed throughput
        would take ~100 minutes, so the caller got a timeout while the work
        continued invisibly in the background. A bounded pass returns a real
        answer within the deadline, and `remaining` tells the caller whether
        to call again.
        """
        if self.state == WorkerState.RUNNING:
            return {"status": "already_running"}

        if not self.embedder:
            self.embedder = self._create_embedder()
        if not self.embedder:
            # Report the failure. Returning a success-shaped dict here is
            # what made the original bug invisible.
            return {"status": "error", "error": "no embedder available"}

        self._stop_requested = False
        self.state = WorkerState.RUNNING
        self.stats = EmbeddingStats(started_at=datetime.now(timezone.utc))
        # Bound this pass without permanently altering the worker's config —
        # start() and any other caller must keep their own limit.
        prior_max = self.config.max_batches
        self.config.max_batches = max_batches
        try:
            await self._run()
        finally:
            self.config.max_batches = prior_max
            self.state = WorkerState.IDLE

        remaining = len(await self.store.get_unembedded(limit=1000000))
        return {
            # "complete" only when nothing is left; otherwise the caller must
            # know to keep going rather than assume one call was enough.
            "status": "complete" if remaining == 0 else "partial",
            "embedded": self.stats.total_embedded,
            "failed": self.stats.total_failed,
            "processed": self.stats.total_processed,
            "skipped": self.stats.total_skipped,
            "remaining": remaining,
        }

    async def stop(self) -> None:
        """Request graceful stop."""
        if self.state != WorkerState.RUNNING:
            return

        self._stop_requested = True
        self.state = WorkerState.STOPPING
        logger.info("embedding_worker_stop_requested")

        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=30.0)
            except asyncio.TimeoutError:
                self._task.cancel()

    def get_status(self) -> Dict[str, Any]:
        """Get current worker status."""
        return {
            "state": self.state.value,
            "stats": self.stats.to_dict(),
            "config": {
                "batch_size": self.config.batch_size,
                "build_relationships": self.config.build_relationships,
                "embedding_model": self.config.embedding_model,
            }
        }

    async def _run(self) -> None:
        """Main worker loop."""
        try:
            # Count total unembedded records
            unembedded = await self.store.get_unembedded(limit=1000000)
            total_count = len(unembedded)
            self.stats.total_batches = (total_count + self.config.batch_size - 1) // self.config.batch_size

            logger.info("embedding_job_starting", total_records=total_count, batches=self.stats.total_batches)

            # Track failed IDs so we don't retry them endlessly
            failed_ids: set = set()
            batch_num = 0
            consecutive_empty = 0
            while not self._stop_requested:
                # Get next batch of unembedded records
                batch = await self.store.get_unembedded(limit=self.config.batch_size)

                # Filter out already-failed records from this run
                if batch and failed_ids:
                    batch = [(rid, content) for rid, content in batch if rid not in failed_ids]

                if not batch:
                    consecutive_empty += 1
                    if consecutive_empty >= 2:
                        logger.info("no_more_unembedded_records",
                                    skipped_failed=len(failed_ids))
                        break
                    # One more try in case of a race
                    await asyncio.sleep(1.0)
                    continue

                consecutive_empty = 0
                batch_num += 1
                self.stats.current_batch = batch_num

                if self.config.max_batches and batch_num > self.config.max_batches:
                    logger.info("max_batches_reached", max=self.config.max_batches)
                    break

                # Process batch
                batch_failed = await self._process_batch(batch)
                failed_ids.update(batch_failed)

                # If the entire batch failed, the embedder is probably down — stop early
                if len(batch_failed) == len(batch):
                    logger.error("entire_batch_failed",
                                 batch=batch_num, error=self.stats.last_error)
                    break

                # Delay between batches (reduce GPU pressure)
                if self.config.delay_between_batches > 0:
                    await asyncio.sleep(self.config.delay_between_batches)

            # Build relationships if enabled
            if self.config.build_relationships and not self._stop_requested:
                await self._build_relationships()

            self.stats.completed_at = datetime.now(timezone.utc)
            self.state = WorkerState.IDLE

            logger.info(
                "embedding_job_completed",
                embedded=self.stats.total_embedded,
                failed=self.stats.total_failed,
                relationships=self.stats.relationships_created,
                duration_seconds=self.stats.to_dict()["duration_seconds"]
            )

        except Exception as e:
            self.stats.last_error = str(e)
            self.state = WorkerState.IDLE
            logger.error("embedding_worker_error", error=str(e))
            raise

    async def _process_batch(self, batch: List[Tuple[str, str]]) -> set:
        """Process a batch of records. Returns set of failed record IDs."""
        updates = []
        batch_failed: set = set()

        for record_id, content in batch:
            self.stats.total_processed += 1

            if self._stop_requested:
                break

            try:
                # Clip to ~500 tokens (1500 chars at ~3 chars/token), the same
                # retrieval-precision size as CHUNK_SIZE_CHARS (not a model
                # limit; the stack's embed num_ctx is 1024 tokens). Backend-side
                # truncation is only a safety net, and we want the embedding to
                # represent the whole clipped content, not a backend-chosen prefix.
                clipped = content[:1500] if len(content) > 1500 else content

                # Generate embedding
                embedding = await asyncio.get_event_loop().run_in_executor(
                    None, self.embedder, clipped
                )
                updates.append((record_id, embedding))
                self.stats.total_embedded += 1

            except Exception as e:
                self.stats.total_failed += 1
                self.stats.last_error = str(e)
                batch_failed.add(record_id)
                logger.warning("embedding_failed", record_id=record_id, error=str(e))
                try:
                    from ..observability import inc_counter
                    await inc_counter("pages_embedding_failures_total")
                except Exception:
                    pass

        # Per-batch counter — bumped once per successful batch insert below,
        # not per-record, so it tracks Prometheus rate() faithfully.

        # Batch update to database
        if updates:
            try:
                count = await self.store.batch_update_embeddings(updates)
                logger.debug("batch_embedded", count=count, batch=self.stats.current_batch)
                try:
                    from ..observability import inc_counter
                    await inc_counter("pages_embeddings_completed_total", value=count)
                except Exception:
                    pass
            except Exception as e:
                logger.error("batch_update_failed", error=str(e))
                self.stats.last_error = str(e)

        return batch_failed

    async def _build_relationships(self) -> None:
        """Build similarity relationships between embedded records."""
        logger.info("building_relationships")

        # Get recently embedded records for relationship building
        recent_records = await self.store.get_recent_embedded(limit=1000)

        for record_id, embedding in recent_records:
            if self._stop_requested:
                break

            # Find similar records via store's semantic search
            similar = await self.store.search_semantic(
                embedding=embedding,
                top_k=self.config.max_relationships_per_record + 1  # +1 to exclude self
            )

            # Filter by threshold and exclude self
            relationships = [
                (entry["id"], score)
                for entry, score in similar
                if entry["id"] != record_id and score >= self.config.similarity_threshold
            ][:self.config.max_relationships_per_record]

            # Store relationships
            for target_id, score in relationships:
                await self.store.store_relationship(record_id, target_id, score)
                self.stats.relationships_created += 1

        logger.info("relationships_built", count=self.stats.relationships_created)

    def _create_embedder(self) -> Optional[Callable[[str], List[float]]]:
        """Create an embedder function from config."""
        try:
            import httpx
        except ImportError:
            logger.error("failed_to_create_embedder", error="httpx not installed")
            return None

        from trellion.live._tls import ssl_verify as _ssl_verify, client_cert as _client_cert

        backend = self.config.embedding_backend
        base_url = self.config.embedding_base_url

        # Transport is the gateway, always — same doctrine as
        # pageindex_embedder. The gateway's /api/external/embeddings owns
        # the ONE implementation of embedding transport: mesh discovery by
        # placement role, mTLS to the node-agent proxy, the per-engine
        # dialect registry and failover. The tei/ollama direct dialects
        # that used to live here were the last copies in the tree; the
        # tei one had been dead in production since the harrier seam (the
        # gateway 404s POST /embed). A configured non-gateway backend is
        # warned about and ignored rather than honoured.
        if backend and backend.lower() not in ("gateway", ""):
            logger.warning(
                "embedding_worker.backend_ignored",
                configured=backend,
                used="gateway",
            )
        base_url = base_url or "https://localhost:8080"
        embedding_space.configure_url(base_url)

        def embed_gateway(text: str) -> List[float]:
            # The one door: gateway_auth. A second signer beside
            # gateway_auth_header() would present a different credential than
            # the rest of the process. The door sends its own key, or raises
            # GatewayCredentialMissing.
            from .gateway_auth import GatewayHmacAuth

            with httpx.Client(
                timeout=60.0, verify=_ssl_verify(), cert=_client_cert(), auth=GatewayHmacAuth()
            ) as client:
                response = client.post(
                    f"{base_url}/api/external/embeddings",
                    json={"input": [text], "model": embedding_space.space_name()},
                )
                response.raise_for_status()
                # A Vector: the list plus the model the response named, so the
                # store records embedding_model for the row it writes.
                out = embedding_space.from_response(response.json())
                if not out:
                    raise RuntimeError("gateway returned no embedding")
                return out[0]

        logger.info("embedder_created", backend="gateway", url=base_url)
        return embed_gateway


def probe_embedding_dim(embedder: Callable[[str], List[float]]) -> int:
    """Send a test string to the embedder and return the output dimension."""
    vec = embedder("dimension probe")
    return len(vec)


# Global worker instance
_worker: Optional[EmbeddingWorker] = None


def get_embedding_worker() -> Optional[EmbeddingWorker]:
    """Get the global embedding worker instance."""
    return _worker


def init_embedding_worker(
    config: EmbeddingConfig,
    store: Optional[Any] = None,
    embedder: Optional[Callable] = None,
    # Deprecated aliases for backward compat
    vector_store: Optional[Any] = None,
    conversation_store: Optional[Any] = None,
) -> EmbeddingWorker:
    """Initialize the global embedding worker."""
    global _worker
    _worker = EmbeddingWorker(
        config=config,
        store=store or conversation_store or vector_store,
        embedder=embedder,
    )
    return _worker
