# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""PageIndex embedding pipeline.

Processes unembedded pages and chunks via an embedding backend
(HuggingFace TEI *or* Ollama), stores vectors back into the
PageIndex, and publishes NATS events.

Supported backends (set via ``backend`` constructor arg):
    - ``"tei"``    — HuggingFace Text Embeddings Inference (``/embed``)
    - ``"ollama"`` — Ollama (``/api/embed``, requires ``model`` param)

Background loop:
    1. Embed pages (primary retrieval unit) in batches
    2. Embed chunks in batches
    3. Sleep, repeat

NATS events published:
    - index.page.embedded  {page_id, doc_id}
    - index.chunk.embedded {chunk_id, page_id}

NATS events consumed:
    - ingest.doc.parsed  -> triggers immediate embedding pass
"""

from __future__ import annotations

import asyncio
from enum import Enum
from typing import Optional

import httpx
import structlog

from trellion import embedding_space
from trellion.live._tls import ssl_verify, client_cert
from trellion.live.embed_throttle import EmbedThrottle, EmbedUnavailable, get_throttle
from trellion.live.pageindex_store import PageIndexStore
from trellion.live.reembed import ReembedWorker

log = structlog.get_logger(__name__)


class EmbedderState(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    STOPPING = "stopping"


class PageIndexEmbedder:
    """Embed pages and chunks via TEI or Ollama and store vectors in PageIndex.

    When a CapabilityRouter is available, embedding calls automatically
    fall back through: preferred node → gossip peers → local Ollama CPU.
    """

    def __init__(
        self,
        store: PageIndexStore,
        embedding_url: str = "https://gateway:8080",
        messaging: Optional[object] = None,
        batch_size: int = 32,
        backend: str = "tei",
        model: Optional[str] = None,
        capability_router: Optional[object] = None,
        throttle: Optional[EmbedThrottle] = None,
    ) -> None:
        self.store = store
        # Process-wide by default so the page, chunk and graph-entity loops
        # (and add-on callers) share one in-flight bound and one AIMD
        # state. Retry/backoff on 503s lives there, not in tenacity here.
        self.throttle = throttle or get_throttle()
        self.embedding_url = embedding_url.rstrip("/")
        embedding_space.configure_url(self.embedding_url)
        self.messaging = messaging  # ClusterMessaging | None
        self.batch_size = batch_size
        # Transport is the gateway, always. `backend` and `capability_router`
        # are accepted so existing callers keep working, but are no longer
        # honoured: engine selection, mesh discovery and failover all belong
        # to the Rust client now, and a second implementation here is what
        # let this module keep posting to a retired TEI after the fleet had
        # moved on. Warn loudly rather than silently ignoring configuration.
        self.backend = "gateway"
        if backend and backend.lower() not in ("gateway", ""):
            log.warning(
                "pageindex_embedder.backend_ignored",
                requested=backend,
                note="embedding transport is gateway-only; the engine is chosen "
                     "by the role=embed placement, not by this setting",
            )
        if capability_router is not None:
            log.warning(
                "pageindex_embedder.capability_router_ignored",
                note="the gateway does its own mesh discovery and failover",
            )
        # The model is NOT configured here. The role=embed placement declares
        # what the fleet's embedder serves, and the node-agent fills it in.
        # Pinning a model in config is what broke the harrier cutover: config
        # said bge-base-en-v1.5 while the placement said harrier.
        self.model = model
        self._capability_router = None

        self.state = EmbedderState.IDLE
        self._stop_requested = False
        self._task: Optional[asyncio.Task] = None
        self._client: Optional[httpx.AsyncClient] = None
        self._wake_event = asyncio.Event()
        self._stop_event = asyncio.Event()
        # Provenance upkeep + space re-embed (live/reembed.py). Built on
        # start(), once the store's engine is known to be usable.
        self._reembed: Optional[ReembedWorker] = None

    # ------------------------------------------------------------------
    # Embedding calls (TEI or Ollama)
    # ------------------------------------------------------------------

    async def embed_text(self, text: str) -> list[float]:
        """Embed a single (interactive) text.

        Not bounded by the backfill semaphore, so a query is never queued
        behind a backlog drain; fewer retries so it fails fast.
        """
        assert self._client is not None

        out = await self.throttle.call(
            lambda: self._embed_gateway([text]),
            what="text", count=1, max_attempts=2, bounded=False,
        )
        return out[0] if out else []

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts in one call (bounded, retried with backoff).

        Raises :class:`EmbedUnavailable` when the backend stays overloaded;
        callers leave the rows unembedded so a later pass retries them.
        """
        if not texts:
            return []
        assert self._client is not None

        return await self.throttle.call(
            lambda: self._embed_gateway(texts), what="batch", count=len(texts)
        )

    # -- Gateway backend (preferred) -----------------------------------
    #
    # Delegates to the Rust gateway's /api/external/embeddings, which owns
    # the ONE implementation of embedding transport: mesh discovery by
    # placement role, mTLS to the node-agent proxy, the per-engine dialect
    # registry, GPU-preferred candidate ordering and failover.
    #
    # The alternative was teaching this client mTLS, node-agent bearer auth,
    # capability discovery and every engine's wire format — a second copy of
    # logic that must then be kept in sync. They had already drifted: this
    # client kept talking to a retired TEI while the Rust client had moved
    # to a mesh-routed embedder, and the corpus silently depended on which
    # one wrote a given row.
    #
    # Python still decides WHAT to embed (which rows are stale, batching,
    # writing vectors back). Rust decides WHERE it goes. The extra hop is
    # loopback and negligible against model inference.

    async def _embed_gateway(
        self, inputs: list[str], model: Optional[str] = None
    ) -> embedding_space.Embeddings:
        """POST one embeddings call, ALWAYS naming the space as ``model``.

        ``model`` defaults to the configured space (TRELLION_EMBEDDING_SPACE);
        the re-embed passes ``"<space>@next"``. The result is a list of
        vectors that also carries the model the RESPONSE named (``.model``,
        and on each vector), which is what a writer records as provenance.
        """
        resp = await self._client.post(
            f"{self.embedding_url}/api/external/embeddings",
            json={"input": inputs, "model": model or embedding_space.space_name()},
        )
        resp.raise_for_status()
        # OpenAI shape, sorted by index (the contract does not promise order).
        return embedding_space.from_response(resp.json())

    async def _embed_next(self, texts: list[str]) -> embedding_space.Embeddings:
        """Embed with the space's NEXT model (a running migration)."""
        assert self._client is not None
        return await self.throttle.call(
            lambda: self._embed_gateway(texts, embedding_space.next_alias()),
            what="reembed", count=len(texts),
        )

    async def _embed_text_gateway(self, text: str) -> list[float]:
        out = await self._embed_gateway([text])
        return out[0] if out else []

    async def _embed_batch_gateway(self, texts: list[str]) -> list[list[float]]:
        return await self._embed_gateway(texts)

    # ------------------------------------------------------------------
    # Processing loops
    # ------------------------------------------------------------------

    async def _gather_batches(self, one, batch_size: int) -> int:
        """Run up to ``throttle.max_inflight`` claimed batches concurrently.

        Each batch claims its own disjoint rows; a batch that fails rolls back
        and releases its rows without failing its siblings. With
        ``max_inflight == 1`` this is one batch at a time, as before.
        """
        n = max(1, self.throttle.max_inflight)
        done = await asyncio.gather(*(one(batch_size) for _ in range(n)))
        return sum(done)

    async def process_unembedded_pages(self, batch_size: int = 32) -> int:
        """Claim unembedded pages, embed, store. Returns count embedded."""
        return await self._gather_batches(self._page_batch, batch_size)

    async def process_unembedded_chunks(self, batch_size: int = 32) -> int:
        """Claim unembedded chunks, embed, store. Returns count embedded."""
        return await self._gather_batches(self._chunk_batch, batch_size)

    async def _page_batch(self, batch_size: int) -> int:
        stage = "claim"
        pages: list[dict] = []
        stored = 0
        try:
            async with self.store.claim_unembedded_pages(
                self.throttle.effective_batch(batch_size)
            ) as claim:
                pages = claim.rows
                if not pages:
                    return 0
                stage = "embed"
                embeddings = await self.embed_batch([p["text"] for p in pages])
                stage = "store"
                stored = await claim.store(
                    [(p["page_id"], e) for p, e in zip(pages, embeddings)]
                )
        except EmbedUnavailable as exc:
            # Expected under load: one line, no traceback, rows released.
            log.warning(
                "page_embed_batch_failed",
                count=len(pages),
                reason=exc.reason,
                attempts=exc.attempts,
                retry_in_s=round(self.throttle.cooldown_s(), 1),
            )
            return 0
        except Exception:
            log.exception(
                "store_page_embeddings_batch_failed" if stage == "store" else "page_embed_batch_failed",
                count=len(pages),
            )
            return 0

        for page in pages[:stored]:
            try:
                await self._publish_page_embedded(page["page_id"])
            except Exception:
                pass

        log.info("pages_embedded", count=stored, batch=len(pages))
        return stored

    async def _chunk_batch(self, batch_size: int) -> int:
        stage = "claim"
        chunks: list[dict] = []
        stored = 0
        try:
            async with self.store.claim_unembedded_chunks(
                self.throttle.effective_batch(batch_size)
            ) as claim:
                chunks = claim.rows
                if not chunks:
                    return 0
                stage = "embed"
                embeddings = await self.embed_batch([c["text"] for c in chunks])
                stage = "store"
                stored = await claim.store(
                    [(c["chunk_id"], e) for c, e in zip(chunks, embeddings)]
                )
        except EmbedUnavailable as exc:
            log.warning(
                "chunk_embed_batch_failed",
                count=len(chunks),
                reason=exc.reason,
                attempts=exc.attempts,
                retry_in_s=round(self.throttle.cooldown_s(), 1),
            )
            return 0
        except Exception:
            log.exception(
                "store_chunk_embeddings_batch_failed" if stage == "store" else "chunk_embed_batch_failed",
                count=len(chunks),
            )
            return 0

        for chunk in chunks[:stored]:
            try:
                await self._publish_chunk_embedded(chunk["chunk_id"], chunk.get("page_id"))
            except Exception:
                pass

        log.info("chunks_embedded", count=stored, batch=len(chunks))
        return stored

    # ------------------------------------------------------------------
    # NATS publishing helpers
    # ------------------------------------------------------------------

    async def _publish_page_embedded(self, page_id: str) -> None:
        if not self.messaging or not getattr(self.messaging, "is_connected", False):
            return
        # Look up doc_id for the page
        page = await self.store.get_page(page_id)
        doc_id = page["doc_id"] if page else None
        await self.messaging.publish_event(
            "index.page.embedded",
            {"page_id": page_id, "doc_id": doc_id},
        )

    async def _publish_chunk_embedded(
        self, chunk_id: str, page_id: Optional[str] = None
    ) -> None:
        if not self.messaging or not getattr(self.messaging, "is_connected", False):
            return
        await self.messaging.publish_event(
            "index.chunk.embedded",
            {"chunk_id": chunk_id, "page_id": page_id},
        )

    # ------------------------------------------------------------------
    # NATS subscription
    # ------------------------------------------------------------------

    async def _subscribe_doc_parsed(self) -> None:
        """Subscribe to ingest.doc.parsed to trigger immediate embedding."""
        if not self.messaging or not getattr(self.messaging, "is_connected", False):
            return

        async def _on_doc_parsed(event: dict) -> None:
            doc_id = event.get("doc_id")
            log.info("doc_parsed_received", doc_id=doc_id)
            # Wake the background loop immediately
            self._wake_event.set()

        await self.messaging.subscribe_events(
            "ingest.doc.parsed",
            _on_doc_parsed,
            durable_name="pageindex_embedder_doc_parsed",
        )
        log.info("subscribed_to_ingest_doc_parsed")

    # ------------------------------------------------------------------
    # Lifecycle (state machine)
    # ------------------------------------------------------------------

    async def start(self, interval: float = 5.0) -> bool:
        """Start the background embedding loop.

        Args:
            interval: Seconds between processing cycles when idle.

        Returns:
            True if started, False if already running.
        """
        if self.state == EmbedderState.RUNNING:
            log.warning("pageindex_embedder_already_running")
            return False

        self._stop_requested = False
        self._wake_event.clear()
        self._stop_event.clear()
        self.state = EmbedderState.RUNNING
        # The credential rides the shared auth hook, not a header built per
        # call: the hook re-signs each request, falls back to the key when a
        # surface refuses an exchanged token, and backs off on a refusal
        # (noomesh followup 436a). This path was the 2026-09-22 403 storm,
        # and a hand-built header had none of the three.
        from trellion.live.gateway_auth import GatewayHmacAuth

        self._client = httpx.AsyncClient(
            timeout=60.0, verify=ssl_verify(), cert=client_cert(), auth=GatewayHmacAuth()
        )

        await self._subscribe_doc_parsed()

        engine = getattr(self.store, "_engine", None)
        self._reembed = (
            ReembedWorker(engine, self._embed_next, batch_size=self.batch_size)
            if engine is not None
            else None
        )

        self._task = asyncio.create_task(self._run_loop(interval))
        log.info(
            "pageindex_embedder_started",
            embedding_url=self.embedding_url,
            batch_size=self.batch_size,
            interval=interval,
        )
        return True

    async def stop(self) -> None:
        """Request graceful shutdown and wait for the loop to finish."""
        if self.state != EmbedderState.RUNNING:
            return

        self._stop_requested = True
        self.state = EmbedderState.STOPPING
        self._wake_event.set()  # unblock any sleep
        self._stop_event.set()
        log.info("pageindex_embedder_stop_requested")

        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=30.0)
            except asyncio.TimeoutError:
                self._task.cancel()
                log.warning("pageindex_embedder_force_cancelled")

        if self._client:
            await self._client.aclose()
            self._client = None

        self.state = EmbedderState.IDLE
        log.info("pageindex_embedder_stopped")

    def get_status(self) -> dict:
        return {
            "state": self.state.value,
            "backend": self.backend,
            "model": self.model,
            "embedding_url": self.embedding_url,
            "batch_size": self.batch_size,
        }

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def _rest(self, seconds: float) -> None:
        """Sleep *seconds*, but wake early on stop."""
        if seconds <= 0 or self._stop_requested:
            return
        try:
            await asyncio.wait_for(self._stop_event.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            pass

    async def _run_loop(self, interval: float) -> None:
        """Process unembedded records in a loop until stopped."""
        try:
            # Containers restarted together must not begin together. Only a
            # stop request cuts the wait short.
            jitter = self.throttle.startup_jitter_s
            if jitter > 0:
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(),
                        timeout=self.throttle._rng.uniform(0, jitter),
                    )
                except asyncio.TimeoutError:
                    pass
            while not self._stop_requested:
                total = 0

                # Provenance upkeep and any running re-embed. Never fatal: a
                # failure here must not stop ordinary embedding.
                swapping = False
                if self._reembed is not None:
                    try:
                        total += await self._reembed.run_pass()
                        swapping = await self._reembed.in_cutover()
                    except Exception:
                        log.exception("reembed_pass_failed")

                if self._stop_requested:
                    break

                # While the space is being swapped to its new model, hold
                # ordinary writes: a vector embedded by the OLD model and
                # written after the swap would be stranded.
                if swapping:
                    await self._rest(interval)
                    continue

                # Pages first (primary retrieval unit)
                pages_done = await self.process_unembedded_pages(self.batch_size)
                total += pages_done

                if self._stop_requested:
                    break

                # Then chunks
                chunks_done = await self.process_unembedded_chunks(self.batch_size)
                total += chunks_done

                if self._stop_requested:
                    break

                if total == 0 and self.throttle.consecutive_failures > 0:
                    # Work exists but the backend refused it: back off (a
                    # doc-parsed wake must not cut this short), then retry the
                    # same rows - nothing was marked embedded.
                    await self._rest(max(interval, self.throttle.cooldown_s()))
                    continue

                if total == 0:
                    # Nothing to process - wait for interval or wake event
                    self._wake_event.clear()
                    try:
                        await asyncio.wait_for(
                            self._wake_event.wait(), timeout=interval
                        )
                    except asyncio.TimeoutError:
                        pass
                    continue

                # More work remains: pace the drain instead of hammering. After
                # failed batches the rest grows (jittered exponential), so a
                # saturated embedder is given room to recover.
                if self.throttle.consecutive_failures > 0:
                    rest = self.throttle.cooldown_s()
                else:
                    rest = self.throttle.pace_s
                await self._rest(rest)

        except httpx.ConnectError:
            log.warning("pageindex_embedder_loop_error", reason="embedding backend unavailable", url=self.embedding_url)
        except Exception:
            log.exception("pageindex_embedder_loop_error")
        finally:
            self.state = EmbedderState.IDLE
