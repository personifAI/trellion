# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""The embedding backlog drains steadily instead of stampeding.

Embedding calls retry on 503 with backoff and bounded concurrency, log one
line per failure rather than a traceback, and do not re-stampede on the next
tick.

These drive the REAL ``PageIndexEmbedder`` through ``httpx.MockTransport``
(the transport is faked, never the call) with a fake store.
"""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager

import httpx
import pytest
import structlog

from trellion.live.pageindex_embedder import PageIndexEmbedder

PAGE_TEXT = "SECRET-PAGE-TEXT-do-not-log"


@pytest.fixture(autouse=True)
def _fresh_throttle(monkeypatch: pytest.MonkeyPatch):
    # Tiny delays so the suite is fast; the knobs are the declared env ones.
    monkeypatch.setenv("TRELLION_EMBED_BACKOFF_BASE_S", "0.01")
    monkeypatch.setenv("TRELLION_EMBED_BACKOFF_CAP_S", "0.05")
    monkeypatch.setenv("TRELLION_EMBED_PACE_S", "0")
    monkeypatch.setenv("TRELLION_EMBED_STARTUP_JITTER_S", "0")
    try:
        from trellion.live.embed_throttle import reset_throttle
    except ImportError:  # pre-change tree: nothing to reset
        yield
        return
    reset_throttle()
    yield
    reset_throttle()


class _FakeClaim:
    def __init__(self, store, rows) -> None:
        self._store, self.rows = store, rows

    async def store(self, items):
        ids = [i for i, _ in items]
        self._store.stored.extend(ids)
        self._store.pending = [p for p in self._store.pending if p["page_id"] not in ids]
        return len(items)


class FakeStore:
    def __init__(self, n_pages: int = 100) -> None:
        self.pending = [{"page_id": f"p{i}", "text": PAGE_TEXT} for i in range(n_pages)]
        self.limits: list[int] = []
        self.stored: list[str] = []

    @asynccontextmanager
    async def claim_unembedded_pages(self, limit: int = 32):
        self.limits.append(limit)
        yield _FakeClaim(self, self.pending[:limit])

    @asynccontextmanager
    async def claim_unembedded_chunks(self, limit: int = 32):
        class Claim:
            rows: list = []

        yield Claim()

    async def get_page(self, page_id):
        return None


def _ok(request: httpx.Request) -> httpx.Response:
    import json

    n = len(json.loads(request.content)["input"])
    return httpx.Response(200, json={"data": [{"index": i, "embedding": [0.1, 0.2]} for i in range(n)]})


def _embedder(handler, store=None) -> PageIndexEmbedder:
    e = PageIndexEmbedder(store or FakeStore(), embedding_url="https://gateway:8080")
    e._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return e


async def test_in_flight_embedding_requests_are_bounded() -> None:
    state = {"now": 0, "peak": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        state["now"] += 1
        state["peak"] = max(state["peak"], state["now"])
        await asyncio.sleep(0.01)
        state["now"] -= 1
        return _ok(request)

    e = _embedder(handler)
    # Page loop, chunk loop, graph embedder and classifier all share a process.
    await asyncio.gather(*(e.embed_batch([f"t{i}"]) for i in range(8)))
    assert state["peak"] == 1


async def test_a_503_backs_off_and_is_retried_not_dropped() -> None:
    calls: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(time.monotonic())
        if len(calls) <= 2:
            return httpx.Response(503, json={"error": "embedder busy"})
        return _ok(request)

    e = _embedder(handler)
    out = await e.embed_batch(["a", "b"])
    assert len(out) == 2 and len(calls) == 3
    assert calls[1] - calls[0] >= 0.004  # waited between attempts, not a hot loop


async def test_retry_after_is_honoured() -> None:
    calls: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(time.monotonic())
        if len(calls) == 1:
            return httpx.Response(503, headers={"Retry-After": "1"})
        return _ok(request)

    from trellion.live.embed_throttle import get_throttle

    get_throttle().backoff_cap_s = 0.3
    e = _embedder(handler)
    await e.embed_batch(["a"])
    assert calls[1] - calls[0] >= 0.29


async def test_a_batch_the_backend_never_takes_stays_unembedded() -> None:
    store = FakeStore(10)
    e = _embedder(lambda r: httpx.Response(503), store)
    assert await e.process_unembedded_pages(32) == 0
    assert store.stored == [] and len(store.pending) == 10  # still queued


async def test_errors_shrink_the_batch_and_success_restores_it() -> None:
    store = FakeStore(400)
    mode = {"fail": True}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503) if mode["fail"] else _ok(request)

    e = _embedder(handler, store)
    for _ in range(3):
        await e.process_unembedded_pages(32)
    shrunk = store.limits[-1]
    assert shrunk < 32 and shrunk >= 1
    mode["fail"] = False
    for _ in range(12):
        await e.process_unembedded_pages(32)
    assert store.limits[-1] == 32


async def test_an_expected_503_logs_a_warning_without_traceback_or_text() -> None:
    store = FakeStore(5)
    e = _embedder(lambda r: httpx.Response(503), store)
    with structlog.testing.capture_logs() as logs:
        await e.process_unembedded_pages(32)
    failed = [x for x in logs if x["event"] == "page_embed_batch_failed"]
    assert failed, logs
    for ev in failed:
        assert ev["log_level"] == "warning"
        assert not ev.get("exc_info")
        assert ev["count"] == 5
    assert not any(PAGE_TEXT in repr(x) for x in logs)


async def test_an_unexpected_exception_keeps_its_traceback() -> None:
    store = FakeStore(3)
    e = _embedder(lambda r: httpx.Response(200, content=b"not json"), store)
    with structlog.testing.capture_logs() as logs:
        assert await e.process_unembedded_pages(32) == 0
    failed = [x for x in logs if x["event"] == "page_embed_batch_failed"]
    assert failed and failed[0]["log_level"] == "error" and failed[0].get("exc_info")


async def test_the_loop_waits_a_start_up_jitter_before_its_first_batch() -> None:
    from trellion.live.embed_throttle import EmbedThrottle

    class Hi:
        def uniform(self, a, b):
            return b

    store = FakeStore(0)
    th = EmbedThrottle(startup_jitter_s=0.3, rng=Hi())
    e = PageIndexEmbedder(store, embedding_url="https://gateway:8080", throttle=th)
    e._client = httpx.AsyncClient(transport=httpx.MockTransport(_ok))
    task = asyncio.create_task(e._run_loop(0.05))
    await asyncio.sleep(0.1)
    assert store.limits == []  # still inside the jitter window
    await asyncio.sleep(0.4)
    assert store.limits
    e._stop_requested = True
    e._stop_event.set()
    e._wake_event.set()
    await asyncio.wait_for(task, 2)
