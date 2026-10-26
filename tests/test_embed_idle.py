"""An idle embedder is cheap: no vector-carrying join, and a backing-off poll.

pop-os, 2026-10-26: the backlog pick ``LEFT JOIN chunk_embeddings ... WHERE
ce.embedding IS NULL`` hashed every vector (974 chunks, ~136 MB) and spilled
it to a temp file on every idle poll, every 5 s, from each process: ~130 MB/s
of writes for a week with nothing to embed.
"""
from __future__ import annotations

import asyncio
import inspect

from trellion.live import pageindex_embedder as pe_mod
from trellion.live import pageindex_store
from trellion.live.pageindex_embedder import IDLE_MAX_S, PageIndexEmbedder


def test_needs_embedding_is_not_exists_not_a_vector_carrying_join():
    src = inspect.getsource(pageindex_store)
    for table, key in (("chunk_embeddings", "chunk_id"), ("page_embeddings", "page_id")):
        assert f"LEFT JOIN {table}" not in src, f"{table}: a LEFT JOIN carries the vector through the join"
        alias = "ce" if table == "chunk_embeddings" else "pe"
        probe = f"NOT EXISTS (SELECT 1 FROM {table} {alias}"
        # get_unembedded_* and claim_unembedded_* both use the probe.
        assert src.count(probe) == 2, f"{table}: expected the NOT EXISTS probe in the get and claim queries"
        assert f"{alias}.embedding IS NOT NULL" in src


class _Idle:
    """A store-free embedder whose backlog is always empty."""


async def test_idle_wait_doubles_to_the_cap_and_a_wake_resets_it(monkeypatch):
    e = PageIndexEmbedder(_Idle(), embedding_url="https://gateway:8080")
    e._reembed = None
    monkeypatch.setattr(e.throttle, "startup_jitter_s", 0)

    async def nothing(_batch_size):
        return 0

    monkeypatch.setattr(e, "process_unembedded_pages", nothing)
    monkeypatch.setattr(e, "process_unembedded_chunks", nothing)

    waits: list[float] = []
    woke_at = 3  # the 4th idle wait is cut short by a wake (a parsed doc)
    real_wait_for = asyncio.wait_for

    async def fake_wait_for(aw, timeout=None):
        if timeout is None:
            return await real_wait_for(aw, timeout)
        waits.append(timeout)
        if inspect.iscoroutine(aw):
            aw.close()
        if len(waits) == woke_at + 1:
            return True  # woken
        if len(waits) >= 12:
            e._stop_requested = True
        raise asyncio.TimeoutError

    monkeypatch.setattr(pe_mod.asyncio, "wait_for", fake_wait_for)
    await real_wait_for(e._run_loop(5.0), 5)

    assert waits[:4] == [5.0, 10.0, 20.0, 40.0], waits
    assert waits[4] == 5.0, f"a wake must reset the back-off: {waits}"
    assert max(waits) == IDLE_MAX_S, waits
    assert all(w <= IDLE_MAX_S for w in waits)


async def test_a_long_idle_spell_never_overflows(monkeypatch):
    """The streak is capped, so days of idling cannot overflow the wait."""
    e = PageIndexEmbedder(_Idle(), embedding_url="https://gateway:8080")
    e._reembed = None
    monkeypatch.setattr(e.throttle, "startup_jitter_s", 0)

    async def nothing(_batch_size):
        return 0

    monkeypatch.setattr(e, "process_unembedded_pages", nothing)
    monkeypatch.setattr(e, "process_unembedded_chunks", nothing)
    n = {"waits": 0}
    real_wait_for = asyncio.wait_for

    async def fake_wait_for(aw, timeout=None):
        if timeout is None:
            return await real_wait_for(aw, timeout)
        if inspect.iscoroutine(aw):
            aw.close()
        n["waits"] += 1
        assert timeout <= IDLE_MAX_S
        if n["waits"] >= 2000:
            e._stop_requested = True
        raise asyncio.TimeoutError

    monkeypatch.setattr(pe_mod.asyncio, "wait_for", fake_wait_for)
    await real_wait_for(e._run_loop(5.0), 10)
    assert n["waits"] >= 2000
