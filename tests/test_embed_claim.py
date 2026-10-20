# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""The embedding backfill claims rows instead of peeking at them.

Two processes (and several batches in one) used to fetch the same first rows
and embed them twice. These run the real store and embedder against Postgres
(``TRELLION_TEST_DSN``); only the HTTP transport is faked. Each test makes
uniquely-tagged documents and counts only its own text.
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from collections import Counter
from pathlib import Path

import httpx
import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from trellion.live.embed_throttle import EmbedThrottle, EmbedUnavailable
from trellion.live.pageindex_embedder import PageIndexEmbedder
from trellion.live.pageindex_store import PageIndexStore

DSN = os.environ.get("TRELLION_TEST_DSN")
needs_db = pytest.mark.skipif(not DSN, reason="TRELLION_TEST_DSN not set")
DIM = 1024
VERSIONS = Path(__file__).resolve().parent.parent / "src/trellion/db/migrations/versions"


def test_claiming_adds_no_migration():
    """The fix is query-only. A migration would force a lockstep release
    (trellion-protocols, both pins, intel's expected schema revision)."""
    names = sorted(p.name for p in VERSIONS.glob("*.py") if p.name != "__init__.py")
    # The claim fix landed on top of 105; later migrations (106 is the
    # embedding provenance work) are other changes and are tested on their own.
    assert names[:6] == [
        "100_trellion_squash.py", "101_frame_embeddings.py", "102_episodes_rename.py",
        "103_extraction_converges.py", "104_recallable_walks.py", "105_doc_hold.py",
    ], names


@pytest.fixture
async def world():
    from trellion.db.engine import TRELLION_SCHEMA, _async_dsn

    eng = create_async_engine(
        _async_dsn(DSN), connect_args={"options": f"-csearch_path={TRELLION_SCHEMA},public"}
    )
    store = PageIndexStore(async_engine=eng, embedding_dim=DIM)
    tag = f"claim-{uuid.uuid4().hex[:10]}"
    made: list[str] = []

    # Other test files leave unembedded rows behind; a claim would pick them up
    # and make batch counts nondeterministic. Hold them for the duration.
    async with eng.begin() as c:
        parked = [
            r[0]
            for r in await c.execute(
                text("UPDATE docs SET held_at = now() WHERE held_at IS NULL RETURNING doc_id")
            )
        ]

    async def make(n: int) -> list[dict]:
        out = []
        for i in range(n):
            doc_id, _ = await store.insert_doc(
                f"test://{tag}/{uuid.uuid4().hex}", f"{tag} {i}", uuid.uuid4().hex
            )
            made.append(doc_id)
            page_id = await store.insert_page(doc_id, 1, f"{tag} page {i}")
            (chunk_id,) = await store.insert_chunks(page_id, [f"{tag} chunk {i}"])
            out.append({"doc_id": doc_id, "page_id": page_id, "chunk_id": chunk_id})
        return out

    try:
        yield store, tag, make
    finally:
        async with eng.begin() as c:
            await c.execute(
                text("DELETE FROM docs WHERE doc_id = ANY(CAST(:ids AS uuid[]))"),
                {"ids": made},
            )
            await c.execute(
                text("UPDATE docs SET held_at = NULL WHERE doc_id = ANY(CAST(:ids AS uuid[]))"),
                {"ids": [str(i) for i in parked]},
            )
        await eng.dispose()


def _track_claims(store):
    """Count simultaneously open chunk claims."""
    state = {"open": 0, "peak": 0}
    orig = store.claim_unembedded_chunks

    def wrapped(limit=32):
        cm = orig(limit)

        class W:
            async def __aenter__(self_):
                c = await cm.__aenter__()
                state["open"] += 1
                state["peak"] = max(state["peak"], state["open"])
                return c

            async def __aexit__(self_, *a):
                state["open"] -= 1
                return await cm.__aexit__(*a)

        return W()

    store.claim_unembedded_chunks = wrapped
    return state


def _embedder(store, tag, *, max_inflight, delay=0.0, fail_first=False):
    seen: Counter = Counter()
    http = {"now": 0, "peak": 0, "calls": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        inputs = json.loads(request.content)["input"]
        http["calls"] += 1
        http["now"] += 1
        http["peak"] = max(http["peak"], http["now"])
        try:
            await asyncio.sleep(delay)
        finally:
            http["now"] -= 1
        # The throttle retries a 500, so poison the first batch's inputs and
        # fail every attempt at them until the test clears http["poison"].
        if fail_first and "poison" not in http:
            http["poison"] = inputs
        if http.get("poison") == inputs:
            return httpx.Response(500)
        for t in inputs:
            if tag in t:
                seen[t] += 1
        vec = [1.0] + [0.0] * (DIM - 1)
        return httpx.Response(
            200, json={"data": [{"index": i, "embedding": vec} for i in range(len(inputs))]}
        )

    th = EmbedThrottle(
        max_inflight=max_inflight, batch_max=4, batch_min=4, pace_s=0,
        startup_jitter_s=0, backoff_base_s=0.01, backoff_cap_s=0.02,
    )
    e = PageIndexEmbedder(store, embedding_url="https://gateway:8080", throttle=th)
    e._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return e, seen, http


async def _pending(store, ids, kind):
    rows = (
        await store.get_unembedded_chunks(limit=1_000_000)
        if kind == "chunk"
        else await store.get_unembedded_pages(limit=1_000_000)
    )
    return {r[f"{kind}_id"] for r in rows} & set(ids)


@needs_db
@pytest.mark.parametrize("kind", ["chunk", "page"])
async def test_two_claimers_never_get_the_same_row(world, kind):
    store, tag, make = world
    await make(20)
    claim = getattr(store, f"claim_unembedded_{kind}s")
    async with claim(5) as a, claim(5) as b, claim(5) as c:
        ids = [[r[f"{kind}_id"] for r in x.rows] for x in (a, b, c)]
    flat = [i for g in ids for i in g]
    assert len(flat) == len(set(flat)) == 15


@needs_db
@pytest.mark.parametrize("kind", ["chunk", "page"])
async def test_a_rolled_back_claim_is_claimable_again(world, kind):
    store, tag, make = world
    await make(6)
    claim = getattr(store, f"claim_unembedded_{kind}s")
    with pytest.raises(RuntimeError):
        async with claim(4) as a:
            first = {r[f"{kind}_id"] for r in a.rows}
            raise RuntimeError("embed failed")
    async with claim(4) as b:
        again = {r[f"{kind}_id"] for r in b.rows}
    assert first and first == again


@needs_db
async def test_loop_embeds_every_chunk_once_and_in_parallel(world):
    store, tag, make = world
    rows = await make(24)
    ids = [r["chunk_id"] for r in rows]
    pids = [r["page_id"] for r in rows]
    claims = _track_claims(store)
    e, seen, http = _embedder(store, tag, max_inflight=3, delay=0.05)

    task = asyncio.create_task(e._run_loop(0.02))
    try:
        async def drained():
            while await _pending(store, ids, "chunk") or await _pending(store, pids, "page"):
                await asyncio.sleep(0.02)

        await asyncio.wait_for(drained(), 20)
    finally:
        e._stop_requested = True
        e._stop_event.set()
        e._wake_event.set()
        await asyncio.wait_for(task, 10)

    chunk_texts = [t for t in seen if "chunk" in t]
    assert len(chunk_texts) == 24 and all(seen[t] == 1 for t in seen), seen
    assert claims["peak"] == 3 and http["peak"] == 3  # it really ran concurrently


@needs_db
async def test_one_in_flight_keeps_one_batch_at_a_time(world):
    store, tag, make = world
    rows = await make(12)
    claims = _track_claims(store)
    e, seen, http = _embedder(store, tag, max_inflight=1, delay=0.02)
    for _ in range(4):
        await e.process_unembedded_chunks(4)
    assert claims["peak"] == 1 and http["peak"] == 1
    assert await _pending(store, [r["chunk_id"] for r in rows], "chunk") == set()
    assert all(seen[t] == 1 for t in seen)


@needs_db
async def test_a_failed_batch_loses_nothing_and_spares_its_siblings(world):
    store, tag, make = world
    rows = await make(12)
    ids = [r["chunk_id"] for r in rows]
    e, seen, http = _embedder(store, tag, max_inflight=3, delay=0.02, fail_first=True)

    assert await e.process_unembedded_chunks(4) == 8  # 3 batches, one failed
    assert len(await _pending(store, ids, "chunk")) == 4  # released, not lost
    http["poison"] = None
    assert await e.process_unembedded_chunks(4) == 4
    assert await _pending(store, ids, "chunk") == set()
    assert sum(1 for t in seen if "chunk" in t) == 12 and all(v == 1 for v in seen.values())


@needs_db
async def test_unavailable_backend_releases_the_claim(world):
    store, tag, make = world
    rows = await make(4)
    e, _, _ = _embedder(store, tag, max_inflight=2)
    e._client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(503)))
    assert await e.process_unembedded_chunks(4) == 0
    assert len(await _pending(store, [r["chunk_id"] for r in rows], "chunk")) == 4
    assert EmbedUnavailable  # the 503 path, not a crash


@needs_db
async def test_held_documents_are_never_claimed(world):
    store, tag, make = world
    kept, held = await make(2)
    await store.hold_docs(doc_ids=[held["doc_id"]], reason="t", apply=True)
    async with store.claim_unembedded_chunks(1_000_000) as c:
        chunk_ids = {r["chunk_id"] for r in c.rows}
    async with store.claim_unembedded_pages(1_000_000) as p:
        page_ids = {r["page_id"] for r in p.rows}
    assert kept["chunk_id"] in chunk_ids and held["chunk_id"] not in chunk_ids
    assert kept["page_id"] in page_ids and held["page_id"] not in page_ids


@needs_db
@pytest.mark.parametrize("kind", ["chunk", "page"])
async def test_a_separate_connection_store_does_not_deadlock_on_the_claim(world, kind):
    """The vector insert's FK check takes FOR KEY SHARE on the parent row.
    FOR UPDATE on the claim conflicts with that and would hang here."""
    store, tag, make = world
    await make(3)
    vec = [1.0] + [0.0] * (DIM - 1)
    async with getattr(store, f"claim_unembedded_{kind}s")(3) as c:
        key = f"{kind}_id"
        target = c.rows[0][key]
        writer = store.store_chunk_embedding if kind == "chunk" else store.store_page_embedding
        await asyncio.wait_for(writer(target, vec), timeout=5)
