# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC
"""search_chunks' global shape under a filter (noomesh-internal i785).

Two properties, against real Postgres (the db-test job sets TRELLION_TEST_DSN):

* audience: a chunk the caller may not read is never returned, even when it is
  the nearest one (a permission check: the filter stays in the one statement);
* fill: when the nearest chunks are ones the caller may not read, the search
  still returns top_k readable chunks instead of silently coming back short
  (hnsw.iterative_scan). The index is forced to drive the plan, with a small
  ef_search, so the test exercises the HNSW path a 285k-chunk corpus uses; it
  asserts that plan, so it cannot pass vacuously on a sequential scan.
"""

from __future__ import annotations

import os
import random
import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from trellion.live.pageindex_store import PageIndexStore

DSN = os.environ.get("TRELLION_TEST_DSN")
needs_db = pytest.mark.skipif(not DSN, reason="TRELLION_TEST_DSN not set")
DIM = 1024
AXIS = 777  # a direction no other test seeds near


def _vec(main: float, side: float = 0.0, jitter: float = 0.0, rng: random.Random | None = None) -> list[float]:
    v = [0.0] * DIM
    v[AXIS] = main
    v[AXIS + 1] = side
    if jitter and rng:
        for i in range(8):
            v[100 + i] = rng.uniform(-jitter, jitter)
    return v


@pytest.fixture
async def world():
    from trellion.db.engine import TRELLION_SCHEMA, _async_dsn

    # The index drives the plan (as at scale), with a small candidate list, so
    # a filtered-out nearest set is enough to empty a non-iterative scan.
    # enable_sort off as well: the joined, filtered statement otherwise finds a
    # sort-based plan at test size, and the property below would be about an
    # exact scan (the first version of this test passed with the iterative scan
    # removed for that reason).
    opts = (f"-csearch_path={TRELLION_SCHEMA},public -c enable_seqscan=off -c enable_sort=off"
            " -c hnsw.ef_search=10")
    eng = create_async_engine(_async_dsn(DSN), connect_args={"options": opts})
    store = PageIndexStore(async_engine=eng, embedding_dim=DIM)
    tag = f"filt-{uuid.uuid4().hex[:10]}"
    made: list[str] = []

    async def make(title: str, roles: list[str] | None, vectors: list[list[float]]) -> list[str]:
        meta = {"roles": roles} if roles is not None else None
        doc_id, _ = await store.insert_doc(f"test://{tag}/{uuid.uuid4().hex}", f"{tag} {title}", uuid.uuid4().hex, metadata=meta)
        made.append(doc_id)
        page_id = await store.insert_page(doc_id, 1, f"page of {title}")
        chunk_ids = await store.insert_chunks(page_id, [f"chunk {i} of {title}" for i in range(len(vectors))])
        for cid, v in zip(chunk_ids, vectors):
            await store.store_chunk_embedding(cid, v)
        return chunk_ids

    try:
        yield store, make, eng
    finally:
        async with eng.begin() as c:
            await c.execute(text("DELETE FROM docs WHERE doc_id = ANY(CAST(:ids AS uuid[]))"), {"ids": made})
        await eng.dispose()


@needs_db
async def test_a_restricted_chunk_is_never_returned_even_when_it_is_the_nearest(world):
    store, make, _ = world
    rng = random.Random(785)
    restricted = await make("finance", ["finance"], [_vec(1.0, jitter=0.001, rng=rng) for _ in range(6)])
    public = await make("public", None, [_vec(0.8, 0.6, jitter=0.001, rng=rng) for _ in range(6)])
    q = _vec(1.0)
    got = {c["chunk_id"] for c in await store.search_chunks(q, top_k=5, roles=["user"])}
    assert not got & set(restricted), "a finance chunk reached a caller without the finance role"
    assert got and got <= set(public), got
    allowed = {c["chunk_id"] for c in await store.search_chunks(q, top_k=5, roles=["user", "finance"])}
    assert allowed and allowed <= set(restricted), "the finance role sees the nearest (finance) chunks"


@needs_db
async def test_a_narrow_audience_still_gets_top_k_when_the_nearest_chunks_are_not_its_own(world):
    store, make, eng = world
    rng = random.Random(786)
    # 60 public chunks nearest the query; 6 finance chunks farther away.
    await make("public", None, [_vec(1.0, jitter=0.01, rng=rng) for _ in range(60)])
    finance = await make("finance", ["finance"], [_vec(0.8, 0.6, jitter=0.01, rng=rng) for _ in range(6)])
    q = _vec(1.0)

    from sqlalchemy import event

    seen: list = []

    @event.listens_for(eng.sync_engine, "before_cursor_execute")
    def grab(conn, cursor, statement, params, context, executemany):  # noqa: ANN001
        if "chunk_embeddings" in statement and "ORDER BY" in statement:
            seen[:] = [(statement, params)]

    got = await store.search_chunks(q, top_k=5, roles=["finance"])

    # The HNSW index drives THIS statement's plan (the shape at scale), so the
    # property is about the index walk, not an exact scan of a small table.
    assert seen, "search_chunks issued no chunk statement"
    stmt, params = seen[0]
    async with eng.connect() as c:
        plan = "\n".join(r[0] for r in (await c.exec_driver_sql("EXPLAIN " + stmt, params)).fetchall())
    assert "chunk_embeddings_hnsw_idx" in plan, plan
    ids = [c["chunk_id"] for c in got]
    assert len(ids) == 5, f"a narrow audience came back short ({len(ids)} of 5): the scan stopped at filtered-out candidates"
    assert set(ids) <= set(finance)
    scores = [c["score"] for c in got]
    assert scores == sorted(scores, reverse=True), "results are re-sorted by distance"
