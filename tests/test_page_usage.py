# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""Page usage: one batched, off-path, never-raising count per retrieve call.

The DB tests run against ``TRELLION_TEST_DSN`` (the db-test CI job migrates it
to head, which includes 107).
"""

from __future__ import annotations

import asyncio
import os
import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from trellion.live import pageindex_retriever as pr
from trellion.live.pageindex_retriever import EvidencePack, PageIndexRetriever
from trellion.live.pageindex_store import PageIndexStore

DSN = os.environ.get("TRELLION_TEST_DSN")
needs_db = pytest.mark.skipif(not DSN, reason="TRELLION_TEST_DSN not set")


# ---- a fake engine that records SQL ---------------------------------------


class _Result:
    def __init__(self, rowcount):
        self.rowcount = rowcount


class _Conn:
    def __init__(self, log, fail):
        self._log, self._fail = log, fail

    async def execute(self, stmt, params=None):
        if self._fail:
            raise RuntimeError("database is down")
        self._log.append((str(stmt), dict(params or {})))
        return _Result(len((params or {}).get("ids", [])))


class _Begin:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *a):
        return False


class FakeEngine:
    def __init__(self, fail=False):
        self.statements: list[tuple[str, dict]] = []
        self._fail = fail

    def begin(self):
        return _Begin(_Conn(self.statements, self._fail))


# ---- no database ----------------------------------------------------------


async def test_record_page_usage_is_one_statement_for_many_pages():
    eng = FakeEngine()
    store = PageIndexStore(async_engine=eng, embedding_dim=8)
    n = await store.record_page_usage(["b", "a", "b", "c", ""])
    assert len(eng.statements) == 1
    sql, params = eng.statements[0]
    assert "INSERT INTO page_usage" in sql and "ON CONFLICT (page_id) DO UPDATE" in sql
    assert "retrieved_count + 1" in sql
    assert params["ids"] == ["a", "b", "c"]  # deduplicated and ordered
    assert n == 3


async def test_record_page_usage_with_nothing_runs_nothing():
    eng = FakeEngine()
    store = PageIndexStore(async_engine=eng, embedding_dim=8)
    assert await store.record_page_usage([]) == 0
    assert eng.statements == []


class _Store:
    def __init__(self, engine_fail=False):
        self.calls: list[list[str]] = []
        self.release = None
        self.fail = engine_fail

    async def record_page_usage(self, ids):
        self.calls.append(list(ids))
        if self.release is not None:
            await self.release.wait()
        if self.fail:
            raise RuntimeError("boom")
        return len(ids)


async def _drain():
    while pr._USAGE_TASKS:
        await asyncio.gather(*list(pr._USAGE_TASKS))


async def test_schedule_is_off_path_and_deduplicates():
    store = _Store()
    store.release = asyncio.Event()
    pr.schedule_usage_record(store, ["p2", "p1", "p2"])
    # scheduled, not awaited: the caller got control back before the write ran
    assert len(pr._USAGE_TASKS) == 1
    store.release.set()
    await _drain()
    assert store.calls == [["p2", "p1"]]


async def test_a_failing_write_is_swallowed():
    store = _Store(engine_fail=True)
    pr.schedule_usage_record(store, ["p1"])
    await _drain()  # raises here if the failure escaped
    assert store.calls == [["p1"]]


async def test_schedule_never_raises():
    class Odd:
        record_page_usage = None

    pr.schedule_usage_record(Odd(), ["p1"])
    pr.schedule_usage_record(object(), ["p1"])
    pr.schedule_usage_record(None, ["p1"])
    pr.schedule_usage_record(_Store(), [])
    assert not pr._USAGE_TASKS


async def test_schedule_outside_a_running_loop_does_not_raise():
    # No event loop in this thread: scheduling must be a no-op, not an error.
    def run():
        pr.schedule_usage_record(_Store(), ["p1"])

    await asyncio.get_running_loop().run_in_executor(None, run)
    assert not pr._USAGE_TASKS


async def test_kill_switch(monkeypatch):
    monkeypatch.setenv("TRELLION_PAGE_USAGE", "off")
    store = _Store()
    pr.schedule_usage_record(store, ["p1"])
    await _drain()
    assert store.calls == []


async def test_in_flight_writes_are_capped(monkeypatch):
    monkeypatch.setattr(pr, "_USAGE_MAX_INFLIGHT", 2)
    store = _Store()
    store.release = asyncio.Event()
    for i in range(5):
        pr.schedule_usage_record(store, [f"p{i}"])
    assert len(pr._USAGE_TASKS) == 2
    store.release.set()
    await _drain()
    assert len(store.calls) == 2


class _Embedder:
    async def embed_text(self, q):
        return [0.0]


async def test_retrieve_schedules_one_batched_record_for_the_returned_pages(monkeypatch):
    store = _Store()
    r = PageIndexRetriever.__new__(PageIndexRetriever)
    r.store, r.embedder, r._graph_ranker = store, _Embedder(), None

    async def pages(*a, **k):
        return [{"page_id": "p1", "score": 0.9}, {"page_id": "p2", "score": 0.5}]

    async def merge(emb, pages_, **k):
        return pages_

    async def top(pages_, k):
        return pages_

    async def chunks(*a, **k):
        return []

    async def evidence(pages_, chunks_):
        return [EvidencePack(p["page_id"] + "-doc", p["page_id"], 1, "t", p["score"], "x") for p in pages_]

    r.retrieve_pages, r._merge_chunk_first_pages = pages, merge
    r._diverse_top, r.retrieve_chunks, r.build_evidence = top, chunks, evidence
    monkeypatch.setattr(pr, "_QUERY_EMBED_CACHE", pr.OrderedDict())
    out = await r.retrieve("a question")
    assert [e.page_id for e in out] == ["p1", "p2"]
    await _drain()
    assert store.calls == [["p1", "p2"]]


async def test_retrieve_returns_even_if_recording_fails(monkeypatch):
    store = _Store(engine_fail=True)
    r = PageIndexRetriever.__new__(PageIndexRetriever)
    r.store, r.embedder, r._graph_ranker = store, _Embedder(), None

    async def pages(*a, **k):
        return [{"page_id": "p1", "score": 0.9}]

    async def ident(pages_, *a, **k):
        return pages_

    async def merge(emb, pages_, **k):
        return pages_

    async def chunks(*a, **k):
        return []

    async def evidence(pages_, chunks_):
        return [EvidencePack("d", "p1", 1, "t", 0.9, "x")]

    r.retrieve_pages, r._merge_chunk_first_pages = pages, merge
    r._diverse_top, r.retrieve_chunks, r.build_evidence = ident, chunks, evidence
    monkeypatch.setattr(pr, "_QUERY_EMBED_CACHE", pr.OrderedDict())
    assert len(await r.retrieve("another question")) == 1
    await _drain()


# ---- real Postgres --------------------------------------------------------


@pytest.fixture
async def world():
    from trellion.db.engine import TRELLION_SCHEMA, _async_dsn

    eng = create_async_engine(
        _async_dsn(DSN), connect_args={"options": f"-csearch_path={TRELLION_SCHEMA},public"}
    )
    store = PageIndexStore(async_engine=eng, embedding_dim=1024)
    tag = f"usage-{uuid.uuid4().hex[:10]}"
    made: list[str] = []

    async def make(title: str) -> str:
        doc_id, _ = await store.insert_doc(f"test://{tag}/{uuid.uuid4().hex}", title, uuid.uuid4().hex)
        made.append(doc_id)
        return await store.insert_page(doc_id, 1, f"page of {title}")

    try:
        yield store, make, eng
    finally:
        async with eng.begin() as c:
            await c.execute(
                text("DELETE FROM docs WHERE doc_id = ANY(CAST(:ids AS uuid[]))"), {"ids": made}
            )
        await eng.dispose()


async def _usage(eng, page_id):
    async with eng.connect() as c:
        return (await c.execute(
            text("SELECT retrieved_count, first_retrieved_at, last_retrieved_at "
                 "FROM page_usage WHERE page_id = CAST(:p AS uuid)"), {"p": page_id}
        )).fetchone()


@needs_db
async def test_upsert_increments_once_per_call_and_skips_unknown_pages(world):
    store, make, eng = world
    a, b = await make("usage a"), await make("usage b")
    ghost = str(uuid.uuid4())
    assert await store.record_page_usage([a, a, b, ghost]) == 2  # a deduplicated, ghost skipped
    assert await store.record_page_usage([a]) == 1
    ra, rb = await _usage(eng, a), await _usage(eng, b)
    assert ra[0] == 2 and rb[0] == 1
    assert ra[2] >= ra[1]  # last >= first
    assert await _usage(eng, ghost) is None


@needs_db
async def test_usage_row_goes_with_its_page(world):
    store, make, eng = world
    a = await make("usage gone")
    await store.record_page_usage([a])
    async with eng.begin() as c:
        await c.execute(
            text("DELETE FROM docs WHERE doc_id = (SELECT doc_id FROM pages WHERE page_id = CAST(:p AS uuid))"),
            {"p": a},
        )
    assert await _usage(eng, a) is None
