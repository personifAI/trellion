# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""Claim, commit, embed, write back with a source recheck.

The re-embed worker and the page/chunk backlog loops used to hold a
``FOR NO KEY UPDATE SKIP LOCKED`` transaction (and a pooled connection) open
while they embedded. With ``NOO_EMBED_BULK`` that wait is minutes. They now pick
rows in a short transaction, embed with nothing open, and write back in a second
short transaction that updates a row only if its source is still what was read
and it still needs the work.

The first half needs no database: a fake engine counts open connections and
transactions, and only the HTTP transport is faked. The second half (``needs_db``)
runs the real SQL against Postgres (``TRELLION_TEST_DSN``) and proves the
recheck itself.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections import namedtuple
from contextlib import asynccontextmanager

import httpx
import pytest
import structlog.testing

from trellion import embedding_space as es
from trellion.embedding_space import Embeddings, SpaceRecord
from trellion.live.embed_throttle import EmbedThrottle
from trellion.live.pageindex_embedder import PageIndexEmbedder
from trellion.live.pageindex_store import PageIndexStore
from trellion.live.reembed import TABLES, ReembedWorker
from tests.test_embed_claim import world  # noqa: F401  (fixture)
from tests.test_embedding_provenance import (  # noqa: F401  (fixtures + helpers)
    _embed_world, _fetch, filling, fresh_space, gw, store_exec,
)

DSN = os.environ.get("TRELLION_TEST_DSN")
needs_db = pytest.mark.skipif(not DSN, reason="TRELLION_TEST_DSN not set")
DIM = 1024
SPEC = {t.name: t for t in TABLES}


def vec(x: float = 1.0) -> list[float]:
    return [x] + [0.0] * (DIM - 1)


def md5(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()


# ----------------------------------------------------------------------
# A fake engine that knows how many connections / transactions are open
# ----------------------------------------------------------------------


class _Res:
    def __init__(self, rows=(), rowcount=0):
        self._rows = list(rows)
        self.rowcount = rowcount

    def fetchall(self):
        return self._rows

    def scalar(self):
        return None


class FakeEngine:
    """Records SQL; ``open`` counts live connect()/begin() blocks.

    ``select`` maps a SQL needle to the rows a SELECT containing it returns
    (a callable gets the bind params). ``rowcount`` is what each UPDATE/INSERT
    reports: a callable of ``(sql, params)``.
    """

    def __init__(self, select=None, rowcount=lambda sql, params: 1):
        self.open = 0
        self.log: list[tuple[str, dict]] = []
        self.select = select or {}
        self.rowcount = rowcount

    @asynccontextmanager
    async def connect(self):
        self.open += 1
        try:
            yield _Conn(self)
        finally:
            self.open -= 1

    begin = connect

    def writes(self):
        return [(s, p) for s, p in self.log
                if s.lstrip().upper().startswith(("UPDATE", "INSERT"))]


class _Conn:
    def __init__(self, eng):
        self.eng = eng

    def begin(self):
        return self.eng.connect()

    async def commit(self):
        pass

    async def execute(self, stmt, params=None):
        sql, params = str(stmt), params or {}
        self.eng.log.append((sql, params))
        if sql.lstrip().upper().startswith("SELECT"):
            for needle, rows in self.eng.select.items():
                if needle in sql:
                    return _Res(rows(params) if callable(rows) else rows)
            return _Res()
        if isinstance(params, list):  # executemany
            return _Res(rowcount=len(params))
        return _Res(rowcount=self.eng.rowcount(sql, params))


PageRow = namedtuple("PageRow", "page_id text seen")


def embedder(store, handler) -> PageIndexEmbedder:
    th = EmbedThrottle(
        max_inflight=1, batch_max=8, batch_min=8, pace_s=0, startup_jitter_s=0,
        backoff_base_s=0.01, backoff_cap_s=0.02,
    )
    e = PageIndexEmbedder(store, embedding_url="https://gateway:8080", throttle=th)
    e._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return e


def reply(request: httpx.Request) -> httpx.Response:
    n = len(json.loads(request.content)["input"])
    return httpx.Response(
        200, json={"data": [{"index": i, "embedding": vec(float(i + 1))} for i in range(n)]}
    )


def page_store(eng) -> PageIndexStore:
    s = PageIndexStore(async_engine=eng, embedding_dim=DIM)
    s._dims_cache = {"page_embeddings.embedding": DIM, "chunk_embeddings.embedding": DIM}
    return s


# ----------------------------------------------------------------------
# No database: nothing is open while the embed runs
# ----------------------------------------------------------------------


async def test_the_page_loop_holds_no_connection_or_transaction_during_the_embed():
    pages = [PageRow(str(uuid.uuid4()), f"page {i}", md5(f"page {i}")) for i in range(3)]
    eng = FakeEngine(select={"FROM pages p": pages})
    during: list[int] = []

    async def handler(request):
        during.append(eng.open)
        return reply(request)

    e = embedder(page_store(eng), handler)
    assert await e.process_unembedded_pages(8) == 3
    assert during == [0]  # the embed ran with the claim committed and released
    assert len(eng.writes()) == 3


async def test_the_chunk_loop_holds_no_connection_or_transaction_during_the_embed():
    Row = namedtuple("Row", "chunk_id page_id text seen")
    rows = [Row(str(uuid.uuid4()), str(uuid.uuid4()), "c", md5("c"))]
    eng = FakeEngine(select={"FROM page_chunks c": rows})
    during: list[int] = []

    async def handler(request):
        during.append(eng.open)
        return reply(request)

    e = embedder(page_store(eng), handler)
    assert await e.process_unembedded_chunks(8) == 1
    assert during == [0]


async def test_the_write_back_passes_the_md5_read_at_pick_time_and_counts_only_written_rows():
    pages = [PageRow(str(uuid.uuid4()), f"page {i}", md5(f"page {i}")) for i in range(3)]
    # the middle row changed (or was embedded elsewhere): the guarded insert hits nothing
    eng = FakeEngine(
        select={"FROM pages p": pages},
        rowcount=lambda sql, p: 0 if p["page_id"] == pages[1].page_id else 1,
    )
    published: list[str] = []

    class Bus:
        is_connected = True

        async def publish_event(self, subject, payload):
            published.append(payload["page_id"])

    st = page_store(eng)

    async def get_page(page_id):
        return {"doc_id": "d"}

    st.get_page = get_page
    e = embedder(st, lambda r: reply(r))
    e.messaging = Bus()
    with structlog.testing.capture_logs() as logs:
        assert await e.process_unembedded_pages(8) == 2  # not 3: no double count
    assert published == [pages[0].page_id, pages[2].page_id]
    sent = {p["page_id"]: p["seen"] for _, p in eng.writes()}
    assert sent == {p.page_id: p.seen for p in pages}
    sql = eng.writes()[0][0]
    assert "md5(p.text) = :seen" in sql and "page_embeddings.embedding IS NULL" in sql
    assert [x["skipped"] for x in logs if x["event"] == "embed_writeback_skipped"] == [1]


async def test_concurrent_batches_in_one_process_never_pick_the_same_rows():
    pages = [PageRow(str(uuid.uuid4()), f"page {i}", md5(f"page {i}")) for i in range(4)]
    eng = FakeEngine(select={
        "FROM pages p": lambda p: [r for r in pages if r.page_id not in p["skip"]][: p["limit"]],
    })
    st = page_store(eng)
    async with st.claim_unembedded_pages(2) as a:
        async with st.claim_unembedded_pages(2) as b:
            assert {r["page_id"] for r in a.rows}.isdisjoint({r["page_id"] for r in b.rows})
            assert len(a.rows) == len(b.rows) == 2
        # b is done: its rows are pickable again, a's are not
        async with st.claim_unembedded_pages(4) as c:
            assert {r["page_id"] for r in c.rows} == {r.page_id for r in pages[2:]}
    assert st._inflight["page_id"] == set()


async def test_picks_started_together_are_still_disjoint():
    """``_gather_batches`` starts its picks at once; they must take turns."""
    import asyncio

    pages = [PageRow(str(uuid.uuid4()), f"page {i}", md5(f"page {i}")) for i in range(6)]
    eng = FakeEngine()

    class Slow(_Conn):
        async def execute(self, stmt, params=None):
            res = await super().execute(stmt, params)
            if str(stmt).lstrip().upper().startswith("SELECT"):
                chosen = [r for r in pages if r.page_id not in params["skip"]][: params["limit"]]
                await asyncio.sleep(0)  # a concurrent pick would interleave here
                res = _Res(chosen)
            return res

    @asynccontextmanager
    async def connect():
        eng.open += 1
        try:
            yield Slow(eng)
        finally:
            eng.open -= 1

    eng.connect = connect
    st = page_store(eng)

    async def pick():
        async with st.claim_unembedded_pages(2) as c:
            await asyncio.sleep(0.01)
            return [r["page_id"] for r in c.rows]

    got = await asyncio.gather(pick(), pick(), pick())
    flat = [i for g in got for i in g]
    assert len(flat) == len(set(flat)) == 6


async def test_a_failed_embed_releases_the_in_process_pick():
    pages = [PageRow(str(uuid.uuid4()), "p", md5("p"))]
    eng = FakeEngine(select={
        "FROM pages p": lambda p: [r for r in pages if r.page_id not in p["skip"]],
    })
    st = page_store(eng)
    with pytest.raises(RuntimeError):
        async with st.claim_unembedded_pages(2):
            raise RuntimeError("embed failed")
    async with st.claim_unembedded_pages(2) as again:
        assert len(again.rows) == 1


NEXT = "n-model"
REC = SpaceRecord("knowledge", "m-old", DIM, True,
                  {"state": "filling", "next_model": NEXT, "next_dims": DIM})


def next_embed(eng, seen_open):
    async def embed(texts):
        seen_open.append(eng.open)
        return Embeddings([vec(float(i + 1)) for i in range(len(texts))], model=NEXT)

    return embed


async def test_the_reembed_text_fill_holds_nothing_open_and_rechecks_the_source():
    rows = [(str(uuid.uuid4()), "alpha", md5("alpha")), (str(uuid.uuid4()), "beta", md5("beta"))]
    eng = FakeEngine(
        select={"md5(": rows},
        rowcount=lambda sql, p: 0 if p["pk"] == rows[1][0] else 1,  # beta changed meanwhile
    )
    during: list[int] = []
    w = ReembedWorker(eng, next_embed(eng, during), batch_size=8)
    with structlog.testing.capture_logs() as logs:
        assert await w._fill_text(SPEC["page_embeddings"], REC) == 2
    assert during == [0]
    sent = {p["pk"]: p for _, p in eng.writes()}
    assert {pk: p["seen"] for pk, p in sent.items()} == {r[0]: r[2] for r in rows}
    assert all(p["next"] == NEXT for p in sent.values())
    sql = eng.writes()[0][0]
    assert "md5(s.text) IS NOT DISTINCT FROM :seen" in sql
    assert "embedding_next_model IS DISTINCT FROM :next" in sql
    assert [x["skipped"] for x in logs if x["event"] == "reembed.writeback_skipped"] == [1]


async def test_the_stale_sweep_holds_nothing_open_and_rechecks_staleness_and_source():
    rows = [(str(uuid.uuid4()), "alpha", md5("alpha"))]
    eng = FakeEngine(select={"md5(": rows}, rowcount=lambda sql, p: 0)  # healed elsewhere
    during: list[int] = []
    w = ReembedWorker(eng, None, embed_live=next_embed(eng, during), batch_size=8)
    handled, written = await w._stale_batch(
        SPEC["page_embeddings"], SpaceRecord("knowledge", NEXT, DIM, True, None), NEXT, "live:x"
    )
    assert (handled, written) == (1, 0) and during == [0]
    ((sql, params),) = eng.writes()
    assert "embedding_model <> :cur" in sql and "md5(s.text) IS NOT DISTINCT FROM :seen" in sql
    assert params["seen"] == rows[0][2] and params["cur"] == NEXT


def _entity_rows(name):
    return [(uuid.UUID(int=1), name, "t", None, None, None, None)]


async def test_an_entity_whose_description_changed_during_the_embed_is_not_written():
    eid = str(uuid.UUID(int=1))
    state = {"name": "before"}
    eng = FakeEngine(select={
        "SELECT t.entity_id FROM entity_embeddings": [(uuid.UUID(int=1),)],
        "FROM entities ent": lambda p: _entity_rows(state["name"]),
    })
    during: list[int] = []

    async def embed(texts):
        during.append(eng.open)
        state["name"] = "after"  # the entity was renamed while the embed was in flight
        return Embeddings([vec()], model=NEXT)

    w = ReembedWorker(eng, embed, batch_size=8)
    with structlog.testing.capture_logs() as logs:
        assert await w._fill_entities(SPEC["entity_embeddings"], REC) == 1
    assert during == [0]
    assert eng.writes() == []  # nothing stale was written
    assert [x["skipped"] for x in logs if x["event"] == "reembed.writeback_skipped"] == [1]

    # unchanged between embed and write: written, with the staged-model guard
    state["name"] = "before"

    async def steady(texts):
        return Embeddings([vec()], model=NEXT)

    w = ReembedWorker(eng, steady, batch_size=8)
    await w._fill_entities(SPEC["entity_embeddings"], REC)
    ((sql, params),) = eng.writes()
    assert params["pk"] == eid and "embedding_next_model IS DISTINCT FROM :next" in sql


# ----------------------------------------------------------------------
# Postgres: the recheck itself
# ----------------------------------------------------------------------


@needs_db
@pytest.mark.parametrize("kind", ["page", "chunk"])
async def test_a_row_whose_text_changed_between_pick_and_write_is_not_written(world, kind):
    store, tag, make = world
    (row,) = await make(1)
    table, col, ident = (
        ("pages", "page_embeddings", "page_id") if kind == "page"
        else ("page_chunks", "chunk_embeddings", "chunk_id")
    )
    async with getattr(store, f"claim_unembedded_{kind}s")(8) as claim:
        mine = [r for r in claim.rows if r[ident] == row[ident]]
        assert mine
        # the source changes while the vector is being computed
        await store_exec(store, f"UPDATE {table} SET text = text || ' edited' "
                         f"WHERE {ident} = CAST(:i AS uuid)", i=row[ident])
        assert await claim.store([(row[ident], vec())]) == 0
        assert claim.written == []
    got = await _fetch(store, f"SELECT 1 FROM {col} WHERE {ident} = CAST(:i AS uuid)", i=row[ident])
    assert got == []  # still unembedded: the next pass picks it up with the new text
    async with getattr(store, f"claim_unembedded_{kind}s")(8) as again:
        (fresh,) = [r for r in again.rows if r[ident] == row[ident]]
        assert fresh["text"].endswith(" edited")
        assert await again.store([(row[ident], vec())]) == 1


@needs_db
@pytest.mark.parametrize("kind", ["page", "chunk"])
async def test_a_row_picked_by_two_workers_is_written_once_and_the_first_vector_stays(world, kind):
    store, tag, make = world
    (row,) = await make(1)
    ident = f"{kind}_id"
    col = f"{kind}_embeddings"
    other = PageIndexStore(async_engine=store._engine, embedding_dim=DIM)  # a second process
    async with getattr(store, f"claim_unembedded_{kind}s")(1_000_000) as a, \
            getattr(other, f"claim_unembedded_{kind}s")(1_000_000) as b:
        assert row[ident] in {r[ident] for r in a.rows}
        assert row[ident] in {r[ident] for r in b.rows}  # no lock: both picked it
        assert await a.store([(row[ident], vec(1.0))]) == 1
        assert await b.store([(row[ident], vec(2.0))]) == 0  # already embedded: a no-op
        assert a.written == [row[ident]] and b.written == []
    got = await _fetch(store, f"SELECT (embedding::real[])[1] FROM {col} WHERE {ident} = CAST(:i AS uuid)",
                       i=row[ident])
    assert [float(r[0]) for r in got] == [1.0]


@needs_db
async def test_the_backlog_loop_holds_no_pooled_connection_while_it_embeds(world):
    store, tag, make = world
    await make(2)
    pool = store._engine.pool
    out: list[int] = []

    async def handler(request):
        out.append(pool.checkedout())
        return reply(request)

    e = embedder(store, handler)
    assert await e.process_unembedded_pages(8) >= 2
    assert out and set(out) == {0}


@needs_db
async def test_an_ordinary_vector_write_is_not_blocked_by_a_pick_that_is_embedding(world):
    import asyncio

    store, tag, make = world
    (row,) = await make(1)
    async with store.claim_unembedded_pages(1_000_000):
        await asyncio.wait_for(
            store.store_page_embedding(row["page_id"], vec()), timeout=5
        )  # the pick's row lock is released: this used to wait for the whole embed


@needs_db
async def test_reembed_skips_a_text_row_changed_during_the_embed_and_stages_it_next_pass(world, gw):
    store, tag, make = world
    gw.model = f"m-{tag}"
    nxt = f"n-{tag}"
    rows, e = await _embed_world(store, gw, make, 3)
    pids = [r["page_id"] for r in rows]
    gw.migration = filling(tag)
    fresh_space()
    changed = {"done": False}

    async def embed(texts):
        out = await e._embed_next(texts)
        if not changed["done"] and any(f"{tag} page 0" in t for t in texts):
            changed["done"] = True
            await store_exec(store, "UPDATE pages SET text = text || ' edited' "
                             "WHERE page_id = CAST(:p AS uuid)", p=pids[0])
        return out

    w = ReembedWorker(store.engine, embed, batch_size=64)
    sql = ("SELECT page_id::text, embedding_next_model FROM page_embeddings "
           "WHERE page_id = ANY(CAST(:ids AS uuid[]))")
    with structlog.testing.capture_logs() as logs:
        await w.run_pass()
    assert changed["done"]
    got = {r[0]: r[1] for r in await _fetch(store, sql, ids=pids)}
    assert got[pids[0]] is None and got[pids[1]] == nxt and got[pids[2]] == nxt
    assert any(x["event"] == "reembed.writeback_skipped" and x["table"] == "page_embeddings"
               for x in logs)
    for _ in range(10):  # the normal selection finds it again, now with the new text
        await w.run_pass()
    got = {r[0]: r[1] for r in await _fetch(store, sql, ids=pids)}
    assert set(got.values()) == {nxt}


@needs_db
async def test_two_reembed_workers_on_one_row_stage_it_once(world, gw):
    store, tag, make = world
    gw.model = f"m-{tag}"
    nxt = f"n-{tag}"
    rows, e = await _embed_world(store, gw, make, 2)
    pids = [r["page_id"] for r in rows]
    gw.migration = filling(tag)
    fresh_space()
    rec = await es.aget_space()
    spec = SPEC["page_embeddings"]
    b = ReembedWorker(store.engine, e._embed_next, batch_size=64)
    ran = {"b": False}

    async def embed_a(texts):
        out = await e._embed_next(texts)
        if not ran["b"]:  # worker B picks the same rows and finishes while A is embedding
            ran["b"] = True
            await b._fill_text(spec, rec)
        return out

    a = ReembedWorker(store.engine, embed_a, batch_size=64)
    with structlog.testing.capture_logs() as logs:
        await a._fill_text(spec, rec)
    assert ran["b"]
    got = await _fetch(store, "SELECT embedding_next_model FROM page_embeddings "
                       "WHERE page_id = ANY(CAST(:ids AS uuid[]))", ids=pids)
    assert {r[0] for r in got} == {nxt}
    skipped = [x for x in logs if x["event"] == "reembed.writeback_skipped"]
    assert skipped and sum(x["skipped"] for x in skipped) >= len(pids)  # A's writes were no-ops


@needs_db
async def test_stale_sweep_does_not_overwrite_a_row_another_worker_already_healed(world, gw):
    store, tag, make = world
    gw.model = f"m-{tag}"
    rows, e = await _embed_world(store, gw, make, 1)
    pid = rows[0]["page_id"]
    await store_exec(store, "UPDATE page_embeddings SET embedding_model = 'm-stale' "
                     "WHERE page_id = CAST(:p AS uuid)", p=pid)
    gw.migration = None
    fresh_space()
    rec = await es.aget_space()

    async def embed_live(texts):
        out = await e.embed_batch(texts)
        # someone else heals the row (current model) while this worker embeds
        await store_exec(store, "UPDATE page_embeddings SET embedding_model = :m "
                         "WHERE page_id = CAST(:p AS uuid)", m=f"m-{tag}", p=pid)
        return Embeddings(out, model="m-from-the-slow-worker")

    w = ReembedWorker(store.engine, e._embed_next, embed_live=embed_live, batch_size=64)
    handled, _written = await w._stale_batch(SPEC["page_embeddings"], rec, rec.model, f"live:{rec.model}")
    got = await _fetch(store, "SELECT embedding_model FROM page_embeddings "
                       "WHERE page_id = CAST(:p AS uuid)", p=pid)
    assert handled >= 1 and got[0][0] == f"m-{tag}"  # not overwritten by the slow worker
