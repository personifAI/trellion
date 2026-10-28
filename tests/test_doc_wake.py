# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""A parsed document wakes the embedding loop through Postgres LISTEN/NOTIFY.

Ingest (noo-trellion) and the embedding loop (noo-trellion-intel) are different
containers: the in-process event and the NATS subject never reached the loop
(the add-on's NATS user cannot subscribe to it, the ingest user cannot publish
it), so the first embed waited out the idle back-off. The database is shared.
"""
from __future__ import annotations

import asyncio

from sqlalchemy.engine import make_url

from trellion.live import doc_wake
from trellion.live import pageindex_embedder as pe_mod
from trellion.live import rag_integration as ri
from trellion.live.pageindex_embedder import PageIndexEmbedder
from trellion.live.pageindex_ingestion import DocumentIngestor


class _Conn:
    def __init__(self):
        self.executed: list = []

    async def execute(self, sql, params=None):
        self.executed.append((str(sql), params))


class _Engine:
    """Records every statement run in a transaction (`begin`)."""

    url = make_url("postgresql+psycopg://u:p@db:5432/noomesh?sslmode=verify-full")

    def __init__(self, fail=False):
        self.conn = _Conn()
        self.fail = fail

    def begin(self):
        engine = self

        class _Tx:
            async def __aenter__(self):
                if engine.fail:
                    raise RuntimeError("db down")
                return engine.conn

            async def __aexit__(self, *a):
                return False

        return _Tx()


class _Store:
    def __init__(self, engine=None):
        self._engine = engine


class _ListenConn:
    """A fake psycopg connection: records LISTEN, lets the test fire notifications."""

    def __init__(self):
        self.listened: list[str] = []
        self.queue: asyncio.Queue = asyncio.Queue()
        self.closed = False

    async def execute(self, sql, *a):
        self.listened.append(sql)

    async def notifies(self):
        while True:
            item = await self.queue.get()
            if isinstance(item, Exception):
                raise item
            yield item

    async def close(self):
        self.closed = True

    def notify(self, payload="doc-1"):
        self.queue.put_nowait(payload)


class _Bus:
    """connect() factory: hands out fake connections, records each attempt."""

    def __init__(self):
        self.conns: list[_ListenConn] = []
        self.conninfos: list[str] = []

    async def __call__(self, conninfo):
        self.conninfos.append(conninfo)
        c = _ListenConn()
        self.conns.append(c)
        return c


async def _until(pred, timeout=2.0):
    end = asyncio.get_event_loop().time() + timeout
    while not pred():
        assert asyncio.get_event_loop().time() < end, "condition not reached"
        await asyncio.sleep(0.01)


# (a) ingest commits -> NOTIFY with the doc_id; listener callback -> wake -------


async def test_ingest_notifies_the_channel_with_the_doc_id():
    eng = _Engine()
    ing = DocumentIngestor(_Store(eng), messaging=None)
    await ing._publish_event("doc-42", "T", 3, "file:///x")
    assert len(eng.conn.executed) == 1
    sql, params = eng.conn.executed[0]
    assert "pg_notify" in sql
    assert params == {"c": doc_wake.CHANNEL, "d": "doc-42"}


async def test_a_failed_notify_never_fails_the_ingest():
    ing = DocumentIngestor(_Store(_Engine(fail=True)), messaging=None)
    await ing._publish_event("doc-42", "T", 3, "file:///x")  # must not raise


async def test_notification_wakes_a_waiting_loop_well_before_the_idle_timeout(monkeypatch):
    bus = _Bus()
    monkeypatch.setattr(doc_wake, "_psycopg_connect", bus)
    e = PageIndexEmbedder(_Store(_Engine()), embedding_url="https://gateway:8080")
    monkeypatch.setattr(e.throttle, "startup_jitter_s", 0)
    monkeypatch.setattr(e.throttle, "idle_max_s", 60.0)
    e._reembed = None
    passes = []

    async def nothing(_n):
        passes.append(asyncio.get_event_loop().time())
        return 0

    monkeypatch.setattr(e, "process_unembedded_pages", nothing)
    monkeypatch.setattr(e, "process_unembedded_chunks", nothing)
    await e.start(interval=30.0)
    try:
        await _until(lambda: len(passes) >= 1 and bus.conns and bus.conns[0].listened)
        assert bus.conns[0].listened == [f"LISTEN {doc_wake.CHANNEL}"]
        # the DSN is the store's database, password included, driver stripped
        assert bus.conninfos[0].startswith("postgresql://u:p@db:5432/noomesh")
        n = len(passes)
        bus.conns[0].notify("doc-42")
        await _until(lambda: len(passes) > n, timeout=1.0)  # idle wait is 30 s
    finally:
        await e.stop()
    assert bus.conns[0].closed, "stop() closes the listener connection"


# (b) no LISTEN unless start() runs -------------------------------------------


async def test_background_workers_off_opens_no_listener(monkeypatch):
    bus = _Bus()
    monkeypatch.setattr(doc_wake, "_psycopg_connect", bus)
    monkeypatch.setenv(ri.BACKGROUND_WORKERS_ENV, "0")

    class _Schema:
        def __init__(self, dsn):
            pass

        async def ensure_tables(self):
            pass

    class _S(_Store):
        def __init__(self, dsn):
            super().__init__(_Engine())

        async def connect(self):
            pass

        async def close(self):
            pass

    monkeypatch.setattr(ri, "PageIndexSchema", _Schema)
    monkeypatch.setattr(ri, "PageIndexStore", _S)
    monkeypatch.setattr(ri, "_ENTITY_EXTRACTOR_AVAILABLE", False)
    monkeypatch.setattr(ri, "_GRAPH_BUILDER_AVAILABLE", False)
    rag = ri.PageIndexRAG(dsn="postgresql://x/y")
    await rag.startup()
    await asyncio.sleep(0.05)
    assert bus.conns == [], "a process with the loops off must never LISTEN"
    assert rag.embedder._doc_wake is None
    await rag.shutdown()


async def test_constructing_an_embedder_opens_no_listener(monkeypatch):
    bus = _Bus()
    monkeypatch.setattr(doc_wake, "_psycopg_connect", bus)
    PageIndexEmbedder(_Store(_Engine()), embedding_url="https://gateway:8080")
    await asyncio.sleep(0.05)
    assert bus.conns == []


# (c) a lost connection reconnects and wakes again ----------------------------


async def test_lost_listener_reconnects_with_backoff_and_wakes_again():
    bus = _Bus()
    woke = []
    lst = doc_wake.DocWakeListener("postgresql://x", lambda: woke.append(1), connect=bus, reconnect_min_s=0.01, reconnect_max_s=0.05)
    await lst.start()
    try:
        await _until(lambda: len(bus.conns) == 1 and bus.conns[0].listened)
        bus.conns[0].queue.put_nowait(ConnectionError("server closed the connection"))
        await _until(lambda: len(bus.conns) == 2 and bus.conns[1].listened)
        assert bus.conns[0].closed
        bus.conns[1].notify()
        await _until(lambda: woke)
    finally:
        await lst.stop()
    assert bus.conns[1].closed


async def test_unreachable_database_backs_off_instead_of_spinning():
    attempts = []

    async def down(conninfo):
        attempts.append(1)
        raise OSError("refused")

    lst = doc_wake.DocWakeListener("postgresql://x", lambda: None, connect=down, reconnect_min_s=0.05, reconnect_max_s=0.2)
    await lst.start()
    await asyncio.sleep(0.4)
    await lst.stop()
    assert 2 <= len(attempts) <= 6, attempts
