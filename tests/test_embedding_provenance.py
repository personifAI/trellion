# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""Embedding provenance and the space re-embed (noomesh i150, stream E2).

Two halves. The first needs no database: it fakes the TRANSPORT (httpx
``MockTransport`` for the gateway, a recording engine for SQL) and runs the
real code over it. The second runs the real store and worker against Postgres
(``TRELLION_TEST_DSN``) with only the gateway faked; every test tags its own
rows and uses a unique model name so it never touches another test's data.
"""

from __future__ import annotations

import importlib.util
import json
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import pytest
import structlog
from sqlalchemy import text

from trellion import embedding_space as es
from trellion.live import reembed
from trellion.live.embed_throttle import EmbedThrottle
from trellion.live.pageindex_embedder import PageIndexEmbedder
from trellion.live.pageindex_store import PageIndexStore
from trellion.live.reembed import TABLES, ReembedWorker
from tests.test_embed_claim import world  # noqa: F401  (fixture)

DSN = os.environ.get("TRELLION_TEST_DSN")
needs_db = pytest.mark.skipif(not DSN, reason="TRELLION_TEST_DSN not set")
DIM = 1024
REAL_CLIENT = httpx.Client
SRC = Path(__file__).resolve().parent.parent / "src/trellion"
VERSIONS = SRC / "db/migrations/versions"

VECTOR_TABLES = {
    "chunk_embeddings", "page_embeddings", "entity_embeddings", "entity_clusters",
    "noo_entries", "trellion_concepts", "trellion_traversal_records", "trellion_episodes",
}


def vec(x: float = 1.0) -> list[float]:
    return [x] + [0.0] * (DIM - 1)


# ----------------------------------------------------------------------
# A fake gateway: capabilities, progress PUTs and embeddings
# ----------------------------------------------------------------------


class Gateway:
    """Serves /capabilities for one space and records every request."""

    def __init__(self, model="m-old", migration=None, up=True):
        self.model = model
        self.migration = migration
        self.up = up
        self.puts: list[dict] = []
        self.embeds: list[dict] = []
        self.caps_calls = 0
        self.poison: set[str] = set()  # inputs the gateway rejects with a 400
        self.embed_model = None  # what the response names; defaults to the alias's model

    def space_entry(self):
        return {"space": "knowledge", "model": self.model, "dims": DIM,
                "default": True, "migration": self.migration}

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if not self.up:
            raise httpx.ConnectError("gateway down", request=request)
        if request.method == "GET" and path == "/api/external/capabilities":
            self.caps_calls += 1
            return httpx.Response(200, json={"embeddings": {"spaces": [self.space_entry()]}})
        if request.method == "PUT" and path == "/api/external/embedding-spaces/knowledge/migration":
            body = json.loads(request.content)
            self.puts.append(body)
            return httpx.Response(200, json=self.space_entry())
        if request.method == "POST" and path == "/api/external/embeddings":
            body = json.loads(request.content)
            self.embeds.append(body)
            if self.poison & set(body["input"]):
                return httpx.Response(400, json={"error": "rejected"})
            if body.get("model") == "knowledge@next":
                served = (self.migration or {}).get("next_model")
            else:
                served = self.model
            served = self.embed_model or served
            return httpx.Response(200, json={
                "model": served,
                "data": [{"index": i, "embedding": vec(float(i + 1))}
                         for i in range(len(body["input"]))],
            })
        return httpx.Response(404)

    def client(self) -> httpx.Client:
        return REAL_CLIENT(transport=httpx.MockTransport(self.handler))

    def async_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self.handler))


@pytest.fixture
def gw(monkeypatch):
    monkeypatch.delenv("TRELLION_EMBEDDING_SPACE", raising=False)
    es.reset()
    g = Gateway()
    monkeypatch.setattr(es, "_client", g.client)
    es.configure_url("https://gateway:8080")
    yield g
    es.reset()


def fresh_space():
    """Forget the cached space record (a new gateway state is now live)."""
    es.reset()
    es.configure_url("https://gateway:8080")


def make_embedder(store, g: Gateway) -> PageIndexEmbedder:
    th = EmbedThrottle(
        max_inflight=1, batch_max=4, batch_min=4, pace_s=0, startup_jitter_s=0,
        backoff_base_s=0.01, backoff_cap_s=0.02,
    )
    e = PageIndexEmbedder(store, embedding_url="https://gateway:8080", throttle=th)
    e._client = g.async_client()
    return e


# ----------------------------------------------------------------------
# The space module
# ----------------------------------------------------------------------


def test_space_name_defaults_to_knowledge_and_is_configurable(monkeypatch):
    monkeypatch.delenv("TRELLION_EMBEDDING_SPACE", raising=False)
    assert es.space_name() == "knowledge"
    assert es.next_alias() == "knowledge@next"
    monkeypatch.setenv("TRELLION_EMBEDDING_SPACE", " code ")
    assert es.space_name() == "code"
    assert es.next_alias() == "code@next"
    monkeypatch.setenv("TRELLION_EMBEDDING_SPACE", "  ")
    assert es.space_name() == "knowledge"


def test_response_model_rides_on_the_vectors():
    out = es.from_response({
        "model": "harrier-x",
        "data": [{"index": 1, "embedding": [2.0]}, {"index": 0, "embedding": [1.0]}],
    })
    assert out == [[1.0], [2.0]]  # sorted by index; still plain lists to old code
    assert out.model == "harrier-x"
    assert es.model_of(out[0]) == "harrier-x"
    assert es.model_of([1.0]) is None
    assert es.from_response({"data": []}).model is None


def test_parse_space_reads_the_migration_block():
    payload = {"embeddings": {"spaces": [
        {"space": "a", "model": "ma", "dims": 1024, "default": False, "migration": None},
        {"space": "knowledge", "model": "m1", "dims": 1024, "default": True, "migration": {
            "next_model": "m2", "next_dims": 1024, "state": "filling",
            "progress": None, "started_at": "t", "reported_at": None}},
    ]}}
    rec = es.parse_space(payload, "knowledge")
    assert (rec.model, rec.migration_state, rec.next_model, rec.next_dims) == (
        "m1", "filling", "m2", 1024)
    assert es.parse_space(payload, "a").migration is None
    assert es.parse_space(payload, "nope") is None
    assert es.parse_space({}, "knowledge") is None
    # The gateway's block is `embeddings` (plural, beside `chat`, `rerank`);
    # a singular key once made the worker find no space against a real gateway
    # while every fixture written to match the reader passed.
    assert es.parse_space({"embedding": payload["embeddings"]}, "knowledge") is None


def test_space_record_is_cached_and_stale_beats_nothing(gw, monkeypatch):
    assert es.current_model() == "m-old"
    assert es.current_model() == "m-old"
    assert gw.caps_calls == 1  # second read served from the cache
    # TTL expiry then an outage: the last good record is kept.
    monkeypatch.setattr(es, "_cached_at", es._cached_at - es.CACHE_TTL_S - 1)
    gw.up = False
    assert es.current_model() == "m-old"


def test_lookup_failure_returns_none_and_warns_once(gw):
    gw.up = False
    with structlog.testing.capture_logs() as logs:
        assert es.current_model() is None
        es._retry_at = 0.0
        assert es.current_model() is None
    warns = [entry for entry in logs if entry["event"] == "embedding_space.lookup_failed"]
    assert len(warns) == 1


async def test_read_filter_uses_vector_model_then_space_then_nothing(gw):
    sql, params = await es.model_clause("pe.embedding_model", query_vector=es.Vector([1.0], "mq"))
    assert (sql, params) == ("pe.embedding_model = :emb_model", {"emb_model": "mq"})
    sql, params = await es.model_clause("pe.embedding_model", query_vector=[1.0])
    assert params == {"emb_model": "m-old"}  # the space's current model
    fresh_space()
    gw.up = False
    # Unknown model: NO filter, never an empty result.
    assert await es.model_clause("pe.embedding_model", query_vector=[1.0]) == ("", {})


def test_progress_payload_matches_the_contract_exactly():
    body = es.build_progress(
        "m2", {"page_embeddings": (3, 10), "trellion_concepts": (0, 4)}, False
    )
    assert body == {
        "next_model": "m2",
        "tables": {
            "page_embeddings": {"done": 3, "total": 10},
            "trellion_concepts": {"done": 0, "total": 4},
        },
        "cutover_done": False,
    }


def test_progress_body_adds_unsourced_when_given():
    body = es.build_progress("m2", {"trellion_concepts": (0, 0, 7), "page_embeddings": (1, 2)}, False)
    assert body["tables"] == {
        "trellion_concepts": {"done": 0, "total": 0, "unsourced": 7},
        "page_embeddings": {"done": 1, "total": 2},
    }


def test_progress_put_goes_to_the_space_migration_route(gw):
    gw.migration = {"next_model": "m2", "next_dims": DIM, "state": "filling"}
    view = es.put_progress("m2", {"page_embeddings": (1, 2)}, cutover_done=True)
    assert view["space"] == "knowledge"
    assert gw.puts == [{
        "next_model": "m2",
        "tables": {"page_embeddings": {"done": 1, "total": 2}},
        "cutover_done": True,
    }]


def test_progress_put_failure_is_reported_not_raised(gw):
    gw.up = False
    assert es.put_progress("m2", {}) is None


# ----------------------------------------------------------------------
# Embed calls name the space; the response names the model
# ----------------------------------------------------------------------


async def test_every_embed_call_sends_the_space_as_model(gw, monkeypatch):
    e = make_embedder(object(), gw)
    out = await e.embed_batch(["a", "b"])
    one = await e.embed_text("c")
    assert [b["model"] for b in gw.embeds] == ["knowledge", "knowledge"]
    # provenance: the RESPONSE's model, on the batch and on each vector
    assert out.model == "m-old" and es.model_of(out[1]) == "m-old"
    assert es.model_of(one) == "m-old"

    monkeypatch.setenv("TRELLION_EMBEDDING_SPACE", "code")
    await e.embed_batch(["d"])
    assert gw.embeds[-1]["model"] == "code"


async def test_reembed_calls_send_the_next_alias(gw):
    gw.migration = {"next_model": "m-new", "next_dims": DIM, "state": "filling"}
    e = make_embedder(object(), gw)
    out = await e._embed_next(["x"])
    assert gw.embeds[-1]["model"] == "knowledge@next"
    assert out.model == "m-new"


def test_the_entry_worker_embedder_sends_the_space_and_keeps_the_model(gw, monkeypatch):
    from trellion.live.embedding_worker import EmbeddingConfig, EmbeddingWorker

    def per_call_client(*a, **kw):
        kw.pop("auth", None)
        kw["transport"] = httpx.MockTransport(gw.handler)
        return REAL_CLIENT(*a, **kw)

    monkeypatch.setattr(httpx, "Client", per_call_client)  # the worker builds one per call
    fn = EmbeddingWorker(
        EmbeddingConfig(embedding_base_url="https://gateway:8080"), store=object()
    )._create_embedder()
    v = fn("hello")
    assert gw.embeds[-1]["model"] == "knowledge"
    assert es.model_of(v) == "m-old"


async def test_the_video_embedder_sends_the_space_and_keeps_the_model(gw, monkeypatch):
    from trellion.live.video.manager import VideoMemoryManager

    real_async = httpx.AsyncClient

    def client(*a, **kw):
        kw.pop("auth", None)
        kw["transport"] = httpx.MockTransport(gw.handler)
        return real_async(*a, **kw)

    monkeypatch.setattr(httpx, "AsyncClient", client)
    mgr = VideoMemoryManager.__new__(VideoMemoryManager)
    mgr.embedding_url = "https://gateway:8080"
    v = await mgr._embed_text("hello")
    assert gw.embeds[-1]["model"] == "knowledge"
    assert es.model_of(v) == "m-old"


# ----------------------------------------------------------------------
# SQL, with a recording engine (no database)
# ----------------------------------------------------------------------


class _Result:
    def __init__(self, scalar=0):
        self._scalar = scalar
        self.rowcount = 0

    def fetchall(self):
        return []

    def scalar(self):
        return self._scalar

    def mappings(self):
        return self

    def all(self):
        return []


class _Conn:
    def __init__(self, log):
        self.log = log

    async def execute(self, stmt, params=None):
        self.log.append((str(stmt), params))
        return _Result()

    async def commit(self):
        pass


class RecordingEngine:
    def __init__(self):
        self.log: list[tuple[str, object]] = []

    @asynccontextmanager
    async def connect(self):
        yield _Conn(self.log)

    @asynccontextmanager
    async def begin(self):
        yield _Conn(self.log)

    def sql(self, needle: str):
        return [(s, p) for s, p in self.log if needle in s]


@pytest.fixture
def store(gw):
    eng = RecordingEngine()
    s = PageIndexStore(async_engine=eng, embedding_dim=DIM)
    s._dims_cache = {"page_embeddings": DIM, "chunk_embeddings": DIM}
    return s, eng


async def test_searches_filter_on_the_current_model(store, gw):
    s, eng = store
    q = [1.0] + [0.0] * (DIM - 1)
    await s.search_pages(q)
    await s.search_chunks(q)
    await s.search_entity_embeddings(q)
    await s.search_clusters(q)
    for alias, needle in (("pe", "FROM page_embeddings pe"), ("ce", "FROM chunk_embeddings ce"),
                          ("ee", "FROM entity_embeddings ee"), ("c", "FROM entity_clusters c")):
        (sql, params), = eng.sql(needle)
        assert f"{alias}.embedding_model = :emb_model" in sql
        assert params["emb_model"] == "m-old"


async def test_searches_drop_the_filter_when_the_model_is_unknown(store, gw):
    s, eng = store
    gw.up = False
    q = [1.0] + [0.0] * (DIM - 1)
    assert await s.search_pages(q) == []  # ran, did not raise
    (sql, params), = eng.sql("FROM page_embeddings pe")
    assert "embedding_model" not in sql and "emb_model" not in params


async def test_query_vector_model_wins_over_the_cached_space_model(store, gw):
    s, eng = store
    await s.search_pages(es.Vector([1.0] + [0.0] * (DIM - 1), "m-query"))
    (_, params), = eng.sql("FROM page_embeddings pe")
    assert params["emb_model"] == "m-query"


async def test_other_readers_filter_too(gw):
    from trellion.live.conversation_store import ConversationStore
    from trellion.live.recall.pg_index import PersonaVectorIndex
    from trellion.live.recall.walk_store import WalkStore

    eng = RecordingEngine()
    q = [1.0] + [0.0] * (DIM - 1)
    cs = ConversationStore.__new__(ConversationStore)
    cs._engine = eng
    cs._embedding_dim = DIM
    await cs.search_semantic(q)
    await PersonaVectorIndex(eng, "p").search("traversals", q)
    await WalkStore(eng).recall_candidates("u", q)
    for needle, col in (("FROM noo_entries e", "e.embedding_model"),
                        ("FROM trellion_episodes", "embedding_model"),
                        ("FROM trellion_traversal_records", "embedding_model")):
        (sql, params), = eng.sql(needle)
        assert f"{col} = :emb_model" in sql and params["emb_model"] == "m-old"


async def test_batch_writers_record_the_response_model_and_dim(store):
    s, eng = store
    batch = es.Embeddings([es.Vector(vec(), "m-served")] * 2, "m-served")
    await s.store_page_embeddings_batch([("p1", batch[0]), ("p2", batch[1])])
    await s.store_chunk_embeddings_batch([("c1", batch[0])])
    (psql, prows), = eng.sql("INSERT INTO page_embeddings")
    assert "embedding_model = EXCLUDED.embedding_model" in psql
    assert "embedding_next = NULL" in psql  # a rewritten vector invalidates its staged twin
    assert [(r["model"], r["dim"]) for r in prows] == [("m-served", DIM)] * 2
    (_, crows), = eng.sql("INSERT INTO chunk_embeddings")
    assert (crows[0]["model"], crows[0]["dim"]) == ("m-served", DIM)


async def test_a_bare_list_records_a_null_model_for_the_legacy_claim(store):
    s, eng = store
    await s.store_page_embeddings_batch([("p1", vec())])
    (_, rows), = eng.sql("INSERT INTO page_embeddings")
    assert rows[0]["model"] is None and rows[0]["dim"] == DIM


async def test_entity_and_cluster_writers_record_provenance(store, gw):
    s, eng = store
    await s.upsert_entity_embedding("e1", es.Vector(vec(), "m-served"), 3)
    (sql, params), = eng.sql("INSERT INTO entity_embeddings")
    assert params["model"] == "m-served" and params["dim"] == DIM
    assert "embedding_next = NULL" in sql
    # A centroid is a bare list: it takes the model it was averaged in.
    await s.upsert_cluster("c1", "n", 0, None, None, vec(), 2, 1)
    (_, params), = eng.sql("INSERT INTO entity_clusters")
    assert params["model"] == "m-old" and params["dim"] == DIM
    await s.upsert_cluster("c2", "n", 0, None, None, es.Vector(vec(), "m-x"), 2, 1)
    assert eng.sql("INSERT INTO entity_clusters")[1][1]["model"] == "m-x"


async def test_archiver_records_the_model_on_traversal_and_episode(gw):
    from trellion.live.recall.archiver import TraversalArchiver

    eng = RecordingEngine()
    arch = TraversalArchiver("/tmp/none", eng)
    res = await arch.capture(
        "p", "q", None, [{"step": 1}], 0.9, query_embedding=es.Vector(vec(), "m-served")
    )
    assert res["archived"] is True
    (esql, eparams), = eng.sql("INSERT INTO trellion_episodes")
    assert "embedding_model, embedding_dim" in esql
    assert (eparams["emb_model"], eparams["emb_dim"]) == ("m-served", DIM)
    (_, tparams), = eng.sql("INSERT INTO trellion_traversal_records")
    assert (tparams["emb_model"], tparams["emb_dim"]) == ("m-served", DIM)
    # An explicit model still wins.
    await arch.capture("p", "q", None, [{"s": 1}], 0.9, query_embedding=vec(),
                       embedding_model="m-explicit")
    assert eng.sql("INSERT INTO trellion_traversal_records")[1][1]["emb_model"] == "m-explicit"


async def test_conversation_store_records_provenance_on_entry_vectors(gw):
    from trellion.live.conversation_store import ConversationStore

    eng = RecordingEngine()
    cs = ConversationStore.__new__(ConversationStore)
    cs._engine = eng
    cs._embedding_dim = DIM
    await cs.batch_update_embeddings([("id1", es.Vector(vec(), "m-served"))])
    (sql, rows), = eng.sql("UPDATE noo_entries")
    assert "embedding_model = :model" in sql and "embedding_next = NULL" in sql
    assert (rows[0]["model"], rows[0]["dim"]) == ("m-served", DIM)


# ----------------------------------------------------------------------
# The worker's statements
# ----------------------------------------------------------------------


def test_worker_tables_are_the_migrations_tables():
    assert {t.name for t in TABLES} == VECTOR_TABLES
    spec = importlib.util.spec_from_file_location(
        "m106", VERSIONS / "106_embedding_provenance.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert set(m.VECTOR_TABLES) == VECTOR_TABLES


def test_legacy_claim_only_touches_unlabelled_vectors_and_skips_locked():
    for spec in TABLES:
        sql = ReembedWorker.legacy_claim_sql(spec)
        assert "embedding_model IS NULL" in sql and f"{spec.vec} IS NOT NULL" in sql
        assert f"embedding_dim = vector_dims({spec.vec})" in sql
        assert "FOR NO KEY UPDATE SKIP LOCKED" in sql and "LIMIT :n" in sql


def test_re_embed_scan_selects_rows_with_a_stale_next_model_and_claims_them():
    for spec in TABLES:
        if spec.kind != "text":
            continue
        sql = ReembedWorker(None, None).claim_sql(spec, cursor=False)
        assert "t.embedding_next_model IS DISTINCT FROM :next" in sql
        assert f"t.{spec.vec} IS NOT NULL" in sql
        assert "FOR NO KEY UPDATE OF t SKIP LOCKED" in sql
        assert f"t.{spec.pk} > CAST(:after AS uuid)" in ReembedWorker(None, None).claim_sql(
            spec, cursor=True)


def test_cutover_sql_swaps_one_table_and_only_staged_rows():
    for spec in TABLES:
        sql = ReembedWorker.cutover_sql(spec)
        assert f"SET {spec.vec} = embedding_next" in sql
        assert "embedding_model = embedding_next_model" in sql
        assert "embedding_next = NULL, embedding_next_model = NULL" in sql
        assert sql.rstrip().endswith("WHERE embedding_next_model = :next")


def test_entity_description_is_the_intel_format():
    assert reembed.entity_description("python", "language", []) == "python (language)"
    nb = [
        {"name": "memory leaks", "type": "concept", "rel_type": "CAUSES", "direction": "outgoing"},
        {"name": "web server", "type": "component", "rel_type": "USED_IN", "direction": "incoming"},
    ]
    assert reembed.entity_description("python", "language", nb) == (
        "python (language): CAUSES -> memory leaks (concept), USED_IN <- web server (component)"
    )


def test_migration_106_content():
    src = (VERSIONS / "106_embedding_provenance.py").read_text()
    assert 'revision: str = "106_embedding_provenance"' in src
    assert 'down_revision: Union[str, None] = "105_doc_hold"' in src
    spec = importlib.util.spec_from_file_location("m106b", VERSIONS / "106_embedding_provenance.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    ran: list[str] = []
    m.op = type("Op", (), {"execute": staticmethod(ran.append)})
    m.upgrade()
    assert "ALTER TABLE trellion_episodes ADD COLUMN IF NOT EXISTS embedding_model text" in ran
    assert "ALTER TABLE trellion_episodes ADD COLUMN IF NOT EXISTS embedding_dim integer" in ran
    assert any("ix_trellion_episodes_embedding_model" in s and "WHERE embedding_model IS NOT NULL"
               in s for s in ran)
    for t in VECTOR_TABLES:
        assert f"ALTER TABLE {t} ADD COLUMN IF NOT EXISTS embedding_next vector(1024)" in ran
        assert f"ALTER TABLE {t} ADD COLUMN IF NOT EXISTS embedding_next_model text" in ran
    # Never backfills: the migration cannot know the model.
    assert not any(s.lstrip().upper().startswith("UPDATE") for s in ran)
    ran.clear()
    m.downgrade()
    assert any("DROP COLUMN IF EXISTS embedding_next" in s for s in ran)


def test_models_mirror_documents_the_new_columns():
    from trellion.db import models

    for cls in (models.PaiEntry, models.PageEmbedding, models.ChunkEmbedding,
                models.EntityEmbedding, models.EntityCluster, models.TrellionConcept,
                models.TrellionEpisode, models.TrellionTraversalRecord):
        cols = set(cls.__table__.columns.keys())
        assert {"embedding_model", "embedding_dim", "embedding_next",
                "embedding_next_model"} <= cols, cls.__name__


# ----------------------------------------------------------------------
# Against Postgres
# ----------------------------------------------------------------------


async def _fetch(store, sql, **params):
    async with store.engine.connect() as c:
        return (await c.execute(text(sql), params)).fetchall()


async def _embed_world(store, gw, make, n=3):
    rows = await make(n)
    e = make_embedder(store, gw)
    await e.process_unembedded_pages(8)
    await e.process_unembedded_chunks(8)
    return rows, e


@needs_db
async def test_migration_106_is_applied(world):
    store, tag, make = world
    for t in VECTOR_TABLES:
        cols = {r[0] for r in await _fetch(
            store,
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = current_schema() AND table_name = :t", t=t)}
        assert {"embedding_model", "embedding_dim", "embedding_next",
                "embedding_next_model"} <= cols, (t, cols)


@needs_db
async def test_backfill_records_the_responses_model_per_row(world, gw):
    store, tag, make = world
    gw.model = f"m-{tag}"
    rows, _ = await _embed_world(store, gw, make)
    got = await _fetch(
        store,
        "SELECT embedding_model, embedding_dim FROM page_embeddings "
        "WHERE page_id = ANY(CAST(:ids AS uuid[]))", ids=[r["page_id"] for r in rows])
    assert got and {tuple(r) for r in got} == {(f"m-{tag}", DIM)}
    got = await _fetch(
        store,
        "SELECT embedding_model FROM chunk_embeddings "
        "WHERE chunk_id = ANY(CAST(:ids AS uuid[]))", ids=[r["chunk_id"] for r in rows])
    assert {r[0] for r in got} == {f"m-{tag}"}


@needs_db
async def test_search_returns_only_vectors_of_the_current_model(world, gw):
    store, tag, make = world
    gw.model = f"m-{tag}"
    rows, e = await _embed_world(store, gw, make, 2)
    ids = {r["page_id"] for r in rows}
    q = es.Vector(vec(), f"m-{tag}")
    hit = {r["page_id"] for r in await store.search_pages(q, top_k=500, doc_ids=[r["doc_id"] for r in rows])}
    assert hit == ids
    # A query in another model finds none of them...
    other = es.Vector(vec(), "m-other")
    assert not {r["page_id"] for r in await store.search_pages(
        other, top_k=500, doc_ids=[r["doc_id"] for r in rows])}
    # ...and an unknown model drops the filter instead of emptying the result.
    fresh_space()
    gw.up = False
    hit = {r["page_id"] for r in await store.search_pages(
        vec(), top_k=500, doc_ids=[r["doc_id"] for r in rows])}
    assert hit == ids


@needs_db
async def test_legacy_claim_labels_unlabelled_vectors(world, gw):
    store, tag, make = world
    rows = await make(2)
    await store.store_page_embeddings_batch([(r["page_id"], vec()) for r in rows])  # bare lists
    ids = [r["page_id"] for r in rows]
    assert {r[0] for r in await _fetch(
        store, "SELECT embedding_model FROM page_embeddings "
        "WHERE page_id = ANY(CAST(:ids AS uuid[]))", ids=ids)} == {None}
    w = ReembedWorker(store.engine, lambda t: None)
    await w.claim_legacy(f"m-{tag}")
    got = await _fetch(
        store, "SELECT embedding_model, embedding_dim FROM page_embeddings "
        "WHERE page_id = ANY(CAST(:ids AS uuid[]))", ids=ids)
    assert {tuple(r) for r in got} == {(f"m-{tag}", DIM)}


def filling(tag):
    return {"next_model": f"n-{tag}", "next_dims": DIM, "state": "filling",
            "progress": None, "started_at": "t", "reported_at": None}


@needs_db
async def test_reembed_fills_the_staging_column_and_reports_progress(world, gw):
    store, tag, make = world
    gw.model = f"m-{tag}"
    rows, e = await _embed_world(store, gw, make, 3)
    pids = [r["page_id"] for r in rows]
    cids = [r["chunk_id"] for r in rows]
    # one row already staged under an OLDER migration's model: must be redone
    await store_exec(store, "UPDATE page_embeddings SET embedding_next = embedding, "
                     "embedding_next_model = 'n-older' WHERE page_id = CAST(:p AS uuid)", p=pids[0])
    gw.migration = filling(tag)
    fresh_space()
    w = ReembedWorker(store.engine, e._embed_next, batch_size=64)
    for _ in range(10):
        await w.run_pass()
    got = await _fetch(
        store, "SELECT embedding_next_model, embedding_model FROM page_embeddings "
        "WHERE page_id = ANY(CAST(:ids AS uuid[]))", ids=pids)
    assert {tuple(r) for r in got} == {(f"n-{tag}", f"m-{tag}")}  # live column untouched
    got = await _fetch(
        store, "SELECT embedding_next_model FROM chunk_embeddings "
        "WHERE chunk_id = ANY(CAST(:ids AS uuid[]))", ids=cids)
    assert {r[0] for r in got} == {f"n-{tag}"}
    # the backfill named the space; every re-embed named its NEXT model
    assert {b["model"] for b in gw.embeds} == {"knowledge", "knowledge@next"}
    # progress: contract-shaped, one entry per table, and every text table reports done
    assert gw.puts, "no progress was reported"
    last = gw.puts[-1]
    assert set(last) == {"next_model", "tables", "cutover_done"}
    assert last["next_model"] == f"n-{tag}" and last["cutover_done"] is False
    assert set(last["tables"]) == VECTOR_TABLES
    for t in last["tables"].values():
        assert set(t) == {"done", "total", "unsourced"} and t["done"] <= t["total"]
    assert last["tables"]["page_embeddings"]["done"] >= 3
    # the underivable table is never invented: not in done/total
    assert last["tables"]["trellion_concepts"]["done"] == 0


async def store_exec(store, sql, **params):
    async with store.engine.begin() as c:
        await c.execute(text(sql), params)


@needs_db
async def test_cutover_swaps_staged_vectors_and_reports_done(world, gw):
    store, tag, make = world
    gw.model = f"m-{tag}"
    rows, e = await _embed_world(store, gw, make, 2)
    pids = [r["page_id"] for r in rows]
    gw.migration = filling(tag)
    fresh_space()
    w = ReembedWorker(store.engine, e._embed_next, batch_size=64)
    for _ in range(6):
        await w.run_pass()
    gw.migration = {**filling(tag), "state": "cutover"}
    fresh_space()
    await w.run_pass()
    got = await _fetch(
        store, "SELECT embedding_model, embedding_dim, embedding_next IS NULL, "
        "embedding_next_model IS NULL FROM page_embeddings "
        "WHERE page_id = ANY(CAST(:ids AS uuid[]))", ids=pids)
    assert {tuple(r) for r in got} == {(f"n-{tag}", DIM, True, True)}
    assert gw.puts[-1]["cutover_done"] is True
    assert gw.puts[-1]["tables"]["page_embeddings"]["done"] >= 2
    # idempotent: a second pass finds nothing staged and just reports again
    n = len(gw.puts)
    await w.run_pass()
    assert len(gw.puts) == n + 1 and gw.puts[-1]["cutover_done"] is True


@needs_db
async def test_cancelled_migration_leaves_no_staged_vectors(world, gw):
    store, tag, make = world
    gw.model = f"m-{tag}"
    rows, e = await _embed_world(store, gw, make, 2)
    pids = [r["page_id"] for r in rows]
    await store_exec(store, "UPDATE page_embeddings SET embedding_next = embedding, "
                     "embedding_next_model = :m WHERE page_id = ANY(CAST(:ids AS uuid[]))",
                     m=f"n-{tag}", ids=pids)
    gw.migration = None  # cancelled (or finished)
    fresh_space()
    w = ReembedWorker(store.engine, e._embed_next)
    await w.run_pass()
    got = await _fetch(
        store, "SELECT count(*) FROM page_embeddings WHERE embedding_next_model IS NOT NULL "
        "OR embedding_next IS NOT NULL")
    assert got[0][0] == 0


@needs_db
async def test_ordinary_rewrite_invalidates_a_staged_vector(world, gw):
    store, tag, make = world
    gw.model = f"m-{tag}"
    rows, e = await _embed_world(store, gw, make, 1)
    pid = rows[0]["page_id"]
    await store_exec(store, "UPDATE page_embeddings SET embedding_next = embedding, "
                     "embedding_next_model = 'n-x' WHERE page_id = CAST(:p AS uuid)", p=pid)
    await store.store_page_embeddings_batch([(pid, es.Vector(vec(2.0), f"m-{tag}"))])
    got = await _fetch(store, "SELECT embedding_next_model FROM page_embeddings "
                       "WHERE page_id = CAST(:p AS uuid)", p=pid)
    assert got[0][0] is None


@needs_db
async def test_derived_tables_episodes_and_clusters(world, gw):
    store, tag, make = world
    nxt = f"n-{tag}"
    frame, trav = str(uuid.uuid4()), str(uuid.uuid4())
    ents = [str(uuid.uuid4()) for _ in range(2)]
    leaf, parent = str(uuid.uuid4()), str(uuid.uuid4())
    lit = "[" + ",".join(["1"] + ["0"] * (DIM - 1)) + "]"
    lit2 = "[" + ",".join(["0", "1"] + ["0"] * (DIM - 2)) + "]"
    try:
        await store_exec(store, "INSERT INTO trellion_episodes (id, persona_id, record_type, "
                         "index_name, content_summary, embedding) VALUES (CAST(:f AS uuid), 'p', "
                         "'traversal', 'traversals', 's', CAST(:v AS vector))", f=frame, v=lit)
        await store_exec(store, "INSERT INTO trellion_traversal_records (id, persona_id, "
                         "query_text, outcome_score, video_frame_id, query_embedding, "
                         "embedding_next, embedding_next_model) VALUES (CAST(:t AS uuid), 'p', "
                         "'q', 0.9, CAST(:f AS uuid), CAST(:v AS vector), CAST(:v2 AS vector), :m)",
                         t=trav, f=frame, v=lit, v2=lit2, m=nxt)
        for i, eid in enumerate(ents):
            await store_exec(store, "INSERT INTO entities (entity_id, canonical_name, type) "
                             "VALUES (CAST(:e AS uuid), :n, 'thing')", e=eid, n=f"{tag}-{i}")
            await store_exec(store, "INSERT INTO entity_embeddings (entity_id, embedding, "
                             "embedding_next, embedding_next_model) VALUES (CAST(:e AS uuid), "
                             "CAST(:v AS vector), CAST(:v2 AS vector), :m)",
                             e=eid, v=lit, v2=lit2, m=nxt)
        await store_exec(store, "INSERT INTO entity_clusters (cluster_id, name, level, centroid, "
                         "member_count) VALUES (CAST(:c AS uuid), 'leaf', 0, CAST(:v AS vector), 2)",
                         c=leaf, v=lit)
        await store_exec(store, "INSERT INTO entity_clusters (cluster_id, name, level, "
                         "parent_cluster_id, centroid, member_count) SELECT CAST(:p AS uuid), "
                         "'parent', 1, NULL, CAST(:v AS vector), 2", p=parent, v=lit)
        await store_exec(store, "UPDATE entity_clusters SET parent_cluster_id = CAST(:p AS uuid) "
                         "WHERE cluster_id = CAST(:c AS uuid)", p=parent, c=leaf)
        for eid in ents:
            await store_exec(store, "INSERT INTO entity_cluster_members (entity_id, cluster_id) "
                             "VALUES (CAST(:e AS uuid), CAST(:c AS uuid))", e=eid, c=leaf)

        gw.model = f"m-{tag}"
        gw.migration = filling(tag)
        fresh_space()
        w = ReembedWorker(store.engine, make_embedder(store, gw)._embed_next, batch_size=64)
        for _ in range(4):  # leaf first, then the parent that waits on it
            await w.run_pass()
            w._idle_until.clear()

        ep = await _fetch(store, "SELECT embedding_next_model, embedding_next::text = :v "
                          "FROM trellion_episodes WHERE id = CAST(:f AS uuid)", f=frame, v=lit2)
        assert tuple(ep[0]) == (nxt, True)  # copied from the traversal record, not re-embedded
        cl = await _fetch(store, "SELECT cluster_id::text, embedding_next_model, "
                          "embedding_next IS NOT NULL FROM entity_clusters "
                          "WHERE cluster_id = ANY(CAST(:ids AS uuid[]))", ids=[leaf, parent])
        assert {r[0]: (r[1], r[2]) for r in cl} == {leaf: (nxt, True), parent: (nxt, True)}
    finally:
        await store_exec(store, "DELETE FROM entity_clusters "
                         "WHERE cluster_id = ANY(CAST(:ids AS uuid[]))", ids=[leaf, parent])
        await store_exec(store, "DELETE FROM entities WHERE entity_id = ANY(CAST(:ids AS uuid[]))",
                         ids=ents)
        await store_exec(store, "DELETE FROM trellion_traversal_records WHERE id = CAST(:t AS uuid)", t=trav)
        await store_exec(store, "DELETE FROM trellion_episodes WHERE id = CAST(:f AS uuid)", f=frame)


async def _fill_all(w, n=6):
    for _ in range(n):
        await w.run_pass()
        w._idle_until.clear()


@needs_db
async def test_unsourced_rows_are_reported_and_dropped_at_cutover(world, gw):
    store, tag, make = world
    gw.model = f"m-{tag}"
    nxt = f"n-{tag}"
    lit = "[" + ",".join(["1"] + ["0"] * (DIM - 1)) + "]"
    concept_snap, concept, orphan = (str(uuid.uuid4()) for _ in range(3))
    try:
        await store_exec(store, "INSERT INTO trellion_snapshots (id, generation, slice_kind) "
                         "VALUES (CAST(:s AS uuid), 1, 'topic')", s=concept_snap)
        await store_exec(store, "INSERT INTO trellion_concepts (id, snapshot_id, canonical_name, "
                         "type, centroid, embedding_model) VALUES (CAST(:c AS uuid), "
                         "CAST(:s AS uuid), :n, 't', CAST(:v AS vector), 'old')",
                         c=concept, s=concept_snap, n=tag, v=lit)
        await store_exec(store, "INSERT INTO trellion_episodes (id, persona_id, record_type, "
                         "index_name, content_summary, embedding, embedding_model) VALUES "
                         "(CAST(:f AS uuid), 'p', 'conversation', 'c', 's', CAST(:v AS vector), 'old')",
                         f=orphan, v=lit)
        gw.migration = filling(tag)
        fresh_space()
        w = ReembedWorker(store.engine, make_embedder(store, gw)._embed_next, batch_size=64)
        await _fill_all(w, 2)
        await w.report(nxt, cutover_done=False)
        tables = gw.puts[-1]["tables"]
        assert tables["trellion_concepts"]["unsourced"] >= 1
        assert tables["trellion_episodes"]["unsourced"] >= 1
        assert tables["trellion_concepts"]["done"] == 0

        gw.migration = {**filling(tag), "state": "cutover"}
        fresh_space()
        await w.run_pass()
        got = await _fetch(store, "SELECT centroid IS NULL, embedding_model IS NULL "
                           "FROM trellion_concepts WHERE id = CAST(:c AS uuid)", c=concept)
        assert tuple(got[0]) == (True, True)  # no old-model vector under the new model
        got = await _fetch(store, "SELECT embedding IS NULL, embedding_model IS NULL "
                           "FROM trellion_episodes WHERE id = CAST(:f AS uuid)", f=orphan)
        assert tuple(got[0]) == (True, True)
    finally:
        await store_exec(store, "DELETE FROM trellion_snapshots WHERE id = CAST(:s AS uuid)", s=concept_snap)
        await store_exec(store, "DELETE FROM trellion_episodes WHERE id = CAST(:f AS uuid)", f=orphan)


@needs_db
async def test_a_poison_row_is_retried_alone_then_skipped_and_the_migration_completes(world, gw):
    store, tag, make = world
    gw.model = f"m-{tag}"
    nxt = f"n-{tag}"
    rows, e = await _embed_world(store, gw, make, 3)
    pids = [r["page_id"] for r in rows]
    gw.poison = {f"{tag} page 1"}
    gw.migration = filling(tag)
    fresh_space()
    w = ReembedWorker(store.engine, e._embed_next, batch_size=64)
    with structlog.testing.capture_logs() as logs:
        await _fill_all(w, 4)
    assert [x["pk"] for x in logs if x["event"] == "reembed.row_skipped"] == [pids[1]]
    got = {r[0]: r[1] for r in await _fetch(
        store, "SELECT page_id::text, embedding_next_model FROM page_embeddings "
        "WHERE page_id = ANY(CAST(:ids AS uuid[]))", ids=pids)}
    assert got == {pids[0]: nxt, pids[1]: None, pids[2]: nxt}
    await w.report(nxt, cutover_done=False)
    # the skipped row counts as unsourced, so done can still reach total
    assert gw.puts[-1]["tables"]["page_embeddings"]["unsourced"] >= 1
    # cutover drops the skipped row's old-model vector instead of leaving it
    gw.migration = {**filling(tag), "state": "cutover"}
    fresh_space()
    await w.run_pass()
    got = {r[0]: r[1] for r in await _fetch(
        store, "SELECT page_id::text, embedding_model FROM page_embeddings "
        "WHERE page_id = ANY(CAST(:ids AS uuid[]))", ids=pids)}
    assert got == {pids[0]: nxt, pids[1]: None, pids[2]: nxt}


@needs_db
async def test_stale_sweep_heals_rows_written_under_an_old_model(world, gw):
    store, tag, make = world
    gw.model = f"m-{tag}"
    rows, e = await _embed_world(store, gw, make, 2)
    pids = [r["page_id"] for r in rows]
    await store_exec(store, "UPDATE page_embeddings SET embedding_model = 'm-stale' "
                     "WHERE page_id = ANY(CAST(:ids AS uuid[]))", ids=pids)
    gw.migration = None
    fresh_space()
    w = ReembedWorker(store.engine, e._embed_next, embed_live=e.embed_batch, batch_size=64)
    before = len(gw.embeds)
    await w.run_pass()
    got = await _fetch(store, "SELECT embedding_model FROM page_embeddings "
                       "WHERE page_id = ANY(CAST(:ids AS uuid[]))", ids=pids)
    assert {r[0] for r in got} == {f"m-{tag}"}
    assert all(b["model"] == "knowledge" for b in gw.embeds[before:])  # the plain space model


@needs_db
async def test_stale_sweep_skips_a_poison_row(world, gw):
    store, tag, make = world
    gw.model = f"m-{tag}"
    rows, e = await _embed_world(store, gw, make, 2)
    pids = [r["page_id"] for r in rows]
    await store_exec(store, "UPDATE page_embeddings SET embedding_model = 'm-stale' "
                     "WHERE page_id = ANY(CAST(:ids AS uuid[]))", ids=pids)
    gw.poison = {f"{tag} page 0"}
    fresh_space()
    w = ReembedWorker(store.engine, e._embed_next, embed_live=e.embed_batch, batch_size=64)
    await w.run_pass()
    got = {r[0]: r[1] for r in await _fetch(
        store, "SELECT page_id::text, embedding_model FROM page_embeddings "
        "WHERE page_id = ANY(CAST(:ids AS uuid[]))", ids=pids)}
    assert got == {pids[0]: "m-stale", pids[1]: f"m-{tag}"}
