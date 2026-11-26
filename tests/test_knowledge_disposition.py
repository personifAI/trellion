# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""The disposition pass (shadow mode): what it records, and that it changes nothing else.

The DB tests run against ``TRELLION_TEST_DSN`` (the db-test CI job migrates it
to head, which includes 107). Every test makes its own uniquely-titled
documents; the stub client answers only for those and leaves any other page
undecided.
"""

from __future__ import annotations

import asyncio
import os
import re
import uuid
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import create_async_engine

from trellion.live import knowledge_store
from trellion.live.decision_client import DecisionError
from trellion.live.knowledge_disposition import (
    NO_ENDPOINT,
    DispositionPass,
    DispositionWorker,
)
from trellion.live.knowledge_spec import default_spec
from trellion.live.knowledge_store import KnowledgeStore, PageBatch
from trellion.live.pageindex_store import PageIndexStore

DSN = os.environ.get("TRELLION_TEST_DSN")
needs_db = pytest.mark.skipif(not DSN, reason="TRELLION_TEST_DSN not set")


# ---- no database: the pass against a fake store ---------------------------


class FakeStore:
    def __init__(self, pages):
        self.pages = list(pages)
        self.written: list[dict] = []
        self.claims = 0
        self._inflight: set[str] = set()

    async def load_active_spec(self):
        return default_spec()

    async def claim_undecided_pages(self, limit, min_age_s=0.0, skip=None):
        self.claims += 1
        skip = set(skip or ())
        done = {w["page_id"] for w in self.written}
        rows = [p for p in self.pages if p["page_id"] not in skip | done][:limit]
        return PageBatch(rows, self, {r["page_id"] for r in rows})  # type: ignore[arg-type]

    async def write_disposition(self, **kw):
        self.written.append(kw)
        return True


class FakeClient:
    model = "decider-1"

    def __init__(self, by_page):
        self.by_page = by_page
        self.asked: list[str] = []

    async def ask(self, spec, page):
        self.asked.append(page["page_id"])
        out = self.by_page.get(page["page_id"], DecisionError("down"))
        if isinstance(out, Exception):
            raise out
        return out

    async def aclose(self):
        pass


def _page(i):
    return {"page_id": f"p{i}", "seen": f"h{i}", "text": f"t{i}", "title": "T", "source_uri": None, "meta": {}}


def _pass(store, client):
    return DispositionPass(store, client, min_age_s=0.0)  # type: ignore[arg-type]


async def test_inert_without_an_endpoint():
    store = FakeStore([_page(1)])
    dpass = _pass(store, None)
    report = await dpass.run_once(10)
    assert report.configured is False and report.message == NO_ENDPOINT
    assert store.claims == 0 and store.written == []
    assert dpass.configured is False


def test_worker_from_env_is_none_without_a_url(monkeypatch):
    monkeypatch.delenv("TRELLION_DECISION_URL", raising=False)
    assert DispositionWorker.from_env(object()) is None


async def test_run_once_records_would_be_dispositions():
    store = FakeStore([_page(1), _page(2), _page(3)])
    client = FakeClient({
        "p1": {"embed": 0.9, "graph": 0.8},
        "p2": {"embed": 0.1, "graph": 0.1},
        "p3": {"embed": 0.4, "graph": 0.1},
    })
    report = await _pass(store, client).run_once(10)
    assert (report.claimed, report.decided, report.failed, report.undecidable) == (3, 3, 0, 0)
    by = {w["page_id"]: w for w in store.written}
    assert (by["p1"]["would_embed"], by["p1"]["would_graph"], by["p1"]["held"]) == (True, True, False)
    assert by["p2"]["held"] is True and by["p3"]["held"] is True
    assert by["p3"]["review_after"] < by["p2"]["review_after"]  # the band is looked at sooner
    assert by["p1"]["answers"] == {"embed": 0.9, "graph": 0.8}  # the raw answers are the audit record
    assert by["p1"]["model"] == "decider-1" and by["p1"]["seen"] == "h1"


async def test_missing_or_malformed_answers_record_nothing():
    store = FakeStore([_page(1), _page(2), _page(3)])
    client = FakeClient({
        "p1": {"embed": 0.9},                       # graph missing
        "p2": {"embed": "yes", "graph": 0.5},       # malformed
        "p3": {"embed": 0.9, "graph": 0.9},
    })
    report = await _pass(store, client).run_once(10)
    assert (report.decided, report.undecidable) == (1, 2)
    assert [w["page_id"] for w in store.written] == ["p3"]


async def test_limit_bounds_the_pass():
    store = FakeStore([_page(i) for i in range(10)])
    client = FakeClient({f"p{i}": {"embed": 0.9, "graph": 0.9} for i in range(10)})
    report = await _pass(store, client).run_once(4)
    assert report.decided == 4 and len(client.asked) == 4


async def test_a_dead_endpoint_ends_the_pass_without_raising():
    store = FakeStore([_page(i) for i in range(40)])
    client = FakeClient({})  # every ask raises DecisionError
    report = await _pass(store, client).run_once(40)
    assert report.decided == 0 and report.failed == 16  # one batch, then stop
    assert "unavailable" in report.message
    assert store.written == []


async def test_a_failed_page_does_not_starve_the_others():
    store = FakeStore([_page(0), _page(1)])
    client = FakeClient({"p1": {"embed": 0.9, "graph": 0.9}})  # p0 always fails
    dpass = _pass(store, client)
    first = await dpass.run_once(10)
    assert (first.decided, first.failed) == (1, 1)
    client.asked.clear()
    second = await dpass.run_once(10)
    assert client.asked == []  # p0 is cooling down, p1 is done
    assert second.claimed == 0


async def test_an_unstorable_answer_is_not_a_crash():
    class Refusing(FakeStore):
        async def write_disposition(self, **kw):
            raise ValueError("Out of range float values are not JSON compliant")

    store = Refusing([_page(1)])
    client = FakeClient({"p1": {"embed": 0.9, "graph": 0.9, "x": float("nan")}})
    report = await _pass(store, client).run_once(5)
    assert report.undecidable == 1 and report.decided == 0


async def test_worker_starts_and_stops():
    store = FakeStore([_page(1)])
    client = FakeClient({"p1": {"embed": 0.9, "graph": 0.9}})
    worker = DispositionWorker(_pass(store, client), interval_s=60.0, batch=5)
    assert await worker.start() is True
    for _ in range(100):
        if store.written:
            break
        await asyncio.sleep(0.02)
    await worker.stop()
    assert [w["page_id"] for w in store.written] == ["p1"]


def test_the_pass_has_no_write_path_to_indexing_state():
    """The store may read pages, docs and embeddings; it writes only its own tables."""
    src = Path(knowledge_store.__file__).read_text()
    sql = re.sub(r"\s+", " ", " ".join(re.findall(r'"""(.*?)"""', src, flags=re.S))).lower()
    for table in ("pages", "docs", "page_embeddings", "page_chunks", "chunk_embeddings", "mentions", "entities"):
        for verb in (f"update {table} ", f"insert into {table} ", f"delete from {table} "):
            assert verb not in sql, verb
    assert "insert into page_dispositions" in sql


# ---- real Postgres --------------------------------------------------------


@pytest.fixture
async def world():
    from trellion.db.engine import TRELLION_SCHEMA, _async_dsn

    eng = create_async_engine(
        _async_dsn(DSN), connect_args={"options": f"-csearch_path={TRELLION_SCHEMA},public"}
    )
    store = PageIndexStore(async_engine=eng, embedding_dim=1024)
    tag = f"kbd-{uuid.uuid4().hex[:10]}"
    made: list[str] = []

    async def make(title: str, body: str = "body", *, embed: bool = False) -> dict:
        doc_id, _ = await store.insert_doc(f"test://{tag}/{uuid.uuid4().hex}", f"{tag} {title}", uuid.uuid4().hex)
        made.append(doc_id)
        page_id = await store.insert_page(doc_id, 1, f"{body} {title}", {"headers": ["h"], "char_count": 10})
        if embed:
            await store.store_page_embedding(page_id, [1.0] + [0.0] * 1023)
        return {"doc_id": doc_id, "page_id": page_id}

    try:
        yield store, tag, make, eng
    finally:
        async with eng.begin() as c:
            await c.execute(
                text("DELETE FROM docs WHERE doc_id = ANY(CAST(:ids AS uuid[]))"), {"ids": made}
            )
            await c.execute(text("DELETE FROM kb_specs WHERE created_by = :t"), {"t": tag})
        await eng.dispose()


class TaggedClient(FakeClient):
    """Answers by title for this test's pages; gives no usable answer for any other."""

    def __init__(self, tag, answers):
        super().__init__({})
        self.tag, self.answers = tag, answers

    async def ask(self, spec, page):
        title = page["title"] or ""
        if not title.startswith(self.tag):
            return {}
        return self.answers[title.removeprefix(self.tag + " ")]


async def _row(eng, sql, **p):
    async with eng.connect() as c:
        return (await c.execute(text(sql), p)).fetchall()


@needs_db
async def test_migration_creates_the_three_tables(world):
    _, _, _, eng = world
    cols = {}
    for t in ("page_usage", "kb_specs", "page_dispositions"):
        rows = await _row(eng, "SELECT column_name FROM information_schema.columns "
                               "WHERE table_schema = current_schema() AND table_name = :t", t=t)
        cols[t] = {r[0] for r in rows}
    assert cols["page_usage"] == {"page_id", "retrieved_count", "first_retrieved_at", "last_retrieved_at"}
    assert cols["kb_specs"] == {"version", "spec", "created_at", "created_by", "note", "active"}
    assert cols["page_dispositions"] == {
        "page_id", "spec_version", "would_embed", "would_graph", "held", "answers",
        "model", "decided_at", "review_after", "mode",
    }


@needs_db
async def test_at_most_one_active_spec(world):
    _, tag, _, eng = world
    ks = KnowledgeStore(eng)
    async with eng.begin() as c:
        await c.execute(text("UPDATE kb_specs SET active = false"))
    assert (await ks.load_active_spec()).version == 1  # none stored: the built-in default
    body = default_spec().to_dict()
    v1 = await ks.save_spec(body, created_by=tag, note="first")
    v2 = await ks.save_spec(body, created_by=tag, note="second")
    assert v2 == v1 + 1 and v1 >= 2
    assert (await ks.load_active_spec()).version == v2
    active = await _row(eng, "SELECT version FROM kb_specs WHERE active")
    assert [r[0] for r in active] == [v2]
    with pytest.raises(IntegrityError):
        async with eng.begin() as c:
            await c.execute(text(
                "INSERT INTO kb_specs (version, spec, active) VALUES (:v, '{}'::jsonb, true)"
            ), {"v": v2 + 1000})


@needs_db
async def test_save_spec_validates(world):
    _, tag, _, eng = world
    from trellion.live.knowledge_spec import SpecError

    with pytest.raises(SpecError):
        await KnowledgeStore(eng).save_spec({"questions": {}}, created_by=tag)


@needs_db
async def test_the_pass_records_dispositions_and_touches_nothing_else(world):
    store, tag, make, eng = world
    keep = await make("keep", embed=True)
    maybe = await make("maybe")
    drop = await make("drop")
    held_doc = await make("heldoc")
    await store.hold_docs(doc_ids=[held_doc["doc_id"]], reason="t", apply=True)
    client = TaggedClient(tag, {
        "keep": {"embed": 0.95, "graph": 0.9, "usefulness": {"score": 4.5}},
        "maybe": {"embed": 0.4, "graph": 0.1},
        "drop": {"embed": {"probabilities": {"true": 0.02, "false": 0.98}}, "graph": 0.0},
        "heldoc": {"embed": 0.9, "graph": 0.9},
    })

    mine = [keep["page_id"], maybe["page_id"], drop["page_id"], held_doc["page_id"]]
    state_sql = ("SELECT p.page_id, p.text, p.meta_json, p.last_accessed_at, p.extracted_at, "
                 "p.extraction_claimed_at, (SELECT count(*) FROM page_embeddings e "
                 "WHERE e.page_id = p.page_id AND e.embedding IS NOT NULL) "
                 "FROM pages p WHERE p.page_id = ANY(CAST(:ids AS uuid[])) ORDER BY p.page_id")
    before = await _row(eng, state_sql, ids=mine)
    pre_embed = await _row(eng, "SELECT count(*) FROM page_embeddings")
    pre_chunk = await _row(eng, "SELECT count(*) FROM chunk_embeddings")

    report = await DispositionPass(KnowledgeStore(eng), client, min_age_s=0.0).run_once(1000)  # type: ignore[arg-type]
    assert report.decided == 3  # the held document's page is not offered

    rows = {str(r[0]): r for r in await _row(
        eng, "SELECT page_id, spec_version, would_embed, would_graph, held, answers, model, "
             "review_after, mode FROM page_dispositions WHERE page_id = ANY(CAST(:ids AS uuid[]))", ids=mine)}
    assert set(rows) == {keep["page_id"], maybe["page_id"], drop["page_id"]}
    k, m, d = rows[keep["page_id"]], rows[maybe["page_id"]], rows[drop["page_id"]]
    assert (k[1], k[2], k[3], k[4], k[8]) == (1, True, True, False, "shadow")
    assert k[5]["usefulness"] == {"score": 4.5} and k[7] is None
    assert (m[2], m[3], m[4]) == (False, False, True) and m[7] is not None and m[7] < d[7]
    assert d[4] is True and d[5]["embed"]["probabilities"]["false"] == 0.98

    assert await _row(eng, state_sql, ids=mine) == before  # pages, embeddings, extraction state untouched
    assert await _row(eng, "SELECT count(*) FROM page_embeddings") == pre_embed
    assert await _row(eng, "SELECT count(*) FROM chunk_embeddings") == pre_chunk

    again = await DispositionPass(KnowledgeStore(eng), client, min_age_s=0.0).run_once(1000)  # type: ignore[arg-type]
    assert again.decided == 0  # decided pages are not asked twice


@needs_db
async def test_min_age_leaves_new_documents_alone(world):
    _, tag, make, eng = world
    await make("fresh")
    batch = await KnowledgeStore(eng).claim_undecided_pages(1000, min_age_s=3600)
    try:
        assert not [r for r in batch.rows if (r["title"] or "").startswith(tag)]
    finally:
        batch.release()


@needs_db
async def test_an_edited_page_is_not_given_a_stale_decision(world):
    store, tag, make, eng = world
    p = await make("edit")
    ks = KnowledgeStore(eng)
    batch = await ks.claim_undecided_pages(1000)
    row = next(r for r in batch.rows if r["page_id"] == p["page_id"])
    batch.release()
    async with eng.begin() as c:
        await c.execute(text("UPDATE pages SET text = 'changed' WHERE page_id = CAST(:p AS uuid)"), {"p": p["page_id"]})
    wrote = await ks.write_disposition(
        page_id=p["page_id"], seen=row["seen"], spec_version=1, would_embed=True,
        would_graph=True, held=False, answers={"embed": 1}, model=None, review_after=None)
    assert wrote is False
    assert await _row(eng, "SELECT 1 FROM page_dispositions WHERE page_id = CAST(:p AS uuid)", p=p["page_id"]) == []


@needs_db
async def test_summary_numbers_and_the_held_but_retrieved_count(world):
    store, tag, make, eng = world
    ks = KnowledgeStore(eng)
    dpass = DispositionPass(ks, None)
    before = await ks.disposition_summary()
    keep = await make("keep", embed=True)
    maybe = await make("maybe", embed=True)   # would be held, already embedded
    drop = await make("drop")
    graphonly = await make("graphonly")
    await store.record_page_usage([maybe["page_id"], maybe["page_id"]])
    await store.record_page_usage([maybe["page_id"], keep["page_id"]])
    client = TaggedClient(tag, {
        "keep": {"embed": 0.95, "graph": 0.9},
        "maybe": {"embed": 0.4, "graph": 0.1},
        "drop": {"embed": 0.0, "graph": 0.0},
        "graphonly": {"embed": 0.1, "graph": 0.9},
    })
    await DispositionPass(ks, client, min_age_s=0.0).run_once(1000)  # type: ignore[arg-type]
    after = await ks.disposition_summary()

    def delta(*path):
        a, b = after, before
        for k in path:
            a, b = a[k], b[k]
        return a - b

    assert delta("decided") == 4
    assert delta("embed_and_graph") == 1 and delta("graph_only") == 1 and delta("embed_only") == 0
    assert delta("held") == 2
    assert delta("embed_calls_saved", "pages") == 3          # maybe, drop, graphonly
    assert delta("embed_calls_saved", "pages_already_embedded") == 1  # maybe
    assert delta("extraction_calls_saved", "pages") == 2     # maybe, drop
    assert delta("safety", "held_retrieved") == 1            # maybe: held, and retrieved
    assert delta("safety", "held_retrieved_count") == 2      # two calls counted it
    assert delta("safety", "unembedded_retrieved") == 1
    ver = (await ks.load_active_spec()).version
    assert after["spec_versions"].get(ver, 0) - before["spec_versions"].get(ver, 0) == 4
    s = await dpass.summary()
    assert s["endpoint_configured"] is False and s["message"] == NO_ENDPOINT
    assert s["mode"] == "shadow"
