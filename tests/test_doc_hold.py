# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""Document hold: keep a document, out of embedding and search.

The DB tests run against ``TRELLION_TEST_DSN`` (the db-test CI job migrates it
to head, which includes 105). Every test makes its own uniquely-titled
documents, so it never reads or disturbs anyone else's rows.
"""

from __future__ import annotations

import os
import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from trellion.live.pageindex_store import PageIndexStore

DSN = os.environ.get("TRELLION_TEST_DSN")
needs_db = pytest.mark.skipif(not DSN, reason="TRELLION_TEST_DSN not set")

DIM = 8


def _vec(hot: int) -> list[float]:
    v = [0.0] * DIM
    v[hot % DIM] = 1.0
    return v


# ---- no database -------------------------------------------------------


def test_escape_like_is_literal():
    assert PageIndexStore.escape_like("a%b_c\\d") == "a\\%b\\_c\\\\d"


async def test_exactly_one_selector():
    store = PageIndexStore(async_engine=object(), embedding_dim=DIM)
    for kw in ({}, {"title_prefix": "x", "doc_ids": ["a"]}):
        with pytest.raises(ValueError):
            await store.hold_docs(reason="r", **kw)
    with pytest.raises(ValueError):
        await store.hold_docs(title_prefix="x", reason="")


async def test_purge_needs_a_reason():
    store = PageIndexStore(async_engine=object(), embedding_dim=DIM)
    for bad in ("", "   "):
        with pytest.raises(ValueError):
            await store.purge_held(bad, apply=True)


# ---- real Postgres -----------------------------------------------------


@pytest.fixture
async def world():
    from trellion.db.engine import TRELLION_SCHEMA, _async_dsn

    eng = create_async_engine(
        _async_dsn(DSN), connect_args={"options": f"-csearch_path={TRELLION_SCHEMA},public"}
    )
    store = PageIndexStore(async_engine=eng, embedding_dim=1024)
    tag = f"hold-{uuid.uuid4().hex[:10]}"
    made: list[str] = []

    async def make(title: str, *, embed: bool = False) -> dict:
        doc_id, _ = await store.insert_doc(
            f"test://{tag}/{uuid.uuid4().hex}", title, uuid.uuid4().hex
        )
        made.append(doc_id)
        page_id = await store.insert_page(doc_id, 1, f"page of {title}")
        chunk_ids = await store.insert_chunks(page_id, [f"chunk of {title}"])
        if embed:
            v = [1.0] + [0.0] * 1023
            await store.store_page_embedding(page_id, v)
            await store.store_chunk_embedding(chunk_ids[0], v)
        return {"doc_id": doc_id, "page_id": page_id, "chunk_id": chunk_ids[0]}

    try:
        yield store, tag, make, eng
    finally:
        async with eng.begin() as c:
            await c.execute(
                text("DELETE FROM docs WHERE doc_id = ANY(CAST(:ids AS uuid[]))"),
                {"ids": made},
            )
        await eng.dispose()


async def _pending(store, ids_pages=(), ids_chunks=()):
    pages = {p["page_id"] for p in await store.get_unembedded_pages(limit=1_000_000)}
    chunks = {c["chunk_id"] for c in await store.get_unembedded_chunks(limit=1_000_000)}
    return pages, chunks


async def _count(eng, sql, **p):
    async with eng.connect() as c:
        return (await c.execute(text(sql), p)).scalar()


@needs_db
async def test_held_doc_is_not_offered_to_the_embedder(world):
    store, tag, make, _ = world
    kept = await make(f"{tag} kept")
    held = await make(f"{tag} held")
    await store.hold_docs(doc_ids=[held["doc_id"]], reason="t", apply=True)
    pages, chunks = await _pending(store)
    assert kept["page_id"] in pages and kept["chunk_id"] in chunks
    assert held["page_id"] not in pages
    assert held["chunk_id"] not in chunks


@needs_db
async def test_held_doc_never_appears_in_search(world):
    store, tag, make, _ = world
    kept = await make(f"{tag} kept", embed=True)
    held = await make(f"{tag} held", embed=True)
    q = [1.0] + [0.0] * 1023
    await store.hold_docs(doc_ids=[held["doc_id"]], reason="t", apply=True)
    for roles in (None, ["user"]):
        pages = {p["page_id"] for p in await store.search_pages(q, top_k=10_000, roles=roles)}
        chunks = {c["chunk_id"] for c in await store.search_chunks(q, top_k=10_000, roles=roles)}
        assert held["page_id"] not in pages and held["chunk_id"] not in chunks
    pages = {p["page_id"] for p in await store.search_pages(q, top_k=10_000)}
    chunks = {c["chunk_id"] for c in await store.search_chunks(q, top_k=10_000)}
    assert kept["page_id"] in pages and kept["chunk_id"] in chunks
    # scoped searches too
    scoped = await store.search_pages(q, top_k=10, doc_ids=[held["doc_id"]])
    assert scoped == []
    scoped = await store.search_chunks(q, top_k=10, page_ids=[held["page_id"]])
    assert scoped == []


@needs_db
async def test_held_pages_are_not_claimed_for_extraction_or_entity_pages(world):
    store, tag, make, eng = world
    held = await make(f"{tag} held")
    ok = await make(f"{tag} ok")
    eid = await store.upsert_entity(f"ent-{tag}", "concept")
    for d in (held, ok):
        await store.insert_mention(eid, d["page_id"])
    await store.hold_docs(doc_ids=[held["doc_id"]], reason="t", apply=True)
    assert await store.claim_page_for_extraction(held["page_id"]) is False
    assert await store.claim_page_for_extraction(ok["page_id"]) is True
    assert await store.get_entity_pages(eid) == [ok["page_id"]]
    assert await store.get_entity_pages_batch([eid]) == [ok["page_id"]]
    async with eng.begin() as c:
        await c.execute(text("DELETE FROM entities WHERE entity_id = CAST(:i AS uuid)"), {"i": eid})


@needs_db
async def test_unembedded_count_excludes_held(world):
    store, tag, make, eng = world
    await make(f"{tag} a")
    held = await make(f"{tag} b")
    sql = (
        "SELECT COUNT(*) FROM pages p JOIN docs d ON d.doc_id = p.doc_id "
        "LEFT JOIN page_embeddings pe ON pe.page_id = p.page_id "
        "WHERE d.held_at IS NULL AND (pe.page_id IS NULL OR pe.embedding IS NULL)"
        " AND d.title LIKE :t"
    )
    before = await _count(eng, sql, t=f"{tag}%")
    assert before == 2
    await store.hold_docs(doc_ids=[held["doc_id"]], reason="t", apply=True)
    assert await _count(eng, sql, t=f"{tag}%") == 1
    assert len(
        [p for p in await store.get_unembedded_pages(limit=1_000_000)
         if p["text"].startswith("page of " + tag)]
    ) == 1


@needs_db
async def test_dry_run_changes_nothing(world):
    store, tag, make, eng = world
    a = await make(f"{tag} one")
    await make(f"{tag} two")
    out = await store.hold_docs(title_prefix=tag, reason="t", apply=False)
    assert out["count"] == 2 and out["applied"] is False and a["doc_id"] in out["doc_ids"]
    assert await _count(eng, "SELECT COUNT(*) FROM docs WHERE held_at IS NOT NULL AND title LIKE :t", t=f"{tag}%") == 0
    # unhold dry run
    await store.hold_docs(title_prefix=tag, reason="t", apply=True)
    out = await store.unhold_docs(title_prefix=tag, apply=False)
    assert out["count"] == 2
    assert await _count(eng, "SELECT COUNT(*) FROM docs WHERE held_at IS NOT NULL AND title LIKE :t", t=f"{tag}%") == 2
    # purge dry run
    out = await store.purge_held("t", apply=False)
    assert out["deleted"] == 0
    assert await _count(eng, "SELECT COUNT(*) FROM docs WHERE title LIKE :t", t=f"{tag}%") == 2


@needs_db
async def test_prefix_wildcards_are_literal(world):
    store, tag, make, _ = world
    pct = await make(f"{tag} 100%sure")
    under = await make(f"{tag} a_b")
    other_pct = await make(f"{tag} 100xsure")
    other_under = await make(f"{tag} axb")
    out = await store.hold_docs(title_prefix=f"{tag} 100%", reason="t", apply=True)
    assert out["doc_ids"] == [pct["doc_id"]]
    out = await store.hold_docs(title_prefix=f"{tag} a_", reason="t", apply=True)
    assert out["doc_ids"] == [under["doc_id"]]
    out = await store.hold_docs(title_prefix=f"{tag} \\", reason="t", apply=False)
    assert out["count"] == 0
    held = {d["doc_id"] for d in (await store.list_held(limit=100_000))["docs"]}
    assert other_pct["doc_id"] not in held and other_under["doc_id"] not in held


@needs_db
async def test_unhold_restores_embedding_and_search_and_nothing_is_deleted(world):
    store, tag, make, eng = world
    d = await make(f"{tag} d", embed=True)
    q = [1.0] + [0.0] * 1023
    out = await store.hold_docs(doc_ids=[d["doc_id"]], reason="t", apply=True)
    assert out["count"] == 1
    # re-holding is a no-op and keeps the first reason
    again = await store.hold_docs(doc_ids=[d["doc_id"]], reason="other", apply=True)
    assert again["count"] == 0
    held = (await store.list_held(reason="t", limit=100_000))["docs"]
    assert [h["doc_id"] for h in held if h["doc_id"] == d["doc_id"]] == [d["doc_id"]]
    assert held[0]["title"] and held[0]["held_at"]
    # nothing deleted
    for tbl, col, val in (("docs", "doc_id", d["doc_id"]), ("pages", "page_id", d["page_id"]),
                          ("page_chunks", "chunk_id", d["chunk_id"])):
        assert await _count(eng, f"SELECT COUNT(*) FROM {tbl} WHERE {col} = CAST(:v AS uuid)", v=val) == 1
    assert d["page_id"] not in {p["page_id"] for p in await store.search_pages(q, top_k=10_000)}
    un = await store.unhold_docs(doc_ids=[d["doc_id"]], apply=True)
    assert un["count"] == 1
    assert d["page_id"] in {p["page_id"] for p in await store.search_pages(q, top_k=10_000)}
    assert d["chunk_id"] in {c["chunk_id"] for c in await store.search_chunks(q, top_k=10_000)}
    # and a never-embedded doc becomes embeddable again
    e = await make(f"{tag} e")
    await store.hold_docs(doc_ids=[e["doc_id"]], reason="t", apply=True)
    assert e["page_id"] not in (await _pending(store))[0]
    await store.unhold_docs(doc_ids=[e["doc_id"]], apply=True)
    assert e["page_id"] in (await _pending(store))[0]


@needs_db
async def test_list_held_pages_through_everything(world):
    store, tag, make, _ = world
    ids = [(await make(f"{tag} {i}"))["doc_id"] for i in range(5)]
    await store.hold_docs(title_prefix=tag, reason=f"r-{tag}", apply=True)
    seen: list[str] = []
    for off in (0, 2, 4):
        page = await store.list_held(reason=f"r-{tag}", limit=2, offset=off)
        assert page["total"] == 5
        seen += [d["doc_id"] for d in page["docs"]]
    assert sorted(seen) == sorted(ids)


@needs_db
async def test_purge_deletes_only_docs_held_under_that_reason(world):
    store, tag, make, eng = world
    mine = [await make(f"{tag} m{i}", embed=True) for i in range(3)]
    unheld = await make(f"{tag} unheld", embed=True)
    other = await make(f"{tag} other", embed=True)
    await store.hold_docs(doc_ids=[m["doc_id"] for m in mine], reason=f"R-{tag}", apply=True)
    await store.hold_docs(doc_ids=[other["doc_id"]], reason=f"S-{tag}", apply=True)
    out = await store.purge_held(f"R-{tag}", apply=True)
    assert out["count"] == 3 and out["deleted"] == 3
    for m in mine:
        for tbl, col, key in (("docs", "doc_id", "doc_id"), ("pages", "page_id", "page_id"),
                              ("page_chunks", "chunk_id", "chunk_id"),
                              ("page_embeddings", "page_id", "page_id"),
                              ("chunk_embeddings", "chunk_id", "chunk_id")):
            assert await _count(eng, f"SELECT COUNT(*) FROM {tbl} WHERE {col} = CAST(:v AS uuid)", v=m[key]) == 0
    for survivor in (unheld, other):
        assert await _count(eng, "SELECT COUNT(*) FROM docs WHERE doc_id = CAST(:v AS uuid)", v=survivor["doc_id"]) == 1
        assert await _count(eng, "SELECT COUNT(*) FROM page_embeddings WHERE page_id = CAST(:v AS uuid)", v=survivor["page_id"]) == 1


@needs_db
async def test_purge_dry_run_and_empty_reason(world):
    store, tag, make, eng = world
    d = await make(f"{tag} d")
    await store.hold_docs(doc_ids=[d["doc_id"]], reason=f"R-{tag}", apply=True)
    out = await store.purge_held(f"R-{tag}", apply=False)
    assert out == {"count": 1, "doc_ids": [d["doc_id"]], "deleted": 0}
    assert await _count(eng, "SELECT COUNT(*) FROM docs WHERE doc_id = CAST(:v AS uuid)", v=d["doc_id"]) == 1
    with pytest.raises(ValueError):
        await store.purge_held("", apply=True)
    assert await _count(eng, "SELECT COUNT(*) FROM docs WHERE doc_id = CAST(:v AS uuid)", v=d["doc_id"]) == 1


@needs_db
async def test_purge_batches(world):
    store, tag, make, eng = world
    ids = [(await make(f"{tag} {i}"))["doc_id"] for i in range(5)]
    await store.hold_docs(doc_ids=ids, reason=f"R-{tag}", apply=True)
    stmts: list[int] = []
    from sqlalchemy import event

    @event.listens_for(eng.sync_engine, "commit")
    def _count_commit(conn):  # one commit per batch transaction
        stmts.append(1)

    out = await store.purge_held(f"R-{tag}", apply=True, batch=2)
    assert out["deleted"] == 5
    assert len(stmts) == 3, "5 docs at batch=2 is three separate transactions"
    assert await _count(eng, "SELECT COUNT(*) FROM docs WHERE title LIKE :t", t=f"{tag}%") == 0


def test_metadata_narrower_never_offers_a_held_doc():
    from trellion.live.narrowers.metadata import MetadataNarrower

    sql, _ = MetadataNarrower._build_query({"title_substr": "x"})
    assert "held_at IS NULL" in sql
