# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""Frame embeddings in Postgres (gateway proposal 28 Part 1).

DB tests run against the migrated schema (TRELLION_TEST_DSN — the
db-test CI job provisions it from trellion's own chain, which now
includes 101_frame_embeddings). They prove the full retirement path:
the archiver's single INSERT makes a frame searchable, tombstones and
prominence need no sync step, and the pure-SQL backfill covers
pre-migration frames from the traversal records' existing vectors.
"""

from __future__ import annotations

import os
import uuid

import pytest
from sqlalchemy import text

DSN = os.environ.get("TRELLION_TEST_DSN")

pytestmark = pytest.mark.skipif(
    not DSN, reason="TRELLION_TEST_DSN not set — see module docstring"
)

DIM = 1024


def _unit_vec(hot: int) -> list[float]:
    v = [0.0] * DIM
    v[hot % DIM] = 1.0
    return v


@pytest.fixture
async def engine():
    from sqlalchemy.ext.asyncio import create_async_engine

    from trellion.db.engine import TRELLION_SCHEMA, _async_dsn

    eng = create_async_engine(
        _async_dsn(DSN),
        connect_args={"options": f"-csearch_path={TRELLION_SCHEMA},public"},
    )
    yield eng
    await eng.dispose()


@pytest.fixture
def persona():
    return f"test-{uuid.uuid4().hex[:8]}"


async def test_archiver_frame_is_searchable_without_faiss(engine, persona):
    from trellion.live.recall import PersonaVectorIndex, TraversalArchiver

    archiver = TraversalArchiver(None, engine, min_outcome_score=0.5)
    out = await archiver.capture(
        persona_id=persona,
        query_text="how do prerequisites chain?",
        intent="causal",
        traversal_path=[{"entity": "a"}, {"entity": "b"}],
        outcome_score=0.9,
        query_embedding=_unit_vec(7),
    )
    assert out and out["archived"], out

    idx = PersonaVectorIndex(engine, persona)
    hits = await idx.search("traversals", _unit_vec(7), k=5)
    assert len(hits) == 1
    hit = hits[0]
    assert hit.frame_id == out["frame_id"]
    assert hit.record_type == "traversal"
    assert hit.similarity > 0.99
    # 60/40: sim ~1.0, prominence = min(1.0, outcome) = 0.9
    assert abs(hit.combined_score - (0.6 * hit.similarity + 0.4 * 0.9)) < 1e-6

    # An orthogonal query still returns the frame, but with ~zero similarity
    # (prominence alone carries it) — the scoring is visible, not hidden.
    far = await idx.search("traversals", _unit_vec(8), k=5)
    assert far and far[0].similarity < 0.01


async def test_tombstoned_frames_never_serve_search(engine, persona):
    from trellion.live.recall import PersonaVectorIndex

    idx = PersonaVectorIndex(engine, persona)
    fid = str(uuid.uuid4())
    await idx.add("conversations", _unit_vec(3), fid, prominence=0.8)
    assert (await idx.search("conversations", _unit_vec(3), k=5))

    async with engine.begin() as conn:
        await conn.execute(
            text(
                "UPDATE trellion_episodes SET is_tombstoned = TRUE "
                "WHERE id = CAST(:id AS uuid)"
            ),
            {"id": fid},
        )
    assert not (await idx.search("conversations", _unit_vec(3), k=5))


async def test_backfill_copies_traversal_embeddings_onto_frames(engine, persona):
    from trellion.live.recall import (
        PersonaVectorIndex,
        backfill_frame_embeddings,
    )
    from trellion.live.recall.pg_index import _vec_literal

    # Simulate a pre-migration frame: row exists, no embedding, but its
    # traversal record carries the vector (the old dual-write's PG half).
    fid, tid = str(uuid.uuid4()), str(uuid.uuid4())
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO trellion_episodes "
                "(id, persona_id, record_type, index_name, "
                " content_summary, prominence_score) "
                "VALUES (CAST(:id AS uuid), :p, 'traversal', 'traversals', 's', 0.7)"
            ),
            {"id": fid, "p": persona},
        )
        await conn.execute(
            text(
                "INSERT INTO trellion_traversal_records "
                "(id, persona_id, query_text, query_embedding, traversal_path, "
                " outcome_score, archived_to_video, video_frame_id) "
                "VALUES (CAST(:id AS uuid), :p, 'q', CAST(:emb AS vector), "
                "        '[]'::jsonb, 0.7, TRUE, CAST(:fid AS uuid))"
            ),
            {"id": tid, "p": persona, "emb": _vec_literal(_unit_vec(5)), "fid": fid},
        )

    out = await backfill_frame_embeddings(engine)
    assert out["updated"] >= 1

    hits = await PersonaVectorIndex(engine, persona).search(
        "traversals", _unit_vec(5), k=5
    )
    assert [h.frame_id for h in hits] == [fid]
    assert hits[0].similarity > 0.99


async def test_search_all_indexes_default(engine, persona):
    """index_name=None covers every index — the fix for the video-search
    surface that was pinned to the never-populated "conversations" index
    and therefore returned [] forever."""
    from trellion.live.recall import PersonaVectorIndex

    idx = PersonaVectorIndex(engine, persona)
    await idx.add("traversals", _unit_vec(11), str(uuid.uuid4()), record_type="traversal")
    await idx.add("conversations", _unit_vec(12), str(uuid.uuid4()))

    hits = await idx.search(None, _unit_vec(11), k=10)
    assert {h.record_type for h in hits} == {"traversal", "conversation"}
    scoped = await idx.search("traversals", _unit_vec(11), k=10)
    assert {h.record_type for h in scoped} == {"traversal"}


async def test_store_dsn_path_pins_search_path():
    """PageIndexStore(dsn=...) must see the trellion schema — a raw engine
    saw only `public` (found by the P1.5 bench harness)."""
    from trellion.live.pageindex_store import PageIndexStore

    store = PageIndexStore(dsn=DSN, embedding_dim=DIM)
    await store.connect()
    try:
        assert isinstance(await store.list_docs(limit=1), list)
    finally:
        await store.close()

