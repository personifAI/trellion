# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""Traversal replay is org-bounded and fails closed (noomesh-internal i434).

``TraversalReplayEngine.find_similar`` used to filter with
``(:org IS NULL OR tr.org_id = :org)``. No caller passed an org and no writer
stamped one, so every replay read every tenant's walks. Now the org is
required and the writers stamp it. Rows written before stay unstamped and are
in no one's org (the gateway's rule, chosen over a backfill: no caller reads
them today, and a backfill would need a schema revision).

The first group needs no database. The last group runs against
``TRELLION_TEST_DSN``, like ``test_video_pgvector.py``: CI's db-test job
sets it.
"""

from __future__ import annotations

import asyncio
import inspect
import os
import uuid
from pathlib import Path

import pytest

from trellion.live.recall import TraversalArchiver
from trellion.live.recall.replay import TraversalReplayEngine
from trellion.live.video.traversal_archiver import TraversalArchiver as VideoShim
from trellion.live.video.types import TraversalSnapshot, TraversalStep

PATH = [{"entity_id": "e1", "entity_name": "E1", "entity_type": "concept"}]


class _Conn:
    def __init__(self, log):
        self.log = log

    async def execute(self, stmt, params=None):
        self.log.append((str(stmt), dict(params or {})))


class _Begin:
    def __init__(self, log):
        self.log = log

    async def __aenter__(self):
        return _Conn(self.log)

    async def __aexit__(self, *a):
        return False


class FakeEngine:
    def __init__(self):
        self.log: list[tuple[str, dict]] = []

    def begin(self):
        return _Begin(self.log)

    def traversal_insert(self):
        (sql, params), = [x for x in self.log if "INSERT INTO trellion_traversal_records" in x[0]]
        return sql, params


async def _capture(**kw):
    eng = FakeEngine()
    out = await TraversalArchiver("/x", eng).capture("p", "q?", "causal", PATH, 0.9, **kw)
    assert out and out["archived"], out
    return eng.traversal_insert()


# ---- the writers stamp an org ------------------------------------------

async def test_the_archiver_writes_the_callers_org():
    sql, params = await _capture(user_id="alice", org_id="acme")
    assert "org_id" in sql and ":org_id" in sql
    assert params["org_id"] == "acme"


@pytest.mark.parametrize("given", [None, "", "  "])
async def test_an_omitted_org_is_the_default_org_never_null(given):
    _, params = await _capture(org_id=given)
    assert params["org_id"] == "default"


async def test_the_video_shim_forwards_a_snapshots_org():
    seen = {}

    class Spy:
        async def capture(self, **kw):
            seen.update(kw)
            return {"archived": True}

    shim = VideoShim(engine=object())
    shim._v2_archiver = Spy()
    snap = TraversalSnapshot(
        snapshot_id="s", query="q", query_intent=None,
        steps=[TraversalStep("e1", "E1", "concept")], outcome_score=0.9,
    )
    shim.capture(snap)
    await asyncio.sleep(0)
    # The protocols snapshot has no org field yet: the shim forwards None and
    # the archiver stamps the default (above), never NULL.
    assert "org_id" in seen and seen["org_id"] is None


def test_a_shard_import_stamps_its_traversals():
    from trellion.transport.parquet import ParquetShardPackager

    src = Path(inspect.getsourcefile(ParquetShardPackager)).read_text()
    body = src.split("async def _apply_traversals(", 1)[1].split("\n    async def ", 1)[0]
    assert "INSERT INTO trellion_traversal_records" in body, "parse found the insert"
    assert "org_id)" in body and ":org)" in body and '"org":' in body


# ---- the reader requires one ---------------------------------------------

def test_find_similar_has_no_any_org_form():
    sig = inspect.signature(TraversalReplayEngine.find_similar)
    assert sig.parameters["org"].default is inspect.Parameter.empty, "org must be required"
    src = inspect.getsource(TraversalReplayEngine.find_similar)
    assert "tr.org_id = :org" in src, "parse found the org filter"
    assert "IS NULL OR" not in src


@pytest.mark.parametrize("org", ["", "   "])
async def test_find_similar_refuses_an_empty_org(org):
    with pytest.raises(ValueError):
        await TraversalReplayEngine(None, object()).find_similar("p", [0.0], org=org)


# ---- real Postgres -----------------------------------------------------

DSN = os.environ.get("TRELLION_TEST_DSN")
needs_db = pytest.mark.skipif(not DSN, reason="TRELLION_TEST_DSN not set")
DIM = 1024


def _unit_vec(hot: int) -> list[float]:
    v = [0.0] * DIM
    v[hot % DIM] = 1.0
    return v


@needs_db
async def test_replay_returns_its_own_orgs_walks_and_none_of_another_orgs():
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    from trellion.db.engine import TRELLION_SCHEMA, _async_dsn

    eng = create_async_engine(
        _async_dsn(DSN), connect_args={"options": f"-csearch_path={TRELLION_SCHEMA},public"}
    )
    persona = f"test-{uuid.uuid4().hex[:8]}"
    try:
        arch = TraversalArchiver(None, eng, min_outcome_score=0.5)
        emb = _unit_vec(42)
        mine = await arch.capture(persona, "acme walk", None, PATH, 0.9,
                                  query_embedding=emb, org_id="acme")
        theirs = await arch.capture(persona, "globex walk", None, PATH, 0.9,
                                    query_embedding=emb, org_id="globex")
        # A row written before this change by an old writer: NULL org.
        legacy = await arch.capture(persona, "legacy walk", None, PATH, 0.9,
                                    query_embedding=emb, org_id="acme")
        async with eng.begin() as c:
            await c.execute(
                text("UPDATE trellion_traversal_records SET org_id = NULL WHERE id = CAST(:i AS uuid)"),
                {"i": legacy["traversal_id"]},
            )
            # Prove the fixture before trusting an absence: three rows, three orgs.
            orgs = dict((await c.execute(
                text("SELECT id::text, org_id FROM trellion_traversal_records WHERE persona_id = :p"),
                {"p": persona},
            )).all())
        assert orgs == {mine["traversal_id"]: "acme", theirs["traversal_id"]: "globex",
                        legacy["traversal_id"]: None}

        replay = TraversalReplayEngine(None, eng)
        acme = {m.traversal_id for m in await replay.find_similar(persona, emb, k=10, org="acme")}
        assert acme == {mine["traversal_id"]}, "acme reads its own walk, not globex's, not the unstamped one"
        # An unstamped (pre-change) row is in no one's org: no org reads it.
        globex = {m.traversal_id for m in await replay.find_similar(persona, emb, k=10, org="globex")}
        assert globex == {theirs["traversal_id"]}
        nobody = await replay.find_similar(persona, emb, k=10, org="initech")
        assert nobody == []
    finally:
        async with eng.begin() as c:
            await c.execute(text("DELETE FROM trellion_traversal_records WHERE persona_id = :p"), {"p": persona})
            await c.execute(text("DELETE FROM trellion_episodes WHERE persona_id = :p"), {"p": persona})
        await eng.dispose()
