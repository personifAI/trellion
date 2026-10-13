# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""Recallable walks, phase 1a (noomesh proposal 57): schema + archiver + store.

Unit tests need no database. The DB tests run against ``TRELLION_TEST_DSN``
(the db-test CI job migrates it to head, which includes 104). The migration
test builds its own scratch database so it can stop at 103 and seed rows.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

import pytest
from sqlalchemy import text

from trellion.live.recall import TraversalArchiver, WalkStore
from trellion.live.video.traversal_archiver import TraversalArchiver as VideoShim
from trellion.live.video.types import TraversalSnapshot, TraversalStep

DSN = os.environ.get("TRELLION_TEST_DSN")
needs_db = pytest.mark.skipif(not DSN, reason="TRELLION_TEST_DSN not set")

DIM = 1024
PATH = [
    {"entity_id": "e1", "entity_name": "E1", "entity_type": "concept"},
    {"entity_id": "e2", "entity_name": "E2", "entity_type": "concept", "edge_type": "uses"},
]
EVIDENCE = {
    "triples": [["e1", "uses", "e2"]],
    "doc_ids": ["d1", "d2"],
    "newest_evidence_at": "2026-10-01T00:00:00+00:00",
}


def unit(hot: int, tilt: float = 0.0) -> list[float]:
    """A unit-ish vector: axis ``hot``, plus ``tilt`` toward axis 0."""
    v = [0.0] * DIM
    v[hot % DIM] = 1.0
    if tilt:
        v[0] += tilt
    return v


# ---- no database -------------------------------------------------------


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

    def insert(self, table):
        (sql, params), = [x for x in self.log if f"INSERT INTO {table}" in x[0]]
        return sql, params


async def test_old_style_capture_is_simulated_with_nulls():
    eng = FakeEngine()
    out = await TraversalArchiver("/x", eng).capture("p", "q", None, PATH, 0.9)
    assert out["archived"]
    _, p = eng.insert("trellion_traversal_records")
    assert p["kind"] == "simulated"
    assert p["user_id"] is None and p["thread_id"] is None
    assert p["turn_index"] is None and p["run_id"] is None
    assert json.loads(p["provenance"]) == {} and json.loads(p["evidence"]) == {}


async def test_thread_capture_needs_an_owner_and_a_uuid_thread():
    eng = FakeEngine()
    arch = TraversalArchiver("/x", eng)
    out = await arch.capture("p", "q", None, PATH, 0.9, kind="thread")
    assert out == {"archived": False, "reason": "a thread walk needs a user_id"}
    out = await arch.capture("p", "q", None, PATH, 0.9, kind="thread", user_id="alice", thread_id="nope")
    assert out["archived"] is False and "UUID" in out["reason"]
    out = await arch.capture("p", "q", None, PATH, 0.9, kind="bogus")
    assert out["archived"] is False and "kind" in out["reason"]
    assert eng.log == []


async def test_thread_episode_row_carries_owner_and_not_the_question():
    eng = FakeEngine()
    await TraversalArchiver("/x", eng).capture(
        "p", "my private question", None, PATH, 0.9, kind="thread", user_id="alice"
    )
    _, ep = eng.insert("trellion_episodes")
    assert ep["user_id"] == "alice"
    assert "private" not in ep["summary"]


async def test_the_video_shim_forwards_walk_fields():
    seen = {}

    class Spy:
        async def capture(self, **kw):
            seen.update(kw)
            return {"archived": True}

    shim = VideoShim(engine=object())
    shim._v2_archiver = Spy()
    shim.capture(TraversalSnapshot(
        snapshot_id="s", query="q", query_intent=None,
        steps=[TraversalStep("e1", "E1", "concept")], outcome_score=0.9,
        user_id="alice", kind="thread", thread_id="t", turn_index=3, run_id="r",
        provenance={"tool": "navigate"}, evidence=EVIDENCE, usefulness=0.7,
    ))
    import asyncio

    await asyncio.sleep(0)
    assert (seen["kind"], seen["thread_id"], seen["turn_index"], seen["run_id"]) == (
        "thread", "t", 3, "r")
    assert seen["provenance"] == {"tool": "navigate"} and seen["evidence"] == EVIDENCE
    assert seen["usefulness"] == 0.7


async def test_the_video_shim_defaults_an_old_snapshot_to_simulated():
    seen = {}

    class Spy:
        async def capture(self, **kw):
            seen.update(kw)
            return {"archived": True}

    shim = VideoShim(engine=object())
    shim._v2_archiver = Spy()
    shim.capture(TraversalSnapshot(snapshot_id="s", query="q", query_intent=None,
                                   steps=[TraversalStep("e1", "E1", "concept")],
                                   outcome_score=0.9))
    import asyncio

    await asyncio.sleep(0)
    assert seen["kind"] == "simulated" and seen["thread_id"] is None


async def test_store_refuses_an_empty_owner():
    store = WalkStore(object())
    for bad in (None, "", "  "):
        with pytest.raises(ValueError):
            await store.delete_for_user(bad)
        with pytest.raises(ValueError):
            await store.stage_for_user(bad)
        with pytest.raises(ValueError):
            await store.recall_candidates(bad, unit(1), kind="thread")


# ---- real Postgres -----------------------------------------------------


@pytest.fixture
async def engine():
    from sqlalchemy.ext.asyncio import create_async_engine

    from trellion.db.engine import TRELLION_SCHEMA, _async_dsn

    eng = create_async_engine(
        _async_dsn(DSN), connect_args={"options": f"-csearch_path={TRELLION_SCHEMA},public"}
    )
    yield eng
    await eng.dispose()


@pytest.fixture
async def users(engine):
    """Two unique users and a persona; their rows are removed afterwards."""
    tag = uuid.uuid4().hex[:8]
    ns = type("NS", (), {})()
    ns.x, ns.y, ns.persona = f"x-{tag}", f"y-{tag}", f"walks-{tag}"
    yield ns
    async with engine.begin() as c:
        await c.execute(text("DELETE FROM trellion_traversal_records WHERE persona_id=:p"), {"p": ns.persona})
        await c.execute(text("DELETE FROM trellion_episodes WHERE persona_id=:p"), {"p": ns.persona})


async def _row(engine, traversal_id):
    async with engine.connect() as c:
        return (await c.execute(
            text("SELECT * FROM trellion_traversal_records WHERE id = CAST(:i AS uuid)"),
            {"i": traversal_id},
        )).mappings().one()


@needs_db
async def test_thread_capture_round_trips_every_field(engine, users):
    thread = str(uuid.uuid4())
    arch = TraversalArchiver(None, engine, min_outcome_score=0.5)
    out = await arch.capture(
        users.persona, "asked by x", "causal", PATH, 0.8,
        query_embedding=unit(3), user_id=users.x, kind="thread", thread_id=thread,
        turn_index=2, run_id="run-9", provenance={"tool": "navigate", "request_id": "r1"},
        evidence=EVIDENCE, usefulness=0.77, embedding_model="harrier",
    )
    assert out["archived"], out
    r = await _row(engine, out["traversal_id"])
    assert r["kind"] == "thread" and r["user_id"] == users.x
    assert str(r["thread_id"]) == thread and r["turn_index"] == 2 and r["run_id"] == "run-9"
    assert r["provenance"] == {"tool": "navigate", "request_id": "r1"}
    assert r["evidence"] == EVIDENCE
    assert r["usefulness"] == pytest.approx(0.77)
    assert r["status"] == "active" and r["recall_count"] == 0
    assert r["last_validated_at"] is not None and r["last_recalled_at"] is None
    assert r["embedding_model"] == "harrier" and r["embedding_dim"] == DIM
    async with engine.connect() as c:
        ep = (await c.execute(
            text("SELECT user_id, content_summary FROM trellion_episodes WHERE id = CAST(:i AS uuid)"),
            {"i": out["frame_id"]},
        )).one()
    assert ep[0] == users.x
    assert "asked by x" not in ep[1]


@needs_db
async def test_old_style_capture_in_postgres_is_simulated_with_nulls(engine, users):
    out = await TraversalArchiver(None, engine, min_outcome_score=0.5).capture(
        users.persona, "heartbeat", None, PATH, 0.9
    )
    r = await _row(engine, out["traversal_id"])
    assert (r["kind"], r["status"], r["user_id"]) == ("simulated", "active", None)
    assert r["thread_id"] is None and r["turn_index"] is None and r["run_id"] is None
    assert r["provenance"] == {} and r["evidence"] == {}
    assert r["usefulness"] is None and r["expires_at"] is None
    async with engine.connect() as c:
        uid = (await c.execute(
            text("SELECT user_id FROM trellion_episodes WHERE id = CAST(:i AS uuid)"),
            {"i": out["frame_id"]},
        )).scalar()
    assert uid is None


async def _seed(engine, users):
    arch = TraversalArchiver(None, engine, min_outcome_score=0.5)
    ids = {}
    # x: near (axis 5), mid (axis 5 tilted), far (axis 9); y: near; sim: near
    for name, owner, kind, emb in [
        ("x_far", users.x, "thread", unit(9)),
        ("x_near", users.x, "thread", unit(5)),
        ("x_mid", users.x, "thread", unit(5, tilt=1.0)),
        ("y_near", users.y, "thread", unit(5)),
        ("sim_near", None, "simulated", unit(5)),
    ]:
        out = await arch.capture(
            users.persona, name, None, PATH, 0.9, query_embedding=emb,
            user_id=owner, kind=kind,
        )
        ids[name] = out["traversal_id"]
    return ids


@needs_db
async def test_candidates_are_only_the_owners_active_threads_nearest_first(engine, users):
    ids = await _seed(engine, users)
    store = WalkStore(engine)
    got = await store.recall_candidates(users.x, unit(5), kind="thread", top_k=10,
                                        persona_id=users.persona)
    assert [g["id"] for g in got] == [ids["x_near"], ids["x_mid"], ids["x_far"]]
    sims = [g["similarity"] for g in got]
    assert sims == sorted(sims, reverse=True) and sims[0] == pytest.approx(1.0)
    assert got[0]["kind"] == "thread" and got[0]["user_id"] == users.x
    assert "query_embedding" not in got[0]
    # top-k cut
    two = await store.recall_candidates(users.x, unit(5), top_k=2, persona_id=users.persona)
    assert [g["id"] for g in two] == [ids["x_near"], ids["x_mid"]]
    # simulated is a separate, unattributed pool
    sim = await store.recall_candidates(None, unit(5), kind="simulated", persona_id=users.persona)
    assert [g["id"] for g in sim] == [ids["sim_near"]]
    # a stale, expired or embedding-less row is not a candidate
    await store.mark_stale([ids["x_near"]])
    async with engine.begin() as c:
        await c.execute(
            text("UPDATE trellion_traversal_records SET expires_at = NOW() - interval '1 day' "
                 "WHERE id = CAST(:i AS uuid)"), {"i": ids["x_mid"]})
    got = await store.recall_candidates(users.x, unit(5), persona_id=users.persona)
    assert [g["id"] for g in got] == [ids["x_far"]]


@needs_db
async def test_touch_stale_and_stage_behave(engine, users):
    ids = await _seed(engine, users)
    store = WalkStore(engine)
    n = await store.touch_recalled([ids["x_near"], ids["x_mid"]])
    assert n == 2
    await store.touch_recalled([ids["x_near"]])
    near = await _row(engine, ids["x_near"])
    mid = await _row(engine, ids["x_mid"])
    far = await _row(engine, ids["x_far"])
    assert near["recall_count"] == 2 and near["last_recalled_at"] is not None
    assert mid["recall_count"] == 1 and far["recall_count"] == 0
    assert far["last_recalled_at"] is None
    # owner guard: y cannot touch or stale x's row
    assert await store.touch_recalled([ids["x_far"]], user_id=users.y) == 0
    assert await store.mark_stale([ids["x_far"]], user_id=users.y) == 0
    assert await store.mark_stale([ids["x_far"]], user_id=users.x) == 1
    assert (await _row(engine, ids["x_far"]))["status"] == "stale"
    assert await store.mark_stale([ids["x_far"]]) == 0  # only active rows move
    assert await store.mark_stale([]) == 0
    # stage: x's rows leave recall, y and simulated untouched
    assert await store.stage_for_user(users.x) == 3
    assert {(await _row(engine, ids[k]))["status"] for k in ("x_near", "x_mid", "x_far")} == {"staged"}
    assert (await _row(engine, ids["y_near"]))["status"] == "active"
    assert (await _row(engine, ids["sim_near"]))["status"] == "active"
    assert await store.recall_candidates(users.x, unit(5), persona_id=users.persona) == []
    assert await store.stage_for_user(users.x) == 0


@needs_db
async def test_delete_for_user_removes_traversals_and_episodes_only_for_that_user(engine, users):
    ids = await _seed(engine, users)
    store = WalkStore(engine)
    assert await store.delete_for_user(users.x, kind="simulated") == {"traversals": 0, "episodes": 0}
    res = await store.delete_for_user(users.x)
    assert res == {"traversals": 3, "episodes": 3}
    async with engine.connect() as c:
        left = (await c.execute(
            text("SELECT user_id FROM trellion_traversal_records WHERE persona_id=:p ORDER BY user_id NULLS LAST"),
            {"p": users.persona})).scalars().all()
        eps = (await c.execute(
            text("SELECT user_id FROM trellion_episodes WHERE persona_id=:p ORDER BY user_id NULLS LAST"),
            {"p": users.persona})).scalars().all()
    assert left == [users.y, None] and eps == [users.y, None]
    assert ids  # y and simulated survive


# ---- the migration -----------------------------------------------------

_ROOT = Path(__file__).resolve().parents[1]


def _alembic(dsn: str, *args: str) -> None:
    env = {**os.environ, "DATABASE_URL": dsn}
    subprocess.run(
        [sys.executable, "-m", "alembic", "-c", str(_ROOT / "src/trellion/db/alembic.ini"), *args],
        check=True, env=env, cwd=_ROOT, capture_output=True,
    )


@needs_db
async def test_migration_104_backfills_existing_rows_and_adds_the_indexes():
    import asyncpg
    from sqlalchemy.engine import make_url

    url = make_url(DSN)
    plain = url.set(drivername="postgresql")
    name = f"walks_mig_{uuid.uuid4().hex[:8]}"
    admin = await asyncpg.connect(plain.render_as_string(hide_password=False))
    try:
        await admin.execute(f'CREATE DATABASE "{name}"')
        scratch = plain.set(database=name).render_as_string(hide_password=False)
        con = await asyncpg.connect(scratch)
        try:
            await con.execute("CREATE EXTENSION IF NOT EXISTS vector; CREATE EXTENSION IF NOT EXISTS pg_trgm")
            _alembic(scratch, "upgrade", "103_extraction_converges")
            await con.execute("SET search_path = trellion, public")
            f1, f2 = str(uuid.uuid4()), str(uuid.uuid4())
            await con.execute(
                "INSERT INTO trellion_episodes (id, persona_id, record_type, index_name, content_summary) "
                "VALUES ($1::uuid,'p','traversal','traversals','a'), ($2::uuid,'p','traversal','traversals','b')",
                f1, f2)
            await con.execute(
                "INSERT INTO trellion_traversal_records (id, persona_id, query_text, outcome_score, user_id, video_frame_id) "
                "VALUES (gen_random_uuid(),'p','hb',0.7,NULL,$1::uuid), (gen_random_uuid(),'p','own',0.9,'alice',$2::uuid)",
                f1, f2)
            _alembic(scratch, "upgrade", "104_recallable_walks")
            rows = await con.fetch(
                "SELECT kind, status, recall_count, provenance::text AS pv, evidence::text AS ev, "
                "thread_id, usefulness FROM trellion_traversal_records")
            assert len(rows) == 2
            for r in rows:
                assert (r["kind"], r["status"], r["recall_count"]) == ("simulated", "active", 0)
                assert (r["pv"], r["ev"], r["thread_id"], r["usefulness"]) == ("{}", "{}", None, None)
            # existing episodes are attributed from the record that points at them
            owners = dict(await con.fetch("SELECT id::text, user_id FROM trellion_episodes"))
            assert owners == {f1: None, f2: "alice"}
            idx = {r[0] for r in await con.fetch(
                "SELECT indexname FROM pg_indexes WHERE schemaname='trellion' "
                "AND tablename IN ('trellion_traversal_records','trellion_episodes')")}
            assert {"ix_trellion_traversal_user_kind_status", "ix_trellion_traversal_thread",
                    "ix_trellion_episodes_user_id"} <= idx
            with pytest.raises(asyncpg.CheckViolationError):
                await con.execute("UPDATE trellion_traversal_records SET kind='bogus'")
            with pytest.raises(asyncpg.CheckViolationError):
                await con.execute("UPDATE trellion_traversal_records SET status='bogus'")
            # idempotent re-run, and reversible
            _alembic(scratch, "downgrade", "103_extraction_converges")
            cols = {r[0] for r in await con.fetch(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema='trellion' AND table_name='trellion_traversal_records'")}
            assert "kind" not in cols and "expires_at" not in cols
            _alembic(scratch, "upgrade", "head")
            assert await con.fetchval(
                "SELECT count(*) FROM trellion_traversal_records WHERE kind='simulated'") == 2
        finally:
            await con.close()
    finally:
        await admin.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
        await admin.close()
