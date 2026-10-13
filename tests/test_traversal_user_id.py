# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""The traversal archiver records WHO ASKED (noomesh followup 1306).

``trellion_traversal_records.user_id`` existed (migration 100) but the archiver
never wrote it, so audience-scoped recall in an add-on treated every
record as unattributed. These pin: the id is written verbatim, omitting it is
NULL (every pre-existing caller), and the video-layer shim forwards it from the
snapshot. The first group needs no database; the last runs against
``TRELLION_TEST_DSN`` like ``test_video_pgvector.py``.
"""

from __future__ import annotations

import os
import uuid

import pytest

from trellion.live.recall import TraversalArchiver
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
        (sql, params), = [x for x in self.log if "trellion_traversal_records" in x[0]]
        return sql, params


async def _capture(**kw):
    eng = FakeEngine()
    out = await TraversalArchiver("/x", eng).capture(
        "p", "what did I ask?", "causal", PATH, 0.9, **kw
    )
    assert out and out["archived"], out
    return eng.traversal_insert()


async def test_user_id_is_written_verbatim_to_the_row():
    sql, params = await _capture(user_id="alice")
    assert "user_id" in sql
    assert params["user_id"] == "alice"


async def test_omitting_user_id_writes_null_so_old_callers_are_unchanged():
    _, params = await _capture()
    assert params["user_id"] is None


async def test_an_empty_user_id_is_null_not_an_empty_string():
    # '' would match no account but would still read as "attributed".
    _, params = await _capture(user_id="")
    assert params["user_id"] is None


async def test_a_snapshot_with_a_user_is_forwarded_by_the_video_shim():
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
        user_id="alice",
    )
    shim.capture(snap)
    import asyncio

    await asyncio.sleep(0)
    assert seen["user_id"] == "alice"


async def test_a_snapshot_without_a_user_forwards_none():
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
    assert seen["user_id"] is None


# ---- real Postgres -----------------------------------------------------

DSN = os.environ.get("TRELLION_TEST_DSN")
needs_db = pytest.mark.skipif(not DSN, reason="TRELLION_TEST_DSN not set")


@needs_db
async def test_the_row_in_postgres_carries_the_user_id():
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    from trellion.db.engine import TRELLION_SCHEMA, _async_dsn

    eng = create_async_engine(
        _async_dsn(DSN), connect_args={"options": f"-csearch_path={TRELLION_SCHEMA},public"}
    )
    persona = f"test-{uuid.uuid4().hex[:8]}"
    try:
        arch = TraversalArchiver(None, eng, min_outcome_score=0.5)
        a = await arch.capture(persona, "asked by alice", None, PATH, 0.9, user_id="alice")
        b = await arch.capture(persona, "no asker", None, PATH, 0.9)
        async with eng.connect() as c:
            rows = dict(
                (await c.execute(
                    text("SELECT id::text, user_id FROM trellion_traversal_records WHERE persona_id=:p"),
                    {"p": persona},
                )).all()
            )
        assert rows[a["traversal_id"]] == "alice"
        assert rows[b["traversal_id"]] is None
    finally:
        async with eng.begin() as c:
            await c.execute(text("DELETE FROM trellion_traversal_records WHERE persona_id=:p"), {"p": persona})
            await c.execute(text("DELETE FROM trellion_episodes WHERE persona_id=:p"), {"p": persona})
        await eng.dispose()
