# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""Consolidation request shape + failure reporting (noomesh followup 1311).

``POST /api/snapshots/consolidate`` failed with ``stats = {"error":
"'canonical_name'"}``: a topic slice (the default kind) reached the slicer with
no params, and ``params["canonical_name"]`` raised a bare KeyError whose
``str()`` is only the key. The slicer now says what is missing and what
arrived, and every stored failure reason carries the exception type.

Unit tests need no database. The DB test (TRELLION_TEST_DSN, see
test_batch_snapshot.py) runs a real consolidation over a small graph.
"""

from __future__ import annotations

import os
import uuid

import pytest
from sqlalchemy import text

from trellion.snapshots import SliceSpec, SnapshotOrchestrator
from trellion.snapshots.slicer import GraphSlicer
from trellion.snapshots.store import describe_failure

DSN = os.environ.get("TRELLION_TEST_DSN")


def test_topic_slice_without_params_names_the_missing_param():
    with pytest.raises(ValueError) as e:
        SliceSpec(kind="topic", params={}).validate()
    msg = str(e.value)
    assert "canonical_name" in msg and "topic" in msg
    # What arrived is part of the diagnosis (a flat body leaves params empty).
    assert "got params keys []" in msg


def test_validate_lists_what_did_arrive():
    with pytest.raises(ValueError, match=r"\['from', 'to'\].*\['hops'\]"):
        SliceSpec(kind="time_window", params={"hops": 2}).validate()


def test_validate_accepts_each_kind_with_its_params():
    SliceSpec("topic", {"canonical_name": "X"}).validate()
    SliceSpec("time_window", {"from": "a", "to": "b"}).validate()
    SliceSpec("user_attention", {"user_id": "alice"}).validate()


def test_unknown_kind_is_a_value_error_naming_the_choices():
    with pytest.raises(ValueError, match="Unknown slice kind.*topic"):
        SliceSpec("nope", {}).validate()


async def test_slicer_never_raises_a_bare_keyerror():
    slicer = GraphSlicer(engine=None)  # type: ignore[arg-type]  # fails before any query
    with pytest.raises(ValueError):
        await slicer.slice(SliceSpec(kind="topic", params={}))


def test_failure_reason_format_is_pinned():
    assert describe_failure(KeyError("canonical_name")) == "KeyError: 'canonical_name'"
    assert describe_failure(ValueError("bad slice")) == "ValueError: bad slice"
    assert len(describe_failure(RuntimeError("x" * 900))) == 500


@pytest.mark.skipif(not DSN, reason="TRELLION_TEST_DSN not set — see test_batch_snapshot.py")
async def test_consolidation_over_a_small_graph_completes():
    from trellion.db.engine import init_engine

    _, engine = init_engine(DSN)
    tag = uuid.uuid4().hex[:8]
    names = [f"fu1311-alpha-{tag}", f"fu1311-beta-{tag}", f"fu1311-gamma-{tag}"]
    ids = [str(uuid.uuid4()) for _ in names]
    async with engine.begin() as conn:
        for i, n in zip(ids, names):
            await conn.execute(
                text(
                    "INSERT INTO noo_entities_typed (id, type, canonical_name) "
                    "VALUES (CAST(:i AS uuid), 'topic', :n)"
                ),
                {"i": i, "n": n},
            )
        for a, b in ((0, 1), (1, 2)):
            await conn.execute(
                text(
                    "INSERT INTO noo_temporal_edges "
                    "(id, subject_id, predicate, object_entity_id) VALUES "
                    "(gen_random_uuid(), CAST(:a AS uuid), 'RELATES', CAST(:b AS uuid))"
                ),
                {"a": ids[a], "b": ids[b]},
            )

    orch = SnapshotOrchestrator(engine)
    res = await orch.consolidate(
        SliceSpec("topic", {"entity_type": "topic", "canonical_name": names[0], "hops": 2})
    )
    assert res.status == "complete", res.stats
    assert res.stats["live_entities"] == 3 and res.stats["concepts"] == 3

    # A spec with no seed is refused before any snapshot row is created.
    async with engine.connect() as conn:
        before = (await conn.execute(text("SELECT count(*) FROM trellion_snapshots"))).scalar()
    with pytest.raises(ValueError, match="canonical_name"):
        await orch.consolidate(SliceSpec("topic", {}))
    async with engine.connect() as conn:
        after = (await conn.execute(text("SELECT count(*) FROM trellion_snapshots"))).scalar()
    assert after == before
