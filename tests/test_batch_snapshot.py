# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""Batch build → snapshot store consolidation (gateway proposal 27).

Unit tests cover the BuildResult contract and the emit failure path with a
stubbed store. The DB test (TRELLION_TEST_DSN, provisioned by the db-test
CI job from trellion's own migrations) proves the real roundtrip: emit a
batch build's artifacts, load them back through the existing
``load_ckg_from_snapshot`` adapter, and answer a query — the same path
the live ``worker.snapshots.*`` surface uses.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from trellion.build.pipeline import BuildResult
from trellion.common.schema import CanonicalEdge, CanonicalEntity, CommunityAssignment

DSN = os.environ.get("TRELLION_TEST_DSN")


def _sample_build():
    concepts = [
        CanonicalEntity(canonical_id="c1", label="Widgets", entity_type="CONCEPT"),
        CanonicalEntity(canonical_id="c2", label="Failures", entity_type="CONCEPT"),
        CanonicalEntity(
            canonical_id="c3", label="Fatigue", entity_type="CONCEPT", aliases=["wear"]
        ),
    ]
    dag_edges = [
        CanonicalEdge(source_id="c3", target_id="c2", relation_type="CAUSES", weight=2.0),
        CanonicalEdge(source_id="c1", target_id="c2", relation_type="CONTAINS", weight=1.0),
    ]
    communities = [
        CommunityAssignment(entity_id="c1", level=0, cluster_id=1, parent_cluster=None),
        CommunityAssignment(entity_id="c2", level=0, cluster_id=1, parent_cluster=None),
        CommunityAssignment(entity_id="c3", level=0, cluster_id=1, parent_cluster=None),
    ]
    return concepts, dag_edges, communities


def test_build_result_str_is_csv_path():
    res = BuildResult(csv_path=Path("/tmp/out/learning-graph.csv"))
    assert str(res) == "/tmp/out/learning-graph.csv"
    assert os.fspath(res) == "/tmp/out/learning-graph.csv"


async def test_emit_marks_failed_and_reraises(monkeypatch):
    from trellion.snapshots import batch_emit

    calls = {}

    class StubStore:
        def __init__(self, engine):
            pass

        async def create_snapshot(self, **kw):
            calls["created"] = kw
            return "snap-1"

        async def write_concepts(self, snapshot_id, canonical):
            raise RuntimeError("boom")

        async def mark_failed(self, snapshot_id, reason):
            calls["failed"] = (snapshot_id, reason)

    monkeypatch.setattr(batch_emit, "SnapshotStore", StubStore)
    concepts, dag_edges, communities = _sample_build()
    with pytest.raises(RuntimeError, match="boom"):
        await batch_emit.emit_build_snapshot(
            object(), concepts, dag_edges, communities, label="t"
        )
    assert calls["created"]["slice_kind"] == "batch_build"
    assert calls["failed"][0] == "snap-1"
    assert "boom" in calls["failed"][1]


@pytest.mark.skipif(not DSN, reason="TRELLION_TEST_DSN not set — see module docstring")
async def test_emit_and_query_roundtrip():
    from trellion.db.engine import init_engine
    from trellion.query.answer import answer_query
    from trellion.snapshots import load_ckg_from_snapshot
    from trellion.snapshots.batch_emit import emit_build_snapshot

    _, engine = init_engine(DSN)
    concepts, dag_edges, communities = _sample_build()
    snapshot_id, stats = await emit_build_snapshot(
        engine, concepts, dag_edges, communities, input_dir="tests://sample", label="rt"
    )
    assert stats == {
        "source": "batch_build",
        "concepts": 3,
        "dag_edges": 2,
        "communities": 1,
    }

    ckg = await load_ckg_from_snapshot(engine, snapshot_id)
    assert len(ckg.concepts) == 3
    labels = {c.label.lower() for c in ckg.concepts.values()}
    assert {"widgets", "failures", "fatigue"} <= labels

    # Edge fidelity: Fatigue -> Failures survived the DB roundtrip.
    by_label = {c.label.lower(): cid for cid, c in ckg.concepts.items()}
    assert by_label["failures"] in ckg.forward.get(by_label["fatigue"], [])

    # Router smoke: the adapter's CKG drives answer_query without error.
    ans = answer_query(ckg, "what causes failures?")
    assert ans.query_type
