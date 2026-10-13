# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""Unit tests for DAG projection (cycle breaking)."""

from __future__ import annotations

from trellion.build.dag_project import project_to_dag
from trellion.common.config import DAGProjectionConfig
from trellion.common.schema import CanonicalEdge


def _edge(s: str, t: str, w: float = 1.0, rel: str = "PREREQ") -> CanonicalEdge:
    return CanonicalEdge(source_id=s, target_id=t, relation_type=rel, weight=w)


def test_passthrough_dag() -> None:
    edges = [_edge("A", "B"), _edge("B", "C"), _edge("A", "C", w=2.0)]
    cfg = DAGProjectionConfig()
    out = project_to_dag(edges, cfg)
    pairs = {(e.source_id, e.target_id) for e in out}
    assert pairs == {("A", "B"), ("B", "C"), ("A", "C")}


def test_breaks_cycle_eades() -> None:
    # A -> B -> C -> A, weight 1/1/1. Eades should linearize and drop one.
    edges = [_edge("A", "B"), _edge("B", "C"), _edge("C", "A")]
    cfg = DAGProjectionConfig(cycle_break_strategy="eades")
    out = project_to_dag(edges, cfg)
    assert len(out) == 2


def test_breaks_cycle_min_weight() -> None:
    # Min-weight strategy drops the lowest-weight edge (C->A at weight 0.5).
    edges = [_edge("A", "B", w=5), _edge("B", "C", w=5), _edge("C", "A", w=0.5)]
    cfg = DAGProjectionConfig(cycle_break_strategy="min_weight")
    out = project_to_dag(edges, cfg)
    pairs = {(e.source_id, e.target_id) for e in out}
    assert ("C", "A") not in pairs
    assert ("A", "B") in pairs and ("B", "C") in pairs


def test_drops_non_priority_relations() -> None:
    edges = [_edge("A", "B", rel="PREREQ"), _edge("B", "C", rel="MENTIONED_WITH")]
    cfg = DAGProjectionConfig(priority_relations=["PREREQ"])
    out = project_to_dag(edges, cfg)
    assert len(out) == 1
    assert out[0].relation_type == "PREREQ"
