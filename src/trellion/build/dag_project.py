# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""Project an arbitrary typed edge set to a DAG suitable for CKG.

Why this exists: GraphRAG (and our extractor) will happily produce cycles and
multiple relation types. CKG retrieval assumes a DAG per relation type. This
module turns one into the other.

Algorithm:
  1. For each priority relation type:
     a. Keep only edges of that type (and high-priority types above it).
     b. Aggregate duplicate (src,tgt) pairs by summing weight.
     c. Detect cycles; break them using either:
        - "min_weight": remove the lowest-weight edge on each cycle.
        - "eades": Eades-Lin-Smyth feedback-arc heuristic (default).
  2. Return a DAG as a list of CanonicalEdge.

The loser edges are logged — useful for debugging why a relation didn't
survive into the final CSV.
"""

from __future__ import annotations

import logging
from collections import defaultdict

import networkx as nx

from trellion.common.config import DAGProjectionConfig
from trellion.common.schema import CanonicalEdge

logger = logging.getLogger(__name__)


def project_to_dag(
    edges: list[CanonicalEdge],
    cfg: DAGProjectionConfig,
) -> list[CanonicalEdge]:
    # Partition by priority relation type; drop anything not on the priority list.
    by_rel: dict[str, list[CanonicalEdge]] = defaultdict(list)
    priorities = set(cfg.priority_relations)
    for e in edges:
        if e.relation_type in priorities:
            by_rel[e.relation_type].append(e)

    out: list[CanonicalEdge] = []
    for rel in cfg.priority_relations:
        rel_edges = by_rel.get(rel, [])
        if not rel_edges:
            continue

        agg = _aggregate(rel_edges)
        g = nx.DiGraph()
        for (s, t), (w, evidence) in agg.items():
            g.add_edge(s, t, weight=w, evidence=evidence)

        before = g.number_of_edges()
        if cfg.cycle_break_strategy == "min_weight":
            _break_cycles_min_weight(g)
        else:
            _break_cycles_eades(g)
        after = g.number_of_edges()
        if before != after:
            logger.info(
                "dag_project[%s]: removed %d edges to break cycles (%d -> %d)",
                rel, before - after, before, after,
            )

        for s, t, data in g.edges(data=True):
            out.append(
                CanonicalEdge(
                    source_id=s,
                    target_id=t,
                    relation_type=rel,
                    weight=float(data.get("weight", 1.0)),
                    evidence_chunks=list(data.get("evidence", [])),
                )
            )

    return out


def _aggregate(edges: list[CanonicalEdge]) -> dict[tuple[str, str], tuple[float, list[str]]]:
    agg: dict[tuple[str, str], tuple[float, list[str]]] = {}
    for e in edges:
        key = (e.source_id, e.target_id)
        if key not in agg:
            agg[key] = (e.weight, list(e.evidence_chunks))
        else:
            w, ev = agg[key]
            agg[key] = (w + e.weight, ev + e.evidence_chunks)
    return agg


def _break_cycles_min_weight(g: nx.DiGraph) -> None:
    while True:
        try:
            cycle = nx.find_cycle(g, orientation="original")
        except nx.NetworkXNoCycle:
            return
        # cycle is list of (u, v, key/direction) tuples.
        victim = min(cycle, key=lambda e: g[e[0]][e[1]].get("weight", 1.0))
        g.remove_edge(victim[0], victim[1])


def _break_cycles_eades(g: nx.DiGraph) -> None:
    """Eades-Lin-Smyth heuristic for the feedback arc set problem.

    Produces a linear ordering; edges that go 'backwards' in the ordering are
    the feedback arcs we remove. Runs in O(|V| + |E|).
    """
    remaining = g.copy()
    s1: list[str] = []
    s2: list[str] = []

    while remaining.number_of_nodes():
        # Repeatedly strip sinks.
        changed = True
        while changed:
            changed = False
            sinks = [n for n in remaining.nodes if remaining.out_degree(n) == 0]
            for n in sinks:
                s2.insert(0, n)
                remaining.remove_node(n)
                changed = True
            sources = [n for n in remaining.nodes if remaining.in_degree(n) == 0]
            for n in sources:
                s1.append(n)
                remaining.remove_node(n)
                changed = True
        if remaining.number_of_nodes() == 0:
            break
        # Pick the vertex maximizing out-degree - in-degree; push to s1.
        pivot = max(
            remaining.nodes,
            key=lambda n: remaining.out_degree(n) - remaining.in_degree(n),
        )
        s1.append(pivot)
        remaining.remove_node(pivot)

    ordering = {n: i for i, n in enumerate(s1 + s2)}
    feedback = [(u, v) for u, v in g.edges if ordering[u] > ordering[v]]
    for u, v in feedback:
        g.remove_edge(u, v)
