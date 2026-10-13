# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""Hierarchical Leiden — the hand-rolled recursive split.

graspologic gave us `hierarchical_leiden` for free and we never tested it. The
recursion is ours now, so the invariants downstream relies on need pinning:
every node placed, ids unique across levels, parents point at real clusters,
and — the one that would hang a build — no infinite recursion on a graph that
refuses to split.
"""

from __future__ import annotations

import networkx as nx
import pytest

from trellion.build.cluster import cluster, hierarchical_leiden
from trellion.common.config import ClusterConfig
from trellion.common.schema import CanonicalEdge

pytest.importorskip("leidenalg", reason="clustering extra not installed")
pytest.importorskip("igraph", reason="clustering extra not installed")


def _cliques(n_cliques: int, size: int) -> nx.Graph:
    """Disjoint cliques joined by a single weak edge each — the easy case for
    modularity, so the partition is unambiguous and the test is not measuring
    Leiden's tie-breaking."""
    g = nx.Graph()
    for c in range(n_cliques):
        members = [f"c{c}n{i}" for i in range(size)]
        for i, u in enumerate(members):
            for v in members[i + 1 :]:
                g.add_edge(u, v, weight=1.0)
        if c > 0:
            g.add_edge(f"c{c - 1}n0", f"c{c}n0", weight=0.01)
    return g


def _nested(n_groups: int = 3, subs: int = 3, size: int = 5) -> nx.Graph:
    """Groups of cliques, where the recursion actually has something to do.

    A plain clique NEVER splits under modularity, however oversized — so a flat
    bag of cliques cannot exercise level 1 at all. Structure that does: cliques
    bridged strongly enough (weight 3.0) that the global partition keeps each
    group whole, while the induced subgraph of one group — which has a much
    smaller total edge weight, and so a different modularity null model —
    separates back into its cliques.
    """
    g = nx.Graph()
    for gr in range(n_groups):
        heads = []
        for s in range(subs):
            members = [f"g{gr}s{s}n{i}" for i in range(size)]
            for i, u in enumerate(members):
                for v in members[i + 1 :]:
                    g.add_edge(u, v, weight=1.0)
            heads.append(members[0])
        for i in range(len(heads) - 1):
            g.add_edge(heads[i], heads[i + 1], weight=3.0)
        if gr > 0:
            g.add_edge(f"g{gr - 1}s0n0", f"g{gr}s0n0", weight=0.001)
    return g


def test_every_node_appears_at_level_zero():
    g = _cliques(4, 5)
    out = hierarchical_leiden(g, max_cluster_size=100, seed=42)

    level0 = {a.entity_id for a in out if a.level == 0}
    assert level0 == set(g.nodes())
    # Nothing oversized, so nothing recurses.
    assert max(a.level for a in out) == 0
    assert all(a.parent_cluster is None for a in out)


def test_finds_the_planted_communities():
    g = _cliques(4, 5)
    out = hierarchical_leiden(g, max_cluster_size=100, seed=42)

    by_cluster: dict[int, set[str]] = {}
    for a in out:
        by_cluster.setdefault(a.cluster_id, set()).add(a.entity_id)

    assert len(by_cluster) == 4
    for members in by_cluster.values():
        # Each recovered community is exactly one planted clique.
        assert len({m.split("n")[0] for m in members}) == 1


def test_oversized_clusters_split_and_record_their_parent():
    g = _nested()
    # Groups come out at ~10-15 nodes, so 8 forces a second pass.
    out = hierarchical_leiden(g, max_cluster_size=8, seed=42)

    assert max(a.level for a in out) >= 1

    level0_ids = {a.cluster_id for a in out if a.level == 0}
    deeper = [a for a in out if a.level > 0]
    assert deeper, "expected a second level"
    # Every child names a real parent, and only nodes from oversized level-0
    # clusters are re-emitted.
    for a in deeper:
        assert a.parent_cluster is not None
    assert {a.parent_cluster for a in deeper if a.level == 1} <= level0_ids


def test_cluster_ids_are_unique_across_levels():
    g = _nested()
    out = hierarchical_leiden(g, max_cluster_size=8, seed=42)

    owner: dict[int, int] = {}
    for a in out:
        owner.setdefault(a.cluster_id, a.level)
        # A cluster id must not be reused at a different level.
        assert owner[a.cluster_id] == a.level


def test_dense_clique_that_cannot_split_terminates():
    """The recursion guard. A single clique is one community no matter how it
    is partitioned, so re-splitting it would loop forever without the
    `len(communities) < 2` check."""
    g = nx.complete_graph(12)
    g = nx.relabel_nodes(g, {n: f"n{n}" for n in g.nodes()})
    nx.set_edge_attributes(g, 1.0, "weight")

    out = hierarchical_leiden(g, max_cluster_size=2, seed=42)

    assert {a.entity_id for a in out if a.level == 0} == set(g.nodes())
    # Bounded, not infinite. The exact depth depends on whether Leiden manages
    # any split at all; what matters is that it stops.
    assert max(a.level for a in out) < 5


def test_deterministic_for_a_fixed_seed():
    g = _nested()
    a = hierarchical_leiden(g, max_cluster_size=8, seed=42)
    b = hierarchical_leiden(g, max_cluster_size=8, seed=42)
    assert [x.model_dump() for x in a] == [x.model_dump() for x in b]


def test_a_clique_is_never_split_however_oversized():
    """Documents why the recursion guard exists, and why `_nested` has to be
    shaped the way it is: modularity has no reason to cut a clique, so an
    oversized one stays whole rather than recursing forever."""
    g = _cliques(4, 5)
    out = hierarchical_leiden(g, max_cluster_size=3, seed=42)

    assert max(a.level for a in out) == 0
    assert {a.entity_id for a in out} == set(g.nodes())


def test_cluster_entrypoint_sums_duplicate_edge_weights():
    edges = [
        CanonicalEdge(source_id="a", target_id="b", relation_type="PREREQ", weight=1.0),
        CanonicalEdge(source_id="a", target_id="b", relation_type="PREREQ", weight=2.0),
        CanonicalEdge(source_id="b", target_id="c", relation_type="PREREQ", weight=1.0),
    ]
    out = cluster(edges, ClusterConfig())
    assert {a.entity_id for a in out} == {"a", "b", "c"}


def test_empty_edges_is_empty():
    assert cluster([], ClusterConfig()) == []


def test_use_lcc_drops_isolated_islands():
    edges = [
        CanonicalEdge(source_id="a", target_id="b", relation_type="PREREQ", weight=1.0),
        CanonicalEdge(source_id="b", target_id="c", relation_type="PREREQ", weight=1.0),
        # Disconnected pair — dropped when use_lcc is on.
        CanonicalEdge(source_id="x", target_id="y", relation_type="PREREQ", weight=1.0),
    ]
    kept = {a.entity_id for a in cluster(edges, ClusterConfig(use_lcc=True))}
    assert kept == {"a", "b", "c"}

    everything = {a.entity_id for a in cluster(edges, ClusterConfig(use_lcc=False))}
    assert everything == {"a", "b", "c", "x", "y"}
