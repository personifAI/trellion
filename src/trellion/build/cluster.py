# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""Hierarchical Leiden over the canonicalized edge set.

Returns assignments at each level; emit_csv uses the top-level partition as
TaxonomyID.

WHY NOT graspologic
-------------------
graspologic ships `hierarchical_leiden` and we used it here, but it pins
`numpy>=1.26.4,<2.0.0` in every published release. That window holds exactly
one numpy version, which predates Python 3.13 and so has no wheel for it —
every 3.13 install compiled numpy from source, and CI's 3.13 leg died on a
missing Python.h. Its successor `graspologic-native` publishes cp39-only
wheels (no abi3), so it would have swapped a numpy source build for a Rust
one.

leidenalg + igraph both publish **abi3** wheels (cp38-abi3 / cp39-abi3), which
install on 3.13 with no toolchain, and neither pulls numpy at all. What they
do not ship is a hierarchical mode — so the recursive split that graspologic
gave us for free is written out below. It is the same construction its
docstring describes: partition, then re-partition any community that is still
larger than `max_cluster_size`.

WHAT "HIERARCHICAL" MEANS HERE (it is not a full tree)
------------------------------------------------------
Level 0 holds every node. A node is re-emitted at level N+1 **only** if the
level-N cluster it landed in was oversized and got split again. So a node that
lands in a small cluster early appears once, at a shallow level, and never
deeper. Consumers that filter on a single level (emit_csv takes `max(level)`)
will therefore miss those nodes — that is inherited graspologic behaviour, not
new here.

Cluster ids are unique across the whole result, not per level.

Output is NOT bit-identical to the old graspologic runs: both are modularity
Leiden, but tie-breaking and community numbering differ. Cluster ids were never
stable identifiers (they are positional), so this only matters if you diff two
snapshots built by different implementations.
"""

from __future__ import annotations

import logging

import networkx as nx

from trellion.common.config import ClusterConfig
from trellion.common.schema import CanonicalEdge, CommunityAssignment

logger = logging.getLogger(__name__)

_INSTALL_HINT = (
    "hierarchical clustering needs leidenalg and igraph; "
    "install trellion[clustering]"
)


def cluster(
    edges: list[CanonicalEdge],
    cfg: ClusterConfig,
) -> list[CommunityAssignment]:
    if not edges:
        return []

    g = nx.Graph()
    for e in edges:
        # Sum weight if the pair already exists (undirected for modularity).
        if g.has_edge(e.source_id, e.target_id):
            g[e.source_id][e.target_id]["weight"] += e.weight
        else:
            g.add_edge(e.source_id, e.target_id, weight=e.weight)

    if cfg.use_lcc and g.number_of_nodes() > 0:
        # Largest connected component — isolated islands get dropped.
        components = sorted(nx.connected_components(g), key=len, reverse=True)
        g = g.subgraph(components[0]).copy()

    return hierarchical_leiden(
        g,
        max_cluster_size=cfg.max_cluster_size,
        seed=cfg.seed,
    )


def hierarchical_leiden(
    g: nx.Graph,
    *,
    max_cluster_size: int,
    seed: int,
) -> list[CommunityAssignment]:
    """Leiden, recursively re-applied to any community over max_cluster_size."""
    try:
        import igraph as ig
        import leidenalg
    except ImportError as exc:  # pragma: no cover - covered by the install docs
        raise RuntimeError(_INSTALL_HINT) from exc

    if g.number_of_nodes() == 0:
        return []

    assignments: list[CommunityAssignment] = []
    next_cluster_id = 0
    level = 0
    # (nodes to partition, id of the cluster they came from)
    frontier: list[tuple[list[str], int | None]] = [(sorted(g.nodes()), None)]

    while frontier:
        next_frontier: list[tuple[list[str], int | None]] = []
        for nodes, parent in frontier:
            communities = _partition(ig, leidenalg, g, nodes, seed)

            # A group that refuses to split would recurse forever: Leiden can
            # return one community for a dense clique no matter how large it
            # is. Accept it as final rather than looping.
            if len(communities) < 2 and level > 0:
                logger.debug(
                    "cluster: %d nodes under cluster %s did not split further",
                    len(nodes),
                    parent,
                )
                continue

            for members in communities:
                cluster_id = next_cluster_id
                next_cluster_id += 1
                for node in members:
                    assignments.append(
                        CommunityAssignment(
                            entity_id=str(node),
                            level=level,
                            cluster_id=cluster_id,
                            parent_cluster=parent,
                        )
                    )
                if len(members) > max_cluster_size and len(communities) > 1:
                    next_frontier.append((members, cluster_id))

        frontier = next_frontier
        level += 1

    logger.info(
        "cluster: %d assignments, %d clusters across %d level(s)",
        len(assignments),
        next_cluster_id,
        level,
    )
    return assignments


def _partition(
    ig,
    leidenalg,
    g: nx.Graph,
    nodes: list[str],
    seed: int,
) -> list[list[str]]:
    """Modularity Leiden over the subgraph induced by `nodes`.

    Returns communities as lists of networkx node ids, each sorted, and the
    list itself sorted — igraph's community ordering is not guaranteed stable,
    and callers number clusters positionally.
    """
    if len(nodes) == 1:
        return [list(nodes)]

    index = {node: i for i, node in enumerate(nodes)}
    sub = g.subgraph(nodes)

    edges: list[tuple[int, int]] = []
    weights: list[float] = []
    for u, v, data in sub.edges(data=True):
        edges.append((index[u], index[v]))
        weights.append(float(data.get("weight", 1.0)))

    graph = ig.Graph(n=len(nodes), edges=edges, directed=False)
    partition = leidenalg.find_partition(
        graph,
        leidenalg.ModularityVertexPartition,
        weights=weights or None,
        seed=seed,
    )

    communities = [sorted(nodes[i] for i in members) for members in partition if members]
    communities.sort()
    return communities
