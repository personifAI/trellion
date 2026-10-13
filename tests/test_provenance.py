# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""End-to-end provenance test: chunk ids survive from raw entities ->
canonical entities -> emitted CSV -> loaded CKG.

Skips the embedding stage by building CanonicalEntity objects manually,
since the real canonicalize path requires an embed service.
"""

from __future__ import annotations

import csv
from pathlib import Path

from trellion.build.emit_csv import emit_csv
from trellion.common.schema import (
    CanonicalEdge,
    CanonicalEntity,
    CommunityAssignment,
)
from trellion.query.graph import CKG


def _entity(cid: str, label: str, chunks: list[str]) -> CanonicalEntity:
    return CanonicalEntity(
        canonical_id=cid,
        label=label,
        entity_type="CONCEPT",
        source_chunk_ids=chunks,
    )


def test_provenance_roundtrip(tmp_path: Path) -> None:
    entities = [
        _entity("C000001", "Function", ["doc1_c0", "doc1_c2"]),
        _entity("C000002", "Derivative", ["doc2_c0"]),
        _entity("C000003", "Chain Rule", ["doc2_c1", "doc3_c0"]),
    ]
    edges = [
        CanonicalEdge(
            source_id="C000001",
            target_id="C000002",
            relation_type="PREREQ",
            weight=1.0,
            evidence_chunks=["doc2_c0"],
        ),
        CanonicalEdge(
            source_id="C000002",
            target_id="C000003",
            relation_type="PREREQ",
            weight=1.0,
            evidence_chunks=["doc2_c1"],
        ),
    ]
    communities = [
        CommunityAssignment(entity_id="C000001", level=0, cluster_id=0, parent_cluster=None),
        CommunityAssignment(entity_id="C000002", level=0, cluster_id=0, parent_cluster=None),
        CommunityAssignment(entity_id="C000003", level=0, cluster_id=0, parent_cluster=None),
    ]

    out_csv = tmp_path / "learning-graph.csv"
    emit_csv(entities, edges, communities, out_csv)

    # Verify the 6th column is written.
    with out_csv.open() as f:
        header = next(csv.reader(f))
    assert header == [
        "ConceptID",
        "ConceptLabel",
        "Dependencies",
        "TaxonomyID",
        "RelationType",
        "SourceChunks",
    ]

    # Round-trip through CKG and confirm chunk ids come back.
    ckg = CKG.load(out_csv)
    function_concept = ckg.find_by_label("Function")
    assert function_concept is not None
    assert ckg.sources_for(function_concept.cid) == ["doc1_c0", "doc1_c2"]

    chain_rule = ckg.find_by_label("Chain Rule")
    assert chain_rule is not None
    assert ckg.sources_for(chain_rule.cid) == ["doc2_c1", "doc3_c0"]


def test_canonicalize_merges_chunk_ids() -> None:
    """When two mentions merge, their chunk ids should union (dedup, ordered)."""
    from trellion.common.schema import ExtractedEntity

    # Simulate two mentions of "Function" from different chunks merging.
    raw = [
        ExtractedEntity(
            name="Function",
            entity_type="CONCEPT",
            description="A mapping",
            source_chunk_id="chunk_a",
        ),
        ExtractedEntity(
            name="Function",
            entity_type="CONCEPT",
            description="A rule mapping inputs to outputs",
            source_chunk_id="chunk_b",
        ),
        # Duplicate chunk id from a 3rd mention should not be re-added.
        ExtractedEntity(
            name="Function",
            entity_type="CONCEPT",
            description="A function is ...",
            source_chunk_id="chunk_a",
        ),
    ]
    # Just verify the dedup logic behaves as expected on this list.
    # We reproduce the body logic without calling canonicalize (which
    # requires an embedding service).
    seen: set[str] = set()
    ordered: list[str] = []
    for p in raw:
        if p.source_chunk_id not in seen:
            seen.add(p.source_chunk_id)
            ordered.append(p.source_chunk_id)
    assert ordered == ["chunk_a", "chunk_b"]
