# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""Serialize canonical entities + DAG edges to the CKG CSV format.

Output schema (CKG's 4-column format plus 5th `RelationType` and 6th
`SourceChunks` columns):

    ConceptID,ConceptLabel,Dependencies,TaxonomyID,RelationType,SourceChunks

`Dependencies` is a pipe-delimited list of `ConceptID` values.
`RelationType` comes from the dominant relation type of this node's incoming edges.
`SourceChunks` is a pipe-delimited list of chunk ids this concept was
extracted from (provenance for citation / grounding). Readers that expect
the 5-column CKG format will simply ignore this column.
"""

from __future__ import annotations

import csv
import logging
from collections import defaultdict
from pathlib import Path

from trellion.common.schema import (
    CanonicalEdge,
    CanonicalEntity,
    CommunityAssignment,
)

logger = logging.getLogger(__name__)


def emit_csv(
    canonical_entities: list[CanonicalEntity],
    edges: list[CanonicalEdge],
    communities: list[CommunityAssignment],
    output_path: Path,
    *,
    default_relation: str = "PREREQ",
) -> None:
    # Assign integer ConceptIDs. Sort for deterministic output.
    entities_sorted = sorted(canonical_entities, key=lambda e: e.canonical_id)
    canon_to_cid = {e.canonical_id: i + 1 for i, e in enumerate(entities_sorted)}

    # Dependencies: for each target, the set of (source_id, relation_type) incoming edges.
    incoming: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for e in edges:
        if e.source_id in canon_to_cid and e.target_id in canon_to_cid:
            incoming[e.target_id].append((e.source_id, e.relation_type))

    # Top-level community assignment per entity -> TaxonomyID.
    top_level = max((a.level for a in communities), default=0)
    entity_to_tax: dict[str, str] = {}
    for a in communities:
        if a.level == top_level:
            entity_to_tax[a.entity_id] = f"CLUSTER_{a.cluster_id}"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "ConceptID",
                "ConceptLabel",
                "Dependencies",
                "TaxonomyID",
                "RelationType",
                "SourceChunks",
            ]
        )
        for e in entities_sorted:
            cid = canon_to_cid[e.canonical_id]
            deps = incoming.get(e.canonical_id, [])
            dep_ids = "|".join(str(canon_to_cid[src]) for src, _ in deps if src in canon_to_cid)
            rel_types = {rt for _, rt in deps}
            rel = next(iter(rel_types)) if len(rel_types) == 1 else default_relation
            tax = entity_to_tax.get(e.canonical_id, "UNASSIGNED")
            chunks = "|".join(e.source_chunk_ids)
            w.writerow([cid, e.label, dep_ids, tax, rel, chunks])

    logger.info("emit_csv: wrote %d rows to %s", len(entities_sorted), output_path)
