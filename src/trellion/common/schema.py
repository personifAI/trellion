# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""Canonical data models that travel between build-stage modules.

Everything is pydantic so we can ser/de to JSONL checkpoints between stages.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    token_count: int
    offset: int  # char offset in source doc


class ExtractedEntity(BaseModel):
    name: str
    entity_type: str
    description: str
    source_chunk_id: str


class ExtractedRelation(BaseModel):
    source: str
    target: str
    relation_type: str
    description: str
    strength: float = 1.0
    source_chunk_id: str


class CanonicalEntity(BaseModel):
    canonical_id: str
    label: str
    entity_type: str
    descriptions: list[str] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)
    # Union of every source chunk from every mention that merged into this
    # canonical entity. Lets downstream tooling cite exactly where a node came
    # from without re-reading the raw entity stream.
    source_chunk_ids: list[str] = Field(default_factory=list)
    # Filled in after canonicalization.
    embedding_preview: list[float] | None = None


class CanonicalEdge(BaseModel):
    source_id: str
    target_id: str
    relation_type: str
    weight: float
    evidence_chunks: list[str] = Field(default_factory=list)


class CommunityAssignment(BaseModel):
    entity_id: str
    level: int
    cluster_id: int
    parent_cluster: int | None


class CKGRow(BaseModel):
    """One row of the 6-column CSV we emit.

    The 6th column `SourceChunks` is our extension to vanilla CKG — a
    pipe-delimited list of the chunk ids this concept was extracted from.
    Readers that only expect the 5 original columns will ignore it.
    """

    concept_id: int
    concept_label: str
    dependencies: list[int] = Field(default_factory=list)
    taxonomy_id: str
    relation_type: str = "PREREQ"
    source_chunks: list[str] = Field(default_factory=list)
