# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""Trellion ORM models — memory-domain tables.

Extracted from ``noomesh/db/models.py`` in Phase 4 of proposal 06.
Owns 19 tables across conversation memory, PageIndex KB, knowledge graph,
bi-temporal graph, and the semantic relationship index.

Shared schema conventions (matching pre-extraction state):
- Vector columns use ``pgvector.sqlalchemy.Vector``.
- Index definitions include PostgreSQL-specific options (HNSW, ivfflat, GIN).
- Tables use explicit ``__tablename__`` matching existing schema names so
  Alembic can detect and migrate existing databases without renames.

Cross-domain notes:
- ``noo_temporal_edges.source_cycle_id`` is a nullable reference to a table
  owned by an add-on. Any FK is added by that owner's migration via raw SQL —
  not an ORM relationship — so the two Bases can stay independent.
- ``noo_temporal_edges.source_entry_id`` and ``source_page_id`` reference
  in-domain tables (``noo_entries``, ``pages``) but were never declared
  as formal FKs (kept loose for ingestion flexibility).
"""

from __future__ import annotations

import os
import uuid as _uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Text,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, relationship

try:
    from pgvector.sqlalchemy import Vector
except ImportError:  # pragma: no cover
    Vector = None  # type: ignore[assignment,misc]


# Read EMBEDDING_DIM at import time. ``ServicesConfig.embedding_dim`` (when
# noomesh's config is present) reads the same env var. Changing this
# requires re-creating the vector columns + re-embedding existing rows.
_EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "768"))


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _genuuid() -> _uuid.UUID:
    return _uuid.uuid4()


class Base(DeclarativeBase):
    """Trellion-owned declarative base.

    Independent of noomesh's Base. Both can coexist on the same database
    with separate Alembic version tables.
    """
    pass


# ======================================================================
# Conversation Memory (source: trellion.live.conversation_store)
# ======================================================================

class PaiConversation(Base):
    __tablename__ = "noo_conversations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=_genuuid)
    user_id = Column(Text)
    org_id = Column(Text, nullable=True)  # PR5b org/tenant boundary; NULL = 'default'
    session_id = Column(Text)
    persona_id = Column(Text, nullable=True, index=True)
    backend = Column(Text)
    model = Column(Text)
    started_at = Column(DateTime(timezone=True), nullable=False, server_default=text("NOW()"))
    ended_at = Column(DateTime(timezone=True))
    total_messages = Column(Integer, default=0, server_default=text("0"))
    total_tool_calls = Column(Integer, default=0, server_default=text("0"))
    total_tokens_in = Column(Integer, default=0, server_default=text("0"))
    total_tokens_out = Column(Integer, default=0, server_default=text("0"))
    total_latency_ms = Column(Integer, default=0, server_default=text("0"))
    metadata_ = Column("metadata", JSONB, server_default=text("'{}'::jsonb"))

    entries = relationship("PaiEntry", back_populates="conversation", cascade="all, delete-orphan")
    tool_executions = relationship("PaiToolExecution", back_populates="conversation", cascade="all, delete-orphan")
    context_retrievals = relationship("PaiContextRetrieval", back_populates="conversation", cascade="all, delete-orphan")
    feedbacks = relationship("PaiConversationFeedback", back_populates="conversation", cascade="all, delete-orphan")


Index("noo_conv_user_idx", PaiConversation.user_id, PaiConversation.started_at.desc())
Index("noo_conv_persona_idx", PaiConversation.persona_id, PaiConversation.started_at.desc())


class PaiEntry(Base):
    __tablename__ = "noo_entries"

    id = Column(UUID(as_uuid=True), primary_key=True, default=_genuuid)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("noo_conversations.id"), nullable=False)
    parent_id = Column(UUID(as_uuid=True))
    type = Column(Text, nullable=False)
    content = Column(Text, nullable=False)
    metadata_ = Column("metadata", JSONB, server_default=text("'{}'::jsonb"))
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=text("NOW()"))
    embedding = Column(Vector(_EMBEDDING_DIM)) if Vector else Column(Text)
    indexed_at = Column(DateTime(timezone=True))

    conversation = relationship("PaiConversation", back_populates="entries")


Index("noo_entries_conv_idx", PaiEntry.conversation_id, PaiEntry.created_at)
Index("noo_entries_type_idx", PaiEntry.type)
Index(
    "noo_entries_embedding_idx",
    PaiEntry.embedding,
    postgresql_using="ivfflat",
    postgresql_with={"lists": 100},
    postgresql_ops={"embedding": "vector_cosine_ops"},
    postgresql_where=PaiEntry.embedding.isnot(None),
)
Index(
    "noo_entries_content_trgm_idx",
    PaiEntry.content,
    postgresql_using="gin",
    postgresql_ops={"content": "gin_trgm_ops"},
)
Index(
    "noo_entries_unembedded_idx",
    PaiEntry.created_at,
    postgresql_where=PaiEntry.embedding.is_(None),
)


class PaiToolExecution(Base):
    __tablename__ = "noo_tool_executions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=_genuuid)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("noo_conversations.id"), nullable=False)
    entry_id = Column(UUID(as_uuid=True), ForeignKey("noo_entries.id"))
    tool_name = Column(Text, nullable=False)
    arguments = Column(JSONB, nullable=False)
    success = Column(Boolean)
    output = Column(Text)
    error = Column(Text)
    execution_ms = Column(Integer)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=text("NOW()"))

    conversation = relationship("PaiConversation", back_populates="tool_executions")


Index("noo_tool_exec_conv_idx", PaiToolExecution.conversation_id)
Index("noo_tool_exec_name_idx", PaiToolExecution.tool_name, PaiToolExecution.success)


class PaiContextRetrieval(Base):
    __tablename__ = "noo_context_retrievals"

    id = Column(UUID(as_uuid=True), primary_key=True, default=_genuuid)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("noo_conversations.id"), nullable=False)
    entry_id = Column(UUID(as_uuid=True), ForeignKey("noo_entries.id"))
    query = Column(Text, nullable=False)
    scope = Column(Text, nullable=False)
    results = Column(JSONB, nullable=False)
    top_score = Column(Float)
    retrieval_ms = Column(Integer)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=text("NOW()"))

    conversation = relationship("PaiConversation", back_populates="context_retrievals")


class PaiConversationFeedback(Base):
    __tablename__ = "noo_feedback"

    id = Column(UUID(as_uuid=True), primary_key=True, default=_genuuid)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("noo_conversations.id"))
    entry_id = Column(UUID(as_uuid=True), ForeignKey("noo_entries.id"))
    feedback_type = Column(Text, nullable=False)
    value = Column(JSONB)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=text("NOW()"))

    conversation = relationship("PaiConversation", back_populates="feedbacks")


# ======================================================================
# PageIndex / Knowledge Graph (source: trellion.live.pageindex_store)
# ======================================================================

class Doc(Base):
    __tablename__ = "docs"

    doc_id = Column(UUID(as_uuid=True), primary_key=True, default=_genuuid)
    source_uri = Column(Text)
    title = Column(Text)
    hash = Column(Text, unique=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=text("NOW()"))
    # Structured doc-level metadata (folder path, drive id, tags, etc).
    # Python attr is metadata_ because SQLAlchemy reserves `metadata` on Base.
    # See migration 004_doc_metadata + memory project_retrieval_narrowers_meta_layer.
    metadata_ = Column("metadata", JSONB, nullable=True, server_default=text("'{}'::jsonb"))
    # Document hold (migration 105): held = kept, but out of embedding and search.
    held_at = Column(DateTime(timezone=True), nullable=True)
    hold_reason = Column(Text, nullable=True)

    pages = relationship("Page", back_populates="doc", cascade="all, delete-orphan")


class Page(Base):
    __tablename__ = "pages"

    page_id = Column(UUID(as_uuid=True), primary_key=True, default=_genuuid)
    doc_id = Column(UUID(as_uuid=True), ForeignKey("docs.doc_id", ondelete="CASCADE"), nullable=False)
    page_no = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)
    meta_json = Column(JSONB)
    # Future-compatibility for cold-page tier-down (migration 006). Populated
    # by retrievers when they touch a page; NULL = never observed.
    last_accessed_at = Column(DateTime(timezone=True), nullable=True)
    # Entity extraction bookkeeping (migration 103). `extracted_at` is set
    # once extraction COMPLETED — including with nothing found — so the
    # extractor selects on it rather than inferring "not done" from the
    # absence of mentions, which re-ran empty pages forever.
    # `extraction_claimed_at` is the atomic claim that keeps two ML
    # containers off the same page; stale after fifteen minutes.
    extracted_at = Column(DateTime(timezone=True), nullable=True)
    extraction_claimed_at = Column(DateTime(timezone=True), nullable=True)

    doc = relationship("Doc", back_populates="pages")
    chunks = relationship("PageChunk", back_populates="page", cascade="all, delete-orphan")
    embedding_rel = relationship("PageEmbedding", back_populates="page", uselist=False, cascade="all, delete-orphan")
    mentions = relationship("Mention", back_populates="page", cascade="all, delete-orphan")


Index("pages_doc_page_idx", Page.doc_id, Page.page_no, unique=True)


class PageChunk(Base):
    __tablename__ = "page_chunks"

    chunk_id = Column(UUID(as_uuid=True), primary_key=True, default=_genuuid)
    page_id = Column(UUID(as_uuid=True), ForeignKey("pages.page_id", ondelete="CASCADE"), nullable=False)
    chunk_no = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)

    page = relationship("Page", back_populates="chunks")
    embedding_rel = relationship("ChunkEmbedding", back_populates="chunk", uselist=False, cascade="all, delete-orphan")


Index("page_chunks_page_idx", PageChunk.page_id)


class PageEmbedding(Base):
    __tablename__ = "page_embeddings"

    page_id = Column(
        UUID(as_uuid=True),
        ForeignKey("pages.page_id", ondelete="CASCADE"),
        primary_key=True,
    )
    embedding = Column(Vector(_EMBEDDING_DIM)) if Vector else Column(Text)

    page = relationship("Page", back_populates="embedding_rel")


Index(
    "page_embeddings_hnsw_idx",
    PageEmbedding.embedding,
    postgresql_using="hnsw",
    postgresql_ops={"embedding": "vector_cosine_ops"},
)


class ChunkEmbedding(Base):
    __tablename__ = "chunk_embeddings"

    chunk_id = Column(
        UUID(as_uuid=True),
        ForeignKey("page_chunks.chunk_id", ondelete="CASCADE"),
        primary_key=True,
    )
    embedding = Column(Vector(_EMBEDDING_DIM)) if Vector else Column(Text)

    chunk = relationship("PageChunk", back_populates="embedding_rel")


Index(
    "chunk_embeddings_hnsw_idx",
    ChunkEmbedding.embedding,
    postgresql_using="hnsw",
    postgresql_ops={"embedding": "vector_cosine_ops"},
)


# Knowledge graph

class Entity(Base):
    __tablename__ = "entities"

    entity_id = Column(UUID(as_uuid=True), primary_key=True, default=_genuuid)
    canonical_name = Column(Text, nullable=False)
    type = Column(Text, nullable=False)
    aliases_json = Column(JSONB)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=text("NOW()"))
    # Multi-user option-value column (PR3b-lite). NULL = shared/global; not
    # written or filtered yet — see docs MULTI_USER_READINESS.md "isolation flip".
    user_id = Column(Text, nullable=True)
    org_id = Column(Text, nullable=True)  # PR5b org/tenant boundary; NULL = 'default'


Index("entities_name_type_idx", Entity.canonical_name, Entity.type, unique=True)
Index("ix_entities_user_id", Entity.user_id, postgresql_where=Entity.user_id.isnot(None))


class EntityGraphSignal(Base):
    """Per-entity graph metrics computed by the graph-metrics provider (an optional add-on).

    Previously created at runtime by a ``CREATE TABLE IF NOT EXISTS`` inside
    the metrics writer, which left it outside both Alembic environments — it
    existed in production owned by no migration. Declared here so the memory
    schema has a single declarative source of truth.

    Column types mirror the original runtime DDL exactly (verified against the
    live table); this model must not drift from it without a migration.
    """

    __tablename__ = "entity_graph_signals"

    entity_id = Column(
        UUID(as_uuid=True),
        ForeignKey("entities.entity_id", ondelete="CASCADE"),
        primary_key=True,
    )
    pagerank = Column(Float, server_default=text("0.0"))
    betweenness = Column(Float, server_default=text("0.0"))
    clustering_coeff = Column(Float, server_default=text("0.0"))
    in_degree = Column(Integer, server_default=text("0"))
    out_degree = Column(Integer, server_default=text("0"))
    triangle_count = Column(Integer, server_default=text("0"))
    motif_type = Column(Text, server_default=text("'unknown'"))
    spectral_embedding = Column(JSONB)
    propagated_type = Column(Text)
    propagated_type_confidence = Column(Float, server_default=text("0.0"))
    signal_generation = Column(Integer, server_default=text("0"))
    computed_at = Column(DateTime(timezone=True), server_default=text("NOW()"))


class Mention(Base):
    __tablename__ = "mentions"

    mention_id = Column(UUID(as_uuid=True), primary_key=True, default=_genuuid)
    entity_id = Column(UUID(as_uuid=True), ForeignKey("entities.entity_id", ondelete="CASCADE"), nullable=False)
    page_id = Column(UUID(as_uuid=True), ForeignKey("pages.page_id", ondelete="CASCADE"), nullable=False)
    span_json = Column(JSONB)
    quote = Column(Text)

    page = relationship("Page", back_populates="mentions")


Index("mentions_entity_idx", Mention.entity_id)
Index("mentions_page_idx", Mention.page_id)


class Edge(Base):
    __tablename__ = "edges"

    edge_id = Column(UUID(as_uuid=True), primary_key=True, default=_genuuid)
    src_entity_id = Column(UUID(as_uuid=True), ForeignKey("entities.entity_id", ondelete="CASCADE"), nullable=False)
    rel_type = Column(Text, nullable=False)
    dst_entity_id = Column(UUID(as_uuid=True), ForeignKey("entities.entity_id", ondelete="CASCADE"), nullable=False)
    evidence_json = Column(JSONB)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=text("NOW()"))
    confidence = Column(Float, server_default=text("1.0"))
    classified_at = Column(DateTime(timezone=True))
    classifier_model = Column(Text)
    # Multi-user option-value column (PR3b-lite). NULL = shared/global.
    user_id = Column(Text, nullable=True)
    org_id = Column(Text, nullable=True)  # PR5b org/tenant boundary; NULL = 'default'


Index("edges_src_rel_idx", Edge.src_entity_id, Edge.rel_type)
Index("ix_edges_user_id", Edge.user_id, postgresql_where=Edge.user_id.isnot(None))
Index("edges_dst_rel_idx", Edge.dst_entity_id, Edge.rel_type)
Index(
    "idx_edges_unclassified",
    Edge.classified_at,
    postgresql_where=Edge.classified_at.is_(None),
)


# ======================================================================
# Temporal Knowledge Graph — bi-temporal edges with provenance
# ======================================================================

class PaiEntityTyped(Base):
    __tablename__ = "noo_entities_typed"

    id = Column(UUID(as_uuid=True), primary_key=True, default=_genuuid)
    type = Column(Text, nullable=False)
    canonical_name = Column(Text, nullable=False)
    extra = Column("metadata", JSONB, server_default=text("'{}'::jsonb"))
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=text("NOW()"))
    # Multi-user option-value column (PR3b-lite). NULL = shared/global. Personal
    # profile facts are the likely first store to flip to per-user isolation.
    user_id = Column(Text, nullable=True)
    org_id = Column(Text, nullable=True)  # PR5b org/tenant boundary; NULL = 'default'

    __table_args__ = (
        Index("uq_entities_typed_type_name", "type", "canonical_name", unique=True),
    )


Index("idx_entities_typed_type", PaiEntityTyped.type)
Index("ix_entities_typed_user_id", PaiEntityTyped.user_id, postgresql_where=PaiEntityTyped.user_id.isnot(None))
Index("idx_entities_typed_name", PaiEntityTyped.canonical_name)


class PaiTemporalEdge(Base):
    __tablename__ = "noo_temporal_edges"

    id = Column(UUID(as_uuid=True), primary_key=True, default=_genuuid)
    subject_id = Column(
        UUID(as_uuid=True),
        ForeignKey("noo_entities_typed.id", ondelete="CASCADE"),
        nullable=False,
    )
    predicate = Column(Text, nullable=False)
    object_entity_id = Column(
        UUID(as_uuid=True),
        ForeignKey("noo_entities_typed.id", ondelete="SET NULL"),
        nullable=True,
    )
    object_value = Column(Text, nullable=True)
    valid_from = Column(DateTime(timezone=True), nullable=False, server_default=text("NOW()"))
    valid_to = Column(DateTime(timezone=True), nullable=True)
    observed_at = Column(DateTime(timezone=True), nullable=False, server_default=text("NOW()"))
    confidence = Column(Float, nullable=False, server_default=text("1.0"))
    source_entry_id = Column(UUID(as_uuid=True), nullable=True)  # FK target: noo_entries.id (loose)
    source_page_id = Column(UUID(as_uuid=True), nullable=True)   # FK target: pages.page_id (loose)
    source_cycle_id = Column(UUID(as_uuid=True), nullable=True)  # nullable reference to an add-on table; no FK declared here
    extra = Column("metadata", JSONB, server_default=text("'{}'::jsonb"))
    # Multi-user option-value column (PR3b-lite). NULL = shared/global.
    user_id = Column(Text, nullable=True)
    org_id = Column(Text, nullable=True)  # PR5b org/tenant boundary; NULL = 'default'


Index("ix_temporal_edges_user_id", PaiTemporalEdge.user_id, postgresql_where=PaiTemporalEdge.user_id.isnot(None))
Index(
    "idx_temporal_edges_current",
    PaiTemporalEdge.subject_id,
    PaiTemporalEdge.predicate,
    postgresql_where=PaiTemporalEdge.valid_to.is_(None),
)
Index(
    "idx_temporal_edges_subject",
    PaiTemporalEdge.subject_id,
    PaiTemporalEdge.valid_from.desc(),
)
Index(
    "idx_temporal_edges_observed",
    PaiTemporalEdge.observed_at.desc(),
)
Index(
    "idx_temporal_edges_entry",
    PaiTemporalEdge.source_entry_id,
    postgresql_where=PaiTemporalEdge.source_entry_id.isnot(None),
)
Index(
    "idx_temporal_edges_object",
    PaiTemporalEdge.object_entity_id,
    postgresql_where=PaiTemporalEdge.object_entity_id.isnot(None),
)


# ======================================================================
# Graph-aware entity embeddings + clusters
# ======================================================================

class EntityEmbedding(Base):
    __tablename__ = "entity_embeddings"

    entity_id = Column(
        UUID(as_uuid=True),
        ForeignKey("entities.entity_id", ondelete="CASCADE"),
        primary_key=True,
    )
    embedding = Column(Vector(_EMBEDDING_DIM), nullable=False) if Vector else Column(Text, nullable=False)
    computed_at = Column(DateTime(timezone=True), nullable=False, server_default=text("NOW()"))
    generation = Column(Integer, nullable=False, server_default=text("0"))


Index(
    "entity_embeddings_hnsw_idx",
    EntityEmbedding.embedding,
    postgresql_using="hnsw",
    postgresql_ops={"embedding": "vector_cosine_ops"},
)


class EntityCluster(Base):
    __tablename__ = "entity_clusters"

    cluster_id = Column(UUID(as_uuid=True), primary_key=True, default=_genuuid)
    name = Column(Text, nullable=False)
    level = Column(Integer, nullable=False, server_default=text("0"))
    parent_cluster_id = Column(
        UUID(as_uuid=True),
        ForeignKey("entity_clusters.cluster_id"),
        nullable=True,
    )
    summary = Column(Text, nullable=True)
    centroid = Column(Vector(_EMBEDDING_DIM)) if Vector else Column(Text)
    member_count = Column(Integer, nullable=False, server_default=text("0"))
    computed_at = Column(DateTime(timezone=True), nullable=False, server_default=text("NOW()"))
    generation = Column(Integer, nullable=False, server_default=text("0"))

    parent = relationship("EntityCluster", remote_side=[cluster_id], backref="children")
    members = relationship("EntityClusterMember", back_populates="cluster", cascade="all, delete-orphan")


Index(
    "entity_clusters_centroid_hnsw_idx",
    EntityCluster.centroid,
    postgresql_using="hnsw",
    postgresql_ops={"centroid": "vector_cosine_ops"},
)
Index("idx_entity_clusters_level", EntityCluster.level)
Index("idx_entity_clusters_parent", EntityCluster.parent_cluster_id)


class EntityClusterMember(Base):
    __tablename__ = "entity_cluster_members"

    entity_id = Column(
        UUID(as_uuid=True),
        ForeignKey("entities.entity_id", ondelete="CASCADE"),
        primary_key=True,
    )
    cluster_id = Column(
        UUID(as_uuid=True),
        ForeignKey("entity_clusters.cluster_id", ondelete="CASCADE"),
        primary_key=True,
    )
    membership_score = Column(Float, nullable=False, server_default=text("1.0"))

    cluster = relationship("EntityCluster", back_populates="members")


Index("idx_entity_cluster_members_cluster", EntityClusterMember.cluster_id)


# ======================================================================
# Embedding worker — semantic relationships
# ======================================================================

class PaiSemanticRelationship(Base):
    __tablename__ = "noo_semantic_relationships"

    id = Column(Integer, primary_key=True, autoincrement=True)
    source_id = Column(UUID(as_uuid=True), nullable=False)
    target_id = Column(UUID(as_uuid=True), nullable=False)
    similarity = Column(Float, nullable=False)
    relationship_type = Column(Text, default="semantic")
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=text("NOW()"))

    __table_args__ = (
        Index("noo_sem_rel_source_idx", "source_id"),
        Index("noo_sem_rel_target_idx", "target_id"),
        Index(
            "noo_sem_rel_unique",
            "source_id", "target_id", "relationship_type",
            unique=True,
        ),
    )


# ======================================================================
# Snapshot consolidation — System 1.5 layer (proposal 06 phase 6)
#
# A consolidation cycle takes a slice of the live bi-temporal graph
# (topic-scoped, time-window, or user-attention) and runs it through
# trellion's existing build pipeline (perturbation-consensus
# canonicalization → Leiden clustering → Eades-Lin-Smyth DAG projection)
# to produce a deterministic, instant-to-query, zero-LLM snapshot.
#
# Snapshots are immutable + dated → diffable. Old snapshots survive new
# builds, so the trellion query router can compare crystallized state
# across consolidation cycles.
# ======================================================================

class TrellionSnapshot(Base):
    """One consolidation cycle's output — immutable and dated."""
    __tablename__ = "trellion_snapshots"

    id = Column(UUID(as_uuid=True), primary_key=True, default=_genuuid)
    generation = Column(Integer, nullable=False)  # monotonically increasing per slice key
    slice_kind = Column(Text, nullable=False)  # 'topic' | 'time_window' | 'user_attention'
    slice_spec = Column(JSONB, nullable=False, server_default=text("'{}'::jsonb"))
    status = Column(Text, nullable=False, server_default=text("'building'"))  # building | complete | failed
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=text("NOW()"))
    completed_at = Column(DateTime(timezone=True), nullable=True)
    stats = Column(JSONB, nullable=False, server_default=text("'{}'::jsonb"))
    # Optional human-readable label for the cycle (e.g., "topic:gemma4 gen=3")
    label = Column(Text, nullable=True)


Index("idx_trellion_snapshots_kind_gen", TrellionSnapshot.slice_kind, TrellionSnapshot.generation.desc())
Index("idx_trellion_snapshots_created", TrellionSnapshot.created_at.desc())
Index("idx_trellion_snapshots_status", TrellionSnapshot.status)


class TrellionConcept(Base):
    """Canonical entity within a snapshot (post-perturbation-consensus dedup)."""
    __tablename__ = "trellion_concepts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=_genuuid)
    snapshot_id = Column(UUID(as_uuid=True), ForeignKey("trellion_snapshots.id", ondelete="CASCADE"), nullable=False)
    canonical_name = Column(Text, nullable=False)
    type = Column(Text, nullable=False)
    aliases = Column(JSONB, nullable=False, server_default=text("'[]'::jsonb"))
    # Optional centroid embedding for hybrid concept matching at query time
    centroid = Column(Vector(_EMBEDDING_DIM)) if Vector else Column(Text)
    extra = Column("metadata", JSONB, server_default=text("'{}'::jsonb"))


Index("idx_trellion_concepts_snapshot", TrellionConcept.snapshot_id)
Index(
    "uq_trellion_concepts_snapshot_name_type",
    TrellionConcept.snapshot_id, TrellionConcept.canonical_name, TrellionConcept.type,
    unique=True,
)


class TrellionDagEdge(Base):
    """DAG-projected edge within a snapshot (post-Eades-Lin-Smyth feedback arc set removal)."""
    __tablename__ = "trellion_dag_edges"

    id = Column(UUID(as_uuid=True), primary_key=True, default=_genuuid)
    snapshot_id = Column(UUID(as_uuid=True), ForeignKey("trellion_snapshots.id", ondelete="CASCADE"), nullable=False)
    src_concept_id = Column(UUID(as_uuid=True), ForeignKey("trellion_concepts.id", ondelete="CASCADE"), nullable=False)
    dst_concept_id = Column(UUID(as_uuid=True), ForeignKey("trellion_concepts.id", ondelete="CASCADE"), nullable=False)
    rel_type = Column(Text, nullable=False)
    weight = Column(Float, nullable=False, server_default=text("1.0"))
    extra = Column("metadata", JSONB, server_default=text("'{}'::jsonb"))


Index("idx_trellion_dag_edges_snapshot", TrellionDagEdge.snapshot_id)
Index("idx_trellion_dag_edges_src", TrellionDagEdge.snapshot_id, TrellionDagEdge.src_concept_id)
Index("idx_trellion_dag_edges_dst", TrellionDagEdge.snapshot_id, TrellionDagEdge.dst_concept_id)


class TrellionCommunity(Base):
    """Hierarchical Leiden community within a snapshot."""
    __tablename__ = "trellion_communities"

    id = Column(UUID(as_uuid=True), primary_key=True, default=_genuuid)
    snapshot_id = Column(UUID(as_uuid=True), ForeignKey("trellion_snapshots.id", ondelete="CASCADE"), nullable=False)
    level = Column(Integer, nullable=False, server_default=text("0"))
    parent_community_id = Column(UUID(as_uuid=True), ForeignKey("trellion_communities.id"), nullable=True)
    name = Column(Text, nullable=False)
    summary = Column(Text, nullable=True)
    member_concept_ids = Column(JSONB, nullable=False, server_default=text("'[]'::jsonb"))
    member_count = Column(Integer, nullable=False, server_default=text("0"))


Index("idx_trellion_communities_snapshot_level", TrellionCommunity.snapshot_id, TrellionCommunity.level)


class TrellionProvenance(Base):
    """Links snapshot concepts back to their live-KG sources.

    No formal FK on source_entity_id — the live KG can supersede entries
    while a snapshot remains immutable. We store the UUID for traceability
    only; readers must tolerate dangling references.
    """
    __tablename__ = "trellion_provenance"

    id = Column(UUID(as_uuid=True), primary_key=True, default=_genuuid)
    snapshot_id = Column(UUID(as_uuid=True), ForeignKey("trellion_snapshots.id", ondelete="CASCADE"), nullable=False)
    concept_id = Column(UUID(as_uuid=True), ForeignKey("trellion_concepts.id", ondelete="CASCADE"), nullable=False)
    # source_kind: 'temporal_edge' | 'page' | 'entry' | 'doc'
    source_kind = Column(Text, nullable=False)
    source_id = Column(UUID(as_uuid=True), nullable=False)  # loose FK — see docstring


Index("idx_trellion_provenance_snapshot", TrellionProvenance.snapshot_id)
Index("idx_trellion_provenance_concept", TrellionProvenance.concept_id)


# ======================================================================
# Video memory v2 — System 1 fast recall (proposal 06 phase 6)
#
# Replaces the old MP4+QR design with the persona service pattern: structured
# data in Postgres, only the FAISS index on disk. Prominence-based
# three-tier lifecycle (S1 ≥0.3 / S1.5 elevated 0.05–0.3 / pruned <0.05
# + >90d).
# ======================================================================

class TrellionEpisode(Base):
    """Lifecycle metadata for a System 1 record (conversation or traversal).

    Replaces the per-MP4-frame metadata. The actual content lives in
    noo_entries (for conversations) or trellion_traversal_records (for
    proven graph walks); this row tracks prominence + access stats.
    """
    __tablename__ = "trellion_episodes"

    id = Column(UUID(as_uuid=True), primary_key=True, default=_genuuid)
    persona_id = Column(Text, nullable=False)
    record_type = Column(Text, nullable=False)  # 'conversation' | 'traversal'
    # Logical index/collection ('conversations' | 'traversals') — was the
    # FAISS file name before proposal 28 P1 moved vectors into Postgres.
    index_name = Column(Text, nullable=False)
    content_summary = Column(Text, nullable=False)
    prominence_score = Column(Float, nullable=False, server_default=text("0.5"))
    access_count = Column(Integer, nullable=False, server_default=text("0"))
    first_accessed_at = Column(DateTime(timezone=True), nullable=True)
    last_accessed_at = Column(DateTime(timezone=True), nullable=True)
    avg_search_score = Column(Float, nullable=True)
    is_tombstoned = Column(Boolean, nullable=False, server_default=text("false"))
    extra = Column("metadata", JSONB, nullable=False, server_default=text("'{}'::jsonb"))
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=text("NOW()"))
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=text("NOW()"))
    # pgvector search column (migration 101); dim matches EMBEDDING_DIM.
    embedding = Column(Vector(_EMBEDDING_DIM)) if Vector else Column(Text)
    # Owner of the episode (migration 104): the asker whose question text and
    # embedding this row carries. NULL = unattributed. Erasure keys on it.
    user_id = Column(Text, nullable=True)


Index("idx_trellion_video_frames_persona", TrellionEpisode.persona_id, TrellionEpisode.is_tombstoned)
TrellionVideoFrame = TrellionEpisode  # compat alias, dies at P2

Index(
    "idx_trellion_video_frames_lifecycle",
    TrellionEpisode.prominence_score,
    TrellionEpisode.created_at,
    postgresql_where=text("is_tombstoned = false"),
)


class TrellionTraversalRecord(Base):
    """Full traversal episode — captured when an outcome score is high enough."""
    __tablename__ = "trellion_traversal_records"

    id = Column(UUID(as_uuid=True), primary_key=True, default=_genuuid)
    persona_id = Column(Text, nullable=False)
    query_text = Column(Text, nullable=False)
    query_embedding = Column(Vector(_EMBEDDING_DIM)) if Vector else Column(Text)
    intent = Column(Text, nullable=True)
    traversal_path = Column(JSONB, nullable=False, server_default=text("'[]'::jsonb"))
    outcome_score = Column(Float, nullable=False)
    exploration_rate = Column(Float, nullable=True)
    uncertainty_fingerprint = Column(JSONB, nullable=False, server_default=text("'{}'::jsonb"))
    archived_to_video = Column(Boolean, nullable=False, server_default=text("false"))
    video_frame_id = Column(UUID(as_uuid=True), ForeignKey("trellion_episodes.id", ondelete="SET NULL"), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=text("NOW()"))
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=text("NOW()"))
    # Multi-user option-value column (PR3b-lite). NULL = shared/global. Video
    # replays are persona-scoped today; user_id enables per-user isolation later.
    user_id = Column(Text, nullable=True)
    org_id = Column(Text, nullable=True)  # PR5b org/tenant boundary; NULL = 'default'
    # Recallable walks (migration 104, noomesh proposal 57).
    kind = Column(Text, nullable=False, server_default=text("'simulated'"))  # thread | simulated
    thread_id = Column(UUID(as_uuid=True), nullable=True)
    turn_index = Column(Integer, nullable=True)
    run_id = Column(Text, nullable=True)
    provenance = Column(JSONB, nullable=False, server_default=text("'{}'::jsonb"))
    evidence = Column(JSONB, nullable=False, server_default=text("'{}'::jsonb"))
    usefulness = Column(Float, nullable=True)
    status = Column(Text, nullable=False, server_default=text("'active'"))  # active | stale | staged
    last_validated_at = Column(DateTime(timezone=True), nullable=True)
    last_recalled_at = Column(DateTime(timezone=True), nullable=True)
    recall_count = Column(Integer, nullable=False, server_default=text("0"))
    expires_at = Column(DateTime(timezone=True), nullable=True)


Index("idx_trellion_traversal_persona", TrellionTraversalRecord.persona_id, TrellionTraversalRecord.created_at.desc())
Index("ix_trellion_traversal_user_id", TrellionTraversalRecord.user_id, postgresql_where=TrellionTraversalRecord.user_id.isnot(None))
Index("ix_trellion_traversal_user_kind_status", TrellionTraversalRecord.user_id, TrellionTraversalRecord.kind, TrellionTraversalRecord.status)
Index("ix_trellion_traversal_thread", TrellionTraversalRecord.thread_id, postgresql_where=TrellionTraversalRecord.thread_id.isnot(None))
Index("idx_trellion_traversal_outcome", TrellionTraversalRecord.outcome_score.desc())
