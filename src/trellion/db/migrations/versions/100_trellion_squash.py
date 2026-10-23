# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""The squashed trellion baseline — twelve revisions become one.

Revision ID: 100_trellion_squash
Revises:
Create Date: 2026-08-05

Trellion segment of the per-system baseline. This replaces the 001–012 chain
with ONE frozen revision: one fresh file per system to build forward from.

FROZEN DDL, taken from a pg_dump of a database built by the FULL old chain —
not from the models. The models do not know everything the chain built: the
HNSW index parameters (012), the partial indexes, the `noo_conv_set_org`
trigger and its function (009). Generating from metadata would have silently
dropped all of them. Dump inventory at squash time: 27 tables, 72 indexes,
60 ALTERs, 1 function, 1 trigger.

UNQUALIFIED on purpose. The env sets `search_path = <schema>, public` before
running, so on a fresh install everything lands in schema `trellion`, created
by `noo_trellion_ddl` and owned correctly at birth. Legacy installs keep
their tables in `public`, adopted in place — they move only at the
export/rebuild/re-import cutover, never in situ.

NO CREATE EXTENSION here. `vector` and `pg_trgm` are untrusted, so creating
them needs superuser regardless of who owns the schema — in-cluster the
BROKER installs them (declared in the `systems:` manifest), and the env
verifies they exist before this runs. Keeping them out of the baseline is
exactly what lets this chain run as an unprivileged DDL role.

ONE CROSS-DOMAIN READ SURVIVES, recorded rather than hidden: the
`noo_conv_set_org` trigger function reads `noo_users` and `noo_kb_teams` —
gateway tables. Unqualified, so it resolves through the CALLER's search_path,
which always includes `public`; correct in both worlds, but it means
trellion's writes consult the gateway's tables at INSERT time. Untangling that is
conversation-identity work (proposal 17), not schema work.

THE OLD VERSION TABLE. Legacy installs carry `public.alembic_version_memory`
stamped at 012_noo_entries_hnsw. This chain uses a NEW version table in the
trellion schema, so that row simply stops being read — left in place, dropped
at cutover. The adopt path below stamps this revision without touching it.

CREATES IF ABSENT, ADOPTS IF PRESENT — the contract of every
system baseline, and what makes the deployment order forgiving: a container running this
against an already-migrated database stamps and stops.

THE CONVERSATION TABLES ARE SHARED. `noo_conversations` and its four
children are the host's chat record as much as trellion's memory, and a host
(noomesh) may build them before trellion ever runs. They are a separate blob,
created only when no schema has them yet; trellion still builds them itself
when it runs standalone.
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "100_trellion_squash"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# One table is enough to detect a legacy install (the old chain built all 27).
# It must be one only trellion creates. `noo_conversations` used to be the
# sentinel, but the conversation tables may now come from the host (below),
# and a host-built `noo_conversations` made this revision stamp an install
# that had none of trellion's own tables.
_SENTINEL = "trellion_snapshots"

# The conversation tables trellion shares with its host. Noomesh can own
# them (it records chats without trellion); standalone trellion still needs
# them. Created here only when no schema has `noo_conversations` yet — a
# second copy in `trellion` would shadow the host's for one side and split
# every conversation in two, silently. Looked up in EVERY schema, not just
# this search_path, for the same reason: a missing relation fails loudly,
# a duplicate one does not.
_CONVERSATION_SENTINEL = "noo_conversations"

_DDL = """\
CREATE TABLE chunk_embeddings (
    chunk_id uuid NOT NULL,
    embedding vector(1024),
    embedding_model text,
    embedding_dim integer
);
CREATE TABLE docs (
    doc_id uuid NOT NULL,
    source_uri text,
    title text,
    hash text,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    metadata jsonb DEFAULT '{}'::jsonb
);
CREATE TABLE edges (
    edge_id uuid NOT NULL,
    src_entity_id uuid NOT NULL,
    rel_type text NOT NULL,
    dst_entity_id uuid NOT NULL,
    evidence_json jsonb,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    confidence double precision DEFAULT 1.0,
    classified_at timestamp with time zone,
    classifier_model text,
    user_id text,
    org_id text
);
CREATE TABLE entities (
    entity_id uuid NOT NULL,
    canonical_name text NOT NULL,
    type text NOT NULL,
    aliases_json jsonb,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    user_id text,
    org_id text
);
CREATE TABLE entity_cluster_members (
    entity_id uuid NOT NULL,
    cluster_id uuid NOT NULL,
    membership_score double precision DEFAULT 1.0 NOT NULL
);
CREATE TABLE entity_clusters (
    cluster_id uuid NOT NULL,
    name text NOT NULL,
    level integer DEFAULT 0 NOT NULL,
    parent_cluster_id uuid,
    summary text,
    centroid vector(1024),
    member_count integer DEFAULT 0 NOT NULL,
    computed_at timestamp with time zone DEFAULT now() NOT NULL,
    generation integer DEFAULT 0 NOT NULL,
    embedding_model text,
    embedding_dim integer
);
CREATE TABLE entity_embeddings (
    entity_id uuid NOT NULL,
    embedding vector(1024) NOT NULL,
    computed_at timestamp with time zone DEFAULT now() NOT NULL,
    generation integer DEFAULT 0 NOT NULL,
    embedding_model text,
    embedding_dim integer
);
CREATE TABLE entity_graph_signals (
    entity_id uuid NOT NULL,
    pagerank double precision DEFAULT 0.0,
    betweenness double precision DEFAULT 0.0,
    clustering_coeff double precision DEFAULT 0.0,
    in_degree integer DEFAULT 0,
    out_degree integer DEFAULT 0,
    triangle_count integer DEFAULT 0,
    motif_type text DEFAULT 'unknown'::text,
    spectral_embedding jsonb,
    propagated_type text,
    propagated_type_confidence double precision DEFAULT 0.0,
    signal_generation integer DEFAULT 0,
    computed_at timestamp with time zone DEFAULT now()
);
CREATE TABLE mentions (
    mention_id uuid NOT NULL,
    entity_id uuid NOT NULL,
    page_id uuid NOT NULL,
    span_json jsonb,
    quote text
);
CREATE TABLE page_chunks (
    chunk_id uuid NOT NULL,
    page_id uuid NOT NULL,
    chunk_no integer NOT NULL,
    text text NOT NULL
);
CREATE TABLE page_embeddings (
    page_id uuid NOT NULL,
    embedding vector(1024),
    embedding_model text,
    embedding_dim integer
);
CREATE TABLE pages (
    page_id uuid NOT NULL,
    doc_id uuid NOT NULL,
    page_no integer NOT NULL,
    text text NOT NULL,
    meta_json jsonb,
    last_accessed_at timestamp with time zone
);
CREATE TABLE noo_entities_typed (
    id uuid NOT NULL,
    type text NOT NULL,
    canonical_name text NOT NULL,
    metadata jsonb DEFAULT '{}'::jsonb,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    user_id text,
    org_id text
);
CREATE TABLE noo_semantic_relationships (
    id integer NOT NULL,
    source_id uuid NOT NULL,
    target_id uuid NOT NULL,
    similarity double precision NOT NULL,
    relationship_type text,
    created_at timestamp with time zone DEFAULT now() NOT NULL
);
CREATE SEQUENCE noo_semantic_relationships_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;
ALTER SEQUENCE noo_semantic_relationships_id_seq OWNED BY noo_semantic_relationships.id;
CREATE TABLE noo_temporal_edges (
    id uuid NOT NULL,
    subject_id uuid NOT NULL,
    predicate text NOT NULL,
    object_entity_id uuid,
    object_value text,
    valid_from timestamp with time zone DEFAULT now() NOT NULL,
    valid_to timestamp with time zone,
    observed_at timestamp with time zone DEFAULT now() NOT NULL,
    confidence double precision DEFAULT 1.0 NOT NULL,
    source_entry_id uuid,
    source_page_id uuid,
    source_cycle_id uuid,
    metadata jsonb DEFAULT '{}'::jsonb,
    user_id text,
    org_id text
);
CREATE TABLE trellion_communities (
    id uuid NOT NULL,
    snapshot_id uuid NOT NULL,
    level integer DEFAULT 0 NOT NULL,
    parent_community_id uuid,
    name text NOT NULL,
    summary text,
    member_concept_ids jsonb DEFAULT '[]'::jsonb NOT NULL,
    member_count integer DEFAULT 0 NOT NULL
);
CREATE TABLE trellion_concepts (
    id uuid NOT NULL,
    snapshot_id uuid NOT NULL,
    canonical_name text NOT NULL,
    type text NOT NULL,
    aliases jsonb DEFAULT '[]'::jsonb NOT NULL,
    centroid vector(1024),
    metadata jsonb DEFAULT '{}'::jsonb,
    embedding_model text,
    embedding_dim integer
);
CREATE TABLE trellion_dag_edges (
    id uuid NOT NULL,
    snapshot_id uuid NOT NULL,
    src_concept_id uuid NOT NULL,
    dst_concept_id uuid NOT NULL,
    rel_type text NOT NULL,
    weight double precision DEFAULT 1.0 NOT NULL,
    metadata jsonb DEFAULT '{}'::jsonb
);
CREATE TABLE trellion_provenance (
    id uuid NOT NULL,
    snapshot_id uuid NOT NULL,
    concept_id uuid NOT NULL,
    source_kind text NOT NULL,
    source_id uuid NOT NULL
);
CREATE TABLE trellion_snapshots (
    id uuid NOT NULL,
    generation integer NOT NULL,
    slice_kind text NOT NULL,
    slice_spec jsonb DEFAULT '{}'::jsonb NOT NULL,
    status text DEFAULT 'building'::text NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    completed_at timestamp with time zone,
    stats jsonb DEFAULT '{}'::jsonb NOT NULL,
    label text
);
CREATE TABLE trellion_traversal_records (
    id uuid NOT NULL,
    persona_id text NOT NULL,
    query_text text NOT NULL,
    query_embedding vector(1024),
    intent text,
    traversal_path jsonb DEFAULT '[]'::jsonb NOT NULL,
    outcome_score double precision NOT NULL,
    exploration_rate double precision,
    uncertainty_fingerprint jsonb DEFAULT '{}'::jsonb NOT NULL,
    archived_to_video boolean DEFAULT false NOT NULL,
    video_frame_id uuid,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL,
    user_id text,
    org_id text,
    embedding_model text,
    embedding_dim integer
);
CREATE TABLE trellion_video_frames (
    id uuid NOT NULL,
    persona_id text NOT NULL,
    record_type text NOT NULL,
    faiss_index_name text NOT NULL,
    content_summary text NOT NULL,
    prominence_score double precision DEFAULT 0.5 NOT NULL,
    access_count integer DEFAULT 0 NOT NULL,
    first_accessed_at timestamp with time zone,
    last_accessed_at timestamp with time zone,
    avg_search_score double precision,
    is_tombstoned boolean DEFAULT false NOT NULL,
    metadata jsonb DEFAULT '{}'::jsonb NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL
);
ALTER TABLE ONLY noo_semantic_relationships ALTER COLUMN id SET DEFAULT nextval('noo_semantic_relationships_id_seq'::regclass);
ALTER TABLE ONLY chunk_embeddings
    ADD CONSTRAINT chunk_embeddings_pkey PRIMARY KEY (chunk_id);
ALTER TABLE ONLY docs
    ADD CONSTRAINT docs_hash_key UNIQUE (hash);
ALTER TABLE ONLY docs
    ADD CONSTRAINT docs_pkey PRIMARY KEY (doc_id);
ALTER TABLE ONLY edges
    ADD CONSTRAINT edges_pkey PRIMARY KEY (edge_id);
ALTER TABLE ONLY entities
    ADD CONSTRAINT entities_pkey PRIMARY KEY (entity_id);
ALTER TABLE ONLY entity_cluster_members
    ADD CONSTRAINT entity_cluster_members_pkey PRIMARY KEY (entity_id, cluster_id);
ALTER TABLE ONLY entity_clusters
    ADD CONSTRAINT entity_clusters_pkey PRIMARY KEY (cluster_id);
ALTER TABLE ONLY entity_embeddings
    ADD CONSTRAINT entity_embeddings_pkey PRIMARY KEY (entity_id);
ALTER TABLE ONLY entity_graph_signals
    ADD CONSTRAINT entity_graph_signals_pkey PRIMARY KEY (entity_id);
ALTER TABLE ONLY mentions
    ADD CONSTRAINT mentions_pkey PRIMARY KEY (mention_id);
ALTER TABLE ONLY page_chunks
    ADD CONSTRAINT page_chunks_pkey PRIMARY KEY (chunk_id);
ALTER TABLE ONLY page_embeddings
    ADD CONSTRAINT page_embeddings_pkey PRIMARY KEY (page_id);
ALTER TABLE ONLY pages
    ADD CONSTRAINT pages_pkey PRIMARY KEY (page_id);
ALTER TABLE ONLY noo_entities_typed
    ADD CONSTRAINT noo_entities_typed_pkey PRIMARY KEY (id);
ALTER TABLE ONLY noo_semantic_relationships
    ADD CONSTRAINT noo_semantic_relationships_pkey PRIMARY KEY (id);
ALTER TABLE ONLY noo_temporal_edges
    ADD CONSTRAINT noo_temporal_edges_pkey PRIMARY KEY (id);
ALTER TABLE ONLY trellion_communities
    ADD CONSTRAINT trellion_communities_pkey PRIMARY KEY (id);
ALTER TABLE ONLY trellion_concepts
    ADD CONSTRAINT trellion_concepts_pkey PRIMARY KEY (id);
ALTER TABLE ONLY trellion_dag_edges
    ADD CONSTRAINT trellion_dag_edges_pkey PRIMARY KEY (id);
ALTER TABLE ONLY trellion_provenance
    ADD CONSTRAINT trellion_provenance_pkey PRIMARY KEY (id);
ALTER TABLE ONLY trellion_snapshots
    ADD CONSTRAINT trellion_snapshots_pkey PRIMARY KEY (id);
ALTER TABLE ONLY trellion_traversal_records
    ADD CONSTRAINT trellion_traversal_records_pkey PRIMARY KEY (id);
ALTER TABLE ONLY trellion_video_frames
    ADD CONSTRAINT trellion_video_frames_pkey PRIMARY KEY (id);
CREATE INDEX chunk_embeddings_hnsw_idx ON chunk_embeddings USING hnsw (embedding vector_cosine_ops);
CREATE INDEX edges_dst_rel_idx ON edges USING btree (dst_entity_id, rel_type);
CREATE INDEX edges_src_rel_idx ON edges USING btree (src_entity_id, rel_type);
CREATE UNIQUE INDEX entities_name_type_idx ON entities USING btree (canonical_name, type);
CREATE INDEX entity_clusters_centroid_hnsw_idx ON entity_clusters USING hnsw (centroid vector_cosine_ops);
CREATE INDEX entity_embeddings_hnsw_idx ON entity_embeddings USING hnsw (embedding vector_cosine_ops);
CREATE INDEX idx_edges_unclassified ON edges USING btree (classified_at) WHERE (classified_at IS NULL);
CREATE INDEX idx_entities_typed_name ON noo_entities_typed USING btree (canonical_name);
CREATE INDEX idx_entities_typed_type ON noo_entities_typed USING btree (type);
CREATE INDEX idx_entity_cluster_members_cluster ON entity_cluster_members USING btree (cluster_id);
CREATE INDEX idx_entity_clusters_level ON entity_clusters USING btree (level);
CREATE INDEX idx_entity_clusters_parent ON entity_clusters USING btree (parent_cluster_id);
CREATE INDEX idx_temporal_edges_current ON noo_temporal_edges USING btree (subject_id, predicate) WHERE (valid_to IS NULL);
CREATE INDEX idx_temporal_edges_entry ON noo_temporal_edges USING btree (source_entry_id) WHERE (source_entry_id IS NOT NULL);
CREATE INDEX idx_temporal_edges_object ON noo_temporal_edges USING btree (object_entity_id) WHERE (object_entity_id IS NOT NULL);
CREATE INDEX idx_temporal_edges_observed ON noo_temporal_edges USING btree (observed_at DESC);
CREATE INDEX idx_temporal_edges_subject ON noo_temporal_edges USING btree (subject_id, valid_from DESC);
CREATE INDEX idx_trellion_communities_snapshot_level ON trellion_communities USING btree (snapshot_id, level);
CREATE INDEX idx_trellion_concepts_snapshot ON trellion_concepts USING btree (snapshot_id);
CREATE INDEX idx_trellion_dag_edges_dst ON trellion_dag_edges USING btree (snapshot_id, dst_concept_id);
CREATE INDEX idx_trellion_dag_edges_snapshot ON trellion_dag_edges USING btree (snapshot_id);
CREATE INDEX idx_trellion_dag_edges_src ON trellion_dag_edges USING btree (snapshot_id, src_concept_id);
CREATE INDEX idx_trellion_provenance_concept ON trellion_provenance USING btree (concept_id);
CREATE INDEX idx_trellion_provenance_snapshot ON trellion_provenance USING btree (snapshot_id);
CREATE INDEX idx_trellion_snapshots_created ON trellion_snapshots USING btree (created_at DESC);
CREATE INDEX idx_trellion_snapshots_kind_gen ON trellion_snapshots USING btree (slice_kind, generation DESC);
CREATE INDEX idx_trellion_snapshots_status ON trellion_snapshots USING btree (status);
CREATE INDEX idx_trellion_traversal_outcome ON trellion_traversal_records USING btree (outcome_score DESC);
CREATE INDEX idx_trellion_traversal_persona ON trellion_traversal_records USING btree (persona_id, created_at DESC);
CREATE INDEX idx_trellion_video_frames_lifecycle ON trellion_video_frames USING btree (prominence_score, created_at) WHERE (is_tombstoned = false);
CREATE INDEX idx_trellion_video_frames_persona ON trellion_video_frames USING btree (persona_id, is_tombstoned);
CREATE INDEX ix_chunk_embeddings_embedding_model ON chunk_embeddings USING btree (embedding_model) WHERE (embedding_model IS NOT NULL);
CREATE INDEX ix_docs_metadata_gin ON docs USING gin (metadata);
CREATE INDEX ix_docs_source_uri ON docs USING btree (source_uri) WHERE ((source_uri IS NOT NULL) AND (source_uri <> ''::text));
CREATE INDEX ix_edges_org ON edges USING btree (org_id) WHERE (org_id IS NOT NULL);
CREATE INDEX ix_edges_user_id ON edges USING btree (user_id) WHERE (user_id IS NOT NULL);
CREATE INDEX ix_entities_org ON entities USING btree (org_id) WHERE (org_id IS NOT NULL);
CREATE INDEX ix_entities_typed_org ON noo_entities_typed USING btree (org_id) WHERE (org_id IS NOT NULL);
CREATE INDEX ix_entities_typed_user_id ON noo_entities_typed USING btree (user_id) WHERE (user_id IS NOT NULL);
CREATE INDEX ix_entities_user_id ON entities USING btree (user_id) WHERE (user_id IS NOT NULL);
CREATE INDEX ix_entity_clusters_embedding_model ON entity_clusters USING btree (embedding_model) WHERE (embedding_model IS NOT NULL);
CREATE INDEX ix_entity_embeddings_embedding_model ON entity_embeddings USING btree (embedding_model) WHERE (embedding_model IS NOT NULL);
CREATE INDEX ix_page_embeddings_embedding_model ON page_embeddings USING btree (embedding_model) WHERE (embedding_model IS NOT NULL);
CREATE INDEX ix_pages_last_accessed ON pages USING btree (last_accessed_at) WHERE (last_accessed_at IS NOT NULL);
CREATE INDEX ix_temporal_edges_org ON noo_temporal_edges USING btree (org_id) WHERE (org_id IS NOT NULL);
CREATE INDEX ix_temporal_edges_user_id ON noo_temporal_edges USING btree (user_id) WHERE (user_id IS NOT NULL);
CREATE INDEX ix_trellion_concepts_embedding_model ON trellion_concepts USING btree (embedding_model) WHERE (embedding_model IS NOT NULL);
CREATE INDEX ix_trellion_traversal_org ON trellion_traversal_records USING btree (org_id) WHERE (org_id IS NOT NULL);
CREATE INDEX ix_trellion_traversal_records_embedding_model ON trellion_traversal_records USING btree (embedding_model) WHERE (embedding_model IS NOT NULL);
CREATE INDEX ix_trellion_traversal_user_id ON trellion_traversal_records USING btree (user_id) WHERE (user_id IS NOT NULL);
CREATE INDEX mentions_entity_idx ON mentions USING btree (entity_id);
CREATE INDEX mentions_page_idx ON mentions USING btree (page_id);
CREATE INDEX page_chunks_page_idx ON page_chunks USING btree (page_id);
CREATE INDEX page_embeddings_hnsw_idx ON page_embeddings USING hnsw (embedding vector_cosine_ops);
CREATE UNIQUE INDEX pages_doc_page_idx ON pages USING btree (doc_id, page_no);
CREATE INDEX noo_sem_rel_source_idx ON noo_semantic_relationships USING btree (source_id);
CREATE INDEX noo_sem_rel_target_idx ON noo_semantic_relationships USING btree (target_id);
CREATE UNIQUE INDEX noo_sem_rel_unique ON noo_semantic_relationships USING btree (source_id, target_id, relationship_type);
CREATE UNIQUE INDEX uq_entities_typed_type_name ON noo_entities_typed USING btree (type, canonical_name);
CREATE UNIQUE INDEX uq_trellion_concepts_snapshot_name_type ON trellion_concepts USING btree (snapshot_id, canonical_name, type);
ALTER TABLE ONLY chunk_embeddings
    ADD CONSTRAINT chunk_embeddings_chunk_id_fkey FOREIGN KEY (chunk_id) REFERENCES page_chunks(chunk_id) ON DELETE CASCADE;
ALTER TABLE ONLY edges
    ADD CONSTRAINT edges_dst_entity_id_fkey FOREIGN KEY (dst_entity_id) REFERENCES entities(entity_id) ON DELETE CASCADE;
ALTER TABLE ONLY edges
    ADD CONSTRAINT edges_src_entity_id_fkey FOREIGN KEY (src_entity_id) REFERENCES entities(entity_id) ON DELETE CASCADE;
ALTER TABLE ONLY entity_cluster_members
    ADD CONSTRAINT entity_cluster_members_cluster_id_fkey FOREIGN KEY (cluster_id) REFERENCES entity_clusters(cluster_id) ON DELETE CASCADE;
ALTER TABLE ONLY entity_cluster_members
    ADD CONSTRAINT entity_cluster_members_entity_id_fkey FOREIGN KEY (entity_id) REFERENCES entities(entity_id) ON DELETE CASCADE;
ALTER TABLE ONLY entity_clusters
    ADD CONSTRAINT entity_clusters_parent_cluster_id_fkey FOREIGN KEY (parent_cluster_id) REFERENCES entity_clusters(cluster_id);
ALTER TABLE ONLY entity_embeddings
    ADD CONSTRAINT entity_embeddings_entity_id_fkey FOREIGN KEY (entity_id) REFERENCES entities(entity_id) ON DELETE CASCADE;
ALTER TABLE ONLY entity_graph_signals
    ADD CONSTRAINT entity_graph_signals_entity_id_fkey FOREIGN KEY (entity_id) REFERENCES entities(entity_id) ON DELETE CASCADE;
ALTER TABLE ONLY mentions
    ADD CONSTRAINT mentions_entity_id_fkey FOREIGN KEY (entity_id) REFERENCES entities(entity_id) ON DELETE CASCADE;
ALTER TABLE ONLY mentions
    ADD CONSTRAINT mentions_page_id_fkey FOREIGN KEY (page_id) REFERENCES pages(page_id) ON DELETE CASCADE;
ALTER TABLE ONLY page_chunks
    ADD CONSTRAINT page_chunks_page_id_fkey FOREIGN KEY (page_id) REFERENCES pages(page_id) ON DELETE CASCADE;
ALTER TABLE ONLY page_embeddings
    ADD CONSTRAINT page_embeddings_page_id_fkey FOREIGN KEY (page_id) REFERENCES pages(page_id) ON DELETE CASCADE;
ALTER TABLE ONLY pages
    ADD CONSTRAINT pages_doc_id_fkey FOREIGN KEY (doc_id) REFERENCES docs(doc_id) ON DELETE CASCADE;
ALTER TABLE ONLY noo_temporal_edges
    ADD CONSTRAINT noo_temporal_edges_object_entity_id_fkey FOREIGN KEY (object_entity_id) REFERENCES noo_entities_typed(id) ON DELETE SET NULL;
ALTER TABLE ONLY noo_temporal_edges
    ADD CONSTRAINT noo_temporal_edges_subject_id_fkey FOREIGN KEY (subject_id) REFERENCES noo_entities_typed(id) ON DELETE CASCADE;
ALTER TABLE ONLY trellion_communities
    ADD CONSTRAINT trellion_communities_parent_community_id_fkey FOREIGN KEY (parent_community_id) REFERENCES trellion_communities(id);
ALTER TABLE ONLY trellion_communities
    ADD CONSTRAINT trellion_communities_snapshot_id_fkey FOREIGN KEY (snapshot_id) REFERENCES trellion_snapshots(id) ON DELETE CASCADE;
ALTER TABLE ONLY trellion_concepts
    ADD CONSTRAINT trellion_concepts_snapshot_id_fkey FOREIGN KEY (snapshot_id) REFERENCES trellion_snapshots(id) ON DELETE CASCADE;
ALTER TABLE ONLY trellion_dag_edges
    ADD CONSTRAINT trellion_dag_edges_dst_concept_id_fkey FOREIGN KEY (dst_concept_id) REFERENCES trellion_concepts(id) ON DELETE CASCADE;
ALTER TABLE ONLY trellion_dag_edges
    ADD CONSTRAINT trellion_dag_edges_snapshot_id_fkey FOREIGN KEY (snapshot_id) REFERENCES trellion_snapshots(id) ON DELETE CASCADE;
ALTER TABLE ONLY trellion_dag_edges
    ADD CONSTRAINT trellion_dag_edges_src_concept_id_fkey FOREIGN KEY (src_concept_id) REFERENCES trellion_concepts(id) ON DELETE CASCADE;
ALTER TABLE ONLY trellion_provenance
    ADD CONSTRAINT trellion_provenance_concept_id_fkey FOREIGN KEY (concept_id) REFERENCES trellion_concepts(id) ON DELETE CASCADE;
ALTER TABLE ONLY trellion_provenance
    ADD CONSTRAINT trellion_provenance_snapshot_id_fkey FOREIGN KEY (snapshot_id) REFERENCES trellion_snapshots(id) ON DELETE CASCADE;
ALTER TABLE ONLY trellion_traversal_records
    ADD CONSTRAINT trellion_traversal_records_video_frame_id_fkey FOREIGN KEY (video_frame_id) REFERENCES trellion_video_frames(id) ON DELETE SET NULL;
"""

# The conversation group: `noo_conversations` and its four children, the
# `noo_conv_set_org` trigger and function. Nothing in `_DDL` references it.
_CONVERSATION_DDL = """\
CREATE FUNCTION noo_conv_set_org() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    IF NEW.org_id IS NULL AND NEW.user_id IS NOT NULL THEN
        IF NEW.user_id LIKE 'team:%' THEN
            NEW.org_id := (SELECT org_id FROM noo_kb_teams
                           WHERE 'team:' || name = NEW.user_id);
        ELSE
            NEW.org_id := (SELECT org_id FROM noo_users
                           WHERE username = NEW.user_id);
        END IF;
    END IF;
    RETURN NEW;
END;
$$;
CREATE TABLE noo_context_retrievals (
    id uuid NOT NULL,
    conversation_id uuid NOT NULL,
    entry_id uuid,
    query text NOT NULL,
    scope text NOT NULL,
    results jsonb NOT NULL,
    top_score double precision,
    retrieval_ms integer,
    created_at timestamp with time zone DEFAULT now() NOT NULL
);
CREATE TABLE noo_conversations (
    id uuid NOT NULL,
    user_id text,
    session_id text,
    persona_id text,
    backend text,
    model text,
    started_at timestamp with time zone DEFAULT now() NOT NULL,
    ended_at timestamp with time zone,
    total_messages integer DEFAULT 0,
    total_tool_calls integer DEFAULT 0,
    total_tokens_in integer DEFAULT 0,
    total_tokens_out integer DEFAULT 0,
    total_latency_ms integer DEFAULT 0,
    metadata jsonb DEFAULT '{}'::jsonb,
    org_id text
);
CREATE TABLE noo_entries (
    id uuid NOT NULL,
    conversation_id uuid NOT NULL,
    parent_id uuid,
    type text NOT NULL,
    content text NOT NULL,
    metadata jsonb DEFAULT '{}'::jsonb,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    embedding vector(1024),
    indexed_at timestamp with time zone,
    embedding_model text,
    embedding_dim integer
);
CREATE TABLE noo_feedback (
    id uuid NOT NULL,
    conversation_id uuid,
    entry_id uuid,
    feedback_type text NOT NULL,
    value jsonb,
    created_at timestamp with time zone DEFAULT now() NOT NULL
);
CREATE TABLE noo_tool_executions (
    id uuid NOT NULL,
    conversation_id uuid NOT NULL,
    entry_id uuid,
    tool_name text NOT NULL,
    arguments jsonb NOT NULL,
    success boolean,
    output text,
    error text,
    execution_ms integer,
    created_at timestamp with time zone DEFAULT now() NOT NULL
);
ALTER TABLE ONLY noo_context_retrievals
    ADD CONSTRAINT noo_context_retrievals_pkey PRIMARY KEY (id);
ALTER TABLE ONLY noo_conversations
    ADD CONSTRAINT noo_conversations_pkey PRIMARY KEY (id);
ALTER TABLE ONLY noo_entries
    ADD CONSTRAINT noo_entries_pkey PRIMARY KEY (id);
ALTER TABLE ONLY noo_feedback
    ADD CONSTRAINT noo_feedback_pkey PRIMARY KEY (id);
ALTER TABLE ONLY noo_tool_executions
    ADD CONSTRAINT noo_tool_executions_pkey PRIMARY KEY (id);
CREATE INDEX ix_conversations_org ON noo_conversations USING btree (org_id) WHERE (org_id IS NOT NULL);
CREATE INDEX ix_noo_conversations_persona_id ON noo_conversations USING btree (persona_id);
CREATE INDEX ix_noo_entries_embedding_model ON noo_entries USING btree (embedding_model) WHERE (embedding_model IS NOT NULL);
CREATE INDEX noo_conv_persona_idx ON noo_conversations USING btree (persona_id, started_at DESC);
CREATE INDEX noo_conv_user_idx ON noo_conversations USING btree (user_id, started_at DESC);
CREATE INDEX noo_entries_content_trgm_idx ON noo_entries USING gin (content gin_trgm_ops);
CREATE INDEX noo_entries_conv_idx ON noo_entries USING btree (conversation_id, created_at);
CREATE INDEX noo_entries_embedding_idx ON noo_entries USING hnsw (embedding vector_cosine_ops) WHERE (embedding IS NOT NULL);
CREATE INDEX noo_entries_type_idx ON noo_entries USING btree (type);
CREATE INDEX noo_entries_unembedded_idx ON noo_entries USING btree (created_at) WHERE (embedding IS NULL);
CREATE INDEX noo_tool_exec_conv_idx ON noo_tool_executions USING btree (conversation_id);
CREATE INDEX noo_tool_exec_name_idx ON noo_tool_executions USING btree (tool_name, success);
CREATE TRIGGER noo_conv_set_org_trg BEFORE INSERT ON noo_conversations FOR EACH ROW EXECUTE FUNCTION noo_conv_set_org();
ALTER TABLE ONLY noo_context_retrievals
    ADD CONSTRAINT noo_context_retrievals_conversation_id_fkey FOREIGN KEY (conversation_id) REFERENCES noo_conversations(id);
ALTER TABLE ONLY noo_context_retrievals
    ADD CONSTRAINT noo_context_retrievals_entry_id_fkey FOREIGN KEY (entry_id) REFERENCES noo_entries(id);
ALTER TABLE ONLY noo_entries
    ADD CONSTRAINT noo_entries_conversation_id_fkey FOREIGN KEY (conversation_id) REFERENCES noo_conversations(id);
ALTER TABLE ONLY noo_feedback
    ADD CONSTRAINT noo_feedback_conversation_id_fkey FOREIGN KEY (conversation_id) REFERENCES noo_conversations(id);
ALTER TABLE ONLY noo_feedback
    ADD CONSTRAINT noo_feedback_entry_id_fkey FOREIGN KEY (entry_id) REFERENCES noo_entries(id);
ALTER TABLE ONLY noo_tool_executions
    ADD CONSTRAINT noo_tool_executions_conversation_id_fkey FOREIGN KEY (conversation_id) REFERENCES noo_conversations(id);
ALTER TABLE ONLY noo_tool_executions
    ADD CONSTRAINT noo_tool_executions_entry_id_fkey FOREIGN KEY (entry_id) REFERENCES noo_entries(id);
"""


def _exists_anywhere(bind, table: str) -> bool:
    return bool(bind.execute(
        sa.text(
            "SELECT 1 FROM information_schema.tables WHERE table_name = :t"
        ),
        {"t": table},
    ).scalar())


def upgrade() -> None:
    bind = op.get_bind()
    if _exists_anywhere(bind, _SENTINEL):
        # Legacy install: everything already exists (in public), built by the
        # old 001–012 chain. Stamp and stop — the tables move at the cutover,
        # not here, because a migration that also relocates data is a failure
        # nobody can attribute.
        return
    # ONE execute per blob, through the RAW DRIVER connection.
    # Two failed cuts bought this shape:
    #   - splitting on ";\n" shattered the `noo_conv_set_org` function, whose
    #     plpgsql body legitimately contains semicolon-newlines of its own —
    #     dollar-quoted bodies make statement splitting a parser problem;
    #   - exec_driver_sql() then failed on the SAME function's
    #     `LIKE 'team:%'` — SQLAlchemy passes an (empty) params tuple, which
    #     makes psycopg parse `%` as a placeholder even in DDL.
    # psycopg's own execute() with params=None sends the text as-is: no
    # splitting, no placeholder scan. Same alembic transaction — it is the
    # same underlying connection.
    raw = bind.connection.driver_connection
    raw.execute(_DDL)
    if not _exists_anywhere(bind, _CONVERSATION_SENTINEL):
        raw.execute(_CONVERSATION_DDL)


def downgrade() -> None:
    # Refuses: dropping these tables deletes every
    # conversation, document, embedding and graph edge the install has.
    raise NotImplementedError(
        "downgrade would drop the entire memory plane; restore from a backup"
    )
