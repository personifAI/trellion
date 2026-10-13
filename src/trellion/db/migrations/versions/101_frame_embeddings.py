# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""Frame embeddings move into Postgres (gateway proposal 28 Part 1).

``trellion_video_frames`` gains the embedding column so vector search
runs in pgvector (HNSW, cosine — the house pattern on five other
embedding tables) instead of per-persona FAISS files. The FAISS files
duplicated both the vectors (already in
``trellion_traversal_records.query_embedding`` for the only populated
index) and the metadata (prominence/tombstone already lived on the
frame row); this migration plus the pure-SQL backfill in
``trellion vidmem-backfill`` retires the last file store.

Dim is 1024, matching every other vector column in the 100 squash.

Revision ID: 101_frame_embeddings
Revises: 100_trellion_squash
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op

revision: str = "101_frame_embeddings"
down_revision: Union[str, None] = "100_trellion_squash"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE trellion_video_frames "
        "ADD COLUMN IF NOT EXISTS embedding vector(1024)"
    )
    # Partial: tombstoned/unembedded rows never serve search.
    op.execute(
        "CREATE INDEX IF NOT EXISTS trellion_video_frames_embedding_hnsw_idx "
        "ON trellion_video_frames USING hnsw (embedding vector_cosine_ops) "
        "WHERE embedding IS NOT NULL"
    )


def downgrade() -> None:
    op.execute(
        "DROP INDEX IF EXISTS trellion_video_frames_embedding_hnsw_idx"
    )
    op.execute(
        "ALTER TABLE trellion_video_frames DROP COLUMN IF EXISTS embedding"
    )
