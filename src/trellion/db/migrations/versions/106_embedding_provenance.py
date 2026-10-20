# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""Embedding provenance and the room to re-embed a space (noomesh i150).

Every vector records the model that produced it (``embedding_model``) and
its width (``embedding_dim``); readers filter on the space's current model,
so a table never ranks vectors from two models against each other.

- ``trellion_episodes`` gains ``embedding_model`` / ``embedding_dim`` (the
  other seven vector tables have carried them since the 100 squash) plus the
  same partial btree on ``embedding_model``.
- Every vector table gains ``embedding_next vector(1024)`` and
  ``embedding_next_model text``: the staging column a re-embed fills while the
  live column keeps serving. Same width as the live column; the gateway
  refuses a migration to another width. A cutover swaps the columns in one
  transaction per table.

NOT backfilled here: this migration cannot know which model produced the
existing vectors. The worker labels every NULL-model vector with the space's
current model at start-up (every pre-provenance vector came from the default
space). Idempotent (IF NOT EXISTS) and additive, so old readers and writers
keep working through the roll.

Revision ID: 106_embedding_provenance
Revises: 105_doc_hold
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op

revision: str = "106_embedding_provenance"
down_revision: Union[str, None] = "105_doc_hold"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

#: Every table holding a vector column that a space's model produces.
VECTOR_TABLES = (
    "chunk_embeddings",
    "page_embeddings",
    "entity_embeddings",
    "entity_clusters",
    "noo_entries",
    "trellion_concepts",
    "trellion_traversal_records",
    "trellion_episodes",
)


def upgrade() -> None:
    op.execute("ALTER TABLE trellion_episodes ADD COLUMN IF NOT EXISTS embedding_model text")
    op.execute("ALTER TABLE trellion_episodes ADD COLUMN IF NOT EXISTS embedding_dim integer")
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_trellion_episodes_embedding_model "
        "ON trellion_episodes (embedding_model) WHERE embedding_model IS NOT NULL"
    )
    for table in VECTOR_TABLES:
        op.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS embedding_next vector(1024)")
        op.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS embedding_next_model text")


def downgrade() -> None:
    for table in reversed(VECTOR_TABLES):
        op.execute(f"ALTER TABLE {table} DROP COLUMN IF EXISTS embedding_next_model")
        op.execute(f"ALTER TABLE {table} DROP COLUMN IF EXISTS embedding_next")
    op.execute("DROP INDEX IF EXISTS ix_trellion_episodes_embedding_model")
    op.execute("ALTER TABLE trellion_episodes DROP COLUMN IF EXISTS embedding_dim")
    op.execute("ALTER TABLE trellion_episodes DROP COLUMN IF EXISTS embedding_model")
