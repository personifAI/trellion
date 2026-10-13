# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""Document hold: keep a document, out of embedding and search.

A held document keeps its row, its pages, its chunks and its ids. It is
simply not offered to the embedder, not returned by search, and not counted
as unembedded. Nothing is deleted by holding; a separate, explicit purge
removes documents held under a named reason.

- `docs.held_at` (timestamptz, NULL = not held)
- `docs.hold_reason` (text, NULL when not held)
- a partial index over the held rows (listing / purge), and a partial index
  over the not-held rows (the anti-join every picker and search now does).

Idempotent (IF NOT EXISTS); both columns are nullable, so an old reader and
an old writer keep working.

Revision ID: 105_doc_hold
Revises: 104_recallable_walks
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op

revision: str = "105_doc_hold"
down_revision: Union[str, None] = "104_recallable_walks"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("ALTER TABLE docs ADD COLUMN IF NOT EXISTS held_at timestamp with time zone")
    op.execute("ALTER TABLE docs ADD COLUMN IF NOT EXISTS hold_reason text")
    op.execute(
        "CREATE INDEX IF NOT EXISTS docs_held_idx ON docs (hold_reason, held_at) "
        "WHERE held_at IS NOT NULL"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS docs_not_held_idx ON docs (doc_id) "
        "WHERE held_at IS NULL"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS docs_not_held_idx")
    op.execute("DROP INDEX IF EXISTS docs_held_idx")
    op.execute("ALTER TABLE docs DROP COLUMN IF EXISTS hold_reason")
    op.execute("ALTER TABLE docs DROP COLUMN IF EXISTS held_at")
