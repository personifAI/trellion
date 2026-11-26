# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""Knowledge disposition, slice 1 (shadow mode): usage, spec versions, decisions.

Three new tables; no existing table changes.

- ``page_usage``: how often a page was returned by retrieval (one row per
  page, upserted off the request path).
- ``kb_specs``: versioned specs ("what good looks like"); at most one row is
  active (partial unique index). With no rows, trellion uses its built-in
  default, reported as version 1.
- ``page_dispositions``: the disposition a decision model would give a page
  (embed it, extract entities from it, or hold it back), with the raw answers
  as the audit record, plus the thresholds in force, a hash of the spec body
  used and the question ids, so a row explains itself even after the built-in
  default changes. ``mode`` is ``'shadow'``: nothing reads these rows to
  change what is embedded or extracted.

Idempotent (IF NOT EXISTS); additive, so an old reader and an old writer keep
working through the roll.

Revision ID: 107_knowledge_disposition
Revises: 106_embedding_provenance
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op

revision: str = "107_knowledge_disposition"
down_revision: Union[str, None] = "106_embedding_provenance"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        "CREATE TABLE IF NOT EXISTS page_usage ("
        "page_id uuid PRIMARY KEY REFERENCES pages (page_id) ON DELETE CASCADE, "
        "retrieved_count bigint NOT NULL DEFAULT 0, "
        "first_retrieved_at timestamp with time zone NOT NULL DEFAULT now(), "
        "last_retrieved_at timestamp with time zone NOT NULL DEFAULT now())"
    )
    op.execute(
        "CREATE TABLE IF NOT EXISTS kb_specs ("
        "version integer PRIMARY KEY, "
        "spec jsonb NOT NULL, "
        "created_at timestamp with time zone NOT NULL DEFAULT now(), "
        "created_by text, "
        "note text, "
        "active boolean NOT NULL DEFAULT false)"
    )
    # At most one active spec: every active row indexes the same constant.
    op.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS kb_specs_one_active_idx "
        "ON kb_specs ((true)) WHERE active"
    )
    op.execute(
        "CREATE TABLE IF NOT EXISTS page_dispositions ("
        "page_id uuid PRIMARY KEY REFERENCES pages (page_id) ON DELETE CASCADE, "
        "spec_version integer NOT NULL, "
        "would_embed boolean NOT NULL, "
        "would_graph boolean NOT NULL, "
        "held boolean NOT NULL, "
        "answers jsonb NOT NULL, "
        "thresholds jsonb NOT NULL, "
        "spec_hash text NOT NULL, "
        "question_ids jsonb NOT NULL, "
        "model text, "
        "decided_at timestamp with time zone NOT NULL DEFAULT now(), "
        "review_after timestamp with time zone, "
        "mode text NOT NULL DEFAULT 'shadow')"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS page_dispositions_review_idx "
        "ON page_dispositions (review_after) WHERE held"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS page_dispositions_review_idx")
    op.execute("DROP TABLE IF EXISTS page_dispositions")
    op.execute("DROP INDEX IF EXISTS kb_specs_one_active_idx")
    op.execute("DROP TABLE IF EXISTS kb_specs")
    op.execute("DROP TABLE IF EXISTS page_usage")
