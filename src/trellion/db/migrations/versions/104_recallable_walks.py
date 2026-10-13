# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""Recallable walks: a user's own threads and the system's simulated walks
share one store (noomesh proposal 57, phase 1a; operator, 2026-10-03).

`trellion_traversal_records` already holds proven graph walks, but nothing
live wrote a RECALLABLE one: the only writer was the heartbeat rehearsal, with
no asker and no embedding. This adds what a recall path needs, additively:

- `kind`: `thread` (a user's real walk, attributed) or `simulated` (a
  system-chosen rehearsal, never attributed). Every row that exists today is
  an unattributed heartbeat rehearsal, so the default and the backfill are
  both `simulated`.
- `thread_id` (uuid, nullable: the conversation id when the caller has one),
  `turn_index`, `run_id` (the heartbeat run of a simulated walk).
- `provenance` jsonb: how the walk was produced (tool, request id, seeds).
  Never the asker; that is `user_id`.
- `evidence` jsonb: the save-time snapshot the recall-time hard gates
  re-check: `{triples: [[src, rel, dst], ...], doc_ids: [...],
  newest_evidence_at}`.
- `usefulness` (nullable, computed by the caller and cached), `status`
  (`active` | `stale` | `staged`), `last_validated_at`, `last_recalled_at`,
  `recall_count`, `expires_at`.

`trellion_episodes.user_id`: every capture writes an episode row whose
`content_summary` carries the first 100 characters of the asker's question and
whose embedding is of that question. It had no owner column, so right-to-be-
forgotten could not reach it. Existing episodes are attributed from the
traversal record that points at them (`video_frame_id`).

Idempotent (IF NOT EXISTS / drop-then-add constraints); every column is
nullable or defaulted, so an old reader and an old writer keep working.

Revision ID: 104_recallable_walks
Revises: 103_extraction_converges
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op

revision: str = "104_recallable_walks"
down_revision: Union[str, None] = "103_extraction_converges"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_T = "trellion_traversal_records"


def upgrade() -> None:
    # DEFAULT 'simulated' backfills every existing row in the same statement.
    op.execute(f"ALTER TABLE {_T} ADD COLUMN IF NOT EXISTS kind text NOT NULL DEFAULT 'simulated'")
    op.execute(f"ALTER TABLE {_T} ADD COLUMN IF NOT EXISTS thread_id uuid")
    op.execute(f"ALTER TABLE {_T} ADD COLUMN IF NOT EXISTS turn_index integer")
    op.execute(f"ALTER TABLE {_T} ADD COLUMN IF NOT EXISTS run_id text")
    op.execute(f"ALTER TABLE {_T} ADD COLUMN IF NOT EXISTS provenance jsonb NOT NULL DEFAULT '{{}}'::jsonb")
    op.execute(f"ALTER TABLE {_T} ADD COLUMN IF NOT EXISTS evidence jsonb NOT NULL DEFAULT '{{}}'::jsonb")
    op.execute(f"ALTER TABLE {_T} ADD COLUMN IF NOT EXISTS usefulness double precision")
    op.execute(f"ALTER TABLE {_T} ADD COLUMN IF NOT EXISTS status text NOT NULL DEFAULT 'active'")
    op.execute(f"ALTER TABLE {_T} ADD COLUMN IF NOT EXISTS last_validated_at timestamp with time zone")
    op.execute(f"ALTER TABLE {_T} ADD COLUMN IF NOT EXISTS last_recalled_at timestamp with time zone")
    op.execute(f"ALTER TABLE {_T} ADD COLUMN IF NOT EXISTS recall_count integer NOT NULL DEFAULT 0")
    op.execute(f"ALTER TABLE {_T} ADD COLUMN IF NOT EXISTS expires_at timestamp with time zone")

    op.execute(f"ALTER TABLE {_T} DROP CONSTRAINT IF EXISTS ck_trellion_traversal_kind")
    op.execute(
        f"ALTER TABLE {_T} ADD CONSTRAINT ck_trellion_traversal_kind "
        "CHECK (kind IN ('thread', 'simulated'))"
    )
    op.execute(f"ALTER TABLE {_T} DROP CONSTRAINT IF EXISTS ck_trellion_traversal_status")
    op.execute(
        f"ALTER TABLE {_T} ADD CONSTRAINT ck_trellion_traversal_status "
        "CHECK (status IN ('active', 'stale', 'staged'))"
    )

    op.execute(
        f"CREATE INDEX IF NOT EXISTS ix_trellion_traversal_user_kind_status "
        f"ON {_T} (user_id, kind, status)"
    )
    op.execute(
        f"CREATE INDEX IF NOT EXISTS ix_trellion_traversal_thread "
        f"ON {_T} (thread_id) WHERE thread_id IS NOT NULL"
    )

    # Episodes: an owner column so erasure can reach the question text.
    op.execute("ALTER TABLE trellion_episodes ADD COLUMN IF NOT EXISTS user_id text")
    op.execute(
        "UPDATE trellion_episodes e SET user_id = r.user_id "
        f"FROM {_T} r WHERE r.video_frame_id = e.id "
        "AND r.user_id IS NOT NULL AND e.user_id IS NULL"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_trellion_episodes_user_id "
        "ON trellion_episodes (user_id) WHERE user_id IS NOT NULL"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_trellion_episodes_user_id")
    op.execute("ALTER TABLE trellion_episodes DROP COLUMN IF EXISTS user_id")
    op.execute("DROP INDEX IF EXISTS ix_trellion_traversal_thread")
    op.execute("DROP INDEX IF EXISTS ix_trellion_traversal_user_kind_status")
    op.execute(f"ALTER TABLE {_T} DROP CONSTRAINT IF EXISTS ck_trellion_traversal_status")
    op.execute(f"ALTER TABLE {_T} DROP CONSTRAINT IF EXISTS ck_trellion_traversal_kind")
    for col in (
        "expires_at", "recall_count", "last_recalled_at", "last_validated_at",
        "status", "usefulness", "evidence", "provenance", "run_id",
        "turn_index", "thread_id", "kind",
    ):
        op.execute(f"ALTER TABLE {_T} DROP COLUMN IF EXISTS {col}")
