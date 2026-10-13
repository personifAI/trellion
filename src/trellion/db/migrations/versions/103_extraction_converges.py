# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""Entity extraction converges: a page is done when it was extracted, not
when it happened to yield a mention (operator, 2026-09-21).

The extractor selected "pages with no mention rows". A page whose
extraction genuinely found nothing never gets a mention row, so it was
selected again every ten seconds, forever — on one node the same fifty pages
went through the chat model sixteen times each per half hour, every call
landing on the one node that serves chat. Two ML containers ran the loop
against the same table with nothing between them, so every page was
extracted twice and every mention was inserted twice: 12,972 duplicate
mention rows and 140 duplicate edges when this was measured.

- `pages.extracted_at`: set once extraction completed, including with an
  empty result. NULL = never extracted. The loop selects on it.
- `pages.extraction_claimed_at`: an atomic claim, so a second extractor
  skips a page another is working on; a stale claim (a crashed worker)
  is reclaimable after fifteen minutes.
- Pages that already have mentions were extracted; they are backfilled as
  done so the roll does not re-run the whole corpus.
- `mentions (entity_id, page_id)` and `edges (src, rel_type, dst)` become
  unique after the duplicates are removed, so a race can never double a
  row again; the store inserts use ON CONFLICT against these.

Revision ID: 103_extraction_converges
Revises: 102_episodes_rename
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op

revision: str = "103_extraction_converges"
down_revision: Union[str, None] = "102_episodes_rename"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("ALTER TABLE pages ADD COLUMN IF NOT EXISTS extracted_at timestamp with time zone")
    op.execute("ALTER TABLE pages ADD COLUMN IF NOT EXISTS extraction_claimed_at timestamp with time zone")
    # Extracted already: anything that produced a mention.
    op.execute(
        "UPDATE pages p SET extracted_at = NOW() "
        "WHERE p.extracted_at IS NULL "
        "AND EXISTS (SELECT 1 FROM mentions m WHERE m.page_id = p.page_id)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_pages_unextracted ON pages (page_id) "
        "WHERE extracted_at IS NULL"
    )
    # Duplicates first, then the constraint that makes them impossible.
    # ctid picks one survivor per group; the rows are identical in every
    # column the graph reads (quote/span differ only by which run wrote
    # them, and both runs read the same page).
    op.execute(
        "DELETE FROM mentions a USING mentions b "
        "WHERE a.entity_id = b.entity_id AND a.page_id = b.page_id AND a.ctid > b.ctid"
    )
    op.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS mentions_entity_page_uniq "
        "ON mentions (entity_id, page_id)"
    )
    op.execute(
        "DELETE FROM edges a USING edges b "
        "WHERE a.src_entity_id = b.src_entity_id AND a.rel_type = b.rel_type "
        "AND a.dst_entity_id = b.dst_entity_id AND a.ctid > b.ctid"
    )
    op.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS edges_src_rel_dst_uniq "
        "ON edges (src_entity_id, rel_type, dst_entity_id)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS edges_src_rel_dst_uniq")
    op.execute("DROP INDEX IF EXISTS mentions_entity_page_uniq")
    op.execute("DROP INDEX IF EXISTS ix_pages_unextracted")
    op.execute("ALTER TABLE pages DROP COLUMN IF EXISTS extraction_claimed_at")
    op.execute("ALTER TABLE pages DROP COLUMN IF EXISTS extracted_at")
