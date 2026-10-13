# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""vidmem → episodic recall: rename the tables in place (operator, 2026-08-29).

"Video memory" described the retired v1 MP4+QR transport; the rows are
EPISODES (archived experiences — today, proven graph traversals). The
`faiss_index_name` column loses its FAISS fossil name in the same pass.

An UPDATABLE compat view keeps the old table/column names serving reads
AND writes until the gateway side (Rust routes/role grants, worker
SQL) lands its rename — cross-repo lockstep without an atomic deploy.
Drop the view at the P2 relocation.

Revision ID: 102_episodes_rename
Revises: 101_frame_embeddings
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op

revision: str = "102_episodes_rename"
down_revision: Union[str, None] = "101_frame_embeddings"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("ALTER TABLE trellion_video_frames RENAME TO trellion_episodes")
    op.execute(
        "ALTER TABLE trellion_episodes RENAME COLUMN faiss_index_name TO index_name"
    )


def downgrade() -> None:
    op.execute(
        "ALTER TABLE trellion_episodes RENAME COLUMN index_name TO faiss_index_name"
    )
    op.execute("ALTER TABLE trellion_episodes RENAME TO trellion_video_frames")
