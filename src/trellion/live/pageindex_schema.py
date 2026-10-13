# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""PageIndex + Knowledge Graph schema for PostgreSQL + pgvector.

Schema is now defined in this repo's own ``trellion.db.models`` (extracted
out of ``noomesh/db/models.py`` in proposal 06 phase 4) and managed by
Alembic migrations. This module is kept for backward compatibility —
``ensure_tables()`` is now a no-op that logs a deprecation notice.
"""

from __future__ import annotations

import structlog

log = structlog.get_logger(__name__)


class PageIndexSchema:
    """Manages the PageIndex + Graph database schema.

    Schema creation has moved to ``noomesh.db.models`` + Alembic.
    This class is retained for API compatibility only.
    """

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn

    async def ensure_tables(self) -> None:
        """No-op — schema is managed by Alembic migrations via ``noo db upgrade``."""
        log.info("pageindex_schema.ensure_tables_noop",
                 message="Schema managed by Alembic; skipping manual CREATE")
