# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""MetadataNarrower — narrows candidate docs by structured metadata.

Filters supported in `ctx['metadata_filters']`:
    {
        # String matchers
        "title_substr":        str,    # ILIKE %X% on docs.title
        "source_uri_prefix":   str,    # docs.source_uri starts with X
        "source_uri_pattern":  str,    # SQL LIKE pattern on docs.source_uri

        # JSONB matchers (against docs.metadata)
        "metadata_contains":   dict,   # docs.metadata @> X (containment)
        "metadata_has_keys":   list,   # docs.metadata ?| X (any key match)

        # Temporal (also lives here for now; TemporalNarrower will own
        # this filter cleanly when it lands)
        "created_after":       str,    # ISO datetime
        "created_before":      str,    # ISO datetime
    }

Any combination is AND-composed at the SQL level. Empty filters → no
opinion (returns NarrowerResult with both ID fields None).

NL extraction (turning "PDFs in /projects/" into the filter dict
automatically) is intentionally NOT in scope for this narrower —
that's a separate query-understanding layer that populates ctx before
the orchestrator runs.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from sqlalchemy import text

from .base import NarrowerResult

log = logging.getLogger(__name__)


class MetadataNarrower:
    """Narrower that filters docs by metadata fields on the `docs` table."""

    name = "metadata"

    def __init__(self, store: Any):
        """Args: store: PageIndexStore (typed Any to avoid circular import)."""
        self.store = store

    async def narrow(self, query: str, ctx: dict) -> NarrowerResult:
        filters = (ctx or {}).get("metadata_filters") or {}
        if not isinstance(filters, dict) or not filters:
            return NarrowerResult(name=self.name)

        sql, params = self._build_query(filters)
        if sql is None:
            # All filter values were empty — no opinion
            return NarrowerResult(name=self.name)

        try:
            engine = getattr(self.store, "_engine", None) or getattr(self.store, "engine", None)
            if engine is None:
                log.warning("metadata_narrower.no_engine_on_store")
                return NarrowerResult(name=self.name)
            async with engine.connect() as conn:
                result = await conn.execute(text(sql), params)
                rows = result.fetchall()
        except Exception:
            log.warning("metadata_narrower.query_failed", exc_info=True)
            return NarrowerResult(name=self.name)

        doc_ids = {str(r[0]) for r in rows}
        log.info(
            "metadata_narrower.matched docs=%d filters=%s",
            len(doc_ids),
            list(filters.keys()),
        )
        return NarrowerResult(
            name=self.name,
            doc_ids=doc_ids,
            debug={"filter_keys": list(filters.keys()), "match_count": len(doc_ids)},
        )

    @staticmethod
    def _build_query(filters: dict) -> tuple[Optional[str], dict]:
        """Compose a parameterized SELECT against `docs` from a filter dict.

        Returns (sql, params) or (None, {}) if no clauses ended up active.
        """
        import json as _json

        clauses: list[str] = []
        params: dict = {}

        title = filters.get("title_substr")
        if title:
            clauses.append("title ILIKE :title_substr")
            params["title_substr"] = f"%{title}%"

        prefix = filters.get("source_uri_prefix")
        if prefix:
            clauses.append("source_uri LIKE :source_uri_prefix")
            params["source_uri_prefix"] = f"{prefix}%"

        pattern = filters.get("source_uri_pattern")
        if pattern:
            clauses.append("source_uri LIKE :source_uri_pattern")
            params["source_uri_pattern"] = pattern

        meta_contains = filters.get("metadata_contains")
        if isinstance(meta_contains, dict) and meta_contains:
            clauses.append("metadata @> CAST(:meta_contains AS jsonb)")
            params["meta_contains"] = _json.dumps(meta_contains)

        meta_keys = filters.get("metadata_has_keys")
        if isinstance(meta_keys, (list, tuple)) and meta_keys:
            # `?|` operator wants a text[] — pass via `= ANY` workaround
            # using array literal syntax for portability with sqlalchemy.
            clauses.append("metadata ?| ARRAY[:meta_keys]")
            # Note: sqlalchemy expands list params to ARRAY when bound this
            # way; if not, we'd need an explicit cast. Most setups handle it.
            params["meta_keys"] = list(meta_keys)

        after = filters.get("created_after")
        if after:
            clauses.append("created_at >= :created_after")
            params["created_after"] = after

        before = filters.get("created_before")
        if before:
            clauses.append("created_at <= :created_before")
            params["created_before"] = before

        if not clauses:
            return None, {}

        # A held document is never a candidate (migration 105).
        clauses.append("held_at IS NULL")
        sql = "SELECT doc_id FROM docs WHERE " + " AND ".join(clauses)
        return sql, params
