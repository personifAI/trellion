# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""Storage for the knowledge spec and recorded dispositions (migration 107).

Raw SQL over an async engine, like :class:`PageIndexStore`. This class only
ever reads ``pages`` / ``docs`` / ``page_embeddings`` and writes
``kb_specs`` and ``page_dispositions``.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any, Mapping, Optional

import structlog
from sqlalchemy import text

from trellion.live.knowledge_spec import (
    DEFAULT_SPEC_VERSION,
    KnowledgeSpec,
    SpecError,
    default_spec,
)

log = structlog.get_logger(__name__)


class KnowledgeStore:
    def __init__(self, engine: Any) -> None:
        self._engine = engine
        # Picks in this process take turns, and a picked page stays excluded
        # until its batch ends (the same arrangement as the embed claim).
        self._pick_lock = asyncio.Lock()
        self._inflight: set[str] = set()

    # ------------------------------------------------------------------
    # Spec versions
    # ------------------------------------------------------------------

    async def load_active_spec(self) -> KnowledgeSpec:
        """The active spec; the built-in default (version 1) when ``kb_specs`` is empty.

        A stored spec that no longer validates is not used: the default stands
        in and the problem is logged, so a bad row cannot stop the pass.
        """
        async with self._engine.connect() as conn:
            row = (await conn.execute(text(
                "SELECT version, spec FROM kb_specs WHERE active"
            ))).fetchone()
        if row is None:
            return default_spec()
        body = row[1]
        if isinstance(body, str):
            body = json.loads(body)
        try:
            return KnowledgeSpec.from_dict(body, int(row[0]))
        except SpecError as exc:
            log.warning("knowledge_spec.stored_spec_invalid", version=int(row[0]), error=str(exc))
            return default_spec()

    async def save_spec(
        self,
        spec: Mapping[str, Any],
        created_by: Optional[str] = None,
        note: Optional[str] = None,
        activate: bool = True,
    ) -> int:
        """Validate ``spec`` (a body, without version), store it as the next
        version and, by default, make it the one active spec. Returns the version.

        The first stored version is 2: version 1 is the built-in default, and a
        stored row never reuses its number.
        """
        KnowledgeSpec.from_dict(spec, DEFAULT_SPEC_VERSION)  # raises SpecError
        async with self._engine.begin() as conn:
            if activate:
                await conn.execute(text("UPDATE kb_specs SET active = false WHERE active"))
            version = (await conn.execute(
                text("""
                INSERT INTO kb_specs (version, spec, created_by, note, active)
                SELECT GREATEST(COALESCE(MAX(version), 0), CAST(:base AS integer)) + 1,
                       CAST(:spec AS jsonb), :by, :note, :active
                FROM kb_specs
                RETURNING version
                """),
                {"base": DEFAULT_SPEC_VERSION, "spec": json.dumps(spec),
                 "by": created_by, "note": note, "active": activate},
            )).scalar_one()
        return int(version)

    # ------------------------------------------------------------------
    # Dispositions
    # ------------------------------------------------------------------

    async def claim_undecided_pages(
        self, limit: int, min_age_s: float = 0.0, skip: Optional[set[str]] = None
    ) -> "PageBatch":
        """Pages of documents that are not held and have no disposition yet.

        Same pick as the embed claim: ``FOR NO KEY UPDATE ... SKIP LOCKED`` in
        a short transaction that has committed before the caller asks the
        endpoint anything. ``min_age_s`` leaves a document alone for that long
        after it was created, so its audience is settled first.
        """
        async with self._pick_lock:
            exclude = sorted(self._inflight | set(skip or ()))
            async with self._engine.connect() as conn:
                async with conn.begin():
                    result = await conn.execute(
                        text("""
                        SELECT p.page_id, p.text, md5(p.text) AS seen, p.meta_json AS meta,
                               d.title, d.source_uri
                        FROM pages p
                        JOIN docs d ON d.doc_id = p.doc_id
                        WHERE d.held_at IS NULL
                          AND d.created_at <= now() - make_interval(secs => CAST(:min_age AS double precision))
                          AND NOT EXISTS (SELECT 1 FROM page_dispositions x
                                          WHERE x.page_id = p.page_id)
                          AND p.page_id <> ALL(CAST(:skip AS uuid[]))
                        ORDER BY p.page_id
                        LIMIT :limit
                        FOR NO KEY UPDATE OF p SKIP LOCKED
                        """),
                        {"limit": limit, "min_age": float(min_age_s), "skip": exclude},
                    )
                    rows = []
                    for r in result.fetchall():
                        row = r._asdict()
                        row["page_id"] = str(row["page_id"])
                        if isinstance(row.get("meta"), str):
                            row["meta"] = json.loads(row["meta"])
                        rows.append(row)
            ids = {r["page_id"] for r in rows}
            self._inflight |= ids
        return PageBatch(rows, self, ids)

    async def write_disposition(
        self,
        page_id: str,
        seen: str,
        spec_version: int,
        would_embed: bool,
        would_graph: bool,
        held: bool,
        answers: Mapping[str, Any],
        model: Optional[str],
        review_after: Optional[datetime],
    ) -> bool:
        """Record a shadow disposition. True when a row was written.

        Written only if the page's text is still what was read (``seen``, an
        md5) and the page has no disposition yet: an edited page is picked
        again, a decided one is not overwritten.
        """
        async with self._engine.begin() as conn:
            res = await conn.execute(
                text("""
                INSERT INTO page_dispositions
                    (page_id, spec_version, would_embed, would_graph, held, answers,
                     model, decided_at, review_after, mode)
                SELECT p.page_id, :ver, :we, :wg, :held, CAST(:answers AS jsonb),
                       :model, now(), CAST(:review_after AS timestamptz), 'shadow'
                FROM pages p
                WHERE p.page_id = CAST(:page_id AS uuid) AND md5(p.text) = :seen
                ON CONFLICT (page_id) DO NOTHING
                """),
                {"page_id": page_id, "seen": seen, "ver": spec_version,
                 "we": would_embed, "wg": would_graph, "held": held,
                 "answers": json.dumps(answers, allow_nan=False), "model": model,
                 "review_after": review_after},
            )
            return (res.rowcount or 0) > 0

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    async def disposition_summary(self) -> dict[str, Any]:
        """What the recorded shadow dispositions say, and what enforcing them would cost.

        ``held_retrieved`` is the safety number: pages the spec would have
        held back that retrieval actually returned (``held_retrieved_count``
        is how many retrievals they account for). ``unembedded_retrieved`` is
        the same for pages that would not be embedded, held or not.
        """
        async with self._engine.connect() as conn:
            d = (await conn.execute(text("""
                SELECT
                  count(*) AS decided,
                  count(*) FILTER (WHERE would_embed AND would_graph) AS embed_and_graph,
                  count(*) FILTER (WHERE would_embed AND NOT would_graph) AS embed_only,
                  count(*) FILTER (WHERE NOT would_embed AND would_graph) AS graph_only,
                  count(*) FILTER (WHERE held) AS held,
                  count(*) FILTER (WHERE held AND review_after <= now()) AS held_review_due,
                  count(*) FILTER (WHERE NOT would_embed) AS embeds_saved_pages,
                  count(*) FILTER (WHERE NOT would_graph) AS extraction_calls_saved
                FROM page_dispositions
            """))).mappings().one()
            spent = (await conn.execute(text("""
                SELECT
                  count(*) FILTER (WHERE NOT x.would_embed AND pe.page_id IS NOT NULL)
                      AS embeds_already_spent,
                  count(*) FILTER (WHERE NOT x.would_graph AND p.extracted_at IS NOT NULL)
                      AS extractions_already_spent
                FROM page_dispositions x
                JOIN pages p ON p.page_id = x.page_id
                LEFT JOIN page_embeddings pe
                       ON pe.page_id = x.page_id AND pe.embedding IS NOT NULL
            """))).mappings().one()
            chunks = (await conn.execute(text("""
                SELECT count(*) FROM page_chunks c
                JOIN page_dispositions x ON x.page_id = c.page_id
                WHERE NOT x.would_embed
            """))).scalar() or 0
            safety = (await conn.execute(text("""
                SELECT
                  count(*) FILTER (WHERE x.held AND u.page_id IS NOT NULL) AS held_retrieved,
                  COALESCE(sum(u.retrieved_count) FILTER (WHERE x.held), 0) AS held_retrieved_count,
                  count(*) FILTER (WHERE NOT x.would_embed AND u.page_id IS NOT NULL)
                      AS unembedded_retrieved,
                  count(*) FILTER (WHERE u.page_id IS NOT NULL) AS decided_retrieved
                FROM page_dispositions x LEFT JOIN page_usage u ON u.page_id = x.page_id
            """))).mappings().one()
            undecided = (await conn.execute(text("""
                SELECT count(*) FROM pages p JOIN docs d ON d.doc_id = p.doc_id
                WHERE d.held_at IS NULL AND NOT EXISTS (
                    SELECT 1 FROM page_dispositions x WHERE x.page_id = p.page_id)
            """))).scalar() or 0
            versions = (await conn.execute(text(
                "SELECT spec_version, count(*) FROM page_dispositions "
                "GROUP BY spec_version ORDER BY spec_version"
            ))).fetchall()
        out: dict[str, Any] = {k: int(v or 0) for k, v in d.items()}
        out["undecided"] = int(undecided)
        out["embed_calls_saved"] = {
            "pages": out.pop("embeds_saved_pages"),
            "chunks": int(chunks),
            "pages_already_embedded": int(spent["embeds_already_spent"] or 0),
        }
        out["extraction_calls_saved"] = {
            "pages": out.pop("extraction_calls_saved"),
            "pages_already_extracted": int(spent["extractions_already_spent"] or 0),
        }
        out["safety"] = {k: int(v or 0) for k, v in safety.items()}
        out["spec_versions"] = {int(v): int(n) for v, n in versions}
        out["mode"] = "shadow"
        return out


class PageBatch:
    """Claimed pages; ``release`` frees them for the next pick."""

    def __init__(self, rows: list[dict], store: KnowledgeStore, ids: set[str]) -> None:
        self.rows = rows
        self._store = store
        self._ids = ids

    def release(self) -> None:
        self._store._inflight -= self._ids
        self._ids = set()
