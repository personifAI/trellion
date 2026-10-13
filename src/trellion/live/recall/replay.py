# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""TraversalReplayEngine — System 1 fast recall of past traversals.

Search the persona's traversal FAISS index for queries similar to the
current one, then fetch the matching ``trellion_traversal_records`` rows.

Lighter than the persona service's full replay engine (which couples to a
seed strategy and a graph narrower that we don't depend on here). Returns
the raw matches; callers can decide what to do with them — extract
seed entities, boost edges, or just surface the recalled query as
context.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from .pg_index import PersonaVectorIndex

logger = structlog.get_logger(__name__)

# Min past outcome to consider as a useful replay
MIN_OUTCOME_SCORE = 0.5


@dataclass
class ReplayMatch:
    """One past traversal that matched the current query."""
    traversal_id: str
    query: str
    intent: str | None
    outcome_score: float
    traversal_path: list[dict[str, Any]]
    text_similarity: float
    combined_score: float
    exploration_rate: float | None = None
    uncertainty_fingerprint: dict[str, Any] | None = None


class TraversalReplayEngine:
    """Reads the persona's traversal frames (pgvector) + Postgres records.

    ``faiss_base_path`` is accepted and ignored for one release — wiring
    code still passes it (proposal 28 Part 1 retired the FAISS files).
    """

    def __init__(self, faiss_base_path: str | None, engine: AsyncEngine):
        self._engine = engine

    async def find_similar(
        self,
        persona_id: str,
        query_embedding: list[float],
        *,
        intent: str | None = None,
        k: int = 10,
        org: str | None = None,
    ) -> list[ReplayMatch]:
        """Return up to k past traversals ranked by combined score.

        Combined score = 60% cosine similarity + 40% prominence
        (filtered to outcome_score >= MIN_OUTCOME_SCORE). When ``intent``
        is provided, intent-matching traversals get a small bonus.

        ``org`` bounds the hydrated records to that tenant (PR5b-3): the
        vector search returns candidate frame_ids, but the Postgres row
        fetch drops any that aren't in the caller's org. ``None`` =
        single-org (no filter).
        """
        try:
            hits = await PersonaVectorIndex(self._engine, persona_id).search(
                index_name="traversals",
                query_embedding=query_embedding,
                k=k * 2,
            )
        except Exception:
            logger.warning("traversal_replay.vector_search_failed", exc_info=True)
            return []

        if not hits:
            return []

        frame_ids = [h.frame_id for h in hits]

        # Fetch records by frame_id, filtering on outcome
        async with self._engine.connect() as conn:
            rows = (await conn.execute(
                text(
                    "SELECT tr.id::text, tr.query_text, tr.intent, tr.outcome_score, "
                    "       tr.traversal_path, tr.exploration_rate, tr.uncertainty_fingerprint, "
                    "       tr.video_frame_id::text "
                    "FROM trellion_traversal_records tr "
                    "WHERE tr.persona_id = :p "
                    "  AND tr.video_frame_id::text = ANY(:fids) "
                    "  AND tr.outcome_score >= :min_out "
                    # PR5b-3 org boundary: NULL org param = single-org (no filter).
                    "  AND (:org::text IS NULL OR tr.org_id = :org)"
                ),
                {"p": persona_id, "fids": frame_ids, "min_out": MIN_OUTCOME_SCORE, "org": org},
            )).fetchall()

        rows_by_frame: dict[str, dict[str, Any]] = {
            r[7]: {
                "id": r[0], "query": r[1], "intent": r[2], "outcome": r[3],
                "path": r[4], "rate": r[5], "uncertainty": r[6],
            }
            for r in rows
        }

        matches: list[ReplayMatch] = []
        for hit in hits:
            row = rows_by_frame.get(hit.frame_id)
            if not row:
                continue

            # Intent bonus — favors past traversals matching the same intent
            intent_bonus = 1.0 if (intent and row["intent"] == intent) else 0.0
            combined = hit.combined_score + 0.1 * intent_bonus

            matches.append(ReplayMatch(
                traversal_id=row["id"],
                query=row["query"],
                intent=row["intent"],
                outcome_score=row["outcome"],
                traversal_path=row["path"] or [],
                text_similarity=hit.similarity,
                combined_score=combined,
                exploration_rate=row["rate"],
                uncertainty_fingerprint=row["uncertainty"],
            ))

        matches.sort(key=lambda m: m.combined_score, reverse=True)
        return matches[:k]
