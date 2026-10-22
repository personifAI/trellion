# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""Per-persona vector search over ``trellion_episodes`` (pgvector).

Replaces ``PersonaFAISSIndex`` (gateway proposal 28 Part 1): the FAISS
files held a brute-force flat index plus a sidecar COPY of metadata the
frame rows already carried — two stores the lifecycle manager had to
keep coherent. Here the frame row is the single source: the embedding
lives on it, search is one SQL query computing the same
60% similarity + 40% prominence score, tombstones are a WHERE clause,
and prominence updates need no sync step at all.

Search results keep the exact ``SearchResult`` shape callers already
consume.
"""

from __future__ import annotations


import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from trellion import embedding_space

from .faiss_index import PROMINENCE_WEIGHT, SIMILARITY_WEIGHT, SearchResult

logger = structlog.get_logger(__name__)


def _vec_literal(values: list[float]) -> str:
    return "[" + ",".join(f"{float(v):.6f}" for v in values) + "]"


class PersonaVectorIndex:
    """Async pgvector search/write for one persona's frames."""

    def __init__(self, engine: AsyncEngine, persona_id: str) -> None:
        self._engine = engine
        self._persona_id = persona_id

    async def add(
        self,
        index_name: str,
        embedding: list[float],
        frame_id: str,
        prominence: float = 0.5,
        record_type: str = "conversation",
        content_summary: str = "",
        embedding_model: str | None = None,
    ) -> None:
        """Attach an embedding to a frame, creating the row if needed.

        The traversal archiver writes its frame row (with embedding)
        directly; this upsert exists for writers that only have a vector
        — it keeps the old ``PersonaFAISSIndex.add`` contract of "one
        call registers a searchable frame". The model recorded is
        ``embedding_model``, else the one the vector carries, else the
        space's current model.
        """
        model = embedding_model or await embedding_space.write_model(embedding)
        async with self._engine.begin() as conn:
            await conn.execute(
                text(
                    "INSERT INTO trellion_episodes "
                    "(id, persona_id, record_type, index_name, "
                    " content_summary, prominence_score, embedding, "
                    " embedding_model, embedding_dim) "
                    "VALUES (CAST(:id AS uuid), :persona, :rtype, :iname, "
                    "        :summary, :prom, CAST(:emb AS vector), :model, :dim) "
                    "ON CONFLICT (id) DO UPDATE SET "
                    "  embedding = EXCLUDED.embedding, "
                    "  embedding_model = EXCLUDED.embedding_model, "
                    "  embedding_dim = EXCLUDED.embedding_dim, "
                    "  embedding_next = NULL, embedding_next_model = NULL, "
                    "  prominence_score = EXCLUDED.prominence_score, "
                    "  updated_at = NOW()"
                ),
                {
                    "id": frame_id,
                    "persona": self._persona_id,
                    "rtype": record_type,
                    "iname": index_name,
                    "summary": content_summary or f"{record_type} frame",
                    "prom": prominence,
                    "emb": _vec_literal(embedding),
                    "model": model,
                    "dim": len(embedding),
                },
            )

    async def search(
        self,
        index_name: str | None,
        query_embedding: list[float],
        k: int = 10,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        """Top-k frames by 60% cosine similarity + 40% prominence.

        ``index_name=None`` searches every index the persona has. That is
        the right default for "search video memory": the old hardcoded
        "conversations" index was never populated (archive_conversations
        is a documented no-op — conversations live in noo_entries), so
        callers pinned to it could only ever get [].
        """
        iname_where = "AND index_name = :iname " if index_name else ""
        params: dict = {
            "vec": _vec_literal(query_embedding),
            "vec2": _vec_literal(query_embedding),
            "persona": self._persona_id,
            "simw": SIMILARITY_WEIGHT,
            "promw": PROMINENCE_WEIGHT,
            "k": k,
        }
        if index_name:
            params["iname"] = index_name
        model_sql, model_params = await embedding_space.model_clause(
            "embedding_model", query_vector=query_embedding
        )
        model_where = f"  AND {model_sql} " if model_sql else ""
        params.update(model_params)
        async with self._engine.connect() as conn:
            rows = (
                await conn.execute(
                    text(
                        "SELECT id::text, record_type, "
                        "       1 - (embedding <=> CAST(:vec AS vector)) AS sim, "
                        "       prominence_score "
                        "FROM trellion_episodes "
                        "WHERE persona_id = :persona "
                        f"  {iname_where}"
                        "  AND is_tombstoned = FALSE "
                        "  AND embedding IS NOT NULL "
                        f"{model_where}"
                        "ORDER BY (:simw * (1 - (embedding <=> CAST(:vec2 AS vector))) "
                        "          + :promw * prominence_score) DESC "
                        "LIMIT :k"
                    ),
                    params,
                )
            ).fetchall()

        results: list[SearchResult] = []
        for frame_id, record_type, sim, prominence in rows:
            combined = (
                SIMILARITY_WEIGHT * float(sim)
                + PROMINENCE_WEIGHT * float(prominence)
            )
            if combined < min_score:
                continue
            results.append(
                SearchResult(
                    frame_id=frame_id,
                    similarity=float(sim),
                    prominence=float(prominence),
                    combined_score=combined,
                    record_type=record_type,
                )
            )
        return results


async def backfill_frame_embeddings(engine: AsyncEngine) -> dict:
    """One-time backfill: copy embeddings already in Postgres onto frames.

    The only FAISS index anything ever wrote (``traversals``) was a
    dual-write — the archiver stored the same vector in
    ``trellion_traversal_records.query_embedding``. So the backfill is a
    single UPDATE...FROM, no FAISS reader required. Frames with no
    matching record (or records archived without an embedding) are
    reported, not guessed at.
    """
    async with engine.begin() as conn:
        updated = (
            await conn.execute(
                text(
                    "UPDATE trellion_episodes f "
                    "SET embedding = tr.query_embedding, "
                    "    embedding_model = tr.embedding_model, "
                    "    embedding_dim = tr.embedding_dim, updated_at = NOW() "
                    "FROM trellion_traversal_records tr "
                    "WHERE tr.video_frame_id = f.id "
                    "  AND tr.query_embedding IS NOT NULL "
                    "  AND f.embedding IS NULL"
                )
            )
        ).rowcount
        remaining = (
            await conn.execute(
                text(
                    "SELECT COUNT(*) FROM trellion_episodes "
                    "WHERE embedding IS NULL AND is_tombstoned = FALSE"
                )
            )
        ).scalar_one()
    logger.info(
        "vidmem.backfill_done", updated=updated, still_unembedded=remaining
    )
    return {"updated": int(updated), "still_unembedded": int(remaining)}
