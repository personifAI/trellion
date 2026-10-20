# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""TraversalArchiver — captures proven graph walks for System 1 replay.

When a graph traversal yields a high outcome score, this archiver:

1. Adds the query embedding to the persona's FAISS ``traversals`` index
2. Inserts a metadata row in ``trellion_episodes`` (lifecycle bookkeeping)
3. Inserts the full traversal record in ``trellion_traversal_records``

No MP4 / QR / ffmpeg — all structured data lives in Postgres, only the
FAISS index goes to disk. Adapted from the persona service's pattern but uses
SQLAlchemy AsyncEngine instead of asyncpg pool, matching trellion's
existing conventions.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Any

import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from trellion import embedding_space


logger = structlog.get_logger(__name__)

DEFAULT_MIN_OUTCOME = 0.6

#: ``thread`` = a user's real, attributed walk; ``simulated`` = a walk the
#: system chose to rehearse (never attributed). Mirrors the CHECK constraint
#: added by migration 104.
WALK_KINDS = ("thread", "simulated")


class TraversalArchiver:
    """Writes proven traversals to Postgres + FAISS for fast replay."""

    def __init__(
        self,
        faiss_base_path: str,
        engine: AsyncEngine,
        *,
        min_outcome_score: float = DEFAULT_MIN_OUTCOME,
    ):
        self._faiss_base_path = faiss_base_path
        self._engine = engine
        self._min_outcome = min_outcome_score

    async def capture(
        self,
        persona_id: str,
        query_text: str,
        intent: str | None,
        traversal_path: list[dict[str, Any]],
        outcome_score: float,
        *,
        query_embedding: list[float] | None = None,
        exploration_rate: float | None = None,
        uncertainty_fingerprint: dict[str, Any] | None = None,
        user_id: str | None = None,
        kind: str = "simulated",
        thread_id: str | None = None,
        turn_index: int | None = None,
        run_id: str | None = None,
        provenance: dict[str, Any] | None = None,
        evidence: dict[str, Any] | None = None,
        usefulness: float | None = None,
        expires_at: datetime | None = None,
        embedding_model: str | None = None,
    ) -> dict[str, Any] | None:
        """Archive a traversal if it meets the outcome threshold.

        ``user_id`` is who ASKED: the account name (``alice``, never the
        connection ``alice/laptop`` and never a ``user:alice`` role tag),
        taken by the caller from a verified principal. ``None`` (the default,
        and what every internal / heartbeat walk passes) writes NULL, the
        shared/unattributed record. Audience-scoped recall treats a NULL row
        as nobody's and a set one as that user's own.

        Recallable walks (noomesh proposal 57). Every new argument is optional
        and an old caller gets ``kind='simulated'`` with NULLs:

        - ``kind``: ``'thread'`` (a user's own walk; requires ``user_id``) or
          ``'simulated'``. The episode row for a thread carries no question
          text (``"thread walk, N steps"``): the private query lives only on
          the traversal row.
        - ``thread_id`` (a UUID string, the conversation id), ``turn_index``,
          ``run_id`` (heartbeat run of a simulated walk).
        - ``provenance``: how the walk was produced (tool, request id, seeds);
          never the asker.
        - ``evidence``: the save-time snapshot the recall hard gates re-check,
          ``{"triples": [[src, rel, dst], ...], "doc_ids": [...],
          "newest_evidence_at": iso8601}``. Stored verbatim.
        - ``usefulness`` / ``expires_at``: stored as given.
        - ``embedding_model``: the model that made ``query_embedding``. When
          omitted it is the model the embed response named (carried on the
          vector), else the space's current model. Recorded on BOTH the
          traversal row and its episode row.

        ``user_id`` is also written onto the ``trellion_episodes`` row so
        erasure reaches the question text and embedding held there.

        Returns:
            ``{archived: True, traversal_id, frame_id}`` on success,
            ``{archived: False, reason: str}`` if below threshold or empty,
            ``None`` on internal error.
        """
        if outcome_score < self._min_outcome:
            return {
                "archived": False,
                "reason": f"outcome {outcome_score:.2f} < threshold {self._min_outcome:.2f}",
            }
        if not traversal_path:
            return {"archived": False, "reason": "empty traversal path"}
        if kind not in WALK_KINDS:
            return {"archived": False, "reason": f"unknown kind {kind!r}"}
        if kind == "thread" and not user_id:
            return {"archived": False, "reason": "a thread walk needs a user_id"}
        thread_uuid: str | None = None
        if thread_id:
            try:
                thread_uuid = str(uuid.UUID(str(thread_id)))
            except ValueError:
                return {"archived": False, "reason": "thread_id is not a UUID"}

        traversal_id = str(uuid.uuid4())
        frame_id = str(uuid.uuid4())
        if query_embedding and not embedding_model:
            embedding_model = await embedding_space.write_model(query_embedding)
        emb_dim = len(query_embedding) if query_embedding else None

        try:
            # Postgres writes — single transaction so frame + traversal stay
            # consistent. The frame carries the embedding directly (proposal
            # 28 Part 1): search runs in pgvector, and the old best-effort
            # FAISS side-write (a duplicate of query_embedding below) is gone.
            if kind == "thread":
                # A user's question is private: keep it on the traversal row
                # (user-scoped, erased with the user) and out of the frame.
                content_summary = (
                    f"thread walk, {len(traversal_path)} steps -> outcome {outcome_score:.2f}"
                )
            else:
                content_summary = f"Traversal: {query_text[:100]} -> outcome {outcome_score:.2f}"
            frame_emb_sql = "CAST(:femb AS vector)" if query_embedding else "NULL::vector"
            async with self._engine.begin() as conn:
                frame_params = {
                    "id": frame_id,
                    "persona": persona_id,
                    "summary": content_summary,
                    "prominence": min(1.0, outcome_score),
                    "meta": json.dumps({
                        "traversal_id": traversal_id,
                        "step_count": len(traversal_path),
                        "kind": kind,
                    }),
                    "user_id": user_id or None,
                    "emb_model": embedding_model if query_embedding else None,
                    "emb_dim": emb_dim,
                }
                if query_embedding:
                    frame_params["femb"] = _vec_literal(query_embedding)
                await conn.execute(
                    text(
                        "INSERT INTO trellion_episodes "
                        "(id, persona_id, record_type, index_name, content_summary, "
                        " prominence_score, metadata, embedding, user_id, "
                        " embedding_model, embedding_dim) "
                        "VALUES (CAST(:id AS uuid), :persona, 'traversal', 'traversals', "
                        f"        :summary, :prominence, CAST(:meta AS jsonb), {frame_emb_sql}, "
                        "        :user_id, :emb_model, :emb_dim)"
                    ),
                    frame_params,
                )
                # Build the embedding clause out-of-band: when query_embedding
                # is None we emit a literal NULL rather than binding `:emb`,
                # because `CASE WHEN :emb IS NULL ...` leaves $4's type
                # un-inferable (psycopg.errors.AmbiguousParameter).
                emb_sql = "CAST(:emb AS vector)" if query_embedding else "NULL::vector"
                params = {
                    "id": traversal_id,
                    "persona": persona_id,
                    "query": query_text,
                    "intent": intent,
                    "path": json.dumps(traversal_path),
                    "outcome": outcome_score,
                    "rate": exploration_rate,
                    "uncertainty": json.dumps(uncertainty_fingerprint or {}),
                    "frame_id": frame_id,
                    "user_id": user_id or None,
                    "kind": kind,
                    "thread_id": thread_uuid,
                    "turn_index": turn_index,
                    "run_id": run_id,
                    "provenance": json.dumps(provenance or {}),
                    "evidence": json.dumps(evidence or {}),
                    "usefulness": usefulness,
                    "expires_at": expires_at,
                    "emb_model": embedding_model,
                    "emb_dim": emb_dim,
                }
                if query_embedding:
                    params["emb"] = _vec_literal(query_embedding)
                await conn.execute(
                    text(
                        "INSERT INTO trellion_traversal_records "
                        "(id, persona_id, query_text, query_embedding, intent, "
                        " traversal_path, outcome_score, exploration_rate, "
                        " uncertainty_fingerprint, archived_to_video, video_frame_id, "
                        " user_id, kind, thread_id, turn_index, run_id, provenance, "
                        " evidence, usefulness, expires_at, embedding_model, "
                        " embedding_dim, last_validated_at) "
                        "VALUES (CAST(:id AS uuid), :persona, :query, "
                        f"        {emb_sql}, "
                        "        :intent, "
                        "        CAST(:path AS jsonb), :outcome, :rate, "
                        "        CAST(:uncertainty AS jsonb), TRUE, "
                        "        CAST(:frame_id AS uuid), :user_id, :kind, "
                        "        CAST(:thread_id AS uuid), :turn_index, :run_id, "
                        "        CAST(:provenance AS jsonb), CAST(:evidence AS jsonb), "
                        "        :usefulness, :expires_at, :emb_model, :emb_dim, NOW())"
                    ),
                    params,
                )

            logger.info(
                "traversal_archived",
                persona_id=persona_id,
                outcome=outcome_score,
                steps=len(traversal_path),
                intent=intent,
                kind=kind,
            )
            return {"archived": True, "traversal_id": traversal_id, "frame_id": frame_id}

        except Exception as exc:
            logger.error(
                "traversal_archiver.failed",
                persona_id=persona_id,
                error=str(exc)[:200],
            )
            return None

    async def get_unarchived_count(self) -> int:
        """For health-router compatibility — always 0 in v2 (no MP4 backlog)."""
        return 0


def _vec_literal(values: list[float]) -> str:
    return "[" + ",".join(f"{float(v):.6f}" for v in values) + "]"
