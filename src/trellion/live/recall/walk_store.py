# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""WalkStore: the read/lifecycle side of recallable walks (proposal 57, 1a).

``TraversalArchiver.capture`` writes a walk; this reads candidates back and
moves them through their lifecycle. It is deliberately FILTER AND ORDER ONLY:
the audience rules (is every step readable by this asker?), the usefulness
score and any further gates belong to the caller and run on what this returns.
Nothing here decides whether a walk is *good*, only which rows are the asker's
own, active, unexpired, and nearest to the query.

Every method that touches a user's rows takes ``user_id`` explicitly and
refuses an empty one: the failure mode to avoid is a missing id widening a
query to everybody's rows.
"""
from __future__ import annotations

from typing import Any, Sequence

import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from .archiver import WALK_KINDS, _vec_literal

logger = structlog.get_logger(__name__)

#: Columns returned for a candidate (everything but the raw embedding).
_CANDIDATE_COLS = (
    "id::text AS id, persona_id, query_text, intent, traversal_path, "
    "outcome_score, kind, user_id, thread_id::text AS thread_id, turn_index, "
    "run_id, provenance, evidence, usefulness, status, last_validated_at, "
    "last_recalled_at, recall_count, expires_at, created_at, "
    "embedding_model, embedding_dim"
)


def _require_user(user_id: str | None) -> str:
    if not user_id or not str(user_id).strip():
        raise ValueError("user_id is required")
    return str(user_id)


def _check_kind(kind: str | None) -> None:
    if kind is not None and kind not in WALK_KINDS:
        raise ValueError(f"unknown kind {kind!r}")


class WalkStore:
    """Postgres store operations over ``trellion_traversal_records``."""

    def __init__(self, engine: AsyncEngine):
        self._engine = engine

    async def recall_candidates(
        self,
        user_id: str | None,
        query_embedding: Sequence[float],
        *,
        kind: str = "thread",
        top_k: int = 10,
        persona_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Nearest ``status='active'``, unexpired walks of one owner and kind.

        ``kind='thread'`` requires ``user_id`` and returns only that user's
        rows. ``kind='simulated'`` walks are never attributed, so pass
        ``user_id=None`` to read them (``user_id IS NULL``); a set ``user_id``
        with ``kind='simulated'`` matches that user's rows, normally none.

        Rows without an embedding are skipped (unrecallable). Each dict has
        the columns above plus ``similarity`` = ``1 - cosine distance``,
        nearest first. No score threshold: the caller gates.
        """
        _check_kind(kind)
        if kind == "thread":
            _require_user(user_id)
        if not query_embedding:
            raise ValueError("query_embedding is required")
        if top_k < 1:
            return []
        owner = "user_id = :uid" if user_id else "user_id IS NULL"
        persona = "AND persona_id = :persona " if persona_id else ""
        params: dict[str, Any] = {
            "vec": _vec_literal(list(query_embedding)),
            "kind": kind,
            "k": int(top_k),
        }
        if user_id:
            params["uid"] = str(user_id)
        if persona_id:
            params["persona"] = persona_id
        async with self._engine.connect() as conn:
            rows = (
                await conn.execute(
                    text(
                        f"SELECT {_CANDIDATE_COLS}, "
                        "       1 - (query_embedding <=> CAST(:vec AS vector)) AS similarity "
                        "FROM trellion_traversal_records "
                        f"WHERE {owner} AND kind = :kind AND status = 'active' "
                        f"  {persona}"
                        "  AND query_embedding IS NOT NULL "
                        "  AND (expires_at IS NULL OR expires_at > NOW()) "
                        "ORDER BY query_embedding <=> CAST(:vec AS vector) "
                        "LIMIT :k"
                    ),
                    params,
                )
            ).mappings().all()
        return [dict(r) for r in rows]

    async def touch_recalled(self, walk_ids: Sequence[str], *, user_id: str | None = None) -> int:
        """Bump ``last_recalled_at`` and ``recall_count`` for walks that were
        served. ``user_id`` (optional) restricts to that owner's rows. Returns
        the rows updated."""
        ids = [str(i) for i in walk_ids]
        if not ids:
            return 0
        owner = "AND user_id = :uid " if user_id else ""
        params: dict[str, Any] = {"ids": ids}
        if user_id:
            params["uid"] = str(user_id)
        async with self._engine.begin() as conn:
            res = await conn.execute(
                text(
                    "UPDATE trellion_traversal_records "
                    "SET last_recalled_at = NOW(), recall_count = recall_count + 1 "
                    f"WHERE id = ANY(CAST(:ids AS uuid[])) {owner}"
                ),
                params,
            )
        return res.rowcount or 0

    async def mark_stale(self, walk_ids: Sequence[str], *, user_id: str | None = None) -> int:
        """``active`` -> ``stale`` (failed revalidation). Only active rows
        move: a ``staged`` row stays staged. Returns rows changed."""
        ids = [str(i) for i in walk_ids]
        if not ids:
            return 0
        owner = "AND user_id = :uid " if user_id else ""
        params: dict[str, Any] = {"ids": ids}
        if user_id:
            params["uid"] = str(user_id)
        async with self._engine.begin() as conn:
            res = await conn.execute(
                text(
                    "UPDATE trellion_traversal_records "
                    "SET status = 'stale', updated_at = NOW() "
                    f"WHERE id = ANY(CAST(:ids AS uuid[])) AND status = 'active' {owner}"
                ),
                params,
            )
        return res.rowcount or 0

    async def stage_for_user(self, user_id: str, *, kind: str | None = None) -> int:
        """Set a user's walks to ``status='staged'`` (out of recall, still
        restorable). Returns rows changed. ``kind`` narrows it."""
        uid = _require_user(user_id)
        _check_kind(kind)
        kind_sql = "AND kind = :kind " if kind else ""
        params: dict[str, Any] = {"uid": uid}
        if kind:
            params["kind"] = kind
        async with self._engine.begin() as conn:
            res = await conn.execute(
                text(
                    "UPDATE trellion_traversal_records "
                    "SET status = 'staged', updated_at = NOW() "
                    f"WHERE user_id = :uid AND status <> 'staged' {kind_sql}"
                ),
                params,
            )
        return res.rowcount or 0

    async def delete_for_user(self, user_id: str, *, kind: str | None = None) -> dict[str, int]:
        """Hard-delete a user's walks and the episode rows they point at.

        Returns ``{"traversals": n, "episodes": m}``. Episodes owned by the
        user (``trellion_episodes.user_id``) go too when no ``kind`` narrows
        the delete. One transaction.
        """
        uid = _require_user(user_id)
        _check_kind(kind)
        kind_sql = "AND kind = :kind " if kind else ""
        params: dict[str, Any] = {"uid": uid}
        if kind:
            params["kind"] = kind
        async with self._engine.begin() as conn:
            gone = (
                await conn.execute(
                    text(
                        "DELETE FROM trellion_traversal_records "
                        f"WHERE user_id = :uid {kind_sql}"
                        "RETURNING video_frame_id"
                    ),
                    params,
                )
            ).all()
            frame_ids = [str(r[0]) for r in gone if r[0] is not None]
            episodes = 0
            if frame_ids:
                res = await conn.execute(
                    text("DELETE FROM trellion_episodes WHERE id = ANY(CAST(:ids AS uuid[]))"),
                    {"ids": frame_ids},
                )
                episodes += res.rowcount or 0
            if kind is None:
                res = await conn.execute(
                    text("DELETE FROM trellion_episodes WHERE user_id = :uid"), {"uid": uid}
                )
                episodes += res.rowcount or 0
        return {"traversals": len(gone), "episodes": episodes}
