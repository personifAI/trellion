# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""Compatibility shim — TraversalArchiver backed by the v2 Postgres+FAISS path.

Existing callers (the add-on retrieval feedback loop,
``heartbeat.capture_traversal``) construct ``TraversalSnapshot`` instances
and pass them to ``.capture(snapshot)``. This shim accepts that
signature and unpacks into ``recall.TraversalArchiver.capture(...)``
which writes to ``trellion_episodes`` + ``trellion_traversal_records``
plus the persona FAISS index. No MP4, no QR, no ffmpeg.
"""
from __future__ import annotations

from typing import Any

import structlog

from .paths import video_memory_dir
from .types import TraversalSnapshot

logger = structlog.get_logger(__name__)


class TraversalArchiver:
    """Thin wrapper preserving the v1 ``capture(snapshot)`` API contract."""

    def __init__(
        self,
        embedding_url: str | None = None,
        engine: Any | None = None,
        faiss_base_path: str | None = None,
        min_outcome_score: float = 0.6,
    ) -> None:
        self._embedding_url = (embedding_url or "").rstrip("/")
        self._engine = engine
        self._faiss_base_path = faiss_base_path
        self._min_outcome = min_outcome_score
        self._v2_archiver = None  # built lazily once engine is wired

    def _resolve_engine(self):
        if self._engine is not None:
            return self._engine
        try:
            from trellion.db.engine import get_async_engine
            self._engine = get_async_engine()
            return self._engine
        except Exception:
            return None

    def _get_v2(self):
        if self._v2_archiver is not None:
            return self._v2_archiver
        engine = self._resolve_engine()
        if engine is None:
            return None
        from trellion.live.recall import TraversalArchiver as V2Archiver
        self._v2_archiver = V2Archiver(
            video_memory_dir(self._faiss_base_path),
            engine,
            min_outcome_score=self._min_outcome,
        )
        return self._v2_archiver

    def capture(self, snapshot: TraversalSnapshot) -> dict[str, Any] | None:
        """Sync entry point — schedules an async write to Postgres + FAISS.

        Some callers (the add-on self-practice loop, heartbeat)
        invoke us synchronously from a context that already has an event
        loop. Schedule the write as a fire-and-forget task and return
        immediately with a "scheduled" marker.
        """
        v2 = self._get_v2()
        if v2 is None:
            logger.debug("traversal_archiver.no_engine_skipped")
            return {"archived": False, "reason": "trellion engine not initialised"}

        try:
            import asyncio
            loop = asyncio.get_event_loop()
            coro = v2.capture(
                persona_id=snapshot.persona_id or "default",
                query_text=snapshot.query,
                intent=snapshot.query_intent,
                traversal_path=snapshot.to_path_json(),
                outcome_score=snapshot.outcome_score,
                query_embedding=snapshot.query_embedding,
                exploration_rate=snapshot.exploration_rate,
                uncertainty_fingerprint=snapshot.uncertainty_fingerprint,
                user_id=getattr(snapshot, "user_id", None),
                kind=getattr(snapshot, "kind", None) or "simulated",
                thread_id=getattr(snapshot, "thread_id", None),
                turn_index=getattr(snapshot, "turn_index", None),
                run_id=getattr(snapshot, "run_id", None),
                provenance=getattr(snapshot, "provenance", None),
                evidence=getattr(snapshot, "evidence", None),
                usefulness=getattr(snapshot, "usefulness", None),
            )
            if loop.is_running():
                loop.create_task(coro)
                return {"archived": True, "scheduled": True}
            return loop.run_until_complete(coro)
        except RuntimeError:
            try:
                import asyncio
                return asyncio.run(coro)
            except Exception:
                logger.warning("traversal_archiver.capture_failed", exc_info=True)
                return None

    async def get_unarchived_count(self) -> int:
        """v2 has no MP4 backlog — always 0."""
        return 0
