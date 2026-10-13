# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""EpisodicRecallManager — opt-in facade for the new Postgres+FAISS path.

Coexists with the existing v1 ``trellion.live.video.manager.VideoMemoryManager``;
no v1 code is touched. Use this when starting greenfield personas; v1
remains the default for installs with existing MP4 archives.

Toggle from the gateway by passing ``use_v2=True`` to the wiring code or
setting ``NOO_VIDEO_MEMORY_V2=1`` in the environment.
"""
from __future__ import annotations

from typing import Any

import structlog
from sqlalchemy.ext.asyncio import AsyncEngine

from ..video.paths import video_memory_dir
from .archiver import TraversalArchiver
from .faiss_index import SearchResult
from .pg_index import PersonaVectorIndex
from .lifecycle import RecallLifecycleManager
from .replay import ReplayMatch, TraversalReplayEngine

logger = structlog.get_logger(__name__)


class EpisodicRecallManager:
    """Slim facade wiring the v2 building blocks for one persona-per-process flow.

    Public surface intentionally smaller than v1's manager — no MP4
    backlog, no archive_conversations / archive_all (those wrote MP4s),
    no startup/shutdown lifecycle (FAISS indexes are loaded lazily).
    Designed to be wired alongside v1 during the transition.
    """

    def __init__(
        self,
        engine: AsyncEngine,
        *,
        faiss_base_path: str | None = None,
        min_outcome_score: float = 0.6,
    ):
        self._engine = engine
        self._faiss_base_path = video_memory_dir(faiss_base_path)
        self.archiver = TraversalArchiver(
            self._faiss_base_path, engine, min_outcome_score=min_outcome_score
        )
        self.replay = TraversalReplayEngine(self._faiss_base_path, engine)
        self.lifecycle = RecallLifecycleManager(self._faiss_base_path, engine)

    # ------------------------------------------------------------------
    # Search — pgvector over the persona's frames
    # ------------------------------------------------------------------

    async def search(
        self,
        persona_id: str,
        query_embedding: list[float],
        *,
        index_name: str | None = None,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        """Vector search over frames, returning typed hits.

        Default searches ALL of the persona's indexes — the historical
        "conversations" default addressed an index nothing ever wrote.
        """
        idx = PersonaVectorIndex(self._engine, persona_id)
        return await idx.search(index_name=index_name, query_embedding=query_embedding,
                                k=top_k, min_score=min_score)

    async def search_traversals(
        self,
        persona_id: str,
        query_embedding: list[float],
        *,
        intent: str | None = None,
        top_k: int = 10,
    ) -> list[ReplayMatch]:
        return await self.replay.find_similar(
            persona_id, query_embedding, intent=intent, k=top_k
        )

    # ------------------------------------------------------------------
    # Status — for the health router + diagnostic UI
    # ------------------------------------------------------------------

    async def status(self) -> dict[str, Any]:
        """Per-persona frame count + prominence summary."""
        from sqlalchemy import text
        async with self._engine.connect() as conn:
            rows = (await conn.execute(
                text(
                    "SELECT persona_id, "
                    "       COUNT(*) FILTER (WHERE is_tombstoned = FALSE) AS active, "
                    "       COUNT(*) FILTER (WHERE is_tombstoned = TRUE) AS tombstoned, "
                    "       AVG(prominence_score) FILTER (WHERE is_tombstoned = FALSE) AS avg_prom "
                    "FROM trellion_episodes "
                    "GROUP BY persona_id"
                )
            )).fetchall()
        per_persona = [
            {
                "persona_id": r[0],
                "active_frames": int(r[1]),
                "tombstoned_frames": int(r[2]),
                "avg_prominence": float(r[3]) if r[3] is not None else 0.0,
            }
            for r in rows
        ]
        return {
            "version": 2,
            "backend": "postgres",
            
            "personas": per_persona,
        }
