# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""RecallLifecycleManager — prominence scoring + tier promotion.

Run periodically (default: every 6 hours) per persona. Three actions:

1. Recompute prominence scores on all active frames.
2. Mark frames whose new prominence is below ELEVATE_THRESHOLD as elevated
   (handed off to System 2 — caller's responsibility to actually copy them
   into pgvector; we just flag them).
3. Tombstone frames whose prominence is below PRUNE_THRESHOLD AND age >
   PRUNE_AGE_DAYS.

Prominence formula:
    recency (35%) + quality (25%) + frequency (20%) + sustained_utility (20%)
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine


logger = structlog.get_logger(__name__)

# Prominence component weights
W_RECENCY = 0.35
W_QUALITY = 0.25
W_FREQUENCY = 0.20
W_UTILITY = 0.20

# Recency decay
HALF_LIFE_DAYS = 30.0

# Lifecycle thresholds
KEEP_THRESHOLD = 0.3        # >= this stays in System 1
ELEVATE_THRESHOLD = 0.05    # below this → tombstone if old enough
PRUNE_AGE_DAYS = 90         # min age before pruning low-prominence frames


class RecallLifecycleManager:
    """Per-persona prominence recomputation + retention enforcement."""

    def __init__(self, faiss_base_path: str, engine: AsyncEngine):
        self._faiss_base_path = faiss_base_path
        self._engine = engine

    async def run_lifecycle(self, persona_id: str) -> dict[str, Any]:
        """Run one full lifecycle pass.

        Returns:
            ``{frames_scored, frames_elevated, frames_pruned, avg_prominence}``
        """
        now = datetime.now(timezone.utc)
        frames_scored = 0
        frames_elevated = 0
        frames_pruned = 0
        total_prominence = 0.0

        # 1. Read all active (non-tombstoned) frames
        async with self._engine.connect() as conn:
            rows = (await conn.execute(
                text(
                    "SELECT id::text, index_name, access_count, "
                    "       last_accessed_at, first_accessed_at, "
                    "       avg_search_score, created_at, prominence_score "
                    "FROM trellion_episodes "
                    "WHERE persona_id = :p AND is_tombstoned = FALSE "
                    "ORDER BY created_at"
                ),
                {"p": persona_id},
            )).fetchall()

        if not rows:
            return {
                "frames_scored": 0,
                "frames_elevated": 0,
                "frames_pruned": 0,
                "avg_prominence": 0.0,
            }


        # 2. For each frame: compute new prominence, update DB + FAISS
        async with self._engine.begin() as conn:
            for row in rows:
                frame_id, _idx_name, access_count, last_at, first_at, avg_score, created_at, prev_prom = row

                new_prom = self._compute_prominence(
                    access_count=access_count or 0,
                    last_accessed_at=last_at,
                    first_accessed_at=first_at,
                    avg_search_score=avg_score,
                    created_at=created_at,
                    now=now,
                )
                frames_scored += 1
                total_prominence += new_prom

                await conn.execute(
                    text(
                        "UPDATE trellion_episodes "
                        "SET prominence_score = :p, updated_at = NOW() "
                        "WHERE id = CAST(:id AS uuid)"
                    ),
                    {"p": new_prom, "id": frame_id},
                )

                # Pruning: tombstone if too low + too old
                age_days = (now - created_at).total_seconds() / 86400.0 if created_at else 0
                if new_prom < ELEVATE_THRESHOLD and age_days > PRUNE_AGE_DAYS:
                    await conn.execute(
                        text(
                            "UPDATE trellion_episodes SET is_tombstoned = TRUE, "
                            "    updated_at = NOW() WHERE id = CAST(:id AS uuid)"
                        ),
                        {"id": frame_id},
                    )
                    frames_pruned += 1
                elif new_prom < KEEP_THRESHOLD:
                    # Elevation marker — caller is responsible for actually
                    # copying the embedding into the System 2 pgvector store.
                    # We just track via metadata so a follow-up sweep can find these.
                    await conn.execute(
                        text(
                            "UPDATE trellion_episodes "
                            "SET metadata = jsonb_set(metadata, '{elevated}', 'true'::jsonb) "
                            "WHERE id = CAST(:id AS uuid)"
                        ),
                        {"id": frame_id},
                    )
                    frames_elevated += 1

        result = {
            "frames_scored": frames_scored,
            "frames_elevated": frames_elevated,
            "frames_pruned": frames_pruned,
            "avg_prominence": total_prominence / frames_scored if frames_scored > 0 else 0.0,
        }
        logger.info("recall.lifecycle_complete", persona=persona_id, **result)
        return result

    # ------------------------------------------------------------------
    # Prominence formula
    # ------------------------------------------------------------------

    def _compute_prominence(
        self,
        access_count: int,
        last_accessed_at: datetime | None,
        first_accessed_at: datetime | None,
        avg_search_score: float | None,
        created_at: datetime | None,
        now: datetime,
    ) -> float:
        """Blend recency + quality + frequency + sustained_utility."""
        # Recency: exponential decay since last access
        if last_accessed_at:
            age = (now - last_accessed_at).total_seconds() / 86400.0
            recency = math.exp(-age * math.log(2.0) / HALF_LIFE_DAYS)
        else:
            # Never accessed — fall back to age-since-created decay
            if created_at:
                age = (now - created_at).total_seconds() / 86400.0
                recency = max(0.0, math.exp(-age * math.log(2.0) / HALF_LIFE_DAYS))
            else:
                recency = 0.0

        # Quality: average search score (already 0..1)
        quality = float(avg_search_score) if avg_search_score is not None else 0.5

        # Frequency: log-scaled access count
        frequency = min(1.0, math.log1p(access_count) / math.log1p(50))

        # Sustained utility: high if frame has been useful over a long span
        if first_accessed_at and last_accessed_at:
            span_days = (last_accessed_at - first_accessed_at).total_seconds() / 86400.0
            utility = min(1.0, span_days / 30.0)
        else:
            utility = 0.0

        prominence = (
            W_RECENCY * recency
            + W_QUALITY * quality
            + W_FREQUENCY * frequency
            + W_UTILITY * utility
        )
        return max(0.0, min(1.0, prominence))
