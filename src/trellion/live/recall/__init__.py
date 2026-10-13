# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""trellion.live.recall — episodic recall (System 1 fast associative memory).

Renamed from ``video_v2`` (2026-08-29, in place — dev-only environment):
"video memory" described the retired v1 MP4+QR transport; what exists is
a Postgres+pgvector store of EPISODES — archived experiences (today:
proven graph traversals) recalled by similarity + prominence. Taxonomy:
the knowledge graph is semantic memory, this is episodic memory.

Reimplements the System 1 storage tier per the persona service's pattern:
- All structured data in Postgres (``trellion_episodes`` +
  ``trellion_traversal_records``)
- Only FAISS index files on disk (one ``.index`` + ``_meta.json`` per
  persona × index_name, written via atomic temp-file rename)

Drops the MP4 + QR + ffmpeg dependencies that v1 pulled in. Keeps the
prominence-based three-tier lifecycle (S1 ≥0.3 / elevated 0.05–0.3 /
pruned <0.05 + >90d).

The v1 manager at ``trellion.live.video.manager`` is unchanged for
backward-compat; v2 is an additive opt-in path. Toggle via constructor
arg or env var ``NOO_VIDEO_MEMORY_V2=1``. See proposal 06 §7(e.1).

Public API:
    from trellion.live.recall import (
        PersonaFAISSIndex, TraversalArchiver, TraversalReplayEngine,
        RecallLifecycleManager, EpisodicRecallManager,
    )
"""
from __future__ import annotations

from .archiver import TraversalArchiver
from .faiss_index import PersonaFAISSIndex, SearchResult  # FAISS class: legacy, unwired
from .pg_index import PersonaVectorIndex, backfill_frame_embeddings
from .lifecycle import RecallLifecycleManager
from .manager import EpisodicRecallManager
from .walk_store import WalkStore
from .replay import ReplayMatch, TraversalReplayEngine

__all__ = [
    "PersonaFAISSIndex",
    "PersonaVectorIndex",
    "backfill_frame_embeddings",
    "ReplayMatch",
    "SearchResult",
    "TraversalArchiver",
    "TraversalReplayEngine",
    "WalkStore",
    "RecallLifecycleManager",
    "EpisodicRecallManager",
]
