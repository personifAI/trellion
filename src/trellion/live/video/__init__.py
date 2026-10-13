# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""Video memory facade — System 1 fast associative recall.

Backed by Postgres + FAISS as of trellion v0.0.2 (proposal 06 phase 6).
The MP4 + QR codec design has been removed; conversations live in
``noo_entries`` and the FAISS sidecar handles fast similarity recall.

Public API surface preserved for callers (gateway/app.py,
worker_pool.py, video_memory_cli.py, heartbeat.py, the add-on retrieval feedback loop).

For new code, use ``trellion.live.recall`` directly — it has the
slimmer API without the v1 compatibility layer.
"""
from __future__ import annotations

from .config import VideoMemoryConfig
from .manager import SearchResult, VideoMemoryManager
from .traversal_archiver import TraversalArchiver
from .types import TraversalSnapshot, TraversalStep

__all__ = [
    "SearchResult",
    "TraversalArchiver",
    "TraversalSnapshot",
    "TraversalStep",
    "VideoMemoryConfig",
    "VideoMemoryManager",
]

# Legacy submodule re-exports — `from trellion.live.video.index import
# SearchResult` worked in the v1 layout. The MP4 index module is gone
# but the SearchResult class moved to manager; re-export under the old
# path so existing tests/imports keep resolving.
import sys as _sys
import types as _types

_index_mod = _types.ModuleType("trellion.live.video.index")
_index_mod.SearchResult = SearchResult  # type: ignore[attr-defined]
_sys.modules[__name__ + ".index"] = _index_mod
del _sys, _types
