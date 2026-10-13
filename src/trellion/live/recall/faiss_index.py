# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""Per-persona FAISS index for System 1 fast recall.

Adapted from the persona service's pattern. Each persona has one or more named FAISS
indexes (typically ``conversations`` and ``traversals``) persisted to
local disk. The Postgres ``trellion_episodes`` table holds the
metadata (prominence, access stats, tombstone flag); FAISS only holds
the embedding vectors and a sidecar ``_meta.json`` mapping vector ids
to ``frame_id``.

Search results are scored as 60% query similarity + 40% prominence —
prominence is recomputed periodically by ``RecallLifecycleManager``.
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)

# Read embedding dim from env at import time (matches trellion.db.models)
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "768"))

SIMILARITY_WEIGHT = 0.6
PROMINENCE_WEIGHT = 0.4


@dataclass
class SearchResult:
    """A single FAISS search hit — scored with prominence weighting."""
    frame_id: str
    similarity: float
    prominence: float
    combined_score: float
    record_type: str


class PersonaFAISSIndex:
    """Per-persona FAISS index manager (thread-safe).

    Indexes live as files at ``<base_path>/<persona_id>/<index_name>.index``
    with a sidecar ``<index_name>_meta.json`` listing per-vector metadata
    (frame_id, prominence, record_type, tombstoned flag).

    LRU eviction: callers can ``evict()`` when memory pressure is high;
    next ``add`` / ``search`` re-loads from disk.
    """

    def __init__(self, base_path: str, persona_id: str, *, dim: int = EMBEDDING_DIM):
        self._base_path = Path(base_path)
        self._persona_id = persona_id
        self._local_dir = self._base_path / persona_id / "faiss"
        self._dim = dim
        self._indexes: dict[str, object] = {}
        self._metadata: dict[str, list[dict]] = {}
        self._loaded: set[str] = set()
        self._lock = threading.Lock()
        self._last_access = time.monotonic()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _load_index(self, index_name: str) -> None:
        """Load an index from disk into RAM. Creates an empty index if missing."""
        if index_name in self._loaded:
            self._last_access = time.monotonic()
            return

        try:
            import faiss
        except ImportError as exc:
            raise ImportError(
                "faiss-cpu required for video memory v2: pip install trellion[video] "
                "or pip install faiss-cpu numpy"
            ) from exc

        index_path = self._local_dir / f"{index_name}.index"
        meta_path = self._local_dir / f"{index_name}_meta.json"

        if index_path.exists() and meta_path.exists():
            self._indexes[index_name] = faiss.read_index(str(index_path))
            self._metadata[index_name] = json.loads(meta_path.read_text())
            logger.debug(
                "faiss.loaded",
                persona=self._persona_id,
                index=index_name,
                vectors=self._indexes[index_name].ntotal,
            )
        else:
            self._indexes[index_name] = faiss.IndexFlatIP(self._dim)
            self._metadata[index_name] = []

        self._loaded.add(index_name)
        self._last_access = time.monotonic()

    def _save_index(self, index_name: str) -> None:
        """Atomic-rename persist to disk."""
        try:
            import faiss
        except ImportError:
            return

        if index_name not in self._indexes:
            return

        self._local_dir.mkdir(parents=True, exist_ok=True)
        index_path = self._local_dir / f"{index_name}.index"
        meta_path = self._local_dir / f"{index_name}_meta.json"

        tmp_index = index_path.with_suffix(".index.tmp")
        tmp_meta = meta_path.with_suffix(".json.tmp")

        faiss.write_index(self._indexes[index_name], str(tmp_index))
        tmp_meta.write_text(json.dumps(self._metadata[index_name]))

        tmp_index.replace(index_path)
        tmp_meta.replace(meta_path)

    def evict(self) -> None:
        """Drop in-memory state. Next access reloads from disk."""
        with self._lock:
            self._indexes.clear()
            self._metadata.clear()
            self._loaded.clear()

    # ------------------------------------------------------------------
    # Read/write
    # ------------------------------------------------------------------

    def add(
        self,
        index_name: str,
        embedding: list[float],
        frame_id: str,
        prominence: float = 0.5,
        record_type: str = "conversation",
    ) -> None:
        try:
            import numpy as np
        except ImportError as exc:
            raise ImportError("numpy required for video memory v2") from exc

        with self._lock:
            self._load_index(index_name)
            vec = np.asarray([embedding], dtype=np.float32)
            norm = np.linalg.norm(vec, axis=1, keepdims=True)
            if float(norm) > 0:
                vec = vec / norm
            self._indexes[index_name].add(vec)
            self._metadata[index_name].append({
                "frame_id": frame_id,
                "prominence": prominence,
                "record_type": record_type,
            })
            self._save_index(index_name)

    def search(
        self,
        index_name: str,
        query_embedding: list[float],
        k: int = 10,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        try:
            import numpy as np
        except ImportError:
            return []

        with self._lock:
            self._load_index(index_name)
            index = self._indexes.get(index_name)
            meta = self._metadata.get(index_name, [])
            if index is None or index.ntotal == 0:
                return []

            vec = np.asarray([query_embedding], dtype=np.float32)
            norm = np.linalg.norm(vec, axis=1, keepdims=True)
            if float(norm) > 0:
                vec = vec / norm

            actual_k = min(k * 2, index.ntotal)
            distances, indices = index.search(vec, actual_k)

            results: list[SearchResult] = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx < 0 or idx >= len(meta):
                    continue
                entry = meta[idx]
                if entry.get("tombstoned"):
                    continue

                similarity = float(dist)
                prominence = float(entry.get("prominence", 0.5))
                combined = SIMILARITY_WEIGHT * similarity + PROMINENCE_WEIGHT * prominence
                if combined < min_score:
                    continue

                results.append(SearchResult(
                    frame_id=entry["frame_id"],
                    similarity=similarity,
                    prominence=prominence,
                    combined_score=combined,
                    record_type=entry.get("record_type", "conversation"),
                ))

            results.sort(key=lambda r: r.combined_score, reverse=True)
            return results[:k]

    def update_prominence(self, index_name: str, frame_id: str, new_prominence: float) -> None:
        with self._lock:
            self._load_index(index_name)
            for entry in self._metadata.get(index_name, []):
                if entry["frame_id"] == frame_id:
                    entry["prominence"] = new_prominence
                    self._save_index(index_name)
                    return

    def mark_tombstoned(self, index_name: str, frame_id: str) -> None:
        with self._lock:
            self._load_index(index_name)
            for entry in self._metadata.get(index_name, []):
                if entry["frame_id"] == frame_id:
                    entry["tombstoned"] = True
                    self._save_index(index_name)
                    return

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def last_access(self) -> float:
        return self._last_access

    @property
    def total_vectors(self) -> int:
        return sum(idx.ntotal for idx in self._indexes.values())

    @property
    def memory_bytes(self) -> int:
        return self.total_vectors * self._dim * 4
