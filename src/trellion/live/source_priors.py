# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""Learned source priors, read from where the tracker persists them.

Per-source rewards are written by something else (an optional add-on) to the
file named by ``RETRIEVAL_TRACKER_PATH``. Queries are served by this package,
so the retriever cannot ask the writer in-process; it reads the file.

This module is the open tier's whole interest in that writer: a read-only,
dependency-free reader. It imports no add-on, so a base install
has nothing to fail, nothing to log per query, and no tier to decide: no file
(or no ``source_rewards`` in it) simply means every prior is the neutral 0.5.

The file is re-read when its mtime changes, checked at most once per
``_RECHECK_SECS``; a half-written or corrupt file keeps the last good values.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import structlog

log = structlog.get_logger(__name__)

NEUTRAL = 0.5
_RECHECK_SECS = 5.0


def default_path() -> Path | None:
    """The add-on's state file named by ``RETRIEVAL_TRACKER_PATH``; None when unset (priors stay neutral)."""
    env = os.environ.get("RETRIEVAL_TRACKER_PATH")
    return Path(env) if env else None


class SourcePriors:
    def __init__(self, path: str | Path | None = None, recheck_secs: float = _RECHECK_SECS) -> None:
        self._path: Path | None = Path(path) if path else default_path()
        self._recheck = recheck_secs
        self._rewards: dict[str, float] = {}
        self._mtime: float | None = None
        self._checked_at: float | None = None
        self._announced = False

    def _refresh(self) -> None:
        now = time.monotonic()
        if self._checked_at is not None and now - self._checked_at < self._recheck:
            return
        self._checked_at = now
        try:
            if self._path is None:
                raise OSError("no state file configured")
            mtime = self._path.stat().st_mtime
        except OSError:
            mtime = None  # no file: neutral priors, quietly
        if mtime is not None and mtime != self._mtime:
            try:
                raw = json.loads(self._path.read_text(encoding="utf-8"))
                rewards = raw.get("source_rewards", {})
                self._rewards = {
                    str(k): float(v) for k, v in rewards.items() if isinstance(v, (int, float))
                }
                self._mtime = mtime
            except (OSError, ValueError, AttributeError):
                # Torn / corrupt write: keep what we had, retry on the next change.
                pass
        if not self._announced:
            self._announced = True
            log.info("source_priors.ready", sources=len(self._rewards), path=str(self._path))

    def get(self, source_id: str) -> float:
        """Learned reward for ``source_id``; 0.5 (neutral) when unseen."""
        self._refresh()
        return self._rewards.get(source_id, NEUTRAL)


_priors: SourcePriors | None = None


def get_source_priors() -> SourcePriors:
    global _priors
    if _priors is None:
        _priors = SourcePriors()
    return _priors
