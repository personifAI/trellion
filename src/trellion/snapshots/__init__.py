# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""trellion.snapshots — System 1.5 consolidation layer.

Takes a slice of the live bi-temporal graph (topic-scoped, time-window,
or user-attention), runs it through trellion's build pipeline
(perturbation-consensus canonicalization + Leiden clustering + Eades-Lin-
Smyth DAG projection), and stores the deterministic snapshot in the
``trellion_snapshots`` family of tables.

Snapshots are immutable + dated → diffable. They're zero-LLM and instant
to query (T1–T5 router via ``trellion.query``).

Public API:
    from trellion.snapshots import (
        SliceSpec, ConsolidationResult,
        GraphSlicer, SnapshotStore, SnapshotOrchestrator,
    )

See proposal 06 §5.1 for design.
"""
from __future__ import annotations

from .ckg_adapter import get_cached_ckg, invalidate_cache, load_ckg_from_snapshot
from .orchestrator import ConsolidationResult, SnapshotOrchestrator
from .slicer import GraphSlicer, SliceSpec
from .store import SnapshotStore

__all__ = [
    "ConsolidationResult",
    "GraphSlicer",
    "SliceSpec",
    "SnapshotOrchestrator",
    "SnapshotStore",
    "get_cached_ckg",
    "invalidate_cache",
    "load_ckg_from_snapshot",
]
