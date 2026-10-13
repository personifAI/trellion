# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""Lightweight per-stage telemetry.

Every build-pipeline stage appends one StageTrace. The collector is JSONL-
serializable so a run's timeline can be replayed offline, diffed across
runs, or fed into RDS-style accounting (tokens / quality).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import Lock


@dataclass
class StageTrace:
    stage: str
    started_at: float = 0.0
    wall_ms: float = 0.0
    records_in: int = 0
    records_out: int = 0
    llm_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    notes: dict = field(default_factory=dict)

    def to_jsonable(self) -> dict:
        d = asdict(self)
        d["wall_ms"] = round(self.wall_ms, 2)
        return d


class TraceCollector:
    """Thread-safe append-only list of StageTrace entries."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._traces: list = []

    def record(self, trace: StageTrace) -> None:
        with self._lock:
            self._traces.append(trace)

    def extend(self, traces) -> None:
        with self._lock:
            self._traces.extend(traces)

    def as_list(self) -> list:
        with self._lock:
            return list(self._traces)

    def dump(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for t in self.as_list():
                f.write(json.dumps(t.to_jsonable()) + "\n")

    def summary(self) -> dict:
        out = {
            "stages": len(self._traces),
            "total_wall_ms": round(sum(t.wall_ms for t in self._traces), 2),
            "total_llm_calls": sum(t.llm_calls for t in self._traces),
            "total_tokens_in": sum(t.tokens_in for t in self._traces),
            "total_tokens_out": sum(t.tokens_out for t in self._traces),
        }
        return out


class stage_timer:
    """Context manager that records wall time and commits a StageTrace."""

    def __init__(self, collector: TraceCollector, stage: str, records_in: int = 0):
        self.collector = collector
        self.trace = StageTrace(stage=stage, records_in=records_in)

    def __enter__(self) -> StageTrace:
        self.trace.started_at = time.time()
        self._t0 = time.perf_counter()
        return self.trace

    def __exit__(self, exc_type, exc, tb) -> None:
        self.trace.wall_ms = (time.perf_counter() - self._t0) * 1000.0
        self.collector.record(self.trace)
