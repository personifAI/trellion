# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""trellion never starts its host's NATS worker pool.

``PageIndexRAG.startup()`` used to call noomesh's ``start_worker_pool`` with
its own messaging handle, which the host leaves unset. The host then started
the real pool on its connected handle, so the call did nothing except log
``worker_pool.no_nats`` at WARNING on every healthy start. Had a host
passed a live handle, it would have subscribed every subject twice. The
host owns its pool; a library reaching into it is the bug.
"""
from __future__ import annotations

from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "trellion"


def test_the_scan_sees_the_sources():
    assert (SRC / "live" / "rag_integration.py").is_file()


def test_no_trellion_module_starts_the_host_worker_pool():
    hits = [
        f"{p.relative_to(SRC)}:{n}"
        for p in sorted(SRC.rglob("*.py"))
        for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
        if "start_worker_pool" in line or "noomesh.gateway.worker_pool" in line
    ]
    assert hits == [], f"trellion reaches into the host's worker pool: {hits}"
