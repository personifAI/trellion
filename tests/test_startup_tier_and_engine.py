# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""Startup noise fixes: tier decided once, video memory survives early start."""

from __future__ import annotations

import sys
import types

import pytest
import structlog

from trellion.live import rag_integration as ri


def _bare_rag():
    rag = object.__new__(ri.PageIndexRAG) if hasattr(ri, "PageIndexRAG") else None
    assert rag is not None
    rag.store = object()
    rag.signal_computer = "unset"
    return rag


@pytest.fixture
def no_intel(monkeypatch):
    # blocking the package root is enough: the loader imports it before any module
    monkeypatch.setitem(sys.modules, "trellion_intel", None)


async def test_open_tier_skips_signal_computer_quietly(no_intel):
    rag = _bare_rag()
    with structlog.testing.capture_logs() as logs:
        await rag._start_signal_computer()
    assert rag.signal_computer is None
    assert not [e for e in logs if e["log_level"] in ("warning", "error")]
    assert not [e for e in logs if e.get("exc_info")]
    assert any(e["event"] == "pageindex_rag.signal_computer_skipped" for e in logs)


async def test_intel_tier_failure_still_warns(monkeypatch):
    class Boom:
        def __init__(self, **kw):
            pass

        async def start(self, interval):
            raise RuntimeError("real failure")

    mod = types.ModuleType("trellion_intel")
    mod.__path__ = []
    gs = types.ModuleType("probe_metrics_provider")
    gs.Metrics = Boom
    mod.TRELLION_PROVIDERS = {"graph_metrics": "probe_metrics_provider:Metrics"}
    monkeypatch.setitem(sys.modules, "trellion_intel", mod)
    monkeypatch.setitem(sys.modules, "probe_metrics_provider", gs)
    # Hermetic loader: no installed entry point (the real add-on may be in the
    # environment) and no package cached from an earlier test.
    # sys.modules' copy: tests that re-import trellion.live leave the package
    # attribute pointing at an older module object than the one callers import.
    import importlib

    _intel = importlib.import_module("trellion.live._intel")
    monkeypatch.setattr(_intel, "_entry_point", lambda: None)
    _intel._reset_for_tests()
    rag = _bare_rag()
    try:
        with structlog.testing.capture_logs() as logs:
            await rag._start_signal_computer()
    finally:
        _intel._reset_for_tests()
    assert rag.signal_computer is None
    assert any(e["event"] == "pageindex_rag.signal_computer_start_failed" for e in logs)


async def test_video_memory_wires_lazily_when_engine_arrives_after_startup(monkeypatch, tmp_path):
    import trellion.db.engine as eng
    from trellion.live.video.manager import VideoMemoryConfig, VideoMemoryManager

    monkeypatch.setattr(eng, "_async_engine", None)
    cfg = VideoMemoryConfig()
    cfg.storage_dir = str(tmp_path)
    vm = VideoMemoryManager(config=cfg, embedding_url="http://x", gateway_url="http://x")
    with structlog.testing.capture_logs() as logs:
        await vm.startup()
    assert not [e for e in logs if e["log_level"] == "warning"]
    assert vm.status()["engine_wired"] is False

    sentinel = types.SimpleNamespace()
    monkeypatch.setattr(eng, "_async_engine", sentinel)
    assert vm.status()["engine_wired"] is True
    assert vm._engine is sentinel
    assert vm.lifecycle is not None and vm.replay_engine is not None
