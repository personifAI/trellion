# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""TRELLION_BACKGROUND_WORKERS=0 starts no backlog loop; the facade still serves.

pop-os, 2026-10-26: the Intelligent-tier container is built on this image and
ran a second copy of every backlog loop beside noo-trellion, so every idle poll
hit Postgres twice. The add-on's overlay turns the loops off in one of them.
"""
from __future__ import annotations

import pytest

from trellion.live import rag_integration as ri


class _Schema:
    def __init__(self, dsn):
        pass

    async def ensure_tables(self):
        pass


class _Store:
    def __init__(self, dsn):
        self.closed = False

    async def connect(self):
        pass

    async def close(self):
        self.closed = True


class _Loop:
    started: list[str] = []
    stopped: list[str] = []

    def __init__(self, *a, **kw):
        pass

    async def start(self):
        _Loop.started.append(type(self).__name__)

    async def stop(self):
        _Loop.stopped.append(type(self).__name__)


class _Embedder(_Loop):
    pass


class _Extractor(_Loop):
    pass


@pytest.fixture
def rag(monkeypatch):
    _Loop.started, _Loop.stopped = [], []
    monkeypatch.setattr(ri, "PageIndexSchema", _Schema)
    monkeypatch.setattr(ri, "PageIndexStore", _Store)
    monkeypatch.setattr(ri, "PageIndexEmbedder", _Embedder)
    monkeypatch.setattr(ri, "EntityExtractor", _Extractor, raising=False)
    monkeypatch.setattr(ri, "_ENTITY_EXTRACTOR_AVAILABLE", True)
    monkeypatch.setattr(ri, "_GRAPH_BUILDER_AVAILABLE", False)
    return ri.PageIndexRAG(dsn="postgresql://x/y")


@pytest.mark.parametrize("value", ["0", "false", "OFF", " no "])
async def test_off_starts_no_loop_and_still_serves(rag, monkeypatch, value):
    monkeypatch.setenv(ri.BACKGROUND_WORKERS_ENV, value)
    await rag.startup()
    assert _Loop.started == []
    assert rag._started and rag.retriever is not None and rag.orchestrator is not None
    await rag.shutdown()
    assert _Loop.stopped == []
    assert rag.store.closed


@pytest.mark.parametrize("value", [None, "", "1", "true", "yes"])
async def test_on_by_default(rag, monkeypatch, value):
    if value is None:
        monkeypatch.delenv(ri.BACKGROUND_WORKERS_ENV, raising=False)
    else:
        monkeypatch.setenv(ri.BACKGROUND_WORKERS_ENV, value)
    await rag.startup()
    assert _Loop.started == ["_Embedder", "_Extractor"]
    await rag.shutdown()
    assert _Loop.stopped == ["_Extractor", "_Embedder"]
