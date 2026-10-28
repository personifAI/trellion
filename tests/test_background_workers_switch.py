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
    closed = False

    async def aclose(self):
        _Embedder.closed = True


class _Extractor(_Loop):
    pass


@pytest.fixture
def rag(monkeypatch):
    _Loop.started, _Loop.stopped = [], []
    _Embedder.closed = False
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
    assert _Embedder.closed, "the client a query may have opened is closed"
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


async def test_off_still_embeds_a_query(monkeypatch):
    """0.6.13-rc.2/rc.3 rehearsal: with the loops off, every rag.query failed
    503 with a blank error. The embedder's gateway client was made only in
    start(), and embed_text asserted on it: a bare AssertionError, empty text."""
    import httpx

    from trellion.live import pageindex_embedder as pe_mod

    seen = []

    def gateway(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        return httpx.Response(200, json={"model": "m", "data": [{"index": 0, "embedding": [0.5, 0.25]}]})

    real_client = httpx.AsyncClient

    def client(**kw):
        kw.pop("verify", None), kw.pop("cert", None), kw.pop("auth", None)
        return real_client(transport=httpx.MockTransport(gateway), **kw)

    monkeypatch.setattr(pe_mod.httpx, "AsyncClient", client)
    monkeypatch.setattr(ri, "PageIndexSchema", _Schema)
    monkeypatch.setattr(ri, "PageIndexStore", _Store)
    monkeypatch.setattr(ri, "_ENTITY_EXTRACTOR_AVAILABLE", False)
    monkeypatch.setattr(ri, "_GRAPH_BUILDER_AVAILABLE", False)
    monkeypatch.setenv(ri.BACKGROUND_WORKERS_ENV, "0")

    rag = ri.PageIndexRAG(dsn="postgresql://x/y", embedding_url="https://gateway:8080")
    await rag.startup()
    assert rag.retriever.embedder is rag.embedder
    vec = await rag.embedder.embed_text("what changed?")
    assert list(vec) == [0.5, 0.25]
    assert seen == ["/api/external/embeddings"]
    await rag.shutdown()
    assert rag.embedder._client is None, "shutdown closes the client the query opened"
