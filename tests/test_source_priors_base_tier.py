# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""Source priors on the base tier: no intel import, no traceback, file-backed.

Every RAG query once tried an add-on import, failed, and
logged a rich traceback (WITH LOCALS, which can carry the query). The writer
is an optional add-on; the retriever reads the priors it persists
(``RETRIEVAL_TRACKER_PATH``) and never imports it.
"""
from __future__ import annotations

import json
import os
import sys

import pytest
import structlog

from trellion.live import source_priors
from trellion.live.pageindex_retriever import PageIndexRetriever


class _Embedder:
    async def embed_text(self, text):
        return [0.1]


class _Store:
    def __init__(self, pages):
        self._pages = pages

    async def search_pages(self, embedding, top_k=10, doc_ids=None, roles=None):
        return [dict(p) for p in self._pages]

    async def search_chunks(self, embedding, top_k=10, page_ids=None, roles=None):
        return []


def _retriever(pages):
    r = PageIndexRetriever.__new__(PageIndexRetriever)
    r.store = _Store(pages)
    r.embedder = _Embedder()
    r._graph_ranker = None

    async def _no_chunks(emb, pages, top_k=0, **kw):
        return pages

    r._merge_chunk_first_pages = _no_chunks

    async def _evidence(pages, chunks):
        return [p["page_id"] for p in pages]

    r.build_evidence = _evidence
    return r


@pytest.fixture(autouse=True)
def _fresh_priors(monkeypatch, tmp_path):
    monkeypatch.setenv("RETRIEVAL_TRACKER_PATH", str(tmp_path / "tracker.json"))
    monkeypatch.setattr(source_priors, "_priors", None)
    monkeypatch.setattr(source_priors, "_RECHECK_SECS", 0.0)
    yield
    source_priors._priors = None


@pytest.fixture
def no_intel(monkeypatch):
    # blocking the package root is enough: the loader imports it before any module
    monkeypatch.setitem(sys.modules, "trellion_intel", None)


PAGES = [
    {"page_id": "a", "score": 0.60},
    {"page_id": "b", "score": 0.58},
]


async def test_base_tier_does_not_import_intel_per_query(no_intel, monkeypatch):
    import trellion.live._intel as seam

    calls = []
    real = seam.import_intel
    monkeypatch.setattr(seam, "import_intel", lambda m: (calls.append(m), real(m))[1])
    r = _retriever(PAGES)
    for _ in range(3):
        await r.retrieve("q", top_k=2)
    assert calls == [], f"retriever attempted intel imports per query: {calls}"


async def test_base_tier_logs_no_traceback(no_intel):
    with structlog.testing.capture_logs() as logs:
        await _retriever(PAGES).retrieve("secret query text", top_k=2)
    assert not [e for e in logs if e.get("exc_info")], logs
    # Only the retriever's and the priors' own events: an unrelated subsystem
    # (e.g. metrics publish when noomesh is absent) may log its own debug line.
    mine = [e for e in logs if e["event"].startswith(("retriever.", "source_priors"))]
    assert not [e for e in mine if "failed" in e["event"]], mine


async def test_persisted_prior_reorders_pages(tmp_path):
    (tmp_path / "tracker.json").write_text(
        json.dumps({"source_rewards": {"b": 1.0, "a": 0.0}})
    )
    out = await _retriever(PAGES).retrieve("q", top_k=2)
    assert out == ["b", "a"], "a high learned prior must outrank a low one"


async def test_no_file_is_neutral_and_quiet():
    with structlog.testing.capture_logs() as logs:
        out = await _retriever(PAGES).retrieve("q", top_k=2)
    assert out == ["a", "b"]
    assert not [e for e in logs if e["log_level"] in ("warning", "error")]


def test_reader_picks_up_flushes_and_survives_torn_writes(tmp_path):
    p = tmp_path / "t.json"
    sp = source_priors.SourcePriors(p, recheck_secs=0.0)
    assert sp.get("x") == 0.5
    p.write_text(json.dumps({"source_rewards": {"x": 0.9}}))
    assert sp.get("x") == 0.9
    p.write_text('{"source_rew')  # torn write
    os.utime(p, (1, 1_000_000_000))
    assert sp.get("x") == 0.9, "keeps the last good values"
    p.write_text(json.dumps({"source_rewards": {"x": 0.2}}))
    os.utime(p, (1, 2_000_000_000))
    assert sp.get("x") == 0.2
