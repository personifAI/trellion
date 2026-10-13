# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""Basic-tier degradation: the seam modules must import and run without
the optional add-on's graph modules.

The fixture simulates a Basic install by blocking the add-on modules
(``None`` in ``sys.modules`` makes ``import X`` raise ImportError) and
forcing the seam modules to re-import under the block-out. When the
Intelligent package is present nothing changes — these tests protect the
*absence* path so the split stays a file move, not a redesign.
"""

from __future__ import annotations

import importlib
import sys
from unittest.mock import MagicMock

import pytest

# The add-on package and the in-tree module paths are blocked, so the
# block-out holds whether or not the add-on happens to be installed.
INTELLIGENT_MODULES = [
    "trellion_intel",
    "trellion.live.graph_builder",
    "trellion.live.graph_extractor",
]

# Modules whose import-time behavior changes under the block-out; they are
# purged so each test imports them fresh, and restored on teardown.
SEAM_MODULES = [
    "trellion.live.pageindex_retriever",
    "trellion.live.rag_generator",
    "trellion.live.rag_orchestrator",
    "trellion.live.rag_integration",
]


@pytest.fixture
def basic_tier(monkeypatch):
    for name in INTELLIGENT_MODULES:
        monkeypatch.setitem(sys.modules, name, None)
    saved = {name: sys.modules.pop(name, None) for name in SEAM_MODULES}
    yield
    # Restore the real seam modules before monkeypatch lifts the block-out,
    # so later tests never see a Basic-flavored module by accident.
    for name, mod in saved.items():
        if mod is not None:
            sys.modules[name] = mod
        else:
            sys.modules.pop(name, None)


def test_provided_graph_narrower_shares_the_result_types():
    """The graph narrower an add-on provides must hand back trellion's own
    Hypothesis/GraphResult classes (one class object each, not copies). When
    no add-on is installed (the open-tier CI), there is nothing to check."""
    pytest.importorskip("trellion_intel")
    from trellion.live._intel import GRAPH_NARROWER, addon_provider

    graph_types = importlib.import_module("trellion.live.graph_types")
    provider = addon_provider(GRAPH_NARROWER)
    home = sys.modules[provider.__module__]
    assert home.Hypothesis is graph_types.Hypothesis
    assert home.GraphResult is graph_types.GraphResult
    assert hasattr(provider, "navigate")


def test_retriever_falls_back_to_vector_only(basic_tier):
    mod = importlib.import_module("trellion.live.pageindex_retriever")
    assert mod.GRAPH_RANKER_AVAILABLE is False
    retriever = mod.PageIndexRetriever(store=MagicMock(), embedder=MagicMock())
    assert retriever._graph_ranker is None


async def test_orchestrator_navigate_returns_none_without_navigator(basic_tier):
    mod = importlib.import_module("trellion.live.rag_orchestrator")
    assert mod.GRAPH_NARROWER_AVAILABLE is False
    orch = mod.RAGOrchestrator(
        navigator=None, retriever=MagicMock(), store=MagicMock()
    )
    assert await orch._navigate("what causes widget failures?") is None


def test_rag_generator_imports_and_formats_hypotheses(basic_tier):
    mod = importlib.import_module("trellion.live.rag_generator")
    graph_types = importlib.import_module("trellion.live.graph_types")
    hyp = graph_types.Hypothesis(
        path=["a", "b"], description="a relates to b", confidence=0.7
    )
    assert "a relates to b" in mod._format_hypotheses([hyp])


def test_rag_integration_flags_off(basic_tier):
    mod = importlib.import_module("trellion.live.rag_integration")
    assert mod._GRAPH_BUILDER_AVAILABLE is False
    assert mod._ENTITY_EXTRACTOR_AVAILABLE is False
    assert mod._GRAPH_NARROWER_AVAILABLE is False


def test_manager_formats_plain_lists_without_router(basic_tier):
    # manager imports the router lazily at call time, so the already-imported
    # module is exercised under the block-out directly.
    mod = importlib.import_module("trellion.live.manager")
    out = mod.MemorySystem._format_retrieval_results([["fact one"], ["fact two"]])
    assert "fact one" in out and "fact two" in out


def test_every_role_an_add_on_declares_is_usable():
    """With an add-on installed, each role trellion defines resolves to a
    callable, and the objects trellion calls methods on have those methods.
    Without one there is nothing to check."""
    pytest.importorskip("trellion_intel")
    from trellion.live import _intel
    from trellion.live.roles import GraphRanker

    roles = (_intel.GRAPH_NARROWER, _intel.GRAPH_RANKER, _intel.QUERY_ROUTER,
             _intel.RELATION_CLASSIFIER, _intel.RELATION_EXTRACTOR,
             _intel.ENTITY_EMBEDDER, _intel.CLUSTER_BUILDER, _intel.GRAPH_METRICS)
    for role in roles:
        assert callable(_intel.addon_provider(role)), role
    assert isinstance(_intel.addon_provider(_intel.GRAPH_RANKER)(), GraphRanker)
    for role, methods in {_intel.RELATION_CLASSIFIER: ("start", "stop"),
                          _intel.RELATION_EXTRACTOR: ("start", "stop"),
                          _intel.ENTITY_EMBEDDER: ("start", "stop"),
                          _intel.GRAPH_METRICS: ("start", "stop"),
                          _intel.CLUSTER_BUILDER: ("run_clustering", "close")}.items():
        for m in methods:
            assert hasattr(_intel.addon_provider(role), m), (role, m)


def test_role_the_add_on_does_not_declare_reads_as_unavailable(monkeypatch):
    """An installed add-on that fills no such role: ImportError (not a crash)."""
    pytest.importorskip("trellion_intel")
    from trellion.live import _intel

    with pytest.raises(ImportError):
        _intel.addon_provider("no_such_role")
