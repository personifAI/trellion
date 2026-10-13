# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""The open tier must stand on its own.

Blocks out the optional add-on package, then imports the open modules that
seam to it. Every `_AVAILABLE` guard must read False and nothing may raise.
"""

from __future__ import annotations

import importlib
import sys

import pytest

OPEN_SEAM_MODULES = [
    "trellion.live.rag_integration",
    "trellion.live.pageindex_retriever",
    "trellion.live.rag_orchestrator",
    "trellion.live.manager",
    "trellion.live",
]

@pytest.fixture
def no_intel(monkeypatch):
    # Setting sys.modules[name] = None makes `import name` raise ImportError.
    monkeypatch.setitem(sys.modules, "trellion_intel", None)
    for m in OPEN_SEAM_MODULES:
        monkeypatch.delitem(sys.modules, m, raising=False)
    yield
    for m in OPEN_SEAM_MODULES:
        sys.modules.pop(m, None)

def test_import_intel_raises_the_tier_message(no_intel):
    from trellion.live._intel import import_intel

    with pytest.raises(ImportError, match="Intelligent tier"):
        import_intel("any_module")

def test_open_seam_modules_import_and_degrade(no_intel):
    ri = importlib.import_module("trellion.live.rag_integration")
    assert ri._GRAPH_NARROWER_AVAILABLE is False
    assert ri._graph_narrower_factory is None
    pr = importlib.import_module("trellion.live.pageindex_retriever")
    assert pr.GRAPH_RANKER_AVAILABLE is False
    ro = importlib.import_module("trellion.live.rag_orchestrator")
    assert ro.GRAPH_NARROWER_AVAILABLE is False
    importlib.import_module("trellion.live.manager")

def test_graph_narrower_provider_is_unavailable(no_intel):
    from trellion.live._intel import GRAPH_NARROWER, addon_provider

    with pytest.raises(ImportError, match="Intelligent tier"):
        addon_provider(GRAPH_NARROWER)

def test_graph_narrower_contract_is_public_and_needs_no_add_on(no_intel):
    live = importlib.import_module("trellion.live")
    from trellion.live.graph_types import GraphNarrower

    assert live.GraphNarrower is GraphNarrower

def test_every_provider_role_is_unavailable(no_intel):
    """Each role trellion defines reads as "not available" with no add-on."""
    from trellion.live import _intel

    roles = [v for k, v in vars(_intel).items() if k.isupper() and isinstance(v, str)
             and k in {"GRAPH_NARROWER", "GRAPH_RANKER", "QUERY_ROUTER", "RELATION_CLASSIFIER",
                       "RELATION_EXTRACTOR", "ENTITY_EMBEDDER", "CLUSTER_BUILDER", "GRAPH_METRICS"}]
    assert len(roles) == 8
    for role in roles:
        with pytest.raises(ImportError, match="Intelligent tier"):
            _intel.addon_provider(role)

def test_rag_integration_roles_read_unavailable(no_intel):
    ri = importlib.import_module("trellion.live.rag_integration")
    assert (ri._CLASSIFIER_AVAILABLE, ri._RELATION_EXTRACTOR_AVAILABLE,
            ri._GRAPH_EMBEDDER_AVAILABLE, ri._GRAPH_CLUSTERER_AVAILABLE) == (False,) * 4

def test_trellion_re_exports_no_add_on_class(no_intel):
    """trellion names no add-on class: those come from the add-on package itself.

    Every name in the public API must resolve to an object defined in trellion
    (or a third party), never in the add-on package the seam loads.
    """
    from trellion.live._intel import INTEL_PACKAGE

    for modname in ("trellion.live", "trellion"):
        mod = importlib.import_module(modname)
        for name in mod.__all__:
            obj = getattr(mod, name, None)
            origin = getattr(obj, "__module__", "") or ""
            assert not origin.startswith(INTEL_PACKAGE), f"{modname}.{name} comes from the add-on"

def test_star_import_skips_intel_names(no_intel):
    """`from trellion.live import *` is what the noomesh.memory shim does;
    with the tier absent it must import cleanly and simply not carry the
    add-on names."""
    sys.modules.pop("trellion.live._intel", None)
    sys.modules.pop("trellion", None)
    live = importlib.import_module("trellion.live")
    ns: dict = {}
    exec("from trellion.live import *", ns)  # noqa: S102 — the shim's exact statement
    assert all(n in ns for n in live.__all__)
    assert "GraphNarrower" in ns
    assert "PageIndexStore" in ns
