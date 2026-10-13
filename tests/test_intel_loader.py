# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""trellion finds the add-on by entry point, falls back by name, and
hands it the host services once.

Uses a throwaway package named ``trellion_intel`` written to tmp_path, so it
runs without the real add-on installed.
"""

from __future__ import annotations

import sys
import types

import pytest

from trellion.live import _intel
from trellion.live.host_services import HOST
from trellion_protocols import ADDON_ENTRY_POINT_GROUP

FAKE_INIT = """
calls = []
def register(host):
    calls.append(host)
"""


@pytest.fixture
def fake_intel(tmp_path, monkeypatch):
    pkg = tmp_path / "trellion_intel"
    pkg.mkdir()
    (pkg / "__init__.py").write_text(FAKE_INIT)
    (pkg / "probe_module.py").write_text("NAME = 'nav'\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    for m in [m for m in sys.modules if m == "trellion_intel" or m.startswith("trellion_intel.")]:
        monkeypatch.delitem(sys.modules, m)
    _intel._reset_for_tests()
    yield pkg
    _intel._reset_for_tests()
    for m in [m for m in sys.modules if m == "trellion_intel" or m.startswith("trellion_intel.")]:
        sys.modules.pop(m, None)


class _EP:
    def __init__(self, name, value):
        self.name, self.value = name, value
        self.module = value.split(":")[0]

    def load(self):
        mod = __import__(self.module, fromlist=["x"])
        return getattr(mod, self.value.split(":")[1])


def _with_entry_points(monkeypatch, eps):
    import importlib.metadata as md

    def fake(*, group):
        assert group == ADDON_ENTRY_POINT_GROUP
        return eps

    monkeypatch.setattr(md, "entry_points", fake)


def test_entry_point_is_preferred_and_registers_the_host_once(fake_intel, monkeypatch):
    _with_entry_points(monkeypatch, [_EP("other_addon", "json:dumps"),
                                     _EP("trellion_intel", "trellion_intel:register")])
    assert _intel.intel_available()
    nav = _intel.import_intel("probe_module")
    assert nav.NAME == "nav"
    _intel.import_intel("probe_module")
    import trellion_intel as addon

    assert addon.calls == [HOST], "registered exactly once, with trellion's host"
    assert _intel.loaded_via() == "entry_point"


def test_without_an_entry_point_it_falls_back_by_name_and_still_registers(fake_intel, monkeypatch):
    _with_entry_points(monkeypatch, [])
    assert _intel.import_intel("probe_module").NAME == "nav"
    import trellion_intel as addon

    assert addon.calls == [HOST]
    assert _intel.loaded_via() == "by_name"


def test_a_package_without_register_still_loads_by_name(fake_intel, monkeypatch):
    (fake_intel / "__init__.py").write_text("")
    _with_entry_points(monkeypatch, [])
    assert _intel.import_intel("probe_module").NAME == "nav"


def test_a_declared_but_broken_entry_point_is_the_tier_error(fake_intel, monkeypatch):
    _with_entry_points(monkeypatch, [_EP("trellion_intel", "trellion_intel_missing:register")])
    with pytest.raises(ImportError, match="Intelligent tier"):
        _intel.import_intel("probe_module")
    assert not _intel.intel_available()


def test_absent_add_on_is_the_tier_error(monkeypatch):
    _intel._reset_for_tests()
    monkeypatch.setitem(sys.modules, "trellion_intel", None)
    _with_entry_points(monkeypatch, [])
    try:
        with pytest.raises(ImportError, match="Intelligent tier"):
            _intel.import_intel("probe_module")
        assert _intel.intel_available() is False
    finally:
        _intel._reset_for_tests()


def test_the_host_module_imports_nothing_heavy_at_import():
    """It is loaded while trellion.live initialises; its members import lazily."""
    import importlib

    import trellion.live.host_services as hs

    src = importlib.util.find_spec("trellion.live.host_services").origin
    top = [ln for ln in open(src).read().splitlines() if ln.startswith(("import ", "from "))]
    assert top == ["from __future__ import annotations", "from typing import Any, Optional"]
    assert isinstance(hs.HOST, object) and not isinstance(hs.HOST, types.ModuleType)


def test_an_add_on_whose_registration_fails_is_the_tier_error(fake_intel, monkeypatch):
    (fake_intel / "__init__.py").write_text("def register(host):\n    raise TypeError('bad host')\n")
    _with_entry_points(monkeypatch, [_EP("trellion_intel", "trellion_intel:register")])
    with pytest.raises(ImportError, match="Intelligent tier"):
        _intel.import_intel("probe_module")
    assert not _intel.intel_available()


def test_an_entry_point_naming_a_missing_object_is_the_tier_error(fake_intel, monkeypatch):
    (fake_intel / "__init__.py").write_text("")
    _with_entry_points(monkeypatch, [_EP("trellion_intel", "trellion_intel:register")])
    assert not _intel.intel_available()


# --- providers: trellion's own role names, the add-on declares what fills them ---


def _declare(fake_intel, body):
    (fake_intel / "__init__.py").write_text(FAKE_INIT + body)


def test_provider_resolves_the_declared_object(fake_intel, monkeypatch):
    (fake_intel / "narrow.py").write_text("class Factory:\n    def __init__(self, store):\n        self.store = store\n")
    _declare(fake_intel, "TRELLION_PROVIDERS = {'graph_narrower': 'trellion_intel.narrow:Factory'}\n")
    _with_entry_points(monkeypatch, [_EP("trellion_intel", "trellion_intel:register")])
    factory = _intel.addon_provider(_intel.GRAPH_NARROWER)
    assert factory(store="s").store == "s"


def test_provider_resolves_by_name_without_an_entry_point(fake_intel, monkeypatch):
    (fake_intel / "narrow.py").write_text("def make(store):\n    return store\n")
    _declare(fake_intel, "TRELLION_PROVIDERS = {'graph_narrower': 'trellion_intel.narrow:make'}\n")
    _with_entry_points(monkeypatch, [])
    assert _intel.addon_provider(_intel.GRAPH_NARROWER)("x") == "x"


def test_provider_absent_when_the_add_on_is_absent(monkeypatch):
    _intel._reset_for_tests()
    monkeypatch.setitem(sys.modules, "trellion_intel", None)
    _with_entry_points(monkeypatch, [])
    try:
        with pytest.raises(ImportError, match="Intelligent tier"):
            _intel.addon_provider(_intel.GRAPH_NARROWER)
    finally:
        _intel._reset_for_tests()


def test_provider_not_declared_is_the_tier_error(fake_intel, monkeypatch):
    _with_entry_points(monkeypatch, [_EP("trellion_intel", "trellion_intel:register")])
    with pytest.raises(ImportError, match="Intelligent tier"):
        _intel.addon_provider(_intel.GRAPH_NARROWER)


def test_provider_declaration_that_does_not_resolve_is_the_tier_error(fake_intel, monkeypatch):
    _declare(fake_intel, "TRELLION_PROVIDERS = {'graph_narrower': 'trellion_intel.nope:Missing'}\n")
    _with_entry_points(monkeypatch, [_EP("trellion_intel", "trellion_intel:register")])
    with pytest.raises(ImportError, match="Intelligent tier"):
        _intel.addon_provider(_intel.GRAPH_NARROWER)
