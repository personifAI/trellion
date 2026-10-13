# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""The seam to the optional commercial add-on package.

The add-on is a separate, separately licensed package (``trellion_intel``);
the open tier works without it. trellion finds it, loads it once and hands it
trellion's host services:

1. **Discovery.** An add-on declares an entry point in the
   ``trellion_protocols.ADDON_ENTRY_POINT_GROUP`` group (``trellion.addons``)
   whose name is the package (``trellion_intel``) and whose object is its
   registration callable. When no such entry point is installed — an image
   that copies the package into site-packages without metadata — trellion
   falls back to importing ``trellion_intel`` by name and calling its
   ``register`` if it has one.
2. **Registration.** The callable receives :data:`trellion.live.host_services.HOST`
   (a ``trellion_protocols.HostServices``): the same throttles, breaker,
   credential, quality gate, metrics path and JSON codec trellion uses.
   Done once per process, before any module of the package is handed out.
3. **Modules.** ``import_intel("<module>")`` imports
   ``trellion_intel.<module>``, or raises ImportError carrying the
   tier message.
4. **Providers.** For a role trellion defines in its own vocabulary (see
   :data:`GRAPH_NARROWER` and the constants after it), :func:`addon_provider` returns the object the
   add-on declares for it. The add-on declares them in a mapping at its
   package root, :data:`PROVIDERS_ATTR` (``TRELLION_PROVIDERS``), from role
   name to an ``"module:attribute"`` reference (entry-point object syntax),
   so trellion never names the add-on's modules or classes. Reading the
   mapping works the same whether the add-on was found by entry point or
   by name.

Tests block the intel package out (``tests/test_basic_without_intel.py``)
to prove the open tier stands on its own.
"""

from __future__ import annotations

import importlib
import logging
import threading
from types import ModuleType
from typing import Any, Optional

from trellion_protocols import ADDON_ENTRY_POINT_GROUP

log = logging.getLogger(__name__)

INTEL_PACKAGE = "trellion_intel"

TIER_MSG = (
    "requires the Intelligent tier (trellion_intel) — not installed on this "
    "controller; the open tier runs vector-only retrieval"
)

_lock = threading.RLock()
#: (package root, how it was found) once loaded; None until then.
_loaded: Optional[tuple[str, str]] = None


def _entry_point() -> Any:
    """The ``trellion_intel`` entry point in the add-on group, or None."""
    try:
        from importlib.metadata import entry_points

        for ep in entry_points(group=ADDON_ENTRY_POINT_GROUP):
            if ep.name == INTEL_PACKAGE:
                return ep
    except Exception:  # noqa: BLE001 - metadata backends differ; fall back by name
        log.debug("intel.entry_points_unreadable", exc_info=True)
    return None


def _load() -> tuple[str, str]:
    """Find, import and register the add-on once. Raises ImportError if absent."""
    global _loaded
    if _loaded is not None:
        return _loaded
    with _lock:
        if _loaded is not None:
            return _loaded
        from trellion.live.host_services import HOST

        ep = _entry_point()
        try:
            if ep is not None:
                register = ep.load()
                package, how = ep.module.split(".")[0], "entry_point"
            else:
                mod = importlib.import_module(INTEL_PACKAGE)
                register = getattr(mod, "register", None)
                package, how = INTEL_PACKAGE, "by_name"
            if register is not None:
                register(HOST)
        except ImportError:
            raise
        except Exception as e:  # noqa: BLE001 - declared but broken is not available
            # An add-on that is installed but cannot load or register is the
            # tier being unavailable, said loudly, not a crash of the caller.
            log.warning("intel.load_failed: %r", e)
            raise ImportError(f"{INTEL_PACKAGE} is installed but failed to load: {e!r}") from e
        _loaded = (package, how)
        return _loaded


#: Provider role: the graph narrower. Built as ``factory(store=<PageIndexStore>)``;
#: the result's ``await navigate(question)`` returns a
#: :class:`trellion.live.graph_types.GraphResult` whose seed entities and
#: hypotheses scope page retrieval (``candidate_page_ids``).
GRAPH_NARROWER = "graph_narrower"

#: Provider role: the graph ranker. Built as ``factory()``; scores and boosts
#: retrieved pages by their graph neighbourhood
#: (:class:`trellion.live.roles.GraphRanker`).
GRAPH_RANKER = "graph_ranker"

#: Provider role: the query router. Built as ``factory(memory_system, user_id)``
#: (:class:`trellion.live.roles.QueryRouter`).
QUERY_ROUTER = "query_router"

#: Provider role: classifies how two entities relate. Built as
#: ``factory(store=, llm_url=, model=, messaging=, embedding_url=)``
#: (:class:`trellion.live.roles.BackgroundWorker`).
RELATION_CLASSIFIER = "relation_classifier"

#: Provider role: extracts relations between entities. Built as
#: ``factory(store=, llm_url=, model=, messaging=)``
#: (:class:`trellion.live.roles.BackgroundWorker`).
RELATION_EXTRACTOR = "relation_extractor"

#: Provider role: embeds graph entities. Built as
#: ``factory(store=, embedder=, messaging=)``
#: (:class:`trellion.live.roles.BackgroundWorker`).
ENTITY_EMBEDDER = "entity_embedder"

#: Provider role: builds entity clusters. Built as
#: ``factory(store=, llm_url=, model=, messaging=)``
#: (:class:`trellion.live.roles.ClusterBuilder`).
CLUSTER_BUILDER = "cluster_builder"

#: Provider role: computes per-entity graph metrics on an interval. Built as
#: ``factory(store=, messaging=)`` (:class:`trellion.live.roles.BackgroundWorker`).
GRAPH_METRICS = "graph_metrics"

#: The add-on package-root attribute that declares its providers.
PROVIDERS_ATTR = "TRELLION_PROVIDERS"


def addon_provider(role: str) -> Any:
    """The object the add-on declares for ``role``, else ImportError with the tier message.

    Absent add-on, an add-on that declares no such role, or a declaration that
    does not resolve all read as "not available" to the caller; the last two are
    also logged, since an installed add-on that cannot fill a role is a fault.
    """
    try:
        package, _ = _load()
    except ImportError as first:
        raise ImportError(f"{role}: {TIER_MSG}") from first
    try:
        declared = getattr(importlib.import_module(package), PROVIDERS_ATTR, None) or {}
        ref = declared.get(role) if isinstance(declared, dict) else None
        if not ref:
            raise LookupError(f"add-on {package!r} declares no provider for {role!r}")
        module, _, attr = str(ref).partition(":")
        obj: Any = importlib.import_module(module)
        for part in filter(None, attr.split(".")):
            obj = getattr(obj, part)
        return obj
    except Exception as e:  # noqa: BLE001 - an unfillable role is the role being unavailable
        log.warning("intel.provider_unavailable role=%s: %r", role, e)
        raise ImportError(f"{role}: {TIER_MSG} ({e})") from e


def loaded_via() -> Optional[str]:
    """``"entry_point"`` / ``"by_name"`` once loaded, else None."""
    return _loaded[1] if _loaded else None


def import_intel(module: str) -> ModuleType:
    """Import ``module`` from the registered add-on, else raise ImportError with the tier message."""
    try:
        package, _ = _load()
        importlib.import_module(package)  # a blocked/removed root must fail even if a child is cached
        return importlib.import_module(f"{package}.{module}")
    except ImportError as first:
        raise ImportError(f"{module}: {TIER_MSG}") from first


def intel_available() -> bool:
    try:
        package, _ = _load()
        importlib.import_module(package)
        return True
    except ImportError:
        return False


def _reset_for_tests() -> None:
    global _loaded
    _loaded = None
