# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""trellion - cognitive substrate facade.

The live memory manager is intentionally eager because lightweight consumers
use it directly. Heavier DB, PageIndex, graph, and RAG
exports are loaded lazily so importing Trellion memory does not require the
whole optional dependency surface.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .live._messaging import MemoryEvents, NoopEvents
from .live.manager import MemoryConfig, MemorySystem, SessionMemory
from .live.store import FileMemoryStore, InMemoryStore, MemoryRecord

_LAZY_EXPORTS = {
    "Base": ("trellion.db", "Base"),
    "ConversationStore": ("trellion.live.conversation_store", "ConversationStore"),
    "MEMORY_TABLE_NAMES": ("trellion.db", "MEMORY_TABLE_NAMES"),
    "PageIndexStore": ("trellion.live.pageindex_store", "PageIndexStore"),
    "memory_tables": ("trellion.db", "memory_tables"),
    "non_memory_tables": ("trellion.db", "non_memory_tables"),
}

__all__ = [
    "Base",
    "ConversationStore",
    "FileMemoryStore",
    "InMemoryStore",
    "MEMORY_TABLE_NAMES",
    "MemoryConfig",
    "MemoryEvents",
    "MemoryRecord",
    "MemorySystem",
    "NoopEvents",
    "PageIndexStore",
    "SessionMemory",
    "memory_tables",
    "non_memory_tables",
]


def __getattr__(name: str) -> Any:
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
