# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""Live Trellion memory exports.

Core memory classes are eager. Graph, PageIndex, RAG, and video helpers are
lazy so applications can use ``MemorySystem`` without importing optional live
retrieval dependencies.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from ._messaging import MemoryEvents, NoopEvents
from .manager import MemoryConfig, MemorySystem, SessionMemory
from .pgvector_store import PGVectorConfig, PGVectorStore
from .store import FileMemoryStore, InMemoryStore, MemoryRecord

MemoryManager = MemorySystem

_LAZY_EXPORTS = {
    "Citation": ("trellion.live.rag_orchestrator", "Citation"),
    "ContextLoader": ("trellion.live.context_files", "ContextLoader"),
    "ConversationEntry": ("trellion.live.conversation_store", "ConversationEntry"),
    "ConversationStore": ("trellion.live.conversation_store", "ConversationStore"),
    "DocumentIngestor": ("trellion.live.pageindex_ingestion", "DocumentIngestor"),
    "EntityExtractor": ("trellion.live.graph_extractor", "EntityExtractor"),
    "Evidence": ("trellion.live.evidence", "Evidence"),
    "EvidencePack": ("trellion.live.pageindex_retriever", "EvidencePack"),
    "GraphBuilder": ("trellion.live.graph_builder", "GraphBuilder"),
    "GraphNarrower": ("trellion.live.graph_types", "GraphNarrower"),
    "GraphResult": ("trellion.live.graph_types", "GraphResult"),
    "Hypothesis": ("trellion.live.graph_types", "Hypothesis"),
    "MessageType": ("trellion.live.conversation_store", "MessageType"),
    "PageIndexEmbedder": ("trellion.live.pageindex_embedder", "PageIndexEmbedder"),
    "PageIndexRAG": ("trellion.live.rag_integration", "PageIndexRAG"),
    "PageIndexRetriever": ("trellion.live.pageindex_retriever", "PageIndexRetriever"),
    "PageIndexSchema": ("trellion.live.pageindex_schema", "PageIndexSchema"),
    "PageIndexStore": ("trellion.live.pageindex_store", "PageIndexStore"),
    "PersonaAdapter": ("trellion.live.persona", "PersonaAdapter"),
    "PersonaLens": ("trellion.live.persona", "PersonaLens"),
    "RAGOrchestrator": ("trellion.live.rag_orchestrator", "RAGOrchestrator"),
    "RAGResponse": ("trellion.live.rag_orchestrator", "RAGResponse"),
    "RAGVerifier": ("trellion.live.rag_verifier", "RAGVerifier"),
    "VerificationResult": ("trellion.live.rag_verifier", "VerificationResult"),
    "VideoMemoryManager": ("trellion.live.video.manager", "VideoMemoryManager"),
    "EpisodicRecallManager": ("trellion.live.recall", "EpisodicRecallManager"),
    "PersonaVectorIndex": ("trellion.live.recall", "PersonaVectorIndex"),
    "evidence_from_chunk": ("trellion.live.evidence", "evidence_from_chunk"),
}

__all__ = [
    "Citation",
    "ContextLoader",
    "ConversationEntry",
    "ConversationStore",
    "DocumentIngestor",
    "EntityExtractor",
    "Evidence",
    "EvidencePack",
    "FileMemoryStore",
    "GraphBuilder",
    "GraphNarrower",
    "GraphResult",
    "Hypothesis",
    "InMemoryStore",
    "MemoryConfig",
    "MemoryEvents",
    "MemoryManager",
    "MemoryRecord",
    "MemorySystem",
    "MessageType",
    "NoopEvents",
    "PageIndexEmbedder",
    "PageIndexRAG",
    "PageIndexRetriever",
    "PageIndexSchema",
    "PageIndexStore",
    "PersonaAdapter",
    "PersonaLens",
    "PGVectorConfig",
    "PGVectorStore",
    "RAGOrchestrator",
    "RAGResponse",
    "RAGVerifier",
    "SessionMemory",
    "VerificationResult",
    "VideoMemoryManager",
    "EpisodicRecallManager",
    "PersonaVectorIndex",
    "evidence_from_chunk",
]


def __getattr__(name: str) -> Any:
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    try:
        value = getattr(import_module(module_name), attr_name)
    except ImportError:
        if name == "VideoMemoryManager":
            value = None
        else:
            raise
    globals()[name] = value
    return value
