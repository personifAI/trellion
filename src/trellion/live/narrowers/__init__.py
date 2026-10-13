# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""Retrieval narrowers — peer components that produce candidate doc/page IDs
to scope vector search.

The narrower architecture:

    Query
      ├──→ graph narrower         → candidate_page_ids (entity-graph traversal; add-on provider) ✓
      ├──→ MetadataNarrower       → candidate_doc_ids  (title / path / tags / jsonb)
      ├──→ TemporalNarrower       → candidate_doc_ids  (created_at, recency)
      ├──→ ConversationNarrower   → candidate_doc_ids  (current-session topics)
            │
            ▼  union (recall) | intersect (precision) | weighted combine
            │
       PageIndexRetriever  (scoped vector search)

Each narrower returns a `NarrowerResult` with optional doc_id and/or page_id
sets. The `NarrowerOrchestrator` runs them in parallel and combines the
outputs according to the configured mode, producing a single
`CombinedNarrowing` that the retriever consumes.

Narrowers are deliberately decoupled from the retriever — adding a new
narrower means dropping a class into this package and registering it,
not editing the retriever or orchestrator.
"""

from .base import (
    Narrower,
    NarrowerResult,
    CombinedNarrowing,
    CombineMode,
)
from .orchestrator import NarrowerOrchestrator
from .metadata import MetadataNarrower

__all__ = [
    "Narrower",
    "NarrowerResult",
    "CombinedNarrowing",
    "CombineMode",
    "NarrowerOrchestrator",
    "MetadataNarrower",
]
