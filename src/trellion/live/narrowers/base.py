# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""Narrower protocol + result types.

Each narrower is a peer component that takes a query (+ optional context)
and returns a `NarrowerResult` declaring which docs and/or pages it thinks
are relevant. A narrower returning `None` for both fields means "no
opinion / didn't apply" — the orchestrator skips it during combine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Protocol, runtime_checkable


class CombineMode(str, Enum):
    """How `NarrowerOrchestrator` merges per-narrower outputs.

    - INTERSECT: doc must appear in EVERY narrower that has an opinion.
      Highest precision, lowest recall. Use when narrowers are confident
      and you want them to agree.
    - UNION: doc appears if ANY narrower picked it. Highest recall,
      lowest precision. Use when narrowers are conservative.
    - WEIGHTED: each narrower contributes scores; final set is top-N
      by combined score. Tunable middle ground (future — not implemented
      in the first cut).
    """

    INTERSECT = "intersect"
    UNION = "union"
    WEIGHTED = "weighted"


@dataclass
class NarrowerResult:
    """Output of one narrower's `narrow()` call.

    A narrower may produce doc_ids, page_ids, both, or neither. `None`
    (the default) means "no opinion" — the orchestrator treats this as
    a non-vote and excludes the narrower from intersect calculations.
    An empty set (`set()`) means "I had an opinion and the answer is
    nothing matches" — under intersect, this collapses the final set to
    empty (the narrower vetoed everything).
    """

    name: str
    doc_ids: Optional[set[str]] = None
    page_ids: Optional[set[str]] = None
    confidence: float = 1.0  # used by WEIGHTED combine; ignored by INTERSECT/UNION
    debug: dict = field(default_factory=dict)  # free-form debug info per narrower

    @property
    def has_opinion(self) -> bool:
        """True if this narrower produced any candidate IDs (even an empty set)."""
        return self.doc_ids is not None or self.page_ids is not None


@dataclass
class CombinedNarrowing:
    """Aggregate output of `NarrowerOrchestrator.run()`.

    Either field may be `None`, which means "no narrowing applied at this
    granularity — let the retriever do unrestricted search at this level".

    `candidate_doc_ids` and `candidate_page_ids` are independent: a query
    may have doc-level narrowing (from metadata) AND page-level narrowing
    (from graph) simultaneously, and the retriever applies both as
    sequential filters.
    """

    candidate_doc_ids: Optional[set[str]] = None
    candidate_page_ids: Optional[set[str]] = None
    contributors: list[str] = field(default_factory=list)  # narrower names that voted
    mode: CombineMode = CombineMode.INTERSECT


@runtime_checkable
class Narrower(Protocol):
    """Contract for a single narrower.

    Implementations should be cheap to construct; expensive setup belongs
    in the orchestrator's lifecycle. The `narrow()` call is the hot path
    and should aim for sub-100ms latency for a typical query.
    """

    name: str

    async def narrow(self, query: str, ctx: dict) -> NarrowerResult:
        """Produce candidate doc/page IDs for this query.

        Args:
            query: The user's natural-language query string.
            ctx: Free-form context dict. Common keys:
                - `metadata_filters`: dict of explicit metadata filters
                - `time_range`: (start, end) ISO datetime tuple
                - `conversation_id`: current chat session ID
                - `user_id`: requesting user
                Implementations should ignore keys they don't understand.

        Returns:
            NarrowerResult. Return `NarrowerResult(name=self.name)` (with
            both ID fields as None) to signal "no opinion / not applicable".
        """
        ...
