# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""NarrowerOrchestrator — runs narrowers in parallel and combines outputs.

This is intentionally separate from `RAGOrchestrator` so the narrower
combine logic can evolve (modes, weighting, fallback policies) without
touching the higher-level retrieval strategy code.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from .base import CombinedNarrowing, CombineMode, Narrower, NarrowerResult

log = logging.getLogger(__name__)


class NarrowerOrchestrator:
    """Runs a list of narrowers and combines their outputs.

    Failure isolation: if one narrower raises, the orchestrator logs and
    continues with the others. A failing narrower is treated as "no
    opinion" — the combine logic doesn't penalize the final set.
    """

    def __init__(self, narrowers: list[Narrower]):
        self.narrowers = narrowers

    async def run(
        self,
        query: str,
        ctx: Optional[dict] = None,
        mode: CombineMode = CombineMode.INTERSECT,
        timeout_s: float = 10.0,
    ) -> CombinedNarrowing:
        """Run all narrowers in parallel, combine results.

        Args:
            query: Natural-language query.
            ctx: Optional context dict passed to each narrower.
            mode: How to combine outputs (intersect / union / weighted).
            timeout_s: Per-narrower timeout. A narrower that exceeds this
                is dropped from the combine (treated as "no opinion").

        Returns:
            CombinedNarrowing with merged candidate sets.
        """
        ctx = ctx or {}
        if not self.narrowers:
            return CombinedNarrowing(mode=mode)

        # Per-narrower timeout — wrap each call so one slow narrower can't
        # block the whole pipeline. Failure isolation: gather with
        # return_exceptions, filter exceptions out before combine.
        async def _run_one(n: Narrower) -> Optional[NarrowerResult]:
            try:
                return await asyncio.wait_for(n.narrow(query, ctx), timeout=timeout_s)
            except asyncio.TimeoutError:
                log.warning("narrower.%s.timeout", n.name)
                return None
            except Exception:
                log.warning("narrower.%s.failed", n.name, exc_info=True)
                return None

        results = await asyncio.gather(*(_run_one(n) for n in self.narrowers))
        valid = [r for r in results if r is not None and r.has_opinion]

        if not valid:
            return CombinedNarrowing(mode=mode)

        return self._combine(valid, mode)

    def _combine(
        self,
        results: list[NarrowerResult],
        mode: CombineMode,
    ) -> CombinedNarrowing:
        """Merge per-narrower results according to `mode`.

        Doc and page sets are combined independently — a narrower that
        only contributes doc_ids doesn't affect the page-level combine,
        and vice versa.
        """
        contributors = [r.name for r in results]

        # Split results by which granularity they contributed at
        doc_results = [r for r in results if r.doc_ids is not None]
        page_results = [r for r in results if r.page_ids is not None]

        if mode == CombineMode.INTERSECT:
            doc_set = self._intersect([r.doc_ids for r in doc_results])
            page_set = self._intersect([r.page_ids for r in page_results])
        elif mode == CombineMode.UNION:
            doc_set = self._union([r.doc_ids for r in doc_results])
            page_set = self._union([r.page_ids for r in page_results])
        elif mode == CombineMode.WEIGHTED:
            # Weighted is more involved (per-doc score accumulation, top-N
            # selection by combined score). Defer to a later iteration; for
            # the first cut, weighted falls back to UNION with a warning.
            log.warning("narrower.combine.weighted_not_implemented_falling_back_to_union")
            doc_set = self._union([r.doc_ids for r in doc_results])
            page_set = self._union([r.page_ids for r in page_results])
        else:
            raise ValueError(f"Unknown combine mode: {mode}")

        return CombinedNarrowing(
            candidate_doc_ids=doc_set,
            candidate_page_ids=page_set,
            contributors=contributors,
            mode=mode,
        )

    @staticmethod
    def _intersect(sets: list[Optional[set[str]]]) -> Optional[set[str]]:
        """Intersect non-None sets. Returns None if no narrower had an opinion
        at this granularity, empty set if narrowers disagreed completely."""
        non_none = [s for s in sets if s is not None]
        if not non_none:
            return None
        result = non_none[0]
        for s in non_none[1:]:
            result = result & s
        return result

    @staticmethod
    def _union(sets: list[Optional[set[str]]]) -> Optional[set[str]]:
        """Union non-None sets. Returns None if no narrower had an opinion."""
        non_none = [s for s in sets if s is not None]
        if not non_none:
            return None
        result: set[str] = set()
        for s in non_none:
            result = result | s
        return result
