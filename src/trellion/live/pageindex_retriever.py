# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""PageIndex retriever with graph-guided page-first retrieval.

Retrieval strategy:
1. Embed the query via PageIndexEmbedder
2. If candidate_page_ids are provided (from the graph narrower), filter to those pages
3. Vector-search page_embeddings for top pages
4. For each selected page, vector-search chunks within that page
5. Assemble EvidencePack objects with full page text and Evidence items
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional

import structlog

from trellion.live.evidence import Evidence, evidence_from_chunk

# Graph ranking is supplied by an optional add-on (provider role
# ``graph_ranker``). Basic installs run vector-only ranking: every runtime
# use below is already behind `if self._graph_ranker`.
try:
    from trellion.live._intel import GRAPH_RANKER, addon_provider
    _graph_ranker_factory = addon_provider(GRAPH_RANKER)
    GRAPH_RANKER_AVAILABLE = True
except ImportError:
    _graph_ranker_factory = None
    GRAPH_RANKER_AVAILABLE = False
from trellion.live.pageindex_embedder import PageIndexEmbedder
from trellion.live.pageindex_store import PageIndexStore
from trellion.live.topic_diversity import diversify, topic_family

log = structlog.get_logger(__name__)

# Query embedding LRU cache — avoids re-embedding repeated/similar queries.
# Each embedding is ~3KB (768 floats × 4 bytes), so 1000 entries ≈ 3MB.
_QUERY_EMBED_CACHE: OrderedDict[str, list[float]] = OrderedDict()
_QUERY_EMBED_CACHE_MAX = 1000

# Candidate pages fetched per answer slot. Was 2: with one topic filling
# the first ten, no other topic was there for the diversity cap to pick.
_POOL_FACTOR = 4


@dataclass
class EvidencePack:
    """A single page's worth of retrieved evidence."""

    doc_id: str
    page_id: str
    page_no: int
    title: str
    relevance_score: float
    page_text: str
    evidence_items: list[Evidence] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


class PageIndexRetriever:
    """Retrieve evidence packs using vector search over pages and chunks."""

    def __init__(
        self,
        store: PageIndexStore,
        embedder: PageIndexEmbedder,
        graph_ranking: bool = True,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self._graph_ranker = (
            _graph_ranker_factory() if (graph_ranking and GRAPH_RANKER_AVAILABLE) else None
        )

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        candidate_page_ids: Optional[list[str]] = None,
        candidate_doc_ids: Optional[list[str]] = None,
        seed_entity_ids: Optional[list[str]] = None,
        roles: Optional[list[str]] = None,
    ) -> list[EvidencePack]:
        """Full retrieval pipeline: embed query -> find pages -> find chunks -> build evidence.

        When ``seed_entity_ids`` are provided and graph-guided retrieval is
        enabled, the pipeline additionally:

        * Expands seed entities via intent-weighted graph traversal to find
          related entities whose edge types match the inferred query intent.
        * After the base vector search returns pages, boosts page scores
          based on which graph-expanded entities are mentioned on each page.
        * Re-ranks the results so that pages with strong graph evidence
          float to the top.

        If no seed entities are available, or graph-guided retrieval produces
        no results, the pipeline falls through to standard vector search.

        Args:
            query: Natural-language query string.
            top_k: Maximum number of evidence packs to return.
            candidate_page_ids: Optional page IDs from graph navigator to scope search.
            candidate_doc_ids: Optional doc IDs from non-graph narrowers (metadata,
                temporal, conversation). Independent of candidate_page_ids — both
                may be supplied; both are applied as filters during the search.
            seed_entity_ids: Optional entity IDs for graph-guided expansion.

        Returns:
            List of EvidencePack sorted by relevance score (descending).
        """
        # Step 1: Embed the query (with LRU cache)
        cache_key = hashlib.sha256(query.encode()).hexdigest()[:16]
        query_embedding = _QUERY_EMBED_CACHE.get(cache_key)
        try:
            from ..observability import inc_counter as _metrics_inc
        except Exception:
            _metrics_inc = None  # type: ignore[assignment]

        if query_embedding is not None:
            _QUERY_EMBED_CACHE.move_to_end(cache_key)
            log.info("retriever.query_embedding_cached", dims=len(query_embedding))
            if _metrics_inc:
                await _metrics_inc("query_embedding_cache_hits_total")
        else:
            query_embedding = await self.embedder.embed_text(query)
            _QUERY_EMBED_CACHE[cache_key] = query_embedding
            if len(_QUERY_EMBED_CACHE) > _QUERY_EMBED_CACHE_MAX:
                _QUERY_EMBED_CACHE.popitem(last=False)
            log.info("retriever.query_embedded", dims=len(query_embedding))
            if _metrics_inc:
                await _metrics_inc("query_embedding_cache_misses_total")

        # Step 1b: Graph entity expansion (intent-aware)
        graph_entities = []
        if self._graph_ranker and seed_entity_ids:
            try:
                graph_entities = await self._graph_ranker.get_graph_boosted_entities(
                    query=query,
                    seed_entity_ids=seed_entity_ids,
                    store=self.store,
                    top_k=20,
                )
                log.info(
                    "retriever.graph_ranker_entities",
                    count=len(graph_entities),
                    intent=self._graph_ranker.classify_query_intent(query),
                )
            except Exception:
                log.warning("retriever.graph_ranker_failed", exc_info=True)

        # Step 2: Retrieve pages
        # Three scoping cases (peer-narrower architecture):
        #   1. candidate_page_ids only       → existing graph-style page filter
        #   2. candidate_doc_ids only        → scope vector search to those docs
        #   3. both                          → docs filter + page-set restriction
        #   4. neither                       → unrestricted vector search
        if candidate_page_ids:
            pages = await self._retrieve_pages_by_ids(
                query_embedding,
                candidate_page_ids,
                top_k=top_k * _POOL_FACTOR,
                doc_id_filter=list(candidate_doc_ids) if candidate_doc_ids else None,
                roles=roles,
            )
        elif candidate_doc_ids:
            # Doc-only scoping: search across pages within those docs.
            # No need to convert page_ids → doc_ids first.
            pages = await self.store.search_pages(
                embedding=query_embedding,
                top_k=top_k * _POOL_FACTOR,
                doc_ids=list(candidate_doc_ids),
                roles=roles,
            )
        else:
            pages = await self.retrieve_pages(query_embedding, top_k=top_k * _POOL_FACTOR, roles=roles)

        # Step 2a: chunk-first recall — let chunks compete directly.
        #
        # Everything above scores PAGES, and a page embedding cannot represent
        # a page larger than the model's input window. The embedder sends the
        # whole page; the backend takes only the gateway's embed num_ctx
        # (1024 tokens) and TRUNCATES the rest silently. So a long page is not
        # ranked poorly — most of it was never encoded. Because chunk search then runs only WITHIN the
        # pages that already won, that page's chunks are unreachable no matter
        # how well they match: indexed, embedded, and invisible.
        #
        # Chunks are sized well inside the window (CHUNK_SIZE_CHARS, ~500 tokens),
        # so they are the one representation that is always faithful. This
        # gives them a route into the candidate set.
        #
        # Observed live: a 51k-char meeting transcript lost to the 1k-char
        # "a summary wasn't produced" page beside it, so "what did we discuss"
        # answered "no summaries were produced" while 4,525 embedded
        # transcript chunks sat unread.
        #
        # This searches chunks across the corpus and folds their pages into
        # the candidate set, scored by their BEST chunk. Additive by design:
        # pages that won on their own are untouched and keep their score, so
        # this can only add reachability, never remove it.
        pages = await self._merge_chunk_first_pages(
            query_embedding,
            pages,
            top_k=top_k,
            candidate_page_ids=candidate_page_ids,
            roles=roles,
        )

        if not pages:
            log.info("retriever.no_pages_found")
            return []

        # Step 2b: Graph page score boosting
        if self._graph_ranker and graph_entities and pages:
            try:
                pages = await self._graph_ranker.boost_page_scores(
                    base_results=pages,
                    graph_entities=graph_entities,
                    store=self.store,
                )
                log.info("retriever.pages_graph_boosted", count=len(pages))
            except Exception:
                log.warning("retriever.graph_boost_failed", exc_info=True)

        # Step 2c: Adjust page scores using learned source priors. Read from
        # a persisted file (live/source_priors.py); this tier never imports
        # whatever writes it, so there is nothing to fail per query.
        try:
            from trellion.live.source_priors import get_source_priors

            priors = get_source_priors()
            adjusted = 0
            for page in pages:
                page_id = page.get("page_id", "")
                if not page_id:
                    continue
                prior = priors.get(page_id)
                if prior != 0.5:  # only adjust if we have signal
                    base = page.get("score", 0.0)
                    page["score"] = base * 0.8 + prior * 0.2
                    adjusted += 1
            if adjusted:
                # Re-sort after adjustment
                pages.sort(key=lambda p: p.get("score", 0.0), reverse=True)
                log.info("retriever.source_priors_applied", adjusted=adjusted)
        except Exception as exc:
            # No exc_info: a rich traceback carries locals (query text) to logs.
            log.warning("retriever.source_prior_adjustment_failed", error=type(exc).__name__)

        # Take top pages, at most a fair share per topic family
        # (live/topic_diversity.py): one prolific source must not hide
        # every other topic from a general question.
        pages = await self._diverse_top(pages, top_k)
        log.info("retriever.pages_found", count=len(pages))

        # Step 3: Retrieve chunks within selected pages
        page_ids = [p["page_id"] for p in pages]
        chunks = await self.retrieve_chunks(
            query_embedding, page_ids, top_k=top_k * 3
        )
        log.info("retriever.chunks_found", count=len(chunks))

        # Step 4: Build evidence packs
        evidence = await self.build_evidence(pages, chunks)
        return evidence

    async def retrieve_pages(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        doc_ids: Optional[list[str]] = None,
        roles: Optional[list[str]] = None,
    ) -> list[dict]:
        """Vector search over page embeddings.

        Args:
            query_embedding: The embedded query vector.
            top_k: Max pages to return.
            doc_ids: Optional document IDs to scope the search to.

        Returns:
            List of page dicts with page_id, doc_id, page_no, text, score.
        """
        return await self.store.search_pages(
            embedding=query_embedding, top_k=top_k, doc_ids=doc_ids, roles=roles
        )

    async def retrieve_chunks(
        self,
        query_embedding: list[float],
        page_ids: list[str],
        top_k: int = 5,
    ) -> list[dict]:
        """Vector search over chunk embeddings scoped to specific pages.

        Args:
            query_embedding: The embedded query vector.
            page_ids: Page IDs to scope chunk search to.
            top_k: Max chunks to return.

        Returns:
            List of chunk dicts with chunk_id, page_id, chunk_no, text, score.
        """
        if not page_ids:
            return []
        return await self.store.search_chunks(
            embedding=query_embedding, top_k=top_k, page_ids=page_ids
        )

    async def build_evidence(
        self, pages: list[dict], chunks: list[dict]
    ) -> list[EvidencePack]:
        """Assemble evidence packs from retrieved pages and chunks.

        Groups chunks by page_id, then builds an EvidencePack for each page
        with its full text and Evidence items built from each chunk.
        """
        # Group chunks by page_id
        chunks_by_page: dict[str, list[dict]] = {}
        for chunk in chunks:
            chunks_by_page.setdefault(chunk["page_id"], []).append(chunk)

        # Look up document titles in a single query (avoids N+1)
        doc_ids_needed = list({p["doc_id"] for p in pages})
        doc_titles = await self._get_doc_titles_batch(doc_ids_needed)

        evidence: list[EvidencePack] = []
        for page in pages:
            page_id = page["page_id"]
            page_chunks = chunks_by_page.get(page_id, [])
            # Sort chunks by score descending
            page_chunks.sort(key=lambda c: c.get("score", 0), reverse=True)

            # Build Evidence items from each chunk
            evidence_items = [
                evidence_from_chunk(chunk, page)
                for chunk in page_chunks
            ]

            # Get page metadata
            page_detail = await self.store.get_page(page_id)
            meta = page_detail.get("meta") or {} if page_detail else {}

            # Propagate graph-guided retrieval annotations into meta
            if page.get("graph_evidence"):
                meta["graph_evidence"] = page["graph_evidence"]
                meta["graph_boost"] = page.get("graph_boost", 0.0)

            pack = EvidencePack(
                doc_id=page["doc_id"],
                page_id=page_id,
                page_no=page.get("page_no", 0),
                title=doc_titles.get(page["doc_id"], ""),
                relevance_score=page.get("score", 0.0),
                page_text=page.get("text", ""),
                evidence_items=evidence_items,
                meta=meta,
            )
            evidence.append(pack)

        # Sort by relevance score descending
        evidence.sort(key=lambda e: e.relevance_score, reverse=True)
        return evidence

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _merge_chunk_first_pages(
        self,
        query_embedding: list[float],
        pages: list[dict],
        top_k: int,
        candidate_page_ids: Optional[list[str]] = None,
        roles: Optional[list[str]] = None,
    ) -> list[dict]:
        """Fold chunk-level winners into a page-level candidate set.

        Scores a page by its best chunk and takes the MAX against any score it
        already had, so a page never loses ground by being found twice. Pages
        discovered only by chunk search are appended with their best chunk's
        score.

        Scoped exactly like the page search it complements: when the graph
        narrowed the search to `candidate_page_ids`, chunk search is confined
        to the same set, so this widens recall within the scope rather than
        escaping it. `roles` is passed through — audience filtering is not
        optional on this path either.

        Failure is non-fatal. This is a recall improvement layered on a
        pipeline that already works; if chunk search errors the page-level
        results stand rather than the query failing.
        """
        try:
            chunks = await self.store.search_chunks(
                embedding=query_embedding,
                top_k=max(top_k * 4, 20),
                page_ids=list(candidate_page_ids) if candidate_page_ids else None,
                roles=roles,
            )
        except Exception:
            log.warning("retriever.chunk_first_failed", exc_info=True)
            return pages
        if not chunks:
            return pages

        best_by_page: dict[str, float] = {}
        for c in chunks:
            pid = str(c.get("page_id") or "")
            if not pid:
                continue
            score = float(c.get("score", 0.0))
            if score > best_by_page.get(pid, float("-inf")):
                best_by_page[pid] = score

        known = {str(p.get("page_id") or "") for p in pages}
        for page in pages:
            pid = str(page.get("page_id") or "")
            if pid in best_by_page:
                page["score"] = max(float(page.get("score", 0.0)), best_by_page[pid])

        new_ids = [pid for pid in best_by_page if pid not in known]
        added = 0
        if new_ids:
            fetched = await self._retrieve_pages_by_ids(
                query_embedding, new_ids, top_k=len(new_ids), roles=roles
            )
            for page in fetched:
                pid = str(page.get("page_id") or "")
                # The page's own embedding is why it lost; its best chunk is
                # the evidence that it should be here.
                page["score"] = max(
                    float(page.get("score", 0.0)), best_by_page.get(pid, 0.0)
                )
                pages.append(page)
                added += 1

        pages.sort(key=lambda p: p.get("score", 0.0), reverse=True)
        log.info(
            "retriever.chunk_first_merged",
            chunks=len(chunks),
            pages_added=added,
            pages_rescored=len(best_by_page) - added,
        )
        return pages

    async def _retrieve_pages_by_ids(
        self,
        query_embedding: list[float],
        candidate_page_ids: list[str],
        top_k: int = 10,
        doc_id_filter: Optional[list[str]] = None,
        roles: Optional[list[str]] = None,
    ) -> list[dict]:
        """Retrieve pages filtered to a set of candidate page_ids.

        Since store.search_pages filters by doc_ids, we first resolve
        candidate_page_ids to their doc_ids, then search within those docs
        and filter the results to only include candidate pages.

        Args:
            doc_id_filter: Optional additional doc-level scope (e.g. from
                MetadataNarrower). When provided, the candidate page set is
                further restricted to pages whose doc_id is in this list —
                AND-composition with the page-level filter.
        """
        if not candidate_page_ids:
            return await self.retrieve_pages(query_embedding, top_k=top_k, roles=roles)

        # Resolve page_ids to doc_ids
        doc_ids: set[str] = set()
        for pid in candidate_page_ids:
            page = await self.store.get_page(pid)
            if page:
                doc_ids.add(page["doc_id"])

        # AND-compose with optional doc-level narrower scope
        if doc_id_filter:
            doc_ids = doc_ids & set(doc_id_filter)

        if not doc_ids:
            # Fallback to role-scoped (not fully unrestricted) search
            return await self.retrieve_pages(query_embedding, top_k=top_k, roles=roles)

        # Search within candidate docs, over-fetch to compensate for filtering
        pages = await self.store.search_pages(
            embedding=query_embedding,
            top_k=top_k * 3,
            doc_ids=list(doc_ids),
            roles=roles,
        )

        # Filter to only candidate page_ids
        candidate_set = set(candidate_page_ids)
        filtered = [p for p in pages if p["page_id"] in candidate_set]

        # If graph filtering is too aggressive, backfill — but stay inside
        # the requester's role scope, never the whole KB.
        if len(filtered) < top_k:
            unrestricted = await self.store.search_pages(
                embedding=query_embedding, top_k=top_k, roles=roles
            )
            seen = {p["page_id"] for p in filtered}
            for p in unrestricted:
                if p["page_id"] not in seen:
                    filtered.append(p)
                    seen.add(p["page_id"])
                if len(filtered) >= top_k:
                    break

        return filtered[:top_k]

    async def _diverse_top(self, pages: list[dict], top_k: int) -> list[dict]:
        """``pages[:top_k]`` with a per-family cap; plain truncation if the
        titles cannot be read."""
        if len(pages) <= top_k:
            return pages
        try:
            titles = await self._get_doc_titles_batch(
                list({str(p.get("doc_id", "")) for p in pages if p.get("doc_id")})
            )
        except Exception as exc:
            log.warning("retriever.topic_diversity_skipped", error=type(exc).__name__)
            return pages[:top_k]
        picked = diversify(
            pages,
            lambda p: topic_family(titles.get(str(p.get("doc_id", "")), ""), str(p.get("doc_id", ""))),
            top_k,
        )
        if [p.get("page_id") for p in picked] != [p.get("page_id") for p in pages[:top_k]]:
            log.info("retriever.topic_diversity_applied", top_k=top_k, pool=len(pages))
        return picked

    async def _get_doc_title(self, doc_id: str) -> str:
        """Look up a document title by doc_id."""
        titles = await self._get_doc_titles_batch([doc_id])
        return titles.get(doc_id, "")

    async def _get_doc_titles_batch(self, doc_ids: list[str]) -> dict[str, str]:
        """Look up document titles for multiple doc_ids in a single query."""
        if not doc_ids:
            return {}
        from sqlalchemy import select
        from ..db.models import Doc
        _doc = Doc.__table__

        async with self.store._engine.connect() as conn:
            result = await conn.execute(
                select(_doc.c.doc_id, _doc.c.title)
                .where(_doc.c.doc_id.in_(doc_ids))
            )
            return {str(row[0]): (row[1] or "") for row in result.fetchall()}
