# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""RAG orchestrator that ties together graph navigation, page retrieval,
persona biasing, and evidence verification.

Given a natural-language question, the orchestrator:
1. Classifies the question intent via keyword heuristics
2. Optionally applies persona lens biasing
3. Routes to the appropriate retrieval strategy (lookup / graph-first / hybrid)
4. Runs verification with retry loop
5. Publishes NATS query lifecycle events
6. Builds structured citations from the evidence packs
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any, Optional

import structlog

from trellion.live.evidence import Evidence
from trellion.live.graph_types import GraphNarrower, GraphResult, Hypothesis

# The graph narrower is an optional collaborator an add-on may provide
# (trellion.live._intel.addon_provider). Every strategy below already
# handles `graph_result is None`.
try:
    from trellion.live._intel import GRAPH_NARROWER, addon_provider
    addon_provider(GRAPH_NARROWER)
    GRAPH_NARROWER_AVAILABLE = True
except ImportError:
    GRAPH_NARROWER_AVAILABLE = False
from trellion.live.pageindex_retriever import EvidencePack, PageIndexRetriever
from trellion.live.pageindex_store import PageIndexStore
from trellion.live.persona import PersonaAdapter, PersonaLens
from trellion.live.rag_generator import RAGAnswerGenerator
from trellion.live.rag_verifier import RAGVerifier, VerificationResult

log = structlog.get_logger(__name__)


@dataclass
class Citation:
    """A structured reference to a specific page in a document."""

    doc_id: str
    title: str
    page_no: int
    page_id: str
    excerpt: str
    span: list[int] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class RAGResponse:
    """Result of a RAG orchestration query."""

    answer: str  # Empty string; filled by LLM caller later
    evidence: list[EvidencePack] = field(default_factory=list)
    graph_context: Optional[GraphResult] = None
    strategy_used: str = ""
    intent: str = ""
    citations: list[Citation] = field(default_factory=list)
    verification: Optional[VerificationResult] = None
    hypotheses: list[Hypothesis] = field(default_factory=list)


# Keyword patterns for intent classification
_INTENT_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(why|cause|reason|caused)\b", re.IGNORECASE), "why"),
    (re.compile(r"\b(compare|comparison|difference|differ|vs\.?|versus)\b", re.IGNORECASE), "comparison"),
    (re.compile(r"\b(what\s+is|define|definition|how\s+much|how\s+many)\b", re.IGNORECASE), "lookup"),
    (re.compile(r"\b(fix|error|problem|debug|issue|broken|failing|crash)\b", re.IGNORECASE), "troubleshooting"),
]

# Map intent to strategy
_INTENT_STRATEGY: dict[str, str] = {
    "lookup": "lookup",
    "why": "graph_first",
    "troubleshooting": "graph_first",
    "comparison": "hybrid",
    "explanation": "hybrid",
}

# Maximum verification retries (3 total attempts)
_MAX_VERIFY_RETRIES = 2


class RAGOrchestrator:
    """Orchestrates graph navigation and page retrieval for question answering."""

    def __init__(
        self,
        navigator: Optional[GraphNarrower],
        retriever: PageIndexRetriever,
        store: PageIndexStore,
        messaging=None,
        verifier: Optional[RAGVerifier] = None,
        generator: Optional[RAGAnswerGenerator] = None,
        video_memory: Optional[Any] = None,
        narrower_orchestrator: Optional[Any] = None,
    ) -> None:
        """Args:
            narrower_orchestrator: Optional NarrowerOrchestrator that runs
                doc-level narrowers (metadata, temporal, conversation) in
                parallel and produces candidate_doc_ids for the retriever.
                When provided, the strategies pass `candidate_doc_ids` to
                `retriever.retrieve()` alongside any `candidate_page_ids`
                from graph navigation. The two filters AND-compose at the
                retriever — narrower_orchestrator's doc-level scope further
                restricts the graph's page-level scope when both are present.
                Optional so existing callers/tests don't need to update.
        """
        self.navigator = navigator  # None on Basic-tier installs: strategies skip graph navigation
        self.retriever = retriever
        self.store = store
        self.messaging = messaging
        self.verifier = verifier
        self.generator = generator
        self.video_memory = video_memory
        self.narrower_orchestrator = narrower_orchestrator

    async def _navigate(self, question: str) -> Optional[GraphResult]:
        """Run graph navigation, or return None when no navigator is wired."""
        if self.navigator is None:
            return None
        return await self.navigator.navigate(question)

    async def query(
        self,
        question: str,
        strategy: str = "hybrid",
        persona_lens: Optional[PersonaLens] = None,
        narrower_ctx: Optional[dict] = None,
        roles: Optional[list[str]] = None,
    ) -> RAGResponse:
        """Run the full RAG pipeline for a question.

        Args:
            question: Natural-language question.
            strategy: Override strategy ("hybrid", "lookup", "graph_first").
                      If "hybrid" (default), the intent classifier picks automatically.
            persona_lens: Optional persona lens for biased retrieval.
            narrower_ctx: Optional dict passed to each registered narrower.
                Common keys: ``metadata_filters`` (dict for MetadataNarrower),
                ``time_range`` (TemporalNarrower, future), ``conversation_id``
                (ConversationNarrower, future). Ignored when no
                narrower_orchestrator is configured.

        Returns:
            RAGResponse with evidence, graph context, citations, verification, and hypotheses.
        """
        await self._publish_query_event("query.received", {
            "question": question[:200],
            "strategy": strategy,
            "persona_id": persona_lens.persona_id if persona_lens else None,
        })

        intent = self.classify_intent(question)

        # Apply persona lens: override strategy and augment query
        effective_query = question
        if persona_lens:
            if strategy == "hybrid":
                strategy = persona_lens.preferred_strategy
            effective_query = PersonaAdapter.apply_to_query(persona_lens, question)

        # Use intent-based strategy unless caller explicitly overrides
        if strategy == "hybrid":
            resolved_strategy = _INTENT_STRATEGY.get(intent, "hybrid")
        else:
            resolved_strategy = strategy

        log.info(
            "rag_orchestrator.query",
            intent=intent,
            strategy=resolved_strategy,
            question=question[:80],
            persona=persona_lens.name if persona_lens else None,
        )

        # System 1: launch parallel video memory search
        video_search_task = None
        if self.video_memory:
            video_search_task = asyncio.create_task(
                self._search_video_memory(effective_query)
            )

        if resolved_strategy == "lookup":
            response = await self.execute_lookup(
                effective_query, persona_lens=persona_lens, narrower_ctx=narrower_ctx,
                roles=roles,
            )
        elif resolved_strategy == "graph_first":
            response = await self.execute_graph_first(
                effective_query, persona_lens=persona_lens, narrower_ctx=narrower_ctx,
                roles=roles,
            )
        else:
            response = await self.execute_hybrid(
                effective_query, persona_lens=persona_lens, narrower_ctx=narrower_ctx,
                roles=roles,
            )

        response.intent = intent
        response.strategy_used = resolved_strategy

        # Copy hypotheses from graph context
        if response.graph_context and response.graph_context.hypotheses:
            response.hypotheses = response.graph_context.hypotheses

        # Merge System 1 (video memory) results if available
        if video_search_task:
            try:
                video_evidence = await video_search_task
                if video_evidence:
                    response.evidence = self._merge_evidence(
                        video_evidence, response.evidence
                    )
                    response.citations = self._build_citations(response.evidence)
                    log.info(
                        "rag_orchestrator.video_memory_merged",
                        count=len(video_evidence),
                    )
            except Exception as e:
                log.warning("rag_orchestrator.video_memory_failed", error=str(e))

        # Run verification with retry
        if self.verifier:
            response = await self._verify_and_retry(response, effective_query, roles=roles)

        # Generate answer from evidence
        if self.generator and response.evidence:
            response.answer = await self.generator.generate(
                query=question,
                evidence_packs=response.evidence,
                verification=response.verification,
                hypotheses=response.hypotheses or None,
            )

        return response

    async def _run_narrowers(
        self, question: str, narrower_ctx: Optional[dict],
    ) -> tuple[Optional[list[str]], list[str]]:
        """Run the configured narrower orchestrator (if any).

        Returns:
            (candidate_doc_ids, contributors). Both are empty/None when no
            orchestrator is configured or no narrower had an opinion. The
            contributors list is purely for telemetry/debug.
        """
        if not self.narrower_orchestrator:
            return None, []
        try:
            combined = await self.narrower_orchestrator.run(question, ctx=narrower_ctx or {})
        except Exception as e:
            log.warning("rag_orchestrator.narrowers_failed", error=str(e))
            return None, []
        if combined.candidate_doc_ids is None:
            return None, combined.contributors
        return list(combined.candidate_doc_ids), combined.contributors

    def classify_intent(self, question: str) -> str:
        """Classify the question intent using keyword heuristics.

        Returns one of: why, comparison, lookup, troubleshooting, explanation.
        """
        for pattern, intent in _INTENT_PATTERNS:
            if pattern.search(question):
                return intent
        return "explanation"

    async def execute_lookup(
        self,
        question: str,
        *,
        persona_lens: Optional[PersonaLens] = None,
        narrower_ctx: Optional[dict] = None,
        roles: Optional[list[str]] = None,
    ) -> RAGResponse:
        """Page retrieval only -- no graph navigation.

        Best for simple fact lookups where relationship traversal adds no value.
        Doc-level narrowers (when configured) still apply — they restrict the
        vector search to a subset of docs without involving the graph.
        """
        candidate_doc_ids, contributors = await self._run_narrowers(question, narrower_ctx)

        evidence = await self.retriever.retrieve(
            question, candidate_doc_ids=candidate_doc_ids, roles=roles,
        )
        citations = self._build_citations(evidence)

        await self._publish_query_event("query.evidence.ready", {
            "evidence_count": len(evidence),
            "strategy": "lookup",
            "narrowers": contributors,
        })

        log.info(
            "rag_orchestrator.lookup_done evidence_count=%d narrowers=%s",
            len(evidence), contributors,
        )
        return RAGResponse(
            answer="",
            evidence=evidence,
            graph_context=None,
            citations=citations,
        )

    async def execute_graph_first(
        self,
        question: str,
        *,
        persona_lens: Optional[PersonaLens] = None,
        narrower_ctx: Optional[dict] = None,
        roles: Optional[list[str]] = None,
    ) -> RAGResponse:
        """Graph navigation first, then page retrieval scoped to candidate pages.

        Best for causal / relational queries where the knowledge graph can
        narrow down which pages contain the answer.

        When a narrower_orchestrator is configured, doc-level narrowers run
        IN PARALLEL with graph navigation (they're independent), and their
        candidate_doc_ids further restrict the page-level scope from the
        graph. AND-composition happens inside the retriever.
        """
        # Graph nav and narrowers are independent — run them concurrently.
        graph_task = asyncio.create_task(self._navigate(question))
        narrower_task = asyncio.create_task(self._run_narrowers(question, narrower_ctx))
        graph_result = await graph_task
        candidate_doc_ids, contributors = await narrower_task

        await self._publish_query_event("query.candidates.ready", {
            "candidate_page_count": len(graph_result.candidate_page_ids) if graph_result else 0,
            "seed_count": len(graph_result.seed_entities) if graph_result else 0,
            "narrower_doc_count": len(candidate_doc_ids) if candidate_doc_ids else 0,
            "narrowers": contributors,
        })

        # Bias seed entities if persona lens provided
        if persona_lens and graph_result and graph_result.seed_entities:
            graph_result.seed_entities = PersonaAdapter.bias_entity_scores(
                persona_lens, graph_result.seed_entities
            )

        candidate_page_ids = graph_result.candidate_page_ids if graph_result else []
        seed_entity_ids = (
            [e["entity_id"] for e in graph_result.seed_entities]
            if graph_result and graph_result.seed_entities
            else None
        )

        if candidate_page_ids:
            evidence = await self.retriever.retrieve(
                question,
                candidate_page_ids=candidate_page_ids,
                candidate_doc_ids=candidate_doc_ids,
                seed_entity_ids=seed_entity_ids,
                roles=roles,
            )
        else:
            # Graph found nothing useful; fall back to retrieval scoped only
            # by doc-level narrowers (if any) or unrestricted otherwise.
            log.info("rag_orchestrator.graph_first_fallback")
            evidence = await self.retriever.retrieve(
                question,
                candidate_doc_ids=candidate_doc_ids,
                seed_entity_ids=seed_entity_ids,
                roles=roles,
            )

        citations = self._build_citations(evidence)

        await self._publish_query_event("query.evidence.ready", {
            "evidence_count": len(evidence),
            "strategy": "graph_first",
            "narrowers": contributors,
        })

        log.info(
            "rag_orchestrator.graph_first_done graph_pages=%d narrower_docs=%s evidence_count=%d",
            len(candidate_page_ids),
            len(candidate_doc_ids) if candidate_doc_ids else 0,
            len(evidence),
        )
        return RAGResponse(
            answer="",
            evidence=evidence,
            graph_context=graph_result,
            citations=citations,
        )

    async def execute_hybrid(
        self,
        question: str,
        *,
        persona_lens: Optional[PersonaLens] = None,
        narrower_ctx: Optional[dict] = None,
        roles: Optional[list[str]] = None,
    ) -> RAGResponse:
        """Graph navigation + unrestricted page search, merged.

        Runs both the graph navigator and the page retriever independently,
        then merges results (graph-scoped evidence first, then unrestricted
        results to fill remaining slots).

        When narrowers are configured, doc-level scope from narrowers
        applies to BOTH retrieval branches — the "unrestricted" branch is
        actually scoped to the narrower's doc set when one is present.
        """
        # Graph nav and narrowers are independent — run concurrently.
        graph_task = asyncio.create_task(self._navigate(question))
        narrower_task = asyncio.create_task(self._run_narrowers(question, narrower_ctx))
        graph_result = await graph_task
        candidate_doc_ids, contributors = await narrower_task

        await self._publish_query_event("query.candidates.ready", {
            "candidate_page_count": len(graph_result.candidate_page_ids) if graph_result else 0,
            "seed_count": len(graph_result.seed_entities) if graph_result else 0,
            "narrower_doc_count": len(candidate_doc_ids) if candidate_doc_ids else 0,
            "narrowers": contributors,
        })

        # Bias seed entities if persona lens provided
        if persona_lens and graph_result and graph_result.seed_entities:
            graph_result.seed_entities = PersonaAdapter.bias_entity_scores(
                persona_lens, graph_result.seed_entities
            )

        seed_entity_ids = (
            [e["entity_id"] for e in graph_result.seed_entities]
            if graph_result and graph_result.seed_entities
            else None
        )

        # Run both retrievals in parallel. Narrower doc-scope (if any)
        # applies to BOTH branches — the unrestricted branch is only
        # unrestricted vs the GRAPH; narrowers still scope it.
        unrestricted_task = asyncio.create_task(
            self.retriever.retrieve(
                question,
                candidate_doc_ids=candidate_doc_ids,
                seed_entity_ids=seed_entity_ids,
                roles=roles,
            )
        )

        graph_evidence: list[EvidencePack] = []
        graph_scoped_task = None
        if graph_result and graph_result.candidate_page_ids:
            graph_scoped_task = asyncio.create_task(
                self.retriever.retrieve(
                    question,
                    candidate_page_ids=graph_result.candidate_page_ids,
                    candidate_doc_ids=candidate_doc_ids,
                    seed_entity_ids=seed_entity_ids,
                    roles=roles,
                )
            )

        unrestricted_evidence = await unrestricted_task
        if graph_scoped_task:
            graph_evidence = await graph_scoped_task

        # Merge: graph-scoped evidence first, then unrestricted (deduplicated)
        evidence = self._merge_evidence(graph_evidence, unrestricted_evidence)
        citations = self._build_citations(evidence)

        await self._publish_query_event("query.evidence.ready", {
            "evidence_count": len(evidence),
            "strategy": "hybrid",
        })

        log.info(
            "rag_orchestrator.hybrid_done",
            graph_evidence=len(graph_evidence),
            unrestricted_evidence=len(unrestricted_evidence),
            merged_evidence=len(evidence),
        )
        return RAGResponse(
            answer="",
            evidence=evidence,
            graph_context=graph_result,
            citations=citations,
        )

    # ------------------------------------------------------------------
    # System 1 (video memory) search
    # ------------------------------------------------------------------

    async def _search_video_memory(self, query: str) -> list[EvidencePack]:
        """Run System 1 video memory search and promote results to EvidencePacks."""
        try:
            return await self.video_memory.search_with_evidence(query, top_k=3)
        except Exception as e:
            log.warning("video_memory_search_error", error=str(e))
            return []

    # ------------------------------------------------------------------
    # Verification retry loop
    # ------------------------------------------------------------------

    async def _verify_and_retry(
        self, response: RAGResponse, query: str,
        roles: Optional[list[str]] = None,
    ) -> RAGResponse:
        """Verify evidence and retry retrieval if claims are unsupported.

        Max 2 retries (3 total attempts).  Uses ``required_retrieval`` hints
        from the verifier for targeted re-retrieval.
        """
        # Collect all evidence items across packs
        all_evidence = self._collect_evidence_items(response.evidence)

        # Use evidence snippets as the "claims" to verify against themselves
        # (In a full pipeline, claims come from the LLM answer; here we verify
        # that retrieved evidence is self-consistent and has adequate coverage.)
        claims = [e.snippet for e in all_evidence if e.snippet]

        if not claims:
            response.verification = VerificationResult(
                status="approved", confidence=1.0
            )
            await self._publish_query_event("query.verified", {
                "status": "approved",
                "confidence": 1.0,
            })
            return response

        verification = self.verifier.verify_with_evidence(all_evidence, claims)

        for attempt in range(_MAX_VERIFY_RETRIES):
            if verification.status == "approved":
                break

            if not verification.required_retrieval:
                break

            log.info(
                "rag_orchestrator.retry_retrieval",
                attempt=attempt + 1,
                hints=verification.required_retrieval[:3],
            )

            # Build a focused re-retrieval query from hints
            hint_query = " ".join(verification.required_retrieval[:3])
            extra_evidence = await self.retriever.retrieve(hint_query, top_k=3, roles=roles)
            response.evidence = self._merge_evidence(response.evidence, extra_evidence)
            response.citations = self._build_citations(response.evidence)

            # Re-verify with enriched evidence
            all_evidence = self._collect_evidence_items(response.evidence)
            claims = [e.snippet for e in all_evidence if e.snippet]
            verification = self.verifier.verify_with_evidence(all_evidence, claims)

        # If still failing after all attempts, mark as partial
        if verification.status == "rejected":
            verification.status = "partial"

        response.verification = verification

        await self._publish_query_event("query.verified", {
            "status": verification.status,
            "confidence": verification.confidence,
            "supported": len(verification.supported_claims),
            "unsupported": len(verification.unsupported_claims),
        })

        return response

    # ------------------------------------------------------------------
    # NATS event publishing
    # ------------------------------------------------------------------

    async def _publish_query_event(self, subject: str, data: dict) -> None:
        """Fire-and-forget NATS event publishing, guarded."""
        if not self.messaging:
            return
        try:
            if self.messaging.is_connected:
                await self.messaging.publish_event(subject, data)
        except Exception:
            log.debug("rag_orchestrator.event_publish_failed", subject=subject)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_citations(evidence: list[EvidencePack]) -> list[Citation]:
        """Build structured citations from evidence packs."""
        citations: list[Citation] = []
        for pack in evidence:
            if pack.evidence_items:
                # Use the top evidence item for the citation
                top = pack.evidence_items[0]
                citations.append(
                    Citation(
                        doc_id=pack.doc_id,
                        title=pack.title,
                        page_no=pack.page_no,
                        page_id=pack.page_id,
                        excerpt=top.snippet,
                        span=list(top.span),
                        confidence=top.confidence,
                    )
                )
            else:
                citations.append(
                    Citation(
                        doc_id=pack.doc_id,
                        title=pack.title,
                        page_no=pack.page_no,
                        page_id=pack.page_id,
                        excerpt=pack.page_text[:200],
                        span=[],
                        confidence=pack.relevance_score,
                    )
                )
        return citations

    @staticmethod
    def _merge_evidence(
        primary: list[EvidencePack],
        secondary: list[EvidencePack],
        max_total: int = 10,
    ) -> list[EvidencePack]:
        """Merge two evidence lists, deduplicating by page_id.

        Primary evidence (e.g. graph-scoped) takes precedence.
        """
        seen_page_ids: set[str] = set()
        merged: list[EvidencePack] = []

        for pack in primary:
            if pack.page_id not in seen_page_ids:
                seen_page_ids.add(pack.page_id)
                merged.append(pack)

        for pack in secondary:
            if pack.page_id not in seen_page_ids:
                seen_page_ids.add(pack.page_id)
                merged.append(pack)
            if len(merged) >= max_total:
                break

        return merged[:max_total]

    @staticmethod
    def _collect_evidence_items(evidence: list[EvidencePack]) -> list[Evidence]:
        """Flatten all Evidence items from a list of EvidencePacks."""
        items: list[Evidence] = []
        for pack in evidence:
            items.extend(pack.evidence_items)
        return items
