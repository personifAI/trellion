# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""RAG verification agent for checking claims against evidence.

Uses heuristic substring matching to verify that every claim has
at least one supporting Evidence item.  Returns structured status
with ``required_retrieval`` hints for retry loops.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import structlog

from trellion.live.evidence import Evidence

log = structlog.get_logger(__name__)

# Minimum sentence length to consider as a verifiable claim
_MIN_CLAIM_LENGTH = 10


@dataclass
class VerificationResult:
    """Result of verifying claims against evidence."""

    status: str  # "approved" | "rejected" | "partial"
    supported_claims: list[str] = field(default_factory=list)
    unsupported_claims: list[str] = field(default_factory=list)
    missing_support: list[str] = field(default_factory=list)
    required_retrieval: list[str] = field(default_factory=list)
    confidence: float = 0.0


class RAGVerifier:
    """Verify that claims are supported by retrieved evidence."""

    def verify_with_evidence(
        self,
        evidence_items: list[Evidence],
        claims: list[str],
    ) -> VerificationResult:
        """Verify claims against a list of Evidence objects.

        Each claim requires >= 1 Evidence with a matching snippet.
        Unsupported claims produce ``required_retrieval`` key-term hints.

        Args:
            evidence_items: Flat list of Evidence from all evidence packs.
            claims: Sentences / assertions to verify.

        Returns:
            VerificationResult with status, supported/unsupported claims,
            and key-term hints for targeted re-retrieval.
        """
        # Filter trivially short claims
        real_claims = [c for c in claims if len(c.strip()) >= _MIN_CLAIM_LENGTH]

        if not real_claims:
            log.info("rag_verifier.no_claims")
            return VerificationResult(status="approved", confidence=1.0)

        # Build lowered snippet corpus from evidence items
        evidence_texts = [e.snippet.lower() for e in evidence_items if e.snippet]

        supported: list[str] = []
        unsupported: list[str] = []
        missing_support: list[str] = []

        for claim in real_claims:
            if self._is_claim_supported(claim, evidence_texts):
                supported.append(claim)
            else:
                unsupported.append(claim)
                missing_support.append(claim)

        total = len(real_claims)
        confidence = len(supported) / total if total > 0 else 0.0

        # Extract key terms from unsupported claims for re-retrieval
        required_retrieval: list[str] = []
        for claim in unsupported:
            terms = self._extract_retrieval_terms(claim)
            for t in terms:
                if t not in required_retrieval:
                    required_retrieval.append(t)

        # Determine status
        if confidence >= 0.8:
            status = "approved"
        elif confidence >= 0.3:
            status = "partial"
        else:
            status = "rejected"

        log.info(
            "rag_verifier.result",
            status=status,
            total_claims=total,
            supported=len(supported),
            unsupported=len(unsupported),
            confidence=round(confidence, 3),
            retrieval_hints=len(required_retrieval),
        )

        return VerificationResult(
            status=status,
            supported_claims=supported,
            unsupported_claims=unsupported,
            missing_support=missing_support,
            required_retrieval=required_retrieval,
            confidence=confidence,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split text into sentences using punctuation boundaries."""
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        return [p.strip() for p in parts if p.strip()]

    @staticmethod
    def _extract_retrieval_terms(claim: str) -> list[str]:
        """Extract multi-word key phrases from an unsupported claim.

        Returns 2-3 word sequences suitable as retrieval queries.
        """
        words = re.findall(r"[a-z0-9]+", claim.lower())
        # Filter stop-like words
        content_words = [w for w in words if len(w) >= 4]
        terms: list[str] = []
        # Build bigrams from content words
        for i in range(len(content_words) - 1):
            terms.append(f"{content_words[i]} {content_words[i + 1]}")
        # Also add individual content words longer than 5 chars
        for w in content_words:
            if len(w) >= 6 and w not in terms:
                terms.append(w)
        return terms[:5]  # Cap at 5 hints per claim

    @staticmethod
    def _is_claim_supported(claim: str, evidence_texts: list[str]) -> bool:
        """Check if a claim is supported by any evidence text.

        Uses a two-pass approach:
        1. Check if any key phrase from the claim appears in evidence
        2. Check if enough individual content words overlap
        """
        claim_lower = claim.lower()

        # Pass 1: Key phrase matching (3+ word sequences)
        words = re.findall(r"[a-z0-9]+", claim_lower)
        for window in range(min(len(words), 5), 2, -1):
            for i in range(len(words) - window + 1):
                phrase = " ".join(words[i : i + window])
                if any(phrase in et for et in evidence_texts):
                    return True

        # Pass 2: Content word overlap -- at least 40% of content words
        # must appear in at least one evidence text
        content_words = [w for w in words if len(w) >= 4]
        if not content_words:
            return False

        for et in evidence_texts:
            matches = sum(1 for w in content_words if w in et)
            if matches / len(content_words) >= 0.4:
                return True

        return False
