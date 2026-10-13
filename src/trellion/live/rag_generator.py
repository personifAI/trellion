# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""RAG answer generator -- synthesizes grounded answers from evidence.

Takes retrieved evidence packs, verification results, and optional
graph hypotheses, then calls an LLM to produce an answer that is
strictly grounded in the provided evidence.

Uses the same OpenAI-compatible chat completions API pattern as
graph_extractor.py (works with Ollama, vLLM, etc.).
"""

from __future__ import annotations

from typing import Optional

import httpx
import structlog

from trellion.live._tls import ssl_verify, client_cert
from trellion.live.gateway_auth import GatewayHmacAuth
from trellion.live.llm_throttle import LLMUnavailable, get_llm_throttle

from trellion.live.graph_types import Hypothesis
from trellion.live.pageindex_retriever import EvidencePack
from trellion.live.rag_verifier import VerificationResult

log = structlog.get_logger(__name__)

_SYSTEM_PROMPT = (
    "You are a precise research assistant. Answer the user's question using ONLY "
    "the numbered evidence excerpts provided below. Follow these rules strictly:\n\n"
    "1. Base every claim on one or more evidence excerpts. Cite sources inline "
    "using [Source N] notation (e.g. [Source 1], [Source 2]).\n"
    "2. If the evidence does not contain enough information to fully answer the "
    "question, say so explicitly. Do NOT guess or add information beyond what is "
    "provided.\n"
    "3. If evidence is contradictory, present both sides and note the conflict.\n"
    "4. Keep the answer concise and well-structured.\n"
    "5. Do not repeat the question or the evidence verbatim; synthesize."
)

_LOW_CONFIDENCE_NOTE = (
    "\nNOTE: The retrieved evidence has LOW confidence. Be especially careful to "
    "acknowledge uncertainty and clearly state which parts of your answer are "
    "well-supported vs. weakly supported."
)

_HYPOTHESES_HEADER = (
    "\nGraph-derived reasoning context (use as interpretive guidance, "
    "not as evidence):\n"
)


def _moderate(text: str) -> str:
    """Run text through the centralised content moderation."""
    try:
        from noomesh.gateway.security import moderate_external_content
        return moderate_external_content(text, source="rag_evidence")
    except ImportError:
        return text


def _format_evidence(evidence_packs: list[EvidencePack]) -> str:
    """Format evidence packs as numbered excerpts for the LLM context."""
    sections: list[str] = []
    for idx, pack in enumerate(evidence_packs, 1):
        if pack.evidence_items:
            snippet = _moderate(pack.evidence_items[0].snippet)
        else:
            snippet = _moderate(pack.page_text[:500])

        header = f"[Source {idx}] {pack.title}"
        if pack.page_no:
            header += f" (page {pack.page_no})"

        sections.append(f"{header}\n{snippet}")

    return "\n\n".join(sections)


def _format_hypotheses(hypotheses: list[Hypothesis]) -> str:
    """Format graph hypotheses as reasoning hints."""
    lines: list[str] = []
    for hyp in hypotheses[:5]:
        path_str = " -> ".join(hyp.path)
        lines.append(
            f"- {path_str}: {hyp.description} (confidence: {hyp.confidence:.2f})"
        )
    return "\n".join(lines)


def _build_messages(
    query: str,
    evidence_packs: list[EvidencePack],
    verification: Optional[VerificationResult],
    hypotheses: Optional[list[Hypothesis]],
) -> list[dict[str, str]]:
    """Assemble the chat messages for the LLM call."""
    system = _SYSTEM_PROMPT

    if verification and verification.status in ("rejected", "partial"):
        system += _LOW_CONFIDENCE_NOTE

    evidence_text = _format_evidence(evidence_packs)
    user_parts = [f"Evidence:\n{evidence_text}"]

    if hypotheses:
        hyp_text = _format_hypotheses(hypotheses)
        user_parts.append(f"{_HYPOTHESES_HEADER}{hyp_text}")

    user_parts.append(f"Question: {query}")

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n\n".join(user_parts)},
    ]


class RAGAnswerGenerator:
    """Generates grounded answers from evidence using an LLM endpoint."""

    def __init__(
        self,
        llm_url: str = "https://gateway:8080",
        model: str = "llama3",
    ) -> None:
        self.llm_url = llm_url.rstrip("/")
        self.model = model
        # Signed per request — see GatewayHmacAuth; the answer is streamed
        # through the gateway's mesh-routed /v1/chat/completions.
        self._http = httpx.AsyncClient(
            timeout=300.0, verify=ssl_verify(), cert=client_cert(), auth=GatewayHmacAuth(),
        )

    async def generate(
        self,
        query: str,
        evidence_packs: list[EvidencePack],
        verification: Optional[VerificationResult] = None,
        hypotheses: Optional[list[Hypothesis]] = None,
    ) -> str:
        """Generate an answer grounded in the provided evidence.

        Uses streaming from vLLM/Ollama to avoid HTTP timeouts on long
        generations. Tokens are accumulated server-side and returned as
        a complete string.
        """
        if not evidence_packs:
            return "No evidence was found to answer this question."

        messages = _build_messages(query, evidence_packs, verification, hypotheses)

        body = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 2048,
            "stream": True,
            # Disable Qwen3 thinking mode for answer generation
            "chat_template_kwargs": {"enable_thinking": False},
        }

        async def _stream() -> str:
            chunks: list[str] = []
            async with self._http.stream(
                "POST",
                f"{self.llm_url}/v1/chat/completions",
                json=body,
            ) as resp:
                if resp.status_code >= 400:
                    await resp.aread()  # so the throttle can see model_not_found
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        import json as _json
                        chunk = _json.loads(payload)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        token = delta.get("content", "")
                        if token:
                            chunks.append(token)
                    except (ValueError, IndexError, KeyError):
                        continue
            return "".join(chunks)

        try:
            # Interactive single call: skips the in-flight bound and does not
            # retry (the user is waiting), but honours the per-model breaker so
            # an absent model fails fast instead of timing out every question.
            answer = await get_llm_throttle().call(
                self.model, _stream, what="rag_answer", bounded=False, max_attempts=1
            )


            # Strip <think> blocks if the model still produced them
            if answer:
                import re
                answer = re.sub(r"<think>.*?</think>\s*", "", answer, flags=re.DOTALL)

            if answer.strip():
                log.info(
                    "rag_generator.done",
                    query=query[:80],
                    answer_len=len(answer),
                    sources=len(evidence_packs),
                )
                return answer.strip()

            log.warning("rag_generator.empty_response", query=query[:80])
            return "The language model returned an empty response. Please try again."

        except LLMUnavailable as exc:
            # Expected outage: the throttle already logged the breaker once.
            log.warning(
                "rag_generator.llm_unavailable",
                model=exc.model,
                reason=exc.reason,
                query=query[:80],
            )
            return (
                "Could not generate an answer because the language model is "
                "unavailable. The retrieved evidence is still available in "
                "the citations."
            )
        except httpx.HTTPError as exc:
            log.error(
                "rag_generator.llm_error",
                error_type=type(exc).__name__,
                error=repr(exc),
                query=query[:80],
            )
            return (
                "Could not generate an answer due to a language model error. "
                "The retrieved evidence is still available in the citations."
            )
        except Exception as exc:
            log.error(
                "rag_generator.unexpected_error", error=str(exc), query=query[:80]
            )
            return (
                "An unexpected error occurred while generating the answer. "
                "The retrieved evidence is still available in the citations."
            )

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._http.aclose()
