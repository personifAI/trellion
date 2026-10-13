# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""Standardized evidence model for the agent team spec.

Every RAG output carries Evidence objects with source_type, span, and
confidence so downstream consumers (verifiers, LLM callers, agents)
always get a uniform representation.
"""

from __future__ import annotations

from trellion_protocols.types import Evidence

__all__ = ["Evidence", "evidence_from_chunk"]


def evidence_from_chunk(
    chunk: dict,
    page: dict,
    *,
    confidence: float | None = None,
) -> Evidence:
    """Build an Evidence from a retrieved chunk and its parent page.

    Args:
        chunk: Dict with keys ``text``, ``page_id``, ``score``, and
               optionally ``start_char``/``end_char``.
        page: Dict with keys ``doc_id``, ``page_id``, ``page_no``, ``text``.
        confidence: Override confidence; defaults to chunk["score"].

    Returns:
        A fully-populated Evidence instance.
    """
    chunk_text = chunk.get("text", "")
    page_text = page.get("text", "")

    # Determine span: prefer explicit offsets, else search page_text
    start = chunk.get("start_char")
    end = chunk.get("end_char")
    if start is None or end is None:
        idx = page_text.find(chunk_text)
        if idx >= 0:
            start = idx
            end = idx + len(chunk_text)
        else:
            start = 0
            end = len(chunk_text)

    return Evidence(
        source_type="chunk",
        doc_id=page.get("doc_id", ""),
        page_no=page.get("page_no", 0),
        page_id=page.get("page_id", chunk.get("page_id", "")),
        snippet=chunk_text,
        span=(start, end),
        confidence=confidence if confidence is not None else chunk.get("score", 0.0),
    )
