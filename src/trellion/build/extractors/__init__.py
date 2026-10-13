# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""Pluggable entity/relation extractors.

Pick a strategy via `ExtractionConfig.strategy`:

- "llm"       — ask the LLM to return structured JSON (the default; our
                GraphRAG-flavored path).
- "classical" — spaCy-only noun phrases + NER with simple co-occurrence
                edges. Useful as a no-LLM ablation baseline when you want to
                measure how much the LLM is actually contributing.

All extractors implement the `Extractor` protocol: `async extract(chunk, nlp)
-> (entities, relations)`.
"""

from trellion.build.extractors.base import Extractor
from trellion.build.extractors.classical import ClassicalExtractor
from trellion.build.extractors.llm import LLMExtractor

__all__ = [
    "Extractor",
    "LLMExtractor",
    "ClassicalExtractor",
    "build_extractor",
]


def build_extractor(cfg, llm=None):
    """Factory: return the extractor instance for a given ExtractionConfig.

    `llm` is only required for strategies that need it; the classical
    strategy ignores it.
    """
    strategy = getattr(cfg, "strategy", "llm")
    if strategy == "llm":
        if llm is None:
            raise ValueError("strategy='llm' requires an LLMClient instance")
        return LLMExtractor(cfg, llm)
    if strategy == "classical":
        return ClassicalExtractor(cfg)
    raise ValueError(f"unknown extraction strategy: {strategy!r}")
