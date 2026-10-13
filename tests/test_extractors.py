# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""Unit tests for the pluggable extractor strategies.

The LLM extractor test uses a fake LLMClient so there's no network hit.
The classical extractor test only requires spaCy's blank English pipeline.
"""

from __future__ import annotations

import asyncio

import pytest

from trellion.build.classical_nlp import ChunkNLP
from trellion.build.extractors import (
    ClassicalExtractor,
    Extractor,
    LLMExtractor,
    build_extractor,
)
from trellion.common.config import ExtractionConfig
from trellion.common.schema import Chunk


class FakeLLM:
    """Mimics just enough of LLMClient for the LLMExtractor."""

    def __init__(self, response: str) -> None:
        self._response = response
        self.calls: list[str] = []

    async def complete(self, prompt, *, system=None, response_format=None, temperature=None):
        self.calls.append(prompt)
        return self._response


def _run(coro):
    return asyncio.run(coro)


def test_build_extractor_dispatches_by_strategy():
    cfg_llm = ExtractionConfig(strategy="llm")
    cfg_classical = ExtractionConfig(strategy="classical")

    llm = FakeLLM("{}")
    assert isinstance(build_extractor(cfg_llm, llm=llm), LLMExtractor)
    assert isinstance(build_extractor(cfg_classical), ClassicalExtractor)

    with pytest.raises(ValueError):
        build_extractor(cfg_llm)  # llm strategy without an LLMClient


def test_classical_satisfies_protocol():
    cfg = ExtractionConfig(strategy="classical")
    extractor = ClassicalExtractor(cfg)
    assert isinstance(extractor, Extractor)


def test_classical_extractor_uses_classical_nlp():
    cfg = ExtractionConfig(strategy="classical", relation_types=["CO_OCCURS", "PREREQ"])
    extractor = ClassicalExtractor(cfg)

    chunk = Chunk(
        chunk_id="c0",
        doc_id="d0",
        text="ignored — the ChunkNLP supplies the mentions",
        token_count=10,
        offset=0,
    )
    nlp = ChunkNLP(
        chunk_id="c0",
        noun_phrases=["Function", "Derivative", "Chain Rule"],
        named_entities=[("Newton", "PERSON")],
    )

    entities, relations = _run(extractor.extract(chunk, nlp))

    names = {e.name for e in entities}
    # Everything got promoted to CONCEPT, normalized uppercase.
    assert names == {"FUNCTION", "DERIVATIVE", "CHAIN RULE", "NEWTON"}
    assert all(e.entity_type == "CONCEPT" for e in entities)
    assert all(e.source_chunk_id == "c0" for e in entities)

    # With 4 mentions we should have C(4,2) = 6 co-occurrence edges.
    assert len(relations) == 6
    assert all(r.relation_type == "CO_OCCURS" for r in relations)
    assert all(r.source_chunk_id == "c0" for r in relations)


def test_llm_extractor_parses_json_response():
    response = (
        '{"entities": ['
        ' {"name": "Function", "entity_type": "CONCEPT", "description": "a mapping"},'
        ' {"name": "Derivative", "entity_type": "CONCEPT", "description": "rate of change"}'
        '], "relationships": ['
        ' {"source": "Function", "target": "Derivative", "relation_type": "PREREQ",'
        '  "description": "function needed before derivative", "strength": 8}'
        ']}'
    )
    fake = FakeLLM(response)
    cfg = ExtractionConfig(strategy="llm")
    extractor = LLMExtractor(cfg, fake)  # type: ignore[arg-type]

    chunk = Chunk(
        chunk_id="c1",
        doc_id="d0",
        text="A function is a rule. Its derivative is a rate.",
        token_count=12,
        offset=0,
    )
    entities, relations = _run(extractor.extract(chunk, None))

    assert {e.name for e in entities} == {"FUNCTION", "DERIVATIVE"}
    assert len(relations) == 1
    r = relations[0]
    assert r.source == "FUNCTION"
    assert r.target == "DERIVATIVE"
    assert r.relation_type == "PREREQ"
    assert r.strength == 8.0
    assert fake.calls, "expected the LLM to be called once"


def test_llm_extractor_filters_hallucinated_edges():
    # "Derivative" is referenced in the relation but not listed as an entity.
    response = (
        '{"entities": ['
        ' {"name": "Function", "entity_type": "CONCEPT", "description": "a mapping"}'
        '], "relationships": ['
        ' {"source": "Function", "target": "Derivative", "relation_type": "PREREQ",'
        '  "description": "bogus", "strength": 1}'
        ']}'
    )
    fake = FakeLLM(response)
    cfg = ExtractionConfig(strategy="llm")
    extractor = LLMExtractor(cfg, fake)  # type: ignore[arg-type]
    chunk = Chunk(chunk_id="c2", doc_id="d0", text="t", token_count=1, offset=0)
    _, relations = _run(extractor.extract(chunk, None))
    assert relations == []
