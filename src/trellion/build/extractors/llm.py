# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""LLM-driven entity + typed-relation extraction (original behavior).

Extraction prompt is parameterized by the run's entity-type and relation-type
vocabularies. Unlike GraphRAG's free-text relations, we force the model to
pick from a closed relation vocabulary so downstream DAG projection has
something to latch onto.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from trellion.build.classical_nlp import ChunkNLP
from trellion.common.config import ExtractionConfig
from trellion.common.llm_client import LLMClient
from trellion.common.schema import Chunk, ExtractedEntity, ExtractedRelation
from trellion.live.entity_quality import DropTally, check_entity

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = (
    "You are an information-extraction system. You read short text passages "
    "and emit structured JSON describing entities and their typed relationships. "
    "You only use the entity types and relation types listed in the task. "
    "You never invent entities that aren't supported by the passage."
)


USER_TEMPLATE = """\
Extract entities and typed relationships from the passage.

Allowed entity types: {entity_types}
Allowed relation types: {relation_types}

Hints from a classical NLP pass (may be incomplete or noisy, use as guidance):
- Noun phrases: {noun_phrases}
- Named entities: {named_entities}

Output JSON with this exact shape:
{{
  "entities": [
    {{"name": str, "entity_type": str, "description": str}}
  ],
  "relationships": [
    {{"source": str, "target": str, "relation_type": str,
      "description": str, "strength": float}}
  ]
}}

`source` and `target` must match an entity `name` you listed. `strength` is
1..10. If you cannot confidently determine a directional relation, omit it.
Do not extract mathematical variables or symbols (a, x, theta, x_t, f(x)),
single letters, bare numbers or amounts, or pronouns as entities.

Passage:
\"\"\"{text}\"\"\"

Return only the JSON object, no prose.
"""


class LLMExtractor:
    """Strategy: delegate extraction to the LLM with JSON-mode output."""

    def __init__(self, cfg: ExtractionConfig, llm: LLMClient):
        self.cfg = cfg
        self.llm = llm

    async def extract(
        self,
        chunk: Chunk,
        nlp: ChunkNLP | None,
    ) -> tuple[list[ExtractedEntity], list[ExtractedRelation]]:
        prompt = USER_TEMPLATE.format(
            entity_types=", ".join(self.cfg.entity_types),
            relation_types=", ".join(self.cfg.relation_types),
            noun_phrases=(nlp.noun_phrases[:20] if nlp else []),
            named_entities=(nlp.named_entities[:20] if nlp else []),
            text=chunk.text,
        )

        response = await self.llm.complete(
            prompt,
            system=SYSTEM_PROMPT,
            response_format={"type": "json_object"},
        )

        parsed = _safe_json(response)
        if parsed is None:
            logger.warning("llm extract: failed to parse chunk %s", chunk.chunk_id)
            return [], []

        entities = [
            ExtractedEntity(
                name=_norm(e.get("name", "")),
                entity_type=_norm(e.get("entity_type", "")),
                description=str(e.get("description", "")).strip(),
                source_chunk_id=chunk.chunk_id,
            )
            for e in parsed.get("entities", [])
            if e.get("name")
        ]
        junk = DropTally()
        kept = []
        for ent in entities:
            reason = check_entity(ent.name, ent.entity_type)
            if reason:
                junk.add(ent.name, reason)
            else:
                kept.append(ent)
        entities = kept
        relations = [
            ExtractedRelation(
                source=_norm(r.get("source", "")),
                target=_norm(r.get("target", "")),
                relation_type=_norm(r.get("relation_type", "")),
                description=str(r.get("description", "")).strip(),
                strength=float(r.get("strength", 1.0)),
                source_chunk_id=chunk.chunk_id,
            )
            for r in parsed.get("relationships", [])
            if r.get("source") and r.get("target") and r.get("relation_type")
        ]

        # Filter out relations whose endpoints the model hallucinated.
        # A dropped junk entity drops its relations here too: no dangling edge.
        entity_names = {e.name for e in entities}
        relations = [r for r in relations if r.source in entity_names and r.target in entity_names]
        junk.log("llm_extract.junk_dropped", chunk_id=chunk.chunk_id, relations_kept=len(relations))

        # TODO: implement gleanings loop (cfg.max_gleanings > 0).
        return entities, relations


def _safe_json(raw: str) -> dict[str, Any] | None:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None


def _norm(s: str) -> str:
    return " ".join(str(s).strip().upper().split())
