# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""Classical-NLP-only extractor: spaCy noun phrases + NER, with co-occurrence
edges between mentions in the same chunk.

This is the ablation baseline. It doesn't invent semantics — it just promotes
every noun phrase / named entity to a CONCEPT entity, then emits a CO_OCCURS
edge between every pair of distinct mentions sharing a chunk. You use this
to measure how much of the final graph's signal actually comes from the LLM
vs. from classical NLP you could have run a decade ago.

The extractor ignores `cfg.entity_types` (everything is CONCEPT) and uses
the first entry of `cfg.relation_types` as the co-occurrence label so the
downstream DAG projector can still filter by relation type.
"""

from __future__ import annotations

import logging

from trellion.build.classical_nlp import ChunkNLP, analyze
from trellion.common.config import ExtractionConfig
from trellion.live.entity_quality import check_entity
from trellion.common.schema import Chunk, ExtractedEntity, ExtractedRelation

logger = logging.getLogger(__name__)


class ClassicalExtractor:
    """Strategy: spaCy NPs + NER with in-chunk co-occurrence edges."""

    def __init__(self, cfg: ExtractionConfig):
        self.cfg = cfg
        # Use the first configured relation type as the co-occurrence tag.
        # Falls back to RELATED_TO if none configured.
        self._cooc_type = (cfg.relation_types[0] if cfg.relation_types else "RELATED_TO").upper()

    async def extract(
        self,
        chunk: Chunk,
        nlp: ChunkNLP | None,
    ) -> tuple[list[ExtractedEntity], list[ExtractedRelation]]:
        # Re-run classical NLP if the caller didn't supply one.
        if nlp is None:
            nlp = analyze(chunk.chunk_id, chunk.text)

        # Collect unique mention strings from noun phrases and named entities.
        mentions: list[str] = []
        seen: set[str] = set()
        for np in nlp.noun_phrases:
            key = _norm(np)
            if key and key not in seen and len(key) > 2 and check_entity(key) is None:
                seen.add(key)
                mentions.append(key)
        for ent_text, _label in nlp.named_entities:
            key = _norm(ent_text)
            if key and key not in seen and len(key) > 2 and check_entity(key) is None:
                seen.add(key)
                mentions.append(key)

        entities = [
            ExtractedEntity(
                name=m,
                entity_type="CONCEPT",
                description=f"Mention observed in chunk {chunk.chunk_id}",
                source_chunk_id=chunk.chunk_id,
            )
            for m in mentions
        ]

        # Fully-connected co-occurrence for mentions in this chunk.
        relations: list[ExtractedRelation] = []
        for i in range(len(mentions)):
            for j in range(i + 1, len(mentions)):
                relations.append(
                    ExtractedRelation(
                        source=mentions[i],
                        target=mentions[j],
                        relation_type=self._cooc_type,
                        description=f"co-occurs in chunk {chunk.chunk_id}",
                        strength=1.0,
                        source_chunk_id=chunk.chunk_id,
                    )
                )

        return entities, relations


def _norm(s: str) -> str:
    return " ".join(str(s).strip().upper().split())
