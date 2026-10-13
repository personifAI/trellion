# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""Classical NLP pass: noun-phrase candidates + NER, pre-LLM.

The goal isn't to replace the LLM extractor — it's to give the extractor a
strong prior for what entities exist in a chunk, and to give canonicalization
a set of anchors that don't depend on LLM consistency.

Lazy spaCy load so the package imports fine even if spaCy models aren't
installed (useful in CI / lightweight environments).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ChunkNLP:
    chunk_id: str
    noun_phrases: list[str] = field(default_factory=list)
    named_entities: list[tuple[str, str]] = field(default_factory=list)  # (text, label)


@lru_cache(maxsize=1)
def _load_spacy(model: str = "en_core_web_sm") -> Any:
    import spacy

    try:
        return spacy.load(model)
    except OSError:
        logger.warning(
            "spaCy model %s not installed. Run `python -m spacy download %s`.", model, model
        )
        return spacy.blank("en")


def analyze(chunk_id: str, text: str, *, model: str = "en_core_web_sm") -> ChunkNLP:
    nlp = _load_spacy(model)
    doc = nlp(text)
    nps = list({nc.text.strip() for nc in getattr(doc, "noun_chunks", [])})
    ents = [(ent.text.strip(), ent.label_) for ent in getattr(doc, "ents", [])]
    return ChunkNLP(chunk_id=chunk_id, noun_phrases=nps, named_entities=ents)
