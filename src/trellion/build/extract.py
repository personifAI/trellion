# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""Facade over the pluggable extractor strategies.

The pipeline calls `extract_from_chunk(chunk, nlp, llm, cfg)` once per chunk;
we dispatch to the configured strategy. This keeps `pipeline.py` stable while
the extractor implementations live under `trellion.build.extractors`.

Adding a new strategy:
  1. Implement the Extractor protocol in `extractors/<name>.py`.
  2. Register it in `extractors/__init__.py::build_extractor`.
  3. Add the literal to `ExtractionConfig.strategy`.
"""

from __future__ import annotations

from trellion.build.classical_nlp import ChunkNLP
from trellion.build.extractors import build_extractor
from trellion.common.config import ExtractionConfig
from trellion.common.llm_client import LLMClient
from trellion.common.schema import Chunk, ExtractedEntity, ExtractedRelation


async def extract_from_chunk(
    chunk: Chunk,
    nlp: ChunkNLP | None,
    llm: LLMClient,
    cfg: ExtractionConfig,
) -> tuple[list[ExtractedEntity], list[ExtractedRelation]]:
    extractor = build_extractor(cfg, llm=llm)
    return await extractor.extract(chunk, nlp)
