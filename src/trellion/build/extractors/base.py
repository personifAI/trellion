# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""Extractor protocol — the minimal surface every strategy must implement."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from trellion.build.classical_nlp import ChunkNLP
from trellion.common.schema import Chunk, ExtractedEntity, ExtractedRelation


@runtime_checkable
class Extractor(Protocol):
    """Turn one Chunk (+ optional classical NLP hints) into entity/relation lists."""

    async def extract(
        self,
        chunk: Chunk,
        nlp: ChunkNLP | None,
    ) -> tuple[list[ExtractedEntity], list[ExtractedRelation]]:
        ...
