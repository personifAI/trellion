# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""Async embedding wrapper. Works against TEI (local) or any OpenAI-compatible
embedding endpoint.

Returns numpy arrays (L2-normalized so cosine == dot product).
"""

from __future__ import annotations

import asyncio
import logging

import numpy as np
from openai import AsyncOpenAI

from trellion.common.config import EmbeddingConfig

logger = logging.getLogger(__name__)


class EmbedClient:
    def __init__(self, cfg: EmbeddingConfig):
        self.cfg = cfg
        self._client = AsyncOpenAI(base_url=cfg.base_url, api_key=cfg.api_key)
        self._sem = asyncio.Semaphore(cfg.max_concurrent)

    async def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of strings. Returns shape (len(texts), dim), L2-normalized."""
        if not texts:
            return np.zeros((0, self.cfg.dimension or 0), dtype=np.float32)

        batches = [
            texts[i : i + self.cfg.batch_size]
            for i in range(0, len(texts), self.cfg.batch_size)
        ]
        results = await asyncio.gather(*(self._embed_batch(b) for b in batches))
        stacked = np.vstack(results)
        return _l2_normalize(stacked)

    async def _embed_batch(self, batch: list[str]) -> np.ndarray:
        async with self._sem:
            resp = await self._client.embeddings.create(model=self.cfg.model, input=batch)
        vectors = np.asarray(
            [d.embedding for d in resp.data], dtype=np.float32
        )
        if self.cfg.dimension is None:
            # cache dimension for downstream callers
            self.cfg.dimension = int(vectors.shape[1])
        return vectors


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, eps, None)
