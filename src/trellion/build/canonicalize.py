# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""Embedding-based entity resolution with perturbation consensus.

This is the module where the "perturb vectors for better labels" idea actually
lives. The perturbation isn't for generating embeddings — it's a stability
probe: we merge two entity mentions only if they remain nearest neighbors
under calibrated noise across a fraction of trials.

Strategy:
  1. Embed each mention's (name + description).
  2. Build a k-NN graph in embedding space.
  3. For T perturbation trials, add N(0, sigma) noise, re-derive k-NN.
  4. Merge pairs whose co-membership fraction >= consensus_fraction.
  5. Assign canonical_id via union-find over the merge pairs.

This is a SKELETON — the union-find + aggregation is real, the embedding
batching is real, but the calibration of sigma (how much noise is "too much
vs. too little") is a tuning TODO.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Iterable

import numpy as np

from trellion.common.config import CanonicalizationConfig
from trellion.common.embed_client import EmbedClient
from trellion.common.schema import CanonicalEntity, ExtractedEntity

logger = logging.getLogger(__name__)


async def canonicalize(
    raw: list[ExtractedEntity],
    embed: EmbedClient,
    cfg: CanonicalizationConfig,
    rng_seed: int = 42,
) -> tuple[list[CanonicalEntity], dict[tuple[str, str], str]]:
    """Return (canonical_entities, mention_key_to_canonical_id)."""
    if not raw:
        return [], {}

    mention_keys = [(e.name, e.entity_type) for e in raw]
    texts = [f"{e.name}. {e.description}" for e in raw]

    logger.info("canonicalize: embedding %d mentions", len(texts))
    vectors = await embed.embed(texts)

    # Group by entity_type — we never merge across types.
    type_to_indices: dict[str, list[int]] = defaultdict(list)
    for i, (_, t) in enumerate(mention_keys):
        type_to_indices[t].append(i)

    uf = _UnionFind(len(raw))
    rng = np.random.default_rng(rng_seed)

    for t, idxs in type_to_indices.items():
        if len(idxs) < 2:
            continue
        sub = vectors[idxs]
        # Trial-0 baseline plus perturbation trials.
        cooc = np.zeros((len(idxs), len(idxs)), dtype=np.int32)
        total = cfg.perturbation_trials + 1
        for t_i in range(total):
            if t_i == 0:
                v = sub
            else:
                noise = rng.normal(0.0, cfg.perturbation_sigma, size=sub.shape)
                v = _l2(sub + noise)
            sims = v @ v.T
            # For each mention, record top-k (excluding self) above threshold.
            for i in range(sims.shape[0]):
                # argpartition for top-k is fine; we'll sort to be deterministic.
                ranked = np.argsort(-sims[i])
                count = 0
                for j in ranked:
                    if j == i:
                        continue
                    if sims[i, j] >= cfg.similarity_threshold and count < cfg.knn_k:
                        cooc[i, j] += 1
                        count += 1
                    if count >= cfg.knn_k:
                        break

        need = int(np.ceil(cfg.consensus_fraction * total))
        # Merge any pair with mutual consensus (i->j AND j->i stable).
        for i in range(cooc.shape[0]):
            for j in range(i + 1, cooc.shape[1]):
                if cooc[i, j] >= need and cooc[j, i] >= need:
                    uf.union(idxs[i], idxs[j])

    # Materialize canonical entities.
    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(len(raw)):
        groups[uf.find(i)].append(i)

    canonical: list[CanonicalEntity] = []
    mention_to_canon: dict[tuple[str, str], str] = {}
    for canon_idx, (root, members) in enumerate(sorted(groups.items())):
        picks = [raw[m] for m in members]
        label = _pick_label([p.name for p in picks])
        entity_type = picks[0].entity_type
        descriptions = [p.description for p in picks if p.description]
        aliases = sorted({p.name for p in picks if p.name != label})
        # Deduplicate while preserving first-seen order — this is the
        # provenance chain: every chunk that contributed a mention.
        seen_chunks: set[str] = set()
        source_chunk_ids: list[str] = []
        for p in picks:
            if p.source_chunk_id and p.source_chunk_id not in seen_chunks:
                seen_chunks.add(p.source_chunk_id)
                source_chunk_ids.append(p.source_chunk_id)
        centroid = vectors[members].mean(axis=0)
        ce = CanonicalEntity(
            canonical_id=f"C{canon_idx:06d}",
            label=label,
            entity_type=entity_type,
            descriptions=descriptions,
            aliases=aliases,
            source_chunk_ids=source_chunk_ids,
            embedding_preview=centroid[:8].tolist(),
        )
        canonical.append(ce)
        for p in picks:
            mention_to_canon[(p.name, p.entity_type)] = ce.canonical_id

    logger.info(
        "canonicalize: %d mentions -> %d canonical entities (%.1f%% reduction)",
        len(raw),
        len(canonical),
        100 * (1 - len(canonical) / max(1, len(raw))),
    )
    return canonical, mention_to_canon


def _pick_label(names: Iterable[str]) -> str:
    """Prefer the longest name as the canonical label (usually the most specific)."""
    return max(names, key=lambda n: (len(n), n))


def _l2(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, 1e-12, None)


class _UnionFind:
    __slots__ = ("parent", "rank")

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
