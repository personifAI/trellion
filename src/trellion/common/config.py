# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""Pydantic config models. One source of truth for every tunable."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """Points at any OpenAI-compatible endpoint (vLLM, OpenAI, Anthropic proxy, etc.)."""

    base_url: str = "http://localhost:8000/v1"
    api_key: str = "not-needed-for-local"
    model: str = "local-llm"
    temperature: float = 0.0
    max_tokens: int = 2048
    timeout_s: float = 120.0
    max_concurrent: int = 32


class EmbeddingConfig(BaseModel):
    """TEI exposes an OpenAI-compatible `/embeddings` endpoint."""

    base_url: str = "http://localhost:8080/v1"
    api_key: str = "not-needed-for-local"
    model: str = "bge-large-en-v1.5"
    batch_size: int = 64
    max_concurrent: int = 16
    dimension: int | None = None  # auto-detected on first call


class ChunkingConfig(BaseModel):
    target_tokens: int = 512
    overlap_tokens: int = 64
    tokenizer: str = "cl100k_base"  # fallback counter; TEI/vLLM do their own


class ExtractionConfig(BaseModel):
    """Controls how the extractor proposes entities and typed relations.

    The `strategy` field selects the implementation (see
    `trellion.build.extractors.build_extractor`). Adding a new strategy is
    the intended way to ablate or experiment with different extraction
    pipelines.
    """

    strategy: Literal["llm", "classical"] = "llm"
    entity_types: list[str] = Field(
        default_factory=lambda: ["CONCEPT", "ENTITY", "EVENT", "PROCESS"]
    )
    relation_types: list[str] = Field(
        default_factory=lambda: ["PREREQ", "CONTAINS", "CAUSES", "RELATED_TO"]
    )
    max_gleanings: int = 1
    prompt_version: str = "v1"


class CanonicalizationConfig(BaseModel):
    """Embedding-based entity resolution with perturbation consensus."""

    similarity_threshold: float = 0.88
    perturbation_sigma: float = 0.02
    perturbation_trials: int = 8
    consensus_fraction: float = 0.625  # merge if >=5/8 trials agree
    knn_k: int = 5


class ClusterConfig(BaseModel):
    algorithm: Literal["leiden"] = "leiden"
    max_cluster_size: int = 20
    seed: int = 42
    use_lcc: bool = True


class DAGProjectionConfig(BaseModel):
    """How to turn an arbitrary typed graph into CKG's DAG CSV."""

    # Relations listed here keep their directionality; others get dropped.
    # Order matters when resolving conflicts — earlier wins.
    priority_relations: list[str] = Field(
        default_factory=lambda: ["PREREQ", "CONTAINS", "CAUSES"]
    )
    cycle_break_strategy: Literal["min_weight", "eades"] = "eades"


class BuildConfig(BaseModel):
    input_dir: Path
    output_dir: Path = Path("out")

    llm: LLMConfig = LLMConfig()
    embed: EmbeddingConfig = EmbeddingConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    extraction: ExtractionConfig = ExtractionConfig()
    canonicalization: CanonicalizationConfig = CanonicalizationConfig()
    cluster: ClusterConfig = ClusterConfig()
    dag: DAGProjectionConfig = DAGProjectionConfig()


class QueryConfig(BaseModel):
    graph_path: Path
    llm: LLMConfig = LLMConfig()
    # Which relation types BFS/DFS traverses by default.
    traverse_relations: list[str] = Field(
        default_factory=lambda: ["PREREQ", "CONTAINS", "CAUSES"]
    )
    max_hops: int = 5
