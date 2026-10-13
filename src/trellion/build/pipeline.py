# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""End-to-end build pipeline.

Stages (each stage writes a JSONL checkpoint so you can resume):

  1. ingest:       docs/ -> chunks.jsonl
  2. classical:    chunks.jsonl -> nlp.jsonl
  3. extract:      chunks + nlp -> entities.jsonl, relations.jsonl
  4. canonicalize: entities -> canonical_entities.jsonl, mention_map.json
  5. edges:        relations + mention_map -> canonical_edges.jsonl
  6. cluster:      canonical_edges -> communities.jsonl
  7. project:      canonical_edges -> dag_edges.jsonl
  8. emit:         canonical_entities + dag_edges + communities -> learning-graph.csv

Every stage appends a StageTrace to `out/trace.jsonl` so runs are auditable
and an RDS-style tokens-per-quality ratio can be computed offline.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from trellion.build import canonicalize as canon_mod
from trellion.build import classical_nlp, cluster, dag_project, emit_csv, extract, ingest
from trellion.common.config import BuildConfig
from trellion.common.embed_client import EmbedClient
from trellion.common.llm_client import LLMClient
from trellion.common.schema import (
    CanonicalEdge,
    Chunk,
    ExtractedEntity,
    ExtractedRelation,
)
from trellion.common.trace import TraceCollector, stage_timer

logger = logging.getLogger(__name__)


@dataclass
class BuildResult:
    """Everything a caller needs after a batch build.

    The in-memory stage-4-7 artifacts ride along so the CLI can persist
    the graph as a snapshot (``trellion.snapshots.batch_emit``) without
    re-reading checkpoints. ``str()`` remains the CSV path for
    backward-compatible logging.
    """

    csv_path: Path
    concepts: list = field(default_factory=list)
    dag_edges: list = field(default_factory=list)
    communities: list = field(default_factory=list)

    def __str__(self) -> str:  # keeps `f"-> {result}"` printing the path
        return str(self.csv_path)

    def __fspath__(self) -> str:
        return str(self.csv_path)


async def run(cfg: BuildConfig) -> BuildResult:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    traces = TraceCollector()

    # 1: ingest
    logger.info("pipeline: stage 1 ingest")
    chunks: list[Chunk] = []
    with stage_timer(traces, "ingest") as st:
        for doc_id, text in ingest.load_documents(cfg.input_dir):
            chunks.extend(ingest.chunk_document(doc_id, text, cfg.chunking))
        st.records_out = len(chunks)
    _dump_jsonl(cfg.output_dir / "chunks.jsonl", (c.model_dump() for c in chunks))

    # 2: classical NLP
    logger.info("pipeline: stage 2 classical")
    with stage_timer(traces, "classical", records_in=len(chunks)) as st:
        nlp_by_chunk = {c.chunk_id: classical_nlp.analyze(c.chunk_id, c.text) for c in chunks}
        st.records_out = len(nlp_by_chunk)

    # 3: extract
    logger.info("pipeline: stage 3 extract (%d chunks)", len(chunks))
    llm = LLMClient(cfg.llm)
    with stage_timer(traces, "extract", records_in=len(chunks)) as st:
        pre = llm.snapshot()
        tasks = [
            extract.extract_from_chunk(c, nlp_by_chunk.get(c.chunk_id), llm, cfg.extraction)
            for c in chunks
        ]
        results = await asyncio.gather(*tasks)

        entities: list[ExtractedEntity] = []
        relations: list[ExtractedRelation] = []
        for ents, rels in results:
            entities.extend(ents)
            relations.extend(rels)
        post = llm.snapshot()
        st.records_out = len(entities) + len(relations)
        st.llm_calls = post["calls"] - pre["calls"]
        st.tokens_in = post["tokens_in"] - pre["tokens_in"]
        st.tokens_out = post["tokens_out"] - pre["tokens_out"]
        st.notes = {"entities": len(entities), "relations": len(relations)}
    _dump_jsonl(cfg.output_dir / "entities.jsonl", (e.model_dump() for e in entities))
    _dump_jsonl(cfg.output_dir / "relations.jsonl", (r.model_dump() for r in relations))

    # 4: canonicalize
    logger.info("pipeline: stage 4 canonicalize")
    embed = EmbedClient(cfg.embed)
    with stage_timer(traces, "canonicalize", records_in=len(entities)) as st:
        canon, mention_map = await canon_mod.canonicalize(entities, embed, cfg.canonicalization)
        st.records_out = len(canon)
    _dump_jsonl(cfg.output_dir / "canonical_entities.jsonl", (c.model_dump() for c in canon))
    (cfg.output_dir / "mention_map.json").write_text(
        json.dumps({f"{k[0]}|{k[1]}": v for k, v in mention_map.items()}, indent=2)
    )

    # 5: project raw relations onto canonical entity ids
    with stage_timer(traces, "canonical_edges", records_in=len(relations)) as st:
        canonical_edges = _relations_to_canonical_edges(relations, mention_map)
        st.records_out = len(canonical_edges)
        st.notes = {"dropped": len(relations) - len(canonical_edges)}
    _dump_jsonl(
        cfg.output_dir / "canonical_edges.jsonl",
        (e.model_dump() for e in canonical_edges),
    )

    # 6: cluster
    logger.info("pipeline: stage 6 cluster")
    with stage_timer(traces, "cluster", records_in=len(canonical_edges)) as st:
        communities = cluster.cluster(canonical_edges, cfg.cluster)
        st.records_out = len(communities)
    _dump_jsonl(cfg.output_dir / "communities.jsonl", (a.model_dump() for a in communities))

    # 7: DAG projection
    logger.info("pipeline: stage 7 dag projection")
    with stage_timer(traces, "dag_project", records_in=len(canonical_edges)) as st:
        dag_edges = dag_project.project_to_dag(canonical_edges, cfg.dag)
        st.records_out = len(dag_edges)
        st.notes = {"edges_removed": len(canonical_edges) - len(dag_edges)}
    _dump_jsonl(cfg.output_dir / "dag_edges.jsonl", (e.model_dump() for e in dag_edges))

    # 8: emit csv — demoted to a debug/fixture artifact (gateway proposal 27
    # consolidation): the canonical destination is the snapshot store, which
    # the CLI writes from this result via trellion.snapshots.batch_emit.
    out_csv = cfg.output_dir / "learning-graph.csv"
    with stage_timer(traces, "emit", records_in=len(canon)) as st:
        emit_csv.emit_csv(canon, dag_edges, communities, out_csv)
        st.records_out = len(canon)

    traces.dump(cfg.output_dir / "trace.jsonl")
    logger.info("pipeline: done -> %s (trace=%s)", out_csv, traces.summary())
    return BuildResult(
        csv_path=out_csv,
        concepts=canon,
        dag_edges=dag_edges,
        communities=communities,
    )


def _relations_to_canonical_edges(
    relations: list[ExtractedRelation],
    mention_map: dict[tuple[str, str], str],
) -> list[CanonicalEdge]:
    """Map each raw relation's src/tgt mentions to canonical IDs.

    Drops relations whose endpoints didn't land in any canonical group (e.g.
    because the entity_type didn't match what the extractor emitted on the
    relation side). In a polished run this should be rare.
    """
    # Build a name-only lookup as a fallback when type agreement fails.
    name_to_canon: dict[str, str] = {}
    for (name, _t), cid in mention_map.items():
        name_to_canon.setdefault(name, cid)

    out: list[CanonicalEdge] = []
    for r in relations:
        src = name_to_canon.get(r.source)
        tgt = name_to_canon.get(r.target)
        if not src or not tgt or src == tgt:
            continue
        out.append(
            CanonicalEdge(
                source_id=src,
                target_id=tgt,
                relation_type=r.relation_type,
                weight=max(1.0, float(r.strength)),
                evidence_chunks=[r.source_chunk_id],
            )
        )
    return out


def _dump_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False))
            f.write("\n")
