# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""Live-path retrieval benchmark: intelligent traversal vs vanilla vector.

Graph-guided retrieval is an optional add-on, so this measures whether it
actually beats vector-only on the live pageindex path. The CSV-lane harness (``eval/benchmark.py``) cannot
answer that — it scores the deterministic CKG router, not live retrieval.

Design:

- **Two arms, same store, same corpus, same embedder.** Vanilla =
  ``RAGOrchestrator.execute_lookup`` (pure vector; no navigation, no
  graph boost). Intelligent = ``execute_graph_first`` (navigator seeds →
  multi-hop expansion → candidate pages → vector ranking within them +
  graph boost). These are the exact strategies the production
  ``worker.rag.query`` subject dispatches.
- **The graph is fixture ground truth, not LLM-extracted** — the
  benchmark isolates retrieval *policy* from extraction quality.
- **Relational golden queries are lexically disjoint from their answer
  docs by construction**, so a vector hit on them is luck and a graph
  hit is traversal. Direct queries keep the vanilla arm honest.
- The bundled ``FakeDeterministicEmbedder`` (token-hash bag-of-words)
  makes the harness runnable in CI with no services. For the real
  scorecard, pass a real embedder (``PageIndexEmbedder``) — the numbers
  you quote must come from real embeddings.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import structlog

logger = structlog.get_logger(__name__)

FIXTURE_DIR = Path(__file__).parent / "bench_fixtures"
CORPUS_PATH = FIXTURE_DIR / "live_corpus.json"
GOLDEN_PATH = FIXTURE_DIR / "live_golden.jsonl"


class FakeDeterministicEmbedder:
    """Token-hash bag-of-words embedder — cosine similarity ≈ token overlap.

    Deterministic and service-free, so the harness mechanics run in CI.
    Stop words are stripped — without that, function-word overlap gives
    every doc a similarity floor that drowns the content signal (a real
    embedding model handles this implicitly). A tiny text-hash component
    breaks score ties so result ordering is stable across runs. NOT for
    the real scorecard.
    """

    _STOP = frozenset(
        "a an the is are was were be been do does did what which how why "
        "we you it on in to for with of from by at as and or has have had "
        "this that its our their there here when where who whom".split()
    )

    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim

    async def embed_text(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        tokens = [
            t
            for t in re.findall(r"[a-zA-Z0-9_\-]+", text.lower())
            if t not in self._STOP
        ]
        for tok in tokens:
            # Signed feature hashing: the sign bit makes cross-token
            # collision noise cancel in expectation, so near-zero cosine
            # actually means "no shared content" instead of "shared
            # hash buckets".
            digest = hashlib.sha256(tok.encode()).hexdigest()
            h = int(digest[:8], 16)
            sign = 1.0 if int(digest[8], 16) % 2 == 0 else -1.0
            vec[h % self.dim] += sign
        # Tie-breaker: a minuscule deterministic per-text component.
        th = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
        vec[th % self.dim] += 1e-4
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed_text(t) for t in texts]


def load_fixture(
    corpus_path: Path = CORPUS_PATH, golden_path: Path = GOLDEN_PATH
) -> tuple[dict, list[dict]]:
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    golden = [
        json.loads(line)
        for line in golden_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return corpus, golden


async def seed_corpus(store: Any, embedder: Any, corpus: dict) -> dict[str, str]:
    """Load the fixture corpus into a pageindex store.

    Idempotent-enough for a bench database: docs dedup on content hash,
    pages upsert on (doc, page_no), entities upsert on (name, type),
    edges reinforce. Stale ``bench://`` docs from an EDITED fixture are
    removed first so renames don't leave orphans.

    Returns {source_uri: doc_id}.
    """
    current_uris = {d["source_uri"] for d in corpus["docs"]}
    for row in await store.list_docs(limit=500):
        uri = row.get("source_uri", "")
        if uri.startswith("bench://") and uri not in current_uris:
            await store.delete_doc(row["doc_id"])

    doc_ids: dict[str, str] = {}
    page_ids: dict[tuple[str, int], str] = {}
    for doc in corpus["docs"]:
        content_hash = hashlib.sha256(
            "\n".join(doc["pages"]).encode()
        ).hexdigest()
        doc_id, _is_new = await store.insert_doc(
            source_uri=doc["source_uri"],
            title=doc["title"],
            content_hash=content_hash,
            metadata={"bench": True},
        )
        doc_ids[doc["source_uri"]] = doc_id
        page_embeds: list[tuple[str, list[float]]] = []
        for page_no, text in enumerate(doc["pages"], start=1):
            page_id = await store.insert_page(doc_id, page_no, text)
            page_ids[(doc["source_uri"], page_no)] = page_id
            page_embeds.append((page_id, await embedder.embed_text(text)))
            chunk_ids = await store.insert_chunks(page_id, [text])
            await store.store_chunk_embeddings_batch(
                [(chunk_ids[0], page_embeds[-1][1])]
            )
        await store.store_page_embeddings_batch(page_embeds)

    entity_ids: dict[str, str] = {}
    for ent in corpus["entities"]:
        entity_ids[ent["name"]] = await store.upsert_entity(
            canonical_name=ent["name"],
            type=ent["type"],
            aliases=ent.get("aliases") or None,
        )
    for m in corpus["mentions"]:
        await store.insert_mention(
            entity_id=entity_ids[m["entity"]],
            page_id=page_ids[(m["doc"], m["page_no"])],
        )
    for e in corpus["edges"]:
        await store.insert_edge(
            src_entity_id=entity_ids[e["src"]],
            rel_type=e["rel"],
            dst_entity_id=entity_ids[e["dst"]],
            evidence={"bench": True},
        )
    logger.info(
        "live_bench.seeded",
        docs=len(doc_ids), entities=len(entity_ids),
        edges=len(corpus["edges"]),
    )
    return doc_ids


@dataclass
class ArmScore:
    hits: int = 0
    total: int = 0
    reciprocal_ranks: list[float] = field(default_factory=list)

    @property
    def recall(self) -> float:
        return self.hits / self.total if self.total else 0.0

    @property
    def mrr(self) -> float:
        return (
            sum(self.reciprocal_ranks) / len(self.reciprocal_ranks)
            if self.reciprocal_ranks
            else 0.0
        )


def _score(evidence: list, expected_doc_ids: set[str], top_k: int) -> tuple[int, float]:
    """(expected docs found in top_k, reciprocal rank of the first hit)."""
    seen: list[str] = []
    for pack in evidence:
        if pack.doc_id not in seen:
            seen.append(pack.doc_id)
    top = seen[:top_k]
    found = sum(1 for d in expected_doc_ids if d in top)
    rr = 0.0
    for rank, d in enumerate(top, start=1):
        if d in expected_doc_ids:
            rr = 1.0 / rank
            break
    return found, rr


async def run_live_bench(
    dsn: str,
    embedder: Optional[Any] = None,
    *,
    top_k: int = 3,
    corpus_path: Path = CORPUS_PATH,
    golden_path: Path = GOLDEN_PATH,
) -> dict:
    """Seed the fixture corpus and score both arms over the golden set.

    Returns a report dict with per-arm, per-query-type recall@top_k and
    MRR, per-query detail, and the headline delta.
    """
    from sqlalchemy.ext.asyncio import create_async_engine

    from trellion.db.engine import TRELLION_SCHEMA, _async_dsn
    from trellion.live._intel import GRAPH_NARROWER, addon_provider
    graph_narrower = addon_provider(GRAPH_NARROWER)
    from trellion.live.pageindex_retriever import PageIndexRetriever
    from trellion.live.pageindex_store import PageIndexStore
    from trellion.live.rag_orchestrator import RAGOrchestrator

    embedder = embedder or FakeDeterministicEmbedder()
    corpus, golden = load_fixture(corpus_path, golden_path)

    # Pin search_path the way every real caller does (trellion's tables
    # live in schema `trellion`; a raw engine sees only `public`) — same
    # convention as trellion.db.engine and the temporal-graph test fixture.
    engine = create_async_engine(
        _async_dsn(dsn),
        connect_args={"options": f"-csearch_path={TRELLION_SCHEMA},public"},
    )
    store = PageIndexStore(
        async_engine=engine,
        embedding_dim=getattr(embedder, "dim", None),
    )
    await store.connect()
    try:
        doc_ids = await seed_corpus(store, embedder, corpus)
        retriever = PageIndexRetriever(store=store, embedder=embedder)
        orchestrator = RAGOrchestrator(
            navigator=graph_narrower(store=store),
            retriever=retriever,
            store=store,
        )

        scores = {
            arm: {qt: ArmScore() for qt in ("direct", "relational")}
            for arm in ("vector", "graph")
        }
        detail: list[dict] = []
        for q in golden:
            expected = {doc_ids[uri] for uri in q["expected_docs"]}
            row: dict = {"id": q["id"], "type": q["type"], "query": q["query"]}
            for arm, run in (
                ("vector", orchestrator.execute_lookup),
                ("graph", orchestrator.execute_graph_first),
            ):
                resp = await run(q["query"])
                found, rr = _score(resp.evidence, expected, top_k)
                s = scores[arm][q["type"]]
                s.hits += found
                s.total += len(expected)
                s.reciprocal_ranks.append(rr)
                row[arm] = {"found": found, "of": len(expected), "rr": round(rr, 3)}
            detail.append(row)

        def _flat(arm: str) -> dict:
            per_type = {
                qt: {"recall": round(s.recall, 3), "mrr": round(s.mrr, 3)}
                for qt, s in scores[arm].items()
            }
            hits = sum(s.hits for s in scores[arm].values())
            total = sum(s.total for s in scores[arm].values())
            return {
                "per_type": per_type,
                "overall_recall": round(hits / total, 3) if total else 0.0,
            }

        report = {
            "top_k": top_k,
            "embedder": type(embedder).__name__,
            "queries": len(golden),
            "vector": _flat("vector"),
            "graph": _flat("graph"),
            "detail": detail,
        }
        report["relational_delta"] = round(
            scores["graph"]["relational"].recall
            - scores["vector"]["relational"].recall,
            3,
        )
        return report
    finally:
        await store.close()


def format_report(report: dict) -> str:
    lines = [
        f"live retrieval bench — {report['queries']} queries, "
        f"recall@{report['top_k']}, embedder={report['embedder']}",
    ]
    for arm in ("vector", "graph"):
        a = report[arm]
        per = ", ".join(
            f"{qt} recall={m['recall']} mrr={m['mrr']}"
            for qt, m in a["per_type"].items()
        )
        lines.append(f"  {arm:<6} overall={a['overall_recall']}  ({per})")
    lines.append(f"  relational delta (graph - vector): {report['relational_delta']}")
    for row in report["detail"]:
        lines.append(
            f"    {row['id']:>3} [{row['type']:<10}] "
            f"vector {row['vector']['found']}/{row['vector']['of']} "
            f"graph {row['graph']['found']}/{row['graph']['of']}  {row['query']}"
        )
    return "\n".join(lines)
