# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""Eval harness: run a golden query set against a CKG and produce a scorecard.

Golden file format (JSONL):

    {"id": "q1", "type": "T2", "query": "prereqs for X",
     "expected_concept_ids": [3, 5], "expected_path": null}
    {"id": "q2", "type": "T3", "query": "from A to B",
     "expected_concept_ids": [1, 3, 4, 7], "expected_path": [1, 3, 4, 7]}

Fields:
  - type: one of T1..T5 — checked against the router's classification.
  - expected_concept_ids: the set of concept ids that should show up in the
    answer (used for F1-on-sets scoring).
  - expected_path: only for T3 / T5 path queries; ordered cid sequence.

Per-row verdict includes set-F1, path scores (if applicable), tokens, and
RDS. The run-level summary averages each metric and reports per-type
breakdowns so regressions localize.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from trellion.eval.metrics import PathScore, SetScore, f1_set, path_match, rds
from trellion.query.answer import Answer, answer_query
from trellion.query.graph import CKG


@dataclass
class GoldenQuery:
    id: str
    type: str
    query: str
    expected_concept_ids: list = field(default_factory=list)
    expected_path: list | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "GoldenQuery":
        return cls(
            id=str(d["id"]),
            type=str(d["type"]),
            query=str(d["query"]),
            expected_concept_ids=list(d.get("expected_concept_ids", []) or []),
            expected_path=list(d["expected_path"]) if d.get("expected_path") else None,
        )


@dataclass
class QueryResult:
    id: str
    expected_type: str
    got_type: str
    type_ok: bool
    set_score: SetScore
    path_score: PathScore | None
    tokens: int
    rds: float
    answer_content: str


def load_golden(path: Path) -> list[GoldenQuery]:
    path = Path(path)
    out: list[GoldenQuery] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            out.append(GoldenQuery.from_dict(json.loads(line)))
    return out


def score_one(ckg: CKG, g: GoldenQuery, *, max_hops: int = 5) -> QueryResult:
    ans: Answer = answer_query(ckg, g.query, max_hops=max_hops)
    set_score = f1_set(ans.concept_ids, g.expected_concept_ids)
    ps = path_match(ans.concept_ids, g.expected_path) if g.expected_path else None
    return QueryResult(
        id=g.id,
        expected_type=g.type,
        got_type=ans.query_type,
        type_ok=(ans.query_type == g.type),
        set_score=set_score,
        path_score=ps,
        tokens=ans.retrieved_tokens,
        rds=rds(set_score.f1, ans.retrieved_tokens),
        answer_content=ans.content,
    )


@dataclass
class RunReport:
    rows: list = field(default_factory=list)
    summary: dict = field(default_factory=dict)

    def to_jsonable(self) -> dict:
        def _row(r: QueryResult) -> dict:
            d = asdict(r)
            # dataclasses nested in dataclasses come through asdict cleanly,
            # but PathScore=None needs to stay as None not {}.
            return d

        return {"rows": [_row(r) for r in self.rows], "summary": self.summary}


def run(
    ckg: CKG,
    golden: Iterable[GoldenQuery],
    *,
    max_hops: int = 5,
) -> RunReport:
    rows: list[QueryResult] = [score_one(ckg, g, max_hops=max_hops) for g in golden]
    if not rows:
        return RunReport(rows=[], summary={"count": 0})

    total = len(rows)
    type_correct = sum(1 for r in rows if r.type_ok)
    avg_f1 = sum(r.set_score.f1 for r in rows) / total
    avg_precision = sum(r.set_score.precision for r in rows) / total
    avg_recall = sum(r.set_score.recall for r in rows) / total
    avg_tokens = sum(r.tokens for r in rows) / total
    avg_rds = sum(r.rds for r in rows) / total

    by_type: dict[str, dict[str, Any]] = {}
    for r in rows:
        bucket = by_type.setdefault(r.expected_type, {"count": 0, "f1_sum": 0.0, "tokens_sum": 0})
        bucket["count"] += 1
        bucket["f1_sum"] += r.set_score.f1
        bucket["tokens_sum"] += r.tokens
    for t, b in by_type.items():
        b["avg_f1"] = b["f1_sum"] / b["count"]
        b["avg_tokens"] = b["tokens_sum"] / b["count"]

    path_rows = [r for r in rows if r.path_score is not None]
    path_exact = (
        sum(1 for r in path_rows if r.path_score and r.path_score.exact) / len(path_rows)
        if path_rows
        else None
    )

    summary = {
        "count": total,
        "type_accuracy": type_correct / total,
        "avg_f1": avg_f1,
        "avg_precision": avg_precision,
        "avg_recall": avg_recall,
        "avg_tokens": avg_tokens,
        "avg_rds": avg_rds,
        "by_type": by_type,
        "path_exact_accuracy": path_exact,
    }
    return RunReport(rows=rows, summary=summary)
