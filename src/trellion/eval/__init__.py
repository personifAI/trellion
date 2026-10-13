# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""Offline eval harness for the T1-T5 CKG query router.

Typical use:

    from trellion.eval.benchmark import load_golden, run
    from trellion.query.graph import CKG

    ckg = CKG.load("out/learning-graph.csv")
    golden = load_golden("src/trellion/eval/golden/learning-graph.jsonl")
    report = run(ckg, golden)
    print(report.summary)
"""

from trellion.eval.benchmark import (
    GoldenQuery,
    QueryResult,
    RunReport,
    load_golden,
    run,
    score_one,
)
from trellion.eval.metrics import PathScore, SetScore, f1_set, path_match, rds

__all__ = [
    "GoldenQuery",
    "QueryResult",
    "RunReport",
    "load_golden",
    "run",
    "score_one",
    "PathScore",
    "SetScore",
    "f1_set",
    "path_match",
    "rds",
]
