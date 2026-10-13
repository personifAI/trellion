# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""Tests for the eval harness: metrics, benchmark runner, golden set."""

from __future__ import annotations

from pathlib import Path

import pytest

from trellion.eval.benchmark import GoldenQuery, load_golden, run, score_one
from trellion.eval.metrics import f1_set, path_match, rds
from trellion.query.graph import CKG


SAMPLE_CSV = Path(__file__).parents[1] / "data" / "sample" / "learning-graph.csv"
GOLDEN = (
    Path(__file__).parents[1]
    / "src"
    / "trellion"
    / "eval"
    / "golden"
    / "learning-graph.jsonl"
)


# --- metrics -----------------------------------------------------------------


def test_f1_set_perfect_empty():
    s = f1_set([], [])
    assert (s.precision, s.recall, s.f1) == (1.0, 1.0, 1.0)


def test_f1_set_basic():
    s = f1_set([1, 2, 3], [2, 3, 4])
    # TP=2, FP=1 (1), FN=1 (4): P=2/3, R=2/3, F1=2/3
    assert s.precision == pytest.approx(2 / 3)
    assert s.recall == pytest.approx(2 / 3)
    assert s.f1 == pytest.approx(2 / 3)


def test_f1_set_zero_overlap():
    s = f1_set([1, 2], [3, 4])
    assert s.f1 == 0.0


def test_path_match_exact():
    ps = path_match([1, 2, 3], [1, 2, 3])
    assert ps.exact
    assert ps.endpoints
    assert ps.intermediate_jaccard == 1.0


def test_path_match_same_endpoints_diff_middle():
    ps = path_match([1, 5, 3], [1, 2, 3])
    assert not ps.exact
    assert ps.endpoints
    # Middle sets {5} vs {2} have Jaccard 0.
    assert ps.intermediate_jaccard == 0.0


def test_path_match_wrong_endpoints():
    ps = path_match([1, 2, 9], [1, 2, 3])
    assert not ps.exact
    assert not ps.endpoints


def test_rds_zero_tokens():
    assert rds(1.0, 0) == 0.0


def test_rds_basic():
    assert rds(0.5, 10) == 0.05


# --- benchmark runner --------------------------------------------------------


@pytest.fixture
def ckg() -> CKG:
    return CKG.load(SAMPLE_CSV)


def test_load_golden_parses_all_rows():
    golden = load_golden(GOLDEN)
    # We hand-wrote 14 rows; if someone adds more, this should still be >=14.
    assert len(golden) >= 14
    ids = {g.id for g in golden}
    # Spot-check a few critical cases.
    assert "t3_fn_taylor" in ids
    assert "t3_rev_taylor_fn" in ids
    assert "t4_found" in ids


def test_score_one_t2(ckg):
    g = GoldenQuery(
        id="x",
        type="T2",
        query="What are the prerequisites for Composite Function?",
        expected_concept_ids=[1, 3],
    )
    r = score_one(ckg, g)
    assert r.got_type == "T2"
    assert r.type_ok
    assert r.set_score.f1 == 1.0
    assert r.tokens > 0


def test_score_one_t3_exact_path(ckg):
    g = GoldenQuery(
        id="x",
        type="T3",
        query="What is the chain from Function to Taylor Series?",
        expected_concept_ids=[1, 6, 8],
        expected_path=[1, 6, 8],
    )
    r = score_one(ckg, g)
    assert r.type_ok
    assert r.path_score is not None
    assert r.path_score.exact
    assert r.path_score.endpoints


def test_score_one_t3_reverse_direction(ckg):
    g = GoldenQuery(
        id="x",
        type="T3",
        query="What is the chain from Taylor Series to Function?",
        expected_concept_ids=[8, 6, 1],
        expected_path=[8, 6, 1],
    )
    r = score_one(ckg, g)
    assert r.type_ok
    assert r.path_score is not None
    assert r.path_score.exact, f"reverse path failed; got {r.answer_content}"


def test_run_full_golden_on_sample(ckg):
    golden = load_golden(GOLDEN)
    report = run(ckg, golden)
    s = report.summary
    assert s["count"] == len(golden)
    # On the sample graph with the built-in router, we should classify types
    # correctly for the overwhelming majority of queries.
    assert s["type_accuracy"] >= 0.85
    # And average F1 should be comfortably nonzero (router isn't perfect on
    # every T5 neighbor-set but should beat random).
    assert s["avg_f1"] > 0.5
    # Every T3 path query should have exact=True on this small graph.
    assert s["path_exact_accuracy"] == 1.0


def test_report_is_jsonable(ckg):
    golden = load_golden(GOLDEN)
    report = run(ckg, golden)
    payload = report.to_jsonable()
    assert "rows" in payload and "summary" in payload
    import json

    # Must round-trip to JSON without errors.
    json.dumps(payload, default=str)
