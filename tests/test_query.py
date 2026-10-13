# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""Smoke tests for the CKG query layer.

These run offline (no LLM, no network) against the bundled sample CSV.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from trellion.query.answer import answer_query, classify
from trellion.query.graph import CKG

SAMPLE = Path(__file__).parents[1] / "data" / "sample" / "learning-graph.csv"


@pytest.fixture
def ckg():
    return CKG.load(SAMPLE)


def test_load(ckg):
    assert len(ckg.concepts) == 8
    assert ckg.find_by_label("Function") is not None
    # Composite Function depends on Function + Function Notation.
    assert set(ckg.forward[4]) == {1, 3}


def test_classify_intents():
    assert classify("What is Composite Function?") == "T1"
    assert classify("What are prerequisites for Composite Function?") == "T2"
    assert classify("What is the chain from Function to Taylor Series?") == "T3"
    assert classify("List all FOUND concepts") == "T4"
    assert classify("How does Domain and Range relate to Inverse Function?") == "T5"


def test_t2_direct_dependency(ckg):
    ans = answer_query(ckg, "What are the prerequisites for Composite Function?")
    assert ans.query_type == "T2"
    assert "Function" in ans.content
    assert "Function Notation" in ans.content
    assert ans.retrieved_tokens < 50


def test_t3_path(ckg):
    ans = answer_query(ckg, "What is the chain from Function to Taylor Series?")
    assert ans.query_type == "T3"
    assert ans.content.index("Function") < ans.content.index("Taylor Series")


def test_t3_respects_direction(ckg):
    ans = answer_query(ckg, "What is the chain from Taylor Series to Function?")
    assert ans.query_type == "T3"
    assert ans.content.index("Taylor Series") < ans.content.index("Function")


def test_t4_taxonomy(ckg):
    ans = answer_query(ckg, "List all FOUND concepts")
    assert ans.query_type == "T4"
    assert "Function" in ans.content
    assert "Composite Function" not in ans.content


def test_t1_is_low_token(ckg):
    ans = answer_query(ckg, "What is Composite Function?")
    assert ans.query_type == "T1"
    assert ans.retrieved_tokens < 30
