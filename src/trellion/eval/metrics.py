# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""Scoring primitives for eval runs.

Every function is deterministic, stateless, and O(n) — we want reproducible
scorecards that diff cleanly between runs.

Metric families:
  - `f1_set`    : set overlap F1 (precision/recall/f1) for the concept-set
                  query families (T2 prerequisites, T4 taxonomy members).
  - `path_match`: exact path equality (T3), plus a softer "endpoints match"
                  and "intermediate Jaccard" variant for partial credit.
  - `rds`       : Reasoning Density Score = F1 / tokens_consumed. Higher =
                  more answer quality per token.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass
class SetScore:
    precision: float
    recall: float
    f1: float


def f1_set(predicted: Iterable, expected: Iterable) -> SetScore:
    """Standard set-based precision/recall/F1. Empty-expected edge case:
    if both are empty, we return perfect 1.0 (correct "nothing to find")."""
    p = set(predicted)
    e = set(expected)
    if not p and not e:
        return SetScore(1.0, 1.0, 1.0)
    tp = len(p & e)
    precision = tp / len(p) if p else 0.0
    recall = tp / len(e) if e else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return SetScore(precision, recall, f1)


@dataclass
class PathScore:
    exact: bool
    endpoints: bool
    intermediate_jaccard: float


def path_match(predicted: list, expected: list) -> PathScore:
    """Path correctness at three granularities.

    - `exact`: identical ordered sequence.
    - `endpoints`: first and last nodes match (direction preserved).
    - `intermediate_jaccard`: set Jaccard of interior nodes (no order
      penalty). Catches "got the hops but BFS chose a sibling path"
      situations where the graph has multiple valid paths.
    """
    exact = list(predicted) == list(expected)
    endpoints = (
        len(predicted) > 0
        and len(expected) > 0
        and predicted[0] == expected[0]
        and predicted[-1] == expected[-1]
    )
    p_mid = set(predicted[1:-1]) if len(predicted) > 2 else set()
    e_mid = set(expected[1:-1]) if len(expected) > 2 else set()
    if not p_mid and not e_mid:
        jaccard = 1.0
    else:
        union = p_mid | e_mid
        jaccard = len(p_mid & e_mid) / len(union) if union else 0.0
    return PathScore(exact=exact, endpoints=endpoints, intermediate_jaccard=jaccard)


def rds(f1: float, tokens_consumed: int) -> float:
    """Reasoning Density Score: quality-per-token.

    Zero-token answers are illegal — return 0 rather than divide-by-zero.
    """
    if tokens_consumed <= 0:
        return 0.0
    return f1 / tokens_consumed
