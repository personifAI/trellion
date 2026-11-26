# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""The SystemOne wire format: request building and tolerant response parsing.

A decision model (for example Clef-Flash) answers structured questions about a
``state`` and generates no text. The request is::

    POST /v1/systemone
    {"model": "...", "state": <string or JSON>,
     "questions": {<id>: {"type": "choice"|"score"|"noul",
                          "instructions": "...", "criteria": {...} | [...]}}}

and the response is::

    {"model": "...", "answers": {<id>: {...}}, "usage": {...}}

* a ``choice`` answer carries ``choice``, ``confidence`` and ``probabilities``
  (option -> probability);
* a ``score`` answer carries ``score`` (the expected value), ``confidence``,
  ``legend`` and ``probabilities``;
* a ``noul`` answer carries the probability that the statement is true, either
  as a bare number or as an object with ``probability`` / ``probabilities``.

Every parser here returns ``None`` for anything it cannot read with
confidence. A probability outside [0, 1], a non-finite number, a boolean
standing in for a number, or an object that contradicts itself is malformed,
never clamped: a decision is recorded only from answers that parse.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional

QUESTION_TYPES = ("choice", "score", "noul")

# How far a pair of true/false probabilities may stray from summing to 1.
_SUM_TOLERANCE = 0.1

_TRUE_KEYS = ("true", "yes", "t", "1")
_FALSE_KEYS = ("false", "no", "f", "0")


def build_request(
    model: Optional[str], state: Any, questions: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """The request body. ``model`` is omitted when empty (the endpoint's default)."""
    body: dict[str, Any] = {}
    if model:
        body["model"] = model
    body["state"] = state
    body["questions"] = {qid: dict(q) for qid, q in questions.items()}
    return body


def _number(value: Any) -> Optional[float]:
    """A finite float from a number or a numeric string; booleans are not numbers."""
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        try:
            value = float(value.strip())
        except ValueError:
            return None
    if isinstance(value, (int, float)):
        f = float(value)
        return f if math.isfinite(f) else None
    return None


def _prob(value: Any) -> Optional[float]:
    f = _number(value)
    return f if f is not None and 0.0 <= f <= 1.0 else None


def _pick(mapping: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for k, v in mapping.items():
        if isinstance(k, str) and k.strip().lower() in keys:
            return v
    return None


def parse_probabilities(raw: Any) -> Optional[dict[str, float]]:
    """option -> probability, or None when it is not a mapping of valid probabilities."""
    if not isinstance(raw, Mapping) or not raw:
        return None
    out: dict[str, float] = {}
    for k, v in raw.items():
        p = _prob(v)
        if not isinstance(k, str) or p is None:
            return None
        out[k] = p
    return out


def noul_probability(answer: Any) -> Optional[float]:
    """The probability of true from a ``noul`` answer, or None if malformed.

    Accepted shapes: ``0.8``; ``{"probability": 0.8}``;
    ``{"probabilities": {"true": 0.8, "false": 0.2}}`` (either side alone is
    enough; both must roughly sum to 1).
    """
    if isinstance(answer, Mapping):
        if "probability" in answer:
            return _prob(answer["probability"])
        if "probabilities" in answer:
            probs = answer["probabilities"]
            if not isinstance(probs, Mapping):
                return None
            t = _pick(probs, _TRUE_KEYS)
            f = _pick(probs, _FALSE_KEYS)
            pt = _prob(t) if t is not None else None
            pf = _prob(f) if f is not None else None
            if (t is not None and pt is None) or (f is not None and pf is None):
                return None
            if pt is not None and pf is not None:
                return pt if abs(pt + pf - 1.0) <= _SUM_TOLERANCE else None
            if pt is not None:
                return pt
            if pf is not None:
                return 1.0 - pf
            return None
        return None
    return _prob(answer)


def score_value(answer: Any) -> Optional[float]:
    """The expected score from a ``score`` answer (an object with ``score``, or a bare number)."""
    if isinstance(answer, Mapping):
        return _number(answer.get("score"))
    return _number(answer)


def choice_answer(answer: Any) -> Optional[dict[str, Any]]:
    """``{"choice", "confidence", "probabilities"}`` from a ``choice`` answer, or None.

    ``confidence`` and ``probabilities`` are None when absent or invalid; the
    choice itself must be a non-empty string, and when probabilities are given
    it must be one of their options.
    """
    if not isinstance(answer, Mapping):
        return None
    choice = answer.get("choice")
    if not isinstance(choice, str) or not choice:
        return None
    probs = parse_probabilities(answer["probabilities"]) if "probabilities" in answer else None
    if "probabilities" in answer and probs is None:
        return None
    if probs is not None and choice not in probs:
        return None
    return {"choice": choice, "confidence": _prob(answer.get("confidence")), "probabilities": probs}


def parse_response(body: Any) -> Optional[dict[str, Any]]:
    """The ``answers`` object of a response, or None when the body is not one."""
    if not isinstance(body, Mapping):
        return None
    answers = body.get("answers")
    if not isinstance(answers, Mapping):
        return None
    return {str(k): v for k, v in answers.items()}
