# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""The SystemOne wire format, the knowledge spec, and ``decide`` (no database)."""

from __future__ import annotations

import copy
import math
from datetime import datetime, timedelta, timezone

import pytest

from trellion.live import systemone
from trellion.live.knowledge_spec import (
    DEFAULT_SPEC_VERSION,
    KnowledgeSpec,
    SpecError,
    decide,
    default_spec,
)

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _answers(embed, graph, usefulness=None):
    out = {"embed": embed, "graph": graph}
    if usefulness is not None:
        out["usefulness"] = usefulness
    return out


# ---- noul parsing --------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        (0.8, 0.8),
        (0, 0.0),
        (1, 1.0),
        ("0.25", 0.25),
        ({"probability": 0.7}, 0.7),
        ({"probabilities": {"true": 0.9, "false": 0.1}}, 0.9),
        ({"probabilities": {"True": 0.9}}, 0.9),
        ({"probabilities": {"false": 0.25}}, 0.75),
        ({"probabilities": {"yes": 0.6, "no": 0.4}, "confidence": 0.5}, 0.6),
    ],
)
def test_noul_probability_accepts_both_shapes(raw, expected):
    assert systemone.noul_probability(raw) == pytest.approx(expected)


@pytest.mark.parametrize(
    "raw",
    [
        None, True, False, "yes", "", [], [0.5], {}, {"probability": 1.5},
        {"probability": -0.1}, {"probability": float("nan")}, {"probability": "x"},
        {"probabilities": []}, {"probabilities": {}}, {"probabilities": {"true": 2}},
        {"probabilities": {"true": 0.9, "false": 0.9}},  # contradicts itself
        {"probabilities": {"maybe": 0.5}},
        1.0000001, float("inf"), {"score": 0.5},
    ],
)
def test_noul_probability_rejects_malformed(raw):
    assert systemone.noul_probability(raw) is None


def test_score_value_and_choice_answer():
    assert systemone.score_value({"score": 3.4, "confidence": 0.9, "legend": {}}) == pytest.approx(3.4)
    assert systemone.score_value(2) == 2.0
    for bad in (None, {}, {"score": "high"}, {"score": float("nan")}, {"score": True}):
        assert systemone.score_value(bad) is None
    ok = systemone.choice_answer(
        {"choice": "a", "confidence": 0.7, "probabilities": {"a": 0.7, "b": 0.3}}
    )
    assert ok == {"choice": "a", "confidence": 0.7, "probabilities": {"a": 0.7, "b": 0.3}}
    bare = systemone.choice_answer({"choice": "a"})
    assert bare == {"choice": "a", "confidence": None, "probabilities": None}
    for bad in (None, "a", {}, {"choice": ""}, {"choice": 3},
                {"choice": "z", "probabilities": {"a": 1.0}},
                {"choice": "a", "probabilities": {"a": 7}},
                {"choice": "a", "probabilities": []}):
        assert systemone.choice_answer(bad) is None


def test_parse_response():
    assert systemone.parse_response({"model": "m", "answers": {"x": 1}, "usage": {}}) == {"x": 1}
    for bad in (None, [], "text", {}, {"answers": []}, {"answers": None}):
        assert systemone.parse_response(bad) is None


def test_build_request_omits_empty_model():
    q = {"a": {"type": "noul", "instructions": "i"}}
    assert "model" not in systemone.build_request(None, "s", q)
    body = systemone.build_request("m", {"k": 1}, q)
    assert body == {"model": "m", "state": {"k": 1}, "questions": q}
    assert body["questions"] is not q


# ---- the default spec ----------------------------------------------------


def test_default_spec_shape():
    spec = default_spec()
    assert spec.version == DEFAULT_SPEC_VERSION == 1
    wire = spec.wire_questions()
    assert wire["embed"]["type"] == "noul" and "future question" in wire["embed"]["instructions"]
    assert wire["graph"]["type"] == "noul" and "durable facts" in wire["graph"]["instructions"]
    assert wire["usefulness"]["type"] == "score"
    crit = wire["usefulness"]["criteria"]
    assert len(crit) == 5 and crit[0] == "noise or boilerplate" and crit[-1] == "core reference"
    t = spec.thresholds
    assert (t.embed_yes, t.graph_yes, t.hold_below) == (0.60, 0.60, 0.25)
    assert (spec.review.held_days, spec.review.band_days) == (90, 14)
    assert set(spec.targets) >= {"retrieval_recall_at_k", "index_size_budget_mb"}
    assert all(v is None for v in spec.targets.values())  # named, not set, not enforced


def test_spec_round_trips_through_its_stored_body():
    spec = default_spec()
    again = KnowledgeSpec.from_dict(spec.to_dict(), 7)
    assert again.version == 7
    assert again.to_dict() == spec.to_dict()


# ---- validation ----------------------------------------------------------


def _body():
    return copy.deepcopy(default_spec().to_dict())


def _bad(mutate, match):
    body = _body()
    mutate(body)
    with pytest.raises(SpecError, match=match):
        KnowledgeSpec.from_dict(body, 2)


def test_validation_rejects_bad_specs():
    _bad(lambda b: b["questions"].pop("embed"), "embed")
    _bad(lambda b: b["questions"].pop("graph"), "graph")
    _bad(lambda b: b["questions"]["embed"].update(type="score", criteria=["a", "b"]), "noul")
    _bad(lambda b: b["questions"]["embed"].update(type="text"), "type")
    _bad(lambda b: b["questions"]["embed"].update(instructions="  "), "instructions")
    _bad(lambda b: b["questions"]["usefulness"].update(criteria=["only one"]), "two criteria")
    _bad(lambda b: b["questions"]["usefulness"].update(criteria="nope"), "criteria")
    _bad(lambda b: b["questions"]["embed"].update(extra=1), "unknown keys")
    _bad(lambda b: b.update(surprise=1), "unknown spec keys")
    _bad(lambda b: b["thresholds"].update(embed_yes=1.2), "embed_yes")
    _bad(lambda b: b["thresholds"].update(graph_yes=True), "graph_yes")
    _bad(lambda b: b["thresholds"].update(hold_below=float("nan")), "hold_below")
    _bad(lambda b: b["thresholds"].update(hold_below=0.9), "hold_below")
    _bad(lambda b: b["review"].update(held_days=0), "held_days")
    _bad(lambda b: b["review"].update(band_days=1.5), "band_days")
    _bad(lambda b: b["review"].update(band_days=100, held_days=10), "band_days")
    _bad(lambda b: b["targets"].update(x="fast"), "targets.x")
    _bad(lambda b: b.update(questions={}), "non-empty")
    with pytest.raises(SpecError):
        KnowledgeSpec.from_dict([], 2)
    with pytest.raises(SpecError):
        KnowledgeSpec.from_dict(_body(), 0)


def test_minimal_spec_takes_defaults():
    body = {"questions": _body()["questions"]}
    spec = KnowledgeSpec.from_dict(body, 3)
    assert spec.thresholds.embed_yes == 0.60 and spec.review.held_days == 90


# ---- decide ---------------------------------------------------------------


SPEC = default_spec()  # embed_yes 0.60, graph_yes 0.60, hold_below 0.25, 90 / 14 days


def test_decide_both_yes():
    d = decide(SPEC, _answers(0.9, 0.8, {"score": 4.2}), NOW)
    assert (d.would_embed, d.would_graph, d.held) == (True, True, False)
    assert d.review_after is None and d.usefulness == pytest.approx(4.2)
    assert d.spec_version == 1


def test_decide_boundaries_are_inclusive_at_the_yes_threshold():
    at = decide(SPEC, _answers(0.60, 0.0), NOW)
    assert at.would_embed is True and at.held is False
    just_under = decide(SPEC, _answers(0.5999, 0.0), NOW)
    assert just_under.would_embed is False and just_under.held is True


def test_decide_graph_only_and_embed_only():
    g = decide(SPEC, _answers(0.1, 0.7), NOW)
    assert (g.would_embed, g.would_graph, g.held) == (False, True, False)
    e = decide(SPEC, _answers(0.7, 0.1), NOW)
    assert (e.would_embed, e.would_graph, e.held) == (True, False, False)


def test_decide_low_is_held_with_the_ordinary_review():
    d = decide(SPEC, _answers(0.05, 0.10), NOW)
    assert d.held and not d.in_band
    assert d.review_after == NOW + timedelta(days=90)


def test_decide_band_is_held_with_the_early_review():
    d = decide(SPEC, _answers(0.40, 0.05), NOW)
    assert d.held and d.in_band
    assert d.review_after == NOW + timedelta(days=14)
    # the lower bound is inclusive: exactly hold_below is in the band
    assert decide(SPEC, _answers(0.25, 0.0), NOW).in_band is True
    assert decide(SPEC, _answers(0.2499, 0.2499), NOW).in_band is False
    # either answer in the band is enough
    assert decide(SPEC, _answers(0.0, 0.5), NOW).in_band is True


def test_decide_accepts_the_object_shapes():
    d = decide(SPEC, _answers({"probability": 0.9}, {"probabilities": {"true": 0.1, "false": 0.9}}), NOW)
    assert d.would_embed and not d.would_graph


@pytest.mark.parametrize(
    "answers",
    [
        None, [], "x", {},
        {"embed": 0.9}, {"graph": 0.9},                      # missing answer
        {"embed": 0.9, "graph": None},
        {"embed": 0.9, "graph": "high"},                     # malformed
        {"embed": 1.7, "graph": 0.5},                        # not a probability
        {"embed": {"probabilities": {"true": 0.9, "false": 0.9}}, "graph": 0.5},
        {"embed": float("nan"), "graph": 0.5},
        {"embed": True, "graph": 0.5},
    ],
)
def test_decide_records_nothing_without_usable_answers(answers):
    assert decide(SPEC, answers, NOW) is None


def test_decide_ignores_a_bad_usefulness_answer():
    d = decide(SPEC, {"embed": 0.9, "graph": 0.9, "usefulness": "five"}, NOW)
    assert d is not None and d.usefulness is None
    assert decide(SPEC, {"embed": 0.9, "graph": 0.9}, NOW).usefulness is None


def test_decide_uses_the_specs_own_thresholds_and_version():
    body = _body()
    body["thresholds"].update(embed_yes=0.9, graph_yes=0.95, hold_below=0.5)
    body["review"].update(held_days=30, band_days=3)
    spec = KnowledgeSpec.from_dict(body, 5)
    d = decide(spec, _answers(0.8, 0.7), NOW)
    assert d.spec_version == 5 and d.held and d.in_band
    assert d.review_after == NOW + timedelta(days=3)
    assert decide(spec, _answers(0.3, 0.3), NOW).review_after == NOW + timedelta(days=30)
    assert not math.isnan(d.p_embed)
