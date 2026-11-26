# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""The knowledge spec: a declarative "what good looks like", and the decision.

A spec holds the questions a decision model is asked about each page, the
thresholds that turn its answers into a disposition, how soon a held page is
looked at again, and targets (named, recorded, not enforced in this slice).

``decide`` is a pure function: spec + the model's answers -> a
:class:`Disposition`, or ``None`` when the answers cannot support one (a
required answer is missing or malformed). A page with no decision simply stays
undecided and is asked again on a later pass.

Slice 1 is shadow mode: a disposition is *recorded*, never acted on.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Optional

from trellion.live import systemone

#: Version reported when ``kb_specs`` holds no row: the built-in default.
DEFAULT_SPEC_VERSION = 1

#: Question ids the decision reads. ``embed`` and ``graph`` must be ``noul``.
Q_EMBED = "embed"
Q_GRAPH = "graph"
Q_USEFULNESS = "usefulness"
REQUIRED_QUESTIONS = (Q_EMBED, Q_GRAPH)

#: Targets named so far. Recorded with the spec, enforced by nothing yet.
KNOWN_TARGETS = (
    "retrieval_recall_at_k",
    "retrieval_p95_latency_ms",
    "embed_budget_pages_per_day",
    "extraction_budget_calls_per_day",
    "index_size_budget_mb",
)

_TOP_KEYS = frozenset({"questions", "thresholds", "review", "targets"})
_MAX_REVIEW_DAYS = 3650


class SpecError(ValueError):
    """The spec is not valid; the message says which field."""


@dataclass(frozen=True)
class Question:
    id: str
    type: str
    instructions: str
    criteria: Any = None  # a list or a dict, passed to the model as given

    def to_wire(self) -> dict[str, Any]:
        out: dict[str, Any] = {"type": self.type, "instructions": self.instructions}
        if self.criteria is not None:
            out["criteria"] = copy.deepcopy(self.criteria)
        return out


@dataclass(frozen=True)
class Thresholds:
    """Probabilities. ``embed_yes`` / ``graph_yes``: at or above is a yes.
    ``hold_below``: a page whose yes-probabilities are all below this is held
    with the ordinary review; one that falls between ``hold_below`` and a yes
    threshold is held with the early (band) review."""

    embed_yes: float = 0.60
    graph_yes: float = 0.60
    hold_below: float = 0.25


@dataclass(frozen=True)
class Review:
    """Days until a held page is re-examined."""

    held_days: int = 90
    band_days: int = 14


@dataclass(frozen=True)
class KnowledgeSpec:
    version: int
    questions: tuple[Question, ...]
    thresholds: Thresholds = field(default_factory=Thresholds)
    review: Review = field(default_factory=Review)
    targets: Mapping[str, Optional[float]] = field(default_factory=dict)

    def wire_questions(self) -> dict[str, dict[str, Any]]:
        return {q.id: q.to_wire() for q in self.questions}

    def to_dict(self) -> dict[str, Any]:
        """The storable body (the version lives in its own column)."""
        return {
            "questions": {
                q.id: q.to_wire() for q in self.questions
            },
            "thresholds": {
                "embed_yes": self.thresholds.embed_yes,
                "graph_yes": self.thresholds.graph_yes,
                "hold_below": self.thresholds.hold_below,
            },
            "review": {
                "held_days": self.review.held_days,
                "band_days": self.review.band_days,
            },
            "targets": dict(self.targets),
        }

    def thresholds_dict(self) -> dict[str, float]:
        return self.to_dict()["thresholds"]

    def question_ids(self) -> list[str]:
        return [q.id for q in self.questions]

    def body_hash(self) -> str:
        """SHA-256 of the canonical spec body (the version is not part of it).

        Recorded with every disposition, so what "version 1" meant when a row
        was written stays recoverable if the built-in default ever changes.
        """
        canon = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canon.encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, data: Any, version: int) -> "KnowledgeSpec":
        """Validate and build. Raises :class:`SpecError`."""
        if isinstance(version, bool) or not isinstance(version, int) or version < 1:
            raise SpecError("version must be an integer >= 1")
        if not isinstance(data, Mapping):
            raise SpecError("spec must be an object")
        unknown = set(data) - _TOP_KEYS
        if unknown:
            raise SpecError(f"unknown spec keys: {sorted(unknown)}")
        questions = _questions(data.get("questions"))
        thresholds = _thresholds(data.get("thresholds", {}))
        review = _review(data.get("review", {}))
        targets = _targets(data.get("targets", {}))
        return cls(version, questions, thresholds, review, targets)


def _questions(raw: Any) -> tuple[Question, ...]:
    if not isinstance(raw, Mapping) or not raw:
        raise SpecError("questions must be a non-empty object")
    out: list[Question] = []
    for qid, q in raw.items():
        if not isinstance(qid, str) or not qid.strip():
            raise SpecError("question ids must be non-empty strings")
        if not isinstance(q, Mapping):
            raise SpecError(f"question {qid!r} must be an object")
        qtype = q.get("type")
        if qtype not in systemone.QUESTION_TYPES:
            raise SpecError(f"question {qid!r}: type must be one of {systemone.QUESTION_TYPES}")
        instr = q.get("instructions")
        if not isinstance(instr, str) or not instr.strip():
            raise SpecError(f"question {qid!r}: instructions must be a non-empty string")
        criteria = q.get("criteria")
        if criteria is not None and not isinstance(criteria, (list, dict)):
            raise SpecError(f"question {qid!r}: criteria must be a list or an object")
        if qtype in ("choice", "score") and (not criteria or len(criteria) < 2):
            raise SpecError(f"question {qid!r}: a {qtype} question needs at least two criteria")
        extra = set(q) - {"type", "instructions", "criteria"}
        if extra:
            raise SpecError(f"question {qid!r}: unknown keys {sorted(extra)}")
        out.append(Question(qid, qtype, instr.strip(), copy.deepcopy(criteria)))
    by_id = {q.id: q for q in out}
    for need in REQUIRED_QUESTIONS:
        if need not in by_id:
            raise SpecError(f"questions must include {need!r}")
        if by_id[need].type != "noul":
            raise SpecError(f"question {need!r} must be of type noul")
    if Q_USEFULNESS in by_id and by_id[Q_USEFULNESS].type != "score":
        raise SpecError(f"question {Q_USEFULNESS!r} must be of type score")
    return tuple(out)


def _unit(name: str, v: Any) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1:
        raise SpecError(f"thresholds.{name} must be a number in [0, 1]")
    return float(v)


def _thresholds(raw: Any) -> Thresholds:
    if not isinstance(raw, Mapping):
        raise SpecError("thresholds must be an object")
    unknown = set(raw) - {"embed_yes", "graph_yes", "hold_below"}
    if unknown:
        raise SpecError(f"unknown thresholds keys: {sorted(unknown)}")
    d = Thresholds()
    t = Thresholds(
        _unit("embed_yes", raw.get("embed_yes", d.embed_yes)),
        _unit("graph_yes", raw.get("graph_yes", d.graph_yes)),
        _unit("hold_below", raw.get("hold_below", d.hold_below)),
    )
    if t.hold_below > min(t.embed_yes, t.graph_yes):
        raise SpecError("thresholds.hold_below must not exceed embed_yes or graph_yes")
    return t


def _days(name: str, v: Any) -> int:
    if isinstance(v, bool) or not isinstance(v, int) or not 1 <= v <= _MAX_REVIEW_DAYS:
        raise SpecError(f"review.{name} must be an integer from 1 to {_MAX_REVIEW_DAYS}")
    return v


def _review(raw: Any) -> Review:
    if not isinstance(raw, Mapping):
        raise SpecError("review must be an object")
    unknown = set(raw) - {"held_days", "band_days"}
    if unknown:
        raise SpecError(f"unknown review keys: {sorted(unknown)}")
    d = Review()
    r = Review(_days("held_days", raw.get("held_days", d.held_days)),
               _days("band_days", raw.get("band_days", d.band_days)))
    if r.band_days > r.held_days:
        raise SpecError("review.band_days must not exceed review.held_days")
    return r


def _targets(raw: Any) -> dict[str, Optional[float]]:
    if not isinstance(raw, Mapping):
        raise SpecError("targets must be an object")
    out: dict[str, Optional[float]] = {name: None for name in KNOWN_TARGETS}
    for k, v in raw.items():
        if not isinstance(k, str) or not k:
            raise SpecError("target names must be non-empty strings")
        if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)):
            raise SpecError(f"targets.{k} must be a number or null")
        out[k] = None if v is None else float(v)
    return out


def default_spec() -> KnowledgeSpec:
    """The built-in spec (version 1), used when ``kb_specs`` is empty."""
    return KnowledgeSpec.from_dict(
        {
            "questions": {
                Q_EMBED: {
                    "type": "noul",
                    "instructions": (
                        "Would retrieving this page help answer a future question "
                        "about its subject?"
                    ),
                },
                Q_GRAPH: {
                    "type": "noul",
                    "instructions": (
                        "Does this page state durable facts about named people, "
                        "organisations, systems or their relationships worth "
                        "recording as entities and relations?"
                    ),
                },
                Q_USEFULNESS: {
                    "type": "score",
                    "instructions": (
                        "How useful is this page as a source of knowledge for "
                        "answering future questions?"
                    ),
                    "criteria": [
                        "noise or boilerplate",
                        "marginal: little beyond what is obvious from its title",
                        "useful context for a narrow topic",
                        "important reference material",
                        "core reference",
                    ],
                },
            },
            "thresholds": {"embed_yes": 0.60, "graph_yes": 0.60, "hold_below": 0.25},
            "review": {"held_days": 90, "band_days": 14},
            "targets": {},
        },
        DEFAULT_SPEC_VERSION,
    )


@dataclass(frozen=True)
class Disposition:
    """What the spec says to do with a page (recorded, not enforced)."""

    spec_version: int
    would_embed: bool
    would_graph: bool
    held: bool
    review_after: Optional[datetime]
    p_embed: float
    p_graph: float
    usefulness: Optional[float] = None
    in_band: bool = False


def decide(
    spec: KnowledgeSpec, answers: Any, now: Optional[datetime] = None
) -> Optional[Disposition]:
    """The disposition for ``answers`` under ``spec``, or None for no decision.

    None when ``answers`` is not an object, or the ``embed`` or ``graph``
    answer is missing or malformed. ``usefulness`` is recorded when it parses
    and is otherwise ignored: it does not gate the decision.

    A page is *embedded* when P(embed) >= ``embed_yes`` and *graphed* when
    P(graph) >= ``graph_yes``; one that is neither is *held*. A held page
    whose probability sits between ``hold_below`` and its yes threshold is in
    the band and gets the early review.
    """
    if not isinstance(answers, Mapping):
        return None
    p_embed = systemone.noul_probability(answers.get(Q_EMBED)) if Q_EMBED in answers else None
    p_graph = systemone.noul_probability(answers.get(Q_GRAPH)) if Q_GRAPH in answers else None
    if p_embed is None or p_graph is None:
        return None
    t = spec.thresholds
    would_embed = p_embed >= t.embed_yes
    would_graph = p_graph >= t.graph_yes
    held = not (would_embed or would_graph)
    in_band = held and (t.hold_below <= p_embed < t.embed_yes or t.hold_below <= p_graph < t.graph_yes)
    review_after = None
    if held:
        base = now or datetime.now(timezone.utc)
        days = spec.review.band_days if in_band else spec.review.held_days
        review_after = base + timedelta(days=days)
    usefulness = systemone.score_value(answers.get(Q_USEFULNESS)) if Q_USEFULNESS in answers else None
    return Disposition(
        spec_version=spec.version,
        would_embed=would_embed,
        would_graph=would_graph,
        held=held,
        review_after=review_after,
        p_embed=p_embed,
        p_graph=p_graph,
        usefulness=usefulness,
        in_band=in_band,
    )
