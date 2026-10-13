# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""Entity-name quality gate: the one place that decides "is this a real entity?".

The extraction prompt asks the model to be thorough and offers ``variable``
and ``parameter`` as types, so a page about a paper comes back with ``a``,
``x_t``, ``theta``, ``0.003`` and ``$1,200`` as "entities". Nothing downstream
objected: the only check was ``if not name``. Those rows then pick up
co-occurrence edges and surface as graph walks like ``model-m -> a -> d -> x``
and cluster labels like "a, theta, example topic".

Everything that creates an entity, or an edge endpoint, asks this module:

* ``trellion.live.graph_extractor`` (live extraction, after parsing)
* ``trellion.live.graph_builder`` (co-occurrence edges; also covers junk
  that is already in the table)
* ``trellion.build.extractors.llm`` (offline build)
* add-on extractors, when installed (typed relations; same helper)
* ``trellion.live.junk_entity_cleanup`` (stages what is already stored)

Design rule: false rejections are worse than leftover junk. Every rule below
is narrow, and the ambiguous ones (a spelled-out ``alpha``, ``s2``) only fire
when the entity type says it is a math symbol (``variable`` / ``parameter``).

The rules are data: ``DEFAULT_CONFIG`` below, optionally overlaid by a JSON
file named in ``TRELLION_ENTITY_QUALITY_CONFIG``::

    {"enabled": true,
     "allow": ["ml", "qa"],             # extra short names that are real
     "stopwords": ["whereas"],          # extra names that are never entities
     "reject_patterns": ["^tmp_\\\\d+$"]}   # extra regexes (case-insensitive)

``TRELLION_ENTITY_QUALITY=off`` is the kill switch.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Optional

import structlog

log = structlog.get_logger(__name__)

# Types under which an ambiguous short token is a math symbol, not a thing.
MATH_TYPES = frozenset({"variable", "parameter"})
# Types for which a short all-caps token is plausibly a real acronym.
ACRONYM_TYPES = frozenset({"organization", "person"})
# Types whose two-letter alphabetic name is a real name in ANY case: a surname
# ("Li", "Wu"), a company ("HP", "EA"), a place ("LA"). Stored names are
# lowercased, so the case-based acronym test above cannot see these when the
# cleanup re-checks what is already in the table.
NAME_TYPES = frozenset({"organization", "person", "location"})
# Types under which a bare 3-5 digit integer is a model number ("3090",
# "4090", "1660"), not a quantity.
MODEL_TYPES = frozenset({"component", "material", "product", "device", "hardware"})

_GREEK_LETTERS = "α-ωΑ-Ωϑϕϰϱϖ"
_GREEK_NAMES = (
    "alpha|beta|gamma|delta|epsilon|zeta|eta|theta|iota|kappa|lambda|mu|nu|xi|"
    "omicron|pi|rho|sigma|tau|upsilon|phi|chi|psi|omega"
)

# Short names that ARE real. Compared case-insensitively, except the
# single-letter entries, which are case-sensitive: "R" the language is real,
# "r" the radius is not.
_DEFAULT_ALLOW = frozenset({
    # fields / technologies
    "ai", "ml", "dl", "rl", "nn", "cv", "ui", "ux", "os", "db", "ci", "cd",
    "vr", "ar", "xr", "tv", "pc", "hr", "pr", "qa", "io", "js", "ts", "py",
    "vm", "ip", "ide", "π0", "t5", "s3", "c#", "f#", "2d", "3d", "4d", "3g", "4g", "5g", "4k", "8k",
    "go",
    # places / bodies
    "un", "eu", "uk", "us", "uae", "nz", "ca", "nl",
    # single letters that are languages
    "R", "C",
})

# Function words and pronouns. Never an entity on their own.
_DEFAULT_STOPWORDS = frozenset({
    "a", "an", "the", "it", "is", "of", "and", "or", "to", "in", "this", "that",
    "these", "those", "there", "we", "i", "he", "she", "they", "you", "them",
    "be", "as", "at", "by", "for", "on", "with", "if", "are", "was", "its",
    "our", "not", "no", "so", "do", "which", "what", "when", "where", "who",
    "how", "why", "here", "then", "than", "such", "from", "into", "but",
})


@dataclass(frozen=True)
class QualityConfig:
    enabled: bool = True
    allow: frozenset[str] = _DEFAULT_ALLOW
    stopwords: frozenset[str] = _DEFAULT_STOPWORDS
    extra_patterns: tuple[re.Pattern, ...] = field(default_factory=tuple)


DEFAULT_CONFIG = QualityConfig()

_config_cache: Optional[QualityConfig] = None


def load_config(path: Optional[str] = None) -> QualityConfig:
    """Defaults, overlaid by the JSON file at ``path`` (or the env var)."""
    path = path or os.environ.get("TRELLION_ENTITY_QUALITY_CONFIG")
    enabled = os.environ.get("TRELLION_ENTITY_QUALITY", "on").strip().lower() not in (
        "off", "0", "false", "no",
    )
    allow, stop, pats = set(_DEFAULT_ALLOW), set(_DEFAULT_STOPWORDS), []
    if path:
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            enabled = enabled and bool(data.get("enabled", True))
            allow |= {str(x) for x in data.get("allow", [])}
            stop |= {str(x).lower() for x in data.get("stopwords", [])}
            for p in data.get("reject_patterns", []):
                pats.append(re.compile(str(p), re.IGNORECASE))
        except (OSError, ValueError, re.error) as exc:
            # A broken override must not take extraction down, and must not
            # silently switch the gate off either: defaults stay in force.
            log.warning("entity_quality.config_unreadable", path=path, error=repr(exc))
    return QualityConfig(enabled, frozenset(allow), frozenset(stop), tuple(pats))


def get_config() -> QualityConfig:
    global _config_cache
    if _config_cache is None:
        _config_cache = load_config()
    return _config_cache


def reset_config_cache() -> None:
    global _config_cache
    _config_cache = None


# --- patterns -------------------------------------------------------------

_RE_HAS_ALNUM = re.compile(r"[^\W_]", re.UNICODE)
# \alpha, \\, ^, _{...}, braces, $...$ math fences
_RE_LATEX = re.compile(r"\\[A-Za-z\\]|\^|_\{|[{}]|\$[^$]*\$")
# 12, 3.5, 1,200, 33%, -4, 1/2, 10:30, 1-2, $25, EUR-ish prefixes
_RE_NUMERIC = re.compile(r"^[-+~]?\s?[$€£¥]?\s?\d[\d.,/:\-]*\s?%?$")
# "$1,400 a month", "$1,500 bucks": an amount of money, not an entity
_RE_MONEY = re.compile(r"^[$€£¥]\s?\d")
_RE_GREEK_CHAR = re.compile(rf"^[{_GREEK_LETTERS}]$")
_RE_GREEK_SUB = re.compile(rf"^[{_GREEK_LETTERS}](?:[_'′’`]|\[).*$")
_RE_GREEK_NAME = re.compile(rf"^(?:{_GREEK_NAMES})$", re.IGNORECASE)
# x_t, x_{t}, a_1, f_θ, z_t1  (one letter, underscore, short subscript)
_RE_SUBSCRIPT = re.compile(rf"^[A-Za-z{_GREEK_LETTERS}]_[A-Za-z0-9{_GREEK_LETTERS}+\-]{{1,3}}$")
# x', x'', x′
_RE_PRIME = re.compile(r"^[A-Za-z]['′’`]+$")
# s[3], a[0], s[t-1], s2[4]
_RE_INDEXED = re.compile(r"^[A-Za-z]\d?\[[^\]]*\]$")
# z+1, t-1
_RE_OFFSET = re.compile(r"^[A-Za-z][+\-]\d+$")
# f(x), l(θ), r(z), kl(sg[q] || p): a 1-2 letter symbol applied to symbols
_RE_SYMBOL_CALL = re.compile(r"^[A-Za-z]{1,2}\((?!.*[A-Za-z]{4}).*\)$")
# one letter + one digit: x1, s2, c1 (math types only: "t5", "s3" are real)
_RE_LETTER_DIGIT = re.compile(r"^[A-Za-z]\d$")
_RE_ACRONYM_FORM = re.compile(r"^(?=.*[A-Z])[A-Z0-9]{2}$")
_RE_TWO_LETTERS = re.compile(r"^[A-Za-z]{2}$")
# a model number: a bare integer of 3-5 digits, no separators
_RE_MODEL_NUMBER = re.compile(r"^\d{3,5}$")


def check_entity(
    name: object, etype: Optional[str] = None, config: Optional[QualityConfig] = None
) -> Optional[str]:
    """Return None if ``name`` may be an entity, else the reason it may not.

    ``name`` should be the name as the model wrote it (before lowercasing):
    the acronym exception needs the case. A name that is already lowercased
    simply gets no acronym exception, which errs toward keeping nothing odd.
    """
    cfg = config or get_config()
    if not cfg.enabled:
        return None
    if not isinstance(name, str):
        return "not_a_string"
    n = re.sub(r"\s+", " ", name).strip()
    if not n:
        return "empty"
    t = (etype or "").strip().lower() or None
    low = n.lower()

    if n in cfg.allow or (len(n) > 1 and low in cfg.allow):
        return None
    if low in cfg.stopwords:
        return "stopword"
    if not _RE_HAS_ALNUM.search(n):
        return "punctuation"
    if _RE_MONEY.match(n):
        return "numeric"
    if _RE_LATEX.search(n):
        return "latex"
    if _RE_NUMERIC.match(n):
        if t in MODEL_TYPES and _RE_MODEL_NUMBER.match(n):
            return None
        return "numeric"
    if _RE_GREEK_CHAR.match(n) or _RE_GREEK_SUB.match(n):
        return "greek_symbol"
    if t in MATH_TYPES and _RE_GREEK_NAME.match(n):
        return "greek_symbol"
    if (
        _RE_SUBSCRIPT.match(n)
        or _RE_PRIME.match(n)
        or _RE_INDEXED.match(n)
        or _RE_OFFSET.match(n)
        or _RE_SYMBOL_CALL.match(n)
    ):
        return "math_symbol"
    if t in MATH_TYPES and _RE_LETTER_DIGIT.match(n):
        return "math_symbol"
    for pat in cfg.extra_patterns:
        if pat.search(n):
            return "configured_pattern"
    if len(n) <= 2:
        if t in ACRONYM_TYPES and _RE_ACRONYM_FORM.match(n):
            return None
        if t in NAME_TYPES and _RE_TWO_LETTERS.match(n):
            return None
        # a letter and a digit outside the math types is a product or model
        # code ("i7", "M2", "T4", "o3", "H2"); under them it returned above
        if t and _RE_LETTER_DIGIT.match(n):
            return None
        return "too_short"
    return None


def is_junk_entity(name: object, etype: Optional[str] = None) -> bool:
    return check_entity(name, etype) is not None


class DropTally:
    """Counts what a pass rejected, for one log line instead of one per name."""

    def __init__(self) -> None:
        self.by_reason: Counter[str] = Counter()
        self.examples: list[str] = []

    def add(self, name: object, reason: str) -> None:
        self.by_reason[reason] += 1
        if len(self.examples) < 5:
            self.examples.append(str(name)[:40])

    @property
    def total(self) -> int:
        return sum(self.by_reason.values())

    def log(self, event: str, **ctx: object) -> None:
        if self.total:
            log.info(
                event, dropped=self.total, by_reason=dict(self.by_reason),
                examples=self.examples, **ctx,
            )


def partition(
    items: Iterable[tuple[str, str, Optional[str]]],
) -> tuple[list[str], DropTally]:
    """Split ``(key, name, type)`` triples; return (junk keys, tally)."""
    tally = DropTally()
    junk: list[str] = []
    for key, name, etype in items:
        reason = check_entity(name, etype)
        if reason:
            junk.append(key)
            tally.add(name, reason)
    return junk, tally
