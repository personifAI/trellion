# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""A completion cut off at the token cap must not lose every entity in it.

`_parse_json_array` had two strategies and both were all-or-nothing: parse
the whole string, or regex a `[ ... ]` block and parse that. A truncated
response has no closing bracket, so both failed and every entity was
discarded — however many had been written in full.

That was the normal case on the live controller, not an edge. Extraction ran
continuously, responses clustered just under the 2048-token ceiling (6,720
characters at the top), and `parsed_elements=0` came back for page after page
whose logged preview plainly began `[{"name": "jia deng", ...`. The graph
showed the cost: 10,039 of 11,619 edges left as generic RELATED_TO,
classification stuck at 14%, entity count moving by single digits across
hours of paid LLM calls.
"""

from __future__ import annotations

import json

import pytest

from trellion.live.graph_extractor import (
    _closes_cleanly,
    _parse_json_array,
    _salvage_objects,
)

WHOLE = [
    {"name": "jia deng", "type": "person", "aliases": [], "quote": "Jia Deng"},
    {"name": "wei dong", "type": "person", "aliases": [], "quote": "Wei Dong"},
    {"name": "imagenet", "type": "concept", "aliases": ["ILSVRC"], "quote": "ImageNet"},
]


def _truncate_mid_element(objs: list[dict]) -> str:
    """A well-formed array cut partway through its LAST element."""
    full = json.dumps(objs, indent=2)
    return full[: full.rindex('{"name"') if '{"name"' in full else len(full) - 40][:-1] + '\n  {"name": "half-writ'


def test_a_complete_array_still_parses_whole():
    assert _parse_json_array(json.dumps(WHOLE)) == WHOLE


def test_a_fenced_array_still_parses():
    assert _parse_json_array("```json\n" + json.dumps(WHOLE) + "\n```") == WHOLE


def test_a_truncated_array_keeps_the_completed_elements():
    """The fix. Two whole objects and a severed third must yield two."""
    raw = '[\n  ' + json.dumps(WHOLE[0]) + ',\n  ' + json.dumps(WHOLE[1]) + ',\n  {"name": "imag'
    got = _parse_json_array(raw)
    assert [g["name"] for g in got] == ["jia deng", "wei dong"]


def test_a_truncated_array_with_one_whole_element_keeps_it():
    raw = '[' + json.dumps(WHOLE[0]) + ', {"name": "cut'
    assert len(_parse_json_array(raw)) == 1


def test_a_severed_first_element_yields_nothing_rather_than_garbage():
    assert _parse_json_array('[{"name": "jia de') == []


def test_an_empty_array_is_still_empty():
    """A genuinely empty page must not start reporting phantom entities."""
    assert _parse_json_array("[]") == []


def test_prose_with_no_array_yields_nothing():
    assert _parse_json_array("I could not find any entities.") == []


def test_salvage_does_not_run_on_a_clean_parse():
    """Ordering matters: the whole-string parse wins, so a complete response
    is never re-derived element by element."""
    raw = json.dumps(WHOLE)
    assert _salvage_objects(raw) == WHOLE  # capable
    assert _parse_json_array(raw) == WHOLE  # but the fast path produced it


@pytest.mark.parametrize(
    "raw,closed",
    [
        (json.dumps(WHOLE), True),
        ("```json\n" + json.dumps(WHOLE) + "\n```", True),
        ('[{"name": "a"}, {"name": "b', False),
        ("[]", True),
        ("", False),
    ],
)
def test_closes_cleanly_detects_the_cap(raw, closed):
    """Drives the `entity_extraction.truncated` warning. Without it an
    operator sees a healthy count and never learns the tail of every page is
    being dropped."""
    assert _closes_cleanly(raw) is closed


def test_salvage_stops_at_the_first_break_and_does_not_resync():
    """Resyncing past a damaged element would invent entities from whatever
    text followed. Stopping is the honest answer."""
    raw = '[{"name": "a"}, {BROKEN}, {"name": "c"}]'
    assert [o["name"] for o in _salvage_objects(raw)] == ["a"]
