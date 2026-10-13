# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""P1.5 live-path retrieval benchmark (gateway proposal 27).

Non-DB tests check fixture integrity (a broken cross-reference would
silently zero a score). The DB test seeds the corpus + hand-authored
graph into the migrated schema and runs BOTH production strategies —
it is simultaneously the harness's own test and an end-to-end proof
that graph-guided retrieval mechanically works: the relational golden
queries are lexically disjoint from their answer docs, so the graph
arm must beat the vector arm on them or something is broken.
"""

from __future__ import annotations

import os

import pytest

# The benchmark compares the add-on's graph arm against the open vector arm;
# the graph arm lives in the add-on package, so without it there is nothing
# to measure. The
# eval harness (trellion.eval.live_bench) stays here so both tiers share it.
pytest.importorskip("trellion_intel", reason="Intelligent tier not installed — benchmark needs the graph arm")

from trellion.eval.live_bench import (
    CORPUS_PATH,
    GOLDEN_PATH,
    FakeDeterministicEmbedder,
    format_report,
    load_fixture,
    run_live_bench,
)

DSN = os.environ.get("TRELLION_TEST_DSN")


def test_fixture_cross_references_are_valid():
    corpus, golden = load_fixture()
    uris = {d["source_uri"] for d in corpus["docs"]}
    names = {e["name"] for e in corpus["entities"]}
    for m in corpus["mentions"]:
        assert m["entity"] in names, m
        assert m["doc"] in uris, m
    for e in corpus["edges"]:
        assert e["src"] in names and e["dst"] in names, e
    for q in golden:
        assert q["type"] in ("direct", "relational"), q
        for uri in q["expected_docs"]:
            assert uri in uris, q


def test_relational_queries_are_lexically_disjoint_from_answers():
    """The design invariant the whole benchmark rests on: a vector hit on
    a relational query would be luck, because the query shares no
    meaningful token with its expected docs."""
    import re

    corpus, golden = load_fixture()
    text_by_uri = {
        d["source_uri"]: " ".join(d["pages"]).lower() for d in corpus["docs"]
    }
    for q in (g for g in golden if g["type"] == "relational"):
        q_tokens = set(re.findall(r"[a-zA-Z0-9_\-]+", q["query"].lower()))
        q_tokens -= {"what", "which", "how", "do", "we", "the", "on", "with",
                     "has", "to", "apply", "a", "of", "does", "why"}
        for uri in q["expected_docs"]:
            doc_tokens = set(
                re.findall(r"[a-zA-Z0-9_\-]+", text_by_uri[uri])
            )
            overlap = q_tokens & doc_tokens
            assert not overlap, (
                f"{q['id']}: query shares tokens {overlap} with {uri} — "
                "the relational arm comparison is no longer meaningful"
            )


async def test_fake_embedder_is_deterministic_and_normalized():
    e = FakeDeterministicEmbedder(dim=64)
    a = await e.embed_text("bearing fatigue spalling")
    b = await e.embed_text("bearing fatigue spalling")
    assert a == b
    assert abs(sum(x * x for x in a) - 1.0) < 1e-6


@pytest.mark.skipif(not DSN, reason="TRELLION_TEST_DSN not set")
async def test_graph_arm_beats_vector_arm_on_relational_queries():
    report = await run_live_bench(DSN, FakeDeterministicEmbedder(dim=1024))

    # Sanity: the harness ran every query through both arms.
    assert report["queries"] == len(load_fixture()[1])
    assert set(report["vector"]["per_type"]) == {"direct", "relational"}

    # Vanilla must stay honest on direct lookups (token overlap is real
    # signal for the fake embedder).
    assert report["vector"]["per_type"]["direct"]["recall"] >= 0.75, report

    # The gate itself: graph-guided retrieval finds what vector cannot.
    rel_graph = report["graph"]["per_type"]["relational"]["recall"]
    rel_vector = report["vector"]["per_type"]["relational"]["recall"]
    assert rel_graph > rel_vector, format_report(report)
    assert report["relational_delta"] > 0

    # And it must not RUIN direct lookups while doing so.
    assert report["graph"]["per_type"]["direct"]["recall"] >= 0.5, report


def test_fixture_paths_exist():
    assert CORPUS_PATH.exists() and GOLDEN_PATH.exists()
