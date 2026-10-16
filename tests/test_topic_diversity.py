# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""One topic may not take every evidence slot (live/topic_diversity.py).

2026-10-16: "best practices" returned five ``pai-trading / experience /
exp-<id>`` pages, the trading agents' per-idea notes that by then made up
most of the knowledge base; every other topic was in the corpus and never
reached the answer.
"""
from __future__ import annotations

import pytest

from trellion.live.pageindex_retriever import PageIndexRetriever
from trellion.live.topic_diversity import diversify, topic_family


def _page(pid: str, doc: str, score: float) -> dict:
    return {"page_id": pid, "doc_id": doc, "page_no": 1, "text": "", "score": score}


TITLES = {
    **{f"t{i}": f"pai-trading / experience / exp-{i}" for i in range(10)},
    "guide": "Engineering best practices",
    "runbook": "ops / runbooks / deploys",
}


def fam(p: dict) -> str:
    return topic_family(TITLES.get(p["doc_id"], ""), p["doc_id"])


def test_family_is_the_title_without_its_last_segment():
    assert topic_family("pai-trading / experience / exp-1", "d") == "pai-trading / experience"
    assert topic_family("Pai-Trading / Experience / exp-2", "d") == "pai-trading / experience"
    assert topic_family("Engineering best practices", "d9") == "doc:d9"
    assert topic_family("", "d9") == "doc:d9"


def test_a_dominant_family_leaves_room_for_other_topics():
    pages = [_page(f"p{i}", f"t{i}", 0.43 - i * 0.001) for i in range(8)]
    pages += [_page("pg", "guide", 0.40), _page("pr", "runbook", 0.39)]
    picked = diversify(pages, fam, top_k=5)
    ids = [p["page_id"] for p in picked]
    assert "pg" in ids and "pr" in ids, ids
    assert sum(fam(p) == "pai-trading / experience" for p in picked) == 3
    # relevance order is kept
    assert ids == [p["page_id"] for p in pages if p["page_id"] in ids]


def test_a_question_about_that_topic_still_gets_it():
    # The other topic is far below the best score: it does not displace.
    pages = [_page(f"p{i}", f"t{i}", 0.80 - i * 0.01) for i in range(8)]
    pages.append(_page("pg", "guide", 0.30))
    picked = diversify(pages, fam, top_k=5)
    assert [p["page_id"] for p in picked] == ["p0", "p1", "p2", "p3", "p4"]


def test_only_one_family_in_the_pool_fills_every_slot():
    pages = [_page(f"p{i}", f"t{i}", 0.5 - i * 0.01) for i in range(8)]
    assert [p["page_id"] for p in diversify(pages, fam, top_k=5)] == ["p0", "p1", "p2", "p3", "p4"]


def test_a_small_pool_is_returned_unchanged():
    pages = [_page("p0", "t0", 0.5), _page("pg", "guide", 0.1)]
    assert diversify(pages, fam, top_k=5) == pages
    assert diversify(pages, fam, top_k=0) == []


def _retriever(titles=None, fail=False):
    r = PageIndexRetriever.__new__(PageIndexRetriever)

    async def _titles(doc_ids):
        if fail:
            raise RuntimeError("db down")
        return {d: (titles or TITLES).get(d, "") for d in doc_ids}

    r._get_doc_titles_batch = _titles
    return r


@pytest.mark.asyncio
async def test_the_retriever_applies_the_cap():
    pages = [_page(f"p{i}", f"t{i}", 0.43 - i * 0.001) for i in range(8)]
    pages.append(_page("pg", "guide", 0.40))
    picked = await _retriever()._diverse_top(pages, 5)
    assert "pg" in [p["page_id"] for p in picked]


@pytest.mark.asyncio
async def test_unreadable_titles_fall_back_to_plain_truncation():
    pages = [_page(f"p{i}", f"t{i}", 0.43 - i * 0.001) for i in range(8)]
    pages.append(_page("pg", "guide", 0.40))
    picked = await _retriever(fail=True)._diverse_top(pages, 5)
    assert [p["page_id"] for p in picked] == ["p0", "p1", "p2", "p3", "p4"]
