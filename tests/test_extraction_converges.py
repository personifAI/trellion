# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""The entity extractor converges (migration 103).

Before: a page was "not done" while it had no mention rows, so a page the
model found nothing on was re-extracted every ten seconds, forever, and
two ML containers ran the loop over the same table with nothing between
them. These pin the three outcomes of one page: found nothing (done),
model unreachable (retry, not done), already taken (no model call).
"""
from __future__ import annotations

import asyncio

import httpx
import pytest

from trellion.live.graph_extractor import EntityExtractor


class FakeStore:
    def __init__(self, claimable: bool = True) -> None:
        self.claimable = claimable
        self.calls: list[tuple] = []

    async def get_page(self, page_id):
        return {"page_id": page_id, "doc_id": "d0", "page_no": 1, "text": "The Krebs cycle."}

    async def claim_page_for_extraction(self, page_id, stale_after_minutes=15):
        self.calls.append(("claim", page_id))
        return self.claimable

    async def mark_page_extracted(self, page_id):
        self.calls.append(("mark", page_id))

    async def release_extraction_claim(self, page_id):
        self.calls.append(("release", page_id))

    async def upsert_entity(self, canonical_name, type, aliases=None):
        self.calls.append(("entity", canonical_name))
        return "e-" + canonical_name

    async def insert_mention(self, entity_id, page_id, span=None, quote=None):
        self.calls.append(("mention", entity_id, page_id))
        return "m1"


def _extractor(store, raw):
    ex = EntityExtractor(store=store, llm_url="http://gw", model="m")
    ex.llm_calls = 0

    async def fake_generate(prompt):
        ex.llm_calls += 1
        if isinstance(raw, Exception):
            raise raw
        return raw

    ex._llm_generate = fake_generate
    return ex


def test_found_nothing_is_done():
    store = FakeStore()
    ex = _extractor(store, "[]")
    ids = asyncio.run(ex.process_page("p1"))
    assert ids == []
    assert ("mark", "p1") in store.calls, store.calls
    assert ("release", "p1") not in store.calls


def test_found_something_is_done_and_recorded():
    store = FakeStore()
    ex = _extractor(store, '[{"name": "Krebs cycle", "type": "concept", "quote": "The Krebs cycle."}]')
    ids = asyncio.run(ex.process_page("p1"))
    assert len(ids) == 1
    kinds = [c[0] for c in store.calls]
    assert kinds.index("entity") < kinds.index("mention") < kinds.index("mark")


def test_model_unreachable_is_not_done():
    store = FakeStore()
    ex = _extractor(store, httpx.ConnectError("refused"))
    ids = asyncio.run(ex.process_page("p1"))
    assert ids == []
    assert ("release", "p1") in store.calls
    assert ("mark", "p1") not in store.calls


def test_http_error_is_not_done():
    store = FakeStore()
    req = httpx.Request("POST", "http://gw/v1/chat/completions")
    ex = _extractor(store, httpx.HTTPStatusError("503", request=req, response=httpx.Response(503, request=req)))
    asyncio.run(ex.process_page("p1"))
    assert ("release", "p1") in store.calls
    assert ("mark", "p1") not in store.calls


def test_a_page_another_extractor_holds_is_not_asked_again():
    store = FakeStore(claimable=False)
    ex = _extractor(store, "[]")
    ids = asyncio.run(ex.process_page("p1"))
    assert ids == []
    assert ex.llm_calls == 0
    assert store.calls == [("claim", "p1")]


@pytest.mark.parametrize("raw", ["[]", "[ ]", "nothing here"])
def test_extract_entities_distinguishes_empty_from_unasked(raw):
    ex = _extractor(FakeStore(), raw)
    assert asyncio.run(ex.extract_entities("p1", "some text")) == []
    ex2 = _extractor(FakeStore(), httpx.ConnectError("refused"))
    assert asyncio.run(ex2.extract_entities("p1", "some text")) is None
