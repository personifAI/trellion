# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""Tests for MetadataNarrower + NarrowerOrchestrator combine logic.

These don't hit Postgres — they exercise:
  - SQL-composition correctness (parameter binding, clause assembly)
  - The "no opinion" contract when filters are empty
  - Orchestrator combine modes (intersect, union)
  - Failure isolation (one narrower throws, others still combine)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pytest

from trellion.live.narrowers import (
    CombineMode,
    MetadataNarrower,
    NarrowerOrchestrator,
    NarrowerResult,
)


# ── Narrower stubs for orchestrator tests ────────────────────────────


@dataclass
class _StubNarrower:
    """Narrower that returns a fixed result (or raises)."""

    name: str
    result: Optional[NarrowerResult] = None
    raise_exc: bool = False

    async def narrow(self, query: str, ctx: dict) -> NarrowerResult:
        if self.raise_exc:
            raise RuntimeError(f"{self.name} blew up")
        return self.result if self.result is not None else NarrowerResult(name=self.name)


# ── MetadataNarrower SQL composition ─────────────────────────────────


class TestMetadataNarrowerQueryBuilder:
    def test_no_filters_returns_no_sql(self):
        sql, params = MetadataNarrower._build_query({})
        assert sql is None
        assert params == {}

    def test_only_empty_filters_returns_no_sql(self):
        # Filter dict has keys but values are falsy/empty
        sql, params = MetadataNarrower._build_query({
            "title_substr": "",
            "metadata_contains": {},
            "metadata_has_keys": [],
        })
        assert sql is None

    def test_title_substr_uses_ilike_wildcards(self):
        sql, params = MetadataNarrower._build_query({"title_substr": "spec"})
        assert "title ILIKE :title_substr" in sql
        assert params["title_substr"] == "%spec%"

    def test_source_uri_prefix_appends_wildcard(self):
        sql, params = MetadataNarrower._build_query({"source_uri_prefix": "/projects/"})
        assert "source_uri LIKE :source_uri_prefix" in sql
        assert params["source_uri_prefix"] == "/projects/%"

    def test_metadata_contains_serialized_to_json(self):
        sql, params = MetadataNarrower._build_query({
            "metadata_contains": {"project": "atlas", "tags": ["security"]},
        })
        assert "metadata @> CAST(:meta_contains AS jsonb)" in sql
        # Value is JSON-serialized so postgres can deserialize at bind time
        import json
        assert json.loads(params["meta_contains"]) == {
            "project": "atlas", "tags": ["security"],
        }

    def test_temporal_filters_use_iso_dates(self):
        sql, params = MetadataNarrower._build_query({
            "created_after": "2026-01-01T00:00:00Z",
            "created_before": "2026-04-01T00:00:00Z",
        })
        assert "created_at >= :created_after" in sql
        assert "created_at <= :created_before" in sql

    def test_multiple_filters_AND_composed(self):
        sql, params = MetadataNarrower._build_query({
            "title_substr": "report",
            "source_uri_prefix": "/projects/",
        })
        assert " AND " in sql
        assert sql.startswith("SELECT doc_id FROM docs WHERE")


# ── MetadataNarrower.narrow() contract ───────────────────────────────


class TestMetadataNarrowerNarrowContract:
    @pytest.mark.asyncio
    async def test_no_ctx_returns_no_opinion(self):
        narrower = MetadataNarrower(store=object())
        result = await narrower.narrow("anything", {})
        assert result.name == "metadata"
        assert result.doc_ids is None
        assert not result.has_opinion

    @pytest.mark.asyncio
    async def test_empty_filters_dict_returns_no_opinion(self):
        narrower = MetadataNarrower(store=object())
        result = await narrower.narrow("anything", {"metadata_filters": {}})
        assert not result.has_opinion

    @pytest.mark.asyncio
    async def test_store_without_engine_returns_no_opinion(self):
        # Defensive: store missing the _engine attr should fail-safe to None
        narrower = MetadataNarrower(store=object())
        result = await narrower.narrow(
            "q",
            {"metadata_filters": {"title_substr": "x"}},
        )
        assert not result.has_opinion


# ── NarrowerOrchestrator combine logic ───────────────────────────────


class TestOrchestratorIntersect:
    @pytest.mark.asyncio
    async def test_intersect_keeps_only_common_doc_ids(self):
        n1 = _StubNarrower(name="a", result=NarrowerResult(name="a", doc_ids={"d1", "d2", "d3"}))
        n2 = _StubNarrower(name="b", result=NarrowerResult(name="b", doc_ids={"d2", "d3", "d4"}))
        orch = NarrowerOrchestrator([n1, n2])
        out = await orch.run("q", mode=CombineMode.INTERSECT)
        assert out.candidate_doc_ids == {"d2", "d3"}
        assert set(out.contributors) == {"a", "b"}

    @pytest.mark.asyncio
    async def test_intersect_with_no_narrowers_returns_none(self):
        out = await NarrowerOrchestrator([]).run("q")
        assert out.candidate_doc_ids is None
        assert out.candidate_page_ids is None

    @pytest.mark.asyncio
    async def test_no_opinion_narrower_excluded_from_intersect(self):
        # A "no opinion" narrower must NOT collapse the set to empty
        n1 = _StubNarrower(name="a", result=NarrowerResult(name="a", doc_ids={"d1", "d2"}))
        n2 = _StubNarrower(name="b", result=NarrowerResult(name="b"))  # no opinion
        orch = NarrowerOrchestrator([n1, n2])
        out = await orch.run("q", mode=CombineMode.INTERSECT)
        assert out.candidate_doc_ids == {"d1", "d2"}
        # b is excluded from contributors because it had no opinion
        assert out.contributors == ["a"]


class TestOrchestratorUnion:
    @pytest.mark.asyncio
    async def test_union_combines_doc_ids(self):
        n1 = _StubNarrower(name="a", result=NarrowerResult(name="a", doc_ids={"d1"}))
        n2 = _StubNarrower(name="b", result=NarrowerResult(name="b", doc_ids={"d2"}))
        orch = NarrowerOrchestrator([n1, n2])
        out = await orch.run("q", mode=CombineMode.UNION)
        assert out.candidate_doc_ids == {"d1", "d2"}


class TestOrchestratorFailureIsolation:
    @pytest.mark.asyncio
    async def test_one_narrower_raising_does_not_break_others(self):
        n1 = _StubNarrower(name="a", result=NarrowerResult(name="a", doc_ids={"d1"}))
        n2 = _StubNarrower(name="boom", raise_exc=True)
        orch = NarrowerOrchestrator([n1, n2])
        out = await orch.run("q", mode=CombineMode.INTERSECT)
        # n2 dropped; result is whatever n1 said
        assert out.candidate_doc_ids == {"d1"}
        assert out.contributors == ["a"]


class TestOrchestratorPageDocIndependence:
    @pytest.mark.asyncio
    async def test_doc_and_page_combined_separately(self):
        # Doc-level narrower + page-level narrower don't interfere
        n1 = _StubNarrower(name="meta", result=NarrowerResult(name="meta", doc_ids={"d1", "d2"}))
        n2 = _StubNarrower(name="graph", result=NarrowerResult(name="graph", page_ids={"p1", "p2"}))
        orch = NarrowerOrchestrator([n1, n2])
        out = await orch.run("q", mode=CombineMode.INTERSECT)
        assert out.candidate_doc_ids == {"d1", "d2"}
        assert out.candidate_page_ids == {"p1", "p2"}
