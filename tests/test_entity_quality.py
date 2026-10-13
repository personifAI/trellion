# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""Junk entities (math symbols, bare numbers, stopwords) never become entities.

Example: a saved walk ``model-m -> a -> d -> x`` and cluster labels
"a, theta, example topic". The extractor accepted any non-empty
name, so ``a``, ``x_t``, ``theta`` and ``0.003`` became entities, collected
co-occurrence edges, and surfaced in answers.

Part 1 pins the validator. Part 2 pins that extraction, the offline LLM
extractor and the edge builders go through it, with no dangling edge. Part 3
(TRELLION_TEST_DSN) pins the cleanup against a real schema: the dry-run finds
exactly the junk and touches nothing; apply stages (restorably) then deletes.
"""
from __future__ import annotations

import asyncio
import os
import uuid

import pytest
from sqlalchemy import text

from trellion.build.extractors import LLMExtractor
from trellion.common.config import ExtractionConfig
from trellion.common.schema import Chunk
from trellion.live import entity_quality as eq
from trellion.live.entity_quality import check_entity, is_junk_entity
from trellion.live.graph_extractor import EXTRACTION_PROMPT, EntityExtractor

# ---------------------------------------------------------------- validator

REAL = [
    ("JEPA", None), ("H-JEPA", None), ("AI", None), ("GPT-4", None),
    ("OpenAI", None), ("Yann LeCun", "person"), ("x86", None), ("C++", None),
    ("R", None), ("ML", "concept"),          # 2-letter acronym on the allow-list
    ("EU", "organization"),                   # 2-letter caps acronym, org type
    ("learning rate", "parameter"), ("a100", "component"), ("t5", "component"),
    ("π0", "concept"), ("alpha release", "concept"), ("go_to_coordinate", "process"),
    ("alpha", "concept"),                     # spelled Greek is fine as a concept
    # short names that must stay valid, in the lowercase the table stores:
    ("xo", "person"), ("zu", "person"), ("qi", "person"),      # short family names
    ("kx", "organization"), ("vz", "organization"),            # short org names
    ("ny", "location"),                                        # places
    ("3000", "component"), ("4000", "material"), ("1600", "component"),  # model numbers
    ("i7", "component"), ("m2", "component"), ("t4", "component"),
    ("o3", "concept"), ("h2", "concept"),                      # letter+digit codes
    ("go", "concept"), ("go", "process"),                      # the language, any type
]
JUNK = [
    "a", "x", "d", "x_t", "θ", "\\alpha", "12", "--", "it", "the", "0.003",
    "$1,200", "33%", "x'", "s[3]", "f(x)", "z+1", "10^8", "α", "s_{t+1}",
    "r",  # a lowercase "r" is a radius, not the language
]


@pytest.mark.parametrize("name,etype", REAL)
def test_validator_accepts_real_entities(name, etype):
    assert check_entity(name, etype) is None, name


@pytest.mark.parametrize("name", JUNK)
def test_validator_rejects_junk(name):
    assert is_junk_entity(name), name


@pytest.mark.parametrize("name,etype,reason", [
    # what the narrowing must NOT let through
    ("he", "person", "stopword"), ("it", "organization", "stopword"), ("she", "person", "stopword"),
    ("x", "person", "too_short"), ("a", "location", "stopword"),       # single letters
    ("12", "component", "numeric"), ("4.5", "component", "numeric"),   # not a model number
    ("123456", "component", "numeric"), ("3090", "concept", "numeric"),  # too long / wrong type
    ("s2", "parameter", "math_symbol"), ("x1", "variable", "math_symbol"),  # math types unchanged
    ("x1", None, "too_short"),                                         # untyped stays strict
    ("ad", "concept", "too_short"), ("hp", None, "too_short"),         # 2 letters, not a name type
])
def test_the_false_positive_fixes_are_narrow(name, etype, reason):
    assert check_entity(name, etype) == reason, (name, etype)


def test_validator_is_type_aware_for_ambiguous_names():
    assert check_entity("beta", "variable") == "greek_symbol"
    assert check_entity("beta", "concept") is None
    assert check_entity("s2", "parameter") == "math_symbol"
    # a letter+digit code is a model outside the math types ("i7", "T4"): the docstring's rule
    assert check_entity("s2", "component") is None
    assert check_entity("AB", "component") == "too_short"
    assert check_entity("AB", "organization") is None


def test_validator_is_data_driven(tmp_path, monkeypatch):
    cfg = tmp_path / "q.json"
    cfg.write_text('{"allow": ["qx"], "stopwords": ["whereas"], "reject_patterns": ["^tmp_\\\\d+$"]}')
    monkeypatch.setenv("TRELLION_ENTITY_QUALITY_CONFIG", str(cfg))
    eq.reset_config_cache()
    try:
        assert check_entity("qx") is None
        assert check_entity("whereas") == "stopword"
        assert check_entity("tmp_42") == "configured_pattern"
        monkeypatch.setenv("TRELLION_ENTITY_QUALITY", "off")
        eq.reset_config_cache()
        assert check_entity("a") is None  # kill switch
    finally:
        monkeypatch.delenv("TRELLION_ENTITY_QUALITY_CONFIG")
        monkeypatch.delenv("TRELLION_ENTITY_QUALITY")
        eq.reset_config_cache()


def test_unreadable_config_keeps_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("TRELLION_ENTITY_QUALITY_CONFIG", str(tmp_path / "missing.json"))
    eq.reset_config_cache()
    try:
        assert check_entity("a") == "stopword"
    finally:
        monkeypatch.delenv("TRELLION_ENTITY_QUALITY_CONFIG")
        eq.reset_config_cache()


# --------------------------------------------------------------- extraction

RAW = (
    '[{"name": "JEPA", "type": "concept", "quote": "JEPA"},'
    ' {"name": "a", "type": "variable", "quote": "a"},'
    ' {"name": "x_t", "type": "variable", "quote": "x_t"},'
    ' {"name": "theta", "type": "parameter", "quote": "theta"},'
    ' {"name": "0.003", "type": "parameter", "quote": "0.003"},'
    ' {"name": "world model", "type": "concept", "aliases": ["WM"], "quote": "world model"}]'
)


class FakeStore:
    def __init__(self) -> None:
        self.entities: list[str] = []
        self.mentions: list[str] = []

    async def get_page(self, page_id):
        return {"page_id": page_id, "doc_id": "d0", "page_no": 1, "text": "JEPA and a world model."}

    async def claim_page_for_extraction(self, page_id, stale_after_minutes=15):
        return True

    async def mark_page_extracted(self, page_id):
        pass

    async def release_extraction_claim(self, page_id):
        pass

    async def upsert_entity(self, canonical_name, type, aliases=None):
        self.entities.append(canonical_name)
        return "e-" + canonical_name

    async def insert_mention(self, entity_id, page_id, span=None, quote=None):
        self.mentions.append(entity_id)
        return "m"


def test_live_extraction_creates_no_junk_entities():
    store = FakeStore()
    ex = EntityExtractor(store=store, llm_url="http://gw", model="m")

    async def gen(prompt):
        return RAW

    ex._llm_generate = gen
    ids = asyncio.run(ex.process_page("p1"))
    assert store.entities == ["jepa", "world model"], store.entities
    assert store.mentions == ids == ["e-jepa", "e-world model"]


def test_extraction_prompt_forbids_symbols():
    assert "mathematical variables" in EXTRACTION_PROMPT


class FakeLLM:
    def __init__(self, response: str) -> None:
        self._response = response

    async def complete(self, prompt, *, system=None, response_format=None, temperature=None):
        return self._response


def test_offline_llm_extraction_drops_junk_and_their_relations():
    resp = (
        '{"entities": ['
        ' {"name": "JEPA", "entity_type": "CONCEPT", "description": ""},'
        ' {"name": "world model", "entity_type": "CONCEPT", "description": ""},'
        ' {"name": "x", "entity_type": "VARIABLE", "description": ""},'
        ' {"name": "12", "entity_type": "CONCEPT", "description": ""}],'
        ' "relationships": ['
        ' {"source": "JEPA", "target": "world model", "relation_type": "RELATED_TO"},'
        ' {"source": "JEPA", "target": "x", "relation_type": "RELATED_TO"},'
        ' {"source": "12", "target": "world model", "relation_type": "RELATED_TO"}]}'
    )
    ex = LLMExtractor(ExtractionConfig(), FakeLLM(resp))
    chunk = Chunk(chunk_id="c1", doc_id="d", text="t", token_count=1, offset=0)
    ents, rels = asyncio.run(ex.extract(chunk, None))
    assert {e.name for e in ents} == {"JEPA", "WORLD MODEL"}
    assert [(r.source, r.target) for r in rels] == [("JEPA", "WORLD MODEL")]
    assert all(r.source in {e.name for e in ents} and r.target in {e.name for e in ents} for r in rels)


# --------------------------------------------------------- DB-backed (DSN)

DSN = os.environ.get("TRELLION_TEST_DSN")
needs_db = pytest.mark.skipif(not DSN, reason="TRELLION_TEST_DSN not set")

_STAGING_DDL = [
    """CREATE TABLE IF NOT EXISTS graph_prune_staging (
        staging_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        entity_id UUID NOT NULL, canonical_name TEXT NOT NULL, entity_type TEXT NOT NULL,
        reason TEXT NOT NULL, payload JSONB NOT NULL,
        staged_at TIMESTAMPTZ NOT NULL DEFAULT now(), restored_at TIMESTAMPTZ, run_id TEXT)""",
]


class Fixture:
    pass


@pytest.fixture
async def graph():
    from trellion.live.pageindex_store import PageIndexStore

    store = PageIndexStore(dsn=DSN)
    tag = uuid.uuid4().hex[:8]
    async with store._engine.connect() as conn:
        for ddl in _STAGING_DDL:
            await conn.execute(text(ddl))
        await conn.commit()
    doc_id, _ = await store.insert_doc(f"entq://{tag}", "t", tag)
    page_id = await store.insert_page(doc_id, 1, "JEPA, world models, a, x_t and 0.003.")
    f = Fixture()
    f.store, f.page_id, f.tag = store, page_id, tag
    f.real, f.junk = {}, {}
    for name, etype in [(f"jepa-{tag}", "concept"), (f"world model {tag}", "concept")]:
        f.real[name] = await store.upsert_entity(name, etype)
    for name, etype in [("a", "variable"), ("x_t", "variable"), ("0.003", "parameter"),
                        ("theta", "parameter")]:
        f.junk[name] = await store.upsert_entity(name, etype)
    for eid in [*f.real.values(), *f.junk.values()]:
        await store.insert_mention(eid, page_id, quote="q")
    # pre-existing junk edges + a cluster membership, as in the live graph
    real = list(f.real.values())
    await store.insert_edge(real[0], "RELATED_TO", f.junk["a"], {})
    await store.insert_edge(f.junk["a"], "RELATED_TO", f.junk["x_t"], {})
    await store.insert_edge(real[0], "RELATED_TO", real[1], {})
    cid = str(uuid.uuid4())
    await store.upsert_cluster(cid, f"c-{tag}", 0, None, None, None, 3, 0)
    async with store._engine.connect() as conn:
        for eid in (real[0], f.junk["a"], f.junk["theta"]):
            await conn.execute(
                text("INSERT INTO entity_cluster_members (entity_id, cluster_id) "
                     "VALUES (CAST(:e AS uuid), CAST(:c AS uuid))"),
                {"e": eid, "c": cid},
            )
        await conn.commit()
    f.cluster_id = cid
    yield f
    ids = [*f.real.values(), *f.junk.values()]
    async with store._engine.connect() as conn:
        await conn.execute(text("DELETE FROM entities WHERE entity_id = ANY(CAST(:i AS uuid[]))"),
                           {"i": ids})
        await conn.execute(text("DELETE FROM graph_prune_staging WHERE entity_id = ANY(CAST(:i AS uuid[]))"),
                           {"i": ids})
        await conn.execute(text("DELETE FROM entity_clusters WHERE cluster_id = CAST(:c AS uuid)"),
                           {"c": cid})
        await conn.execute(text("DELETE FROM docs WHERE source_uri LIKE :u"), {"u": f"entq://{f.tag}%"})
        await conn.commit()
    await store._engine.dispose()


async def _count(store, sql, **p):
    async with store._engine.connect() as conn:
        return (await conn.execute(text(sql), p)).scalar()


async def _pairs(store, ids):
    """Unordered RELATED_TO pairs among ``ids``."""
    from sqlalchemy import text
    async with store._engine.connect() as conn:
        res = await conn.execute(
            text(
                "SELECT src_entity_id::text, dst_entity_id::text FROM edges "
                "WHERE rel_type = 'RELATED_TO' AND src_entity_id = ANY(CAST(:i AS uuid[])) "
                "AND dst_entity_id = ANY(CAST(:i AS uuid[]))"
            ),
            {"i": [str(i) for i in ids]},
        )
        return {frozenset((a, b)) for a, b in res.fetchall()}


@needs_db
async def test_graph_builder_gives_junk_no_edges(graph):
    from trellion.live.graph_builder import GraphBuilder

    b = GraphBuilder(graph.store)
    ids = await b._get_page_entities(graph.page_id)
    assert set(ids) == set(graph.real.values())
    # a second page: two real entities, a new real one, two junk ones
    doc_id, _ = await graph.store.insert_doc(f"entq://{graph.tag}-2", "t", graph.tag + "b")
    page2 = await graph.store.insert_page(doc_id, 1, "more")
    third = await graph.store.upsert_entity(f"third {graph.tag}", "concept")
    graph.real["third"] = third
    for eid in [*list(graph.real.values())[:2], third, graph.junk["theta"], graph.junk["x_t"]]:
        await graph.store.insert_mention(eid, page2, quote="q")
    await b.build_edges_for_page(page2)
    # Every real pair on page 2 is linked, in whichever direction the builder
    # orders a pair (the fixture's (r0, r1) edge may or may not match it, so
    # the NUMBER of new rows is not stable; the set of linked pairs is).
    reals = [*list(graph.real.values())[:2], third]
    linked = await _pairs(graph.store, reals)
    want = {frozenset((str(x), str(y))) for i, x in enumerate(reals) for y in reals[i + 1:]}
    assert want <= linked, f"missing real-real edges: {want - linked}"
    junk_ids = list(graph.junk.values())
    # no NEW edge touches junk (the fixture's two pre-existing ones are the only)
    assert await _count(
        graph.store,
        "SELECT count(*) FROM edges WHERE (src_entity_id = ANY(CAST(:i AS uuid[])) "
        "OR dst_entity_id = ANY(CAST(:i AS uuid[]))) AND rel_type = 'RELATED_TO'", i=junk_ids,
    ) == 2


@needs_db
async def test_cleanup_dry_run_finds_exactly_the_junk_and_changes_nothing(graph):
    from trellion.live import junk_entity_cleanup as jec

    before = {
        "entities": await _count(graph.store, "SELECT count(*) FROM entities"),
        "edges": await _count(graph.store, "SELECT count(*) FROM edges"),
        "mentions": await _count(graph.store, "SELECT count(*) FROM mentions"),
        "members": await _count(graph.store, "SELECT count(*) FROM entity_cluster_members"),
        "staged": await _count(graph.store, "SELECT count(*) FROM graph_prune_staging"),
    }
    mine = {*graph.real.values(), *graph.junk.values()}
    rep = await jec.run(graph.store._engine, apply=False, restrict_to=mine)
    assert {c.entity_id for c in rep.candidates} == set(graph.junk.values())
    by_name = {c.name: c for c in rep.candidates}
    assert by_name["a"].edges == 2 and by_name["a"].clusters == 1 and by_name["a"].mentions == 1
    assert by_name["theta"].edges == 0 and by_name["theta"].clusters == 1
    assert "a" in rep.render() and rep.dry_run
    after = {
        "entities": await _count(graph.store, "SELECT count(*) FROM entities"),
        "edges": await _count(graph.store, "SELECT count(*) FROM edges"),
        "mentions": await _count(graph.store, "SELECT count(*) FROM mentions"),
        "members": await _count(graph.store, "SELECT count(*) FROM entity_cluster_members"),
        "staged": await _count(graph.store, "SELECT count(*) FROM graph_prune_staging"),
    }
    assert after == before


@needs_db
async def test_cleanup_apply_stages_restorably_then_deletes_only_junk(graph):
    from trellion.live import junk_entity_cleanup as jec

    mine = {*graph.real.values(), *graph.junk.values()}
    rep = await jec.run(graph.store._engine, apply=True, restrict_to=mine)
    assert rep.staged == 4
    ids = list(graph.junk.values())
    assert await _count(graph.store, "SELECT count(*) FROM entities WHERE entity_id = ANY(CAST(:i AS uuid[]))", i=ids) == 0
    real = list(graph.real.values())
    assert await _count(graph.store, "SELECT count(*) FROM entities WHERE entity_id = ANY(CAST(:i AS uuid[]))", i=real) == 2
    # the real<->real edge survives; every edge that touched junk is gone, none dangling
    assert await _count(graph.store, "SELECT count(*) FROM edges WHERE src_entity_id = CAST(:a AS uuid) AND dst_entity_id = CAST(:b AS uuid)", a=real[0], b=real[1]) == 1
    assert await _count(graph.store, "SELECT count(*) FROM edges WHERE src_entity_id = ANY(CAST(:i AS uuid[])) OR dst_entity_id = ANY(CAST(:i AS uuid[]))", i=ids) == 0
    assert await _count(graph.store, "SELECT count(*) FROM entity_cluster_members WHERE entity_id = ANY(CAST(:i AS uuid[]))", i=ids) == 0
    # staged with the restore payload shape + memberships
    async with graph.store._engine.connect() as conn:
        row = (await conn.execute(
            text("SELECT reason, payload FROM graph_prune_staging WHERE canonical_name = 'a' "
                 "AND entity_id = CAST(:e AS uuid)"), {"e": graph.junk["a"]})).one()
    assert row[0] == jec.STAGING_REASON
    payload = row[1]
    assert payload["entity"]["canonical_name"] == "a"
    assert len(payload["mentions"]) == 1
    # A junk<->junk edge is snapshotted with whichever endpoint goes first
    # (the same behaviour as graph_prune); across the run every edge is staged.
    async with graph.store._engine.connect() as conn:
        payloads = [r[0] for r in (await conn.execute(
            text("SELECT payload FROM graph_prune_staging WHERE entity_id = ANY(CAST(:i AS uuid[]))"),
            {"i": ids})).all()]
    staged_edges = {e["edge_id"] for p in payloads for e in p["edges"]}
    assert len(staged_edges) == 2
    assert [m["cluster_id"] for m in payload["cluster_members"]] == [graph.cluster_id]


# --------------------------------------------- where the staging table lives

def test_staging_table_is_unqualified_by_default_and_qualified_on_request():
    from trellion.live import junk_entity_cleanup as jec

    assert jec.staging_table(None) == "graph_prune_staging"
    assert jec.staging_table("mesh") == '"mesh".graph_prune_staging'


@pytest.mark.parametrize("bad", ["", "Mesh", "mesh; DROP TABLE entities", 'me"sh', "mesh.x", "1mesh", "a" * 64, 7])
def test_staging_schema_is_a_plain_identifier(bad):
    from trellion.live import junk_entity_cleanup as jec

    with pytest.raises(ValueError):
        jec.staging_table(bad)


async def test_a_bad_staging_schema_is_refused_before_anything_is_read():
    from trellion.live import junk_entity_cleanup as jec

    class Boom:
        def connect(self):  # pragma: no cover - must not be reached
            raise AssertionError("touched the database")

    with pytest.raises(ValueError):
        await jec.run(Boom(), apply=True, staging_schema="x;y")
