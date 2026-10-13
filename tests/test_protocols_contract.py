# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""trellion keeps the contract it publishes in ``trellion_protocols``.

Add-ons import only ``trellion_protocols``. Nothing checks at
import time that trellion still matches it, so this file does:

* SCHEMA_REV is the head of the Alembic chain — adding a migration without
  bumping it fails here (and an add-on pinned to the old rev then refuses a
  database migrated past it, loudly, instead of reading a schema it was not
  tested against);
* TABLES is every column of every listed table: equal to information_schema
  of a migrated database, and a superset of the ORM models;
* every Protocol member exists on trellion's concrete class with the same
  parameters, and an instance passes the runtime ``isinstance`` check;
* the dataclasses trellion re-exports are the protocol package's own class
  objects;
* the package stays a contract: standard library only, no dependencies.
"""

from __future__ import annotations

import ast
import inspect
import os
import re
from pathlib import Path

import pytest

import trellion_protocols as tp
from trellion_protocols import host, schema, stores
from trellion_protocols import types as ptypes

ROOT = Path(__file__).resolve().parents[1]
VERSIONS = ROOT / "src" / "trellion" / "db" / "migrations" / "versions"

DSN = os.environ.get("TRELLION_TEST_DSN")
needs_db = pytest.mark.skipif(not DSN, reason="TRELLION_TEST_DSN not set")


# ---- schema revision ------------------------------------------------------


def _chain() -> dict[str, str | None]:
    """revision -> down_revision for every migration file, by parsing it."""
    out: dict[str, str | None] = {}
    files = sorted(p for p in VERSIONS.glob("*.py") if p.name != "__init__.py")
    assert files, f"no migrations found under {VERSIONS}"
    for path in files:
        found: dict[str, object] = {}
        for node in ast.parse(path.read_text()).body:
            target = None
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                target, value = node.target.id, node.value
            elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                target, value = node.targets[0].id, node.value
            if target in ("revision", "down_revision") and value is not None:
                found[target] = ast.literal_eval(value)
        assert "revision" in found and "down_revision" in found, f"{path.name}: revision ids not found"
        out[found["revision"]] = found["down_revision"]  # type: ignore[index]
    return out


def test_chain_parse_found_every_migration():
    chain = _chain()
    n_files = len([p for p in VERSIONS.glob("*.py") if p.name != "__init__.py"])
    assert len(chain) == n_files >= 6


def test_schema_rev_is_the_single_migration_head():
    chain = _chain()
    parents = {down for down in chain.values() if down is not None}
    heads = sorted(set(chain) - parents)
    assert len(heads) == 1, f"the chain has {len(heads)} heads: {heads}"
    assert schema.SCHEMA_REV == heads[0], (
        f"a migration was added ({heads[0]}) but trellion_protocols.schema.SCHEMA_REV "
        f"is still {schema.SCHEMA_REV!r}: release a trellion-protocols with it bumped (personifAI/"
        "trellion-protocols, schema.py), update TABLES if the migration changed columns, and move this repo's pin. "
        "An add-on refuses a database whose revision it was not built against."
    )


def test_version_table_is_the_one_alembic_writes():
    env = (VERSIONS.parent / "env.py").read_text()
    assert re.search(rf'^VERSION_TABLE\s*=\s*"{schema.VERSION_TABLE}"', env, re.M)
    assert f'os.environ.get("{schema.SCHEMA_ENV}", "{schema.DEFAULT_SCHEMA}")' in env


# ---- tables and columns ---------------------------------------------------


def test_orm_models_map_a_subset_of_tables():
    """The models may leave a column unmapped (embedding_model/embedding_dim
    exist only in SQL), never invent one the migrations do not create. The
    migrated-database test below is the equality check."""
    from trellion.db import models

    by_name = {t.name: t for t in models.Base.metadata.tables.values()}
    for table, cols in schema.TABLES.items():
        assert table in by_name, f"{table}: listed in trellion_protocols but not a trellion model"
        mapped = {c.name for c in by_name[table].columns}
        assert mapped <= set(cols), f"{table}: ORM columns not in the contract: {mapped - set(cols)}"
    assert len(schema.TABLES) >= 13


@needs_db
async def test_tables_equal_a_migrated_database():
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    from trellion.db.engine import TRELLION_SCHEMA, _async_dsn

    eng = create_async_engine(_async_dsn(DSN))
    try:
        async with eng.connect() as conn:
            rev = (await conn.execute(text(
                f'SELECT version_num FROM "{TRELLION_SCHEMA}".{schema.VERSION_TABLE}'
            ))).scalars().all()
            assert rev == [schema.SCHEMA_REV]
            for table, cols in schema.TABLES.items():
                got = set((await conn.execute(text(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_schema = :s AND table_name = :t"
                ), {"s": TRELLION_SCHEMA, "t": table})).scalars().all())
                assert got == set(cols), table
    finally:
        await eng.dispose()


# ---- the concrete classes satisfy the protocols ---------------------------


def _members(proto: type) -> dict[str, object]:
    """Members a Protocol declares itself or inherits from another Protocol."""
    out: dict[str, object] = {}
    for klass in reversed(proto.__mro__):
        if klass is object or not getattr(klass, "_is_protocol", False):
            continue
        for name, value in vars(klass).items():
            if name.startswith("_") or name in ("_is_protocol",):
                continue
            out[name] = value
        for name in getattr(klass, "__annotations__", {}):
            if not name.startswith("_"):
                out.setdefault(name, None)
    return out


def _params(fn) -> list[tuple[str, object, object]]:
    sig = inspect.signature(fn)
    return [(p.name, p.kind, p.default) for p in sig.parameters.values()]


def _assert_satisfies(concrete: type, proto: type, instance: object = None) -> int:
    members = _members(proto)
    assert members, f"{proto.__name__}: no members found"
    for name, decl in members.items():
        if decl is None:  # an annotated attribute: may live on the instance
            holder = instance if instance is not None else concrete
            assert hasattr(holder, name), f"{concrete.__name__} lacks {proto.__name__}.{name}"
            continue
        assert hasattr(concrete, name), f"{concrete.__name__} lacks {proto.__name__}.{name}"
        impl = inspect.getattr_static(concrete, name)
        if isinstance(decl, property):
            assert isinstance(impl, property), f"{concrete.__name__}.{name} must be a property"
        elif inspect.isfunction(decl):
            assert inspect.isfunction(impl), f"{concrete.__name__}.{name} must be a method"
            assert inspect.iscoroutinefunction(impl) == inspect.iscoroutinefunction(decl), (
                f"{concrete.__name__}.{name}: async-ness differs from {proto.__name__}"
            )
            assert _params(impl) == _params(decl), (
                f"{concrete.__name__}.{name}{inspect.signature(impl)} != "
                f"{proto.__name__}.{name}{inspect.signature(decl)}"
            )
    return len(members)


def test_the_checker_rejects_a_drifted_signature():
    """The satisfaction check is only worth something if it can fail."""

    class Drifted:
        async def recall_candidates(self, user_id, query_embedding, *, kind="walk", top_k=10, persona_id=None):
            return []

        async def touch_recalled(self, walk_ids, *, user_id=None):
            return 0

        def mark_stale(self, walk_ids, *, user_id=None):  # not async
            return 0

    with pytest.raises(AssertionError, match="recall_candidates"):
        _assert_satisfies(Drifted, stores.WalkStore)
    Drifted.recall_candidates = stores.WalkStore.recall_candidates  # type: ignore[assignment]
    with pytest.raises(AssertionError, match="async-ness"):
        _assert_satisfies(Drifted, stores.WalkStore)


def test_pageindex_store_satisfies_graph_store():
    from trellion.live.pageindex_store import PageIndexStore

    assert _assert_satisfies(PageIndexStore, stores.GraphStore) == 24
    store = PageIndexStore(async_engine=object(), embedding_dim=4)
    assert isinstance(store, stores.GraphStore)
    assert isinstance(store, stores.GraphReader)
    assert store.ROLES_WHERE is schema.ROLES_WHERE is PageIndexStore._ROLES_WHERE
    assert store.vec([1, 2, 3, 4]) == store._vec([1, 2, 3, 4])
    eng = object()
    assert PageIndexStore(async_engine=eng, embedding_dim=4).engine is eng
    assert isinstance(store.walk_store, stores.WalkStore)


def test_walk_store_satisfies_its_protocol():
    from trellion.live.recall.walk_store import WalkStore

    assert _assert_satisfies(WalkStore, stores.WalkStore) == 3
    assert isinstance(WalkStore(object()), stores.WalkStore)


def test_traversal_archiver_shim_satisfies_its_protocol():
    from trellion.live.video.traversal_archiver import TraversalArchiver

    assert _assert_satisfies(TraversalArchiver, stores.TraversalArchiver) == 1


def test_reexported_dataclasses_are_the_protocol_classes():
    from trellion.live import evidence, graph_types
    from trellion.live.video import types as vtypes

    assert graph_types.Hypothesis is ptypes.Hypothesis is tp.Hypothesis
    assert graph_types.GraphResult is ptypes.GraphResult
    assert evidence.Evidence is ptypes.Evidence
    assert vtypes.TraversalStep is ptypes.TraversalStep
    assert vtypes.TraversalSnapshot is ptypes.TraversalSnapshot
    # evidence_from_chunk stays trellion's and still builds the shared class.
    ev = evidence.evidence_from_chunk({"text": "b", "page_id": "p"}, {"doc_id": "d", "page_id": "p", "page_no": 1, "text": "abc"})
    assert isinstance(ev, ptypes.Evidence) and ev.span == (1, 2)


# ---- host services: the same singletons trellion uses -----------------------


def test_host_services_satisfy_the_protocol():
    from trellion.live.host_services import HOST, TrellionHostServices

    assert _assert_satisfies(TrellionHostServices, host.HostServices) == 14
    assert isinstance(HOST, host.HostServices)


def test_host_services_hand_out_trellions_own_objects():
    from trellion import json_fast
    from trellion.live import _tls, embed_throttle, entity_quality, gateway_auth, llm_throttle
    from trellion.live.host_services import HOST

    assert HOST.embed_throttle() is embed_throttle.get_throttle()
    assert HOST.llm_throttle() is llm_throttle.get_llm_throttle()
    assert HOST.embed_unavailable is embed_throttle.EmbedUnavailable
    assert HOST.llm_unavailable is llm_throttle.LLMUnavailable
    assert HOST.llm_breaker_open is llm_throttle.LLMBreakerOpen
    assert HOST.json is json_fast
    assert isinstance(HOST.gateway_auth(), gateway_auth.GatewayHmacAuth)
    assert isinstance(HOST.drop_tally(), entity_quality.DropTally)
    assert HOST.client_cert() == _tls.client_cert()
    assert HOST.check_entity("x") == entity_quality.check_entity("x")
    assert HOST.classify_embed_failure(ValueError()) == embed_throttle.classify(ValueError())


def test_throttles_tally_and_codec_satisfy_their_protocols():
    from trellion import json_fast
    from trellion.live.embed_throttle import EmbedThrottle
    from trellion.live.entity_quality import DropTally
    from trellion.live.llm_throttle import LLMThrottle

    assert _assert_satisfies(EmbedThrottle, host.EmbedThrottle, instance=EmbedThrottle()) == 6
    assert _assert_satisfies(LLMThrottle, host.LLMThrottle) == 3
    assert _assert_satisfies(DropTally, host.DropTally) == 3
    assert isinstance(EmbedThrottle(), host.EmbedThrottle)
    assert isinstance(LLMThrottle(), host.LLMThrottle)
    assert isinstance(json_fast, host.JsonCodec)


# ---- the dependency is pinned, not floating ----------------------------------


def test_protocols_dependency_is_pinned_by_full_commit():
    import tomllib

    deps = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["dependencies"]
    pins = [d for d in deps if d.startswith("trellion-protocols")]
    assert len(pins) == 1
    assert re.fullmatch(
        r"trellion-protocols @ git\+https://github\.com/personifAI/trellion-protocols\.git@[0-9a-f]{40}",
        pins[0],
    ), f"pin trellion-protocols by full commit SHA (a tag can move): {pins[0]}"
