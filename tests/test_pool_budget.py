# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""trellion's pools take their size from trellion.db.pool_budget (noomesh i815).

noomesh's tests/test_pg_pool_budget.py and noomesh-internal's release-align
read POOL_CEILINGS from the SOURCE of src/trellion/db/pool_budget.py, so these
tests hold the three things that reading relies on: the declaration is a plain
literal, it sums to the approved budget, and every pool in src/trellion is
sized from it (no literal size, no SQLAlchemy default 5 + 10).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from trellion.db import pool_budget as pb

SRC = Path(__file__).resolve().parents[1] / "src"
BUDGET_FILE = SRC / "trellion/db/pool_budget.py"
POOL_CALLS = {"create_engine", "create_async_engine", "create_pool", "ConnectionPool", "AsyncConnectionPool"}
SIZE_KWARGS = {"pool_size", "max_overflow", "max_size", "min_size"}
# The one raw connection trellion holds open: the embedder's LISTEN.
LISTEN_SITE = "trellion/live/doc_wake.py"


def _literal(name: str, src: str):
    for node in ast.parse(src).body:
        target = node.target if isinstance(node, ast.AnnAssign) else (
            node.targets[0] if isinstance(node, ast.Assign) and len(node.targets) == 1 else None
        )
        if isinstance(target, ast.Name) and target.id == name:
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} is not a module-level literal in pool_budget.py")


def test_the_declaration_reads_without_importing():
    src = BUDGET_FILE.read_text()
    assert _literal("POOL_CEILINGS", src) == pb.POOL_CEILINGS
    assert _literal("STANDALONE_POOLS", src) == pb.STANDALONE_POOLS


def test_the_noo_trellion_pools_are_the_approved_budget():
    # noomesh i815 (2026-11-14). A change here is a budget change: noomesh's
    # tests/test_pg_pool_budget.py pins the same table.
    assert pb.POOL_CEILINGS == {
        "async_engine": (5, 10),
        "sync_engine": (2, 3),
        "pageindex_store": (5, 3),
        "doc_wake_listen": (1, 0),
    }
    assert pb.process_ceiling() == 29


def test_every_entry_is_positive_and_names_are_unique():
    assert not set(pb.POOL_CEILINGS) & set(pb.STANDALONE_POOLS)
    for name, (size, overflow) in {**pb.POOL_CEILINGS, **pb.STANDALONE_POOLS}.items():
        assert size >= 1 and overflow >= 0, name


def _call_name(func: ast.AST) -> str:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _budget_name(call: ast.Call) -> str | None:
    """The ``name`` in ``**pool_args("name")``, or None."""
    for kw in call.keywords:
        if kw.arg is None and isinstance(kw.value, ast.Call) and _call_name(kw.value.func) == "pool_args":
            args = kw.value.args
            if len(args) == 1 and isinstance(args[0], ast.Constant) and isinstance(args[0].value, str):
                return args[0].value
    return None


def _is_nullpool(call: ast.Call) -> bool:
    return any(kw.arg == "poolclass" and _call_name(kw.value) == "NullPool" for kw in call.keywords)


def scan(src: str, rel: str) -> tuple[list[str], set[str]]:
    """(offences, budget names used) for one module's source."""
    out, used = [], set()
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node.func)
        if name in POOL_CALLS:
            literal = [kw.arg for kw in node.keywords if kw.arg in SIZE_KWARGS]
            if literal:
                out.append(f"{rel}:{node.lineno} {name}() sizes itself ({', '.join(literal)}); use **pool_args(name)")
                continue
            if _is_nullpool(node):
                continue
            budget = _budget_name(node)
            if budget is None:
                out.append(f"{rel}:{node.lineno} {name}() takes SQLAlchemy's default size; use **pool_args(name)")
            elif budget not in pb.POOL_CEILINGS and budget not in pb.STANDALONE_POOLS:
                out.append(f"{rel}:{node.lineno} {name}() names {budget!r}, which pool_budget does not declare")
            else:
                used.add(budget)
        elif name == "connect" and isinstance(node.func, ast.Attribute) and _call_name(node.func.value) in (
            "AsyncConnection", "Connection", "asyncpg", "psycopg", "psycopg2",
        ):
            if rel == LISTEN_SITE:
                used.add("doc_wake_listen")
            else:
                out.append(f"{rel}:{node.lineno} a raw Postgres connection outside {LISTEN_SITE}; declare it in pool_budget")
    return out, used


def _production() -> tuple[list[str], set[str]]:
    offences, used = [], set()
    for path in sorted((SRC / "trellion").rglob("*.py")):
        rel = path.relative_to(SRC).as_posix()
        o, u = scan(path.read_text(encoding="utf-8-sig"), rel)
        offences += o
        used |= u
    return offences, used


def test_no_pool_in_src_sizes_itself_or_takes_the_default():
    offences, _ = _production()
    assert offences == []


def test_every_declared_pool_is_one_the_code_opens():
    # A declaration that outlives its pool overstates; one the code never
    # names is not what noomesh thinks it sums.
    _, used = _production()
    assert used == set(pb.POOL_CEILINGS) | set(pb.STANDALONE_POOLS)


@pytest.mark.parametrize(
    "src,bad",
    [
        ("create_engine(url, pool_size=7)\n", True),
        ("create_async_engine(url)\n", True),
        ("create_async_engine(url, **pool_args('nope'))\n", True),
        ("create_async_engine(url, **pool_args('async_engine'), pool_size=3)\n", True),
        ("psycopg.AsyncConnection.connect(c)\n", True),
        ("create_async_engine(url, **pool_args('async_engine'))\n", False),
        ("create_engine(url, poolclass=pool.NullPool)\n", False),
    ],
)
def test_the_guard_sees_what_it_is_for(src, bad):
    offences, _ = scan(src, "trellion/x.py")
    assert bool(offences) is bad, offences


def test_init_engine_builds_the_budgeted_sizes():
    from trellion.db import engine as eng

    sync, aeng = eng.init_engine("postgresql://u:p@127.0.0.1:1/db")
    try:
        assert (sync.pool.size(), sync.pool._max_overflow) == pb.POOL_CEILINGS["sync_engine"]
        assert (aeng.pool.size(), aeng.pool._max_overflow) == pb.POOL_CEILINGS["async_engine"]
    finally:
        sync.dispose()
        eng._engine = eng._async_engine = None


def test_a_dsn_built_page_index_store_is_budgeted():
    from trellion.live.pageindex_store import PageIndexStore

    s = PageIndexStore(dsn="postgresql://u:p@127.0.0.1:1/db", embedding_dim=4)
    assert (s._engine.pool.size(), s._engine.pool._max_overflow) == pb.POOL_CEILINGS["pageindex_store"]


def test_page_index_rag_uses_a_handed_engine():
    import asyncio

    from trellion.live import rag_integration as ri

    marker = object()
    rag = ri.PageIndexRAG(dsn="postgresql://x/y", async_engine=marker)
    built = {}

    class _Store:
        def __init__(self, async_engine=None, dsn=None, **kw):
            built.update(engine=async_engine, dsn=dsn)

        async def connect(self):
            raise RuntimeError("stop after the store")

    orig = ri.PageIndexStore
    ri.PageIndexStore = _Store
    try:
        with pytest.raises(RuntimeError, match="stop after the store"):
            asyncio.run(rag.startup())
    finally:
        ri.PageIndexStore = orig
    assert built == {"engine": marker, "dsn": None}
