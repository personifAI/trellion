# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""Every Postgres pool trellion opens, and how big it may get (noomesh i815).

A pool's ceiling is ``pool_size + max_overflow``. Inside noomesh, trellion's
pools share the ``noo-trellion`` process with noomesh's own (audit sink,
HeartbeatStore, ConversationStore), and every process's ceilings add up
against Postgres ``max_connections``. noomesh's ``tests/test_pg_pool_budget.py``
sums them; it reads ``POOL_CEILINGS`` below from this file's SOURCE (a literal
dict, parsed with ``ast.literal_eval``, never imported), and noomesh-internal's
``release-align.sh`` reads it at the sha the release pins. So:

* keep ``POOL_CEILINGS`` and ``STANDALONE_POOLS`` plain literals;
* every engine in ``src/trellion`` takes its size from here, through
  ``pool_args(name)`` (``tests/test_pool_budget.py`` fails on a pool built
  with a literal or SQLAlchemy's default size);
* a change to a number here is a budget change: noomesh's table pins it.

Approved budget (noomesh i815, 2026-11-14): trellion's own pools in
noo-trellion = 29. Do not starve the async engine to fit; it is the one ingest
and retrieval wait on.
"""

from __future__ import annotations

from typing import Dict, Tuple

# The pools that open in the noo-trellion process. name -> (pool_size, max_overflow).
# Sum of ceilings: 15 + 5 + 8 + 1 = 29, the approved budget.
POOL_CEILINGS: Dict[str, Tuple[int, int]] = {
    # init_engine's async engine: snapshot/consolidation handlers, recall,
    # autonomy (noomesh workers/common._trellion_engine calls init_engine(dsn)
    # and warms it at worker start). Was 10 + 20.
    "async_engine": (5, 10),
    # init_engine's sync engine. A SQLAlchemy pool connects on first use, so
    # it holds nothing until a sync caller (the CLI, migrations helpers)
    # checks one out. Was 10 + 20.
    "sync_engine": (2, 3),
    # PageIndexStore, built in PageIndexRAG.startup from the DSN unless the
    # host hands it an engine. noomesh's POOL_BUDGET["pageindex_store"] names
    # the same 5 + 3. Was 5 + 10 (SQLAlchemy default).
    "pageindex_store": (5, 3),
    # live/doc_wake.py: the embedder's one dedicated LISTEN connection (not a
    # pool; one psycopg connection while background workers run).
    "doc_wake_listen": (1, 0),
}

# Pools trellion can open that do NOT open in noo-trellion under noomesh:
# each is a DSN fallback noomesh does not take, or a tool. Sized here so no
# pool anywhere takes SQLAlchemy's default 5 + 10; not part of the sum.
STANDALONE_POOLS: Dict[str, Tuple[int, int]] = {
    # ConversationStore(dsn=...). noomesh builds this store's engine itself
    # (its POOL_BUDGET["conversation_store"]) and hands it over.
    "conversation_store_dsn": (5, 3),
    # PGVectorStore(PGVectorConfig(dsn=...)): legacy flat memory table. Opened
    # only by noomesh factory._build_pgvector_store when memory.pgvector is
    # enabled (default off).
    "pgvector_store_dsn": (2, 3),
    # eval/live_bench.py: the offline retrieval bench, its own process.
    "live_bench": (2, 3),
}


def ceiling(name: str) -> int:
    size, overflow = {**POOL_CEILINGS, **STANDALONE_POOLS}[name]
    return size + overflow


def process_ceiling() -> int:
    """trellion's own pools' ceiling in the noo-trellion process."""
    return sum(size + overflow for size, overflow in POOL_CEILINGS.values())


def pool_args(name: str) -> Dict[str, int]:
    """``pool_size`` / ``max_overflow`` for pool ``name``; KeyError if undeclared."""
    size, overflow = {**POOL_CEILINGS, **STANDALONE_POOLS}[name]
    return {"pool_size": size, "max_overflow": overflow}
