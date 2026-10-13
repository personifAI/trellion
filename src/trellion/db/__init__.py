# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""Trellion database layer — ORM models + Alembic migrations + manifest.

This package owns the memory-domain tables that were previously defined
in ``noomesh/db/models.py``. Extracted in Phase 4 of proposal 06.

The ORM ``Base`` here is independent of noomesh's Base — trellion is a
standalone library, so its tables live under their own ``MetaData``.
The two Bases coexist in the same Postgres database, with separate
Alembic version tables (``alembic_version_memory`` / ``alembic_version_mesh``)
managed by noomesh's ``migration_runner``.

Public API:
    from trellion.db import Base, MEMORY_TABLE_NAMES
    from trellion.db.models import PaiConversation, PaiEntry, ...
    from trellion.db.engine import init_engine, get_async_engine, get_engine
"""
from __future__ import annotations

from .models import Base


# Tables owned by trellion's memory subsystem, by ``__tablename__``.
#
# DERIVED, not restated. This was a hand-maintained frozenset until
# 2026-08-04, and it had drifted twice by then: ``entity_graph_signals``
# (declared by migration 011) was never added, and neither were the two
# prune staging tables. Both drifts had the same silent failure mode —
# the table keeps whatever owner it was created with, and trellion only
# finds out when a later migration tries to ALTER it.
#
# The gateway's revision 038 documented the fix as an upstream filing against
# this file ("they belong in MEMORY_TABLE_NAMES in the trellion repo").
# This is that fix, done so it cannot drift a third time: the model
# tables come from the metadata that defines them.
_DECLARED_TABLES: frozenset[str] = frozenset(Base.metadata.tables)

# Memory-plane tables trellion OWNS but does not DECLARE. `graph_prune`
# is their only writer and it runs in the memory worker pool, but they
# are created by the gateway's chain (revisions 014/037) because the prune
# safety net predates the split. Ownership follows the writer, so they
# are adopted here rather than left owned by `pai` — without them a
# prune cannot stage, which is what makes prune reversible.
#
# Adopting is not declaring: they stay out of ``Base.metadata``, so
# trellion's Alembic autogenerate will never try to create or drop them.
_ADOPTED_TABLES: frozenset[str] = frozenset({
    "graph_prune_staging",
    "graph_edge_prune_staging",
})

MEMORY_TABLE_NAMES: frozenset[str] = _DECLARED_TABLES | _ADOPTED_TABLES


def memory_tables(metadata=None):
    """Return Table objects for memory-owned tables.

    By default uses trellion's own Base.metadata (which only contains memory
    tables). Pass an external metadata to filter — useful for callers that
    have a unified metadata across multiple Bases.
    """
    md = metadata if metadata is not None else Base.metadata
    return [md.tables[name] for name in MEMORY_TABLE_NAMES if name in md.tables]


def non_memory_tables(metadata):
    """Return Table objects in *metadata* that are NOT memory-owned.

    Used by the gateway's Alembic env to filter out memory tables when its Base
    legacy-imports trellion models for backward compatibility.
    """
    return [t for name, t in metadata.tables.items() if name not in MEMORY_TABLE_NAMES]


__all__ = [
    "Base",
    "MEMORY_TABLE_NAMES",
    "memory_tables",
    "non_memory_tables",
]
