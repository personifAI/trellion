# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""The squashed baseline (100) shares the conversation tables with its host.

Three installs, each on a scratch database under ``TRELLION_TEST_DSN``:

* fresh: trellion builds everything, conversation tables included (standalone);
* host-first: the host already built ``noo_conversations`` in its own schema,
  so trellion builds its own tables and NOT a second conversation copy;
* legacy: trellion's tables already exist, so the revision only stamps.
"""
from __future__ import annotations

import os
import subprocess
import sys
import uuid
from pathlib import Path

import pytest


DSN = os.environ.get("TRELLION_TEST_DSN")
needs_db = pytest.mark.skipif(not DSN, reason="TRELLION_TEST_DSN not set")

_ROOT = Path(__file__).resolve().parents[1]

_CONVERSATION_TABLES = {
    "noo_conversations", "noo_entries", "noo_feedback",
    "noo_tool_executions", "noo_context_retrievals",
}


def _alembic(dsn: str, *args: str) -> None:
    env = {**os.environ, "DATABASE_URL": dsn}
    subprocess.run(
        [sys.executable, "-m", "alembic", "-c", str(_ROOT / "src/trellion/db/alembic.ini"), *args],
        check=True, env=env, cwd=_ROOT, capture_output=True,
    )


async def _scratch(setup_sql: str, target: str) -> dict[str, set[str]]:
    """Run ``setup_sql`` on a fresh database, migrate to ``target``, and
    return the base tables per schema."""
    import asyncpg
    from sqlalchemy.engine import make_url

    plain = make_url(DSN).set(drivername="postgresql")
    name = f"squash_{uuid.uuid4().hex[:8]}"
    admin = await asyncpg.connect(plain.render_as_string(hide_password=False))
    try:
        await admin.execute(f'CREATE DATABASE "{name}"')
        scratch = plain.set(database=name).render_as_string(hide_password=False)
        con = await asyncpg.connect(scratch)
        try:
            await con.execute("CREATE EXTENSION IF NOT EXISTS vector; CREATE EXTENSION IF NOT EXISTS pg_trgm")
            if setup_sql:
                await con.execute(setup_sql)
            _alembic(scratch, "upgrade", target)
            tables: dict[str, set[str]] = {}
            for r in await con.fetch(
                "SELECT table_schema, table_name FROM information_schema.tables "
                "WHERE table_type = 'BASE TABLE' AND table_schema IN ('public','trellion','mesh')"
            ):
                tables.setdefault(r[0], set()).add(r[1])
            tables["_triggers"] = {r[0] for r in await con.fetch(
                "SELECT tgname FROM pg_trigger WHERE NOT tgisinternal")}
            return tables
        finally:
            await con.close()
    finally:
        await admin.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')


@needs_db
async def test_fresh_install_builds_the_conversation_tables_too():
    t = await _scratch("", "head")
    assert _CONVERSATION_TABLES <= t["trellion"]
    assert {"pages", "trellion_snapshots", "noo_temporal_edges"} <= t["trellion"]
    assert "noo_conv_set_org_trg" in t["_triggers"]


@needs_db
async def test_host_built_conversations_are_not_duplicated():
    t = await _scratch(
        "CREATE SCHEMA mesh; CREATE TABLE mesh.noo_conversations (id uuid PRIMARY KEY)",
        "head",
    )
    # trellion's own plane is built ...
    assert {"pages", "trellion_snapshots", "noo_temporal_edges"} <= t["trellion"]
    # ... but no second conversation copy to shadow the host's.
    assert not (_CONVERSATION_TABLES & t["trellion"])
    assert t["mesh"] == {"noo_conversations"}
    assert "noo_conv_set_org_trg" not in t["_triggers"]


@needs_db
async def test_legacy_install_is_stamped_not_rebuilt():
    t = await _scratch("CREATE TABLE public.trellion_snapshots (id uuid PRIMARY KEY)", "100_trellion_squash")
    assert "trellion_snapshots" in t["public"]
    assert t.get("trellion", set()) == {"alembic_version_memory"}
