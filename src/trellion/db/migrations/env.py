# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""Alembic environment for the trellion memory subsystem.

Owns the memory-domain tables defined in ``trellion.db.models``, in schema
``trellion`` on installs built after proposal 19 phase B′, adopted in place
in ``public`` on installs that predate it.

RUNS AS ``noo_trellion_ddl``, NOT AS SUPERUSER. This env used to require the
privileged ``pai`` connection for exactly two reasons, and both are gone:

* ``CREATE EXTENSION vector / pg_trgm`` — untrusted extensions, superuser
  only. The BROKER installs them now (``systems:`` manifest in
  noomesh.yaml); this env merely verifies they exist and says whose job it
  was if they do not.
* creating the ``noo_trellion`` role — the broker's job since phase A′
  superseded this env's own provisioning (which itself replaced the gateway's
  038/039). Third home for that logic, and the right one: provisioning
  requires privilege, and the broker is the one component that is privileged
  by design.

VERSION TABLE ``alembic_version_memory``, in the TRELLION SCHEMA. Legacy
installs carry ``public.alembic_version_memory`` stamped by the old 001–012
chain; that row simply stops being read once this env takes over — dropped at
the cutover, not here.
"""
from __future__ import annotations

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool, text

from trellion.db.models import Base


config = context.config

if config.config_file_name and os.path.exists(config.config_file_name):
    try:
        fileConfig(config.config_file_name)
    except Exception:
        pass

target_metadata = Base.metadata

VERSION_TABLE = "alembic_version_memory"

#: Where fresh installs put the memory plane. Legacy installs have it in
#: `public` and are adopted in place.
TRELLION_SCHEMA = os.environ.get("NOO_TRELLION_SCHEMA", "trellion")

#: Installed by the broker in-cluster; standalone installs get them from
#: whoever owns their database. Verified below, never created here — that is
#: precisely the superuser requirement this env shed.
REQUIRED_EXTENSIONS = ("vector", "pg_trgm")


def _get_url() -> str:
    """Resolve the MIGRATION DSN, in priority order.

    ``NOO_TRELLION_DDL_DATABASE_URL`` first and deliberately: migrations run
    as the DDL role, the worker pool runs as the runtime role, and conflating
    them leaves the runtime holding CREATE for its whole life.

    ``NOO_MIGRATIONS_DSN`` next — the legacy privileged path, kept so a
    container that predates the split keeps migrating during the transition.
    """
    url = config.get_main_option("sqlalchemy.url")
    if url:
        return url
    for key in (
        "NOO_TRELLION_DDL_DATABASE_URL",
        "NOO_MIGRATIONS_DSN",
        "PAGEINDEX_DSN",
        "PGVECTOR_URL",
        "DATABASE_URL",
    ):
        val = os.environ.get(key)
        if val:
            return val
    raise RuntimeError(
        "No database URL configured. Set NOO_TRELLION_DDL_DATABASE_URL (the "
        "migration identity) or sqlalchemy.url in alembic.ini."
    )


def run_migrations_offline() -> None:
    context.configure(
        url=_get_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        version_table=VERSION_TABLE,
        version_table_schema=TRELLION_SCHEMA,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    url = _get_url()

    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+psycopg://", 1)
    elif url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)

    connectable = engine_from_config(
        {"sqlalchemy.url": url},
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        # VERIFY the extensions, do not create them. Creating needs superuser
        # (both are untrusted), which is the requirement this env shed. The
        # error names whose job it was, because "type vector does not exist"
        # three tables into a migration is a much worse place to find out.
        for ext in REQUIRED_EXTENSIONS:
            present = connection.execute(
                text("SELECT 1 FROM pg_extension WHERE extname = :e"),
                {"e": ext},
            ).scalar()
            if not present:
                raise RuntimeError(
                    f"extension {ext!r} is not installed. In-cluster the "
                    f"BROKER installs it — declare it under the system's "
                    f"`extensions:` in noomesh.yaml's systems manifest. "
                    f"Standalone, install it as your database owner before "
                    f"migrating."
                )

        # CHECK-THEN-CREATE for the schema (Postgres checks privilege BEFORE
        # `IF NOT EXISTS`). In-cluster the broker already made
        # it and this role cannot; standalone the connecting role owns its
        # database and can.
        exists = connection.execute(
            text("SELECT 1 FROM pg_namespace WHERE nspname = :n"),
            {"n": TRELLION_SCHEMA},
        ).scalar()
        if not exists:
            connection.execute(text(f"CREATE SCHEMA {TRELLION_SCHEMA}"))

        # Unqualified DDL lands in the FIRST search_path entry, so the
        # baseline creates into the trellion schema on a fresh install while
        # its adopt-check still sees legacy tables in public.
        #
        # The COMMIT is load-bearing, not tidiness. `execute()` autobegins a
        # transaction, and alembic decides who owns transaction management by
        # whether one is already in progress at configure() time: in progress
        # means THE CALLER commits — and this exact pattern's first cut
        # did not. The entire migration applied and rolled back on
        # connection close: `upgrade` exited 0, `current` showed the stamp
        # from inside its own transaction, and the database contained
        # nothing. SET SESSION survives the commit; only SET LOCAL would not.
        connection.execute(text(f"SET search_path = {TRELLION_SCHEMA}, public"))
        connection.commit()

        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            version_table=VERSION_TABLE,
            # With the tables it versions — defaulting to public would need
            # CREATE there, which noo_trellion_ddl deliberately lacks.
            version_table_schema=TRELLION_SCHEMA,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
