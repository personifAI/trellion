# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""Trellion's optional shared SQLAlchemy engine pool.

When trellion runs inside noomesh, the gateway-side ``init_db`` creates
engines and (via ``set_engines``) shares them with trellion so all
cross-domain operations (joins, FK checks) hit the same connection pool.

For standalone trellion use, callers can use ``init_engine(dsn)`` directly.
Either way, ``get_engine()`` / ``get_async_engine()`` return the live pool.
"""
from __future__ import annotations

import os
from typing import Optional

import structlog
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

logger = structlog.get_logger(__name__)

#: Schema the memory plane lives in post-B′. Overridable for embedded/test
#: setups that keep everything in public.
TRELLION_SCHEMA = os.environ.get("NOO_TRELLION_SCHEMA", "trellion")


_engine: Optional[Engine] = None
_async_engine: Optional[AsyncEngine] = None


def _sync_dsn(dsn: str) -> str:
    if dsn.startswith("postgresql+psycopg://"):
        return dsn
    if dsn.startswith("postgresql+asyncpg://"):
        return dsn.replace("postgresql+asyncpg://", "postgresql+psycopg://")
    if dsn.startswith("postgresql://"):
        return dsn.replace("postgresql://", "postgresql+psycopg://", 1)
    if dsn.startswith("postgres://"):
        return dsn.replace("postgres://", "postgresql+psycopg://", 1)
    return dsn


def _async_dsn(dsn: str) -> str:
    # psycopg3 supports async natively, so the same prefix works for both
    return _sync_dsn(dsn)


def init_engine(dsn: str, pool_size: int = 10, max_overflow: int = 20) -> tuple[Engine, AsyncEngine]:
    """Create and store the shared engines.

    Idempotent — repeated calls dispose the prior engines first.
    """
    global _engine, _async_engine

    if _engine is not None:
        _engine.dispose()
    if _async_engine is not None:
        # Async dispose requires await; schedule it best-effort
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(_async_engine.dispose())
            else:
                loop.run_until_complete(_async_engine.dispose())
        except Exception:
            pass

    sync_url = _sync_dsn(dsn)
    async_url = _async_dsn(dsn)

    # The engines pin their own search_path: trellion's tables live in schema
    # `trellion` on installs built after proposal 19 phase B′ and in `public`
    # on ones that predate it, and every query in this package is
    # unqualified. Pinned HERE, by the consumer, not by ALTER ROLE from the
    # broker — each service declares where its own tables live. Names are
    # unique across both schemas, so order only decides lookup priority.
    _sp = {"options": f"-csearch_path={TRELLION_SCHEMA},public"}
    _engine = create_engine(
        sync_url,
        pool_size=pool_size,
        max_overflow=max_overflow,
        pool_timeout=30,
        pool_pre_ping=True,
        pool_recycle=3600,
        connect_args=_sp,
    )
    _async_engine = create_async_engine(
        async_url,
        pool_size=pool_size,
        max_overflow=max_overflow,
        pool_timeout=30,
        pool_pre_ping=True,
        pool_recycle=3600,
        connect_args=_sp,
    )
    logger.info("trellion.engines_created")
    return _engine, _async_engine


def set_engines(engine: Engine, async_engine: AsyncEngine) -> None:
    """Inject engines created by an external caller (e.g. noomesh).

    Used during noomesh's ``init_db`` lifespan to share its connection
    pool with trellion instead of opening a second pool.
    """
    global _engine, _async_engine
    _engine = engine
    _async_engine = async_engine
    logger.debug("trellion.engines_set_externally")


def get_engine() -> Engine:
    """Return the shared sync Engine. Raises if not initialised."""
    if _engine is None:
        raise RuntimeError(
            "trellion sync engine not initialised. "
            "Call trellion.db.engine.init_engine(dsn) or set_engines(...) first."
        )
    return _engine


def get_async_engine() -> AsyncEngine:
    """Return the shared AsyncEngine. Raises if not initialised."""
    if _async_engine is None:
        raise RuntimeError(
            "trellion async engine not initialised. "
            "Call trellion.db.engine.init_engine(dsn) or set_engines(...) first."
        )
    return _async_engine


async def close_engines() -> None:
    """Dispose both engine pools."""
    global _engine, _async_engine
    if _async_engine is not None:
        await _async_engine.dispose()
        _async_engine = None
    if _engine is not None:
        _engine.dispose()
        _engine = None
    logger.info("trellion.engines_closed")
