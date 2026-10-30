# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""Wake the embedding loop when a document is ingested, across processes.

Ingest and the embedding loop run in different containers in the stack, so an
in-process event (or a bus subject the ingest process may not publish) never
reaches the loop. The one thing both already share is the database: ingest
issues ``NOTIFY`` on :data:`CHANNEL` once the document's pages are committed,
and the embedder holds one ``LISTEN`` connection. The loop's idle back-off
(``TRELLION_EMBED_IDLE_MAX_S``) stays the fallback for a missed notification.

``LISTEN`` needs a session: a direct connection or a session-mode pooler, not
a transaction-mode one.
"""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

import structlog

log = structlog.get_logger(__name__)

#: The Postgres channel both sides use.
CHANNEL = "trellion_doc_parsed"

#: Reconnect back-off bounds (seconds) for a lost listener connection.
RECONNECT_MIN_S = 1.0
RECONNECT_MAX_S = 30.0


async def notify_doc_parsed(engine: Any, doc_id: str) -> None:
    """``NOTIFY`` the channel with *doc_id*; never raises (ingest must not fail on it)."""
    if engine is None:
        return
    try:
        from sqlalchemy import text

        async with engine.begin() as conn:
            await conn.execute(text("SELECT pg_notify(:c, :d)"), {"c": CHANNEL, "d": str(doc_id)})
    except Exception:
        log.warning("doc_wake.notify_failed", doc_id=doc_id, exc_info=True)


def conninfo_from_engine(engine: Any) -> str:
    """A libpq URL (with password) for the engine's database."""
    return engine.url.set(drivername="postgresql").render_as_string(hide_password=False)


async def _psycopg_connect(conninfo: str) -> Any:
    import psycopg

    return await psycopg.AsyncConnection.connect(conninfo, autocommit=True)


class DocWakeListener:
    """One dedicated ``LISTEN`` connection that calls *on_wake* per notification."""

    def __init__(
        self,
        conninfo: str,
        on_wake: Callable[[], None],
        connect: Callable[[str], Awaitable[Any]] | None = None,
        reconnect_min_s: float = RECONNECT_MIN_S,
        reconnect_max_s: float = RECONNECT_MAX_S,
    ) -> None:
        self._conninfo = conninfo
        self._on_wake = on_wake
        self._connect = connect or _psycopg_connect
        self._min = reconnect_min_s
        self._max = reconnect_max_s
        self._task: asyncio.Task | None = None
        self._conn: Any = None

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        conn, self._conn = self._conn, None
        if conn is not None:
            try:
                await conn.close()
            except Exception:
                pass

    async def _run(self) -> None:
        delay = self._min
        warned = False
        while True:
            try:
                conn = await self._connect(self._conninfo)
                self._conn = conn
                await conn.execute(f"LISTEN {CHANNEL}")
                if warned:
                    log.info("doc_wake.listener_restored")
                warned = False
                delay = self._min
                log.info("doc_wake.listening", channel=CHANNEL)
                async for _ in conn.notifies():
                    self._on_wake()
                raise ConnectionError("notification stream ended")
            except asyncio.CancelledError:
                raise
            except Exception as e:
                if not warned:
                    log.warning("doc_wake.listener_lost", error=str(e), retry_in_s=delay)
                    warned = True
                old, self._conn = self._conn, None
                if old is not None:
                    try:
                        await old.close()
                    except Exception:
                        pass
                await asyncio.sleep(delay)
                delay = min(delay * 2, self._max)
