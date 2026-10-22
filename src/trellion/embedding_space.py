# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""The embedding space trellion's vectors live in, and its provenance.

One module owns four things, so nothing else reads the environment or
talks to the capabilities door about embeddings:

* the configured SPACE name (``TRELLION_EMBEDDING_SPACE``, default
  ``knowledge``). Every embed call sends it as ``model``; the gateway maps a
  space to the model that currently serves it (noomesh followup i150);
* a short-TTL cached read of that space's record from
  ``GET /api/external/capabilities`` (``embeddings.spaces``): the current
  model, the vector width and any running re-embed migration;
* the **current model**, which every similarity query filters on
  (``embedding_model = <current model>``) so a table holding vectors from two
  models never ranks one against the other;
* the progress report (``PUT /api/external/embedding-spaces/{space}/migration``)
  the re-embed worker makes.

Provenance travels with the vector. The embed response names the model that
actually served the call; :class:`Embeddings` / :class:`Vector` are plain
``list`` subclasses that carry it as ``.model``, so every existing
``list[list[float]]`` consumer keeps working and a writer can record
``embedding_model`` without a signature change.

A failed lookup never empties a result. If the current model cannot be
determined the read filter is dropped (logged once) rather than returning
nothing.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

import httpx
import structlog

log = structlog.get_logger(__name__)

DEFAULT_SPACE = "knowledge"

#: Seconds a fetched space record is trusted. Short: a cutover must reach
#: readers within a minute, and the fetch is one small GET.
CACHE_TTL_S = 30.0

#: After a failed fetch with nothing cached, do not ask again for this long:
#: every similarity query would otherwise wait out the timeout.
RETRY_AFTER_FAILURE_S = 10.0

#: Where the gateway is, when no component told us (``configure_url``).
_DEFAULT_URL = "https://gateway:8080"


# ----------------------------------------------------------------------
# Vectors that remember the model that produced them
# ----------------------------------------------------------------------


class Vector(list):
    """One embedding. A ``list[float]`` that also knows its model."""

    model: Optional[str] = None

    def __init__(self, values=(), model: Optional[str] = None) -> None:
        super().__init__(values)
        self.model = model


class Embeddings(list):
    """A batch of embeddings. A ``list`` of :class:`Vector` plus the model."""

    model: Optional[str] = None

    def __init__(self, vectors=(), model: Optional[str] = None) -> None:
        super().__init__(vectors)
        self.model = model


def model_of(obj: Any) -> Optional[str]:
    """The model a vector (or batch) carries, or None for a bare list."""
    model = getattr(obj, "model", None)
    return model if isinstance(model, str) and model else None


def from_response(data: dict) -> Embeddings:
    """Build :class:`Embeddings` from an OpenAI-shaped embeddings response.

    Sorted by ``index`` (the contract does not promise order). The top-level
    ``model`` is the model that actually served the call.
    """
    model = data.get("model")
    model = model if isinstance(model, str) and model else None
    rows = sorted(data.get("data", []) or [], key=lambda d: d.get("index", 0))
    return Embeddings([Vector(r.get("embedding", []), model) for r in rows], model)


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

_url: Optional[str] = None


def space_name() -> str:
    """The configured space (``TRELLION_EMBEDDING_SPACE``, default ``knowledge``)."""
    return (os.environ.get("TRELLION_EMBEDDING_SPACE") or "").strip() or DEFAULT_SPACE


def next_alias() -> str:
    """The ``model`` value that embeds with the space's *next* model."""
    return f"{space_name()}@next"


def configure_url(url: Optional[str]) -> None:
    """Tell the module where the gateway is (first caller with a URL wins
    until reset; components pass the URL they already hold)."""
    global _url
    if url and not _url:
        _url = url.rstrip("/")


def gateway_url() -> str:
    return (_url or os.environ.get("PAGEINDEX_EMBEDDING_URL") or _DEFAULT_URL).rstrip("/")


# ----------------------------------------------------------------------
# The space record
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class SpaceRecord:
    """One entry of ``capabilities.embeddings.spaces``."""

    space: str
    model: str
    dims: Optional[int]
    default: bool
    migration: Optional[dict]

    @property
    def migration_state(self) -> Optional[str]:
        return (self.migration or {}).get("state")

    @property
    def next_model(self) -> Optional[str]:
        return (self.migration or {}).get("next_model")

    @property
    def next_dims(self) -> Optional[int]:
        return (self.migration or {}).get("next_dims")


def parse_space(payload: dict, space: str) -> Optional[SpaceRecord]:
    """Pick ``space`` out of a capabilities payload; None when absent."""
    spaces = ((payload or {}).get("embeddings") or {}).get("spaces") or []
    for s in spaces:
        if isinstance(s, dict) and s.get("space") == space and s.get("model"):
            mig = s.get("migration")
            return SpaceRecord(
                space=space,
                model=s["model"],
                dims=s.get("dims"),
                default=bool(s.get("default")),
                migration=mig if isinstance(mig, dict) else None,
            )
    return None


_lock = threading.Lock()
_cached: Optional[SpaceRecord] = None
_cached_at = 0.0
_retry_at = 0.0
_warned = False


def reset() -> None:
    """Forget everything cached (tests, and a config change at runtime)."""
    global _cached, _cached_at, _retry_at, _warned, _url
    with _lock:
        _cached = None
        _cached_at = 0.0
        _retry_at = 0.0
        _warned = False
        _url = None


def _client() -> httpx.Client:
    """The one HTTP client this module uses (a test seam)."""
    from trellion.live._tls import client_cert, ssl_verify
    from trellion.live.gateway_auth import GatewayHmacAuth

    return httpx.Client(
        timeout=5.0, verify=ssl_verify(), cert=client_cert(), auth=GatewayHmacAuth()
    )


def _fetch() -> SpaceRecord:
    with _client() as client:
        resp = client.get(f"{gateway_url()}/api/external/capabilities")
        resp.raise_for_status()
        rec = parse_space(resp.json(), space_name())
    if rec is None:
        raise LookupError(f"capabilities names no embedding space {space_name()!r}")
    return rec


def get_space(*, force: bool = False) -> Optional[SpaceRecord]:
    """The space's record, cached for :data:`CACHE_TTL_S`.

    On a failed fetch the last good record is kept (stale beats nothing);
    with none, returns None and warns once.
    """
    global _cached, _cached_at, _retry_at, _warned
    now = time.monotonic()
    with _lock:
        if not force and _cached is not None and now - _cached_at < CACHE_TTL_S:
            return _cached
        if not force and _cached is None and now < _retry_at:
            return None
    try:
        rec = _fetch()
    except Exception as exc:  # noqa: BLE001 - any failure degrades the same way
        with _lock:
            _retry_at = time.monotonic() + RETRY_AFTER_FAILURE_S
            if not _warned:
                _warned = True
                log.warning(
                    "embedding_space.lookup_failed",
                    space=space_name(),
                    error=str(exc)[:200],
                    note="read filters on embedding_model are dropped until the "
                         "gateway answers; results are not emptied",
                )
            return _cached
    with _lock:
        _cached, _cached_at, _warned, _retry_at = rec, time.monotonic(), False, 0.0
    return rec


async def aget_space(*, force: bool = False) -> Optional[SpaceRecord]:
    """Async :func:`get_space`; the blocking fetch runs in a thread."""
    now = time.monotonic()
    with _lock:
        if not force and _cached is not None and now - _cached_at < CACHE_TTL_S:
            return _cached
        if not force and _cached is None and now < _retry_at:
            return None
    return await asyncio.to_thread(get_space, force=force)


def current_model() -> Optional[str]:
    rec = get_space()
    return rec.model if rec else None


async def acurrent_model() -> Optional[str]:
    rec = await aget_space()
    return rec.model if rec else None


async def model_clause(
    column: str, *, query_vector: Any = None, param: str = "emb_model"
) -> tuple[str, dict]:
    """The read filter for a similarity query: ``(sql, params)``.

    The model is the query vector's own (it came from the embed response, so
    it is the space the vector was embedded in) and otherwise the space's
    current model. When neither is known the filter is dropped: ``("", {})``.
    """
    model = model_of(query_vector) or await acurrent_model()
    if not model:
        return "", {}
    return f"{column} = :{param}", {param: model}


def model_clause_sync(
    column: str, *, query_vector: Any = None, param: str = "emb_model"
) -> tuple[str, dict]:
    """:func:`model_clause` for synchronous stores."""
    model = model_of(query_vector) or current_model()
    if not model:
        return "", {}
    return f"{column} = :{param}", {param: model}


async def write_model(vector: Any = None) -> Optional[str]:
    """The model to record for a vector being written.

    The vector's own model when it carries one (the embed response), else the
    space's current model (a caller that built a bare list), else None.
    """
    return model_of(vector) or await acurrent_model()


# ----------------------------------------------------------------------
# Progress report
# ----------------------------------------------------------------------


def build_progress(next_model: str, tables: dict[str, tuple], cutover_done: bool) -> dict:
    """The exact body of the progress PUT.

    ``tables`` maps table -> ``(done, total)`` or ``(done, total, unsourced)``.
    ``unsourced`` counts rows that have an embedding but no stored source or
    derivation to re-embed from (or that were skipped); they are left out of
    ``done``/``total``. The gateway ignores keys it does not know.
    """
    out: dict[str, dict] = {}
    for t, counts in tables.items():
        entry = {"done": int(counts[0]), "total": int(counts[1])}
        if len(counts) > 2:
            entry["unsourced"] = int(counts[2])
        out[t] = entry
    return {"next_model": next_model, "tables": out, "cutover_done": bool(cutover_done)}


def put_progress(
    next_model: str, tables: dict[str, tuple], cutover_done: bool = False
) -> Optional[dict]:
    """``PUT /api/external/embedding-spaces/{space}/migration``.

    Returns the gateway's view of the space, or None when the report was
    refused or the gateway is unreachable (logged; the next loop retries).
    """
    body = build_progress(next_model, tables, cutover_done)
    try:
        with _client() as client:
            resp = client.put(
                f"{gateway_url()}/api/external/embedding-spaces/{space_name()}/migration",
                json=body,
            )
            resp.raise_for_status()
            return resp.json()
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "embedding_space.progress_report_failed",
            space=space_name(),
            error=str(exc)[:200],
        )
        return None


async def aput_progress(
    next_model: str, tables: dict[str, tuple], cutover_done: bool = False
) -> Optional[dict]:
    return await asyncio.to_thread(put_progress, next_model, tables, cutover_done)
