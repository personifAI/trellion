# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""Bulk embedding through the gateway's job route (noomesh i179).

Background callers (the re-embed, the page/chunk backlog, the graph entity
sweep) are not waiting on a vector, so they hand their batch to the gateway as
a *job* that fleet workers claim, instead of holding a synchronous embed slot.
A caller that IS waiting (a query) never comes through here.

    POST /api/external/embeddings/jobs {"input": [...], "space": space}
        -> 202 {"job_id", "path", "model"}
    GET  /api/external/embeddings/jobs/{id}?vectors=1
        -> {"status", "model", "n", "dims", "error", "vectors"?}

Failure vocabulary, chosen so the existing callers keep working:

* 429 / 503 on submit or poll are raised as ``httpx.HTTPStatusError`` for the
  caller's ``EmbedThrottle.call``, which backs off honouring Retry-After and
  ends in :class:`EmbedUnavailable` (an outage, never a poisoned row).
* A job that ends ``dead`` raises :class:`BulkJobFailed`, an ordinary error:
  the re-embed retries those rows alone.
* A job still pending at the deadline raises :class:`BulkJobTimeout`; the
  embedder turns it into ``EmbedUnavailable``.
* 404/405 on submit means a gateway without the route: :class:`BulkUnsupported`.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Awaitable, Callable, Optional

import httpx
import structlog

from trellion import embedding_space

log = structlog.get_logger(__name__)

FLAG = "NOO_EMBED_BULK"
TIMEOUT_ENV = "NOO_EMBED_BULK_TIMEOUT_S"
DEFAULT_TIMEOUT_S = 900.0
MAX_INPUTS = 64  # the gateway's per-job cap (embed_jobs::MAX_INPUTS)
MAX_CONCURRENT_JOBS = 2
POLL_START_S = 0.5
POLL_CAP_S = 5.0

_TRUE = frozenset({"1", "true", "yes", "on"})
# Terminal statuses that are not a success. Anything else but "completed"
# (queued, running, ...) is still in flight.
_FAILED = frozenset({"dead", "failed", "cancelled", "canceled", "expired"})


def enabled() -> bool:
    """``NOO_EMBED_BULK`` (default OFF), read at call time."""
    return os.environ.get(FLAG, "").strip().lower() in _TRUE


def timeout_s() -> float:
    try:
        v = float(os.environ.get(TIMEOUT_ENV, DEFAULT_TIMEOUT_S))
    except (TypeError, ValueError):
        return DEFAULT_TIMEOUT_S
    return v if v > 0 else DEFAULT_TIMEOUT_S


class BulkUnsupported(Exception):
    """The gateway has no embedding-job route (404/405 on submit)."""


class BulkJobFailed(RuntimeError):
    """The job ended without vectors. Not an outage: retry the rows alone."""


class BulkJobTimeout(Exception):
    """The job was still pending at the deadline."""


def _raise_for(resp: httpx.Response) -> None:
    resp.raise_for_status()


async def _one_job(
    client: httpx.AsyncClient,
    base: str,
    texts: list[str],
    model: str,
    *,
    deadline: float,
    sleep: Callable[[float], Awaitable[None]],
    poll_start: float,
    poll_cap: float,
    resubmitted: bool = False,
) -> embedding_space.Embeddings:
    resp = await client.post(
        f"{base}/api/external/embeddings/jobs", json={"input": texts, "space": model}
    )
    if resp.status_code in (404, 405):
        raise BulkUnsupported(f"HTTP {resp.status_code}")
    if resp.status_code == 413 and len(texts) > 1:
        # Over the byte cap: halve it, in order.
        mid = len(texts) // 2
        kw = dict(deadline=deadline, sleep=sleep, poll_start=poll_start, poll_cap=poll_cap)
        a = await _one_job(client, base, texts[:mid], model, **kw)
        b = await _one_job(client, base, texts[mid:], model, **kw)
        return embedding_space.Embeddings([*a, *b], a.model or b.model)
    _raise_for(resp)
    job_id = resp.json().get("job_id")
    if not job_id:
        raise BulkJobFailed("the gateway accepted the job without a job_id")

    delay = poll_start
    while True:
        r = await client.get(
            f"{base}/api/external/embeddings/jobs/{job_id}", params={"vectors": "1"}
        )
        if r.status_code == 410:
            if resubmitted:
                raise BulkJobFailed(f"job {job_id}: vectors expired twice")
            log.warning("embed_bulk.vectors_expired_resubmitting", job_id=job_id)
            return await _one_job(
                client, base, texts, model, deadline=deadline, sleep=sleep,
                poll_start=poll_start, poll_cap=poll_cap, resubmitted=True,
            )
        if r.status_code == 404:
            raise BulkJobFailed(f"job {job_id} is unknown to the gateway")
        _raise_for(r)
        body = r.json()
        status = str(body.get("status") or "").lower()
        if status == "completed":
            vectors = body.get("vectors")
            served = body.get("model")
            served = served if isinstance(served, str) and served else None
            if not isinstance(vectors, list) or len(vectors) != len(texts):
                got = len(vectors) if isinstance(vectors, list) else None
                raise BulkJobFailed(
                    f"job {job_id}: {got} vectors for {len(texts)} inputs"
                )
            return embedding_space.Embeddings(
                [embedding_space.Vector(v, served) for v in vectors], served
            )
        if status in _FAILED:
            raise BulkJobFailed(
                f"job {job_id} {status}: {str(body.get('error') or '')[:200]}"
            )
        if time.monotonic() >= deadline:
            raise BulkJobTimeout(f"job {job_id} still {status or 'pending'}")
        await sleep(delay)
        delay = min(poll_cap, delay * 1.6)


async def embed_bulk(
    client: httpx.AsyncClient,
    base: str,
    texts: list[str],
    model: str,
    *,
    timeout: Optional[float] = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    poll_start: float = POLL_START_S,
    poll_cap: float = POLL_CAP_S,
) -> embedding_space.Embeddings:
    """Embed ``texts`` as <=64-text jobs, at most two in flight, in order."""
    deadline = time.monotonic() + (timeout if timeout is not None else timeout_s())
    chunks = [texts[i : i + MAX_INPUTS] for i in range(0, len(texts), MAX_INPUTS)]
    sem = asyncio.Semaphore(MAX_CONCURRENT_JOBS)

    async def run(chunk: list[str]) -> embedding_space.Embeddings:
        async with sem:
            return await _one_job(
                client, base, chunk, model, deadline=deadline, sleep=sleep,
                poll_start=poll_start, poll_cap=poll_cap,
            )

    parts = await asyncio.gather(*(run(c) for c in chunks))
    served = next((p.model for p in parts if p.model), None)
    return embedding_space.Embeddings([v for p in parts for v in p], served)
