# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""Client-side throttle for gateway embedding calls.

The gateway's ``/api/external/embeddings`` admits at most 4 requests per
embedder target; a request that cannot get a slot within about 1.5 s is shed
with a 503 and a ``Retry-After`` of 2-10 s (scaled to queue depth) while the
GPU embedder is saturated. Every process
that backfills embeddings used to issue 32-text batches back to back, all
started together by a stack restart, and none of them slowed down on failure:
the callers retried only transport errors (never a 503), and a failed pass
simply came round again on the next 5 s tick. Several loops timing out
together kept the queue full, so ~75% of calls failed for as long as the
backlog lasted.

This module is the one place that decides *how hard* a process may push:

* **Bounded concurrency.** At most ``max_inflight`` (default 1) embedding
  requests per process, shared by every loop in it (page embedder, chunk
  embedder, graph entity embedder, classifier templates).
* **Retry with exponential backoff + full jitter** on 429/502/503/504 and
  transport errors, honouring ``Retry-After`` when the gateway sends one.
  A batch that exhausts its attempts raises :class:`EmbedUnavailable`; the
  caller leaves the rows unembedded so a later pass picks them up again.
* **AIMD batch size.** Each failed call halves the batch the loops are
  allowed to send (down to ``batch_min``); each success grows it back by
  ``batch_step`` up to ``batch_max``.
* **Cool-down.** After consecutive failed batches the loop sleeps an
  exponentially growing, jittered interval before the next one; after a
  success it paces by ``pace_s``.
* **Start-up jitter.** Loops wait ``uniform(0, startup_jitter_s)`` before
  their first pass so containers restarted together do not begin together.

Everything is overridable by environment (``TRELLION_EMBED_*``); the values
below are the declared defaults.
"""

from __future__ import annotations

import asyncio
import os
import random
from collections.abc import Awaitable, Callable
from email.utils import parsedate_to_datetime
from typing import Any

import httpx
import structlog

log = structlog.get_logger(__name__)

RETRYABLE_STATUS = frozenset({429, 502, 503, 504})

# Default longest wait between idle backlog polls (TRELLION_EMBED_IDLE_MAX_S).
# It is also the most a newly ingested document waits if its wake is missed:
# ingest NOTIFYs Postgres (live/doc_wake.py) and the embedder LISTENs, so the
# wake crosses containers through the shared database; this cap is the
# fallback. (Before that, the wake only reached a loop in the ingesting
# process and 60 s made the first citation of a console upload take 37-40 s,
# 0.6.13 rehearsal.) The idle poll is two NOT EXISTS index probes, so 10 s
# costs nothing that matters.
IDLE_MAX_S = 10.0


def _env_num(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


class EmbedUnavailable(Exception):  # noqa: N818 - reads as a condition, not an error
    """The embedding backend stayed overloaded/unreachable for every attempt.

    Expected under load: callers log one warning line and leave the work
    queued. It is deliberately NOT an ``HTTPStatusError`` so nothing treats
    it as an unexpected failure.
    """

    def __init__(self, reason: str, attempts: int, retry_after: float | None = None) -> None:
        super().__init__(f"{reason} after {attempts} attempts")
        self.reason = reason
        self.attempts = attempts
        # The last Retry-After the gateway sent, when it sent one.
        self.retry_after = retry_after


def retry_after_seconds(resp: httpx.Response | None) -> float | None:
    """Parse a ``Retry-After`` header (delta-seconds or HTTP date)."""
    if resp is None:
        return None
    raw = resp.headers.get("retry-after")
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        pass
    try:
        from datetime import UTC, datetime

        dt = parsedate_to_datetime(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return max(0.0, (dt - datetime.now(UTC)).total_seconds())
    except (TypeError, ValueError):
        return None


def classify(exc: BaseException) -> str | None:
    """Short reason for an *expected* overload error, else ``None``.

    ``None`` means "unexpected": the caller keeps its traceback.
    """
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        return f"http_{code}" if code in RETRYABLE_STATUS else None
    if isinstance(exc, httpx.TimeoutException):
        return "timeout"
    if isinstance(exc, httpx.TransportError):
        return type(exc).__name__
    if isinstance(exc, EmbedUnavailable):
        return exc.reason
    return None


class EmbedThrottle:
    def __init__(
        self,
        *,
        max_inflight: int | None = None,
        batch_max: int | None = None,
        batch_min: int | None = None,
        batch_step: int | None = None,
        max_attempts: int | None = None,
        backoff_base_s: float | None = None,
        backoff_cap_s: float | None = None,
        pace_s: float | None = None,
        startup_jitter_s: float | None = None,
        idle_max_s: float | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        rng: random.Random | None = None,
    ) -> None:
        def pick(v: Any, env: str, default: float) -> float:
            return _env_num(env, default) if v is None else v

        self.max_inflight = max(1, int(pick(max_inflight, "TRELLION_EMBED_MAX_INFLIGHT", 1)))
        self.batch_max = max(1, int(pick(batch_max, "TRELLION_EMBED_BATCH_MAX", 32)))
        self.batch_min = max(1, min(self.batch_max, int(pick(batch_min, "TRELLION_EMBED_BATCH_MIN", 4))))
        self.batch_step = max(1, int(pick(batch_step, "TRELLION_EMBED_BATCH_STEP", 4)))
        self.max_attempts = max(1, int(pick(max_attempts, "TRELLION_EMBED_MAX_ATTEMPTS", 4)))
        self.backoff_base_s = pick(backoff_base_s, "TRELLION_EMBED_BACKOFF_BASE_S", 2.0)
        self.backoff_cap_s = pick(backoff_cap_s, "TRELLION_EMBED_BACKOFF_CAP_S", 60.0)
        self.pace_s = pick(pace_s, "TRELLION_EMBED_PACE_S", 0.5)
        self.startup_jitter_s = pick(startup_jitter_s, "TRELLION_EMBED_STARTUP_JITTER_S", 30.0)
        self.idle_max_s = pick(idle_max_s, "TRELLION_EMBED_IDLE_MAX_S", IDLE_MAX_S)
        self._sleep = sleep
        self._rng = rng or random.Random()
        self._sem: asyncio.Semaphore | None = None
        self.batch_size = self.batch_max
        self.consecutive_failures = 0
        self.inflight = 0
        self.peak_inflight = 0

    # -- concurrency --------------------------------------------------

    def _semaphore(self) -> asyncio.Semaphore:
        if self._sem is None:
            self._sem = asyncio.Semaphore(self.max_inflight)
        return self._sem

    # -- AIMD -----------------------------------------------------------

    def record_success(self) -> None:
        self.consecutive_failures = 0
        self.batch_size = min(self.batch_max, self.batch_size + self.batch_step)

    def record_failure(self) -> None:
        self.consecutive_failures += 1
        self.batch_size = max(self.batch_min, self.batch_size // 2)

    def effective_batch(self, requested: int) -> int:
        return max(1, min(requested, self.batch_size))

    # -- delays ---------------------------------------------------------

    def backoff_delay(self, attempt: int, retry_after: float | None = None) -> float:
        """Full-jitter exponential delay for retry number *attempt* (1-based)."""
        ceiling = min(self.backoff_cap_s, self.backoff_base_s * (2 ** (attempt - 1)))
        delay = self._rng.uniform(ceiling / 2, ceiling)
        if retry_after is not None:
            delay = max(delay, min(retry_after, self.backoff_cap_s))
        return delay

    def cooldown_s(self) -> float:
        """How long a loop should rest before its next batch."""
        if self.consecutive_failures <= 0:
            return self.pace_s
        return self.backoff_delay(self.consecutive_failures)

    async def startup_delay(self) -> float:
        d = self._rng.uniform(0, self.startup_jitter_s) if self.startup_jitter_s > 0 else 0.0
        if d > 0:
            await self._sleep(d)
        return d

    # -- the call ---------------------------------------------------------

    async def call(
        self,
        fn: Callable[[], Awaitable[Any]],
        *,
        what: str,
        count: int,
        max_attempts: int | None = None,
        bounded: bool = True,
    ) -> Any:
        """Run ``fn`` under the concurrency bound with retry/backoff.

        ``bounded=False`` skips the semaphore (interactive query embeds must
        not queue behind a backfill) but keeps retry and AIMD accounting.
        Never logs the request payload.
        """
        attempts = max_attempts or self.max_attempts
        last_reason = "unknown"
        last_retry_after: float | None = None
        for attempt in range(1, attempts + 1):
            try:
                if bounded:
                    async with self._semaphore():
                        self._enter()
                        try:
                            result = await fn()
                        finally:
                            self.inflight -= 1
                else:
                    result = await fn()
            except Exception as exc:  # noqa: BLE001 - classified below
                reason = classify(exc)
                if reason is None:
                    raise
                last_reason = reason
                if isinstance(exc, EmbedUnavailable):
                    last_retry_after = exc.retry_after
                elif isinstance(exc, httpx.HTTPStatusError):
                    last_retry_after = retry_after_seconds(exc.response)
                self.record_failure()
                if attempt >= attempts:
                    break
                resp = exc.response if isinstance(exc, httpx.HTTPStatusError) else None
                delay = self.backoff_delay(attempt, retry_after_seconds(resp))
                log.warning(
                    "embed_backpressure",
                    what=what,
                    count=count,
                    reason=reason,
                    attempt=attempt,
                    max_attempts=attempts,
                    backoff_s=round(delay, 2),
                    batch_size=self.batch_size,
                )
                await self._sleep(delay)
                continue
            self.record_success()
            return result
        raise EmbedUnavailable(last_reason, attempts, last_retry_after)

    def _enter(self) -> None:
        self.inflight += 1
        self.peak_inflight = max(self.peak_inflight, self.inflight)


_default: EmbedThrottle | None = None


def get_throttle() -> EmbedThrottle:
    """The process-wide throttle shared by every embedding loop."""
    global _default
    if _default is None:
        _default = EmbedThrottle()
    return _default


def reset_throttle() -> None:
    """Drop the process-wide instance (tests)."""
    global _default
    _default = None
