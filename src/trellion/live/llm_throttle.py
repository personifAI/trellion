# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""Client-side throttle + per-model circuit breaker for gateway LLM calls.

When no node serves the chat model, a loop that re-selects the same
unextracted pages every tick would ask again and fail again without end. This
module bounds that: it is the LLM-side counterpart of
``embed_throttle`` (same shape, same helpers):

* **Bounded concurrency.** At most ``max_inflight`` (default 2) LLM requests
  per process, shared by every loop in it. Interactive single calls pass
  ``bounded=False`` and skip the bound (they must not queue behind a backfill)
  but still honour the breaker.
* **Retry with exponential backoff + full jitter** on 429/502/503/504,
  timeouts and transport errors, honouring ``Retry-After``.
* **Per-model circuit breaker.** A 404 ``model_not_found`` (the gateway says
  nobody serves that model) opens the breaker for *that model* at once;
  ``failure_threshold`` consecutive 502/503/504/transport failures open it
  too. While open, no call is issued for that model: :meth:`call` raises
  :class:`LLMBreakerOpen` without touching the network, and loops ask
  :meth:`blocked_for` to pause the work that needs the model (leaving it
  queued, never marked done). After ``cooldown_s`` one probe call is let
  through (doubling to ``cooldown_cap_s`` while probes fail); the first
  success closes the breaker. One warning is logged when it opens and one
  info line when it closes (model, how long it was open, how many items
  were deferred), not one line per failed call.
* Other models are unaffected: state is keyed by model.

Everything is overridable by environment (``TRELLION_LLM_*``); the values
below are the declared defaults.
"""

from __future__ import annotations

import asyncio
import os
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import structlog

from trellion.live.embed_throttle import RETRYABLE_STATUS, retry_after_seconds

log = structlog.get_logger(__name__)

# Statuses that count toward the "model is unavailable" breaker threshold.
# 429 is the gateway shedding load, not absence: it backs off but never trips.
_UNAVAILABLE_STATUS = frozenset({502, 503, 504})


def _env_num(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


class LLMUnavailable(Exception):  # noqa: N818 - reads as a condition, not an error
    """The model could not be asked (overloaded, unreachable, not served).

    Expected under outage: callers leave the work queued and log nothing per
    item. Deliberately NOT an ``HTTPStatusError`` so nothing treats it as an
    unexpected failure. ``cause`` is the last underlying exception, if any.
    """

    def __init__(
        self, model: str, reason: str, attempts: int = 0, cause: BaseException | None = None
    ) -> None:
        super().__init__(f"{model}: {reason} after {attempts} attempts")
        self.model = model
        self.reason = reason
        self.attempts = attempts
        self.cause = cause


class LLMBreakerOpen(LLMUnavailable):
    """The breaker for this model is open: no request was sent."""


def _response(exc: BaseException) -> Any:
    return getattr(exc, "response", None)


def _status(exc: BaseException) -> int | None:
    resp = _response(exc)
    code = getattr(resp, "status_code", None)
    if isinstance(code, int):
        return code
    code = getattr(exc, "status_code", None)  # openai.APIStatusError
    return code if isinstance(code, int) else None


def is_model_not_found(exc: BaseException) -> bool:
    """404 whose body says the *model* is not served (not a wrong URL)."""
    if _status(exc) != 404:
        return False
    if getattr(exc, "code", None) == "model_not_found":
        return True
    resp = _response(exc)
    text = ""
    try:
        text = resp.text if resp is not None else ""
    except Exception:  # noqa: BLE001 - unread streaming body
        text = ""
    return "model_not_found" in (text or "") or "model_not_found" in str(exc)


def classify_llm(exc: BaseException) -> tuple[str, bool] | None:
    """``(reason, counts_toward_breaker)`` for an *expected* error, else ``None``.

    ``None`` means "unexpected": the caller keeps its traceback. Duck-typed on
    ``.response.status_code`` so httpx and the OpenAI SDK both classify.
    """
    if isinstance(exc, LLMUnavailable):
        return exc.reason, False
    status = _status(exc)
    if status is not None:
        if is_model_not_found(exc):
            return "model_not_found", True
        if status in RETRYABLE_STATUS:
            return f"http_{status}", status in _UNAVAILABLE_STATUS
        return None
    name = type(exc).__name__
    try:
        import httpx

        if isinstance(exc, httpx.TimeoutException):
            return "timeout", True
        if isinstance(exc, httpx.TransportError):
            return name, True
    except ImportError:  # pragma: no cover
        pass
    if name in ("APITimeoutError",):
        return "timeout", True
    if name in ("APIConnectionError",):
        return name, True
    return None


@dataclass
class _Breaker:
    failures: int = 0
    open: bool = False
    opened_at: float = 0.0
    next_probe_at: float = 0.0
    cooldown_s: float = 0.0
    probing: bool = False
    deferred: int = 0
    reason: str = ""
    probes_failed: int = 0
    opens: int = field(default=0)


class LLMThrottle:
    def __init__(
        self,
        *,
        max_inflight: int | None = None,
        max_attempts: int | None = None,
        backoff_base_s: float | None = None,
        backoff_cap_s: float | None = None,
        failure_threshold: int | None = None,
        cooldown_s: float | None = None,
        cooldown_cap_s: float | None = None,
        startup_jitter_s: float | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        clock: Callable[[], float] = time.monotonic,
        rng: random.Random | None = None,
    ) -> None:
        def pick(v: Any, env: str, default: float) -> float:
            return _env_num(env, default) if v is None else v

        self.max_inflight = max(1, int(pick(max_inflight, "TRELLION_LLM_MAX_INFLIGHT", 2)))
        self.max_attempts = max(1, int(pick(max_attempts, "TRELLION_LLM_MAX_ATTEMPTS", 3)))
        self.backoff_base_s = pick(backoff_base_s, "TRELLION_LLM_BACKOFF_BASE_S", 2.0)
        self.backoff_cap_s = pick(backoff_cap_s, "TRELLION_LLM_BACKOFF_CAP_S", 60.0)
        self.failure_threshold = max(
            1, int(pick(failure_threshold, "TRELLION_LLM_BREAKER_THRESHOLD", 5))
        )
        self.cooldown_s = max(0.0, pick(cooldown_s, "TRELLION_LLM_BREAKER_COOLDOWN_S", 30.0))
        self.cooldown_cap_s = max(
            self.cooldown_s, pick(cooldown_cap_s, "TRELLION_LLM_BREAKER_COOLDOWN_CAP_S", 600.0)
        )
        self.startup_jitter_s = pick(startup_jitter_s, "TRELLION_LLM_STARTUP_JITTER_S", 30.0)
        self._sleep = sleep
        self._clock = clock
        self._rng = rng or random.Random()
        self._sem: asyncio.Semaphore | None = None
        self._breakers: dict[str, _Breaker] = {}
        self.inflight = 0
        self.peak_inflight = 0

    # -- concurrency ------------------------------------------------------

    def _semaphore(self) -> asyncio.Semaphore:
        if self._sem is None:
            self._sem = asyncio.Semaphore(self.max_inflight)
        return self._sem

    # -- delays -----------------------------------------------------------

    def backoff_delay(self, attempt: int, retry_after: float | None = None) -> float:
        ceiling = min(self.backoff_cap_s, self.backoff_base_s * (2 ** (attempt - 1)))
        delay = self._rng.uniform(ceiling / 2, ceiling)
        if retry_after is not None:
            delay = max(delay, min(retry_after, self.backoff_cap_s))
        return delay

    async def startup_delay(self) -> float:
        d = self._rng.uniform(0, self.startup_jitter_s) if self.startup_jitter_s > 0 else 0.0
        if d > 0:
            await self._sleep(d)
        return d

    # -- breaker ------------------------------------------------------------

    def _b(self, model: str) -> _Breaker:
        b = self._breakers.get(model)
        if b is None:
            b = self._breakers[model] = _Breaker()
        return b

    def is_open(self, model: str) -> bool:
        b = self._breakers.get(model)
        return bool(b and b.open)

    def blocked_for(self, model: str) -> float | None:
        """Seconds until a call for *model* may be issued, ``None`` if one may now.

        Read-only: does not claim the probe slot. Loops use it to pause the
        work that needs the model instead of selecting items to fail.
        """
        b = self._breakers.get(model)
        if b is None or not b.open:
            return None
        if b.probing:
            return max(1.0, self.cooldown_s)
        wait = b.next_probe_at - self._clock()
        return wait if wait > 0 else None

    def defer(self, model: str, n: int = 1) -> None:
        """Note *n* items left queued because the model's breaker is open."""
        b = self._breakers.get(model)
        if b is not None and b.open and n > 0:
            b.deferred += n

    def _admit(self, model: str) -> bool:
        """Return True when this call is the probe; raise when blocked."""
        b = self._breakers.get(model)
        if b is None or not b.open:
            return False
        if not b.probing and self._clock() >= b.next_probe_at:
            b.probing = True
            return True
        b.deferred += 1
        raise LLMBreakerOpen(model, b.reason, 0)

    def _record_success(self, model: str) -> None:
        b = self._breakers.get(model)
        if b is None:
            return
        if b.open:
            log.info(
                "llm_breaker_closed",
                model=model,
                open_for_s=round(self._clock() - b.opened_at, 1),
                deferred=b.deferred,
                probes_failed=b.probes_failed,
            )
        self._breakers[model] = _Breaker(opens=b.opens)

    def _record_unavailable(self, model: str, reason: str, *, immediate: bool) -> None:
        b = self._b(model)
        now = self._clock()
        if b.open:
            # A failed probe: stay open, back the next probe off.
            b.probing = False
            b.probes_failed += 1
            b.cooldown_s = min(self.cooldown_cap_s, max(b.cooldown_s, self.cooldown_s) * 2)
            b.next_probe_at = now + b.cooldown_s
            b.reason = reason
            return
        b.failures += 1
        if immediate or b.failures >= self.failure_threshold:
            b.open = True
            b.opened_at = now
            b.cooldown_s = self.cooldown_s
            b.next_probe_at = now + b.cooldown_s
            b.reason = reason
            b.opens += 1
            log.warning(
                "llm_breaker_open",
                model=model,
                reason=reason,
                consecutive_failures=b.failures,
                next_probe_in_s=round(b.cooldown_s, 1),
                note="calls for this model are paused; work stays queued",
            )

    # -- the call -------------------------------------------------------------

    async def call(
        self,
        model: str,
        fn: Callable[[], Awaitable[Any]],
        *,
        what: str,
        max_attempts: int | None = None,
        bounded: bool = True,
    ) -> Any:
        """Run ``fn`` for *model* under the bound, breaker and retry policy.

        Raises :class:`LLMBreakerOpen` without calling ``fn`` while the
        breaker is open; :class:`LLMUnavailable` when attempts are exhausted
        or the breaker opens mid-call. Unexpected exceptions propagate
        unchanged (the caller logs them with their traceback). Never logs
        the request payload.
        """
        is_probe = self._admit(model)
        attempts = 1 if is_probe else (max_attempts or self.max_attempts)
        last_reason = "unknown"
        last_exc: BaseException | None = None
        try:
            for attempt in range(1, attempts + 1):
                try:
                    if bounded:
                        async with self._semaphore():
                            self.inflight += 1
                            self.peak_inflight = max(self.peak_inflight, self.inflight)
                            try:
                                result = await fn()
                            finally:
                                self.inflight -= 1
                    else:
                        result = await fn()
                except Exception as exc:  # noqa: BLE001 - classified below
                    verdict = classify_llm(exc)
                    if verdict is None:
                        raise
                    reason, counts = verdict
                    last_reason, last_exc = reason, exc
                    if counts:
                        self._record_unavailable(
                            model, reason, immediate=reason == "model_not_found"
                        )
                        if self.is_open(model):
                            break  # stop retrying: the breaker now speaks for this model
                    if attempt >= attempts:
                        break
                    delay = self.backoff_delay(attempt, retry_after_seconds(_response(exc)))
                    log.warning(
                        "llm_backpressure",
                        what=what,
                        model=model,
                        reason=reason,
                        attempt=attempt,
                        max_attempts=attempts,
                        backoff_s=round(delay, 2),
                    )
                    await self._sleep(delay)
                    continue
                self._record_success(model)
                return result
        finally:
            if is_probe:
                b = self._breakers.get(model)
                if b is not None:
                    b.probing = False
        raise LLMUnavailable(model, last_reason, attempts, last_exc)


_default: LLMThrottle | None = None


def get_llm_throttle() -> LLMThrottle:
    """The process-wide throttle shared by every LLM caller."""
    global _default
    if _default is None:
        _default = LLMThrottle()
    return _default


def set_llm_throttle(t: LLMThrottle | None) -> None:
    """Install (or drop) the process-wide instance (tests)."""
    global _default
    _default = t


def reset_llm_throttle() -> None:
    set_llm_throttle(None)
