# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""The disposition pass (slice 1, shadow mode).

For each page that has no disposition yet, ask the decision endpoint the
active spec's questions, apply :func:`decide` and RECORD what would be done:
embed the page, extract entities from it, or hold it. Nothing reads the record
to change behaviour. The pass never writes ``pages``, embeddings or extraction
state; it only reads them and writes ``page_dispositions``.

Boundaries:

* No endpoint (``TRELLION_DECISION_URL`` unset): inert.
* Only documents readable by a plain ``user`` are sent to the endpoint;
  restricted-audience documents are skipped and counted in the summary.
* An invalid ACTIVE stored spec stops the pass (the built-in default is NOT
  substituted) and is reported.
* Load is bounded: at most ``TRELLION_DECISION_MAX_PAGES_PER_HOUR`` pages are
  asked per hour (default 600), and while the endpoint is failing the pass
  sends one probe page per backoff period, the period doubling up to an hour.

Deployment settings (each documented in docs/configuration.md):
``TRELLION_DECISION_INTERVAL_S``, ``TRELLION_DECISION_BATCH``,
``TRELLION_DECISION_CONCURRENCY``, ``TRELLION_DECISION_MAX_PAGES_PER_HOUR``.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any, Callable, Optional

import structlog

from trellion.live.decision_client import (
    DecisionClient,
    DecisionError,
    _env_number,
    non_private_host,
)
from trellion.live.knowledge_spec import KnowledgeSpec, decide
from trellion.live.knowledge_store import KnowledgeStore, StoredSpecError

log = structlog.get_logger(__name__)

#: A document is left alone this long after it was created, so its audience is
#: settled (it is set just after ingest) before anything is sent. Not a setting.
MIN_AGE_S = 300.0
DEFAULT_MAX_PAGES_PER_HOUR = 600
BACKOFF_MAX_S = 3600.0

NO_ENDPOINT = "no decision endpoint configured"
#: Pages claimed per batch.
_BATCH_SIZE = 16
#: A page that got no decision (endpoint error, unusable answers) is not
#: offered again by this process for this long, so one stubborn page at the
#: front of the queue cannot starve the rest. In memory only: a restart retries.
_COOLDOWN_S = 1800.0
_COOLDOWN_MAX = 10_000


@dataclass
class PassReport:
    configured: bool
    message: str = ""
    claimed: int = 0
    decided: int = 0
    undecidable: int = 0  # answers did not support a decision; page stays undecided
    failed: int = 0  # the endpoint gave no answer
    stale: int = 0  # the page changed (or was decided elsewhere) before the write
    skipped: int = 0  # claimed, then not asked because the endpoint tripped
    spec_invalid: bool = False
    capped: bool = False  # the hourly cap stopped (or prevented) the pass
    backing_off: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DispositionPass:
    def __init__(
        self,
        store: KnowledgeStore,
        client: Optional[DecisionClient],
        min_age_s: float = MIN_AGE_S,
        concurrency: int = 4,
        max_pages_per_hour: int = DEFAULT_MAX_PAGES_PER_HOUR,
        backoff_base_s: float = 120.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._store = store
        self._client = client
        self._min_age_s = min_age_s
        self._concurrency = max(1, concurrency)
        self._max_per_hour = max(1, int(max_pages_per_hour))
        self._backoff_base_s = backoff_base_s
        self._clock = clock
        self._cooldown: dict[str, float] = {}
        self._asked: deque[float] = deque()  # when each recent ask began
        self._fail_streak = 0
        self._next_probe_at = 0.0
        self._spec_error: Optional[str] = None

    @classmethod
    def from_env(cls, engine: Any) -> "DispositionPass":
        return cls(
            KnowledgeStore(engine),
            DecisionClient.from_env(),
            concurrency=int(_env_number("TRELLION_DECISION_CONCURRENCY", 4, 1, 16)),
            max_pages_per_hour=int(_env_number(
                "TRELLION_DECISION_MAX_PAGES_PER_HOUR", DEFAULT_MAX_PAGES_PER_HOUR, 1, 100000)),
        )

    @property
    def endpoint_url(self) -> Optional[str]:
        return getattr(self._client, "url", None)

    @property
    def configured(self) -> bool:
        return self._client is not None

    def _cooling(self) -> set[str]:
        now = self._clock()
        self._cooldown = {p: t for p, t in self._cooldown.items() if t > now}
        return set(self._cooldown)

    def _cool(self, page_id: str) -> None:
        if len(self._cooldown) >= _COOLDOWN_MAX:
            self._cooldown.clear()
        self._cooldown[page_id] = self._clock() + _COOLDOWN_S

    def _hour_budget(self) -> int:
        now = self._clock()
        while self._asked and self._asked[0] <= now - 3600.0:
            self._asked.popleft()
        return max(0, self._max_per_hour - len(self._asked))

    async def run_once(self, limit: int = 32) -> PassReport:
        """Decide up to ``limit`` undecided pages. Never raises on an endpoint failure."""
        if self._client is None:
            return PassReport(configured=False, message=NO_ENDPOINT)
        report = PassReport(configured=True)
        try:
            spec = await self._store.load_active_spec()
            self._spec_error = None
        except StoredSpecError as exc:
            self._spec_error = str(exc)
            report.spec_invalid = True
            report.message = "active spec is invalid; pass skipped"
            log.warning("knowledge_disposition.spec_invalid", error=self._spec_error)
            return report
        if self._clock() < self._next_probe_at:
            report.backing_off = True
            report.message = "decision endpoint failing; backing off"
            return report
        remaining = max(0, int(limit))
        if self._fail_streak:
            remaining = min(remaining, 1)  # one probe page until it answers again
        budget = self._hour_budget()
        if budget < remaining:
            report.capped = True
            remaining = budget
        if remaining <= 0:
            report.message = "hourly page cap reached"
            return report
        failed_ids: set[str] = self._cooling()
        ok = 0  # requests the endpoint answered (even if the answer was unusable)
        tripped = False
        while remaining > 0 and not tripped:
            batch = await self._store.claim_undecided_pages(
                min(_BATCH_SIZE, remaining), self._min_age_s, skip=failed_ids
            )
            try:
                rows = batch.rows
                if not rows:
                    break
                report.claimed += len(rows)
                remaining -= len(rows)
                sem = asyncio.Semaphore(self._concurrency)

                async def one(row: dict) -> str:
                    nonlocal tripped, ok
                    async with sem:
                        if tripped:
                            return "skipped"
                        self._asked.append(self._clock())
                        outcome = await self._decide_one(spec, row)
                        if outcome != "failed":
                            ok += 1
                        elif ok == 0:
                            tripped = True  # nothing has answered yet: stop sending
                        return outcome

                outcomes = await asyncio.gather(*(one(r) for r in rows))
            finally:
                batch.release()
            for row, outcome in zip(rows, outcomes):
                if outcome == "skipped":
                    report.skipped += 1
                    continue
                if outcome == "decided":
                    report.decided += 1
                elif outcome == "stale":
                    report.stale += 1
                elif outcome == "undecidable":
                    report.undecidable += 1
                else:
                    report.failed += 1
                if outcome in ("undecidable", "failed"):
                    failed_ids.add(row["page_id"])
                    self._cool(row["page_id"])
        if report.failed and ok == 0:
            self._fail_streak += 1
            delay = min(self._backoff_base_s * 2 ** (self._fail_streak - 1), BACKOFF_MAX_S)
            self._next_probe_at = self._clock() + delay
            report.message = "decision endpoint unavailable; backing off"
        elif ok:
            self._fail_streak = 0
            self._next_probe_at = 0.0
        if remaining <= 0 and report.capped:
            report.message = report.message or "hourly page cap reached"
        log.info("knowledge_disposition.pass", **report.to_dict())
        return report

    async def _decide_one(self, spec: KnowledgeSpec, row: dict) -> str:
        try:
            answers = await self._client.ask(spec, row)  # type: ignore[union-attr]
        except DecisionError as exc:
            log.debug("knowledge_disposition.ask_failed", page_id=row["page_id"], error=str(exc))
            return "failed"
        except Exception as exc:  # a client bug must not end the pass
            log.warning("knowledge_disposition.ask_error", error=type(exc).__name__)
            return "failed"
        disposition = decide(spec, answers)
        if disposition is None:
            return "undecidable"
        try:
            written = await self._store.write_disposition(
                page_id=row["page_id"],
                seen=row["seen"],
                spec_version=disposition.spec_version,
                would_embed=disposition.would_embed,
                would_graph=disposition.would_graph,
                held=disposition.held,
                thresholds=spec.thresholds_dict(),
                spec_hash=spec.body_hash(),
                question_ids=spec.question_ids(),
                answers=answers,
                model=self._client.model,  # type: ignore[union-attr]
                review_after=disposition.review_after,
            )
        except (ValueError, TypeError) as exc:  # answers not storable as JSON
            log.debug("knowledge_disposition.unstorable", page_id=row["page_id"], error=type(exc).__name__)
            return "undecidable"
        return "decided" if written else "stale"

    async def summary(self) -> dict[str, Any]:
        out = await self._store.disposition_summary()
        try:
            out["active_spec_version"] = (await self._store.load_active_spec()).version
            out["active_spec_error"] = None
        except StoredSpecError as exc:
            out["active_spec_version"] = None
            out["active_spec_error"] = str(exc)
        out["endpoint_configured"] = self.configured
        out["hourly_page_cap"] = self._max_per_hour
        out["asks_in_last_hour"] = self._max_per_hour - self._hour_budget()
        out["endpoint_failing"] = self._fail_streak > 0
        if not self.configured:
            out["message"] = NO_ENDPOINT
        return out

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()


class DispositionWorker:
    """Runs :meth:`DispositionPass.run_once` on an interval until stopped.

    Same ``start`` / ``stop`` shape as the other backlog loops.
    """

    def __init__(self, dpass: DispositionPass, interval_s: float = 120.0, batch: int = 32) -> None:
        self._pass = dpass
        self._interval_s = interval_s
        self._batch = batch
        self._task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()

    @classmethod
    def from_env(cls, engine: Any) -> Optional["DispositionWorker"]:
        """A worker, or None when no decision endpoint is configured."""
        dpass = DispositionPass.from_env(engine)
        if not dpass.configured:
            return None
        return cls(
            dpass,
            interval_s=_env_number("TRELLION_DECISION_INTERVAL_S", 120.0, 5.0, 86400.0),
            batch=int(_env_number("TRELLION_DECISION_BATCH", 32, 1, 500)),
        )

    async def start(self) -> bool:
        if self._task is not None and not self._task.done():
            return False
        host = non_private_host(self._pass.endpoint_url or "")
        if host:
            log.warning(
                "knowledge_disposition.endpoint_not_private",
                host=host,
                note="page titles, source names, headers and text are sent to this host; "
                     "it should be a service you run yourself",
            )
        self._stop.clear()
        self._task = asyncio.create_task(self._loop())
        return True

    async def stop(self) -> None:
        self._stop.set()
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        await self._pass.aclose()

    async def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                report = await self._pass.run_once(self._batch)
                busy = report.decided > 0
            except Exception as exc:
                log.warning("knowledge_disposition.pass_failed", error=type(exc).__name__)
                busy = False
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=1.0 if busy else self._interval_s)
            except asyncio.TimeoutError:
                pass
