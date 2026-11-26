# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""The disposition pass (slice 1, shadow mode).

For each page that has no disposition yet, ask the decision endpoint the
active spec's questions, apply :func:`decide` and RECORD what would be done:
embed the page, extract entities from it, or hold it. Nothing reads the record
to change behaviour. The pass never writes ``pages``, embeddings or extraction
state; it only reads them and writes ``page_dispositions``.

With no endpoint configured (``TRELLION_DECISION_URL`` unset) the pass is inert.

Other knobs: ``TRELLION_DECISION_INTERVAL_S`` (idle sleep between passes,
default 120), ``TRELLION_DECISION_BATCH`` (pages per pass, default 32),
``TRELLION_DECISION_MIN_AGE_S`` (a document is left alone this long after it
was created, so its audience is settled first; default 300),
``TRELLION_DECISION_CONCURRENCY`` (requests in flight, default 4).
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import asdict, dataclass
from typing import Any, Optional

import structlog

from trellion.live.decision_client import DecisionClient, DecisionError, _env_number
from trellion.live.knowledge_spec import KnowledgeSpec, decide
from trellion.live.knowledge_store import KnowledgeStore

log = structlog.get_logger(__name__)

NO_ENDPOINT = "no decision endpoint configured"
#: A batch in which every request failed ends the pass: the endpoint is down.
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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DispositionPass:
    def __init__(
        self,
        store: KnowledgeStore,
        client: Optional[DecisionClient],
        min_age_s: float = 300.0,
        concurrency: int = 4,
    ) -> None:
        self._store = store
        self._client = client
        self._min_age_s = min_age_s
        self._concurrency = max(1, concurrency)
        self._cooldown: dict[str, float] = {}

    def _cooling(self) -> set[str]:
        now = time.monotonic()
        self._cooldown = {p: t for p, t in self._cooldown.items() if t > now}
        return set(self._cooldown)

    def _cool(self, page_id: str) -> None:
        if len(self._cooldown) >= _COOLDOWN_MAX:
            self._cooldown.clear()
        self._cooldown[page_id] = time.monotonic() + _COOLDOWN_S

    @classmethod
    def from_env(cls, engine: Any) -> "DispositionPass":
        return cls(
            KnowledgeStore(engine),
            DecisionClient.from_env(),
            min_age_s=_env_number("TRELLION_DECISION_MIN_AGE_S", 300.0, 0.0, 86400.0),
            concurrency=int(_env_number("TRELLION_DECISION_CONCURRENCY", 4, 1, 16)),
        )

    @property
    def configured(self) -> bool:
        return self._client is not None

    async def run_once(self, limit: int = 32) -> PassReport:
        """Decide up to ``limit`` undecided pages. Never raises on an endpoint failure."""
        if self._client is None:
            return PassReport(configured=False, message=NO_ENDPOINT)
        report = PassReport(configured=True)
        spec = await self._store.load_active_spec()
        failed_ids: set[str] = self._cooling()
        remaining = max(0, int(limit))
        while remaining > 0:
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
                    async with sem:
                        return await self._decide_one(spec, row)

                outcomes = await asyncio.gather(*(one(r) for r in rows))
            finally:
                batch.release()
            for row, outcome in zip(rows, outcomes):
                if outcome == "decided":
                    report.decided += 1
                elif outcome == "stale":
                    report.stale += 1
                elif outcome == "undecidable":
                    report.undecidable += 1
                    failed_ids.add(row["page_id"])
                    self._cool(row["page_id"])
                else:
                    report.failed += 1
                    failed_ids.add(row["page_id"])
                    self._cool(row["page_id"])
            if all(o == "failed" for o in outcomes):
                report.message = "decision endpoint unavailable; pass ended early"
                break
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
        spec = await self._store.load_active_spec()
        out["active_spec_version"] = spec.version
        out["endpoint_configured"] = self.configured
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
