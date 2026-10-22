# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""Background embedding opts in to the gateway's bulk job route (noomesh i179 D).

The transport is faked (``httpx.MockTransport`` playing the gateway), the code
under test is real. No database.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest

from trellion.live import embed_bulk, pageindex_embedder
from trellion.live.embed_throttle import EmbedThrottle, EmbedUnavailable
from trellion.live.pageindex_embedder import PageIndexEmbedder

DIM = 4


def vec_for(text: str) -> list[float]:
    return [float(len(text)), float(sum(map(ord, text)) % 97), 0.0, 1.0]


class JobGateway:
    """Serves the sync embeddings route and the bulk job routes."""

    def __init__(self) -> None:
        self.sync: list[dict] = []
        self.submits: list[dict] = []
        self.polls: list[str] = []
        self.jobs: dict[str, dict] = {}
        self.pending_polls = 1  # polls answered "queued"/"running" first
        self.submit_status: int | None = None  # force a status on submit
        self.submit_headers: dict = {}
        self.final: str = "completed"
        self.expire_first: set[int] = set()  # submit ordinals whose first read is 410
        self.vectors_drop = 0  # return this many fewer vectors
        self.model = "m-served"
        self.max_bytes: int | None = None
        self._n = 0

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.method == "POST" and path == "/api/external/embeddings":
            body = json.loads(request.content)
            self.sync.append(body)
            return httpx.Response(200, json={
                "model": self.model,
                "data": [{"index": i, "embedding": vec_for(t)} for i, t in enumerate(body["input"])],
            })
        if request.method == "POST" and path == "/api/external/embeddings/jobs":
            body = json.loads(request.content)
            if self.submit_status:
                return httpx.Response(self.submit_status, headers=self.submit_headers)
            if self.max_bytes is not None and sum(len(t) for t in body["input"]) > self.max_bytes:
                return httpx.Response(413)
            self.submits.append(body)
            self._n += 1
            jid = f"job{self._n}"
            self.jobs[jid] = {"input": body["input"], "polls": 0, "ordinal": self._n}
            return httpx.Response(202, json={"job_id": jid, "path": "pull", "model": self.model})
        if request.method == "GET" and path.startswith("/api/external/embeddings/jobs/"):
            jid = path.rsplit("/", 1)[1]
            self.polls.append(jid)
            job = self.jobs[jid]
            job["polls"] += 1
            if job["polls"] <= self.pending_polls:
                return httpx.Response(200, json={"status": "queued" if job["polls"] == 1 else "running"})
            if self.final != "completed":
                return httpx.Response(200, json={"status": self.final, "error": "refused: boom"})
            if job["ordinal"] in self.expire_first:
                self.expire_first.discard(job["ordinal"])
                return httpx.Response(410, json={"error": "gone"})
            vs = [vec_for(t) for t in job["input"]]
            if self.vectors_drop:
                vs = vs[: -self.vectors_drop]
            return httpx.Response(200, json={
                "status": "completed", "model": self.model, "n": len(vs), "dims": DIM,
                "error": "", "vectors": vs,
            })
        return httpx.Response(404)


def make(gw: JobGateway) -> PageIndexEmbedder:
    th = EmbedThrottle(
        max_inflight=1, batch_max=4, batch_min=4, pace_s=0, startup_jitter_s=0,
        backoff_base_s=0.0, backoff_cap_s=0.0,
        sleep=_nosleep,
    )
    e = PageIndexEmbedder(SimpleNamespace(), embedding_url="https://gateway:8080", throttle=th)
    e._client = httpx.AsyncClient(transport=httpx.MockTransport(gw.handler))
    return e


async def _nosleep(_s: float) -> None:
    return None


@pytest.fixture(autouse=True)
def _fast(monkeypatch):
    monkeypatch.delenv("TRELLION_EMBEDDING_SPACE", raising=False)
    monkeypatch.delenv(embed_bulk.FLAG, raising=False)
    real = embed_bulk.embed_bulk

    async def fast(*a, **kw):
        kw.setdefault("sleep", _nosleep)
        return await real(*a, **kw)

    monkeypatch.setattr(embed_bulk, "embed_bulk", fast)
    yield


def on(monkeypatch):
    monkeypatch.setenv(embed_bulk.FLAG, "1")


async def test_flag_off_uses_the_sync_route():
    gw = JobGateway()
    out = await make(gw).embed_bulk(["a", "bb"])
    assert len(gw.sync) == 1 and not gw.submits and not gw.polls
    assert out == [vec_for("a"), vec_for("bb")] and out.model == "m-served"


async def test_flag_on_submits_polls_and_returns_vectors_in_order(monkeypatch):
    on(monkeypatch)
    gw = JobGateway()
    out = await make(gw).embed_bulk(["a", "bb", "ccc"])
    assert not gw.sync
    assert gw.submits == [{"input": ["a", "bb", "ccc"], "space": "knowledge"}]
    assert len(gw.polls) == 2  # queued/running, then completed
    assert out == [vec_for(t) for t in ("a", "bb", "ccc")]
    assert out.model == "m-served" and out[0].model == "m-served"


async def test_next_alias_is_sent_as_the_space(monkeypatch):
    on(monkeypatch)
    gw = JobGateway()
    await make(gw)._embed_next_bulk(["a"])
    assert gw.submits[0]["space"] == "knowledge@next"


async def test_more_than_64_inputs_are_chunked_in_order(monkeypatch):
    on(monkeypatch)
    gw = JobGateway()
    texts = [f"t{i}" for i in range(150)]
    out = await make(gw).embed_bulk(texts)
    assert [len(s["input"]) for s in gw.submits] == [64, 64, 22]
    assert out == [vec_for(t) for t in texts]


async def test_413_halves_the_job_and_keeps_order(monkeypatch):
    on(monkeypatch)
    gw = JobGateway()
    gw.max_bytes = 6
    texts = ["aaa", "bbb", "ccc", "ddd"]
    out = await make(gw).embed_bulk(texts)
    assert out == [vec_for(t) for t in texts]
    assert all(len(s["input"]) == 2 for s in gw.submits)


@pytest.mark.parametrize("status", [404, 405])
async def test_old_gateway_falls_back_to_sync_and_logs_once(monkeypatch, status):
    import structlog

    on(monkeypatch)
    gw = JobGateway()
    gw.submit_status = status
    e = make(gw)
    with structlog.testing.capture_logs() as logs:
        a = await e.embed_bulk(["a"])
        b = await e.embed_bulk(["b"])
    assert a == [vec_for("a")] and b == [vec_for("b")]
    assert len(gw.sync) == 2
    assert [x["event"] for x in logs].count("embed_bulk.gateway_without_job_route") == 1


@pytest.mark.parametrize("status", [429, 503])
async def test_overload_raises_embed_unavailable_with_retry_after(monkeypatch, status):
    on(monkeypatch)
    gw = JobGateway()
    gw.submit_status = status
    gw.submit_headers = {"Retry-After": "7"}
    e = make(gw)
    with pytest.raises(EmbedUnavailable) as ei:
        await e.embed_bulk(["a"])
    assert ei.value.reason == f"http_{status}" and ei.value.retry_after == 7.0
    assert e.throttle.consecutive_failures > 0


async def test_a_failed_job_is_not_an_outage(monkeypatch):
    on(monkeypatch)
    gw = JobGateway()
    gw.final = "dead"
    with pytest.raises(embed_bulk.BulkJobFailed) as ei:
        await make(gw).embed_bulk(["a"])
    assert not isinstance(ei.value, EmbedUnavailable) and "boom" in str(ei.value)


async def test_410_resubmits_once(monkeypatch):
    on(monkeypatch)
    gw = JobGateway()
    gw.expire_first = {1}
    out = await make(gw).embed_bulk(["a", "b"])
    assert len(gw.submits) == 2 and out == [vec_for("a"), vec_for("b")]


async def test_a_second_410_fails_the_job(monkeypatch):
    on(monkeypatch)
    gw = JobGateway()
    gw.expire_first = {1, 2}
    with pytest.raises(embed_bulk.BulkJobFailed):
        await make(gw).embed_bulk(["a"])
    assert len(gw.submits) == 2


async def test_vector_count_mismatch_is_refused(monkeypatch):
    on(monkeypatch)
    gw = JobGateway()
    gw.vectors_drop = 1
    with pytest.raises(embed_bulk.BulkJobFailed, match="vectors for"):
        await make(gw).embed_bulk(["a", "b"])


async def test_a_job_pending_past_the_deadline_is_an_outage(monkeypatch):
    on(monkeypatch)
    monkeypatch.setenv(embed_bulk.TIMEOUT_ENV, "0.001")
    gw = JobGateway()
    gw.pending_polls = 10**6
    with pytest.raises(EmbedUnavailable) as ei:
        await make(gw).embed_bulk(["a"])
    assert ei.value.reason.startswith("bulk_timeout")


async def test_query_time_paths_never_touch_the_jobs_route(monkeypatch):
    on(monkeypatch)
    gw = JobGateway()
    e = make(gw)
    await e.embed_text("q")
    await e.embed_batch(["q1", "q2"])
    assert len(gw.sync) == 2 and not gw.submits and not gw.polls


async def test_the_backlog_loop_and_the_reembed_are_wired_to_bulk(monkeypatch):
    on(monkeypatch)
    gw = JobGateway()
    e = make(gw)

    class Claim:
        rows = [{"page_id": "p1", "text": "hello"}, {"page_id": "p2", "text": "x"}]
        stored: list = []

        async def store(self, items):
            Claim.stored = items
            return len(items)

    class Ctx:
        async def __aenter__(self):
            return Claim()

        async def __aexit__(self, *a):
            return False

    e.store = SimpleNamespace(claim_unembedded_pages=lambda n: Ctx())
    assert await e._page_batch(32) == 2
    assert len(gw.submits) == 1 and not gw.sync

    seen = {}

    class W:
        def __init__(self, engine, embed_next, embed_live=None, **kw):
            seen.update(next=embed_next, live=embed_live)

    async def idle(self, interval):
        return None

    async def nosub(self):
        return None

    monkeypatch.setattr(pageindex_embedder, "ReembedWorker", W)
    monkeypatch.setattr(PageIndexEmbedder, "_run_loop", idle)
    monkeypatch.setattr(PageIndexEmbedder, "_subscribe_doc_parsed", nosub)
    e2 = make(gw)
    e2.store = SimpleNamespace(_engine=object())
    await e2.start()
    await e2._task
    assert seen["next"] == e2._embed_next_bulk and seen["live"] == e2.embed_bulk
    await e2._client.aclose()
    e2._client = None
    e2.state = pageindex_embedder.EmbedderState.IDLE
