# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""LLM calls back off, and stop entirely while a model is known absent.

When no node serves the model, the extraction loop must not re-select the
same pages every tick and ask again without backoff.

These drive the REAL ``EntityExtractor`` (and its real ``_loop``) through
``httpx.MockTransport`` (the transport is faked, never the call) with a fake
store, a fake clock and a recording sleep. Nothing here imports ``noomesh``.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from structlog.testing import capture_logs

from trellion.live.graph_extractor import EntityExtractor

MODEL = "google/gemma-4-12B-it-qat-w4a16-ct"
NOT_FOUND = {"error": {"code": "model_not_found", "message": f"no node serves {MODEL}"}}
GOOD = {"choices": [{"message": {"content": '[{"name": "krebs cycle", "type": "concept", "quote": "q"}]'}}]}


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class Sleeps:
    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, d: float) -> None:
        self.delays.append(d)
        await asyncio.sleep(0)


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def sleeps() -> Sleeps:
    return Sleeps()


@pytest.fixture(autouse=True)
def _fresh_throttle():
    try:
        from trellion.live.llm_throttle import reset_llm_throttle
    except ImportError:  # pre-change tree: nothing to reset
        yield
        return
    reset_llm_throttle()
    yield
    reset_llm_throttle()


def _install(clock: Clock, sleeps: Sleeps, **kw):
    """Build + install the process-wide throttle with a fake clock/sleep."""
    from trellion.live.llm_throttle import LLMThrottle, set_llm_throttle

    kw.setdefault("backoff_base_s", 2.0)
    t = LLMThrottle(
        startup_jitter_s=0, sleep=sleeps, clock=clock, rng=__import__("random").Random(7), **kw
    )
    set_llm_throttle(t)
    return t


def _try_install(clock: Clock, sleeps: Sleeps, **kw):
    try:
        return _install(clock, sleeps, **kw)
    except ImportError:
        return None  # old tree: the extractor uses its own un-throttled path


class FakeStore:
    def __init__(self, n: int) -> None:
        self.pending = {f"p{i}" for i in range(n)}
        self.extracted: set[str] = set()
        self.claimed: set[str] = set()
        self.claims = 0

    async def get_page(self, page_id):
        return {"page_id": page_id, "doc_id": "d0", "page_no": 1, "text": "The Krebs cycle."}

    async def claim_page_for_extraction(self, page_id, stale_after_minutes=15):
        self.claims += 1
        self.claimed.add(page_id)
        return True

    async def release_extraction_claim(self, page_id):
        self.claimed.discard(page_id)

    async def mark_page_extracted(self, page_id):
        self.extracted.add(page_id)
        self.pending.discard(page_id)
        self.claimed.discard(page_id)

    async def upsert_entity(self, canonical_name, type, aliases=None):
        return "e-" + canonical_name

    async def insert_mention(self, entity_id, page_id, span=None, quote=None):
        return "m"


class Gateway:
    """Scripted gateway: ``respond(call_no, model)`` -> httpx.Response."""

    def __init__(self, respond) -> None:
        self.respond = respond
        self.calls: list[str] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        model = json.loads(request.content)["model"]
        self.calls.append(model)
        return self.respond(len(self.calls), model)


def _extractor(store, gw: Gateway, model: str = MODEL) -> EntityExtractor:
    ex = EntityExtractor(store=store, llm_url="http://gw", model=model)
    ex._http = httpx.AsyncClient(transport=httpx.MockTransport(gw.handler))

    async def pending(limit: int = 50):
        return [{"page_id": p, "text": "The Krebs cycle."} for p in sorted(store.pending - store.claimed)][:limit]

    ex._get_unextracted_pages = pending  # the SQL is not what is under test
    return ex


async def _run_loop(ex: EntityExtractor, until, *, timeout: float = 3.0) -> None:
    ex._running = True
    task = asyncio.create_task(ex._loop(0.001))
    try:
        end = asyncio.get_running_loop().time() + timeout
        while not until() and asyncio.get_running_loop().time() < end:
            await asyncio.sleep(0.01)
    finally:
        ex._running = False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


def _events(logs, name):
    return [e for e in logs if e["event"] == name]


def _no_tracebacks(logs):
    bad = [e for e in logs if e.get("exc_info") or e["event"] == "entity_extraction.llm_error"]
    assert not bad, bad


async def test_a_burst_of_404s_opens_the_breaker_and_stops_calls(clock, sleeps):
    _try_install(clock, sleeps)
    store = FakeStore(50)
    gw = Gateway(lambda n, m: httpx.Response(404, json=NOT_FOUND))
    ex = _extractor(store, gw)
    with capture_logs() as logs:
        await _run_loop(ex, until=lambda: False, timeout=0.5)
    # The old loop asked once per pending page, every tick (hundreds in 0.5 s).
    assert len(gw.calls) <= 2, f"{len(gw.calls)} calls reached the gateway"
    assert store.extracted == set(), "no page may be marked done on a failed call"
    assert store.claimed == set(), "claims were handed back"
    _no_tracebacks(logs)


async def test_one_warning_per_open_not_per_call(clock, sleeps):
    t = _install(clock, sleeps)
    store = FakeStore(30)
    gw = Gateway(lambda n, m: httpx.Response(404, json=NOT_FOUND))
    ex = _extractor(store, gw)
    with capture_logs() as logs:
        await _run_loop(ex, until=lambda: False, timeout=0.3)
        # Two failed probes later it is still ONE warning (and one open).
        for _ in range(2):
            clock.now += 700
            await _run_loop(ex, until=lambda: False, timeout=0.2)
    opens = _events(logs, "llm_breaker_open")
    assert len(opens) == 1, [e["event"] for e in logs]
    assert opens[0]["model"] == MODEL and opens[0]["log_level"] == "warning"
    assert t.is_open(MODEL)
    assert len(gw.calls) == 3  # first call + one probe per cool-down
    _no_tracebacks(logs)


async def test_pages_stay_unextracted_then_are_picked_up_after_it_closes(clock, sleeps):
    t = _install(clock, sleeps)
    store = FakeStore(8)
    state = {"up": False}

    def respond(n, m):
        return httpx.Response(200, json=GOOD) if state["up"] else httpx.Response(404, json=NOT_FOUND)

    gw = Gateway(respond)
    ex = _extractor(store, gw)
    with capture_logs() as logs:
        await _run_loop(ex, until=lambda: False, timeout=0.3)
        assert store.extracted == set() and len(store.pending) == 8
        calls_while_down = len(gw.calls)
        await _run_loop(ex, until=lambda: False, timeout=0.2)
        assert len(gw.calls) == calls_while_down, "no probe before the cool-down"
        # The model comes back; the cool-down elapses; the first probe closes it.
        state["up"] = True
        clock.now += 31
        await _run_loop(ex, until=lambda: not store.pending)
    assert store.extracted == {f"p{i}" for i in range(8)}
    assert not t.is_open(MODEL)
    closed = _events(logs, "llm_breaker_closed")
    assert len(closed) == 1 and closed[0]["log_level"] == "info"
    assert closed[0]["model"] == MODEL and closed[0]["open_for_s"] >= 31
    assert closed[0]["deferred"] >= 1
    _no_tracebacks(logs)


async def test_probe_cooldown_doubles_to_a_cap(clock, sleeps):
    t = _install(clock, sleeps, cooldown_s=30, cooldown_cap_s=100)
    gw = Gateway(lambda n, m: httpx.Response(404, json=NOT_FOUND))
    ex = _extractor(FakeStore(1), gw)
    await ex.extract_entities("p0", "text")
    assert t.blocked_for(MODEL) == pytest.approx(30)
    seen = []
    for _ in range(3):
        clock.now += t.blocked_for(MODEL) + 0.1
        await ex.extract_entities("p0", "text")  # the probe fails
        seen.append(round(t.blocked_for(MODEL)))
    assert seen == [60, 100, 100]


async def test_503s_back_off_honour_retry_after_and_retry(clock, sleeps):
    _install(clock, sleeps, backoff_base_s=2.0)
    store = FakeStore(1)

    def respond(n, m):
        if n == 1:
            return httpx.Response(503, headers={"retry-after": "7"}, json={"error": "busy"})
        if n == 2:
            return httpx.Response(502, json={"error": "bad gateway"})
        return httpx.Response(200, json=GOOD)

    gw = Gateway(respond)
    ex = _extractor(store, gw)
    with capture_logs() as logs:
        ids = await ex.process_page("p0")
    assert len(gw.calls) == 3 and ids == ["e-krebs cycle"] and store.extracted == {"p0"}
    assert len(sleeps.delays) == 2
    assert sleeps.delays[0] >= 7, "Retry-After honoured"
    assert 1.0 <= sleeps.delays[1] <= 4.0, "exponential backoff with jitter"
    _no_tracebacks(logs)


async def test_consecutive_502s_open_the_breaker_after_n(clock, sleeps):
    t = _install(clock, sleeps, failure_threshold=5, max_attempts=3)
    store = FakeStore(20)
    gw = Gateway(lambda n, m: httpx.Response(502, json={"error": "backend_unavailable"}))
    ex = _extractor(store, gw)
    with capture_logs() as logs:
        await _run_loop(ex, until=lambda: False, timeout=0.5)
    assert t.is_open(MODEL)
    assert len(gw.calls) == 5, "exactly the threshold, then silence"
    assert store.extracted == set() and store.claimed == set()
    assert len(_events(logs, "llm_breaker_open")) == 1
    assert not [e for e in logs if e["event"] == "entity_extraction.llm_error"]
    _no_tracebacks(logs)


async def test_a_different_model_is_unaffected(clock, sleeps):
    t = _install(clock, sleeps)

    def respond(n, m):
        return httpx.Response(404, json=NOT_FOUND) if m == MODEL else httpx.Response(200, json=GOOD)

    gw = Gateway(respond)
    down = _extractor(FakeStore(1), gw, model=MODEL)
    up_store = FakeStore(3)
    up = _extractor(up_store, gw, model="other-model")
    await down.extract_entities("p0", "text")
    assert t.is_open(MODEL) and not t.is_open("other-model")
    before = len(gw.calls)
    for p in sorted(up_store.pending):
        assert await up.process_page(p)
    assert len(gw.calls) == before + 3
    assert up_store.extracted == {"p0", "p1", "p2"}


async def test_unexpected_errors_keep_their_traceback_and_do_not_trip(clock, sleeps):
    from trellion.live.llm_throttle import LLMThrottle

    t = _install(clock, sleeps)
    gw = Gateway(lambda n, m: httpx.Response(400, json={"error": "bad request"}))
    ex = _extractor(FakeStore(1), gw)
    with capture_logs() as logs:
        assert await ex.extract_entities("p0", "text") is None
    assert [e["event"] for e in logs] == ["entity_extraction.llm_error"]  # as before
    assert not t.is_open(MODEL) and len(gw.calls) == 1

    async def boom():
        raise ValueError("a bug")

    with pytest.raises(ValueError):
        await LLMThrottle(sleep=sleeps, clock=clock).call(MODEL, boom, what="x")


async def test_a_bare_404_is_a_wrong_url_not_an_absent_model(clock, sleeps):
    t = _install(clock, sleeps)
    gw = Gateway(lambda n, m: httpx.Response(404, text="not found"))
    ex = _extractor(FakeStore(1), gw)
    await ex.extract_entities("p0", "text")
    assert not t.is_open(MODEL)


async def test_the_in_flight_bound_holds_and_interactive_calls_bypass_it(clock, sleeps):
    t = _install(clock, sleeps, max_inflight=2)
    gate = asyncio.Event()

    async def slow():
        await gate.wait()
        return "ok"

    batch = [asyncio.create_task(t.call("m", slow, what="bulk")) for _ in range(6)]
    await asyncio.sleep(0.05)
    assert t.peak_inflight == 2
    # A user waiting on an answer is not queued behind the backfill.
    assert await t.call("m", _ok, what="interactive", bounded=False) == "ok"
    gate.set()
    assert await asyncio.gather(*batch) == ["ok"] * 6
    assert t.peak_inflight == 2


async def _ok():
    return "ok"


async def test_the_build_path_pauses_on_an_open_breaker_instead_of_hammering(
    clock, sleeps, monkeypatch
):
    openai = pytest.importorskip("openai")
    from trellion.common.config import LLMConfig
    from trellion.common.llm_client import LLMClient
    from trellion.live.llm_throttle import LLMUnavailable

    t = _install(clock, sleeps)
    monkeypatch.setenv("TRELLION_LLM_BUILD_WAIT_S", "0")
    calls = {"n": 0}

    class Completions:
        async def create(self, **kw):
            calls["n"] += 1
            req = httpx.Request("POST", "http://gw/v1/chat/completions")
            raise openai.NotFoundError(
                "model_not_found", response=httpx.Response(404, json=NOT_FOUND, request=req), body=NOT_FOUND
            )

    class Chat:
        completions = Completions()

    llm = LLMClient(LLMConfig(model=MODEL))
    llm._client.chat = Chat()
    results = await asyncio.gather(
        *(llm.complete(f"chunk {i}") for i in range(40)), return_exceptions=True
    )
    assert all(isinstance(r, LLMUnavailable) for r in results)
    assert calls["n"] == 1, f"{calls['n']} calls for 40 chunks"
    assert t.is_open(MODEL)
