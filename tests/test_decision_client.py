# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""The decision client: request building, and a local stub endpoint (no database)."""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from trellion.live.decision_client import (
    DecisionClient,
    DecisionError,
    build_state,
    endpoint_url,
)
from trellion.live.knowledge_spec import default_spec

SPEC = default_spec()
PAGE = {
    "page_id": "p1",
    "title": "Quarterly notes",
    "source_uri": "file:///docs/notes.md",
    "meta": {"headers": ["Intro", "Plan"], "char_count": 5000, "has_table": True, "has_figure": False},
    "text": "x" * 5000,
}


# ---- request building -----------------------------------------------------


def test_build_state_fields_and_truncation():
    s = build_state(PAGE, 100)
    assert s["title"] == "Quarterly notes" and s["kind"] == "md"
    assert s["headers"] == ["Intro", "Plan"]
    assert s["char_count"] == 5000 and s["has_table"] is True and s["has_figure"] is False
    assert len(s["text"]) == 100 and s["text_truncated"] is True
    short = build_state({**PAGE, "text": "abc", "meta": None}, 100)
    assert short["text"] == "abc" and short["text_truncated"] is False
    assert short["char_count"] == 3 and short["headers"] == []


def test_kind_falls_back_to_the_scheme():
    assert build_state({"source_uri": "https://example.com/page"}, 10)["kind"] == "https"
    assert build_state({"source_uri": None}, 10)["kind"] == "unknown"


def test_build_request_has_the_systemone_shape():
    client = DecisionClient("http://127.0.0.1:1/v1/systemone", model="decider-1", max_chars=50)
    req = client.build_request(SPEC, PAGE)
    assert req["model"] == "decider-1"
    assert set(req["questions"]) == {"embed", "graph", "usefulness"}
    assert req["questions"]["embed"]["type"] == "noul"
    assert len(req["state"]["text"]) == 50
    json.dumps(req)  # serialisable
    assert "model" not in DecisionClient("http://127.0.0.1:1/").build_request(SPEC, PAGE)


def test_endpoint_url_appends_the_default_path_to_a_bare_host():
    assert endpoint_url("http://127.0.0.1:9") == "http://127.0.0.1:9/v1/systemone"
    assert endpoint_url("http://127.0.0.1:9/") == "http://127.0.0.1:9/v1/systemone"
    assert endpoint_url("http://127.0.0.1:9/custom/path") == "http://127.0.0.1:9/custom/path"


def test_from_env_is_inert_without_a_url(monkeypatch):
    for k in ("TRELLION_DECISION_URL", "TRELLION_DECISION_MODEL", "TRELLION_DECISION_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    assert DecisionClient.from_env() is None
    monkeypatch.setenv("TRELLION_DECISION_URL", "   ")
    assert DecisionClient.from_env() is None


def test_from_env_reads_and_bounds_its_knobs(monkeypatch):
    monkeypatch.setenv("TRELLION_DECISION_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("TRELLION_DECISION_MODEL", "m")
    monkeypatch.setenv("TRELLION_DECISION_TIMEOUT_S", "100000")
    monkeypatch.setenv("TRELLION_DECISION_MAX_CHARS", "nonsense")
    c = DecisionClient.from_env()
    assert c.url.endswith("/v1/systemone") and c.model == "m"
    assert c.timeout_s == 120.0 and c.max_chars == 4000


# ---- a stub endpoint on 127.0.0.1 -----------------------------------------


class _Stub:
    """Serves scripted responses; records what it received."""

    def __init__(self, script):
        self.script = list(script)  # (status, body_bytes_or_obj, delay_s)
        self.seen: list[dict] = []
        stub = self

        class H(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                stub.seen.append({"path": self.path, "auth": self.headers.get("Authorization"),
                                  "body": json.loads(raw)})
                status, body, delay = stub.script.pop(0) if stub.script else (200, {"answers": {}}, 0)
                if delay:
                    time.sleep(delay)
                data = body if isinstance(body, bytes) else json.dumps(body).encode()
                try:
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                except OSError:
                    pass  # the client gave up (timeout test)

            def log_message(self, *a):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def stub():
    made: list[_Stub] = []

    def make(script):
        s = _Stub(script)
        made.append(s)
        return s

    yield make
    for s in made:
        s.close()


GOOD = {"model": "m", "answers": {"embed": 0.9, "graph": {"probability": 0.2}}, "usage": {}}


async def test_ask_posts_the_request_and_returns_the_answers(stub):
    s = stub([(200, GOOD, 0)])
    client = DecisionClient(s.url, model="m", api_key="sekret-value")
    try:
        answers = await client.ask(SPEC, PAGE)
    finally:
        await client.aclose()
    assert answers == GOOD["answers"]
    got = s.seen[0]
    assert got["path"] == "/v1/systemone" and got["auth"] == "Bearer sekret-value"
    assert got["body"]["model"] == "m" and "embed" in got["body"]["questions"]


async def test_non_200_is_an_error_and_a_client_error_is_not_retried(stub):
    s = stub([(400, {"error": "bad"}, 0), (200, GOOD, 0)])
    client = DecisionClient(s.url)
    try:
        with pytest.raises(DecisionError, match="HTTP 400"):
            await client.ask(SPEC, PAGE)
    finally:
        await client.aclose()
    assert len(s.seen) == 1


async def test_a_server_error_is_retried_once_then_gives_up(stub):
    s = stub([(503, {}, 0), (503, {}, 0), (200, GOOD, 0)])
    client = DecisionClient(s.url)
    try:
        with pytest.raises(DecisionError, match="HTTP 503"):
            await client.ask(SPEC, PAGE)
    finally:
        await client.aclose()
    assert len(s.seen) == 2


async def test_a_transient_error_then_success_returns_the_answers(stub):
    s = stub([(500, {}, 0), (200, GOOD, 0)])
    client = DecisionClient(s.url)
    try:
        assert await client.ask(SPEC, PAGE) == GOOD["answers"]
    finally:
        await client.aclose()
    assert len(s.seen) == 2


async def test_timeout_is_bounded_and_retried_once(stub):
    s = stub([(200, GOOD, 1.0), (200, GOOD, 1.0)])
    client = DecisionClient(s.url, timeout_s=0.2)
    t0 = time.monotonic()
    try:
        with pytest.raises(DecisionError, match="timeout"):
            await client.ask(SPEC, PAGE)
    finally:
        await client.aclose()
    assert time.monotonic() - t0 < 1.5
    assert len(s.seen) == 2


@pytest.mark.parametrize("body", [b"not json", b"[]", b'{"answers": []}', b'{"model": "m"}'])
async def test_garbage_200_is_an_error(stub, body):
    s = stub([(200, body, 0)])
    client = DecisionClient(s.url)
    try:
        with pytest.raises(DecisionError, match="no answers"):
            await client.ask(SPEC, PAGE)
    finally:
        await client.aclose()


async def test_connection_refused_is_an_error():
    client = DecisionClient("http://127.0.0.1:9", timeout_s=0.5)
    try:
        with pytest.raises(DecisionError, match="transport error"):
            await client.ask(SPEC, PAGE)
    finally:
        await client.aclose()


async def test_the_api_key_never_reaches_the_error_or_the_log(stub, capsys):
    s = stub([(500, {}, 0), (500, {}, 0)])
    client = DecisionClient(s.url, api_key="sekret-value")
    try:
        with pytest.raises(DecisionError) as exc:
            await client.ask(SPEC, PAGE)
    finally:
        await client.aclose()
    out = capsys.readouterr()
    assert "sekret-value" not in str(exc.value) + out.out + out.err
