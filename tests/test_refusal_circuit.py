# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""A refused surface is not asked again every call.

The callers loop per item, so the shared auth hook closes a surface locally
after a refusal — 1 s doubling to 60 s — and the first success reopens it.

Tests drive the REAL client through an ``httpx.MockTransport`` (never a fake
``httpx.post``: see test_addon_token_exchange.py for why) and a controllable
clock.
"""

from __future__ import annotations

import httpx
import pytest

from trellion.live import gateway_auth


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


@pytest.fixture(autouse=True)
def _clean(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(gateway_auth, "_circuit", {}, raising=False)
    monkeypatch.setattr(gateway_auth, "_token", None, raising=False)
    monkeypatch.setattr(gateway_auth, "_next_attempt", 0.0, raising=False)
    # The key itself travels: no exchange inside the window (the door is not
    # what these test), and there is no HMAC lane to fall back to any more.
    monkeypatch.setattr(gateway_auth, "_next_attempt", float("inf"), raising=False)
    monkeypatch.setenv("NOO_GATEWAY_KEY", "noo-sk-the-key")
    clock = Clock()
    monkeypatch.setattr(gateway_auth.time, "monotonic", clock)
    yield clock


def _client(status_for_call):
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(status_for_call(len(calls)), json={})

    client = httpx.Client(auth=gateway_auth.GatewayHmacAuth(), transport=httpx.MockTransport(handler),
                          base_url="https://gateway:8080")
    return client, calls


def test_a_refused_surface_is_not_asked_again_inside_the_window(_clean: Clock) -> None:
    client, calls = _client(lambda n: 403)
    assert client.post("/api/external/embeddings", json={}).status_code == 403
    for _ in range(50):
        with pytest.raises(gateway_auth.GatewayRefused) as e:
            client.post("/api/external/embeddings", json={})
        assert e.value.response.status_code == 403, "the local refusal carries the real status"
    assert len(calls) == 1, f"{len(calls)} requests reached the gateway; the circuit let them through"


def test_the_window_doubles_and_caps_at_a_minute(_clean: Clock) -> None:
    client, calls = _client(lambda n: 403)
    client.post("/api/external/embeddings", json={})
    windows = []
    for _ in range(9):
        until, backoff, _status = gateway_auth._circuit["gateway:8080/api/external/embeddings"]
        windows.append(backoff)
        _clean.t = until + 0.001  # just past the window: one real attempt
        client.post("/api/external/embeddings", json={})
    assert windows[:4] == [1.0, 2.0, 4.0, 8.0]
    assert max(windows) == 60.0 and windows[-1] == 60.0
    assert len(calls) == 10


def test_the_first_success_reopens_the_surface(_clean: Clock) -> None:
    client, calls = _client(lambda n: 403 if n == 1 else 200)
    client.post("/api/external/embeddings", json={})
    _clean.t += 1.5
    assert client.post("/api/external/embeddings", json={}).status_code == 200
    assert "gateway:8080/api/external/embeddings" not in gateway_auth._circuit
    assert client.post("/api/external/embeddings", json={}).status_code == 200
    assert len(calls) == 3


def test_one_surface_refusing_does_not_close_another(_clean: Clock) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403 if request.url.path.endswith("embeddings") else 200, json={})

    client = httpx.Client(auth=gateway_auth.GatewayHmacAuth(), transport=httpx.MockTransport(handler),
                          base_url="https://gateway:8080")
    client.post("/api/external/embeddings", json={})
    assert client.post("/v1/chat/completions", json={}).status_code == 200


def test_server_errors_and_not_found_do_not_trip_it(_clean: Clock) -> None:
    for status in (404, 500, 502):
        client, calls = _client(lambda n, s=status: s)
        client.post("/api/external/embeddings", json={})
        client.post("/api/external/embeddings", json={})
        assert len(calls) == 2, f"{status} closed the circuit"


def test_the_embedder_uses_the_shared_hook() -> None:
    """The 2026-09-22 storm path built its header by hand and so had neither
    the key fallback nor the circuit. Pin that it rides the hook now."""
    import inspect

    from trellion.live import pageindex_embedder

    src = inspect.getsource(pageindex_embedder)
    assert "auth=GatewayHmacAuth()" in src
    assert "headers=self._gateway_auth_header()" not in src
