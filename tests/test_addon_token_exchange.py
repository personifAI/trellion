# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""An add-on trades its key for a short-lived token, and degrades safely.

`NOO_GATEWAY_KEY` is a named row in
`noo_service_keys` — scoped, revocable, with a `last_used_at`. Presenting it
on every call makes the thing that travels as long-lived as the thing that
grants. The exchange keeps the key at one door and puts a ten-minute token on
every other request.

What these pin is mostly the DEGRADING. The exchange sits in front of every
gateway call trellion makes, so the ways it can fail matter more than the way
it succeeds: an older gateway has no such route, a key without scopes is
refused, and the network can simply not answer. Each of those must fall back
to presenting the key — which is exactly what every install did before the
door existed — rather than failing the call.
"""

from __future__ import annotations

import time

import httpx
import pytest

from trellion.live import gateway_auth


@pytest.fixture(autouse=True)
def _clean_cache(monkeypatch: pytest.MonkeyPatch):
    """Module-level cache: tests share one process, so reset it between them
    or the first test's token decides every later assertion."""
    monkeypatch.setattr(gateway_auth, "_token", None, raising=False)
    monkeypatch.setattr(gateway_auth, "_next_attempt", 0.0, raising=False)
    monkeypatch.setattr(gateway_auth, "_transport", None, raising=False)
    monkeypatch.setenv("NOO_GATEWAY_KEY", "noo-sk-the-key")
    monkeypatch.delenv("NOO_CONTROLLER_SECRET", raising=False)
    monkeypatch.delenv("NOO_CLUSTER_SECRET", raising=False)
    yield


def _transport(status: int, body: dict | None = None, calls: list | None = None):
    """An httpx.MockTransport, injected through gateway_auth._transport.

    NOT a replacement for `httpx.post`. The first version of these tests
    monkeypatched `httpx.post` with a fake that accepted any keyword — and so
    passed while production raised `TypeError: post() got an unexpected
    keyword argument 'cert'` on every single exchange. A transport sits UNDER
    the real Client, so the real constructor and the real `.post()` run, and
    a bad keyword surfaces here first.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        if calls is not None:
            calls.append(request)
        return httpx.Response(status, json=body or {})

    return httpx.MockTransport(handler)


def test_the_exchanged_token_is_what_travels(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gateway_auth, "_transport", _transport(200, {"token": "short-lived", "expires_in": 600}))
    assert gateway_auth.gateway_auth_header() == {"Authorization": "Bearer short-lived"}


def test_the_key_itself_authenticates_the_exchange(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never a token. A token that could open this door would renew itself
    past the revocation of the key it came from."""
    calls: list = []
    monkeypatch.setattr(gateway_auth, "_transport", _transport(200, {"token": "short-lived", "expires_in": 600}, calls))
    gateway_auth.gateway_auth_header()
    assert calls, "no exchange was attempted"
    assert calls[0].headers["Authorization"] == "Bearer noo-sk-the-key"


def test_one_exchange_serves_many_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """Per token lifetime, not per call — the point of exchanging at all."""
    calls: list = []
    monkeypatch.setattr(gateway_auth, "_transport", _transport(200, {"token": "short-lived", "expires_in": 600}, calls))
    for _ in range(25):
        gateway_auth.gateway_auth_header()
    assert len(calls) == 1, f"minted {len(calls)} tokens for 25 calls"


def test_an_expired_token_is_replaced(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gateway_auth, "_token", ("stale", time.monotonic() - 1), raising=False)
    monkeypatch.setattr(gateway_auth, "_transport", _transport(200, {"token": "fresh", "expires_in": 600}))
    assert gateway_auth.gateway_auth_header() == {"Authorization": "Bearer fresh"}


def test_a_gateway_without_the_door_still_gets_the_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """404 is not an error. Every gateway older than PR 343 answers this way,
    and the key is a perfectly good bearer — that lane is why it was kept."""
    monkeypatch.setattr(gateway_auth, "_transport", _transport(404))
    assert gateway_auth.gateway_auth_header() == {"Authorization": "Bearer noo-sk-the-key"}


def test_a_refused_exchange_still_gets_the_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """403 means the key carries no scopes beyond `addon`. Failing the call
    here would turn a re-issue-the-key problem into an outage."""
    monkeypatch.setattr(gateway_auth, "_transport", _transport(403))
    assert gateway_auth.gateway_auth_header() == {"Authorization": "Bearer noo-sk-the-key"}


def test_a_transport_failure_still_gets_the_key(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to gateway")

    monkeypatch.setattr(gateway_auth, "_transport", httpx.MockTransport(boom))
    assert gateway_auth.gateway_auth_header() == {"Authorization": "Bearer noo-sk-the-key"}


def test_a_failing_door_is_not_asked_on_every_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """The amplifier this must not become: a failing door asked on every
    call turns one missing route into a flood, and a 404 on each would triple the
    traffic a missing route already cannot serve."""
    calls: list = []
    monkeypatch.setattr(gateway_auth, "_transport", _transport(404, None, calls))
    for _ in range(50):
        gateway_auth.gateway_auth_header()
    assert len(calls) == 1, f"asked a dead door {len(calls)} times in 50 calls"


def test_a_nonsense_body_falls_back_rather_than_presenting_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 200 carrying no token is the silent shape: it would otherwise cache
    an empty bearer and every later call would 401 with nothing logged."""
    for body in ({}, {"token": "", "expires_in": 600}, {"token": "t", "expires_in": 0}):
        gateway_auth._token = None
        gateway_auth._next_attempt = 0.0
        monkeypatch.setattr(gateway_auth, "_transport", _transport(200, body))
        assert gateway_auth.gateway_auth_header() == {
            "Authorization": "Bearer noo-sk-the-key"
        }, f"body {body!r} must fall back to the key"


def test_no_key_means_no_exchange_and_no_hmac(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no key there is nothing to exchange, and the HMAC lane that used
    to stand in here is retired (noomesh followup 1264): the call raises,
    naming the fix, and nothing is sent — not even to the exchange door."""
    monkeypatch.delenv("NOO_GATEWAY_KEY", raising=False)
    monkeypatch.setenv("NOO_CONTROLLER_SECRET", "shared-hmac-value")

    def never(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request may be sent without a key")

    monkeypatch.setattr(gateway_auth, "_transport", httpx.MockTransport(never))
    with pytest.raises(gateway_auth.GatewayCredentialMissing) as ei:
        gateway_auth.gateway_auth_header()
    assert "noo security service-key provision trellion" in str(ei.value)


def test_the_exchange_runs_through_the_real_client_api(monkeypatch: pytest.MonkeyPatch) -> None:
    """The regression these tests exist for. With a transport underneath, the
    real `httpx.Client(...)` and `.post(...)` execute; a keyword one of them
    does not accept raises inside `_exchange`, is caught, and comes back as
    None — which this asserts against. The old `httpx.post` mock could not
    fail here, and did not."""
    monkeypatch.setattr(gateway_auth, "_transport", _transport(200, {"token": "t", "expires_in": 600}))
    assert gateway_auth._exchange("noo-sk-the-key") is not None, (
        "the exchange raised inside the real client call — see the warning it logged"
    )


# ── the door mints; a surface refuses (noomesh PR 421) ────────────────────


def _surface(calls: list, refuse_token: str | None):
    """A gateway whose exchange door works and whose embeddings surface
    refuses the exchanged token but accepts the key — 2026-09-22 exactly."""

    def handler(request: httpx.Request) -> httpx.Response:
        # Record the header AS SENT: the auth flow retries by mutating the
        # same Request object, so keeping the object would show the retry's
        # header on the first call too.
        bearer = request.headers.get("Authorization", "")
        calls.append((request.url.path, bearer))
        if request.url.path == "/api/addon/token":
            return httpx.Response(200, json={"token": "short-lived", "expires_in": 600})
        if refuse_token and bearer == f"Bearer {refuse_token}":
            return httpx.Response(403, json={"error": "forbidden"})
        return httpx.Response(200, json={"ok": True})

    return httpx.MockTransport(handler)


def test_a_token_refused_on_a_surface_is_retried_with_the_key(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list = []
    transport = _surface(calls, refuse_token="short-lived")
    monkeypatch.setattr(gateway_auth, "_transport", transport)
    client = httpx.Client(auth=gateway_auth.GatewayHmacAuth(), transport=transport, base_url="https://gateway:8080")
    r = client.post("/api/external/embeddings", json={"input": "x"})
    assert r.status_code == 200, "the call must succeed on the key after the token is refused"
    surface = [b for p, b in calls if p == "/api/external/embeddings"]
    assert surface == ["Bearer short-lived", "Bearer noo-sk-the-key"]


def test_after_a_surface_refusal_the_key_travels_without_asking_again(monkeypatch: pytest.MonkeyPatch) -> None:
    """The amplifier, one door later: a refused token must not be exchanged
    for another refused token on every call."""
    calls: list = []
    transport = _surface(calls, refuse_token="short-lived")
    monkeypatch.setattr(gateway_auth, "_transport", transport)
    client = httpx.Client(auth=gateway_auth.GatewayHmacAuth(), transport=transport, base_url="https://gateway:8080")
    for _ in range(20):
        assert client.post("/api/external/embeddings", json={"input": "x"}).status_code == 200
    exchanges = [p for p, _ in calls if p == "/api/addon/token"]
    refused = [p for p, b in calls if p != "/api/addon/token" and b == "Bearer short-lived"]
    assert len(exchanges) == 1, f"exchanged {len(exchanges)} times across 20 calls"
    assert len(refused) == 1, f"presented the refused token {len(refused)} times"


def test_a_refusal_of_the_key_itself_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only a refused TOKEN has a fallback. A refused key is the answer."""
    calls: list = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.url.path, request.headers.get("Authorization", "")))
        if request.url.path == "/api/addon/token":
            return httpx.Response(404)
        return httpx.Response(403)

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(gateway_auth, "_transport", transport)
    client = httpx.Client(auth=gateway_auth.GatewayHmacAuth(), transport=transport, base_url="https://gateway:8080")
    assert client.post("/api/external/embeddings", json={}).status_code == 403
    surface = [p for p, _ in calls if p == "/api/external/embeddings"]
    assert len(surface) == 1, "a refused key was retried"
