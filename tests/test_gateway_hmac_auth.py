# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""Every gateway call trellion makes carries THIS container's own key, and
nothing else (noomesh followup 1264).

2026-09-09: graph extraction, RAG synthesis and MarkItDown image descriptions
went straight to ``PAGEINDEX_LLM_URL`` with no credential; routed through the
gateway they carry the auth hook on every request. Until followup 1264 that
hook fell back to a cluster HMAC signed with ``NOO_CONTROLLER_SECRET`` /
``NOO_CLUSTER_SECRET`` when no key was set. That lane is retired and the
gateway refuses an add-on's HMAC, so with no key the call is NOT sent: the
hook raises ``GatewayCredentialMissing`` naming
``noo security service-key provision <addon>``, and logs it.

Every test here drives a real ``httpx`` client over a ``MockTransport`` that
records what reaches "the gateway" — "sends nothing" is asserted on the wire,
not on a helper's return value.
"""

from __future__ import annotations

import logging

import httpx
import pytest

from trellion.live import gateway_auth
from trellion.live.gateway_auth import GatewayCredentialMissing, GatewayHmacAuth, gateway_auth_header

SHARED = {"NOO_CONTROLLER_SECRET": "ctl", "NOO_CLUSTER_SECRET": "fleet"}


@pytest.fixture(autouse=True)
def _clean(monkeypatch: pytest.MonkeyPatch):
    for n in ("NOO_GATEWAY_KEY", "NOO_GATEWAY_KEY_FILE", "NOO_SECRET_NOO_GATEWAY_KEY", "NOO_WORKER_SURFACES"):
        monkeypatch.delenv(n, raising=False)
    # The shared secrets ARE present, as they are in the container (the
    # noomesh worker_pool beside trellion still holds the controller secret
    # for its bus principal). The door must not touch them.
    for k, v in SHARED.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("NODE_ID", "node-1")
    monkeypatch.setattr(gateway_auth, "_token", None, raising=False)
    monkeypatch.setattr(gateway_auth, "_next_attempt", 0.0, raising=False)
    monkeypatch.setattr(gateway_auth, "_circuit", {}, raising=False)
    monkeypatch.setattr(gateway_auth, "_missing_logged_at", None, raising=False)
    yield


def _recording():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path == "/api/addon/token":
            return httpx.Response(200, json={"token": "short-lived", "expires_in": 600})
        return httpx.Response(200, json={"ok": True})

    return httpx.MockTransport(handler), seen


def _looks_like_hmac(bearer: str) -> bool:
    parts = bearer.removeprefix("Bearer ").split(":")
    return len(parts) == 3 and parts[1].isdigit() and len(parts[2]) == 64


# ── no key: refuse loudly, send nothing ──────────────────────────────────


def test_no_key_sends_no_request_and_no_hmac(monkeypatch, caplog):
    transport, seen = _recording()
    monkeypatch.setattr(gateway_auth, "_transport", transport)
    client = httpx.Client(auth=GatewayHmacAuth(), transport=transport, base_url="https://gateway:8080")
    with caplog.at_level(logging.ERROR, logger=gateway_auth.__name__):
        with pytest.raises(GatewayCredentialMissing) as ei:
            client.post("/v1/chat/completions", json={})
    assert seen == [], f"{len(seen)} request(s) left the process with no key: {[r.headers for r in seen]}"
    assert "noo security service-key provision trellion" in str(ei.value)
    assert "NOT sent" in str(ei.value)
    logged = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert any("gateway_credential_missing" in m and "provision trellion" in m for m in logged), logged


async def test_no_key_sends_nothing_on_the_async_client_either(monkeypatch):
    transport, seen = _recording()
    async with httpx.AsyncClient(auth=GatewayHmacAuth(), transport=transport) as c:
        with pytest.raises(GatewayCredentialMissing):
            await c.post("https://gateway:8080/api/external/embeddings", json={})
    assert seen == []


def test_the_bare_helper_raises_rather_than_signing_or_returning_empty():
    """The pre-1264-retirement helper returned an HMAC here, or ``{}`` with no secret.
    Either is a request the gateway refuses with nothing naming the fix."""
    with pytest.raises(GatewayCredentialMissing):
        gateway_auth_header()


@pytest.mark.parametrize("blank", ["", "   ", "\n"])
def test_a_blank_key_is_no_key(monkeypatch, blank):
    monkeypatch.setenv("NOO_GATEWAY_KEY", blank)
    with pytest.raises(GatewayCredentialMissing):
        gateway_auth_header()


def test_the_intel_container_is_told_its_own_provision_command(monkeypatch):
    monkeypatch.setenv("NOO_WORKER_SURFACES", "ml-intel")
    with pytest.raises(GatewayCredentialMissing) as ei:
        gateway_auth_header()
    assert "noo security service-key provision trellion-intel" in str(ei.value)
    assert "svc:trellion-intel" in str(ei.value)


def test_the_missing_key_error_is_logged_once_a_minute_not_per_call(monkeypatch, caplog):
    with caplog.at_level(logging.ERROR, logger=gateway_auth.__name__):
        for _ in range(20):
            with pytest.raises(GatewayCredentialMissing):
                gateway_auth_header()
    assert sum("gateway_credential_missing" in r.getMessage() for r in caplog.records) == 1


# ── a key: it, or its exchanged token, and never an HMAC ─────────────────


def test_the_exchanged_token_travels(monkeypatch):
    transport, seen = _recording()
    monkeypatch.setattr(gateway_auth, "_transport", transport)
    monkeypatch.setenv("NOO_GATEWAY_KEY", "noo-sk-the-key")
    client = httpx.Client(auth=GatewayHmacAuth(), transport=transport, base_url="https://gateway:8080")
    client.post("/v1/chat/completions", json={})
    sent = [(r.url.path, r.headers.get("Authorization")) for r in seen]
    assert sent == [("/api/addon/token", "Bearer noo-sk-the-key"), ("/v1/chat/completions", "Bearer short-lived")]


@pytest.mark.parametrize("status", [404, 401, 403, 500])
def test_an_unavailable_exchange_never_falls_back_to_an_hmac(monkeypatch, status):
    """The door missing or refusing: the key itself travels (it is still this
    container's own credential), never a shared-secret signature."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/addon/token":
            return httpx.Response(status, json={})
        return httpx.Response(200, json={})

    monkeypatch.setattr(gateway_auth, "_transport", httpx.MockTransport(handler))
    monkeypatch.setenv("NOO_GATEWAY_KEY", "noo-sk-the-key")
    bearer = gateway_auth_header()["Authorization"]
    assert bearer == "Bearer noo-sk-the-key"
    assert not _looks_like_hmac(bearer)


def test_the_hook_resolves_the_credential_per_request(monkeypatch):
    """A long-lived client must not reuse a header computed at construction:
    a rotated key (or a refreshed token) is picked up on the next send."""
    monkeypatch.setattr(gateway_auth, "_next_attempt", float("inf"))  # the key itself
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers["Authorization"])
        return httpx.Response(200, json={"choices": []})

    client = httpx.Client(auth=GatewayHmacAuth(), transport=httpx.MockTransport(handler))
    monkeypatch.setenv("NOO_GATEWAY_KEY", "noo-sk-old")
    client.post("https://gateway:8080/v1/chat/completions", json={})
    monkeypatch.setenv("NOO_GATEWAY_KEY", "noo-sk-new")
    client.post("https://gateway:8080/v1/chat/completions", json={})
    assert seen == ["Bearer noo-sk-old", "Bearer noo-sk-new"]


@pytest.mark.parametrize("module,cls", [
    ("trellion.live.graph_extractor", "EntityExtractor"),
    ("trellion.live.rag_generator", "RAGAnswerGenerator"),
])
def test_llm_clients_carry_the_auth_hook(module, cls):
    import importlib
    from unittest.mock import MagicMock

    mod = importlib.import_module(module)
    kwargs = {"store": MagicMock()} if cls == "EntityExtractor" else {}
    inst = getattr(mod, cls)(llm_url="https://gateway:8080", **kwargs)
    assert isinstance(inst._http.auth, GatewayHmacAuth)


def test_a_missing_key_fails_like_a_request_that_could_not_be_sent():
    """Callers already treat an ``httpx.RequestError`` as a failed item (the
    extractors keep the page queued and back off). A new exception type would
    escape those handlers and kill a worker loop instead."""
    assert issubclass(GatewayCredentialMissing, httpx.RequestError)
