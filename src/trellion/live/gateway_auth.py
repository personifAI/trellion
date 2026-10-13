# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""Gateway auth for Python → gateway self-calls (trellion's `live/` package).

One implementation: every caller of the Rust gateway's embedding surface
(the PageIndex embedder, the graph classifier) takes its auth header from
here, so no caller keeps its own unauthenticated client.

The credential is this container's OWN service key, ``NOO_GATEWAY_KEY``
(``svc:trellion``, or the add-on's own identity when the add-on runs this
package in its own container), exchanged at ``/api/addon/token`` for a short-lived bearer
token; the key itself travels only when the exchange door is missing or
refuses (see ``_addon_bearer``). There is NO shared-secret fallback any more
(the shared secret is retired): with no key, :func:`gateway_auth_header`
logs ``gateway_credential_missing`` and raises
:class:`GatewayCredentialMissing`, naming
``noo security service-key provision <addon>``. It never signs with
``NOO_CONTROLLER_SECRET`` / ``NOO_CLUSTER_SECRET`` and never sends a request
with no credential at all — the gateway refuses an add-on's HMAC now.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Generator

import httpx

from .secret_env import secret as _secret

logger = logging.getLogger(__name__)

# The exchanged token and the monotonic time it stops being usable.
_token: tuple[str, float] | None = None
_token_lock = threading.Lock()

# Refresh this far before expiry. A token that expires mid-flight is a 401
# the caller cannot distinguish from a revoked key, so never cut it fine.
_REFRESH_MARGIN_SECS = 60.0

# How long to sit on a failed exchange before trying again. Without this a
# gateway that predates the door is asked on EVERY call, turning one missing
# route into a request amplifier.
_RETRY_AFTER_FAILURE_SECS = 300.0
_next_attempt = 0.0

# Test seam. Injecting an `httpx.MockTransport` here exercises the real
# `httpx.Client` construction and `.post()` call; monkeypatching `httpx.post`
# does not. The first version of this module passed a `cert=` keyword to
# `httpx.post`, which httpx 0.28 does not accept — every exchange raised
# TypeError, was caught, and fell back to presenting the key. Ten tests were
# green, because every one of them replaced `httpx.post` with a fake that
# accepted anything.
_transport: httpx.BaseTransport | None = None


def _gateway_url() -> str:
    return os.environ.get("NOO_GATEWAY_URL", "https://gateway:8080").rstrip("/")


def _exchange(key: str) -> tuple[str, float] | None:
    """Trade the long-lived key for a short-lived token.

    Returns (token, expires_at) or None — None meaning "present the key
    directly", which is what every install did before the door existed and
    what any gateway older than it still requires.

    The exchange itself authenticates with the KEY, never with a token: a
    token that could open this door would renew itself past the revocation of
    the key it came from, and the gateway drops the `addon` scope from what it
    mints for exactly that reason.
    """
    from ._tls import client_cert, ssl_verify  # noqa: PLC0415 — avoids an import cycle

    try:
        # A Client, not the module-level `httpx.post`: `cert=` is a Client
        # argument in httpx 0.28 and `post()` refuses it with TypeError.
        with httpx.Client(
            verify=ssl_verify(), cert=client_cert(), timeout=10.0, transport=_transport
        ) as client:
            resp = client.post(
                f"{_gateway_url()}/api/addon/token",
                headers={"Authorization": f"Bearer {key}"},
            )
    except Exception as exc:  # noqa: BLE001 — any transport failure falls back
        # The reason goes IN the message. `extra=` is not rendered by the
        # default formatter, so the first version logged the event name and
        # nothing else — the TypeError above was invisible for a full deploy.
        logger.warning("addon_token_exchange_failed: %s: %s", type(exc).__name__, exc)
        return None
    if resp.status_code == 404:
        # This gateway predates the door. Not an error: the key still works
        # as a bearer, which is the whole point of keeping that lane.
        logger.info("addon_token_exchange_unavailable: HTTP 404 (the gateway has no door)")
        return None
    if resp.status_code != 200:
        # 401 = the key does not carry the `addon` scope; 403 = it carries
        # nothing else. Either way the remedy is re-issuing the key, not
        # retrying, so it is a warning and it says which.
        logger.warning(
            "addon_token_exchange_refused: HTTP %s: %s", resp.status_code, resp.text[:160]
        )
        return None
    try:
        body = resp.json()
        token = str(body["token"])
        ttl = float(body.get("expires_in") or 0)
    except Exception as exc:  # noqa: BLE001
        logger.warning("addon_token_exchange_unreadable: %s", exc)
        return None
    if not token or ttl <= 0:
        return None
    return token, time.monotonic() + max(ttl - _REFRESH_MARGIN_SECS, 1.0)


def _addon_bearer(key: str) -> str:
    """The token to present: exchanged when the door answers, else the key.

    One exchange per token lifetime, not per call. The lock keeps a burst of
    concurrent requests from minting a token each — the entity extractor and
    the answer generator both hold long-lived clients.
    """
    global _token, _next_attempt

    now = time.monotonic()
    cached = _token
    if cached and now < cached[1]:
        return cached[0]

    with _token_lock:
        # Re-check inside the lock: another thread may have refreshed while
        # this one waited, and minting twice is the thing the lock is for.
        cached = _token
        now = time.monotonic()
        if cached and now < cached[1]:
            return cached[0]
        if now < _next_attempt:
            return key
        fresh = _exchange(key)
        if fresh is None:
            _next_attempt = now + _RETRY_AFTER_FAILURE_SECS
            return key
        _token = fresh
        return fresh[0]


def _presented_exchanged_token(header: dict[str, str]) -> bool:
    """Was the bearer in `header` the exchanged token (not the key, not HMAC)?"""
    cached = _token
    if not cached:
        return False
    return header.get("Authorization") == f"Bearer {cached[0]}"


def _refuse_token(status: int, path: str) -> None:
    """A surface refused the exchanged token: forget it and stop exchanging
    for the cooldown, so the key travels instead. Logged once per refusal
    rather than once per call — the call rate is the caller's, and the point
    is that the caller keeps working."""
    global _token, _next_attempt
    with _token_lock:
        _token = None
        _next_attempt = time.monotonic() + _RETRY_AFTER_FAILURE_SECS
    logger.warning(
        "addon_token_refused_on_surface: HTTP %s on %s — presenting the key for the next %ss "
        "(the gateway minted a token its own door refuses; noomesh PR 421)",
        status, path, int(_RETRY_AFTER_FAILURE_SECS),
    )


class GatewayCredentialMissing(httpx.RequestError):
    """This container holds no gateway key of its own, so there is nothing to
    present. Raised instead of sending the request: the shared-secret HMAC
    that used to stand in here is gone (noomesh followup 1264), and a request
    with no credential is a 401 that names nothing.

    An ``httpx.RequestError`` (a request that could not be sent), like
    :class:`GatewayRefused` is an ``HTTPStatusError``: every caller that
    already treats a failed gateway call as a failed item — the extractors
    keep the page queued and back off — handles this one the same way,
    instead of a new exception type killing a worker loop. The ERROR log
    line is what names the fix."""


#: How often the missing-key ERROR repeats. Every gateway call raises; the log
#: says why once a minute rather than once per call.
_MISSING_LOG_EVERY_SECS = 60.0
_missing_logged_at: float | None = None


def addon_name() -> str:
    """The add-on this process is, as `noo security service-key provision`
    names it. An add-on that runs this same package in its own container is
    told apart by the worker surface it serves (``NOO_WORKER_SURFACES``)."""
    surfaces = os.environ.get("NOO_WORKER_SURFACES", "")
    return "trellion-intel" if "ml-intel" in surfaces.split(",") else "trellion"


def missing_credential_reason() -> str:
    name = addon_name()
    return (
        f"no gateway key for svc:{name} (NOO_GATEWAY_KEY / NOO_GATEWAY_KEY_FILE is unset or "
        f"empty), so this call was NOT sent. The shared-secret fallback is retired. "
        f"Fix: noo security service-key provision {name}"
    )


def gateway_auth_header(request: httpx.Request | None = None) -> dict[str, str]:
    """The bearer for Python -> Rust self-calls: this container's own key,
    exchanged for a short-lived token (the key itself where the gateway has no
    exchange door or it refuses).

    With no key at all this RAISES :class:`GatewayCredentialMissing` and logs
    ``gateway_credential_missing`` (at most once a minute). It never falls
    back to ``NOO_CONTROLLER_SECRET`` / ``NOO_CLUSTER_SECRET`` and never returns
    an empty header: the request is not sent.
    """
    # A key of trellion's own: a named, scoped, quota-capped, revocable service
    # key, where the shared secrets it replaced were one value every
    # controller service held (which made a rotation all-or-nothing).
    key = (_secret("NOO_GATEWAY_KEY") or "").strip()
    if key:
        # Exchanged for a short-lived token where the gateway offers that door
        # so what travels on each request is
        # worth ten minutes rather than until someone revokes the key. Falls
        # back to the key itself against an older gateway.
        return {"Authorization": f"Bearer {_addon_bearer(key)}"}

    global _missing_logged_at
    reason = missing_credential_reason()
    now = time.monotonic()
    if _missing_logged_at is None or now - _missing_logged_at >= _MISSING_LOG_EVERY_SECS:
        _missing_logged_at = now
        logger.error("gateway_credential_missing: %s", reason)
    raise GatewayCredentialMissing(reason, request=request)


# ── The refusal circuit ──────────────────────────────────────────────────
#
# A surface that refuses the credential will refuse the next call too, and the
# callers here loop per ITEM, not per batch, so unchecked they would flood the
# surface with refused requests. After
# a refusal (401/403/429) the surface is closed locally for a growing interval
# (1 s doubling to 60 s); a call inside it raises the same kind of error the
# refused request would have, without leaving the process. The first success
# closes it. 60 s at most, so a rotated or re-scoped key recovers within a
# minute on its own.
_CIRCUIT_REFUSALS = (401, 403, 429)
_CIRCUIT_MIN_SECS = 1.0
_CIRCUIT_MAX_SECS = 60.0
# surface -> (open_until monotonic, current backoff secs, last refused status)
_circuit: dict[str, tuple[float, float, int]] = {}
_circuit_lock = threading.Lock()


class GatewayRefused(httpx.HTTPStatusError):
    """Raised locally while a surface's refusal circuit is open. An
    ``HTTPStatusError`` so every caller that handles the real refusal
    handles this one the same way."""


def _surface(request: httpx.Request) -> str:
    return f"{request.url.host}:{request.url.port}{request.url.path}"


def _circuit_check(request: httpx.Request) -> None:
    key = _surface(request)
    with _circuit_lock:
        state = _circuit.get(key)
    if state and time.monotonic() < state[0]:
        status = state[2]
        raise GatewayRefused(
            f"{request.url.path}: refused {status} moments ago — not retrying for "
            f"{state[0] - time.monotonic():.0f}s (refusal circuit, noomesh followup 436a)",
            request=request,
            response=httpx.Response(status, request=request, text="refusal circuit open"),
        )


def _circuit_record(request: httpx.Request, status: int) -> None:
    key = _surface(request)
    with _circuit_lock:
        if status in _CIRCUIT_REFUSALS:
            prev = _circuit.get(key)
            backoff = _CIRCUIT_MIN_SECS if prev is None else min(prev[1] * 2, _CIRCUIT_MAX_SECS)
            _circuit[key] = (time.monotonic() + backoff, backoff, status)
            opened = prev is None
        else:
            _circuit.pop(key, None)
            opened = False
    if opened:
        logger.warning(
            "gateway_refusal_circuit_open: HTTP %s on %s — backing off (1 s doubling to %ss) "
            "until a call succeeds", status, request.url.path, int(_CIRCUIT_MAX_SECS),
        )


class GatewayHmacAuth(httpx.Auth):
    """``httpx.Auth`` form of :func:`gateway_auth_header`.

    The name is historical: it once signed a cluster HMAC, and add-ons import
    it by this name. It presents this
    container's own key (exchanged) and nothing else; with no key it raises
    :class:`GatewayCredentialMissing` before the request leaves the process.

    The exchanged token expires, so a header computed once at client
    construction goes stale on a long-lived client — the entity extractor and
    the answer generator keep one ``AsyncClient`` for the life of the worker.
    An ``Auth`` hook resolves the bearer at send time instead, and works on the
    sync ``httpx.Client`` MarkItDown's OpenAI client wraps as well as the
    async ones.

    2026-09-09: trellion's LLM calls (graph extraction, RAG synthesis,
    MarkItDown image descriptions) went STRAIGHT to ``PAGEINDEX_LLM_URL``
    with no credential, which pinned them to a backend reachable by plain
    HTTP from this container — the controller's own ollama. Routed through
    the gateway's ``/v1/chat/completions`` they follow the same mesh
    resolver every chat channel uses (placements on any node, role=chat,
    exact-model preference), and the gateway carries the node-agent
    credential for the last hop. That needs the cluster bearer on every
    call, which is this class.
    """

    requires_request_body = False

    def auth_flow(self, request: httpx.Request) -> Generator[httpx.Request, httpx.Response, None]:
        _circuit_check(request)
        header = gateway_auth_header(request)
        for k, v in header.items():
            request.headers[k] = v
        response = yield request

        # An exchanged token the gateway then REFUSES on the surface it is
        # presented to. A token the door minted
        # can be answered 403 on the embeddings surface while the key itself
        # would have worked, and without this every call would present the
        # same refused token forever. So: drop the token, sit out the
        # exchange for the cooldown, and retry this one request with the
        # key. Same shape as a refused exchange, one door later.
        if response.status_code in (401, 403) and _presented_exchanged_token(header):
            key = (_secret("NOO_GATEWAY_KEY") or "").strip()
            if key:
                _refuse_token(response.status_code, request.url.path)
                request.headers["Authorization"] = f"Bearer {key}"
                response = yield request
        _circuit_record(request, response.status_code)
