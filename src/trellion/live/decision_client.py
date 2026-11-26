# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""Asks a decision endpoint (SystemOne wire format) the spec's questions about a page.

Configuration is by environment variable; with ``TRELLION_DECISION_URL`` unset
the feature is inert (:meth:`DecisionClient.from_env` returns None).

=================================  =========================================
``TRELLION_DECISION_URL``          The endpoint. A bare host URL gets
                                   ``/v1/systemone`` appended.
``TRELLION_DECISION_MODEL``        Model name sent in the request (omitted
                                   when unset).
``TRELLION_DECISION_API_KEY``      Optional bearer token. Never logged.
``TRELLION_DECISION_TIMEOUT_S``    Per-request timeout, 1-120 (default 20).
``TRELLION_DECISION_MAX_CHARS``    Page text sent, truncated to this many
                                   characters, 200-50000 (default 4000).
=================================  =========================================

The page's title, source, headers and text are sent to the configured
endpoint: point it only at a service you are content to see that text.
A request is tried twice at most (one retry, on a timeout, a connection
error, a 429 or a 5xx). The URL's query string, the key and the page text are
never written to the log.
"""

from __future__ import annotations

import os
from typing import Any, Mapping, Optional
from urllib.parse import urlsplit

import httpx
import structlog

from trellion.live import systemone
from trellion.live.knowledge_spec import KnowledgeSpec

log = structlog.get_logger(__name__)

URL_ENV = "TRELLION_DECISION_URL"
MODEL_ENV = "TRELLION_DECISION_MODEL"
KEY_ENV = "TRELLION_DECISION_API_KEY"
TIMEOUT_ENV = "TRELLION_DECISION_TIMEOUT_S"
MAX_CHARS_ENV = "TRELLION_DECISION_MAX_CHARS"

DEFAULT_PATH = "/v1/systemone"
DEFAULT_TIMEOUT_S = 20.0
DEFAULT_MAX_CHARS = 4000
_RETRY_STATUS = frozenset({429, 500, 502, 503, 504})


class DecisionError(RuntimeError):
    """The endpoint could not give an answer. The message carries no secret."""


def _env_number(name: str, default: float, lo: float, hi: float) -> float:
    raw = os.environ.get(name)
    try:
        value = float(raw) if raw not in (None, "") else default
    except ValueError:
        value = default
    return min(max(value, lo), hi)


def endpoint_url(raw: str) -> str:
    """``raw`` with ``/v1/systemone`` appended when it names only a host."""
    parts = urlsplit(raw.strip())
    if parts.path in ("", "/"):
        return raw.strip().rstrip("/") + DEFAULT_PATH
    return raw.strip()


def _kind(source_uri: Optional[str]) -> str:
    """A short kind for a source: its file extension, else its URI scheme."""
    if not source_uri:
        return "unknown"
    tail = source_uri.rsplit("/", 1)[-1]
    if "." in tail:
        ext = tail.rsplit(".", 1)[-1].lower()
        if 0 < len(ext) <= 8 and ext.isalnum():
            return ext
    scheme = urlsplit(source_uri).scheme
    return scheme.lower() if scheme else "unknown"


def build_state(page: Mapping[str, Any], max_chars: int) -> dict[str, Any]:
    """The ``state`` sent for a page: metadata plus truncated text."""
    meta = page.get("meta") or {}
    if not isinstance(meta, Mapping):
        meta = {}
    text = page.get("text") or ""
    headers = meta.get("headers")
    return {
        "title": page.get("title") or "",
        "source": page.get("source_uri") or "",
        "kind": _kind(page.get("source_uri")),
        "headers": [str(h) for h in headers][:20] if isinstance(headers, list) else [],
        "char_count": meta.get("char_count") if isinstance(meta.get("char_count"), int) else len(text),
        "has_table": bool(meta.get("has_table")),
        "has_figure": bool(meta.get("has_figure")),
        "text": text[:max_chars],
        "text_truncated": len(text) > max_chars,
    }


class DecisionClient:
    def __init__(
        self,
        url: str,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_chars: int = DEFAULT_MAX_CHARS,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ) -> None:
        self.url = endpoint_url(url)
        self.model = model or None
        self._api_key = api_key or None
        self.timeout_s = timeout_s
        self.max_chars = max_chars
        self._transport = transport
        self._http: Optional[httpx.AsyncClient] = None

    @classmethod
    def from_env(cls) -> Optional["DecisionClient"]:
        """A client from the environment, or None when no endpoint is configured."""
        url = (os.environ.get(URL_ENV) or "").strip()
        if not url:
            return None
        return cls(
            url,
            model=(os.environ.get(MODEL_ENV) or "").strip() or None,
            api_key=(os.environ.get(KEY_ENV) or "").strip() or None,
            timeout_s=_env_number(TIMEOUT_ENV, DEFAULT_TIMEOUT_S, 1.0, 120.0),
            max_chars=int(_env_number(MAX_CHARS_ENV, DEFAULT_MAX_CHARS, 200, 50000)),
        )

    def build_request(self, spec: KnowledgeSpec, page: Mapping[str, Any]) -> dict[str, Any]:
        return systemone.build_request(
            self.model, build_state(page, self.max_chars), spec.wire_questions()
        )

    def _client(self) -> httpx.AsyncClient:
        if self._http is None:
            headers = {"Content-Type": "application/json"}
            if self._api_key:
                # The decision endpoint's own key, set by the operator for that
                # service: not a gateway credential, so not the gateway door's.
                headers["Authorization"] = "Bearer " + self._api_key
            self._http = httpx.AsyncClient(
                timeout=self.timeout_s, headers=headers, transport=self._transport
            )
        return self._http

    async def aclose(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    async def ask(self, spec: KnowledgeSpec, page: Mapping[str, Any]) -> dict[str, Any]:
        """The ``answers`` object for ``page``. Raises :class:`DecisionError`."""
        body = self.build_request(spec, page)
        host = urlsplit(self.url).netloc.rsplit("@", 1)[-1]
        last = "no attempt"
        for attempt in (1, 2):
            try:
                resp = await self._client().post(self.url, json=body)
            except httpx.TimeoutException:
                last = "timeout"
            except httpx.HTTPError as exc:
                last = f"transport error ({type(exc).__name__})"
            else:
                if resp.status_code == 200:
                    try:
                        parsed = systemone.parse_response(resp.json())
                    except ValueError:
                        parsed = None
                    if parsed is None:
                        raise DecisionError("endpoint returned a response with no answers object")
                    return parsed
                last = f"HTTP {resp.status_code}"
                if resp.status_code not in _RETRY_STATUS:
                    break
            log.debug("decision_client.attempt_failed", host=host, attempt=attempt, reason=last)
        raise DecisionError(f"decision request failed: {last}")
