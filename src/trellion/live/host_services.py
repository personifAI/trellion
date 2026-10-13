# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""trellion's implementation of ``trellion_protocols.HostServices``.

Handed to an add-on at load time (``trellion.live._intel``). Every member
delegates to the SAME process-wide object trellion's own code uses — the
embed throttle, the LLM throttle and breaker, the gateway credential, the
entity quality gate, the metrics path, the JSON codec — so an add-on shares
backpressure, breaker state and credentials with trellion instead of keeping
copies. Imports are deferred to the call: this module is loaded while
``trellion.live`` itself is still initialising.
"""

from __future__ import annotations

from typing import Any, Optional


class TrellionHostServices:
    """``trellion_protocols.HostServices`` over trellion's own singletons."""

    # -- transport ---------------------------------------------------------

    def ssl_verify(self) -> Any:
        from trellion.live._tls import ssl_verify

        return ssl_verify()

    def client_cert(self) -> Optional[tuple[str, str]]:
        from trellion.live._tls import client_cert

        return client_cert()

    def gateway_auth(self) -> Any:
        from trellion.live.gateway_auth import GatewayHmacAuth

        return GatewayHmacAuth()

    def gateway_auth_header(self) -> dict[str, str]:
        from trellion.live.gateway_auth import gateway_auth_header

        return gateway_auth_header()

    # -- throttles -----------------------------------------------------------

    def embed_throttle(self) -> Any:
        from trellion.live.embed_throttle import get_throttle

        return get_throttle()

    @property
    def embed_unavailable(self) -> type[Exception]:
        from trellion.live.embed_throttle import EmbedUnavailable

        return EmbedUnavailable

    def classify_embed_failure(self, exc: BaseException) -> Optional[str]:
        from trellion.live.embed_throttle import classify

        return classify(exc)

    def llm_throttle(self) -> Any:
        from trellion.live.llm_throttle import get_llm_throttle

        return get_llm_throttle()

    @property
    def llm_unavailable(self) -> type[Exception]:
        from trellion.live.llm_throttle import LLMUnavailable

        return LLMUnavailable

    @property
    def llm_breaker_open(self) -> type[Exception]:
        from trellion.live.llm_throttle import LLMBreakerOpen

        return LLMBreakerOpen

    # -- entity quality --------------------------------------------------------

    def check_entity(self, name: object, etype: Optional[str] = None) -> Optional[str]:
        from trellion.live.entity_quality import check_entity

        return check_entity(name, etype)

    def drop_tally(self) -> Any:
        from trellion.live.entity_quality import DropTally

        return DropTally()

    # -- observability and encoding ------------------------------------------

    async def inc_counter(
        self, name: str, value: int = 1, labels: Optional[dict[str, str]] = None
    ) -> None:
        from trellion.observability import inc_counter

        await inc_counter(name, value=value, labels=labels)

    @property
    def json(self) -> Any:
        from trellion import json_fast

        return json_fast


#: The one instance trellion hands every add-on.
HOST = TrellionHostServices()
