# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""MemoryEvents — the abstract surface trellion expects of an event bus.

Trellion's stores and graph-builders publish lightweight events
(``graph.edges.built``, ``graph.entity.extracted``, ``ingest.doc.parsed``,
etc.) to coordinate downstream work — embedding, classification,
clustering. They subscribe to the same events.

In a noomesh deployment, NATS provides the implementation. In a
standalone trellion deployment, an in-memory pub/sub or a noop
implementation is sufficient. Trellion does not import a concrete
NATS client — callers inject any object satisfying this Protocol.

The Protocol is structural — existing noomesh ``ClusterMessaging``
already matches without any changes. Phase 3 of proposal 06 just makes
the contract explicit.
"""
from __future__ import annotations

from typing import Any, Awaitable, Callable, Protocol, runtime_checkable


@runtime_checkable
class MemoryEvents(Protocol):
    """The pub/sub surface trellion needs.

    Implementations:
    - ``noomesh.cluster.messaging.ClusterMessaging`` — NATS-backed
      (used in noomesh runtime, satisfies this Protocol structurally).
    - ``trellion.live._messaging.NoopEvents`` — drops everything (safe
      default for tests / standalone use without an event bus).
    """

    is_connected: bool

    async def publish_event(self, subject: str, data: dict) -> None:
        """Fire-and-forget publish to *subject*."""
        ...

    async def subscribe_events(
        self,
        subject: str,
        handler: Callable[[dict], Awaitable[None]],
        **kwargs: Any,
    ) -> None:
        """Subscribe *handler* to messages on *subject*."""
        ...


class NoopEvents:
    """Drop-everything implementation — satisfies the Protocol with no I/O.

    Use as a default when no event bus is wired. Trellion's graph builders
    check ``messaging.is_connected`` before publishing, so this short-
    circuits cleanly without spamming logs.
    """

    is_connected: bool = False

    async def publish_event(self, subject: str, data: dict) -> None:
        return None

    async def subscribe_events(
        self,
        subject: str,
        handler: Callable[[dict], Awaitable[None]],
        **kwargs: Any,
    ) -> None:
        return None
