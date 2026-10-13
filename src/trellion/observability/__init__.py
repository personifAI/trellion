# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""Best-effort metric publishers for trellion.

Mirrors ``noomesh.observability.metrics``: tries to publish JSON events to
NATS subjects (``metrics.counter`` / ``metrics.gauge``) so the Rust gateway
can render them at ``/metrics``. Falls back to a silent no-op when no NATS
context is available — metrics are best-effort and must never break the
caller.

Trellion is import-safe in environments without NATS (CLI, tests,
non-cluster hosts). The publish path uses lazy imports for the same
reason.
"""
from __future__ import annotations

import json
from typing import Any, Optional

import structlog

logger = structlog.get_logger(__name__)


async def _publish(subject: str, body: dict[str, Any]) -> None:
    """Publish a metric event to NATS. Returns silently on any failure."""
    try:
        # Trellion is embedded inside noomesh today, so the noomesh
        # messaging singleton is the most reliable hook. If trellion is
        # being used standalone (CLI, tests), this import will fail and
        # we degrade to a no-op.
        from noomesh.cluster.messaging import get_messaging  # type: ignore
        msg = get_messaging()
        if msg is None or not getattr(msg, "nc", None):
            return
        await msg.nc.publish(subject, json.dumps(body).encode("utf-8"))
    except Exception as e:
        logger.debug("trellion.metrics.publish_failed", subject=subject, error=str(e))


async def inc_counter(
    name: str,
    value: int = 1,
    labels: Optional[dict[str, str]] = None,
) -> None:
    body: dict[str, Any] = {"type": "counter", "name": name, "value": value}
    if labels:
        body["labels"] = labels
    await _publish("metrics.counter", body)


async def set_gauge(
    name: str,
    value: float,
    labels: Optional[dict[str, str]] = None,
) -> None:
    body: dict[str, Any] = {"type": "gauge", "name": name, "value": float(value)}
    if labels:
        body["labels"] = labels
    await _publish("metrics.gauge", body)
