# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""TLS helpers for memory subsystem HTTP clients.

Pure env-var readers — no gateway dependency. Memory modules call these
to construct ``httpx.Client(verify=..., cert=...)`` for cluster-internal
endpoints (TEI, Ollama, vLLM) that present cluster-CA-signed certs.

Behaviorally identical to ``noomesh.gateway.deps.ssl_verify`` /
``client_cert``. Duplicated here so the memory package has zero inbound
imports from ``noomesh.gateway`` — the seam that lets memory move out
of the gateway repo into this repo (proposal 06, phase 1).
"""
from __future__ import annotations

import os


def ssl_verify():
    """Return the TLS verify setting for memory-internal HTTP clients.

    Creates an SSL context that verifies the CA chain but skips hostname
    checking. Cluster nodes register with LAN IPs, Docker container names,
    or hostnames — strict hostname matching against IP SANs is unreliable
    across Docker bridge / WSL / LAN boundaries.
    """
    ca = os.environ.get("NOO_TLS_CA", "")
    if ca and os.path.isfile(ca):
        import ssl
        ctx = ssl.create_default_context(cafile=ca)
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_REQUIRED
        return ctx
    return True


def client_cert() -> tuple[str, str] | None:
    """Return (cert_path, key_path) for mTLS client authentication.

    Returns None if no client cert is configured.
    """
    cert = os.environ.get("NOO_TLS_CERT", "")
    key = os.environ.get("NOO_TLS_KEY", "")
    if cert and key and os.path.isfile(cert) and os.path.isfile(key):
        return (cert, key)
    return None
