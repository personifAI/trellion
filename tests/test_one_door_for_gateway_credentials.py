# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""A trellion -> gateway call gets its credential from ONE place.

`gateway_auth.gateway_auth_header()` (and its httpx.Auth form) is the door.
It prefers this container's own key — exchanged for a short-lived token —
and nothing else: the shared-HMAC fallback is retired (noomesh followup
1264). A second signer anywhere else would keep
presenting the shared secret after the door has moved on, and the install
reads as migrated while that caller is not.

`embedding_worker.py` once signed its own bearer from NOO_CLUSTER_SECRET,
so embedding calls presented a different credential than every other call.
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "trellion"
DOOR = SRC / "live" / "gateway_auth.py"


def _own_signers() -> list[str]:
    hits: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        if path == DOOR:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        for node in ast.walk(tree):
            # hmac.new(...) anywhere outside the door is a second signer
            if isinstance(node, ast.Call):
                fn = node.func
                if isinstance(fn, ast.Attribute) and fn.attr == "new" and getattr(fn.value, "id", None) == "hmac":
                    hits.append(f"{path.relative_to(SRC.parent.parent)}:{node.lineno}  hmac.new(...)")
            # a hand-built "Bearer ..." string is the other shape of it
            if isinstance(node, ast.JoinedStr):
                head = node.values[0] if node.values else None
                if isinstance(head, ast.Constant) and str(head.value).startswith("Bearer "):
                    hits.append(f"{path.relative_to(SRC.parent.parent)}:{node.lineno}  a hand-built Bearer f-string")
    return hits


def test_the_scan_sees_the_door_itself() -> None:
    """Sanity: the pattern must find a bearer builder SOMEWHERE, or an empty
    result below is a pattern that matches nothing rather than a clean tree.
    The door builds ``f"Bearer {...}"`` from this container's own key."""
    tree = ast.parse(DOOR.read_text(encoding="utf-8-sig"))
    found = any(
        isinstance(n, ast.JoinedStr)
        and n.values
        and isinstance(n.values[0], ast.Constant)
        and str(n.values[0].value).startswith("Bearer ")
        for n in ast.walk(tree)
    )
    assert found, "the door builds no Bearer f-string — the scan below would be vacuous"


def test_nothing_in_trellion_signs_an_hmac_any_more() -> None:
    """noomesh followup 1264: the door's shared-secret fallback is retired, so
    NO module — the door included — calls ``hmac.new`` or reads either shared
    secret. The gateway refuses an add-on's HMAC; a signer left here would be
    a refused call, not a working fallback."""
    hits: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        text = path.read_text(encoding="utf-8-sig")
        tree = ast.parse(text, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                fn = node.func
                if isinstance(fn, ast.Attribute) and fn.attr == "new" and getattr(fn.value, "id", None) == "hmac":
                    hits.append(f"{path.relative_to(SRC.parent.parent)}:{node.lineno}  hmac.new(...)")
            if isinstance(node, ast.Constant) and node.value in ("NOO_CONTROLLER_SECRET", "NOO_CLUSTER_SECRET"):
                hits.append(f"{path.relative_to(SRC.parent.parent)}:{node.lineno}  reads {node.value}")
    assert not hits, "a shared-secret signer is back:\n  " + "\n  ".join(hits)


def test_only_the_door_signs_a_gateway_bearer() -> None:
    hits = _own_signers()
    assert not hits, (
        "these build a gateway credential outside gateway_auth.py:\\n  "
        + "\\n  ".join(hits)
        + "\\nRoute them through gateway_auth_header() / GatewayHmacAuth(). A second signer "
        "keeps presenting the shared secret after the door prefers a per-consumer key."
    )
