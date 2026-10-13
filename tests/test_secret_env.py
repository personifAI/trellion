# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""Tier-1 secrets read NAME_FILE > NOO_SECRET_NAME > NAME (noomesh followup 1247).

Compose will hand this container ``/run/secrets/<name>`` and set ``NAME_FILE``
instead of ``NAME``. A reader that only knows ``NAME`` then starts with the
secret unset and every gateway call 401s — noomesh followup 247's shape. These
pin the helper's semantics (identical to noomesh ``secret_env``) and that each
migrated call site in ``gateway_auth`` actually honours the file.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from trellion.live import gateway_auth, secret_env
from trellion.live.secret_env import (
    FILE_BACKED_SECRETS,
    SecretConflictError,
    SecretFileError,
    refuse_on_conflicts,
    secret,
)

NAMES = FILE_BACKED_SECRETS


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch):
    for n in NAMES:
        for var in (n, n + "_FILE", "NOO_SECRET_" + n):
            monkeypatch.delenv(var, raising=False)
    monkeypatch.delenv("NODE_ID", raising=False)
    monkeypatch.setattr(gateway_auth, "_token", None, raising=False)
    monkeypatch.setattr(gateway_auth, "_next_attempt", 0.0, raising=False)
    monkeypatch.setattr(gateway_auth, "_transport", None, raising=False)
    monkeypatch.setattr(gateway_auth, "_circuit", {}, raising=False)
    yield


def _file(tmp_path: Path, content: str, name: str = "s") -> str:
    p = tmp_path / name
    p.write_bytes(content.encode())
    return str(p)


# ── the helper ────────────────────────────────────────────────────────────


def test_manifest_names_the_one_gateway_credential() -> None:
    """The shared secrets left with the HMAC fallback (noomesh followup 1264)."""
    assert set(NAMES) == {"NOO_GATEWAY_KEY"}


def test_precedence_file_beats_prefixed_beats_plain(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("NOO_GATEWAY_KEY", "plain")
    assert secret("NOO_GATEWAY_KEY") == "plain"
    monkeypatch.setenv("NOO_SECRET_NOO_GATEWAY_KEY", "prefixed")
    assert secret("NOO_GATEWAY_KEY") == "prefixed"
    monkeypatch.setenv("NOO_GATEWAY_KEY_FILE", _file(tmp_path, "from-file\n"))
    assert secret("NOO_GATEWAY_KEY") == "from-file"


def test_unset_is_none_and_empty_values_fall_through(monkeypatch) -> None:
    assert secret("NOO_GATEWAY_KEY") is None
    monkeypatch.setenv("NOO_GATEWAY_KEY_FILE", "")
    monkeypatch.setenv("NOO_SECRET_NOO_GATEWAY_KEY", "")
    monkeypatch.setenv("NOO_GATEWAY_KEY", "plain")
    assert secret("NOO_GATEWAY_KEY") == "plain"


def test_strips_exactly_one_trailing_newline(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("NOO_CLUSTER_SECRET_FILE", _file(tmp_path, "abc\n\n"))
    assert secret("NOO_CLUSTER_SECRET") == "abc\n"
    monkeypatch.setenv("NOO_CLUSTER_SECRET_FILE", _file(tmp_path, "abc", "b"))
    assert secret("NOO_CLUSTER_SECRET") == "abc"
    monkeypatch.setenv("NOO_CLUSTER_SECRET_FILE", _file(tmp_path, " abc \n", "c"))
    assert secret("NOO_CLUSTER_SECRET") == " abc ", "only the newline is stripped, not whitespace"


def test_strips_one_crlf(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("NOO_CLUSTER_SECRET_FILE", _file(tmp_path, "abc\r\n"))
    assert secret("NOO_CLUSTER_SECRET") == "abc"
    monkeypatch.setenv("NOO_CLUSTER_SECRET_FILE", _file(tmp_path, "abc\r\n\r\n", "b"))
    assert secret("NOO_CLUSTER_SECRET") == "abc\r\n"


def test_unreadable_file_raises_naming_the_path(monkeypatch, tmp_path) -> None:
    missing = str(tmp_path / "nope")
    monkeypatch.setenv("NOO_CONTROLLER_SECRET_FILE", missing)
    monkeypatch.setenv("NOO_CONTROLLER_SECRET", "would-mask-a-mis-mount")
    with pytest.raises(SecretFileError) as ei:
        secret("NOO_CONTROLLER_SECRET")
    assert missing in str(ei.value)
    assert "NOO_CONTROLLER_SECRET_FILE" in str(ei.value)


def test_empty_file_raises(monkeypatch, tmp_path) -> None:
    for content in ("", "\n", "\r\n"):
        path = _file(tmp_path, content, f"e{len(content)}")
        monkeypatch.setenv("NOO_CONTROLLER_SECRET_FILE", path)
        with pytest.raises(SecretFileError, match="is empty") as ei:
            secret("NOO_CONTROLLER_SECRET")
        assert path in str(ei.value)


def test_both_set_is_refused_at_boot(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("NOO_GATEWAY_KEY", "plain")
    monkeypatch.setenv("NOO_GATEWAY_KEY_FILE", _file(tmp_path, "file\n"))
    with pytest.raises(SecretConflictError) as ei:
        refuse_on_conflicts()
    assert ei.value.names == ["NOO_GATEWAY_KEY"]


def test_boot_check_passes_one_way_delivery_and_fails_a_broken_file(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("NOO_GATEWAY_KEY_FILE", _file(tmp_path, "file\n"))
    # A name outside the manifest (a shared secret since followup 1264) is not
    # this check's business, whichever way it is delivered.
    monkeypatch.setenv("NOO_CLUSTER_SECRET", "plain")
    monkeypatch.setenv("NOO_CLUSTER_SECRET_FILE", _file(tmp_path, "x\n", "x"))
    refuse_on_conflicts()
    monkeypatch.setenv("NOO_GATEWAY_KEY_FILE", str(tmp_path / "missing"))
    with pytest.raises(SecretFileError):
        refuse_on_conflicts()


def test_the_cli_entry_refuses_a_both_set_conflict(monkeypatch, tmp_path) -> None:
    from trellion.cli import app

    monkeypatch.setenv("NOO_GATEWAY_KEY", "plain")
    monkeypatch.setenv("NOO_GATEWAY_KEY_FILE", _file(tmp_path, "file\n"))
    result = CliRunner().invoke(app, ["inspect", "--graph", str(tmp_path / "g.csv")])
    assert isinstance(result.exception, SecretConflictError), result.output


# ── each migrated call site honours NAME_FILE ─────────────────────────────


def test_gateway_key_file_is_the_key_that_is_exchanged(monkeypatch, tmp_path) -> None:
    """gateway_auth_header: the NOO_GATEWAY_KEY read."""
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json={"token": "short-lived", "expires_in": 600})

    monkeypatch.setattr(gateway_auth, "_transport", httpx.MockTransport(handler))
    monkeypatch.setenv("NOO_GATEWAY_KEY_FILE", _file(tmp_path, "noo-sk-from-file\n"))
    assert gateway_auth.gateway_auth_header() == {"Authorization": "Bearer short-lived"}
    assert calls[0].headers["Authorization"] == "Bearer noo-sk-from-file"


def test_a_shared_secret_file_is_not_a_gateway_credential(monkeypatch, tmp_path) -> None:
    """Both shared secrets delivered as files, no key: the door raises rather
    than signing with either (noomesh followup 1264)."""
    monkeypatch.setenv("NOO_CONTROLLER_SECRET_FILE", _file(tmp_path, "ctl-from-file\n", "c"))
    monkeypatch.setenv("NOO_CLUSTER_SECRET_FILE", _file(tmp_path, "fleet-from-file\n", "f"))
    with pytest.raises(gateway_auth.GatewayCredentialMissing):
        gateway_auth.gateway_auth_header()


def test_refused_token_retry_uses_the_key_file(monkeypatch, tmp_path) -> None:
    """GatewayHmacAuth.auth_flow: the NOO_GATEWAY_KEY read on the retry path."""
    calls: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bearer = request.headers.get("Authorization", "")
        calls.append((request.url.path, bearer))
        if request.url.path == "/api/addon/token":
            return httpx.Response(200, json={"token": "short-lived", "expires_in": 600})
        if bearer == "Bearer short-lived":
            return httpx.Response(403, json={"error": "forbidden"})
        return httpx.Response(200, json={"ok": True})

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(gateway_auth, "_transport", transport)
    monkeypatch.setenv("NOO_GATEWAY_KEY_FILE", _file(tmp_path, "noo-sk-from-file\n"))
    client = httpx.Client(auth=gateway_auth.GatewayHmacAuth(), transport=transport, base_url="https://gateway:8080")
    r = client.post("/api/external/embeddings", json={"input": "x"})
    assert r.status_code == 200
    surface = [b for p, b in calls if p == "/api/external/embeddings"]
    assert surface == ["Bearer short-lived", "Bearer noo-sk-from-file"]


def test_gateway_auth_reads_secrets_only_through_the_helper() -> None:
    """No call site in the door reads one of the manifest names straight from
    the environment — the next edit cannot quietly reintroduce one."""
    src = Path(gateway_auth.__file__).read_text(encoding="utf-8")
    assert "_secret(" in src, "the scan below would be vacuous"
    for n in NAMES:
        assert f'environ.get("{n}"' not in src and f'getenv("{n}"' not in src, n
    assert secret_env.FILE_BACKED_SECRETS == NAMES


def test_a_values_own_carriage_returns_survive(monkeypatch, tmp_path) -> None:
    """No universal-newline translation: the Rust reader keeps the bytes."""
    monkeypatch.setenv("NOO_CLUSTER_SECRET_FILE", _file(tmp_path, "a\rb\r"))
    assert secret("NOO_CLUSTER_SECRET") == "a\rb\r"
