# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""One reader for tier-1 secrets: ``NAME_FILE`` > ``NOO_SECRET_NAME`` > ``NAME``.

Mirrors the noomesh stack's ``secret_env`` readers; keep the semantics
identical. The stack hands a container ``/run/secrets/<name>`` and sets
``NAME_FILE`` instead of ``NAME``. A reader that only knew ``NAME`` would
start with the secret unset and every gateway call would 401. So the readers
learn ``_FILE`` first, and either roll order is safe.

Resolution, for ``secret("NAME")``:

1. ``NAME_FILE`` set and non-empty → read the file, strip exactly one
   trailing ``\\n`` (or ``\\r\\n``). Unreadable or empty → :class:`SecretFileError`
   naming the path. Never falls through to the env: a mis-mounted secret must
   not look like a missing one.
2. ``NOO_SECRET_NAME`` non-empty → that.
3. ``NAME`` non-empty → that.
4. ``None``.

Both ``NAME`` and ``NAME_FILE`` non-empty is a compose/launcher skew, not a
choice: :func:`refuse_on_conflicts` is the boot check that refuses it. The
per-call reader still answers (``_FILE`` wins) so the refusal is reported
once, by the check that owns it.

In the stack's trellion containers the process entry is
noomesh's ``worker_pool``, which runs noomesh's own ``refuse_on_conflicts``
over a manifest that includes every name below. This module's check is for
trellion's own entry point (the ``trellion`` CLI) and for any embedder that
does not run through noomesh.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

#: Every name trellion reads through :func:`secret`. Add a reader, add the name.
#: ``NOO_CONTROLLER_SECRET`` / ``NOO_CLUSTER_SECRET`` left with the HMAC
#: fallback (noomesh followup 1264): nothing in trellion reads them. The
#: noomesh worker_pool that hosts trellion in its containers still does (its
#: bus principal), through noomesh's own reader and manifest.
FILE_BACKED_SECRETS: tuple[str, ...] = (
    "NOO_GATEWAY_KEY",
)

PREFIX = "NOO_SECRET_"
FILE_SUFFIX = "_FILE"


class SecretFileError(RuntimeError):
    """``NAME_FILE`` names a file that cannot be read, or is empty."""

    def __init__(self, name: str, path: str, reason: str) -> None:
        self.name = name
        self.path = path
        self.reason = reason
        super().__init__(
            f"{name}{FILE_SUFFIX} names {path}, which {reason}; "
            "a named secret file must exist, be readable by this process, "
            "and be non-empty"
        )


class SecretConflictError(RuntimeError):
    """Both ``NAME`` and ``NAME_FILE`` are set for one or more names."""

    def __init__(self, names: list[str]) -> None:
        self.names = names
        super().__init__(
            "both NAME and NAME_FILE are set for "
            + ", ".join(names)
            + " — a compose/launcher skew, not a choice; deliver each secret "
            "ONE way (unset the env value, or the _FILE pointer)"
        )


def strip_one_newline(s: str) -> str:
    """Strip exactly one trailing newline (``\\r\\n`` or ``\\n``)."""
    if s.endswith("\r\n"):
        return s[:-2]
    if s.endswith("\n"):
        return s[:-1]
    return s


def read_secret_file(name: str, path: str) -> str:
    # newline="": no universal-newline translation. Text mode would turn a
    # value's own "\r\n" or lone "\r" into "\n" — the Rust reader
    # (std::fs::read_to_string) keeps the bytes, and so must this one.
    try:
        with open(path, encoding="utf-8", newline="") as fh:
            raw = fh.read()
    except OSError as e:
        raise SecretFileError(name, path, f"cannot be read ({e.strerror or e})") from e
    value = strip_one_newline(raw)
    if not value:
        raise SecretFileError(name, path, "is empty")
    return value


def secret(name: str, env: Mapping[str, str] | None = None) -> str | None:
    """The secret called *name*, or ``None`` when nothing is configured."""
    env = os.environ if env is None else env
    path = env.get(name + FILE_SUFFIX) or ""
    if path:
        return read_secret_file(name, path)
    return env.get(PREFIX + name) or env.get(name) or None


def both_set_conflicts(
    manifest: tuple[str, ...] = FILE_BACKED_SECRETS,
    env: Mapping[str, str] | None = None,
) -> list[str]:
    """Names in *manifest* where BOTH ``NAME`` and ``NAME_FILE`` are non-empty."""
    env = os.environ if env is None else env
    return [n for n in manifest if (env.get(n) or "") and (env.get(n + FILE_SUFFIX) or "")]


def refuse_on_conflicts(
    manifest: tuple[str, ...] = FILE_BACKED_SECRETS,
    env: Mapping[str, str] | None = None,
) -> None:
    """Boot-time check: raise :class:`SecretConflictError` on an env/file skew
    and :class:`SecretFileError` on a named file that cannot be read."""
    env = os.environ if env is None else env
    both = both_set_conflicts(manifest, env)
    if both:
        raise SecretConflictError(both)
    for n in manifest:
        secret(n, env)  # raises SecretFileError on a broken file
