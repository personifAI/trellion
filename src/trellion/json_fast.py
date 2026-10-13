# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""Fast JSON — drop-in ``json`` replacement backed by orjson.

Usage (preferred — module-level alias keeps existing call sites unchanged)::

    from noomesh import json_fast as json   # relative: from . import json_fast as json

    json.dumps(obj)
    json.loads(data)
    json.dump(obj, fp, indent=2)
    json.JSONDecodeError

``dumps`` always returns ``str`` (not bytes) for drop-in compat with
stdlib ``json.dumps``.  ``loads`` accepts both ``str`` and ``bytes``.

Extra kwargs accepted by stdlib (``sort_keys``, ``ensure_ascii``,
``separators``) are mapped to orjson options where possible and silently
ignored otherwise — this keeps call sites 100 % unchanged.
"""

from __future__ import annotations

from json import JSONDecodeError  # noqa: F401 — re-export for consumers
from typing import IO, Any

try:
    import orjson as _orjson

    def dumps(
        obj: Any,
        *,
        default: Any = None,
        indent: Any = None,
        sort_keys: bool = False,
        ensure_ascii: bool = True,
        separators: tuple[str, str] | None = None,
    ) -> str:
        """Serialize *obj* to a JSON string (orjson backend)."""
        opts = _orjson.OPT_NON_STR_KEYS
        if indent:
            opts |= _orjson.OPT_INDENT_2
        if sort_keys:
            opts |= _orjson.OPT_SORT_KEYS
        # ensure_ascii / separators: orjson always writes UTF-8 compact
        return _orjson.dumps(obj, default=default, option=opts).decode()

    def loads(data: str | bytes) -> Any:
        """Deserialize JSON string or bytes (orjson backend)."""
        return _orjson.loads(data)

    BACKEND = "orjson"

except ImportError:
    import json as _json

    def dumps(  # type: ignore[misc]
        obj: Any,
        *,
        default: Any = None,
        indent: Any = None,
        sort_keys: bool = False,
        ensure_ascii: bool = True,
        separators: tuple[str, str] | None = None,
    ) -> str:
        """Serialize *obj* to a JSON string (stdlib fallback)."""
        kw: dict[str, Any] = {}
        if default is not None:
            kw["default"] = default
        if indent is not None and indent is not False:
            kw["indent"] = 2 if indent is True else indent
        if sort_keys:
            kw["sort_keys"] = True
        if not ensure_ascii:
            kw["ensure_ascii"] = False
        if separators is not None:
            kw["separators"] = separators
        return _json.dumps(obj, **kw)

    def loads(data: str | bytes) -> Any:  # type: ignore[misc]
        """Deserialize JSON string or bytes (stdlib fallback)."""
        return _json.loads(data)

    BACKEND = "json"


# ---------------------------------------------------------------------------
# File I/O helpers (thin wrappers so callers can use json.dump / json.load)
# ---------------------------------------------------------------------------

def dump(obj: Any, fp: IO[str], **kw: Any) -> None:
    """Serialize *obj* as JSON and write to *fp* (text-mode file)."""
    fp.write(dumps(obj, **kw))


def load(fp: IO[str], **kw: Any) -> Any:
    """Deserialize JSON from *fp* (text-mode file)."""
    return loads(fp.read())
