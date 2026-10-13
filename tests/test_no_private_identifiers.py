# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""The tracked tree must not carry private network addresses, credentials or
host-specific paths: this repository is published.

Two layers:

* GENERIC patterns live here (private-network ranges, tailnet names,
  credential shapes, host-specific path shapes). They always run.
* The list of private NAMES (people, hosts, products, add-on internals) is NOT
  stored in this repository, because a list of banned names is itself a leak.
  It is read at run time from the file named by ``TRELLION_PRIVATE_NAMES_FILE``
  (one case-insensitive literal per line; ``#`` starts a comment; a line
  prefixed ``addon:`` names an add-on internal, which the seam files in
  ``ADDON_SEAM`` may carry). When the variable is unset or the file is absent,
  the name checks skip with a notice.
"""

import os
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
NAMES_ENV = "TRELLION_PRIVATE_NAMES_FILE"

# Generic shapes, safe to publish.
GENERIC = [
    re.compile(r"\b100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.\d{1,3}\.\d{1,3}\b"),  # tailnet CGNAT
    re.compile(r"\b10\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),
    re.compile(r"\b192\.168\.\d{1,3}\.\d{1,3}\b"),
    re.compile(r"\b172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}\b"),
    re.compile(r"(?<!example)\.ts\.net\b"),  # example.ts.net is the neutral fixture
    re.compile(r"-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bsk-[A-Za-z0-9]{32,}\b"),
    re.compile(r"/mnt/[a-z]/"),
    re.compile(r"[A-Za-z]:\\Users\\"),
]

# Files that name the add-on because the seam needs its name to find and load
# it. Reason per path.
ADDON_SEAM = {
    "src/trellion/live/_intel.py": "the loader: the package name and the tier error message",
    "src/trellion/live/gateway_auth.py": "service identity of the add-on's own container",
    "tests/test_intel_loader.py": "exercises the loader with a stand-in package",
    "tests/test_basic_without_intel.py": "blocks the add-on to prove the open tier stands alone",
    "tests/test_basic_tier_fallback.py": "blocks the add-on to prove the open tier stands alone",
    "tests/test_startup_tier_and_engine.py": "blocks the add-on to prove the open tier stands alone",
    "tests/test_source_priors_base_tier.py": "blocks the add-on to prove the open tier stands alone",
    "tests/test_live_bench.py": "skips when the add-on is not installed",
    "tests/test_gateway_hmac_auth.py": "asserts the add-on's service identity in the error",
    "COMMERCIAL-LICENSE.md": "licence text: states that the add-on is separately licensed",
}

# Maintainer-only paths the public snapshot removes. Reason per path.
ALLOWED_PATHS = {
    "docs/WORKTREES.md": "maintainer workflow doc; dropped from the public cut",
}
ALLOWED_PREFIXES = (
    ".githooks/",  # maintainer policy hooks; dropped from the public cut
)


def _tracked():
    out = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True
    ).stdout.decode()
    return [p for p in out.split("\0") if p]


def _text_files(skip=lambda rel: False):
    for rel in _tracked():
        if skip(rel):
            continue
        path = ROOT / rel
        if not path.is_file():
            continue
        try:
            yield rel, path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue


def _private_names():
    """(names, addon_names) from the external file, or None when absent."""
    raw = os.environ.get(NAMES_ENV)
    if not raw or not Path(raw).is_file():
        return None
    names, addon = [], []
    for line in Path(raw).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("addon:"):
            addon.append(line[len("addon:"):].strip())
        else:
            names.append(line)
    return names, addon


def _compile(items):
    return re.compile("|".join(re.escape(i) for i in items), re.IGNORECASE) if items else None


def _skip_notice():
    pytest.skip(f"{NAMES_ENV} not set (or file absent): private-name checks skipped")


def _scan(pattern_search, skip):
    hits = []
    for rel, text in _text_files(skip):
        for n, line in enumerate(text.splitlines(), 1):
            if pattern_search(line):
                hits.append(f"{rel}:{n}")
    return hits


def _is_maintainer_only(rel):
    return rel in ALLOWED_PATHS or rel.startswith(ALLOWED_PREFIXES)


def test_tracked_tree_has_no_generic_private_shapes():
    hits = _scan(lambda line: any(p.search(line) for p in GENERIC), _is_maintainer_only)
    assert not hits, "private-network or credential shapes in the tracked tree:\n" + "\n".join(hits)


def test_tracked_tree_has_no_private_names():
    loaded = _private_names()
    if loaded is None:
        _skip_notice()
    names, addon = loaded
    pattern = _compile(names + addon)
    # Add-on names are allowed at the seam; every other name is allowed nowhere.
    plain = _compile(names)
    hits = []
    for rel, text in _text_files(_is_maintainer_only):
        pat = plain if rel in ADDON_SEAM else pattern
        if pat is None:
            continue
        for n, line in enumerate(text.splitlines(), 1):
            if pat.search(line):
                hits.append(f"{rel}:{n}")
    assert not hits, "private names in the tracked tree (locations only):\n" + "\n".join(hits)


def test_seam_waivers_are_not_stale():
    loaded = _private_names()
    if loaded is None:
        _skip_notice()
    _, addon = loaded
    pattern = _compile(addon)
    tracked = set(_tracked())
    stale = [
        p for p in ADDON_SEAM
        if p not in tracked or pattern is None or not pattern.search((ROOT / p).read_text(encoding="utf-8"))
    ]
    assert not stale, f"ADDON_SEAM entries that no longer need a waiver: {stale}"
