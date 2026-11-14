# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC
"""The scale benchmark (noomesh-internal i785) only ever runs on a throw-away,
seeded database: dispatch-only, localhost-only, refuses a database with docs."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WF = ROOT / ".github/workflows/kb-scale-bench.yml"


def _bench():
    spec = importlib.util.spec_from_file_location("bench_chunk_search", ROOT / "scripts/bench_chunk_search.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_benchmark_is_dispatch_only():
    wf = yaml.safe_load(WF.read_text())
    on = wf.get("on", wf.get(True))
    assert set(on) == {"workflow_dispatch"}, f"the benchmark must never run on a schedule or a push: {set(on)}"


def test_the_workflow_only_ever_talks_to_its_own_container():
    text = WF.read_text()
    assert "docker run -d --name bench-pg" in text and "-p 127.0.0.1:5432:5432" in text
    for line in text.splitlines():
        if "postgresql" in line and "://" in line:
            assert "@localhost:5432/" in line, line


@pytest.mark.parametrize("dsn", [
    "postgresql+asyncpg://noo:x@pop-os:5432/noomesh",
    "postgresql+asyncpg://noo:x@100.64.0.7:5432/noomesh",
    "postgresql+asyncpg://noo:x@noo-postgres:5432/noomesh",
])
def test_the_script_refuses_any_database_but_localhost(dsn):
    with pytest.raises(SystemExit) as e:
        _bench().refuse_unless_local(dsn)
    assert "only runs on localhost" in str(e.value)


def test_the_script_accepts_localhost_and_refuses_a_database_with_docs():
    b = _bench()
    b.refuse_unless_local("postgresql+asyncpg://noo:x@localhost:5432/noomesh")
    src = (ROOT / "scripts/bench_chunk_search.py").read_text()
    assert 'if await conn.fetchval("SELECT count(*) FROM trellion.docs"):' in src
    assert "only runs on a database it seeded itself" in src
