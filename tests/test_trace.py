# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""Unit tests for the telemetry module."""

from __future__ import annotations

import json
import time

from trellion.common.trace import StageTrace, TraceCollector, stage_timer


def test_stage_trace_defaults():
    st = StageTrace(stage="ingest")
    assert st.stage == "ingest"
    assert st.wall_ms == 0.0
    assert st.records_in == 0
    assert st.records_out == 0
    assert st.tokens_in == 0
    assert st.tokens_out == 0
    assert st.notes == {}


def test_stage_timer_records_wall_time():
    traces = TraceCollector()
    with stage_timer(traces, "sleep", records_in=3) as st:
        time.sleep(0.01)
        st.records_out = 7
    entries = traces.as_list()
    assert len(entries) == 1
    entry = entries[0]
    assert entry.stage == "sleep"
    assert entry.records_in == 3
    assert entry.records_out == 7
    # Should have measured at least a few ms of wall time
    assert entry.wall_ms >= 5.0
    assert entry.started_at > 0.0


def test_stage_timer_commits_even_on_exception():
    traces = TraceCollector()
    try:
        with stage_timer(traces, "boom") as st:
            st.records_out = 1
            raise RuntimeError("kaboom")
    except RuntimeError:
        pass
    # The trace should still be recorded for debugging
    assert len(traces.as_list()) == 1
    assert traces.as_list()[0].stage == "boom"


def test_summary_aggregates():
    traces = TraceCollector()
    with stage_timer(traces, "a") as st:
        st.llm_calls = 2
        st.tokens_in = 100
        st.tokens_out = 50
    with stage_timer(traces, "b") as st:
        st.llm_calls = 3
        st.tokens_in = 200
        st.tokens_out = 80
    s = traces.summary()
    assert s["stages"] == 2
    assert s["total_llm_calls"] == 5
    assert s["total_tokens_in"] == 300
    assert s["total_tokens_out"] == 130
    assert s["total_wall_ms"] >= 0.0


def test_dump_roundtrip(tmp_path):
    traces = TraceCollector()
    with stage_timer(traces, "ingest") as st:
        st.records_out = 4
        st.notes = {"chunks": 4}
    dst = tmp_path / "trace.jsonl"
    traces.dump(dst)
    lines = dst.read_text().strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["stage"] == "ingest"
    assert row["records_out"] == 4
    assert row["notes"] == {"chunks": 4}
    assert "wall_ms" in row
