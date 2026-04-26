"""Tests for teacher trace generation and SFT export."""
from __future__ import annotations

import json
from pathlib import Path

from server.training_data import export_sft_jsonl, export_traces_jsonl, generate_teacher_traces


def test_teacher_traces_are_successful(tmp_path: Path) -> None:
    """Teacher traces should end in successful submissions for the shipped repos."""
    traces = generate_teacher_traces(episodes_per_repo=1, logs_root=tmp_path / "logs")
    terminal_steps = [trace for trace in traces if trace.done]
    assert terminal_steps
    assert all(trace.success for trace in terminal_steps)


def test_trace_and_sft_exports_write_jsonl(tmp_path: Path) -> None:
    """Teacher traces should export cleanly to trace and SFT JSONL files."""
    traces = generate_teacher_traces(episodes_per_repo=1, logs_root=tmp_path / "logs")
    traces_path = tmp_path / "teacher_traces.jsonl"
    sft_path = tmp_path / "teacher_sft.jsonl"

    export_traces_jsonl(traces, traces_path)
    export_sft_jsonl(traces, sft_path)

    trace_rows = [json.loads(line) for line in traces_path.read_text(encoding="utf-8").splitlines()]
    sft_rows = [json.loads(line) for line in sft_path.read_text(encoding="utf-8").splitlines()]
    assert trace_rows
    assert sft_rows
    assert "messages" in sft_rows[0]
    assert sft_rows[0]["messages"][0]["role"] == "system"
