"""Export chat-format SFT rows from teacher traces."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.training_data import TraceStep, export_sft_jsonl


def load_traces(path: Path) -> list[TraceStep]:
    """Load trace rows from JSONL."""
    traces: list[TraceStep] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        payload = json.loads(line)
        traces.append(TraceStep(**payload))
    return traces


def main() -> int:
    parser = argparse.ArgumentParser(description="Export teacher traces as SFT chat data")
    parser.add_argument("--input", default="logs/teacher_traces.jsonl")
    parser.add_argument("--output", default="logs/teacher_sft.jsonl")
    parser.add_argument("--include-failures", action="store_true")
    args = parser.parse_args()

    traces = load_traces(Path(args.input))
    export_sft_jsonl(
        traces,
        Path(args.output),
        success_only=not args.include_failures,
    )
    print(f"Wrote SFT dataset to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
