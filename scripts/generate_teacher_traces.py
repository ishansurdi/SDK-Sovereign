"""Generate successful teacher traces and persist them as JSONL."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.training_data import export_traces_jsonl, generate_teacher_traces


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate teacher traces for SDK-Sovereign")
    parser.add_argument("--episodes-per-repo", type=int, default=50)
    parser.add_argument("--output", default="logs/teacher_traces.jsonl")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    traces = generate_teacher_traces(
        episodes_per_repo=args.episodes_per_repo,
        seed=args.seed,
    )
    output_path = Path(args.output)
    export_traces_jsonl(traces, output_path)
    successes = sum(1 for trace in traces if trace.done and trace.success)
    print(f"Wrote {len(traces)} trace rows to {output_path}")
    print(f"Successful terminal turns: {successes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
