"""Rebuild Phase 8 plots from eval_results.json without re-running the models."""
from __future__ import annotations

import argparse
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def _load_eval_results(path: Path) -> dict[str, list[dict[str, Any]]]:
	"""Load baseline/trained eval results from disk."""
	payload = json.loads(path.read_text(encoding="utf-8"))
	baseline = payload.get("baseline")
	trained = payload.get("trained")
	if not isinstance(baseline, list) or not isinstance(trained, list):
		raise ValueError("eval_results.json must contain 'baseline' and 'trained' lists")
	return {"baseline": baseline, "trained": trained}


def _safe_mean(values: Iterable[float]) -> float:
	"""Return the arithmetic mean or zero for an empty iterable."""
	values = list(values)
	return statistics.mean(values) if values else 0.0


def _write_placeholder_curve(plot_path: Path, role: str) -> None:
	"""Write an honest placeholder plot when WandB metrics are unavailable."""
	import matplotlib.pyplot as plt

	plt.figure(figsize=(8, 4))
	plt.text(
		0.5,
		0.5,
		f"No training curve available for {role}.\nRun Phase 7 with WandB or export metrics first.",
		ha="center",
		va="center",
		fontsize=12,
	)
	plt.axis("off")
	plt.title(f"GRPO training - {role}")
	plt.tight_layout()
	plt.savefig(plot_path, dpi=150, bbox_inches="tight")
	plt.close()


def build_plots(eval_results_path: Path, plots_dir: Path) -> list[Path]:
	"""Generate the six PRD plot artifacts from saved evaluation results."""
	import matplotlib.pyplot as plt
	import numpy as np

	results = _load_eval_results(eval_results_path)
	baseline_results = results["baseline"]
	trained_results = results["trained"]
	plots_dir.mkdir(parents=True, exist_ok=True)

	created: list[Path] = []

	b_rate = sum(bool(item.get("success")) for item in baseline_results) / max(len(baseline_results), 1)
	t_rate = sum(bool(item.get("success")) for item in trained_results) / max(len(trained_results), 1)
	pass_rate_path = plots_dir / "pass_rate_baseline_vs_trained.png"
	plt.figure(figsize=(6, 4))
	plt.bar(["Baseline (untrained)", "Trained (two LoRAs)"], [b_rate, t_rate], color=["#bbbbbb", "#1f77b4"])
	plt.ylabel("Pass rate (all tests passed)")
	plt.title("SDK-Sovereign - pass rate")
	plt.ylim(0, 1)
	for index, value in enumerate([b_rate, t_rate]):
		plt.text(index, value + 0.02, f"{value:.0%}", ha="center", fontweight="bold")
	plt.tight_layout()
	plt.savefig(pass_rate_path, dpi=150, bbox_inches="tight")
	plt.close()
	created.append(pass_rate_path)

	mean_reward_path = plots_dir / "mean_reward.png"
	plt.figure(figsize=(6, 4))
	plt.bar(
		["Baseline", "Trained"],
		[_safe_mean(item.get("total_reward", 0.0) for item in baseline_results), _safe_mean(item.get("total_reward", 0.0) for item in trained_results)],
		color=["#bbbbbb", "#1f77b4"],
	)
	plt.ylabel("Mean episode reward")
	plt.axhline(0, color="k", lw=0.5)
	plt.title("Mean total reward per episode")
	plt.tight_layout()
	plt.savefig(mean_reward_path, dpi=150, bbox_inches="tight")
	plt.close()
	created.append(mean_reward_path)

	for role in ("lead", "auditor"):
		curve_path = plots_dir / f"reward_curve_{role}.png"
		_write_placeholder_curve(curve_path, role)
		created.append(curve_path)

	b_per: defaultdict[str, list[bool]] = defaultdict(list)
	t_per: defaultdict[str, list[bool]] = defaultdict(list)
	for item in baseline_results:
		b_per[str(item.get("repo", "unknown_repo"))].append(bool(item.get("success")))
	for item in trained_results:
		t_per[str(item.get("repo", "unknown_repo"))].append(bool(item.get("success")))
	repos = sorted(set(b_per) | set(t_per))
	per_repo_path = plots_dir / "per_repo_pass_rate.png"
	plt.figure(figsize=(8, 4))
	x = np.arange(len(repos))
	width = 0.35
	plt.bar(x - width / 2, [_safe_mean(b_per[repo]) for repo in repos], width, label="Baseline", color="#bbbbbb")
	plt.bar(x + width / 2, [_safe_mean(t_per[repo]) for repo in repos], width, label="Trained", color="#1f77b4")
	plt.xticks(x, repos, rotation=15)
	plt.ylabel("Pass rate")
	plt.legend()
	plt.title("Pass rate by repo")
	plt.tight_layout()
	plt.savefig(per_repo_path, dpi=150, bbox_inches="tight")
	plt.close()
	created.append(per_repo_path)

	completion_turns_path = plots_dir / "completion_turns.png"
	b_turns = [int(item.get("turns", 0)) for item in baseline_results if item.get("success")]
	t_turns = [int(item.get("turns", 0)) for item in trained_results if item.get("success")]
	plt.figure(figsize=(7, 4))
	plt.hist([b_turns, t_turns], bins=list(range(1, 9)), label=["Baseline", "Trained"], color=["#bbbbbb", "#1f77b4"], edgecolor="white")
	plt.xlabel("Turns to completion (successful episodes only)")
	plt.ylabel("Count")
	plt.legend()
	plt.title("Distribution of completion turns")
	plt.tight_layout()
	plt.savefig(completion_turns_path, dpi=150, bbox_inches="tight")
	plt.close()
	created.append(completion_turns_path)

	return created


def main() -> None:
	"""CLI entrypoint for regenerating plots from saved evaluation JSON."""
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--eval-results", default="eval_results.json", help="Path to eval_results.json")
	parser.add_argument("--plots-dir", default="plots", help="Directory for generated PNGs")
	args = parser.parse_args()

	created = build_plots(Path(args.eval_results), Path(args.plots_dir))
	print("Generated plots:")
	for path in created:
		size = os.path.getsize(path)
		print(f"- {path} ({size} bytes)")


if __name__ == "__main__":
	main()
