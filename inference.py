"""Smoke runner for SDK-Sovereign API and demo endpoints."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from models import SDKAction


@dataclass
class CheckResult:
	"""Structured result for one smoke-check step."""

	name: str
	passed: bool
	details: str


class HttpRunner:
	"""Small HTTP helper that speaks to the deployed API."""

	def __init__(self, base_url: str):
		self.base_url = base_url.rstrip("/")

	def get_json(self, path: str) -> dict[str, Any]:
		request = Request(f"{self.base_url}{path}", method="GET")
		with urlopen(request, timeout=30) as response:
			return json.loads(response.read().decode("utf-8"))

	def post_json(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
		body = None if payload is None else json.dumps(payload).encode("utf-8")
		request = Request(
			f"{self.base_url}{path}",
			data=body,
			headers={"Content-Type": "application/json"},
			method="POST",
		)
		with urlopen(request, timeout=30) as response:
			return json.loads(response.read().decode("utf-8"))


def _ok(name: str, details: str) -> CheckResult:
	return CheckResult(name=name, passed=True, details=details)


def _fail(name: str, details: str) -> CheckResult:
	return CheckResult(name=name, passed=False, details=details)


def check_openenv_flow(base_url: str, repo_id: str) -> list[CheckResult]:
	"""Verify the core OpenEnv reset/step contract via the typed client."""
	from client import SDKSovereignEnv

	results: list[CheckResult] = []
	with SDKSovereignEnv(base_url=base_url).sync() as env:
		observation = env.reset(repo_id=repo_id)
		if observation.current_role != "auditor":
			results.append(_fail("openenv.reset", f"expected auditor first, got {observation.current_role}"))
		else:
			results.append(_ok("openenv.reset", f"repo={repo_id} current_role={observation.current_role}"))

		next_observation = env.step(SDKAction(role="auditor", action_type="pass", reasoning="smoke test"))
		if next_observation.current_role != "lead":
			results.append(_fail("openenv.step", f"expected lead next, got {next_observation.current_role}"))
		else:
			results.append(_ok("openenv.step", f"reward={next_observation.reward:+.2f} next_role={next_observation.current_role}"))
	return results


def check_demo_routes(base_url: str, repo_id: str, mode: str) -> list[CheckResult]:
	"""Verify the public demo endpoints used by the HF Space UI."""
	runner = HttpRunner(base_url)
	results: list[CheckResult] = []

	health_payload = runner.get_json("/health")
	if health_payload.get("status") == "healthy":
		results.append(_ok("health", json.dumps(health_payload)))
	else:
		results.append(_fail("health", json.dumps(health_payload)))

	catalog = runner.get_json("/play/catalog")
	mode_ids = [item["id"] for item in catalog.get("available_modes", [])]
	if "rule" in mode_ids:
		results.append(_ok("play.catalog", f"modes={mode_ids}"))
	else:
		results.append(_fail("play.catalog", f"missing rule mode: {mode_ids}"))

	query = urlencode({"repo_id": repo_id, "mode": mode})
	reset_payload = runner.post_json(f"/play/reset?{query}")
	observation = reset_payload.get("observation", {})
	if observation.get("current_role") == "auditor":
		results.append(_ok("play.reset", f"mode={reset_payload.get('agent_mode')} repo={repo_id}"))
	else:
		results.append(_fail("play.reset", json.dumps(reset_payload)))

	step_payload = runner.post_json("/play/agent_step")
	if step_payload.get("agent_mode") == mode and step_payload.get("transcript_entry"):
		entry = step_payload["transcript_entry"]
		results.append(_ok("play.agent_step", f"role={entry.get('role')} action={entry.get('action_type')}"))
	else:
		results.append(_fail("play.agent_step", json.dumps(step_payload)))

	state_payload = runner.get_json("/play/state")
	if state_payload.get("repo_id") == repo_id:
		results.append(_ok("play.state", f"terminated={state_payload.get('terminated_reason')} step_count={state_payload.get('step_count')}"))
	else:
		results.append(_fail("play.state", json.dumps(state_payload)))

	return results


def main() -> int:
	"""Run the end-to-end API smoke checks and print a concise report."""
	parser = argparse.ArgumentParser(description="SDK-Sovereign smoke runner")
	parser.add_argument("--base-url", default=None, help="Target server URL, for example http://127.0.0.1:8000")
	parser.add_argument("--repo-id", default="payments_repo", choices=["payments_repo", "maps_repo", "comms_repo"])
	parser.add_argument("--mode", default="rule", help="Demo policy mode to exercise")
	args = parser.parse_args()

	base_url = args.base_url
	if not base_url:
		print("Missing --base-url. Example: python inference.py --base-url http://127.0.0.1:8000")
		return 2

	results: list[CheckResult] = []
	try:
		results.extend(check_openenv_flow(base_url, args.repo_id))
		results.extend(check_demo_routes(base_url, args.repo_id, args.mode))
	except (HTTPError, URLError, OSError) as exc:
		print(f"Request failed: {exc}")
		return 1
	except Exception as exc:  # pragma: no cover - defensive smoke script
		print(f"Smoke run crashed: {exc}")
		return 1

	passed = sum(1 for result in results if result.passed)
	for result in results:
		status = "PASS" if result.passed else "FAIL"
		print(f"[{status}] {result.name}: {result.details}")

	print(f"\nSummary: {passed}/{len(results)} checks passed")
	return 0 if passed == len(results) else 1


if __name__ == "__main__":
	raise SystemExit(main())
