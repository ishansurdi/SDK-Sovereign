"""Custom /play routes: serve the demo HTML and run agents server-side."""
from __future__ import annotations

import difflib
import os
from pathlib import Path
from typing import Any, Callable, Optional

from fastapi import HTTPException
from fastapi.responses import FileResponse

PolicyMap = dict[str, Callable[[Any], Any]]


_AGENT_CACHE: dict[str, PolicyMap] = {}
_CURRENT_MODE = "rule"


def _serialize(value: Any) -> dict[str, Any]:
	"""Convert dataclass-style models into JSON-serialisable dictionaries."""
	if hasattr(value, "model_dump"):
		return value.model_dump()
	if hasattr(value, "__dict__"):
		return dict(value.__dict__)
	raise TypeError(f"Cannot serialize object of type {type(value)!r}")


def _resolve_env(env_source: Any) -> Any:
	"""Return a singleton environment instance from a class or instance input."""
	if isinstance(env_source, type):
		return env_source()
	if callable(env_source) and not hasattr(env_source, "step"):
		return env_source()
	return env_source


def _agent_mode() -> str:
	"""Report the currently selected play policy mode."""
	return _CURRENT_MODE


def _configured_live_modes() -> list[dict[str, str]]:
	"""Return the policy modes the current deployment can actually serve."""
	modes = [
		{"id": "rule", "label": "Rule fallback", "description": "Deterministic fallback for demo reliability."},
	]
	if os.environ.get("SDK_SOVEREIGN_AGENTS_LIVE"):
		modes.append(
			{"id": "baseline", "label": "Baseline model", "description": "Base model with fresh adapters, before RL improvement."}
		)
		lead_repo = os.environ.get("SDK_SOVEREIGN_LEAD_ADAPTER_REPO")
		auditor_repo = os.environ.get("SDK_SOVEREIGN_AUDITOR_ADAPTER_REPO")
		shared_repo = os.environ.get("SDK_SOVEREIGN_ADAPTER_REPO")
		if shared_repo or (lead_repo and auditor_repo):
			modes.append(
				{"id": "trained", "label": "Trained adapters", "description": "Loads RL-tuned adapters so behavior can differ from baseline."}
			)
	return modes


def _mode_diagnostics() -> dict[str, Any]:
	"""Explain which policy modes are available and what is still missing."""
	live_enabled = bool(os.environ.get("SDK_SOVEREIGN_AGENTS_LIVE"))
	lead_repo, auditor_repo = _resolve_adapter_repos()
	diagnostics = {
		"live_enabled": live_enabled,
		"available_mode_ids": [item["id"] for item in _configured_live_modes()],
		"notes": [],
	}
	if not live_enabled:
		diagnostics["notes"].append("Set SDK_SOVEREIGN_AGENTS_LIVE=1 to enable model-generated baseline and trained modes.")
	if live_enabled and not (lead_repo and auditor_repo):
		diagnostics["notes"].append(
			"Configure SDK_SOVEREIGN_ADAPTER_REPO or both SDK_SOVEREIGN_LEAD_ADAPTER_REPO and SDK_SOVEREIGN_AUDITOR_ADAPTER_REPO to expose trained mode."
		)
	if live_enabled:
		diagnostics["notes"].append("Baseline mode uses the base model with fresh adapters. Trained mode uses RL-tuned adapters when configured.")
	return diagnostics


def _current_issue(play_env: Any) -> dict[str, Any]:
	"""Return issue-like repo metadata for the active or requested repository."""
	state = play_env.state
	repo_id = state.repo_id if state is not None else None
	if repo_id is None:
		return {}
	repo = play_env.repos[repo_id]
	return {
		"repo_id": repo_id,
		"category": repo.get("category"),
		"entrypoint": repo.get("entrypoint"),
		"deprecated_sdk": repo.get("deprecated_sdk"),
		"target_sdk": repo.get("ground_truth_replacement"),
		"issue_title": f"Migrate {repo_id} off {repo.get('deprecated_sdk')}",
		"issue_summary": repo.get("error_log"),
	}


def _build_patch_preview(play_env: Any, action: Any) -> Optional[str]:
	"""Build a unified diff preview when a patch is submitted."""
	if not getattr(action, "patched_code", None) or play_env.state is None:
		return None
	repo = play_env.repos[play_env.state.repo_id]
	before = (repo.get("broken_code") or "").splitlines()
	after = (action.patched_code or "").splitlines()
	diff = difflib.unified_diff(before, after, fromfile="broken.py", tofile="patched.py", lineterm="")
	return "\n".join(list(diff)[:40])


def _summarize_feedback(observation: Any) -> str:
	"""Render reward feedback as a readable policy-learning signal."""
	breakdown = getattr(observation, "reward_breakdown", {}) or {}
	if not breakdown:
		return "No explicit reward feedback on this step."
	parts: list[str] = []
	for key, value in breakdown.items():
		direction = "+" if value > 0 else ""
		parts.append(f"{key}={direction}{value:.2f}")
	return "Reward signal: " + ", ".join(parts)


def _build_transcript_entry(play_env: Any, action: Any, observation: Any) -> dict[str, Any]:
	"""Expose a richer step narrative for the frontend."""
	proposal = getattr(action, "proposed_sdk", None) or getattr(observation, "current_proposal", None)
	return {
		"role": getattr(action, "role", None),
		"action_type": getattr(action, "action_type", None),
		"discussion": getattr(action, "reasoning", None) or "No explicit reasoning emitted.",
		"proposal": proposal,
		"approved_replacement": getattr(observation, "approved_replacement", None),
		"issue_update": getattr(action, "rejection_reason", None)
		or (f"Patch prepared for {proposal}." if getattr(action, "patched_code", None) else None)
		or "Issue is being triaged.",
		"learning_signal": _summarize_feedback(observation),
		"reward": getattr(observation, "reward", 0.0),
		"reward_breakdown": getattr(observation, "reward_breakdown", {}) or {},
		"patch_preview": _build_patch_preview(play_env, action),
		"policy_mode": _agent_mode(),
	}


def _load_rule_agents() -> PolicyMap:
	"""Return deterministic fallback agents."""
	from server.rule_agents import auditor_rule_agent, lead_rule_agent

	return {"auditor": auditor_rule_agent, "lead": lead_rule_agent}


def _resolve_adapter_repos() -> tuple[Optional[str], Optional[str]]:
	"""Resolve lead and auditor adapter repositories from environment variables."""
	shared_repo = os.environ.get("SDK_SOVEREIGN_ADAPTER_REPO")
	if shared_repo:
		return f"{shared_repo}/lead", f"{shared_repo}/auditor"
	return os.environ.get("SDK_SOVEREIGN_LEAD_ADAPTER_REPO"), os.environ.get("SDK_SOVEREIGN_AUDITOR_ADAPTER_REPO")


def _load_model_agents(mode: str) -> PolicyMap:
	"""Load baseline or trained model-driven agents for the play demo."""
	from server.llm_agents import load_model_with_two_adapters, make_agent_pair

	model, tokenizer = load_model_with_two_adapters()
	agents = make_agent_pair(model, tokenizer)
	if mode == "trained":
		lead_repo, auditor_repo = _resolve_adapter_repos()
		if not lead_repo or not auditor_repo:
			raise RuntimeError("trained mode requested but adapter repositories are not configured")
		model.load_adapter(lead_repo, adapter_name="lead_adapter_trained")
		model.load_adapter(auditor_repo, adapter_name="auditor_adapter_trained")
		agents["lead"].adapter_name = "lead_adapter_trained"
		agents["auditor"].adapter_name = "auditor_adapter_trained"
	return agents


def _load_agents(mode: str) -> PolicyMap:
	"""Load or reuse agents for the requested policy mode."""
	if mode in _AGENT_CACHE:
		return _AGENT_CACHE[mode]

	if mode == "rule":
		agents = _load_rule_agents()
		_AGENT_CACHE[mode] = agents
		return agents

	if not os.environ.get("SDK_SOVEREIGN_AGENTS_LIVE"):
		raise RuntimeError(f"{mode} mode requested but SDK_SOVEREIGN_AGENTS_LIVE is not enabled")

	agents = _load_model_agents(mode)
	_AGENT_CACHE[mode] = agents
	return agents


def _select_mode(requested_mode: Optional[str]) -> str:
	"""Validate and persist the requested policy mode."""
	global _CURRENT_MODE
	mode = requested_mode or "rule"
	available = {item["id"] for item in _configured_live_modes()}
	if mode not in available:
		raise HTTPException(status_code=400, detail=f"unsupported mode '{mode}'")
	_CURRENT_MODE = mode
	return mode


def register_play_routes(app: Any, env: Any) -> None:
	"""Register the Phase 9 HTML demo plus server-side stepping routes."""
	frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
	play_env = _resolve_env(env)

	@app.get("/")
	def index() -> FileResponse:
		return FileResponse(str(frontend_dir / "play.html"))

	@app.get("/play")
	def play_index() -> FileResponse:
		return FileResponse(str(frontend_dir / "play.html"))

	@app.get("/play/catalog")
	def play_catalog() -> dict[str, Any]:
		items = []
		for repo_id, repo in sorted(play_env.repos.items()):
			items.append(
				{
					"repo_id": repo_id,
					"category": repo.get("category"),
					"deprecated_sdk": repo.get("deprecated_sdk"),
					"target_sdk": repo.get("ground_truth_replacement"),
					"entrypoint": repo.get("entrypoint"),
					"issue_summary": repo.get("error_log"),
				}
			)
		return {
			"repos": items,
			"agent_mode": _agent_mode(),
			"available_modes": _configured_live_modes(),
			"diagnostics": _mode_diagnostics(),
		}

	@app.post("/play/reset")
	def play_reset(repo_id: str | None = None, mode: str | None = None) -> dict[str, Any]:
		selected_mode = _select_mode(mode)
		observation = play_env.reset(repo_id=repo_id)
		return {
			"observation": _serialize(observation),
			"issue": _current_issue(play_env),
			"agent_mode": selected_mode,
		}

	@app.get("/play/state")
	def play_state() -> dict[str, Any]:
		if play_env.state is None:
			raise HTTPException(status_code=400, detail="call /play/reset first")
		return _serialize(play_env.state)

	@app.post("/play/agent_step")
	def play_agent_step() -> dict[str, Any]:
		if play_env.state is None:
			raise HTTPException(status_code=400, detail="call /play/reset first")
		try:
			agents = _load_agents(_agent_mode())
		except Exception as exc:
			raise HTTPException(status_code=503, detail=str(exc)) from exc
		observation = play_env._build_observation(play_env._next_role(), last_reward=0.0)
		action = agents[observation.current_role](observation)
		new_observation = play_env.step(action)
		return {
			"action": _serialize(action),
			"observation": _serialize(new_observation),
			"issue": _current_issue(play_env),
			"transcript_entry": _build_transcript_entry(play_env, action, new_observation),
			"agent_mode": _agent_mode(),
		}
