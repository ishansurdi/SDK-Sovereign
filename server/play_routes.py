"""Custom /play routes: serve the demo HTML and run agents server-side."""
from __future__ import annotations

import difflib
import os
from pathlib import Path
from typing import Any, Callable, Optional

from fastapi import HTTPException
from fastapi.responses import FileResponse


_AGENTS: Optional[dict[str, Callable[[Any], Any]]] = None
_LOADED = False


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
	"""Report whether the play demo is using live or rule agents."""
	return "live" if os.environ.get("SDK_SOVEREIGN_AGENTS_LIVE") else "rule"


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
	}


def _try_load_trained_agents() -> dict[str, Callable[[Any], Any]]:
	"""Load live agents when enabled, otherwise fall back to rule agents."""
	global _AGENTS, _LOADED
	if _LOADED and _AGENTS is not None:
		return _AGENTS

	_LOADED = True
	if not os.environ.get("SDK_SOVEREIGN_AGENTS_LIVE"):
		from server.rule_agents import auditor_rule_agent, lead_rule_agent

		_AGENTS = {"auditor": auditor_rule_agent, "lead": lead_rule_agent}
		return _AGENTS

	try:
		from server.llm_agents import load_model_with_two_adapters, make_agent_pair

		model, tokenizer = load_model_with_two_adapters()
		adapter_repo = os.environ.get("SDK_SOVEREIGN_ADAPTER_REPO")
		if adapter_repo:
			model.load_adapter(f"{adapter_repo}/lead", adapter_name="lead_adapter")
			model.load_adapter(f"{adapter_repo}/auditor", adapter_name="auditor_adapter")
		_AGENTS = make_agent_pair(model, tokenizer)
	except Exception:
		from server.rule_agents import auditor_rule_agent, lead_rule_agent

		_AGENTS = {"auditor": auditor_rule_agent, "lead": lead_rule_agent}
	return _AGENTS


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
		return {"repos": items, "agent_mode": _agent_mode()}

	@app.post("/play/reset")
	def play_reset(repo_id: str | None = None) -> dict[str, Any]:
		observation = play_env.reset(repo_id=repo_id)
		return {
			"observation": _serialize(observation),
			"issue": _current_issue(play_env),
			"agent_mode": _agent_mode(),
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
		agents = _try_load_trained_agents()
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
