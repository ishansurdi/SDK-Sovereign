"""Custom /play routes: serve the demo HTML and run agents server-side."""
from __future__ import annotations

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

	@app.get("/play")
	def play_index() -> FileResponse:
		return FileResponse(str(frontend_dir / "play.html"))

	@app.post("/play/reset")
	def play_reset(repo_id: str | None = None) -> dict[str, Any]:
		observation = play_env.reset(repo_id=repo_id)
		return _serialize(observation)

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
		return {"action": _serialize(action), "observation": _serialize(new_observation)}
