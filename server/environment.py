"""Main SDK-Sovereign environment implementation."""
from __future__ import annotations

import json
import random
import uuid
from pathlib import Path
from typing import Dict, List, Optional

try:
	from openenv.core.env_server.interfaces import Environment
except ImportError:
	class Environment:  # type: ignore[no-redef]
		def __init__(self, *args, **kwargs):
			pass

from models import ActionType, Role, SDKAction, SDKObservation, SDKState
from server.rubric import SDKMigrationRubric
from server.verifier import Verifier


class SDKSovereignEnvironment(Environment[SDKAction, SDKObservation, SDKState]):
	"""Environment that alternates auditor/lead turns under partial observability."""

	MAX_TURNS = 7
	SUPPORTS_CONCURRENT_SESSIONS = True

	def __init__(self, repos_root: Optional[Path] = None, seed: int = 0):
		"""Initialise environment repositories, verifier, and rubric."""
		super().__init__()
		repos_root = repos_root or Path(__file__).parent / "repos"
		self.repos_root = Path(repos_root)
		self.allowlist_path = Path(__file__).parent / "allowlist.json"
		self._load_allowlist()
		self._discover_repos()
		self.verifier = Verifier(self.repos_root)
		self.rubric = SDKMigrationRubric(
			allowlist=self.allowlist,
			deprecated_sdks=[repo["deprecated_sdk"] for repo in self.repos.values()],
		)
		self._state: Optional[SDKState] = None
		self._history: List[Dict] = []
		self._rng = random.Random(seed)

	def _load_allowlist(self) -> None:
		"""Load sovereign allowlist metadata from disk."""
		data = json.loads(self.allowlist_path.read_text())
		self.allowlist = data["allowlist"]
		self.allowlist_metadata = data["metadata"]

	def _discover_repos(self) -> None:
		"""Discover all configured repositories from repos_root."""
		self.repos: Dict[str, Dict] = {}
		for directory in self.repos_root.iterdir():
			if not directory.is_dir() or directory.name.startswith("_"):
				continue
			meta_path = directory / "meta.json"
			if not meta_path.exists():
				continue
			meta = json.loads(meta_path.read_text())
			meta["broken_code"] = (directory / "broken.py").read_text()
			self.repos[meta["repo_id"]] = meta

	def reset(
		self,
		seed: Optional[int] = None,
		episode_id: Optional[str] = None,
		**kwargs,
	) -> SDKObservation:
		"""Start a new episode and return the first auditor observation."""
		if seed is not None:
			self._rng.seed(seed)
		repo_id = self._rng.choice(list(self.repos.keys()))
		repo = self.repos[repo_id]
		self._state = SDKState(
			episode_id=episode_id or str(uuid.uuid4())[:8],
			repo_id=repo_id,
			deprecated_sdk=repo["deprecated_sdk"],
			ground_truth_replacement=repo["ground_truth_replacement"],
		)
		self._history = []
		return self._build_observation(Role.AUDITOR.value, last_reward=0.0)

	def step(
		self,
		action: SDKAction,
		timeout_s: Optional[float] = None,
		**kwargs,
	) -> SDKObservation:
		"""Apply one action and return the next role-masked observation."""
		if self._state is None:
			raise ValueError("Call reset() before step().")

		expected_role = self._next_role()
		if action.role != expected_role:
			self._state.step_count += 1
			self._history.append(
				{
					"turn": self._state.step_count,
					"role": action.role,
					"expected_role": expected_role,
					"action_type": "WRONG_ROLE",
					"reward": -1.0,
				}
			)
			return self._build_observation(
				self._next_role(),
				last_reward=-1.0,
				breakdown={"wrong_role_penalty": -1.0},
			)

		self._apply_action(action)

		result = self.rubric.score_step(action, self._state, self.verifier)
		step_reward = result.total
		breakdown = dict(result.components)

		self._history.append(
			{
				"turn": self._state.step_count,
				"role": action.role,
				"action_type": action.action_type,
				"proposed_sdk": action.proposed_sdk,
				"rejection_reason": (action.rejection_reason or "")[:200],
				"reasoning": (action.reasoning or "")[:300],
				"reward": step_reward,
			}
		)

		done = False
		if action.action_type == ActionType.SUBMIT_PATCH.value:
			done = True
			self._state.terminated_reason = "submitted"
			self._state.final_patch = action.patched_code
			self._state.test_results = self.verifier.run_parity_tests(
				action.patched_code or "", self._state.repo_id
			)
		elif self._state.step_count >= self.MAX_TURNS - 1:
			done = True
			self._state.terminated_reason = "max_turns"

		if done:
			terminal_breakdown = self.rubric.score_terminal(self._state, self._state.terminated_reason)
			for key, value in terminal_breakdown.items():
				breakdown[key] = breakdown.get(key, 0.0) + value
				step_reward += value

		self._state.cumulative_reward_by_role[action.role] += step_reward
		self._state.step_count += 1

		next_role = self._next_role() if not done else action.role
		return self._build_observation(next_role, done=done, last_reward=step_reward, breakdown=breakdown)

	@property
	def state(self) -> SDKState:
		"""Return full unmasked environment state."""
		return self._state

	def _next_role(self) -> str:
		"""Return whose turn it is based on step parity."""
		return Role.AUDITOR.value if self._state.step_count % 2 == 0 else Role.LEAD.value

	def _apply_action(self, action: SDKAction) -> None:
		"""Apply state mutations implied by a valid action."""
		if action.action_type == ActionType.PROPOSE_REPLACEMENT.value and action.proposed_sdk:
			self._state.proposals_history.append(action.proposed_sdk)
		elif action.action_type == ActionType.APPROVE.value:
			if self._state.proposals_history:
				self._state.approved_replacement = self._state.proposals_history[-1]
		elif action.action_type == ActionType.REJECT.value:
			if self._state.proposals_history:
				self._state.rejected_so_far.append(self._state.proposals_history[-1])

	def _build_observation(
		self,
		role: str,
		done: bool = False,
		last_reward: float = 0.0,
		breakdown: Optional[Dict] = None,
	) -> SDKObservation:
		"""Build a role-masked observation from the current state."""
		repo = self.repos[self._state.repo_id]
		visible_codebase = repo["broken_code"] if role == Role.LEAD.value else None
		visible_filename = "broken.py" if role == Role.LEAD.value else None
		visible_allowlist = self.allowlist if role == Role.AUDITOR.value else None
		return SDKObservation(
			current_role=role,
			turn_index=self._state.step_count,
			max_turns=self.MAX_TURNS,
			error_log=repo["error_log"],
			conversation_history=list(self._history),
			visible_codebase=visible_codebase,
			visible_filename=visible_filename,
			visible_allowlist=visible_allowlist,
			current_proposal=self._state.proposals_history[-1] if self._state.proposals_history else None,
			approved_replacement=self._state.approved_replacement,
			done=done,
			reward=last_reward,
			reward_breakdown=breakdown or {},
		)
