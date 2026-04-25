"""
Action / Observation / State schemas for SDK-Sovereign.

Design notes
------------
- Role-conditional visibility is enforced server-side: the env zeros out
  visible_codebase when current_role == AUDITOR and zeros out
  visible_allowlist when current_role == LEAD.
- ActionType is a finite enum so the policy learns a discrete action structure.
- reasoning is a free-text channel that the *other* agent reads next turn -
  this is the negotiation surface and the theory-of-mind lever.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

try:
    from core.env_server import Action as _OEAction, Observation as _OEObservation, State as _OEState
except ImportError:
    class _OEAction: ...  # type: ignore[no-redef]
    class _OEObservation: ...  # type: ignore[no-redef]
    class _OEState: ...  # type: ignore[no-redef]


class Role(str, Enum):
	"""Enumerates the two environment roles."""

	AUDITOR = "auditor"
	LEAD = "lead"


class ActionType(str, Enum):
	"""Enumerates all valid action types across both roles."""

	PROPOSE_REPLACEMENT = "propose_replacement"
	SUBMIT_PATCH = "submit_patch"
	REQUEST_HINT = "request_hint"
	APPROVE = "approve"
	REJECT = "reject"
	GIVE_HINT = "give_hint"
	PASS = "pass"


@dataclass
class SDKAction(_OEAction):
	"""Represents one agent action for a turn."""

	role: str
	action_type: str
	proposed_sdk: Optional[str] = None
	rejection_reason: Optional[str] = None
	patched_code: Optional[str] = None
	hint_request: Optional[str] = None
	hint_response: Optional[str] = None
	reasoning: Optional[str] = None


@dataclass
class SDKObservation(_OEObservation):
	"""Represents the role-masked observation returned each turn."""

	current_role: str
	turn_index: int
	max_turns: int
	error_log: str
	conversation_history: List[Dict[str, Any]] = field(default_factory=list)
	visible_codebase: Optional[str] = None
	visible_filename: Optional[str] = None
	visible_allowlist: Optional[List[str]] = None
	current_proposal: Optional[str] = None
	approved_replacement: Optional[str] = None
	done: bool = False
	reward: float = 0.0
	reward_breakdown: Dict[str, float] = field(default_factory=dict)


@dataclass
class SDKState(_OEState):
	"""Represents the full hidden environment state."""

	episode_id: str
	repo_id: str
	deprecated_sdk: str
	ground_truth_replacement: str
	step_count: int = 0
	proposals_history: List[str] = field(default_factory=list)
	rejected_so_far: List[str] = field(default_factory=list)
	approved_replacement: Optional[str] = None
	final_patch: Optional[str] = None
	test_results: Optional[Dict[str, bool]] = None
	terminated_reason: Optional[str] = None
	cumulative_reward_by_role: Dict[str, float] = field(
		default_factory=lambda: {"auditor": 0.0, "lead": 0.0}
	)
