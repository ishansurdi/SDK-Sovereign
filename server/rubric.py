"""Composable reward rubric for SDK-Sovereign."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from models import ActionType, Role, SDKAction


WEIGHTS = {
	"format_valid": 0.5,
	"bad_format": -1.0,
	"pass_action_penalty": -0.5,
	"lead_identifies_deprecated": 1.0,
	"lead_proposes_in_history": -0.5,
	"lead_syntax_valid": 2.0,
	"lead_uses_approved_sdk": 1.0,
	"lead_split_brain": -2.0,
	"auditor_correct_approval": 1.5,
	"auditor_correct_rejection": 1.0,
	"auditor_wrong_approval": -2.0,
	"auditor_wrong_rejection": -1.0,
	"passes_parity_test": 3.0,
	"terminal_success": 2.0,
	"terminal_failure_max_turns": -1.0,
	"early_completion_bonus": 1.0,
}


@dataclass
class RubricResult:
	"""Represents step-level reward output with component breakdown."""

	total: float
	components: Dict[str, float]
	role_attribution: Dict[str, float]


class SDKMigrationRubric:
	"""Scores each action and terminal event for the migration task."""

	def __init__(self, allowlist: List[str], deprecated_sdks: List[str]):
		"""Initialise rubric with allowlist and deprecated SDK set."""
		self.allowlist = set(allowlist)
		self.deprecated_sdks = set(deprecated_sdks)

	def score_step(self, action: SDKAction, env_state, verifier) -> RubricResult:
		"""Score a single environment step action."""
		components: Dict[str, float] = {}
		role = action.role

		if self._is_valid_format(action):
			components["format_valid"] = WEIGHTS["format_valid"]
		else:
			components["bad_format"] = WEIGHTS["bad_format"]
			return self._finalise(components, role)

		if action.action_type == ActionType.PASS.value:
			components["pass_action_penalty"] = WEIGHTS["pass_action_penalty"]
			return self._finalise(components, role)

		if (
			role == Role.LEAD.value
			and env_state.step_count <= 2
			and action.reasoning
			and env_state.deprecated_sdk.lower() in action.reasoning.lower()
		):
			components["lead_identifies_deprecated"] = WEIGHTS["lead_identifies_deprecated"]

		if action.action_type == ActionType.PROPOSE_REPLACEMENT.value:
			if action.proposed_sdk in env_state.rejected_so_far:
				components["lead_proposes_in_history"] = WEIGHTS["lead_proposes_in_history"]

		if action.action_type == ActionType.APPROVE.value:
			proposed = env_state.proposals_history[-1] if env_state.proposals_history else None
			if proposed is None:
				pass
			elif proposed in self.allowlist:
				components["auditor_correct_approval"] = WEIGHTS["auditor_correct_approval"]
			else:
				components["auditor_wrong_approval"] = WEIGHTS["auditor_wrong_approval"]

		if action.action_type == ActionType.REJECT.value:
			proposed = env_state.proposals_history[-1] if env_state.proposals_history else None
			if proposed and proposed not in self.allowlist:
				components["auditor_correct_rejection"] = WEIGHTS["auditor_correct_rejection"]
			elif proposed == env_state.ground_truth_replacement:
				components["auditor_wrong_rejection"] = WEIGHTS["auditor_wrong_rejection"]

		if action.action_type == ActionType.SUBMIT_PATCH.value:
			components.update(self._score_patch(action, env_state, verifier))

		return self._finalise(components, role)

	def _score_patch(self, action: SDKAction, env_state, verifier) -> Dict[str, float]:
		"""Score patch quality and parity outcomes for submit_patch actions."""
		components: Dict[str, float] = {}
		code = action.patched_code or ""
		if not verifier.syntax_ok(code):
			return components
		components["lead_syntax_valid"] = WEIGHTS["lead_syntax_valid"]

		imports = verifier.extract_imports(code)
		if env_state.approved_replacement:
			if env_state.approved_replacement in imports:
				components["lead_uses_approved_sdk"] = WEIGHTS["lead_uses_approved_sdk"]
			else:
				components["lead_split_brain"] = WEIGHTS["lead_split_brain"]
				return components

		results = verifier.run_parity_tests(code, env_state.repo_id)
		passed = sum(1 for passed_flag in results.values() if passed_flag)
		components["passes_parity_test"] = WEIGHTS["passes_parity_test"] * passed
		return components

	def score_terminal(self, env_state, terminated_reason: str) -> Dict[str, float]:
		"""Score terminal bonuses/penalties when an episode ends."""
		components: Dict[str, float] = {}
		if terminated_reason == "submitted":
			results = env_state.test_results or {}
			if results and all(results.values()):
				components["terminal_success"] = WEIGHTS["terminal_success"]
				if env_state.step_count <= 5:
					components["early_completion_bonus"] = WEIGHTS["early_completion_bonus"]
		elif terminated_reason == "max_turns":
			components["terminal_failure_max_turns"] = WEIGHTS["terminal_failure_max_turns"]
		return components

	@staticmethod
	def _is_valid_format(action: SDKAction) -> bool:
		"""Validate role/action pairing and required payload fields."""
		if action.role not in (Role.AUDITOR.value, Role.LEAD.value):
			return False
		try:
			action_type = ActionType(action.action_type)
		except ValueError:
			return False

		required_field = {
			ActionType.PROPOSE_REPLACEMENT: "proposed_sdk",
			ActionType.SUBMIT_PATCH: "patched_code",
			ActionType.REJECT: "rejection_reason",
			ActionType.GIVE_HINT: "hint_response",
			ActionType.REQUEST_HINT: "hint_request",
		}.get(action_type)
		if required_field and not getattr(action, required_field):
			return False

		lead_actions = {
			ActionType.PROPOSE_REPLACEMENT,
			ActionType.SUBMIT_PATCH,
			ActionType.REQUEST_HINT,
			ActionType.PASS,
		}
		auditor_actions = {
			ActionType.APPROVE,
			ActionType.REJECT,
			ActionType.GIVE_HINT,
			ActionType.PASS,
		}
		if action.role == Role.LEAD.value and action_type not in lead_actions:
			return False
		if action.role == Role.AUDITOR.value and action_type not in auditor_actions:
			return False
		return True

	@staticmethod
	def _finalise(components: Dict[str, float], role: str) -> RubricResult:
		"""Create a RubricResult from scored components and acting role."""
		total = sum(components.values())
		attribution = {Role.AUDITOR.value: 0.0, Role.LEAD.value: 0.0}
		attribution[role] = total
		return RubricResult(total=total, components=components, role_attribution=attribution)
