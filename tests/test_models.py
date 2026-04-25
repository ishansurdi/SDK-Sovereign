"""Smoke tests for schemas."""
from __future__ import annotations

from models import ActionType, SDKAction, SDKObservation, SDKState


def test_action_minimal() -> None:
	"""Validate a minimal action payload."""
	action = SDKAction(role="lead", action_type="pass")
	assert action.role == "lead"
	assert action.proposed_sdk is None


def test_observation_required_fields() -> None:
	"""Validate an observation with only required fields."""
	observation = SDKObservation(
		current_role="auditor",
		turn_index=0,
		max_turns=7,
		error_log="boom",
	)
	assert observation.current_role == "auditor"
	assert observation.done is False
	assert observation.reward == 0.0


def test_state_default_cumulative() -> None:
	"""Validate default cumulative rewards."""
	state = SDKState(
		episode_id="e1",
		repo_id="payments_repo",
		deprecated_sdk="stripe",
		ground_truth_replacement="razorpay",
	)
	assert state.cumulative_reward_by_role == {"auditor": 0.0, "lead": 0.0}


def test_action_type_enum() -> None:
	"""Validate enum values for lead actions."""
	assert ActionType.PROPOSE_REPLACEMENT.value == "propose_replacement"
	assert ActionType.SUBMIT_PATCH.value == "submit_patch"
