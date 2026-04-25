"""Reward function tests for rubric behavior."""
from __future__ import annotations

import pytest

from models import SDKAction, SDKState
from server.rubric import SDKMigrationRubric, WEIGHTS


@pytest.fixture
def rubric() -> SDKMigrationRubric:
	"""Return a test rubric with a minimal allowlist."""
	return SDKMigrationRubric(
		allowlist=["razorpay", "mmi_sdk", "kaleyra"],
		deprecated_sdks=["stripe", "googlemaps", "twilio"],
	)


@pytest.fixture
def state() -> SDKState:
	"""Return a baseline state for rubric tests."""
	return SDKState(
		episode_id="t",
		repo_id="payments_repo",
		deprecated_sdk="stripe",
		ground_truth_replacement="razorpay",
	)


class _DummyVerifier:
	"""Dummy verifier for rubric unit tests."""

	def syntax_ok(self, code: str) -> bool:
		"""Return True when code appears to define a function."""
		return "def " in code

	def extract_imports(self, code: str) -> set[str]:
		"""Return razorpay import only when present in code."""
		return {"razorpay"} if "razorpay" in code else set()

	def run_parity_tests(self, code: str, repo_id: str) -> dict[str, bool]:
		"""Return all pass when razorpay appears, otherwise fail."""
		_ = repo_id
		return {"t1": True, "t2": True, "t3": True} if "razorpay" in code else {"t1": False}


def test_format_valid_pass(rubric: SDKMigrationRubric, state: SDKState) -> None:
	"""Validate format score for a pass action."""
	action = SDKAction(role="lead", action_type="pass")
	result = rubric.score_step(action, state, _DummyVerifier())
	assert result.components["format_valid"] == WEIGHTS["format_valid"]


def test_bad_format_unknown_action(rubric: SDKMigrationRubric, state: SDKState) -> None:
	"""Validate unknown action format rejection."""
	action = SDKAction(role="lead", action_type="bogus_action")
	result = rubric.score_step(action, state, _DummyVerifier())
	assert "bad_format" in result.components


def test_bad_format_role_action_mismatch(rubric: SDKMigrationRubric, state: SDKState) -> None:
	"""Validate role/action mismatch rejection."""
	action = SDKAction(role="lead", action_type="approve")
	result = rubric.score_step(action, state, _DummyVerifier())
	assert "bad_format" in result.components


def test_auditor_correct_approval(rubric: SDKMigrationRubric, state: SDKState) -> None:
	"""Validate positive reward for correct approval."""
	state.proposals_history.append("razorpay")
	action = SDKAction(role="auditor", action_type="approve", reasoning="razorpay is on the allowlist")
	result = rubric.score_step(action, state, _DummyVerifier())
	assert result.components["auditor_correct_approval"] == WEIGHTS["auditor_correct_approval"]


def test_auditor_wrong_approval(rubric: SDKMigrationRubric, state: SDKState) -> None:
	"""Validate negative reward for wrong approval."""
	state.proposals_history.append("not_on_list")
	action = SDKAction(role="auditor", action_type="approve")
	result = rubric.score_step(action, state, _DummyVerifier())
	assert result.components["auditor_wrong_approval"] == WEIGHTS["auditor_wrong_approval"]


def test_lead_split_brain(rubric: SDKMigrationRubric, state: SDKState) -> None:
	"""Validate split-brain penalty for wrong SDK import after approval."""
	state.approved_replacement = "razorpay"
	action = SDKAction(
		role="lead",
		action_type="submit_patch",
		patched_code="import kaleyra\ndef charge_customer(a, c):\n    return {}",
	)
	result = rubric.score_step(action, state, _DummyVerifier())
	assert "lead_split_brain" in result.components


def test_expert_trajectory_total_positive(rubric: SDKMigrationRubric, state: SDKState) -> None:
	"""Validate expert trajectory has strongly positive return."""
	verifier = _DummyVerifier()
	total = 0.0

	action0 = SDKAction(role="auditor", action_type="pass", reasoning="Awaiting Lead's proposal.")
	total += rubric.score_step(action0, state, verifier).total
	state.step_count += 1

	action1 = SDKAction(
		role="lead",
		action_type="propose_replacement",
		proposed_sdk="razorpay",
		reasoning="The code uses stripe; razorpay is sovereign.",
	)
	state.proposals_history.append("razorpay")
	total += rubric.score_step(action1, state, verifier).total
	state.step_count += 1

	action2 = SDKAction(role="auditor", action_type="approve")
	state.approved_replacement = "razorpay"
	total += rubric.score_step(action2, state, verifier).total
	state.step_count += 1

	action3 = SDKAction(
		role="lead",
		action_type="submit_patch",
		patched_code="import razorpay\ndef charge_customer(a, c):\n    return {}",
	)
	total += rubric.score_step(action3, state, verifier).total

	state.test_results = {"t1": True, "t2": True, "t3": True}
	total += sum(rubric.score_terminal(state, "submitted").values())

	assert total > 12.0, f"Expert trajectory only scored {total}"


def test_random_trajectory_total_negative(rubric: SDKMigrationRubric, state: SDKState) -> None:
	"""Validate random trajectory has negative return."""
	verifier = _DummyVerifier()
	total = 0.0

	action = SDKAction(role="lead", action_type="propose_replacement", proposed_sdk="malware_sdk")
	state.proposals_history.append("malware_sdk")
	total += rubric.score_step(action, state, verifier).total
	state.step_count += 1

	action = SDKAction(role="auditor", action_type="approve")
	state.approved_replacement = "malware_sdk"
	total += rubric.score_step(action, state, verifier).total
	state.step_count += 1

	action = SDKAction(
		role="lead",
		action_type="submit_patch",
		patched_code="import some_other\ndef charge_customer(a, c):\n    return {}",
	)
	total += rubric.score_step(action, state, verifier).total

	state.test_results = {"t1": False}
	total += sum(rubric.score_terminal(state, "submitted").values())
	assert total < 0, f"Random trajectory scored {total} (should be negative)"
