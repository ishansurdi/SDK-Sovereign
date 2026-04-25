"""Environment orchestration tests using fixture repos."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from models import Role, SDKAction


@pytest.fixture
def fixture_env(tmp_path: Path):
	"""Create an environment backed by a temporary fixture repo."""
	repos_root = tmp_path / "repos"
	repos_root.mkdir()
	repo = repos_root / "fixture_repo"
	repo.mkdir()
	(repo / "broken.py").write_text("import stripe\ndef charge_customer(a, c): pass\n")
	(repo / "meta.json").write_text(
		json.dumps(
			{
				"repo_id": "fixture_repo",
				"deprecated_sdk": "stripe",
				"ground_truth_replacement": "razorpay",
				"category": "payments",
				"entrypoint": "charge_customer",
				"error_log": "stripe banned in IN region",
			}
		)
	)
	(repo / "tests.json").write_text(
		json.dumps(
			{
				"test_basic": {
					"args": [100, "c1"],
					"expected": {"id": {"type": "str"}},
				}
			}
		)
	)
	from server.environment import SDKSovereignEnvironment

	env = SDKSovereignEnvironment(repos_root=repos_root, seed=42)
	return env


def test_reset_returns_auditor_first(fixture_env) -> None:
	"""Validate reset returns an auditor-first observation."""
	observation = fixture_env.reset()
	assert observation.current_role == Role.AUDITOR.value
	assert observation.visible_codebase is None
	assert observation.visible_allowlist is not None


def test_lead_sees_codebase_not_allowlist(fixture_env) -> None:
	"""Validate role masking for lead observations."""
	fixture_env.reset()
	observation = fixture_env.step(SDKAction(role="auditor", action_type="pass"))
	assert observation.current_role == Role.LEAD.value
	assert observation.visible_codebase is not None
	assert observation.visible_allowlist is None


def test_role_alternation(fixture_env) -> None:
	"""Validate deterministic turn alternation."""
	observation = fixture_env.reset()
	assert observation.current_role == "auditor"
	observation = fixture_env.step(SDKAction(role="auditor", action_type="pass"))
	assert observation.current_role == "lead"
	observation = fixture_env.step(SDKAction(role="lead", action_type="pass"))
	assert observation.current_role == "auditor"


def test_wrong_role_penalised(fixture_env) -> None:
	"""Validate wrong-role actions are penalized."""
	fixture_env.reset()
	observation = fixture_env.step(SDKAction(role="lead", action_type="pass"))
	assert observation.reward == -1.0


def test_episode_terminates_on_submit(fixture_env) -> None:
	"""Validate submit_patch immediately ends the episode."""
	fixture_env.reset()
	fixture_env.step(SDKAction(role="auditor", action_type="pass"))
	fixture_env.step(
		SDKAction(role="lead", action_type="propose_replacement", proposed_sdk="razorpay")
	)
	fixture_env.step(SDKAction(role="auditor", action_type="approve"))
	observation = fixture_env.step(
		SDKAction(
			role="lead",
			action_type="submit_patch",
			patched_code="import razorpay\ndef charge_customer(a, c):\n    return {'id': 'x'}",
		)
	)
	assert observation.done is True


def test_episode_terminates_on_max_turns(fixture_env) -> None:
	"""Validate episode terminates on max turn cap."""
	fixture_env.reset()
	observation = None
	for index in range(7):
		role = "auditor" if index % 2 == 0 else "lead"
		observation = fixture_env.step(SDKAction(role=role, action_type="pass"))
	assert observation.done is True
