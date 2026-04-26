"""Tests for play-policy runtime gating."""
from __future__ import annotations

import pytest

from server import policy_runtime


def test_configured_live_modes_hide_model_modes_when_runtime_unavailable(monkeypatch) -> None:
	"""Baseline and trained modes should be hidden when the runtime cannot load models."""
	monkeypatch.setenv("SDK_SOVEREIGN_AGENTS_LIVE", "1")
	monkeypatch.setattr(
		policy_runtime,
		"model_runtime_status",
		lambda: {"ready": False, "issues": ["missing dependency: unsloth"]},
	)

	mode_ids = [item["id"] for item in policy_runtime.configured_live_modes()]
	assert mode_ids == ["rule", "teacher"]

	diagnostics = policy_runtime.mode_diagnostics()
	assert diagnostics["model_runtime_ready"] is False
	assert "missing dependency: unsloth" in diagnostics["notes"][0] or "missing dependency: unsloth" in " ".join(diagnostics["notes"])


def test_configured_live_modes_include_baseline_and_trained_when_runtime_ready(monkeypatch) -> None:
	"""Model-backed modes should appear only when the runtime and adapter config are both ready."""
	monkeypatch.setenv("SDK_SOVEREIGN_AGENTS_LIVE", "1")
	monkeypatch.setenv("SDK_SOVEREIGN_LEAD_ADAPTER_REPO", "lead-repo")
	monkeypatch.setenv("SDK_SOVEREIGN_AUDITOR_ADAPTER_REPO", "auditor-repo")
	monkeypatch.setattr(
		policy_runtime,
		"model_runtime_status",
		lambda: {"ready": True, "issues": []},
	)

	mode_ids = [item["id"] for item in policy_runtime.configured_live_modes()]
	assert mode_ids == ["rule", "teacher", "baseline", "trained"]


def test_load_model_agents_fails_with_clear_runtime_error(monkeypatch) -> None:
	"""Forced model loading should fail with a clear runtime error instead of an import traceback."""
	monkeypatch.setattr(
		policy_runtime,
		"model_runtime_status",
		lambda: {"ready": False, "issues": ["missing dependency: unsloth", "CUDA GPU not available"]},
	)

	with pytest.raises(RuntimeError, match="missing dependency: unsloth"):
		policy_runtime.load_model_agents("baseline")