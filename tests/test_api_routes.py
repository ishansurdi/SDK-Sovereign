"""Additional local API contract tests for root, health, and play state flows."""
from __future__ import annotations

from fastapi.testclient import TestClient

from server.app import app


def test_health_endpoint_reports_healthy() -> None:
	"""The app should expose a healthy status endpoint for smoke automation."""
	client = TestClient(app)
	response = client.get("/health")
	assert response.status_code == 200
	payload = response.json()
	assert payload["status"] == "healthy"


def test_root_serves_play_html() -> None:
	"""The root route should serve the interactive play demo page."""
	client = TestClient(app)
	response = client.get("/")
	assert response.status_code == 200
	assert "text/html" in response.headers["content-type"]


def test_play_state_and_agent_step_round_trip() -> None:
	"""Resetting the demo should make state and step endpoints usable immediately."""
	client = TestClient(app)
	reset_response = client.post("/play/reset?repo_id=payments_repo&mode=rule")
	assert reset_response.status_code == 200

	state_response = client.get("/play/state")
	assert state_response.status_code == 200
	state_payload = state_response.json()
	assert state_payload["repo_id"] == "payments_repo"

	step_response = client.post("/play/agent_step")
	assert step_response.status_code == 200
	step_payload = step_response.json()
	assert step_payload["agent_mode"] == "rule"
	assert step_payload["transcript_entry"]["policy_mode"] == "rule"
	assert step_payload["observation"]["current_role"] in {"auditor", "lead"}
