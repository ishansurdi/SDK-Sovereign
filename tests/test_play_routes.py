"""Route-level tests for the play demo API."""
from __future__ import annotations

from fastapi.testclient import TestClient

from server.app import app


def test_play_catalog_exposes_available_modes() -> None:
	"""The catalog should advertise repo options and supported policy modes."""
	client = TestClient(app)
	response = client.get("/play/catalog")
	assert response.status_code == 200
	payload = response.json()
	assert payload["repos"]
	assert any(mode["id"] == "rule" for mode in payload["available_modes"])
	assert "diagnostics" in payload
	assert payload["diagnostics"]["available_mode_ids"]


def test_play_reset_accepts_rule_mode() -> None:
	"""The demo reset route should accept an explicit rule mode selection."""
	client = TestClient(app)
	response = client.post("/play/reset?repo_id=comms_repo&mode=rule")
	assert response.status_code == 200
	payload = response.json()
	assert payload["agent_mode"] == "rule"
	assert payload["issue"]["repo_id"] == "comms_repo"
	assert payload["observation"]["current_role"] == "auditor"


def test_play_reset_rejects_unknown_mode() -> None:
	"""Unsupported policy modes should fail fast with a 400."""
	client = TestClient(app)
	response = client.post("/play/reset?repo_id=comms_repo&mode=unknown")
	assert response.status_code == 400