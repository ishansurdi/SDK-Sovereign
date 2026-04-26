"""Focused tests for the optional GitHub repo analysis helper."""
from __future__ import annotations

from server.repo_analysis import GitHubSnapshot, build_local_analysis, generate_gemini_analysis


def test_build_local_analysis_detects_known_sdk() -> None:
	"""Known benchmark SDKs should map to a sovereign recommendation offline."""
	snapshot = GitHubSnapshot(
		owner="acme",
		repo="payments-demo",
		html_url="https://github.com/acme/payments-demo",
		default_branch="main",
		description="Stripe payment integration service",
		readme_text="This service uses stripe webhooks and stripe checkout.",
		top_level_files=["README.md", "payments.py"],
	)
	analysis = build_local_analysis(snapshot)
	assert analysis["detected_sdk"] == "stripe"
	assert analysis["recommended_replacement"] == "razorpay"


def test_generate_gemini_analysis_uses_gemini_api_secret(monkeypatch) -> None:
	"""Gemini support should read the Hugging Face secret name GEMINI_API."""
	monkeypatch.delenv("GEMINI_API", raising=False)
	snapshot = GitHubSnapshot(
		owner="acme",
		repo="demo",
		html_url="https://github.com/acme/demo",
		default_branch="main",
		description="demo",
		readme_text="demo",
		top_level_files=["README.md"],
	)
	result = generate_gemini_analysis(snapshot, {"recommended_replacement": "razorpay"})
	assert result["enabled"] is False
	assert result["status"] == "missing_api"