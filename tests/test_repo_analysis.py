"""Focused tests for the optional GitHub repo analysis helper."""
from __future__ import annotations

import io
import json
from urllib import error

import server.repo_analysis as repo_analysis
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


def test_generate_gemini_analysis_summarizes_quota_error(monkeypatch) -> None:
	"""Quota failures should be compact and UI-safe instead of dumping raw JSON."""
	monkeypatch.setenv("GEMINI_API", "token")
	payload = {
		"error": {
			"code": 429,
			"status": "RESOURCE_EXHAUSTED",
			"message": "Quota exceeded for model gemini-2.0-flash. Please retry in 38.991083151s.",
		}
	}

	def fake_urlopen(*_args, **_kwargs):
		raise error.HTTPError(
			url="https://example.test",
			code=429,
			msg="Too Many Requests",
			hdrs=None,
			fp=io.BytesIO(json.dumps(payload).encode("utf-8")),
		)

	monkeypatch.setattr(repo_analysis.request, "urlopen", fake_urlopen)
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
	assert result["enabled"] is True
	assert result["status"] == "resource_exhausted"
	assert result["code"] == 429
	assert "quota exceeded" in result["summary"].lower()
	assert result["retry_after_seconds"] == 38.991083151