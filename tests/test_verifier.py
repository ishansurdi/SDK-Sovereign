"""Tests for the AST + stubbed-exec verifier."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from server.verifier import StubRegistry, Verifier


@pytest.fixture
def repos_root(tmp_path: Path) -> Path:
	"""Create a temporary fake repo for verifier tests."""
	repo = tmp_path / "fake_repo"
	repo.mkdir()
	(repo / "broken.py").write_text("import stripe\n")
	(repo / "meta.json").write_text(
		json.dumps(
			{
				"repo_id": "fake_repo",
				"deprecated_sdk": "stripe",
				"ground_truth_replacement": "razorpay",
				"category": "payments",
				"entrypoint": "do_thing",
				"error_log": "boom",
			}
		)
	)
	(repo / "tests.json").write_text(
		json.dumps(
			{
				"test_one": {
					"args": [42],
					"expected": {"value": {"type": "int", "equals": 42}},
				}
			}
		)
	)
	return tmp_path


def test_stub_registry_installs_stripe() -> None:
	"""Validate deprecated stripe stub installation."""
	registry = StubRegistry()
	registry.install_in_sys_modules()
	import stripe  # noqa: F401

	assert stripe is not None


def test_stub_registry_installs_razorpay() -> None:
	"""Validate sovereign razorpay stub installation."""
	registry = StubRegistry()
	registry.install_in_sys_modules()
	import razorpay

	client = razorpay.Client(auth=("k", "s"))
	result = client.payment.create({"amount": 100, "customer_id": "c1"})
	assert "id" in result
	assert result["status"] == "captured"


def test_verifier_passes_correct_patch(repos_root: Path) -> None:
	"""Validate that a correct patch passes parity tests."""
	verifier = Verifier(repos_root)
	code = "def do_thing(x):\n    return {'value': x}"
	results = verifier.run_parity_tests(code, "fake_repo")
	assert results["test_one"] is True


def test_verifier_fails_wrong_output(repos_root: Path) -> None:
	"""Validate that a wrong output fails parity tests."""
	verifier = Verifier(repos_root)
	code = "def do_thing(x):\n    return {'value': 999}"
	results = verifier.run_parity_tests(code, "fake_repo")
	assert results["test_one"] is False


def test_verifier_reinstalls_stubs_per_run(repos_root: Path) -> None:
	"""Parity checks should recover if SDK stub modules were removed from sys.modules."""
	import sys

	verifier = Verifier(repos_root)
	sys.modules.pop("stripe", None)
	code = "import stripe\ndef do_thing(x):\n    return {'value': x}"
	results = verifier.run_parity_tests(code, "fake_repo")
	assert results["test_one"] is True


def test_verifier_handles_syntax_error(repos_root: Path) -> None:
	"""Validate graceful handling of syntax errors."""
	verifier = Verifier(repos_root)
	code = "def broken(:"
	results = verifier.run_parity_tests(code, "fake_repo")
	assert results["test_one"] is False


def test_verifier_handles_missing_function(repos_root: Path) -> None:
	"""Validate graceful handling of a missing entrypoint."""
	verifier = Verifier(repos_root)
	code = "def something_else(x):\n    return x"
	results = verifier.run_parity_tests(code, "fake_repo")
	assert results["test_one"] is False


def test_verifier_handles_runtime_exception(repos_root: Path) -> None:
	"""Validate graceful handling of runtime exceptions."""
	verifier = Verifier(repos_root)
	code = "def do_thing(x):\n    raise RuntimeError('boom')"
	results = verifier.run_parity_tests(code, "fake_repo")
	assert results["test_one"] is False


def test_extract_imports() -> None:
	"""Validate import extraction across import styles."""
	verifier = Verifier(Path("."))
	code = "import razorpay\nfrom mmi_sdk import Client\nimport os.path"
	imports = verifier.extract_imports(code)
	assert "razorpay" in imports
	assert "mmi_sdk" in imports
	assert "os" in imports
