"""
Verifier: runs submitted patches against golden parity tests using
stubbed SDK modules.

Why stubs? At runtime in HF Space we have no network and we don't want
to depend on real SDKs. Stubs let us inject imports cleanly and check that
the patch is structurally and behaviourally correct.
"""
from __future__ import annotations

import ast
import json
import signal
import sys
import types
from pathlib import Path
from typing import Any, Dict


class StubRegistry:
	"""Holds stub modules that replace real SDKs at exec time."""

	def __init__(self) -> None:
		"""Initialise and populate the stub module registry."""
		self._stubs: Dict[str, types.ModuleType] = {}
		self._build_all()

	def _build_all(self) -> None:
		"""Build all deprecated and sovereign SDK stubs."""
		stripe = types.ModuleType("stripe")
		stripe.api_key = ""

		class _StripeCharge:
			@staticmethod
			def create(amount: int, currency: str, customer: str) -> Any:
				return type(
					"Charge",
					(),
					{
						"id": f"ch_stub_{customer}_{amount}",
						"status": "succeeded" if amount > 0 else "failed",
					},
				)()

		stripe.Charge = _StripeCharge
		self._stubs["stripe"] = stripe

		googlemaps = types.ModuleType("googlemaps")

		class _GMapsClient:
			def __init__(self, key: str | None = None) -> None:
				self.key = key

			def geocode(self, address: str) -> list[dict[str, Any]]:
				hsh = abs(hash(address)) % 1000
				return [{"geometry": {"location": {"lat": 12.0 + hsh * 0.001, "lng": 77.0 + hsh * 0.001}}}]

		googlemaps.Client = _GMapsClient
		self._stubs["googlemaps"] = googlemaps

		twilio = types.ModuleType("twilio")
		twilio_rest = types.ModuleType("twilio.rest")

		class _TwilioClient:
			def __init__(self, *args: Any, **kwargs: Any) -> None:
				_ = (args, kwargs)

			class _Messages:
				@staticmethod
				def create(to: str, from_: str, body: str) -> Any:
					_ = (from_, body)
					return type("Message", (), {"sid": f"SM_stub_{to}", "status": "queued"})()

			messages = _Messages()

		twilio_rest.Client = _TwilioClient
		twilio.rest = twilio_rest
		self._stubs["twilio"] = twilio
		self._stubs["twilio.rest"] = twilio_rest

		razorpay = types.ModuleType("razorpay")

		class _RazorpayPayment:
			@staticmethod
			def create(data: dict[str, Any]) -> dict[str, Any]:
				amount = data.get("amount", 0)
				customer_id = data.get("customer_id", "unknown")
				return {
					"id": f"pay_stub_{customer_id}_{amount}",
					"status": "captured" if amount > 0 else "failed",
				}

		class _RazorpayClient:
			def __init__(self, auth: tuple[str, str] | None = None) -> None:
				self.auth = auth

			payment = _RazorpayPayment()

		razorpay.Client = _RazorpayClient
		self._stubs["razorpay"] = razorpay

		mmi_sdk = types.ModuleType("mmi_sdk")

		class _MMIClient:
			def __init__(self, api_key: str | None = None) -> None:
				self.api_key = api_key

			def get_location(self, address: str) -> dict[str, float]:
				hsh = abs(hash(address)) % 1000
				return {"lat": 12.0 + hsh * 0.001, "lng": 77.0 + hsh * 0.001}

		mmi_sdk.Client = _MMIClient
		self._stubs["mmi_sdk"] = mmi_sdk

		kaleyra = types.ModuleType("kaleyra")

		class _KaleyraClient:
			def __init__(self, api_key: str | None = None) -> None:
				self.api_key = api_key

			def send_sms(self, to: str, sender: str, message: str) -> dict[str, str]:
				_ = (sender, message)
				return {"message_id": f"klr_stub_{to}", "status": "sent"}

		kaleyra.Client = _KaleyraClient
		self._stubs["kaleyra"] = kaleyra

	def install_in_sys_modules(self) -> None:
		"""Install all stub modules into sys.modules."""
		for name, module in self._stubs.items():
			sys.modules[name] = module


class _Timeout:
	"""Raise TimeoutError when a sandboxed step takes too long."""

	def __init__(self, seconds: int) -> None:
		"""Store the timeout duration."""
		self.seconds = seconds

	def __enter__(self) -> None:
		"""Arm the alarm on supported platforms."""
		if hasattr(signal, "SIGALRM"):
			signal.signal(signal.SIGALRM, self._handler)
			signal.alarm(self.seconds)

	def __exit__(self, *args: Any) -> None:
		"""Disarm the alarm on exit."""
		_ = args
		if hasattr(signal, "SIGALRM"):
			signal.alarm(0)

	@staticmethod
	def _handler(signum: int, frame: Any) -> None:
		"""Handle SIGALRM by raising a timeout error."""
		_ = (signum, frame)
		raise TimeoutError("verifier timeout")


class Verifier:
	"""Runs golden parity tests for a given repo against a submitted patch."""

	def __init__(self, repos_root: Path) -> None:
		"""Initialise the verifier and install SDK stubs."""
		self.repos_root = Path(repos_root)
		self.stubs = StubRegistry()
		self.stubs.install_in_sys_modules()

	def load_meta(self, repo_id: str) -> Dict[str, Any]:
		"""Load repo metadata for a given repo identifier."""
		return json.loads((self.repos_root / repo_id / "meta.json").read_text())

	def load_tests(self, repo_id: str) -> Dict[str, Any]:
		"""Load parity test specifications for a repo."""
		return json.loads((self.repos_root / repo_id / "tests.json").read_text())

	def load_broken_code(self, repo_id: str) -> str:
		"""Load the shipped broken code for a repo."""
		return (self.repos_root / repo_id / "broken.py").read_text()

	def run_parity_tests(self, code: str, repo_id: str) -> Dict[str, bool]:
		"""Execute code and evaluate it against the repo's parity tests."""
		# Reinstall stubs per run so remote workers do not drift onto real or missing SDK modules.
		self.stubs.install_in_sys_modules()
		tests = self.load_tests(repo_id)
		meta = self.load_meta(repo_id)
		target_fn = meta["entrypoint"]
		results = {test_id: False for test_id in tests.keys()}
		local_ns: Dict[str, Any] = {}

		try:
			with _Timeout(2):
				exec(code, local_ns, local_ns)
		except Exception:
			return results

		fn = local_ns.get(target_fn)
		if not callable(fn):
			return results

		for test_id, spec in tests.items():
			try:
				with _Timeout(2):
					output = fn(*spec.get("args", []), **spec.get("kwargs", {}))
				results[test_id] = self._matches(output, spec["expected"])
			except Exception:
				results[test_id] = False
		return results

	def syntax_ok(self, code: str) -> bool:
		"""Return whether the supplied code parses successfully."""
		try:
			ast.parse(code)
			return True
		except SyntaxError:
			return False

	def extract_imports(self, code: str) -> set[str]:
		"""Extract top-level imported module names from code."""
		try:
			tree = ast.parse(code)
		except SyntaxError:
			return set()

		imports: set[str] = set()
		for node in ast.walk(tree):
			if isinstance(node, ast.Import):
				for alias in node.names:
					imports.add(alias.name.split(".")[0])
			elif isinstance(node, ast.ImportFrom) and node.module:
				imports.add(node.module.split(".")[0])
		return imports

	@staticmethod
	def _matches(output: Any, expected: Dict[str, Any]) -> bool:
		"""Return whether an output object matches the expected spec."""
		for key, spec in expected.items():
			value = output.get(key) if isinstance(output, dict) else getattr(output, key, None)
			if value is None:
				return False
			if "type" in spec:
				expected_type = spec["type"]
				if expected_type == "str" and not isinstance(value, str):
					return False
				if expected_type == "int" and not isinstance(value, int):
					return False
				if expected_type == "float" and not isinstance(value, (int, float)):
					return False
				if expected_type == "dict" and not isinstance(value, dict):
					return False
			if "contains" in spec and spec["contains"] not in str(value):
				return False
			if "equals" in spec and value != spec["equals"]:
				return False
		return True
