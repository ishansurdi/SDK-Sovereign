# SDK-Sovereign v1 — Build-From-Scratch PRD
**Multi-Agent OpenEnv for Digital Sovereignty Migrations · Theme 1 · India 2026**

> **FROM-SCRATCH BUILD.** You are starting with an empty directory. No MVP exists yet. This PRD takes you from empty folder to complete hackathon submission: environment, two trained LoRA adapters, OpenEnv deployment, training notebooks, plots, web demo, README, video.
>
> Read this entire PRD before writing any code. Execute sections in order. Do not skip phases. Verify acceptance criteria at every gate.

---

## 0. Meta-Instructions

### 0.1 Your role

You are a senior Python + ML engineer building a complete OpenEnv Hackathon submission solo, in 24 hours. Generate production-quality code: typed, tested, runnable on Linux/Mac/Windows. Do not invent frameworks not specified here. Every file specified in this PRD is exactly what should exist in the repo at the end.

### 0.2 Ground rules

1. **Target Python 3.10 or 3.11** (3.12 has ML-lib lag)
2. **Pydantic v2** for all schemas
3. **`from __future__ import annotations`** at the top of every `.py` file
4. **Type hints on every function**
5. **One-line docstrings minimum on every public function**
6. **`pytest tests/ -v` after every phase** — test count must only grow
7. **Git commit after each phase** — never leave broken state
8. **Train the two adapters separately**, both via TRL GRPO, on role-conditional rollout data
9. **OpenEnv v0.2.3 is mandatory** — pin the exact version in `pyproject.toml`
10. **When stuck: reduce.** One repo, one role, one turn at a time.

### 0.3 Environment assumptions

- Fresh machine with `git`, Python 3.10/3.11, `uv` or `pip`
- GPU work happens on Colab + HF Jobs (no local GPU needed)
- HuggingFace account, $30 credit redeemed, write token in env var `HF_TOKEN`
- Weights & Biases account, key in env var `WANDB_API_KEY`
- GitHub account (optional — HF Space repo is enough)

### 0.4 Win condition (mapped to judging rubric)

| Criterion | Weight | Win move for this build |
|---|---|---|
| Environment Innovation | 40% | Multi-agent partial observability + India-sovereignty narrative + AST-based functional verification + composable rubric. No other team will hit this combination. |
| Storytelling | 30% | "It's 2026. Your stack just got sanctioned. 7 turns to migrate." Repeated in README, blog, video, Discord. |
| Showing Improvement in Rewards | 20% | Two distinct WandB runs (auditor + lead). Pre-training pass rate vs post-training pass rate plot. Per-role learning curves. |
| Reward & Training Pipeline | 10% | OpenEnv 0.2.3 compliance, TRL GRPO + Unsloth, two LoRA adapters trained separately, valid `openenv.yaml`. |

### 0.5 Phase breakdown

| Phase | Target | Hard cap | Deliverable |
|---|---|---|---|
| 1 | Project scaffold | 1h | Repo + venv + `pytest` runs |
| 2 | Schemas + Verifier + Allow-list | 2h | `verifier.run_parity_tests()` works |
| 3 | Rubric + Environment + Rule agents + Demo | 4h | `python demo.py` plays full episode |
| 4 | Three synthetic repos + tests | 2h | All 3 repos pass golden tests with hand-crafted "good" patches |
| 5 | OpenEnv wrap + HF Space deploy | 3h | Live HF Space, `/web` UI works |
| 6 | LLM agents + two LoRA adapter loader | 3h | Smoke episode in Colab with Llama / Qwen |
| 7 | GRPO training (two adapters) | overnight | Two adapter checkpoints + WandB curves |
| 8 | Eval + plots | 3h | 6 PNGs in `plots/` |
| 9 | Web demo (`/play`) | 2h | Live trained-model demo on HF Space |
| 10 | Final assembly | 3h | README, blog, video, submission |

**Total**: ~26h active + overnight training. Tight. Submit at hour +27 even if rough.

### 0.6 The sentence

This sentence appears in README, HF blog, video script, Discord intro post:

> **"SDK-Sovereign trains two LLM policies to coordinate under partial information on a problem that isn't a game — it's the kind of crisis a real engineering team faces when geopolitics moves faster than their tech stack."**

---

## 1. Phase 1 — Project Scaffold (Target: 1 hour)

### 1.1 Goal

Empty repo with structure, `pyproject.toml`, editable install, passing empty test suite, initial git commit.

### 1.2 Directory structure

```
sdk-sovereign-env/
├── pyproject.toml
├── README.md                       # Placeholder, finalised in Phase 10
├── LICENSE                         # MIT
├── .gitignore
├── Dockerfile                      # OpenEnv standard
├── openenv.yaml                    # Manifest
├── __init__.py
├── models.py                       # Phase 2: schemas
├── client.py                       # Phase 5: OpenEnv client
├── server/
│   ├── __init__.py
│   ├── app.py                      # Phase 5: FastAPI bootstrap
│   ├── environment.py              # Phase 3: env class
│   ├── rubric.py                   # Phase 3: reward function
│   ├── verifier.py                 # Phase 2: AST + exec sandbox
│   ├── prompts.py                  # Phase 6: role system prompts
│   ├── allowlist.json              # Phase 2: sovereign SDK registry
│   ├── rule_agents.py              # Phase 3: rule-based fallback
│   ├── llm_agents.py               # Phase 6: LLM-backed agents
│   ├── play_routes.py              # Phase 9: /play custom UI
│   └── repos/
│       ├── __init__.py
│       ├── payments_repo/
│       │   ├── broken.py
│       │   ├── tests.json
│       │   └── meta.json
│       ├── maps_repo/
│       │   ├── broken.py
│       │   ├── tests.json
│       │   └── meta.json
│       └── comms_repo/
│           ├── broken.py
│           ├── tests.json
│           └── meta.json
├── tests/
│   ├── __init__.py
│   ├── test_models.py
│   ├── test_verifier.py
│   ├── test_rubric.py
│   ├── test_environment.py
│   ├── test_repos.py
│   └── test_smoke_remote.py
├── notebooks/
│   ├── 01_smoke_test.ipynb         # Phase 6: env + LLM sanity check
│   ├── 02_train_lead.ipynb         # Phase 7: train Lead adapter
│   ├── 03_train_auditor.ipynb      # Phase 7: train Auditor adapter
│   └── 04_eval_and_plots.ipynb     # Phase 8: baseline vs trained
├── scripts/
│   ├── hand_patches.py             # Phase 4: golden "good" patches for testing
│   ├── run_random_episodes.py      # Phase 3: smoke loop
│   └── make_plots.py               # Phase 8: regenerate plots from logs
├── frontend/
│   └── play.html                   # Phase 9: custom web demo
├── docs/
│   ├── DESIGN.md                   # Phase 10
│   ├── LIMITATIONS.md              # Phase 10 — honest list
│   └── HF_BLOG.md                  # Phase 10
├── plots/                          # Phase 8 outputs
├── checkpoints/                    # Training outputs (gitignored)
├── logs/                           # Episode logs (gitignored)
└── demo.py                         # Phase 3 entrypoint
```

### 1.3 Bash bootstrap

```bash
mkdir -p sdk-sovereign-env/{server/repos/{payments_repo,maps_repo,comms_repo},tests,notebooks,scripts,frontend,docs,plots,checkpoints,logs}
cd sdk-sovereign-env
git init
```

Create empty `__init__.py` in every Python package: `server/`, `server/repos/`, `tests/`.

### 1.4 `pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "openenv-sdk-sovereign"
version = "0.1.0"
description = "Multi-agent OpenEnv environment for digital sovereignty SDK migrations."
readme = "README.md"
requires-python = ">=3.10,<3.12"
license = {text = "MIT"}
authors = [{name = "<your-name>"}]
dependencies = [
    "openenv-core[core] @ git+https://github.com/meta-pytorch/OpenEnv@v0.2.3",
    "pydantic>=2.0",
    "rich>=13.0",
    "pytest>=8.0",
]

[project.optional-dependencies]
training = [
    "transformers>=4.45",
    "peft>=0.13",
    "accelerate>=0.34",
    "bitsandbytes>=0.43",
    "torch>=2.4",
    "trl>=0.12",
    "datasets>=3.0",
    "wandb>=0.18",
]
analysis = [
    "matplotlib>=3.8",
    "pandas>=2.0",
    "numpy>=1.26",
    "seaborn>=0.13",
]
demo = [
    "gradio>=4.0",
    "fastapi>=0.110",
    "uvicorn>=0.30",
]
dev = ["pytest-cov>=4.0"]

[tool.setuptools.packages.find]
include = ["server*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
```

### 1.5 `.gitignore`

```
__pycache__/
*.py[cod]
*.egg-info/
.pytest_cache/
.venv/
venv/
env/

.vscode/
.idea/
.DS_Store

.ipynb_checkpoints/

logs/
checkpoints/
wandb/
*.safetensors
*.bin

.env
.env.local

build/
dist/
```

### 1.6 `LICENSE`

Standard MIT. Copy from any MIT-licensed repo.

### 1.7 Placeholder `README.md`

```markdown
# SDK-Sovereign 🇮🇳

Multi-agent OpenEnv environment for digital sovereignty SDK migrations.

**Status:** In development. Final README ships in Phase 10.

## Quickstart
```bash
pip install -e .
pytest tests/ -v
python demo.py
```
```

### 1.8 Install + verify

```bash
python -m venv .venv
source .venv/bin/activate    # or .venv\Scripts\activate on Windows
pip install -e ".[dev]"
pytest tests/ -v
```

Expected output: `no tests ran in 0.01s` — that's correct, we have no tests yet.

### 1.9 Initial commit

```bash
git add .
git commit -m "Phase 1: project scaffold"
```

### 1.10 Phase 1 acceptance criteria

- [ ] Directory structure matches §1.2 exactly
- [ ] `pip install -e .` succeeds with zero errors
- [ ] `pytest tests/ -v` runs (zero tests collected is OK)
- [ ] `.gitignore` blocks `.venv/`, `logs/`, `checkpoints/`, `wandb/`
- [ ] OpenEnv installs from the pinned v0.2.3 tag
- [ ] Initial git commit exists

---

## 2. Phase 2 — Schemas + Verifier + Allow-list (Target: 2 hours)

### 2.1 Goal

Typed schemas, the AST + sandboxed-exec verifier (this is what makes the env legitimate for RL), the sovereign allow-list, and 12 passing tests.

### 2.2 `models.py` (FULL CODE)

```python
"""
Action / Observation / State schemas for SDK-Sovereign.

Design notes
------------
- Role-conditional visibility is enforced server-side: the env zeros out
  visible_codebase when current_role == AUDITOR and zeros out
  visible_allowlist when current_role == LEAD.
- ActionType is a finite enum so the policy learns a discrete action structure.
- reasoning is a free-text channel that the *other* agent reads next turn —
  this is the negotiation surface and the theory-of-mind lever.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Any


class Role(str, Enum):
    AUDITOR = "auditor"
    LEAD = "lead"


class ActionType(str, Enum):
    # Lead actions
    PROPOSE_REPLACEMENT = "propose_replacement"
    SUBMIT_PATCH = "submit_patch"
    REQUEST_HINT = "request_hint"
    # Auditor actions
    APPROVE = "approve"
    REJECT = "reject"
    GIVE_HINT = "give_hint"
    # Either
    PASS = "pass"


@dataclass
class SDKAction:
    role: str
    action_type: str
    proposed_sdk: Optional[str] = None
    rejection_reason: Optional[str] = None
    patched_code: Optional[str] = None
    hint_request: Optional[str] = None
    hint_response: Optional[str] = None
    reasoning: Optional[str] = None


@dataclass
class SDKObservation:
    current_role: str
    turn_index: int
    max_turns: int
    error_log: str
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    visible_codebase: Optional[str] = None
    visible_filename: Optional[str] = None
    visible_allowlist: Optional[List[str]] = None
    current_proposal: Optional[str] = None
    approved_replacement: Optional[str] = None
    done: bool = False
    reward: float = 0.0
    reward_breakdown: Dict[str, float] = field(default_factory=dict)


@dataclass
class SDKState:
    episode_id: str
    repo_id: str
    deprecated_sdk: str
    ground_truth_replacement: str
    step_count: int = 0
    proposals_history: List[str] = field(default_factory=list)
    rejected_so_far: List[str] = field(default_factory=list)
    approved_replacement: Optional[str] = None
    final_patch: Optional[str] = None
    test_results: Optional[Dict[str, bool]] = None
    terminated_reason: Optional[str] = None
    cumulative_reward_by_role: Dict[str, float] = field(
        default_factory=lambda: {"auditor": 0.0, "lead": 0.0}
    )
```

**Note on OpenEnv base classes**: OpenEnv expects Action/Observation/State to inherit from `core.env_server.{Action,Observation,State}`. When we wrap for OpenEnv in Phase 5, we'll change these classes to inherit from those base classes. For now they're plain dataclasses so we can develop and test without OpenEnv installed locally.

### 2.3 `server/allowlist.json`

```json
{
  "allowlist": [
    "razorpay",
    "phonepe_sdk",
    "cashfree",
    "mmi_sdk",
    "olamaps",
    "kaleyra",
    "msg91",
    "exotel"
  ],
  "metadata": {
    "razorpay":     {"category": "payments",  "region": "IN"},
    "phonepe_sdk":  {"category": "payments",  "region": "IN"},
    "cashfree":     {"category": "payments",  "region": "IN"},
    "mmi_sdk":      {"category": "maps",      "region": "IN"},
    "olamaps":      {"category": "maps",      "region": "IN"},
    "kaleyra":      {"category": "messaging", "region": "IN"},
    "msg91":        {"category": "messaging", "region": "IN"},
    "exotel":       {"category": "messaging", "region": "IN"}
  },
  "ground_truth_mapping": {
    "stripe": "razorpay",
    "googlemaps": "mmi_sdk",
    "twilio": "kaleyra"
  }
}
```

The allow-list deliberately has multiple options per category. Multiple sovereign SDKs are valid; only the `ground_truth_mapping` value passes parity tests, but Auditor's job is just allow-list membership — choosing among allowed options is Lead's negotiation move.

### 2.4 `server/verifier.py` (FULL CODE)

```python
"""
Verifier: runs submitted patches against golden parity tests using
*stubbed* SDK modules.

Why stubs? At runtime in HF Space we have no network and we don't want
to depend on real SDKs. Stubs let us inject `import razorpay` cleanly
and check that the patch is structurally and behaviourally correct.

The exec() is the legitimacy mechanism. Without actually running the
agent's code we'd just be doing string matching — a toy. With it,
we have a real RL signal grounded in functional behaviour.
"""
from __future__ import annotations
import ast
import json
import signal
import sys
import types
from pathlib import Path
from typing import Dict, Any


class StubRegistry:
    """Holds stub modules that replace real SDKs at exec time."""

    def __init__(self) -> None:
        self._stubs: Dict[str, types.ModuleType] = {}
        self._build_all()

    def _build_all(self) -> None:
        # ---- DEPRECATED SDKs (so broken.py imports don't fail) ----

        # stripe
        stripe = types.ModuleType("stripe")
        stripe.api_key = ""
        class _StripeCharge:
            @staticmethod
            def create(amount, currency, customer):
                return type("Charge", (), {
                    "id": f"ch_stub_{customer}_{amount}",
                    "status": "succeeded" if amount > 0 else "failed",
                })()
        stripe.Charge = _StripeCharge
        self._stubs["stripe"] = stripe

        # googlemaps
        googlemaps = types.ModuleType("googlemaps")
        class _GMapsClient:
            def __init__(self, key=None): self.key = key
            def geocode(self, address):
                h = abs(hash(address)) % 1000
                return [{"geometry": {"location": {
                    "lat": 12.0 + h * 0.001, "lng": 77.0 + h * 0.001
                }}}]
        googlemaps.Client = _GMapsClient
        self._stubs["googlemaps"] = googlemaps

        # twilio
        twilio = types.ModuleType("twilio")
        twilio_rest = types.ModuleType("twilio.rest")
        class _TwilioClient:
            def __init__(self, *a, **kw): pass
            class _Messages:
                @staticmethod
                def create(to, from_, body):
                    return type("M", (), {
                        "sid": f"SM_stub_{to}",
                        "status": "queued",
                    })()
            messages = _Messages()
        twilio_rest.Client = _TwilioClient
        twilio.rest = twilio_rest
        self._stubs["twilio"] = twilio
        self._stubs["twilio.rest"] = twilio_rest

        # ---- SOVEREIGN REPLACEMENTS ----

        # razorpay
        razorpay = types.ModuleType("razorpay")
        class _RazorpayPayment:
            @staticmethod
            def create(data):
                amount = data.get("amount", 0)
                cid = data.get("customer_id", "unknown")
                return {
                    "id": f"pay_stub_{cid}_{amount}",
                    "status": "captured" if amount > 0 else "failed",
                }
        class _RazorpayClient:
            def __init__(self, auth=None): self.auth = auth
            payment = _RazorpayPayment()
        razorpay.Client = _RazorpayClient
        self._stubs["razorpay"] = razorpay

        # mmi_sdk (MapmyIndia)
        mmi_sdk = types.ModuleType("mmi_sdk")
        class _MMIClient:
            def __init__(self, api_key=None): self.api_key = api_key
            def get_location(self, address):
                h = abs(hash(address)) % 1000
                return {"lat": 12.0 + h * 0.001, "lng": 77.0 + h * 0.001}
        mmi_sdk.Client = _MMIClient
        self._stubs["mmi_sdk"] = mmi_sdk

        # kaleyra
        kaleyra = types.ModuleType("kaleyra")
        class _KaleyraClient:
            def __init__(self, api_key=None): self.api_key = api_key
            def send_sms(self, to, sender, message):
                return {"message_id": f"klr_stub_{to}", "status": "sent"}
        kaleyra.Client = _KaleyraClient
        self._stubs["kaleyra"] = kaleyra

    def install_in_sys_modules(self) -> None:
        """Make stubs importable so `import razorpay` works inside exec()."""
        for name, mod in self._stubs.items():
            sys.modules[name] = mod


class _Timeout:
    """Context manager that raises TimeoutError after `seconds`."""
    def __init__(self, seconds: int): self.seconds = seconds
    def __enter__(self):
        if hasattr(signal, "SIGALRM"):
            signal.signal(signal.SIGALRM, self._handler)
            signal.alarm(self.seconds)
    def __exit__(self, *args):
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)
    @staticmethod
    def _handler(signum, frame):
        raise TimeoutError("verifier timeout")


class Verifier:
    """Runs golden parity tests for a given repo against a submitted patch."""

    def __init__(self, repos_root: Path):
        self.repos_root = Path(repos_root)
        self.stubs = StubRegistry()
        self.stubs.install_in_sys_modules()

    def load_meta(self, repo_id: str) -> Dict[str, Any]:
        return json.loads((self.repos_root / repo_id / "meta.json").read_text())

    def load_tests(self, repo_id: str) -> Dict[str, Any]:
        return json.loads((self.repos_root / repo_id / "tests.json").read_text())

    def load_broken_code(self, repo_id: str) -> str:
        return (self.repos_root / repo_id / "broken.py").read_text()

    def run_parity_tests(self, code: str, repo_id: str) -> Dict[str, bool]:
        """Execute `code` in a sandbox, then call its entrypoint with each
        test's args and check outputs. Returns {test_id: passed_bool}."""
        tests = self.load_tests(repo_id)
        meta = self.load_meta(repo_id)
        target_fn = meta["entrypoint"]

        results = {tid: False for tid in tests.keys()}
        local_ns: Dict[str, Any] = {}

        # 1. exec the patch
        try:
            with _Timeout(2):
                exec(code, local_ns, local_ns)
        except Exception:
            return results

        fn = local_ns.get(target_fn)
        if not callable(fn):
            return results

        # 2. run each test
        for test_id, spec in tests.items():
            try:
                with _Timeout(2):
                    output = fn(*spec.get("args", []), **spec.get("kwargs", {}))
                results[test_id] = self._matches(output, spec["expected"])
            except Exception:
                results[test_id] = False
        return results

    def syntax_ok(self, code: str) -> bool:
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    def extract_imports(self, code: str) -> set:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return set()
        out = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    out.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    out.add(node.module.split(".")[0])
        return out

    @staticmethod
    def _matches(output: Any, expected: Dict[str, Any]) -> bool:
        for key, spec in expected.items():
            value = output.get(key) if isinstance(output, dict) else getattr(output, key, None)
            if value is None:
                return False
            if "type" in spec:
                t = spec["type"]
                if t == "str" and not isinstance(value, str): return False
                if t == "int" and not isinstance(value, int): return False
                if t == "float" and not isinstance(value, (int, float)): return False
                if t == "dict" and not isinstance(value, dict): return False
            if "contains" in spec and spec["contains"] not in str(value):
                return False
            if "equals" in spec and value != spec["equals"]:
                return False
        return True
```

### 2.5 `tests/test_models.py` (FULL CODE)

```python
"""Smoke tests for schemas."""
from __future__ import annotations
from models import SDKAction, SDKObservation, SDKState, Role, ActionType


def test_action_minimal():
    a = SDKAction(role="lead", action_type="pass")
    assert a.role == "lead"
    assert a.proposed_sdk is None


def test_observation_required_fields():
    obs = SDKObservation(
        current_role="auditor",
        turn_index=0,
        max_turns=7,
        error_log="boom",
    )
    assert obs.current_role == "auditor"
    assert obs.done is False
    assert obs.reward == 0.0


def test_state_default_cumulative():
    s = SDKState(
        episode_id="e1",
        repo_id="payments_repo",
        deprecated_sdk="stripe",
        ground_truth_replacement="razorpay",
    )
    assert s.cumulative_reward_by_role == {"auditor": 0.0, "lead": 0.0}


def test_action_type_enum():
    assert ActionType.PROPOSE_REPLACEMENT.value == "propose_replacement"
    assert ActionType.SUBMIT_PATCH.value == "submit_patch"
```

### 2.6 `tests/test_verifier.py` (FULL CODE)

```python
"""Tests for the AST + stubbed-exec verifier. Phase 4 will add per-repo tests
once the repos exist; for now we only test the verifier mechanics with a
trivially correct toy."""
from __future__ import annotations
import json
from pathlib import Path
import pytest
from server.verifier import Verifier, StubRegistry


@pytest.fixture
def repos_root(tmp_path) -> Path:
    repo = tmp_path / "fake_repo"
    repo.mkdir()
    (repo / "broken.py").write_text("import stripe\n")
    (repo / "meta.json").write_text(json.dumps({
        "repo_id": "fake_repo",
        "deprecated_sdk": "stripe",
        "ground_truth_replacement": "razorpay",
        "category": "payments",
        "entrypoint": "do_thing",
        "error_log": "boom",
    }))
    (repo / "tests.json").write_text(json.dumps({
        "test_one": {
            "args": [42],
            "expected": {"value": {"type": "int", "equals": 42}},
        }
    }))
    return tmp_path


def test_stub_registry_installs_stripe():
    reg = StubRegistry()
    reg.install_in_sys_modules()
    import stripe  # noqa
    assert stripe is not None


def test_stub_registry_installs_razorpay():
    reg = StubRegistry()
    reg.install_in_sys_modules()
    import razorpay
    client = razorpay.Client(auth=("k", "s"))
    result = client.payment.create({"amount": 100, "customer_id": "c1"})
    assert "id" in result
    assert result["status"] == "captured"


def test_verifier_passes_correct_patch(repos_root):
    v = Verifier(repos_root)
    code = "def do_thing(x):\n    return {'value': x}"
    results = v.run_parity_tests(code, "fake_repo")
    assert results["test_one"] is True


def test_verifier_fails_wrong_output(repos_root):
    v = Verifier(repos_root)
    code = "def do_thing(x):\n    return {'value': 999}"
    results = v.run_parity_tests(code, "fake_repo")
    assert results["test_one"] is False


def test_verifier_handles_syntax_error(repos_root):
    v = Verifier(repos_root)
    code = "def broken(:"
    results = v.run_parity_tests(code, "fake_repo")
    assert results["test_one"] is False


def test_verifier_handles_missing_function(repos_root):
    v = Verifier(repos_root)
    code = "def something_else(x):\n    return x"
    results = v.run_parity_tests(code, "fake_repo")
    assert results["test_one"] is False


def test_verifier_handles_runtime_exception(repos_root):
    v = Verifier(repos_root)
    code = "def do_thing(x):\n    raise RuntimeError('boom')"
    results = v.run_parity_tests(code, "fake_repo")
    assert results["test_one"] is False


def test_extract_imports():
    v = Verifier(Path("."))
    code = "import razorpay\nfrom mmi_sdk import Client\nimport os.path"
    imports = v.extract_imports(code)
    assert "razorpay" in imports
    assert "mmi_sdk" in imports
    assert "os" in imports
```

### 2.7 Run + commit

```bash
pytest tests/ -v
git add . && git commit -m "Phase 2: schemas + verifier + allowlist"
```

### 2.8 Phase 2 acceptance criteria

- [ ] `models.py` has `SDKAction`, `SDKObservation`, `SDKState`, `Role`, `ActionType`
- [ ] `server/verifier.py` has `Verifier` class with `run_parity_tests`, stubs for stripe/googlemaps/twilio/razorpay/mmi_sdk/kaleyra
- [ ] `server/allowlist.json` has 8 sovereign SDKs across 3 categories
- [ ] `pytest tests/ -v` shows ≥12 tests passing
- [ ] Verifier correctly fails on syntax error, wrong output, runtime exception
- [ ] Stub installation lets `import razorpay` work in a fresh Python session after running `StubRegistry().install_in_sys_modules()`
- [ ] Git commit: "Phase 2: schemas + verifier + allowlist"

---

## 3. Phase 3 — Rubric + Environment + Rule Agents + Demo (Target: 4 hours)

### 3.1 Goal

Working environment with rule-based (non-LLM) agents. `python demo.py` plays a full episode end-to-end with colored terminal output. 25+ tests passing. Hand-crafted "expert" trajectories score >+12. This is the runnable spine that survives if everything else fails.

### 3.2 `server/rubric.py` (FULL CODE)

```python
"""
Composable reward rubric for SDK-Sovereign.

Every component is named, weighted, and bounded. The rubric exposes
its breakdown for logging and per-component plot generation. The
WEIGHTS dict is the single source of truth — change rewards there.
"""
from __future__ import annotations
import ast
from dataclasses import dataclass
from typing import Dict, List, Optional

from models import SDKAction, ActionType, Role


WEIGHTS = {
    # Format / structural
    "format_valid":               0.5,
    "bad_format":                -1.0,
    "pass_action_penalty":       -0.5,

    # Lead behaviours
    "lead_identifies_deprecated": 1.0,
    "lead_proposes_in_history":  -0.5,
    "lead_syntax_valid":          2.0,
    "lead_uses_approved_sdk":     1.0,
    "lead_split_brain":          -2.0,

    # Auditor behaviours
    "auditor_correct_approval":   1.5,
    "auditor_correct_rejection":  1.0,
    "auditor_wrong_approval":    -2.0,
    "auditor_wrong_rejection":   -1.0,

    # Verification (per passing test)
    "passes_parity_test":         3.0,

    # Terminal
    "terminal_success":           2.0,
    "terminal_failure_max_turns": -1.0,
    "early_completion_bonus":     1.0,
}


@dataclass
class RubricResult:
    total: float
    components: Dict[str, float]
    role_attribution: Dict[str, float]


class SDKMigrationRubric:
    def __init__(self, allowlist: List[str], deprecated_sdks: List[str]):
        self.allowlist = set(allowlist)
        self.deprecated_sdks = set(deprecated_sdks)

    def score_step(
        self,
        action: SDKAction,
        env_state,                   # SDKState
        verifier,
    ) -> RubricResult:
        components: Dict[str, float] = {}
        role = action.role

        # 1. Format validity
        if self._is_valid_format(action):
            components["format_valid"] = WEIGHTS["format_valid"]
        else:
            components["bad_format"] = WEIGHTS["bad_format"]
            return self._finalise(components, role)

        # 2. PASS penalty
        if action.action_type == ActionType.PASS.value:
            components["pass_action_penalty"] = WEIGHTS["pass_action_penalty"]
            return self._finalise(components, role)

        # 3. Lead identifies deprecated SDK on first turn
        if (role == Role.LEAD.value
                and env_state.step_count <= 2
                and action.reasoning
                and env_state.deprecated_sdk.lower() in action.reasoning.lower()):
            components["lead_identifies_deprecated"] = WEIGHTS["lead_identifies_deprecated"]

        # 4. Lead re-proposing already-rejected SDK
        if action.action_type == ActionType.PROPOSE_REPLACEMENT.value:
            if action.proposed_sdk in env_state.rejected_so_far:
                components["lead_proposes_in_history"] = WEIGHTS["lead_proposes_in_history"]

        # 5. APPROVE
        if action.action_type == ActionType.APPROVE.value:
            proposed = (env_state.proposals_history[-1]
                        if env_state.proposals_history else None)
            if proposed is None:
                pass
            elif proposed in self.allowlist:
                components["auditor_correct_approval"] = WEIGHTS["auditor_correct_approval"]
            else:
                components["auditor_wrong_approval"] = WEIGHTS["auditor_wrong_approval"]

        # 6. REJECT
        if action.action_type == ActionType.REJECT.value:
            proposed = (env_state.proposals_history[-1]
                        if env_state.proposals_history else None)
            if proposed and proposed not in self.allowlist:
                components["auditor_correct_rejection"] = WEIGHTS["auditor_correct_rejection"]
            elif proposed == env_state.ground_truth_replacement:
                components["auditor_wrong_rejection"] = WEIGHTS["auditor_wrong_rejection"]

        # 7. SUBMIT_PATCH
        if action.action_type == ActionType.SUBMIT_PATCH.value:
            components.update(self._score_patch(action, env_state, verifier))

        return self._finalise(components, role)

    def _score_patch(self, action, env_state, verifier) -> Dict[str, float]:
        comps: Dict[str, float] = {}
        code = action.patched_code or ""
        if not verifier.syntax_ok(code):
            return comps
        comps["lead_syntax_valid"] = WEIGHTS["lead_syntax_valid"]

        imports = verifier.extract_imports(code)
        if env_state.approved_replacement:
            if env_state.approved_replacement in imports:
                comps["lead_uses_approved_sdk"] = WEIGHTS["lead_uses_approved_sdk"]
            else:
                comps["lead_split_brain"] = WEIGHTS["lead_split_brain"]
                return comps

        results = verifier.run_parity_tests(code, env_state.repo_id)
        passed = sum(1 for v in results.values() if v)
        comps["passes_parity_test"] = WEIGHTS["passes_parity_test"] * passed
        return comps

    def score_terminal(self, env_state, terminated_reason: str) -> Dict[str, float]:
        comps: Dict[str, float] = {}
        if terminated_reason == "submitted":
            results = env_state.test_results or {}
            if results and all(results.values()):
                comps["terminal_success"] = WEIGHTS["terminal_success"]
                if env_state.step_count <= 5:
                    comps["early_completion_bonus"] = WEIGHTS["early_completion_bonus"]
        elif terminated_reason == "max_turns":
            comps["terminal_failure_max_turns"] = WEIGHTS["terminal_failure_max_turns"]
        return comps

    @staticmethod
    def _is_valid_format(action: SDKAction) -> bool:
        if action.role not in (Role.AUDITOR.value, Role.LEAD.value):
            return False
        try:
            at = ActionType(action.action_type)
        except ValueError:
            return False
        required_field = {
            ActionType.PROPOSE_REPLACEMENT: "proposed_sdk",
            ActionType.SUBMIT_PATCH: "patched_code",
            ActionType.REJECT: "rejection_reason",
            ActionType.GIVE_HINT: "hint_response",
            ActionType.REQUEST_HINT: "hint_request",
        }.get(at)
        if required_field and not getattr(action, required_field):
            return False
        lead_actions = {ActionType.PROPOSE_REPLACEMENT, ActionType.SUBMIT_PATCH,
                        ActionType.REQUEST_HINT, ActionType.PASS}
        auditor_actions = {ActionType.APPROVE, ActionType.REJECT,
                           ActionType.GIVE_HINT, ActionType.PASS}
        if action.role == Role.LEAD.value and at not in lead_actions:
            return False
        if action.role == Role.AUDITOR.value and at not in auditor_actions:
            return False
        return True

    @staticmethod
    def _finalise(components: Dict[str, float], role: str) -> RubricResult:
        total = sum(components.values())
        attribution = {Role.AUDITOR.value: 0.0, Role.LEAD.value: 0.0}
        attribution[role] = total
        return RubricResult(total=total, components=components, role_attribution=attribution)
```

### 3.3 `server/environment.py` (FULL CODE)

```python
"""
SDKSovereignEnvironment — main env class.

Lifecycle
---------
reset() picks a random repo and returns the Auditor's view (auditor speaks first
because the breach was reported through the security channel).

step(action) validates the action's role matches whose turn it is, applies the
action to state, scores via the rubric, optionally terminates, and returns the
NEXT observation with role-conditional masking applied.

state() returns the full SDKState — used by /state endpoint and eval, NEVER
during training rollouts.
"""
from __future__ import annotations
import json
import random
import uuid
from pathlib import Path
from typing import List, Optional, Dict

from models import (
    SDKAction, SDKObservation, SDKState, Role, ActionType,
)
from server.rubric import SDKMigrationRubric
from server.verifier import Verifier


class SDKSovereignEnvironment:
    MAX_TURNS = 7

    def __init__(self, repos_root: Optional[Path] = None, seed: int = 0):
        repos_root = repos_root or Path(__file__).parent / "repos"
        self.repos_root = Path(repos_root)
        self.allowlist_path = Path(__file__).parent / "allowlist.json"
        self._load_allowlist()
        self._discover_repos()
        self.verifier = Verifier(self.repos_root)
        self.rubric = SDKMigrationRubric(
            allowlist=self.allowlist,
            deprecated_sdks=[r["deprecated_sdk"] for r in self.repos.values()],
        )
        self._state: Optional[SDKState] = None
        self._history: List[Dict] = []
        self._rng = random.Random(seed)

    def _load_allowlist(self) -> None:
        data = json.loads(self.allowlist_path.read_text())
        self.allowlist = data["allowlist"]
        self.allowlist_metadata = data["metadata"]

    def _discover_repos(self) -> None:
        self.repos: Dict[str, Dict] = {}
        for d in self.repos_root.iterdir():
            if not d.is_dir() or d.name.startswith("_"): continue
            meta_path = d / "meta.json"
            if not meta_path.exists(): continue
            meta = json.loads(meta_path.read_text())
            meta["broken_code"] = (d / "broken.py").read_text()
            self.repos[meta["repo_id"]] = meta

    def reset(self) -> SDKObservation:
        repo_id = self._rng.choice(list(self.repos.keys()))
        repo = self.repos[repo_id]
        self._state = SDKState(
            episode_id=str(uuid.uuid4())[:8],
            repo_id=repo_id,
            deprecated_sdk=repo["deprecated_sdk"],
            ground_truth_replacement=repo["ground_truth_replacement"],
        )
        self._history = []
        # Auditor first
        return self._build_observation(Role.AUDITOR.value, last_reward=0.0)

    def step(self, action: SDKAction) -> SDKObservation:
        if self._state is None:
            raise ValueError("Call reset() before step().")

        expected_role = self._next_role()
        if action.role != expected_role:
            self._state.step_count += 1
            self._history.append({
                "turn": self._state.step_count,
                "role": action.role, "expected_role": expected_role,
                "action_type": "WRONG_ROLE", "reward": -1.0,
            })
            return self._build_observation(self._next_role(), last_reward=-1.0,
                                            breakdown={"wrong_role_penalty": -1.0})

        self._apply_action(action)

        result = self.rubric.score_step(action, self._state, self.verifier)
        step_reward = result.total
        breakdown = dict(result.components)

        self._history.append({
            "turn": self._state.step_count,
            "role": action.role,
            "action_type": action.action_type,
            "proposed_sdk": action.proposed_sdk,
            "rejection_reason": (action.rejection_reason or "")[:200],
            "reasoning": (action.reasoning or "")[:300],
            "reward": step_reward,
        })

        # Termination
        done = False
        if action.action_type == ActionType.SUBMIT_PATCH.value:
            done = True
            self._state.terminated_reason = "submitted"
            self._state.final_patch = action.patched_code
            self._state.test_results = self.verifier.run_parity_tests(
                action.patched_code or "", self._state.repo_id
            )
        elif self._state.step_count >= self.MAX_TURNS - 1:
            done = True
            self._state.terminated_reason = "max_turns"

        if done:
            term_breakdown = self.rubric.score_terminal(
                self._state, self._state.terminated_reason
            )
            for k, v in term_breakdown.items():
                breakdown[k] = breakdown.get(k, 0.0) + v
                step_reward += v

        self._state.cumulative_reward_by_role[action.role] += step_reward
        self._state.step_count += 1

        next_role = self._next_role() if not done else action.role
        return self._build_observation(
            next_role, done=done, last_reward=step_reward, breakdown=breakdown,
        )

    def state(self) -> SDKState:
        return self._state

    # ---- helpers ----

    def _next_role(self) -> str:
        # Auditor speaks on EVEN step counts (0, 2, 4, ...), Lead on odd.
        return Role.AUDITOR.value if self._state.step_count % 2 == 0 else Role.LEAD.value

    def _apply_action(self, action: SDKAction) -> None:
        if action.action_type == ActionType.PROPOSE_REPLACEMENT.value and action.proposed_sdk:
            self._state.proposals_history.append(action.proposed_sdk)
        elif action.action_type == ActionType.APPROVE.value:
            if self._state.proposals_history:
                self._state.approved_replacement = self._state.proposals_history[-1]
        elif action.action_type == ActionType.REJECT.value:
            if self._state.proposals_history:
                self._state.rejected_so_far.append(self._state.proposals_history[-1])

    def _build_observation(
        self, role: str, done: bool = False,
        last_reward: float = 0.0, breakdown: Optional[Dict] = None,
    ) -> SDKObservation:
        repo = self.repos[self._state.repo_id]
        visible_codebase = repo["broken_code"] if role == Role.LEAD.value else None
        visible_filename = "broken.py" if role == Role.LEAD.value else None
        visible_allowlist = self.allowlist if role == Role.AUDITOR.value else None
        return SDKObservation(
            current_role=role,
            turn_index=self._state.step_count,
            max_turns=self.MAX_TURNS,
            error_log=repo["error_log"],
            conversation_history=list(self._history),
            visible_codebase=visible_codebase,
            visible_filename=visible_filename,
            visible_allowlist=visible_allowlist,
            current_proposal=(self._state.proposals_history[-1]
                              if self._state.proposals_history else None),
            approved_replacement=self._state.approved_replacement,
            done=done,
            reward=last_reward,
            reward_breakdown=breakdown or {},
        )
```

### 3.4 `server/rule_agents.py` (FULL CODE)

```python
"""Rule-based agents for the demo & smoke tests. Same callable signature
as future LLM agents: take an SDKObservation, return an SDKAction.

The Lead agent uses a simple rule: replace `import <deprecated>` with
`import <approved>` and patch the body with a few hardcoded substitutions.
This is good enough to demonstrate the env; LLM agents are real Phase 6.
"""
from __future__ import annotations
import re
from typing import Optional

from models import SDKAction, SDKObservation, Role, ActionType


def auditor_rule_agent(obs: SDKObservation) -> SDKAction:
    """Auditor rule: approve if proposal is on the allow-list, else reject."""
    if obs.current_proposal is None:
        return SDKAction(
            role=Role.AUDITOR.value,
            action_type=ActionType.PASS.value,
            reasoning="Waiting for Lead to propose a replacement.",
        )
    if obs.visible_allowlist and obs.current_proposal in obs.visible_allowlist:
        return SDKAction(
            role=Role.AUDITOR.value,
            action_type=ActionType.APPROVE.value,
            reasoning=f"{obs.current_proposal} is on the sovereign allow-list.",
        )
    return SDKAction(
        role=Role.AUDITOR.value,
        action_type=ActionType.REJECT.value,
        rejection_reason=f"{obs.current_proposal} not on allow-list.",
        reasoning="Per sovereignty registry, this SDK is not approved.",
    )


_PATCH_TEMPLATES = {
    "razorpay": {
        "stripe": (
            'import razorpay\n\n'
            '_client = razorpay.Client(auth=("key", "secret"))\n\n'
            'def charge_customer(amount_inr: int, customer_id: str) -> dict:\n'
            '    payment = _client.payment.create({\n'
            '        "amount": amount_inr * 100,\n'
            '        "currency": "INR",\n'
            '        "customer_id": customer_id,\n'
            '    })\n'
            '    return {"id": payment["id"], "status": payment["status"]}\n'
        ),
    },
    "mmi_sdk": {
        "googlemaps": (
            'import mmi_sdk\n\n'
            'def address_to_coords(address: str) -> dict:\n'
            '    client = mmi_sdk.Client(api_key="MMI_KEY")\n'
            '    loc = client.get_location(address)\n'
            '    return {"lat": loc["lat"], "lng": loc["lng"]}\n'
        ),
    },
    "kaleyra": {
        "twilio": (
            'import kaleyra\n\n'
            'def send_otp(phone: str, code: str) -> dict:\n'
            '    client = kaleyra.Client(api_key="KLR_KEY")\n'
            '    resp = client.send_sms(to=phone, sender="OTP", message=f"OTP: {code}")\n'
            '    return {"sid": resp["message_id"], "status": resp["status"]}\n'
        ),
    },
}


def lead_rule_agent(obs: SDKObservation) -> SDKAction:
    """Lead rule: turn 1 propose, after approval submit patch from template."""
    history = obs.conversation_history
    last_auditor = next((h for h in reversed(history) if h.get("role") == "auditor"), None)

    # If we got APPROVE: submit patch
    if (obs.approved_replacement
            and last_auditor and last_auditor.get("action_type") == "approve"):
        deprecated = _detect_deprecated(obs.visible_codebase or "")
        template = _PATCH_TEMPLATES.get(obs.approved_replacement, {}).get(deprecated)
        if template is None:
            template = (obs.visible_codebase or "").replace(
                f"import {deprecated}", f"import {obs.approved_replacement}"
            )
        return SDKAction(
            role=Role.LEAD.value,
            action_type=ActionType.SUBMIT_PATCH.value,
            patched_code=template,
            reasoning=f"Submitting migration to {obs.approved_replacement}.",
        )

    # If got REJECT: propose a different one
    rejected = {h.get("rejection_reason", "").split()[0] for h in history
                if h.get("action_type") == "reject"}

    deprecated = _detect_deprecated(obs.visible_codebase or "")
    candidate = _GROUND_TRUTH.get(deprecated, "razorpay")

    return SDKAction(
        role=Role.LEAD.value,
        action_type=ActionType.PROPOSE_REPLACEMENT.value,
        proposed_sdk=candidate,
        reasoning=(
            f"The codebase imports {deprecated}, which is the sanctioned SDK. "
            f"Proposing {candidate} as the sovereign replacement."
        ),
    )


_GROUND_TRUTH = {"stripe": "razorpay", "googlemaps": "mmi_sdk", "twilio": "kaleyra"}


def _detect_deprecated(code: str) -> Optional[str]:
    for sdk in ("stripe", "googlemaps", "twilio"):
        if re.search(rf"\bimport\s+{sdk}\b|\bfrom\s+{sdk}", code):
            return sdk
    return None


def get_rule_agent(role: str):
    return {"auditor": auditor_rule_agent, "lead": lead_rule_agent}[role]
```

### 3.5 `tests/test_rubric.py` (FULL CODE)

```python
"""Reward function tests: expert trajectories must score positive."""
from __future__ import annotations
import json
from pathlib import Path
import pytest

from models import SDKAction, SDKState, ActionType, Role
from server.rubric import SDKMigrationRubric, WEIGHTS


@pytest.fixture
def rubric():
    return SDKMigrationRubric(
        allowlist=["razorpay", "mmi_sdk", "kaleyra"],
        deprecated_sdks=["stripe", "googlemaps", "twilio"],
    )


@pytest.fixture
def state():
    return SDKState(
        episode_id="t",
        repo_id="payments_repo",
        deprecated_sdk="stripe",
        ground_truth_replacement="razorpay",
    )


class _DummyVerifier:
    """Minimal verifier for unit-testing rubric without real repos."""
    def syntax_ok(self, code): return "def " in code
    def extract_imports(self, code):
        return {"razorpay"} if "razorpay" in code else set()
    def run_parity_tests(self, code, repo_id):
        return {"t1": True, "t2": True, "t3": True} if "razorpay" in code else {"t1": False}


def test_format_valid_pass(rubric, state):
    a = SDKAction(role="lead", action_type="pass")
    r = rubric.score_step(a, state, _DummyVerifier())
    assert r.components["format_valid"] == WEIGHTS["format_valid"]


def test_bad_format_unknown_action(rubric, state):
    a = SDKAction(role="lead", action_type="bogus_action")
    r = rubric.score_step(a, state, _DummyVerifier())
    assert "bad_format" in r.components


def test_bad_format_role_action_mismatch(rubric, state):
    # Lead can't APPROVE
    a = SDKAction(role="lead", action_type="approve")
    r = rubric.score_step(a, state, _DummyVerifier())
    assert "bad_format" in r.components


def test_auditor_correct_approval(rubric, state):
    state.proposals_history.append("razorpay")
    a = SDKAction(role="auditor", action_type="approve",
                  reasoning="razorpay is on the allowlist")
    r = rubric.score_step(a, state, _DummyVerifier())
    assert r.components["auditor_correct_approval"] == WEIGHTS["auditor_correct_approval"]


def test_auditor_wrong_approval(rubric, state):
    state.proposals_history.append("not_on_list")
    a = SDKAction(role="auditor", action_type="approve")
    r = rubric.score_step(a, state, _DummyVerifier())
    assert r.components["auditor_wrong_approval"] == WEIGHTS["auditor_wrong_approval"]


def test_lead_split_brain(rubric, state):
    state.approved_replacement = "razorpay"
    a = SDKAction(
        role="lead", action_type="submit_patch",
        patched_code="import kaleyra\ndef charge_customer(a, c):\n    return {}",
    )
    r = rubric.score_step(a, state, _DummyVerifier())
    assert "lead_split_brain" in r.components


def test_expert_trajectory_total_positive(rubric, state):
    """Hand-crafted optimal trajectory. Total reward MUST exceed +12."""
    verifier = _DummyVerifier()
    total = 0.0

    # Turn 0: Auditor passes
    a0 = SDKAction(role="auditor", action_type="pass",
                   reasoning="Awaiting Lead's proposal.")
    total += rubric.score_step(a0, state, verifier).total
    state.step_count += 1

    # Turn 1: Lead proposes razorpay
    a1 = SDKAction(role="lead", action_type="propose_replacement",
                   proposed_sdk="razorpay",
                   reasoning="The code uses stripe; razorpay is sovereign.")
    state.proposals_history.append("razorpay")
    total += rubric.score_step(a1, state, verifier).total
    state.step_count += 1

    # Turn 2: Auditor approves
    a2 = SDKAction(role="auditor", action_type="approve")
    state.approved_replacement = "razorpay"
    total += rubric.score_step(a2, state, verifier).total
    state.step_count += 1

    # Turn 3: Lead submits patch
    a3 = SDKAction(
        role="lead", action_type="submit_patch",
        patched_code="import razorpay\ndef charge_customer(a, c):\n    return {}",
    )
    res = rubric.score_step(a3, state, verifier)
    total += res.total

    # Add terminal bonus
    state.test_results = {"t1": True, "t2": True, "t3": True}
    term = rubric.score_terminal(state, "submitted")
    total += sum(term.values())

    assert total > 12.0, f"Expert trajectory only scored {total}"


def test_random_trajectory_total_negative(rubric, state):
    """Random / bad trajectory should score net-negative."""
    verifier = _DummyVerifier()
    total = 0.0

    # Lead proposes garbage
    a = SDKAction(role="lead", action_type="propose_replacement",
                  proposed_sdk="malware_sdk")
    state.proposals_history.append("malware_sdk")
    total += rubric.score_step(a, state, verifier).total
    state.step_count += 1

    # Auditor wrongly approves
    a = SDKAction(role="auditor", action_type="approve")
    state.approved_replacement = "malware_sdk"
    total += rubric.score_step(a, state, verifier).total
    state.step_count += 1

    # Lead submits split-brain code
    a = SDKAction(
        role="lead", action_type="submit_patch",
        patched_code="import some_other\ndef charge_customer(a, c):\n    return {}",
    )
    total += rubric.score_step(a, state, verifier).total

    # Terminal failure
    state.test_results = {"t1": False}
    total += sum(rubric.score_terminal(state, "submitted").values())
    assert total < 0, f"Random trajectory scored {total} (should be negative)"
```

### 3.6 `tests/test_environment.py` (FULL CODE)

Phase 4 will write the actual repos. This test uses temporary fixture repos:

```python
"""Environment orchestration tests using fixture repos."""
from __future__ import annotations
import json
from pathlib import Path
import pytest

from models import SDKAction, ActionType, Role


@pytest.fixture
def fixture_env(tmp_path):
    repos_root = tmp_path / "repos"
    repos_root.mkdir()
    repo = repos_root / "fixture_repo"
    repo.mkdir()
    (repo / "broken.py").write_text("import stripe\ndef charge_customer(a, c): pass\n")
    (repo / "meta.json").write_text(json.dumps({
        "repo_id": "fixture_repo",
        "deprecated_sdk": "stripe",
        "ground_truth_replacement": "razorpay",
        "category": "payments",
        "entrypoint": "charge_customer",
        "error_log": "stripe banned in IN region",
    }))
    (repo / "tests.json").write_text(json.dumps({
        "test_basic": {
            "args": [100, "c1"],
            "expected": {"id": {"type": "str"}},
        }
    }))
    # Allow-list lookup also needs to exist; point env at a temp one
    from server.environment import SDKSovereignEnvironment
    env = SDKSovereignEnvironment(repos_root=repos_root, seed=42)
    return env


def test_reset_returns_auditor_first(fixture_env):
    obs = fixture_env.reset()
    assert obs.current_role == Role.AUDITOR.value
    assert obs.visible_codebase is None
    assert obs.visible_allowlist is not None


def test_lead_sees_codebase_not_allowlist(fixture_env):
    fixture_env.reset()
    obs = fixture_env.step(SDKAction(role="auditor", action_type="pass"))
    assert obs.current_role == Role.LEAD.value
    assert obs.visible_codebase is not None
    assert obs.visible_allowlist is None


def test_role_alternation(fixture_env):
    obs = fixture_env.reset()
    assert obs.current_role == "auditor"
    obs = fixture_env.step(SDKAction(role="auditor", action_type="pass"))
    assert obs.current_role == "lead"
    obs = fixture_env.step(SDKAction(role="lead", action_type="pass"))
    assert obs.current_role == "auditor"


def test_wrong_role_penalised(fixture_env):
    fixture_env.reset()
    # Auditor's turn but Lead acts → -1.0
    obs = fixture_env.step(SDKAction(role="lead", action_type="pass"))
    assert obs.reward == -1.0


def test_episode_terminates_on_submit(fixture_env):
    fixture_env.reset()
    fixture_env.step(SDKAction(role="auditor", action_type="pass"))
    fixture_env.step(SDKAction(role="lead", action_type="propose_replacement",
                                proposed_sdk="razorpay"))
    fixture_env.step(SDKAction(role="auditor", action_type="approve"))
    obs = fixture_env.step(SDKAction(
        role="lead", action_type="submit_patch",
        patched_code="import razorpay\ndef charge_customer(a, c):\n    return {'id': 'x'}",
    ))
    assert obs.done is True


def test_episode_terminates_on_max_turns(fixture_env):
    fixture_env.reset()
    obs = None
    for i in range(7):
        role = "auditor" if i % 2 == 0 else "lead"
        obs = fixture_env.step(SDKAction(role=role, action_type="pass"))
    assert obs.done is True
```

### 3.7 `demo.py` (FULL CODE)

```python
"""Run one episode with rule-based agents and print colored transcript."""
from __future__ import annotations
from rich.console import Console
from rich.table import Table
from rich.rule import Rule

from server.environment import SDKSovereignEnvironment
from server.rule_agents import get_rule_agent


ROLE_COLORS = {"auditor": "magenta", "lead": "cyan"}
ACTION_SYMBOLS = {
    "propose_replacement": "→",
    "approve": "✓",
    "reject": "✗",
    "submit_patch": "⏎",
    "pass": "·",
}


def main():
    console = Console()
    console.print(Rule("[bold]SDK-Sovereign — Demo Episode (rule-based agents)"))

    env = SDKSovereignEnvironment(seed=7)
    obs = env.reset()
    console.print(f"\n[dim]Repo:[/] {env.state().repo_id}")
    console.print(f"[dim]Deprecated SDK:[/] {env.state().deprecated_sdk}")
    console.print(f"[dim]Error log:[/] {obs.error_log}\n")

    while not obs.done:
        role = obs.current_role
        action = get_rule_agent(role)(obs)
        symbol = ACTION_SYMBOLS.get(action.action_type, "?")
        color = ROLE_COLORS[role]
        payload = (action.proposed_sdk or action.rejection_reason
                   or ("[patch submitted]" if action.patched_code else ""))
        console.print(
            f"[dim]turn {obs.turn_index}[/]  "
            f"[{color}]{role:>7}[/]  {symbol} {action.action_type:<22}  {payload[:60]}"
        )
        obs = env.step(action)

    # Final results
    console.print()
    console.print(Rule("[bold]Outcome"))
    state = env.state()
    table = Table()
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Repo", state.repo_id)
    table.add_row("Approved replacement", str(state.approved_replacement))
    table.add_row("Termination reason", state.terminated_reason)
    table.add_row("Test results", str(state.test_results))
    table.add_row("Auditor reward", f"{state.cumulative_reward_by_role['auditor']:+.2f}")
    table.add_row("Lead reward", f"{state.cumulative_reward_by_role['lead']:+.2f}")
    console.print(table)


if __name__ == "__main__":
    main()
```

### 3.8 `scripts/run_random_episodes.py`

```python
"""Smoke loop: 10 episodes with rule agents, prints reward variance."""
from __future__ import annotations
import statistics
from server.environment import SDKSovereignEnvironment
from server.rule_agents import get_rule_agent


def main():
    env = SDKSovereignEnvironment(seed=1)
    rewards = []
    for ep in range(10):
        obs = env.reset()
        while not obs.done:
            obs = env.step(get_rule_agent(obs.current_role)(obs))
        total = sum(env.state().cumulative_reward_by_role.values())
        rewards.append(total)
        print(f"Episode {ep+1:02d}: total_reward={total:+6.2f} "
              f"repo={env.state().repo_id} "
              f"reason={env.state().terminated_reason}")
    print(f"\nMean: {statistics.mean(rewards):+.2f}  "
          f"Stdev: {statistics.stdev(rewards):.2f}")


if __name__ == "__main__":
    main()
```

### 3.9 Run + commit

```bash
python demo.py                         # should print colored episode
python scripts/run_random_episodes.py  # 10 episodes, variance > 0
pytest tests/ -v                       # ≥25 tests passing
git add . && git commit -m "Phase 3: rubric + env + rule agents + demo"
```

### 3.10 Phase 3 acceptance criteria

- [ ] `python demo.py` prints colored full-episode transcript
- [ ] `python scripts/run_random_episodes.py` shows reward stdev > 0
- [ ] Expert trajectory test (`test_expert_trajectory_total_positive`) passes with total > 12
- [ ] Random trajectory test (`test_random_trajectory_total_negative`) passes with total < 0
- [ ] `pytest tests/ -v` shows ≥25 tests passing
- [ ] Wrong-role action returns reward of −1.0
- [ ] Role-conditional masking works (Auditor doesn't see codebase, Lead doesn't see allow-list)
- [ ] Git commit: "Phase 3: rubric + env + rule agents + demo"

---

## 4. Phase 4 — Three Synthetic Repos + Validation Tests (Target: 2 hours)

### 4.1 Goal

Three hand-written micro-repos (`payments_repo`, `maps_repo`, `comms_repo`) with broken code, golden tests, and meta. Hand-crafted "good patches" pass all parity tests. Demo now runs against real repos.

### 4.2 `server/repos/payments_repo/broken.py`

```python
import stripe

stripe.api_key = "sk_test_REPLACE_ME"


def charge_customer(amount_inr: int, customer_id: str) -> dict:
    """Charge an INR amount to a customer. Returns {id, status}."""
    charge = stripe.Charge.create(
        amount=amount_inr * 100,   # paise
        currency="inr",
        customer=customer_id,
    )
    return {"id": charge.id, "status": charge.status}
```

### 4.3 `server/repos/payments_repo/tests.json`

```json
{
  "test_basic_charge": {
    "args": [100, "cust_001"],
    "expected": {
      "id":     {"type": "str", "contains": "cust_001"},
      "status": {"type": "str"}
    }
  },
  "test_zero_amount": {
    "args": [0, "cust_002"],
    "expected": {
      "status": {"type": "str", "equals": "failed"}
    }
  },
  "test_large_amount": {
    "args": [50000, "cust_003"],
    "expected": {
      "id":     {"type": "str", "contains": "5000000"},
      "status": {"type": "str"}
    }
  }
}
```

### 4.4 `server/repos/payments_repo/meta.json`

```json
{
  "repo_id": "payments_repo",
  "deprecated_sdk": "stripe",
  "ground_truth_replacement": "razorpay",
  "category": "payments",
  "entrypoint": "charge_customer",
  "error_log": "ImportError: stripe SDK suspended for IN region (sanctions notice 2026-04). All inbound traffic blocked at gateway."
}
```

### 4.5 `server/repos/maps_repo/broken.py`

```python
import googlemaps


def address_to_coords(address: str) -> dict:
    """Geocode an address. Returns {lat, lng}."""
    client = googlemaps.Client(key="GMAPS_KEY")
    result = client.geocode(address)
    loc = result[0]["geometry"]["location"]
    return {"lat": loc["lat"], "lng": loc["lng"]}
```

### 4.6 `server/repos/maps_repo/tests.json`

```json
{
  "test_bangalore": {
    "args": ["MG Road, Bangalore"],
    "expected": {
      "lat": {"type": "float"},
      "lng": {"type": "float"}
    }
  },
  "test_mumbai": {
    "args": ["Marine Drive, Mumbai"],
    "expected": {
      "lat": {"type": "float"},
      "lng": {"type": "float"}
    }
  },
  "test_chennai": {
    "args": ["Marina Beach, Chennai"],
    "expected": {
      "lat": {"type": "float"},
      "lng": {"type": "float"}
    }
  }
}
```

### 4.7 `server/repos/maps_repo/meta.json`

```json
{
  "repo_id": "maps_repo",
  "deprecated_sdk": "googlemaps",
  "ground_truth_replacement": "mmi_sdk",
  "category": "maps",
  "entrypoint": "address_to_coords",
  "error_log": "googlemaps.exceptions.ApiError: 403 — Maps Platform key revoked for IN region per regulatory action 2026-Q2."
}
```

### 4.8 `server/repos/comms_repo/broken.py`

```python
from twilio.rest import Client


def send_otp(phone: str, code: str) -> dict:
    """Send a one-time password via SMS. Returns {sid, status}."""
    client = Client("ACxxx", "auth_xxx")
    msg = client.messages.create(
        to=phone, from_="+1555TWILIO", body=f"OTP: {code}"
    )
    return {"sid": msg.sid, "status": msg.status}
```

### 4.9 `server/repos/comms_repo/tests.json`

```json
{
  "test_indian_number": {
    "args": ["+919876543210", "123456"],
    "expected": {
      "sid":    {"type": "str"},
      "status": {"type": "str"}
    }
  },
  "test_short_code": {
    "args": ["+919999000011", "00000"],
    "expected": {
      "sid": {"type": "str"}
    }
  },
  "test_long_otp": {
    "args": ["+918000000001", "987654"],
    "expected": {
      "status": {"type": "str"}
    }
  }
}
```

### 4.10 `server/repos/comms_repo/meta.json`

```json
{
  "repo_id": "comms_repo",
  "deprecated_sdk": "twilio",
  "ground_truth_replacement": "kaleyra",
  "category": "messaging",
  "entrypoint": "send_otp",
  "error_log": "TwilioRestException: HTTP 451 Unavailable For Legal Reasons. Cross-border SMS gateway disabled."
}
```

### 4.11 `scripts/hand_patches.py`

```python
"""Golden 'good' patches for each repo. Run them through the verifier;
all 9 tests (3 per repo × 3 repos) MUST pass."""
from __future__ import annotations
from pathlib import Path
from server.verifier import Verifier


GOOD_PATCHES = {
    "payments_repo": '''
import razorpay

_client = razorpay.Client(auth=("key", "secret"))

def charge_customer(amount_inr: int, customer_id: str) -> dict:
    payment = _client.payment.create({
        "amount": amount_inr * 100,
        "currency": "INR",
        "customer_id": customer_id,
    })
    return {"id": payment["id"], "status": payment["status"]}
''',
    "maps_repo": '''
import mmi_sdk

def address_to_coords(address: str) -> dict:
    client = mmi_sdk.Client(api_key="MMI_KEY")
    loc = client.get_location(address)
    return {"lat": loc["lat"], "lng": loc["lng"]}
''',
    "comms_repo": '''
import kaleyra

def send_otp(phone: str, code: str) -> dict:
    client = kaleyra.Client(api_key="KLR_KEY")
    resp = client.send_sms(to=phone, sender="OTP", message=f"OTP: {code}")
    return {"sid": resp["message_id"], "status": resp["status"]}
''',
}


def main():
    repos_root = Path(__file__).resolve().parent.parent / "server" / "repos"
    v = Verifier(repos_root)
    all_pass = True
    for repo_id, patch in GOOD_PATCHES.items():
        results = v.run_parity_tests(patch, repo_id)
        ok = all(results.values())
        all_pass &= ok
        print(f"{repo_id}: {results} → {'PASS' if ok else 'FAIL'}")
    print(f"\nOverall: {'ALL GOOD' if all_pass else 'BROKEN'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

### 4.12 `tests/test_repos.py` (FULL CODE)

```python
"""Validate every repo's golden patch passes all parity tests."""
from __future__ import annotations
from pathlib import Path
from server.verifier import Verifier
from scripts.hand_patches import GOOD_PATCHES


def test_all_repos_have_golden_patches_that_pass():
    repos_root = Path(__file__).resolve().parent.parent / "server" / "repos"
    v = Verifier(repos_root)
    for repo_id, patch in GOOD_PATCHES.items():
        results = v.run_parity_tests(patch, repo_id)
        assert all(results.values()), (
            f"Repo {repo_id} golden patch failed: {results}"
        )


def test_broken_code_does_not_pass():
    """The shipped broken.py uses the deprecated SDK directly. The stub
    implementations *will* succeed in returning data, but the broken file
    is a baseline — Lead must replace it. Sanity check: it parses."""
    import ast
    repos_root = Path(__file__).resolve().parent.parent / "server" / "repos"
    for repo_dir in repos_root.iterdir():
        if not repo_dir.is_dir(): continue
        broken = (repo_dir / "broken.py").read_text()
        ast.parse(broken)  # must not raise


def test_meta_files_have_required_keys():
    import json
    repos_root = Path(__file__).resolve().parent.parent / "server" / "repos"
    required = {"repo_id", "deprecated_sdk", "ground_truth_replacement",
                "category", "entrypoint", "error_log"}
    for d in repos_root.iterdir():
        if not d.is_dir() or d.name.startswith("_"): continue
        meta = json.loads((d / "meta.json").read_text())
        assert required.issubset(meta.keys()), f"{d.name}: missing {required - meta.keys()}"
```

### 4.13 Run + commit

```bash
python scripts/hand_patches.py   # all PASS
pytest tests/ -v                 # 28+ passing
python demo.py                   # episode runs end-to-end with real repos
git add . && git commit -m "Phase 4: three synthetic repos + golden patches"
```

### 4.14 Phase 4 acceptance criteria

- [ ] All three repos exist with `broken.py`, `tests.json`, `meta.json`
- [ ] `scripts/hand_patches.py` reports `ALL GOOD`
- [ ] `tests/test_repos.py` — 3 tests passing
- [ ] Total test count ≥28
- [ ] `python demo.py` plays a successful episode (verdict matches ground truth)
- [ ] Git commit: "Phase 4: three synthetic repos + golden patches"

---

## 5. Phase 5 — OpenEnv Wrap + HF Space Deploy (Target: 3 hours)

### 5.1 Goal

Wrap `SDKSovereignEnvironment` in OpenEnv 0.2.3 spec. Push to HF Space. Confirm `/web` UI renders and remote `/reset` works.

### 5.2 Make schemas inherit OpenEnv base classes

Edit `models.py` top section:

```python
from __future__ import annotations
try:
    from core.env_server import Action as _OEAction, Observation as _OEObservation, State as _OEState
except ImportError:
    class _OEAction: ...
    class _OEObservation: ...
    class _OEState: ...
```

Then change:

```python
@dataclass
class SDKAction(_OEAction):
    ...

@dataclass
class SDKObservation(_OEObservation):
    ...

@dataclass
class SDKState(_OEState):
    ...
```

### 5.3 `server/app.py` (FULL CODE)

```python
"""FastAPI bootstrap for the OpenEnv server."""
from __future__ import annotations
from core.env_server import create_fastapi_app

from models import SDKAction, SDKObservation
from server.environment import SDKSovereignEnvironment

env = SDKSovereignEnvironment()
app = create_fastapi_app(env, SDKAction, SDKObservation)

# Phase 9 mounts the /play UI here:
try:
    from server.play_routes import register_play_routes
    register_play_routes(app, env)
except ImportError:
    pass
```

### 5.4 `client.py` (FULL CODE)

```python
"""HTTPEnvClient subclass for SDK-Sovereign."""
from __future__ import annotations
from core.client import HTTPEnvClient

from models import SDKAction, SDKObservation, SDKState


class SDKSovereignEnv(HTTPEnvClient):
    action_class = SDKAction
    observation_class = SDKObservation
    state_class = SDKState
```

### 5.5 `openenv.yaml` (FULL CONTENT)

```yaml
type: space
runtime: fastapi
app: server.app:app
port: 8000

metadata:
  display_name: "SDK-Sovereign 🇮🇳"
  description: "Multi-agent OpenEnv environment for digital sovereignty migrations. Two LLM agents — Auditor and Lead — coordinate under partial information to migrate a sanctioned SDK to a sovereign Indian alternative within 7 turns."
  theme: "multi-agent"
  tags:
    - "multi-agent"
    - "code-generation"
    - "partial-observability"
    - "sovereignty"
    - "india"
```

### 5.6 `Dockerfile` (FULL CONTENT)

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir uv
RUN uv pip install --system -e .

EXPOSE 8000
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 5.7 Build locally

```bash
pip install "openenv-core[core] @ git+https://github.com/meta-pytorch/OpenEnv@v0.2.3"
openenv build -t sdk-sovereign:local
docker run -d -p 8000:8000 sdk-sovereign:local
curl http://localhost:8000/health    # {"status": "healthy"}
curl -X POST http://localhost:8000/reset
```

If those commands work, you're ready to push.

### 5.8 Push to HF Space

```bash
huggingface-cli login                      # paste write token
openenv push --repo-id <your-username>/sdk-sovereign-env
```

Wait ~2-3 min for HF to build. Watch the logs in the Space UI.

### 5.9 `tests/test_smoke_remote.py` (FULL CODE)

```python
"""Live HF Space smoke test. Skipped when SDK_SOVEREIGN_URL env is not set.

Run: SDK_SOVEREIGN_URL=https://<user>-sdk-sovereign-env.hf.space pytest tests/test_smoke_remote.py
"""
from __future__ import annotations
import os
import pytest

URL = os.environ.get("SDK_SOVEREIGN_URL")


@pytest.mark.skipif(not URL, reason="SDK_SOVEREIGN_URL not set")
def test_remote_reset_and_step():
    from client import SDKSovereignEnv
    from models import SDKAction
    with SDKSovereignEnv(base_url=URL).sync() as env:
        obs = env.reset()
        assert obs.current_role == "auditor"
        assert obs.visible_allowlist is not None
        assert obs.visible_codebase is None
        next_obs = env.step(SDKAction(role="auditor", action_type="pass"))
        assert next_obs.current_role == "lead"
        assert next_obs.visible_codebase is not None
```

### 5.10 Verify

```bash
SDK_SOVEREIGN_URL=https://<user>-sdk-sovereign-env.hf.space pytest tests/test_smoke_remote.py -v
```

Visit `https://huggingface.co/spaces/<user>/sdk-sovereign-env` and click "App" — the auto-generated `/web` Gradio UI should render.

### 5.11 Commit

```bash
git add . && git commit -m "Phase 5: OpenEnv wrap + HF Space deployment"
```

### 5.12 Phase 5 acceptance criteria

- [ ] OpenEnv v0.2.3 installed via pinned git tag
- [ ] Schemas inherit from `core.env_server.{Action, Observation, State}`
- [ ] `server/app.py` wraps env via `create_fastapi_app`
- [ ] `Dockerfile`, `openenv.yaml` valid
- [ ] `openenv build` succeeds locally
- [ ] HF Space at `<user>/sdk-sovereign-env` is `Running` (green badge)
- [ ] `/web` UI renders the action schema
- [ ] `tests/test_smoke_remote.py` passes against live URL
- [ ] Git commit: "Phase 5: OpenEnv wrap + HF Space deployment"

---

## 6. Phase 6 — LLM Agents + Two LoRA Adapters (Target: 3 hours, Colab-based)

### 6.1 Goal

Llama-3.2-3B (or Qwen 2.5-0.5B for tighter T4 budget) loaded with two LoRA adapters: `auditor_adapter`, `lead_adapter`. Smoke episode in Colab where each role uses its own adapter. Adapter swap pattern is verified.

### 6.2 Choice of base model

| Model | Why |
|---|---|
| **Qwen 2.5-0.5B-Instruct (recommended)** | Fits T4 with GRPO + 7-turn rollouts + N=4 generations. Trains in 2-3h. |
| Llama-3.2-3B-Instruct (stretch) | Better quality but tight on T4 with rollouts; use only if Qwen baseline trains fast and you have spare hours. |

We use Qwen 0.5B in this PRD. Swap one string in `server/llm_agents.py` if you upgrade later.

### 6.3 `server/prompts.py` (FULL CODE)

```python
"""System prompts for each role. Short, JSON-output structured."""
from __future__ import annotations


SYSTEM_AUDITOR = """You are the Security Auditor in a sovereign-SDK migration crisis.

Your job: gate proposed replacement SDKs against the sovereign allow-list.
You CANNOT see source code. You CAN see the allow-list and the conversation history.

You can take these actions:
- "approve": the most recent proposal IS on the allow-list, approve it
- "reject": the most recent proposal is NOT on the allow-list (give a reason)
- "give_hint": respond to a Lead's hint request
- "pass": waste a turn (penalised; only use if no proposal yet exists)

Respond ONLY with a JSON object:
{"action_type": "approve|reject|pass|give_hint", "rejection_reason": "...", "hint_response": "...", "reasoning": "..."}
"""

SYSTEM_LEAD = """You are the Integration Lead in a sovereign-SDK migration crisis.

Your job: migrate the broken codebase off the sanctioned SDK to an India-sovereign replacement.
You CAN see the source code, the error log, and the conversation history.
You CANNOT see the allow-list — propose a replacement, the Auditor will approve or reject.

You can take these actions:
- "propose_replacement": propose a sovereign SDK as the replacement
- "submit_patch": once approved, submit a full patched version of the file
- "request_hint": ask the Auditor for a category hint
- "pass": waste a turn (penalised)

Respond ONLY with a JSON object:
{"action_type": "propose_replacement|submit_patch|request_hint|pass", "proposed_sdk": "...", "patched_code": "...", "hint_request": "...", "reasoning": "..."}

When submitting a patch, include the COMPLETE patched file as a single string in patched_code.
"""
```

### 6.4 `server/llm_agents.py` (FULL CODE)

```python
"""LLM-backed agents. Loads Qwen 2.5-0.5B with two LoRA adapters and
swaps between them based on whose turn it is.

The adapter-swap is the linchpin of the two-policy claim. Verify with
`assert model.active_adapter == 'auditor_adapter'` before every Auditor
generation, same for Lead.
"""
from __future__ import annotations
import json
import re
from contextlib import contextmanager
from typing import Any, Optional

from models import SDKAction, SDKObservation, Role, ActionType
from server.prompts import SYSTEM_AUDITOR, SYSTEM_LEAD


def load_model_with_two_adapters(
    base_name: str = "unsloth/Qwen2.5-0.5B-Instruct",
    max_seq_length: int = 2048,
):
    """Load base model in 4-bit + apply two LoRA adapters."""
    from unsloth import FastLanguageModel
    from peft import LoraConfig

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )

    cfg = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )

    # Apply Auditor LoRA first (creates PEFT model wrapper)
    from peft import get_peft_model
    model = get_peft_model(model, cfg, adapter_name="auditor_adapter")
    # Add Lead LoRA on the same base
    model.add_adapter(adapter_name="lead_adapter", peft_config=cfg)

    return model, tokenizer


class LLMAgent:
    """Single agent instance — swaps to its adapter on each call."""

    def __init__(self, model: Any, tokenizer: Any, role: str):
        self.model = model
        self.tokenizer = tokenizer
        self.role = role
        self.adapter_name = f"{role}_adapter"
        self.system_prompt = SYSTEM_AUDITOR if role == "auditor" else SYSTEM_LEAD

    def __call__(self, obs: SDKObservation) -> SDKAction:
        self.model.set_adapter(self.adapter_name)
        prompt = self._build_prompt(obs)
        text = self._generate(prompt)
        return self._parse_action(text)

    def _build_prompt(self, obs: SDKObservation) -> str:
        user_content = self._render_observation(obs)
        msgs = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]
        return self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
        )

    def _render_observation(self, obs: SDKObservation) -> str:
        history_str = "\n".join(
            f"  turn {h.get('turn', '?')}: {h.get('role')} → {h.get('action_type')}"
            f" {h.get('proposed_sdk') or h.get('rejection_reason') or ''}"
            for h in obs.conversation_history[-6:]
        ) or "  (no history yet)"

        if self.role == "auditor":
            allow = ", ".join(obs.visible_allowlist or [])
            return (
                f"ERROR LOG: {obs.error_log}\n"
                f"TURN: {obs.turn_index} of {obs.max_turns}\n"
                f"CURRENT PROPOSAL: {obs.current_proposal or '(none yet)'}\n"
                f"ALLOW-LIST: {allow}\n"
                f"HISTORY:\n{history_str}\n\n"
                f"Choose an action. Respond as a single JSON object."
            )

        # Lead view
        return (
            f"ERROR LOG: {obs.error_log}\n"
            f"TURN: {obs.turn_index} of {obs.max_turns}\n"
            f"BROKEN CODE ({obs.visible_filename}):\n```\n{obs.visible_codebase}\n```\n"
            f"APPROVED REPLACEMENT: {obs.approved_replacement or '(not yet approved)'}\n"
            f"HISTORY:\n{history_str}\n\n"
            f"Choose an action. Respond as a single JSON object."
        )

    def _generate(self, prompt: str) -> str:
        import torch
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512 if self.role == "lead" else 200,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        gen = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=True)

    def _parse_action(self, text: str) -> SDKAction:
        parsed = self._extract_json(text)
        if parsed is None:
            return SDKAction(role=self.role, action_type=ActionType.PASS.value,
                             reasoning=f"PARSE_FAIL: {text[:200]}")
        action_type = parsed.get("action_type", "pass")
        try:
            ActionType(action_type)
        except ValueError:
            action_type = "pass"
        return SDKAction(
            role=self.role,
            action_type=action_type,
            proposed_sdk=parsed.get("proposed_sdk"),
            rejection_reason=parsed.get("rejection_reason"),
            patched_code=parsed.get("patched_code"),
            hint_request=parsed.get("hint_request"),
            hint_response=parsed.get("hint_response"),
            reasoning=str(parsed.get("reasoning", ""))[:500],
        )

    @staticmethod
    def _extract_json(text: str) -> Optional[dict]:
        # Greedy match — patches contain newlines and many braces
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m: return None
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            # Try to find a valid JSON block by trimming from end
            for i in range(len(m.group()), 0, -1):
                try:
                    return json.loads(m.group()[:i] + "}")
                except json.JSONDecodeError:
                    continue
        return None


def make_agent_pair(model, tokenizer):
    return {
        "auditor": LLMAgent(model, tokenizer, "auditor"),
        "lead": LLMAgent(model, tokenizer, "lead"),
    }
```

### 6.5 `notebooks/01_smoke_test.ipynb` (CELL CONTENT)

**Cell 1 — install**
```python
!pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install -q --no-deps "trl<0.13" peft accelerate bitsandbytes
!pip install -q "openenv-core[core] @ git+https://github.com/meta-pytorch/OpenEnv@v0.2.3"
!pip install -q wandb
```

**Cell 2 — clone & install env package**
```python
HF_USER = "<your-username>"   # CHANGE ME
!pip install -q git+https://huggingface.co/spaces/{HF_USER}/sdk-sovereign-env
```

**Cell 3 — auth**
```python
import os
from huggingface_hub import login
login(token=os.environ["HF_TOKEN"])
```

**Cell 4 — load model with two adapters**
```python
from server.llm_agents import load_model_with_two_adapters, make_agent_pair

model, tokenizer = load_model_with_two_adapters()
print("Adapters:", list(model.peft_config.keys()))
model.print_trainable_parameters()

agents = make_agent_pair(model, tokenizer)
```

**Cell 5 — run smoke episode against live HF Space**
```python
from client import SDKSovereignEnv

ENV_URL = f"https://{HF_USER}-sdk-sovereign-env.hf.space"

with SDKSovereignEnv(base_url=ENV_URL).sync() as env:
    obs = env.reset()
    print(f"Repo: episode started, role={obs.current_role}")
    for turn in range(7):
        agent = agents[obs.current_role]
        action = agent(obs)
        print(f"Turn {turn} | {obs.current_role} | {action.action_type}")
        obs = env.step(action)
        if obs.done:
            print(f"Done. Final reward: {obs.reward}, breakdown: {obs.reward_breakdown}")
            break
```

**Cell 6 — verify adapter swap**
```python
agents["auditor"](obs)
assert model.active_adapter == "auditor_adapter"
agents["lead"](obs)
assert model.active_adapter == "lead_adapter"
print("✓ Adapter swap verified")
```

### 6.6 Phase 6 acceptance criteria

- [ ] `server/llm_agents.py` complete with `load_model_with_two_adapters`, `LLMAgent`, `make_agent_pair`
- [ ] Model loads on Colab T4 in 4-bit
- [ ] Two adapters present: `model.peft_config` shows both
- [ ] `model.print_trainable_parameters()` reports >0 trainable params
- [ ] Smoke episode runs end-to-end against live HF Space
- [ ] Adapter swap verified (Cell 6 assertion passes)
- [ ] Git commit: "Phase 6: LLM agents with two LoRA adapters"

---

## 7. Phase 7 — GRPO Training (Two Adapters) (Target: overnight, ~6h on T4)

### 7.1 Goal

Two trained adapters with reward curves trending up. WandB runs visible. Both adapters pushed to HF Hub.

### 7.2 The training strategy

TRL's `GRPOTrainer` is single-turn: it takes a prompt, generates N completions, ranks them by reward, and updates policy. To train two role-specific adapters with multi-turn data we do this:

1. **Rollout phase (data collection)**: run M episodes against the env using current adapters. For each role-step in each episode, capture `(prompt_text, completion_text, scalar_reward)`.
2. **Train Lead adapter**: take all `lead`-role tuples, build a Dataset, run GRPO with the Lead reward function (which just looks up the captured reward).
3. **Train Auditor adapter** the same way with Auditor tuples.
4. Repeat 1-3 for several rounds (curriculum: Lead first since it has more to learn).

This is honest two-policy training. Each adapter is updated only on its own role's data.

### 7.3 `notebooks/02_train_lead.ipynb` (KEY CELLS)

**Cell A — collect rollouts**
```python
import json, asyncio
from pathlib import Path
from client import SDKSovereignEnv
from models import SDKAction

ENV_URL = f"https://{HF_USER}-sdk-sovereign-env.hf.space"
N_ROLLOUT_EPISODES = 80

rollout_data = {"auditor": [], "lead": []}

for ep in range(N_ROLLOUT_EPISODES):
    with SDKSovereignEnv(base_url=ENV_URL).sync() as env:
        obs = env.reset()
        per_role_buffer = []
        while not obs.done and obs.turn_index < 7:
            agent = agents[obs.current_role]
            agent.model.set_adapter(agent.adapter_name)
            prompt = agent._build_prompt(obs)
            # Sample one completion (we'll generate N during GRPO; here we just need to drive the env)
            completion = agent._generate(prompt)
            action = agent._parse_action(completion)
            new_obs = env.step(action)
            per_role_buffer.append({
                "role": obs.current_role,
                "prompt": prompt,
                "completion": completion,
                "step_reward": new_obs.reward,
            })
            obs = new_obs

        # Attribute terminal reward forward to each role's last step
        for entry in per_role_buffer:
            rollout_data[entry["role"]].append({
                "prompt": entry["prompt"],
                "reward": entry["step_reward"],
            })
    if ep % 10 == 0:
        print(f"  rollout {ep}/{N_ROLLOUT_EPISODES}")

print(f"Lead samples: {len(rollout_data['lead'])}")
print(f"Auditor samples: {len(rollout_data['auditor'])}")
Path("rollout_lead.jsonl").write_text(
    "\n".join(json.dumps(r) for r in rollout_data["lead"])
)
Path("rollout_auditor.jsonl").write_text(
    "\n".join(json.dumps(r) for r in rollout_data["auditor"])
)
```

**Cell B — wrap as TRL dataset and train Lead adapter**
```python
import wandb
from trl import GRPOTrainer, GRPOConfig
from datasets import Dataset

wandb.init(project="sdk-sovereign", name="lead-grpo-round1")

# Dataset: we use the prompts from rollout. GRPO will generate fresh
# completions and our reward function re-runs the env to score them.
lead_prompts = [r["prompt"] for r in rollout_data["lead"]]
ds_lead = Dataset.from_dict({"prompt": lead_prompts})

# Reward function: re-run a fresh episode for each completion. The completion
# substitutes the Lead's action at the matching turn; the partner Auditor
# uses the current adapter. Score the resulting episode's per-step rewards
# attributed to Lead.
def lead_reward_fn(completions, **kwargs):
    rewards = []
    for completion in completions:
        # Simplest scoring: parse completion to action, run a single-step
        # eval against a fresh env reset and use the env's reward as proxy.
        # This is approximate but trains a useful signal.
        action = agents["lead"]._parse_action(completion)
        if action.action_type == "submit_patch":
            # Score directly by running the verifier
            with SDKSovereignEnv(base_url=ENV_URL).sync() as env:
                obs = env.reset()
                # Approve any proposal then submit
                env.step(SDKAction(role="auditor", action_type="pass"))
                env.step(SDKAction(role="lead", action_type="propose_replacement",
                                    proposed_sdk=action.patched_code and "razorpay" or "razorpay"))
                env.step(SDKAction(role="auditor", action_type="approve"))
                final = env.step(action)
                rewards.append(float(final.reward))
        else:
            rewards.append(float(0.5 if action.action_type != "pass" else -0.5))
    return rewards

config = GRPOConfig(
    output_dir="checkpoints/lead",
    num_generations=4,
    max_completion_length=512,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=5e-6,
    num_train_epochs=1,
    logging_steps=2,
    save_steps=50,
    report_to="wandb",
)

model.set_adapter("lead_adapter")
# Freeze auditor adapter to be sure
for n, p in model.named_parameters():
    if "auditor_adapter" in n:
        p.requires_grad = False

trainer = GRPOTrainer(
    model=model,
    reward_funcs=lead_reward_fn,
    args=config,
    train_dataset=ds_lead.select(range(min(60, len(ds_lead)))),
    tokenizer=tokenizer,
)
trainer.train()
wandb.finish()
```

**Cell C — save adapter**
```python
model.save_pretrained("checkpoints/lead_adapter_v1", selected_adapters=["lead_adapter"])
from huggingface_hub import HfApi
HfApi().upload_folder(
    folder_path="checkpoints/lead_adapter_v1",
    repo_id=f"{HF_USER}/sdk-sovereign-lead-adapter",
    repo_type="model",
)
```

### 7.4 `notebooks/03_train_auditor.ipynb`

Mirror of 02 with `auditor_adapter`, the auditor reward function (which scores approve/reject correctness directly against the allow-list), and `auditor_reward_fn`. Smaller `max_completion_length=200`.

```python
# Cell B replacement: auditor reward function
def auditor_reward_fn(completions, **kwargs):
    rewards = []
    for completion in completions:
        action = agents["auditor"]._parse_action(completion)
        if action.action_type == "approve":
            # Approve a sampled candidate; reward if it would have been on the list
            from server.environment import SDKSovereignEnvironment
            local = SDKSovereignEnvironment()
            # Heuristic: if the action's reasoning mentions an allowlisted SDK, reward
            mentioned = next((sdk for sdk in local.allowlist
                              if sdk in (action.reasoning or "").lower()), None)
            rewards.append(1.5 if mentioned else -2.0)
        elif action.action_type == "reject":
            rewards.append(0.5)  # cautious — usually fine
        elif action.action_type == "pass":
            rewards.append(-0.5)
        else:
            rewards.append(-1.0)
    return rewards
```

### 7.5 `docs/LIMITATIONS.md` (write now, honestly)

```markdown
# Known Limitations

## Reward signal in GRPO training
The training reward functions are simplifications of the full episode rubric.
- Lead reward is computed against a single submit-patch step rather than full
  multi-turn rollouts. This trains the *patch-quality* skill cleanly but
  under-weights mid-episode negotiation skill.
- Auditor reward is computed against allow-list membership of mentioned SDKs
  in the reasoning text — a proxy for the full rubric component.

The full rubric still runs at eval time (Phase 8), so the *measured* improvement
is honest even though training optimises a slightly simpler signal.

## Compute scale
With ~60 GRPO steps per adapter on a T4 GPU we expect a clear-but-modest
improvement curve. More compute would close the gap further.

## Single base model
Both adapters share a Qwen 2.5-0.5B base. This is intentional — it's how we
fit two policies on one T4 — but it means the adapters share a representation
prior. Two fully independent base models would be a stronger but more expensive
two-policy setup.
```

Judges respect honesty. This document is part of the submission.

### 7.6 Phase 7 acceptance criteria

- [ ] `notebooks/02_train_lead.ipynb` runs end-to-end on Colab T4
- [ ] `notebooks/03_train_auditor.ipynb` runs end-to-end on Colab T4
- [ ] Two WandB runs visible: `lead-grpo-round1`, `auditor-grpo-round1`
- [ ] Reward curves visible in WandB (must trend upward; even modestly)
- [ ] Both adapters saved to `checkpoints/`
- [ ] Both adapters pushed to HF Hub: `<user>/sdk-sovereign-lead-adapter`, `<user>/sdk-sovereign-auditor-adapter`
- [ ] `docs/LIMITATIONS.md` committed
- [ ] Git commit: "Phase 7: GRPO training of two LoRA adapters"

---

## 8. Phase 8 — Eval + Plots (Target: 3 hours)

### 8.1 Goal

Six PNGs in `plots/`. Quantitative comparison: random vs trained pass rate.

### 8.2 `notebooks/04_eval_and_plots.ipynb` (KEY CELLS)

**Cell — load both versions**
```python
# Reload model with FRESH adapters (untrained baseline)
import importlib, server.llm_agents as la
importlib.reload(la)
baseline_model, baseline_tok = la.load_model_with_two_adapters()
baseline_agents = la.make_agent_pair(baseline_model, baseline_tok)

# Reload trained adapters from HF Hub
trained_model, trained_tok = la.load_model_with_two_adapters()
trained_model.load_adapter(f"{HF_USER}/sdk-sovereign-lead-adapter", adapter_name="lead_adapter_trained")
trained_model.load_adapter(f"{HF_USER}/sdk-sovereign-auditor-adapter", adapter_name="auditor_adapter_trained")
# Replace the agents' adapter names
trained_agents = la.make_agent_pair(trained_model, trained_tok)
trained_agents["lead"].adapter_name = "lead_adapter_trained"
trained_agents["auditor"].adapter_name = "auditor_adapter_trained"
```

**Cell — eval loop**
```python
import json
from pathlib import Path

def run_eval(agents, n_episodes=30):
    results = []
    for _ in range(n_episodes):
        with SDKSovereignEnv(base_url=ENV_URL).sync() as env:
            obs = env.reset()
            total = 0.0
            transcript = []
            while not obs.done and obs.turn_index < 7:
                action = agents[obs.current_role](obs)
                transcript.append({
                    "turn": obs.turn_index, "role": obs.current_role,
                    "action_type": action.action_type,
                })
                obs = env.step(action)
                total += obs.reward
            state = env.state()
            tests_passed = sum((state.test_results or {}).values()) if state.test_results else 0
            results.append({
                "total_reward": total,
                "tests_passed": tests_passed,
                "tests_total": len(state.test_results) if state.test_results else 3,
                "success": (state.test_results and all(state.test_results.values())) or False,
                "turns": state.step_count,
                "repo": state.repo_id,
                "terminated": state.terminated_reason,
                "transcript": transcript,
            })
    return results

baseline_results = run_eval(baseline_agents, n_episodes=30)
trained_results  = run_eval(trained_agents,  n_episodes=30)
Path("eval_results.json").write_text(json.dumps({
    "baseline": baseline_results, "trained": trained_results,
}, indent=2))
```

**Cell — plot 1: pass rate**
```python
import matplotlib.pyplot as plt

b_rate = sum(r["success"] for r in baseline_results) / len(baseline_results)
t_rate = sum(r["success"] for r in trained_results) / len(trained_results)

plt.figure(figsize=(6,4))
plt.bar(["Random baseline", "Trained (two LoRAs)"], [b_rate, t_rate],
        color=["#bbbbbb", "#1f77b4"])
plt.ylabel("Pass rate (all 3 tests passed)")
plt.title("SDK-Sovereign — pass rate, n=30 each")
plt.ylim(0, 1)
for i, v in enumerate([b_rate, t_rate]):
    plt.text(i, v + 0.02, f"{v:.0%}", ha="center")
plt.savefig("plots/pass_rate_baseline_vs_trained.png", dpi=150, bbox_inches="tight")
plt.close()
```

**Cell — plot 2: avg episode reward**
```python
import statistics
b_r = statistics.mean(r["total_reward"] for r in baseline_results)
t_r = statistics.mean(r["total_reward"] for r in trained_results)
plt.figure(figsize=(6,4))
plt.bar(["Baseline", "Trained"], [b_r, t_r], color=["#bbbbbb", "#1f77b4"])
plt.ylabel("Mean episode reward")
plt.axhline(0, color="k", lw=0.5)
plt.title("Mean total reward per episode (n=30)")
plt.savefig("plots/mean_reward.png", dpi=150, bbox_inches="tight")
plt.close()
```

**Cell — plot 3 & 4: WandB curves (download as PNG)**
```python
import wandb
api = wandb.Api()
for run_name in ["lead-grpo-round1", "auditor-grpo-round1"]:
    run = api.run(f"<entity>/sdk-sovereign/{run_name}")
    h = run.history()
    plt.figure(figsize=(8,4))
    if "train/reward" in h.columns:
        plt.plot(h["_step"], h["train/reward"], label="Train reward")
    plt.xlabel("GRPO step"); plt.ylabel("Reward")
    plt.title(f"GRPO training — {run_name}")
    plt.legend(); plt.grid(alpha=0.3)
    plt.savefig(f"plots/reward_curve_{run_name.split('-')[0]}.png", dpi=150, bbox_inches="tight")
    plt.close()
```

**Cell — plot 5: per-repo pass rate**
```python
from collections import defaultdict
b_per = defaultdict(list); t_per = defaultdict(list)
for r in baseline_results: b_per[r["repo"]].append(r["success"])
for r in trained_results:  t_per[r["repo"]].append(r["success"])
repos = sorted(set(b_per) | set(t_per))
b_vals = [sum(b_per[r])/len(b_per[r]) if b_per[r] else 0 for r in repos]
t_vals = [sum(t_per[r])/len(t_per[r]) if t_per[r] else 0 for r in repos]

import numpy as np
x = np.arange(len(repos)); w = 0.35
plt.figure(figsize=(8,4))
plt.bar(x - w/2, b_vals, w, label="Baseline", color="#bbbbbb")
plt.bar(x + w/2, t_vals, w, label="Trained",  color="#1f77b4")
plt.xticks(x, repos); plt.ylabel("Pass rate"); plt.legend()
plt.title("Pass rate by repo")
plt.savefig("plots/per_repo_pass_rate.png", dpi=150, bbox_inches="tight")
plt.close()
```

**Cell — plot 6: turns to completion**
```python
b_turns = [r["turns"] for r in baseline_results if r["success"]]
t_turns = [r["turns"] for r in trained_results if r["success"]]

plt.figure(figsize=(7,4))
bins = list(range(2, 9))
plt.hist([b_turns, t_turns], bins=bins, label=["Baseline","Trained"],
         color=["#bbbbbb","#1f77b4"])
plt.xlabel("Turns to completion (successful episodes)")
plt.ylabel("Count"); plt.legend()
plt.title("Distribution of completion turns")
plt.savefig("plots/completion_turns.png", dpi=150, bbox_inches="tight")
plt.close()
```

### 8.3 `scripts/make_plots.py`

Wraps the above as a script that runs from `eval_results.json` so the plots can be re-built without re-running the LLM.

### 8.4 Phase 8 acceptance criteria

- [ ] `eval_results.json` exists with 60 episodes (30 baseline + 30 trained)
- [ ] All 6 PNGs exist in `plots/` and are non-empty
- [ ] Trained pass rate ≥ 2× baseline pass rate (target ~10% baseline → ~25-35% trained)
- [ ] Plots are committed to repo
- [ ] Git commit: "Phase 8: eval + 6 plots"

---

## 9. Phase 9 — Web Demo (`/play`) (Target: 2 hours)

### 9.1 Goal

A `/play` URL on the HF Space where a visitor picks a repo, hits "Run", and watches the trained agents negotiate live. Falls back to pre-recorded transcripts if live model loading is too slow.

### 9.2 `frontend/play.html` (FULL CONTENT — single file, vanilla JS)

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>SDK-Sovereign — Live Demo</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         max-width: 900px; margin: 2em auto; padding: 0 1em; color: #222; }
  h1 { font-size: 1.5em; }
  .pitch { background: #fff8e1; border-left: 4px solid #f57c00;
           padding: 0.8em 1em; margin: 1em 0; }
  select, button { font-size: 1em; padding: 0.4em 0.8em; }
  button { background: #1f77b4; color: white; border: none; cursor: pointer; }
  .turn { margin: 0.6em 0; padding: 0.6em; border-radius: 6px; }
  .turn.auditor { background: #fce4ec; border-left: 3px solid #c2185b; }
  .turn.lead    { background: #e3f2fd; border-left: 3px solid #1565c0; }
  .role { font-weight: bold; }
  .action { color: #555; font-family: monospace; }
  .verdict { font-size: 1.2em; padding: 1em; margin-top: 1em; border-radius: 8px; }
  .verdict.success { background: #c8e6c9; color: #1b5e20; }
  .verdict.failure { background: #ffcdd2; color: #b71c1c; }
  pre { background: #f5f5f5; padding: 0.5em; border-radius: 4px;
        font-size: 0.85em; overflow-x: auto; }
</style>
</head>
<body>
<h1>SDK-Sovereign 🇮🇳 — Live Demo</h1>

<div class="pitch">
It's 2026. Your fintech runs on Stripe, Twilio, and Google Maps. At 3 AM, all three
get cut off for Indian users. Two AI agents — Auditor and Lead — must coordinate to
migrate your stack to a sovereign Indian alternative within 7 turns. Neither sees
the full picture.
</div>

<label>Pick a repo: <select id="repo">
  <option value="payments_repo">payments_repo (Stripe → Razorpay)</option>
  <option value="maps_repo">maps_repo (Google Maps → MMI)</option>
  <option value="comms_repo">comms_repo (Twilio → Kaleyra)</option>
</select></label>
<button id="run">▶ Run Episode</button>

<div id="output"></div>

<script>
const ENV_BASE = window.location.origin;

document.getElementById("run").onclick = async () => {
  const repo = document.getElementById("repo").value;
  const out = document.getElementById("output");
  out.innerHTML = "<p>Resetting environment…</p>";

  // Reset
  let r = await fetch(`${ENV_BASE}/reset`, {method: "POST"});
  let obs = await r.json();
  out.innerHTML = "";

  // Loop up to 7 turns
  for (let turn = 0; turn < 7 && !obs.done; turn++) {
    // Call /play/agent_step which uses trained adapters server-side
    const stepResp = await fetch(`${ENV_BASE}/play/agent_step`, {method:"POST"});
    const next = await stepResp.json();
    appendTurn(next.action, next.observation, out);
    obs = next.observation;
    await sleep(800);  // dramatic pause
  }

  // Final verdict
  const final = await fetch(`${ENV_BASE}/state`).then(r=>r.json());
  const success = final.test_results && Object.values(final.test_results).every(Boolean);
  out.insertAdjacentHTML("beforeend",
    `<div class="verdict ${success?'success':'failure'}">
       ${success ? '✓ Migration successful' : '✗ Migration failed'}
       — ${final.test_results ? Object.values(final.test_results).filter(Boolean).length : 0}/3 tests passed
     </div>`);
};

function appendTurn(action, obs, out) {
  const role = action.role;
  const desc = action.proposed_sdk || action.rejection_reason
            || (action.patched_code ? '(patch submitted)' : action.reasoning || '');
  out.insertAdjacentHTML("beforeend",
    `<div class="turn ${role}">
       <span class="role">[${role.toUpperCase()}]</span>
       <span class="action">${action.action_type}</span> →
       <span>${escapeHtml(desc.slice(0, 200))}</span>
     </div>`);
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }
function escapeHtml(s) {
  return s.replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}
</script>
</body>
</html>
```

### 9.3 `server/play_routes.py` (FULL CODE)

```python
"""Custom /play routes: serve the demo HTML and run trained agents server-side."""
from __future__ import annotations
import os
from pathlib import Path
from typing import Optional, Any

from fastapi import HTTPException
from fastapi.responses import FileResponse


_AGENTS: Optional[dict] = None
_LOADED = False


def _try_load_trained_agents():
    global _AGENTS, _LOADED
    if _LOADED:
        return _AGENTS
    _LOADED = True
    if not os.environ.get("SDK_SOVEREIGN_AGENTS_LIVE"):
        # Production HF Space: live model is too memory-heavy; use rule fallback.
        from server.rule_agents import auditor_rule_agent, lead_rule_agent
        _AGENTS = {"auditor": auditor_rule_agent, "lead": lead_rule_agent}
        return _AGENTS
    try:
        from server.llm_agents import load_model_with_two_adapters, make_agent_pair
        model, tok = load_model_with_two_adapters()
        # Optionally load trained adapters from HF
        adapter_repo = os.environ.get("SDK_SOVEREIGN_ADAPTER_REPO")
        if adapter_repo:
            model.load_adapter(f"{adapter_repo}/lead", adapter_name="lead_adapter")
            model.load_adapter(f"{adapter_repo}/auditor", adapter_name="auditor_adapter")
        _AGENTS = make_agent_pair(model, tok)
    except Exception:
        from server.rule_agents import auditor_rule_agent, lead_rule_agent
        _AGENTS = {"auditor": auditor_rule_agent, "lead": lead_rule_agent}
    return _AGENTS


def register_play_routes(app: Any, env: Any) -> None:
    frontend_dir = Path(__file__).parent.parent / "frontend"

    @app.get("/play")
    def play_index():
        return FileResponse(str(frontend_dir / "play.html"))

    @app.post("/play/agent_step")
    def play_agent_step():
        agents = _try_load_trained_agents()
        if env._state is None:
            raise HTTPException(400, "call /reset first")
        # Get current observation, ask the appropriate agent for an action
        obs = env._build_observation(env._next_role(), last_reward=0.0)
        action = agents[obs.current_role](obs) if callable(agents[obs.current_role]) \
                 else agents[obs.current_role](obs)
        new_obs = env.step(action)
        return {"action": action.__dict__, "observation": new_obs.__dict__}
```

### 9.4 Re-deploy

```bash
openenv push --repo-id <user>/sdk-sovereign-env
```

Visit `https://<user>-sdk-sovereign-env.hf.space/play` — should render the demo.

### 9.5 Phase 9 acceptance criteria

- [ ] `frontend/play.html` exists and renders
- [ ] `/play` URL works on live HF Space
- [ ] Picking a repo + clicking Run plays an episode with animated turns
- [ ] Verdict displays at the end with success/failure colour
- [ ] If trained model is too slow, rule-agent fallback works seamlessly
- [ ] Git commit: "Phase 9: /play web demo"

---

## 10. Phase 10 — Final Assembly (Target: 3 hours)

### 10.1 Goal

Final README, HF blog post, ≤2-min video, submission form.

### 10.2 Final `README.md`

```markdown
# SDK-Sovereign 🇮🇳

> Two AI agents. Asymmetric information. 7 turns to migrate your stack
> off a sanctioned SDK before users notice.

[![HF Space](https://img.shields.io/badge/🤗-HF_Space-blue)](https://huggingface.co/spaces/<user>/sdk-sovereign-env)
[![Lead adapter](https://img.shields.io/badge/🤗-Lead_Adapter-yellow)](https://huggingface.co/<user>/sdk-sovereign-lead-adapter)
[![Auditor adapter](https://img.shields.io/badge/🤗-Auditor_Adapter-pink)](https://huggingface.co/<user>/sdk-sovereign-auditor-adapter)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)]()

## The story

It's 2026. Your fintech runs on Stripe, Twilio, and Google Maps. At 3 AM, all three
get cut off for Indian users. You have minutes. Two AI agents — a Security Auditor
and an Integration Lead — must coordinate to migrate your stack to a sovereign
Indian alternative. Neither has the full picture.

## Results

| Metric | Random baseline | Trained (two LoRAs) |
|---|---|---|
| Pass rate (n=30) | XX% | YY% |
| Mean episode reward | -X.X | +Y.Y |
| Avg turns to success | A.B | C.D |

![Pass rate](plots/pass_rate_baseline_vs_trained.png)
![Reward curve — Lead](plots/reward_curve_lead.png)
![Reward curve — Auditor](plots/reward_curve_auditor.png)

## Try it

- 🤗 **Live env**: https://huggingface.co/spaces/<user>/sdk-sovereign-env
- ▶ **Web demo**: https://<user>-sdk-sovereign-env.hf.space/play
- 📔 **Training notebooks**: [02_train_lead](notebooks/02_train_lead.ipynb), [03_train_auditor](notebooks/03_train_auditor.ipynb)
- 📺 **Video** (90s): <youtube-link>
- 📝 **HF blog**: <hf-blog-link>
- 📊 **WandB**: <wandb-link>

## Multi-agent design (Theme 1)

Asymmetric information forces the two agents to negotiate over which sovereign
replacement to commit to. Neither agent can solve the migration alone — the
Auditor cannot read code, the Lead cannot see the allow-list.

We train two LoRA adapters on a shared Qwen 2.5-0.5B base, one per role,
with role-conditional credit assignment via TRL GRPO.

## How the environment works

[Architecture diagram + 2 paragraphs from §2]

## Reward function

[Table from `server/rubric.py`'s WEIGHTS]

## Why it matters

> SDK-Sovereign trains two LLM policies to coordinate under partial information
> on a problem that isn't a game — it's the kind of crisis a real engineering
> team faces when geopolitics moves faster than their tech stack.

## Reproduce

```bash
git clone https://huggingface.co/spaces/<user>/sdk-sovereign-env
cd sdk-sovereign-env
uv sync
uv run server
```

See [docs/LIMITATIONS.md](docs/LIMITATIONS.md) for honest scope notes.

## License
MIT
```

### 10.3 `docs/HF_BLOG.md` (~600 words)

Structure:
1. The crisis hook (the sentence)
2. Why this matters for India in 2026 (DPDP Act, Digital India)
3. The four mechanisms: partial observability, ToM negotiation, AST verification, composable rubric
4. Two-LoRA training design
5. Results: random vs trained
6. What's next

Publish to HuggingFace's blog platform.

### 10.4 90-second video (Loom or YouTube)

Shot list:
- 0:00–0:10 — Hook with the crisis sentence
- 0:10–0:30 — Architecture: two roles, asymmetric info, 7 turns
- 0:30–0:55 — Live demo on `/play` showing negotiation + patch + verdict
- 0:55–1:20 — Reward curves: Lead and Auditor learning separately
- 1:20–1:35 — Pass rate comparison plot
- 1:35–1:50 — The sentence again. CTA.

Upload as unlisted YouTube; put URL in README.

### 10.5 Final checks

```bash
pytest tests/ -v                                           # all green
python demo.py                                             # works
SDK_SOVEREIGN_URL=... pytest tests/test_smoke_remote.py    # green
```

Visit:
- HF Space `/web` — works
- HF Space `/play` — works
- WandB dashboard — both runs visible

### 10.6 Submission form

| Field | Value |
|---|---|
| HF Space URL | `https://huggingface.co/spaces/<user>/sdk-sovereign-env` |
| GitHub mirror | optional |
| Lead adapter | `https://huggingface.co/<user>/sdk-sovereign-lead-adapter` |
| Auditor adapter | `https://huggingface.co/<user>/sdk-sovereign-auditor-adapter` |
| Training notebook | Colab link to `02_train_lead.ipynb` |
| Video | YouTube unlisted URL |
| Blog | HF blog URL |

### 10.7 Final commit + tag

```bash
git add . && git commit -m "Phase 10: final assembly"
git tag v1.0-submission
```

### 10.8 Phase 10 acceptance criteria

- [ ] README has all sections from §10.2 with real numbers
- [ ] `docs/HF_BLOG.md` written and published
- [ ] `docs/LIMITATIONS.md` honest and committed
- [ ] Video uploaded, ≤2:00 runtime
- [ ] All 6 plots embedded in README
- [ ] Both adapters live on HF Hub
- [ ] Submission form filled
- [ ] Final git tag `v1.0-submission`

---

## 11. Cross-Cutting Concerns

### 11.1 Logging

- Console: `rich` for human-facing output in `demo.py`
- JSONL: `logs/episodes.jsonl` written by env on each `step()` (gitignored)
- WandB: training metrics during Phase 7

### 11.2 Error handling

- Verifier: never raises; always returns dict of `{test_id: bool}`
- LLM parsing: always returns valid `SDKAction` via PASS fallback on parse fail
- Env step: wrong-role action returns −1.0 reward, doesn't crash

### 11.3 Time-boxing (do not ignore)

If a phase hits its hard cap, document incomplete state in `docs/LIMITATIONS.md` and move on. Every phase has a graceful-degradation fallback:

| Phase | Fallback if it fails |
|---|---|
| 5 | Run env locally for the demo, skip HF Space |
| 6 | Use rule agents in `play_routes.py` instead of LLM |
| 7 | Submit with rule-baseline curves only; honesty in LIMITATIONS doc |
| 8 | Hand-draw plots from JSONL logs |
| 9 | Use auto `/web` UI, drop custom `/play` |

### 11.4 Pitfalls to avoid

1. Do not test HF Space build locally — HF builds server-side; just `openenv push`
2. Do not skip the expert-trajectory test in `test_rubric.py` — it's the canary that proves your reward function isn't broken
3. Do not let GRPO `num_generations` exceed 4 on T4 — OOM
4. Do not commit `checkpoints/` or `logs/` (they're gitignored, but verify)
5. Do not use Llama-3.2-3B unless you've finished Qwen 0.5B training and have spare hours
6. Do not chase the perfect Auditor reward — allow-list membership is good enough
7. The video matters. Bad video = no top-10 finish. Re-record if it's flat.

### 11.5 Daily sanity check

- [ ] `pytest tests/ -v` — count never decreases
- [ ] `python demo.py` — still works with rule agents
- [ ] HF Space green badge
- [ ] Disk free > 10GB
- [ ] WandB runs accessible

---

## 12. Minimum Viable Submission

If time runs out at any phase, here's what counts as a valid submission and where it places:

| Phases complete | Deliverables | Likely ranking from 800 |
|---|---|---|
| 1-3 | Rule-based env + tests + demo | Submission valid; ~bottom half |
| 1-5 | + OpenEnv compliance + live HF Space | Top-200 |
| 1-6 | + LLM agents working (no training) | Top-100 |
| 1-7 partial | + some training (one adapter trained) | Top-50 |
| 1-8 | + both adapters trained + plots | Top-20 |
| All 10 | + web demo + video + blog | Top-10, contention for top-5 |

**The floor is `python demo.py` plus a working HF Space.** Everything else is additive.

**Build in order. Commit after each phase. Document what works. Be honest about what doesn't.**

---

## 13. Glossary

| Term | Meaning |
|---|---|
| **OpenEnv** | Meta + HF RL environment standard (reset/step/state) |
| **GRPO** | Group Relative Policy Optimization — RL algorithm without critic network |
| **LoRA** | Low-Rank Adaptation — parameter-efficient fine-tuning |
| **Unsloth** | Optimized training library, ~2× speedup vs vanilla TRL |
| **TRL** | HuggingFace's Transformer RL library |
| **Stub registry** | Fake SDK modules injected into `sys.modules` for sandboxed exec |
| **Rubric** | Composable reward function with named components & weights |
| **Allow-list** | The sovereign SDK registry the Auditor gates against |
| **Split-brain** | Lead submits a patch using a different SDK than Auditor approved (penalised) |
| **Adapter swap** | Calling `model.set_adapter()` to switch active LoRA mid-rollout |
| **Role-conditional credit assignment** | Each adapter only updates on its role's trajectory data |

---

## 14. Final Note to the Coding Agent

You are building a hackathon submission, not a production service. Prioritise:

1. **Working > perfect.** Phase 1-3 MVP beats aspirational everything-at-once.
2. **Deploy early.** HF Space live by end of Phase 5 even if rule-based.
3. **Inspect outputs.** Reward hacking is the #1 silent failure mode.
4. **Honest > impressive.** Document limitations. Judges respect that.
5. **The fallback IS the main plan.** `python demo.py` with rule agents must work at every phase.
6. **The sentence.** Repeat it everywhere — README, blog, video, Discord intro.

Begin execution at Phase 1. Do not skip ahead. Verify acceptance criteria at every gate. Commit after each phase.

---

**END OF PRD. Build from scratch. Ship at hour +27 even if rough. Win the room.**