"""Policy loading helpers for teacher and model-backed agents."""
from __future__ import annotations

import importlib.util
import os
from typing import Any, Callable, Optional

PolicyMap = dict[str, Callable[[Any], Any]]


def model_runtime_status() -> dict[str, Any]:
    """Report whether the current runtime can actually serve model-backed modes."""
    issues: list[str] = []
    if importlib.util.find_spec("unsloth") is None:
        issues.append("missing dependency: unsloth")
    if importlib.util.find_spec("torch") is None:
        issues.append("missing dependency: torch")
    else:
        import torch

        if not torch.cuda.is_available():
            issues.append("CUDA GPU not available")
    return {
        "ready": not issues,
        "issues": issues,
    }


def resolve_adapter_repos() -> tuple[Optional[str], Optional[str]]:
    """Resolve lead and auditor adapter repositories from environment variables."""
    shared_repo = os.environ.get("SDK_SOVEREIGN_ADAPTER_REPO")
    if shared_repo:
        return f"{shared_repo}/lead", f"{shared_repo}/auditor"
    return os.environ.get("SDK_SOVEREIGN_LEAD_ADAPTER_REPO"), os.environ.get("SDK_SOVEREIGN_AUDITOR_ADAPTER_REPO")


def configured_live_modes() -> list[dict[str, str]]:
    """Return the policy modes the current deployment can actually serve."""
    modes = [
        {"id": "rule", "label": "Rule fallback", "description": "Deterministic fallback for demo reliability."},
        {"id": "teacher", "label": "Teacher policy", "description": "Deterministic successful teacher policy for trace generation."},
    ]
    runtime_status = model_runtime_status()
    if os.environ.get("SDK_SOVEREIGN_AGENTS_LIVE") and runtime_status["ready"]:
        modes.append(
            {"id": "baseline", "label": "Baseline model", "description": "Base model with fresh adapters, decoded deterministically."}
        )
        lead_repo, auditor_repo = resolve_adapter_repos()
        if lead_repo and auditor_repo:
            modes.append(
                {"id": "trained", "label": "Trained adapters", "description": "Loads adapter checkpoints from Hugging Face Hub."}
            )
    return modes


def mode_diagnostics() -> dict[str, Any]:
    """Explain which policy modes are available and what is still missing."""
    live_enabled = bool(os.environ.get("SDK_SOVEREIGN_AGENTS_LIVE"))
    lead_repo, auditor_repo = resolve_adapter_repos()
    runtime_status = model_runtime_status()
    diagnostics = {
        "live_enabled": live_enabled,
        "available_mode_ids": [item["id"] for item in configured_live_modes()],
        "model_runtime_ready": runtime_status["ready"],
        "model_runtime_issues": runtime_status["issues"],
        "notes": [],
    }
    if not live_enabled:
        diagnostics["notes"].append("Set SDK_SOVEREIGN_AGENTS_LIVE=1 to enable model-generated baseline and trained modes.")
    if live_enabled and not runtime_status["ready"]:
        diagnostics["notes"].append(
            "Model-backed modes are hidden because this runtime cannot load them yet: " + "; ".join(runtime_status["issues"])
        )
    if live_enabled and runtime_status["ready"] and not (lead_repo and auditor_repo):
        diagnostics["notes"].append(
            "Configure SDK_SOVEREIGN_ADAPTER_REPO or both SDK_SOVEREIGN_LEAD_ADAPTER_REPO and SDK_SOVEREIGN_AUDITOR_ADAPTER_REPO to expose trained mode."
        )
    diagnostics["notes"].append("Teacher mode is the recommended source for successful trace generation and SFT exports.")
    diagnostics["notes"].append("Model-backed modes now decode deterministically so pass-rate comparisons are stable.")
    return diagnostics


def load_rule_agents() -> PolicyMap:
    """Return deterministic fallback agents."""
    from server.rule_agents import auditor_rule_agent, lead_rule_agent

    return {"auditor": auditor_rule_agent, "lead": lead_rule_agent}


def load_teacher_agents() -> PolicyMap:
    """Return the current teacher policy used for dataset generation."""
    return load_rule_agents()


def load_model_agents(mode: str, *, deterministic: bool = True) -> PolicyMap:
    """Load baseline or trained model-driven agents."""
    runtime_status = model_runtime_status()
    if not runtime_status["ready"]:
        raise RuntimeError("model runtime unavailable: " + "; ".join(runtime_status["issues"]))

    from server.llm_agents import load_model_with_two_adapters, make_agent_pair

    model, tokenizer = load_model_with_two_adapters()
    agents = make_agent_pair(model, tokenizer, deterministic=deterministic)
    if mode == "trained":
        lead_repo, auditor_repo = resolve_adapter_repos()
        if not lead_repo or not auditor_repo:
            raise RuntimeError("trained mode requested but adapter repositories are not configured")
        model.load_adapter(lead_repo, adapter_name="lead_adapter_trained")
        model.load_adapter(auditor_repo, adapter_name="auditor_adapter_trained")
        agents["lead"].adapter_name = "lead_adapter_trained"
        agents["auditor"].adapter_name = "auditor_adapter_trained"
    return agents
