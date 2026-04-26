"""Teacher-trace generation and SFT export helpers."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from models import SDKAction, SDKObservation
from server.environment import SDKSovereignEnvironment
from server.policy_runtime import load_teacher_agents
from server.prompts import SYSTEM_AUDITOR, SYSTEM_LEAD

PolicyMap = dict[str, Callable[[SDKObservation], SDKAction]]


@dataclass
class TraceStep:
    """Serializable single-turn trace record."""

    episode_id: str
    repo_id: str
    role: str
    observation: dict[str, Any]
    action: dict[str, Any]
    reward: float
    done: bool
    success: bool
    policy_mode: str


def serialize_model(value: Any) -> dict[str, Any]:
    """Convert dataclass-style values into plain dictionaries."""
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    raise TypeError(f"Cannot serialize object of type {type(value)!r}")


def run_episode(
    env: SDKSovereignEnvironment,
    policies: PolicyMap,
    *,
    policy_mode: str,
    initial_observation: SDKObservation,
) -> list[TraceStep]:
    """Run one full episode and capture all transitions."""
    observation = initial_observation
    trace: list[TraceStep] = []
    while True:
        acting_observation = observation
        action = policies[acting_observation.current_role](acting_observation)
        observation = env.step(action)
        success = bool(observation.done and env.state.test_results and all(env.state.test_results.values()))
        trace.append(
            TraceStep(
                episode_id=env.state.episode_id,
                repo_id=env.state.repo_id,
                role=action.role,
                observation=serialize_model(acting_observation),
                action=serialize_model(action),
                reward=observation.reward,
                done=observation.done,
                success=success,
                policy_mode=policy_mode,
            )
        )
        if observation.done:
            return trace


def generate_teacher_traces(
    *,
    episodes_per_repo: int = 10,
    repos_root: Path | None = None,
    logs_root: Path | None = None,
    seed: int = 0,
) -> list[TraceStep]:
    """Generate successful teacher traces across all repos."""
    env = SDKSovereignEnvironment(repos_root=repos_root, logs_root=logs_root, seed=seed)
    policies = load_teacher_agents()
    traces: list[TraceStep] = []
    for repo_id in sorted(env.repos.keys()):
        for _ in range(episodes_per_repo):
            observation = env.reset(repo_id=repo_id)
            traces.extend(run_episode(env, policies, policy_mode="teacher", initial_observation=observation))
    return traces


def write_jsonl(records: Iterable[dict[str, Any]], output_path: Path) -> None:
    """Write records as JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def export_traces_jsonl(traces: Iterable[TraceStep], output_path: Path) -> None:
    """Persist trace records for later SFT / RL preprocessing."""
    write_jsonl((trace.__dict__ for trace in traces), output_path)


def build_sft_messages(role: str, observation: dict[str, Any], action: dict[str, Any]) -> list[dict[str, str]]:
    """Convert one trace item into a chat-format SFT sample."""
    system_prompt = SYSTEM_AUDITOR if role == "auditor" else SYSTEM_LEAD
    user_payload = json.dumps(observation, ensure_ascii=True)
    assistant_payload = json.dumps(action, ensure_ascii=True)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_payload},
        {"role": "assistant", "content": assistant_payload},
    ]


def export_sft_jsonl(
    traces: Iterable[TraceStep],
    output_path: Path,
    *,
    success_only: bool = True,
) -> None:
    """Export successful teacher traces as chat-format SFT rows."""
    rows = []
    for trace in traces:
        if success_only and not trace.success:
            continue
        rows.append(
            {
                "messages": build_sft_messages(trace.role, trace.observation, trace.action),
                "role": trace.role,
                "repo_id": trace.repo_id,
                "episode_id": trace.episode_id,
                "reward": trace.reward,
                "success": trace.success,
                "policy_mode": trace.policy_mode,
            }
        )
    write_jsonl(rows, output_path)
