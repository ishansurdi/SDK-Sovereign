"""Typed client for the SDK-Sovereign OpenEnv server."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict

try:
    from openenv.core.env_client import EnvClient as _AsyncEnvClient
    from openenv.core.sync_client import SyncEnvClient as _SyncEnvClient
    from openenv.core.client_types import StepResult
except ImportError:
    class _AsyncEnvClient:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("openenv is required to use SDKSovereignEnv")

    class _SyncEnvClient:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("openenv is required to use SDKSovereignEnv")

    class StepResult:  # type: ignore[no-redef]
        def __init__(self, observation: Any, reward: float | None = None, done: bool = False):
            self.observation = observation
            self.reward = reward
            self.done = done

from models import SDKAction, SDKObservation, SDKState


def _to_payload(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "__dict__"):
        return vars(value)
    return dict(value)


class _SDKSovereignSyncEnv(_SyncEnvClient[SDKAction, SDKObservation, SDKState]):
    """Compatibility wrapper that returns observations directly."""

    def reset(self, **kwargs: Any) -> SDKObservation:
        return super().reset(**kwargs).observation

    def step(self, action: SDKAction, **kwargs: Any) -> SDKObservation:
        return super().step(action, **kwargs).observation


class SDKSovereignEnv(_AsyncEnvClient[SDKAction, SDKObservation, SDKState]):
    """Typed OpenEnv client for SDK-Sovereign."""

    action_class = SDKAction
    observation_class = SDKObservation
    state_class = SDKState

    def _step_payload(self, action: SDKAction) -> Dict[str, Any]:
        return _to_payload(action)

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[SDKObservation]:
        observation_payload = dict(payload.get("observation", {}))
        if "reward" in payload and "reward" not in observation_payload:
            observation_payload["reward"] = payload["reward"]
        if "done" in payload and "done" not in observation_payload:
            observation_payload["done"] = payload["done"]
        if "reward_breakdown" in payload and "reward_breakdown" not in observation_payload:
            observation_payload["reward_breakdown"] = payload["reward_breakdown"]
        observation = SDKObservation(**observation_payload)
        return StepResult(
            observation=observation,
            reward=payload.get("reward", observation.reward),
            done=payload.get("done", observation.done),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> SDKState:
        return SDKState(**payload)

    def sync(self) -> _SDKSovereignSyncEnv:
        return _SDKSovereignSyncEnv(self)
