"""HTTPEnvClient subclass for SDK-Sovereign."""
from __future__ import annotations
try:
    from openenv.core import SyncEnvClient as _BaseClient
except ImportError:
    class _BaseClient:  # type: ignore[no-redef]
        ...

from models import SDKAction, SDKObservation, SDKState


class SDKSovereignEnv(_BaseClient):
    """Typed HTTP client for the SDK-Sovereign OpenEnv server."""

    action_class = SDKAction
    observation_class = SDKObservation
    state_class = SDKState
