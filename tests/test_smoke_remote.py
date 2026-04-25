"""Live HF Space smoke test. Skipped when SDK_SOVEREIGN_URL env is not set.

Run: SDK_SOVEREIGN_URL=https://<user>-sdk-sovereign-env.hf.space pytest tests/test_smoke_remote.py
"""
from __future__ import annotations
import os
import pytest

URL = os.environ.get("SDK_SOVEREIGN_URL")


@pytest.mark.skipif(not URL, reason="SDK_SOVEREIGN_URL not set")
def test_remote_reset_and_step():
    """Verify /reset and /step work against the live HF Space."""
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
