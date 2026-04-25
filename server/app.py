"""FastAPI bootstrap for the OpenEnv server."""
from __future__ import annotations
from openenv.core import create_fastapi_app

from models import SDKAction, SDKObservation
from server.environment import SDKSovereignEnvironment

app = create_fastapi_app(SDKSovereignEnvironment, SDKAction, SDKObservation)

# Phase 9 mounts the /play UI here:
try:
    from server.play_routes import register_play_routes
    register_play_routes(app, SDKSovereignEnvironment)
except ImportError:
    pass
