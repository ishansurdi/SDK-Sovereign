"""FastAPI bootstrap for the OpenEnv server."""
from __future__ import annotations

from fastapi import FastAPI

try:
    from openenv.core import create_fastapi_app
except ImportError:
    def create_fastapi_app(env_cls, action_cls, observation_cls):  # type: ignore[no-redef]
        _ = (action_cls, observation_cls)
        app = FastAPI()
        env = env_cls()

        @app.get("/health")
        def health() -> dict[str, str]:
            return {"status": "healthy"}

        @app.post("/reset")
        def reset(**kwargs):
            return env.reset(**kwargs).model_dump()

        @app.post("/step")
        def step(payload: dict):
            return env.step(action_cls.model_validate(payload)).model_dump()

        return app

from models import SDKAction, SDKObservation
from server.environment import SDKSovereignEnvironment

app = create_fastapi_app(SDKSovereignEnvironment, SDKAction, SDKObservation)

# Phase 9 mounts the /play UI here:
try:
    from server.play_routes import register_play_routes
    register_play_routes(app, SDKSovereignEnvironment)
except ImportError:
    pass
