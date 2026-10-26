"""OpenSpiel matrix-game environment adapter."""

from .base import EnvAdapter, ObsView, StepResult, TransitionLog
from .matrix_game import MatrixGameAdapter

__all__ = [
    "EnvAdapter",
    "ObsView",
    "StepResult",
    "TransitionLog",
    "MatrixGameAdapter",
    "make_env",
]


def make_env(spec: str | dict, **overrides) -> EnvAdapter:
    """Construct an OpenSpiel matrix-game adapter."""
    if isinstance(spec, dict):
        if spec.get("kind", "matrix") != "matrix":
            raise ValueError(f"unknown env kind: {spec['kind']!r}")
        spec = spec["game_name"]
    if not isinstance(spec, str):
        raise TypeError(f"env spec must be str or dict, got {type(spec).__name__}")
    if not spec.startswith("matrix_"):
        raise ValueError(f"unknown env spec: {spec!r}")
    return MatrixGameAdapter(game_name=spec, **overrides)
