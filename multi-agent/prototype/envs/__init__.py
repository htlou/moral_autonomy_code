"""Environment adapters. All environments expose the same EnvAdapter API
so the runner is independent of OpenSpiel vs. Melting Pot vs. anything else."""

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
    """Factory: dispatch on a string spec like 'matrix_pd' or a dict.

    Examples:
      make_env("matrix_pd")
      make_env("matrix_sh")
      make_env({"kind": "melting_pot", "substrate": "prisoners_dilemma_in_the_matrix__repeated"})
    """
    if isinstance(spec, str):
        if spec.startswith("matrix_"):
            return MatrixGameAdapter(game_name=spec, **overrides)
        if spec.startswith("mp:") or spec.startswith("meltingpot:"):
            from .melting_pot import MeltingPotAdapter
            substrate = spec.split(":", 1)[1]
            return MeltingPotAdapter(substrate_name=substrate, **overrides)
        raise ValueError(f"unknown env spec: {spec!r}")
    if isinstance(spec, dict):
        kind = spec.get("kind", "matrix")
        if kind == "matrix":
            return MatrixGameAdapter(game_name=spec["game_name"], **overrides)
        if kind in ("melting_pot", "mp"):
            from .melting_pot import MeltingPotAdapter
            return MeltingPotAdapter(substrate_name=spec["substrate"], **overrides)
        raise ValueError(f"unknown env kind: {kind!r}")
    raise TypeError(f"env spec must be str or dict, got {type(spec).__name__}")
