"""EnvAdapter: a small abstraction so the same runner drives matrix games
(OpenSpiel) today and Melting Pot substrates later.

Design constraints:
  - Observations must be renderable as TEXT for the LLM agent.
  - Action space is a discrete list of NAMED actions (strings).
  - Joint-action step semantics: all players supply an action each turn.
    Single-agent envs are a degenerate case (n_players=1).
  - Stateful: the adapter owns all environment state. Agents are stateless
    w.r.t. the env between steps; their memory lives in their constitution
    store and the runner-supplied history.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


# ── Data classes ────────────────────────────────────────────────────────────

@dataclass
class ObsView:
    """A single player's observation rendered for prompting.

    `text` is the canonical LLM-readable form. `extras` carries anything
    structured (positions, inventory, raw tensors) for logging/analysis.
    """
    player_id: int
    text: str
    legal_actions: list[str]
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class StepResult:
    """Outcome of stepping the env with a joint action."""
    rewards: list[float]              # rewards[i] for player i
    done: bool
    obs: list[ObsView]                # next observation per player
    info: dict[str, Any] = field(default_factory=dict)


@dataclass
class TransitionLog:
    """One env step, recorded in env-agnostic form for offline analysis."""
    step_idx: int
    actions: list[int]                 # int indices the env actually consumed
    action_names: list[str]            # names corresponding to those indices
    rewards: list[float]
    info: dict[str, Any] = field(default_factory=dict)


# ── Adapter contract ────────────────────────────────────────────────────────

class EnvAdapter(ABC):
    """Joint-action multi-agent env. All players step in lockstep.

    Concrete adapters: MatrixGameAdapter (OpenSpiel), MeltingPotAdapter,
    and any future substrate.
    """

    name: str
    n_players: int

    # ── lifecycle ───────────────────────────────────────────────────
    @abstractmethod
    def reset(self, seed: int | None = None) -> list[ObsView]:
        """Begin a new episode; return per-player observations."""

    @abstractmethod
    def step(self, joint_action: list[int]) -> StepResult:
        """Apply per-player actions; return rewards + next observation."""

    # ── action space ────────────────────────────────────────────────
    @abstractmethod
    def legal_action_names(self, player_id: int) -> list[str]:
        """Names the LLM is allowed to output. Position in list = action index."""

    def action_index(self, player_id: int, name: str) -> int:
        """Map a (loose) name to the index. Override for custom matching."""
        legals = self.legal_action_names(player_id)
        norm = name.strip().lower()
        for i, n in enumerate(legals):
            if n.lower() == norm:
                return i
        # prefix-tolerant fallback
        for i, n in enumerate(legals):
            if n.lower().startswith(norm) or norm.startswith(n.lower()):
                return i
        raise ValueError(
            f"unknown action {name!r} for player {player_id}; legal={legals}"
        )

    # ── prompt helpers ──────────────────────────────────────────────
    @abstractmethod
    def system_preamble(self, player_id: int) -> str:
        """Static text describing the env to the player (rules, payoffs)."""

    # Default: action names as comma-separated list. Override for nicer prose.
    def actions_help(self, player_id: int) -> str:
        return ", ".join(self.legal_action_names(player_id))

    # ── analysis hooks (optional) ───────────────────────────────────
    def is_cooperative_action(self, player_id: int, action_idx: int) -> bool:
        """Best-effort: which discrete action counts as 'cooperative' for
        downstream cooperation-rate metrics. Default: action 0.
        Subclasses should override when this heuristic fails."""
        return action_idx == 0
