"""Melting Pot adapter — high-level intent wrapper.

Implements EnvAdapter so the runner sees one matrix-style "round" per
`step()` call. Internally the adapter owns two `IntentController`s (one
per player) and steps the underlying MP substrate until the next
*interaction event* (or a timeout, or episode end).

Intent space exposed to the LLM (per player, per round):
  - "Cooperate" : controller heads to cooperate-resources (green)
  - "Defect"    : controller heads to defect-resources (red)
After the controller is READY_TO_SHOOT and has at least one resource,
it pathfinds to the opponent and fires the INTERACT beam, which triggers
the matrix payoff.

A "round" = one consummated interaction (or a timeout if no interaction
happens within `max_steps_per_round`). The reward returned is the reward
delivered between the previous and the current interaction.

Dependencies:
  - dm-meltingpot (lazy-imported in `_build_env`).
  - The installed substrate must expose POSITION + ORIENTATION (the
    `_ENABLE_DEBUG_OBSERVATIONS` flag, plus the corresponding entries
    in `individual_observation_names` / `timestep_spec`). This prototype
    edits the installed copy of `prisoners_dilemma_in_the_matrix__repeated.py`
    to enable that — see README.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from .base import EnvAdapter, ObsView, StepResult
from .mp_controller import (
    INTENT_COLLECT_COOP, INTENT_COLLECT_DEFECT, IntentController,
)
from .mp_world import ASCII_MAP_PD_REPEATED, WorldGrid

logger = logging.getLogger(__name__)


# Intent label → controller intent code.
_INTENT_MAP = {
    "Cooperate": INTENT_COLLECT_COOP,
    "Defect": INTENT_COLLECT_DEFECT,
}
_INTENT_NAMES = list(_INTENT_MAP.keys())  # exposed to the LLM as legal actions


# Per-substrate static descriptions for the LLM preamble.
_SUBSTRATE_DESCRIPTIONS: dict[str, str] = {
    "prisoners_dilemma_in_the_matrix__repeated": (
        "Spatial Prisoner's Dilemma. You and another agent walk a 23×15 grid "
        "collecting two colored resources (green = cooperate, red = defect). "
        "When you 'interact' (zap each other), the dominant color in each "
        "player's inventory selects a row/column of a Prisoner's-Dilemma payoff "
        "table:\n"
        "       opp-C   opp-D\n"
        "  C  ( 3 ,  3) ( 0 ,  5)\n"
        "  D  ( 5 ,  0) ( 1 ,  1)\n"
        "After each interaction both inventories reset and play continues."
    ),
}


# Substrate -> ASCII map for static world parsing.
_SUBSTRATE_ASCII: dict[str, str] = {
    "prisoners_dilemma_in_the_matrix__repeated": ASCII_MAP_PD_REPEATED,
}


class MeltingPotAdapter(EnvAdapter):
    """High-level intent adapter over a Melting Pot matrix-game substrate."""

    def __init__(self, substrate_name: str,
                 max_steps_per_round: int = 250,
                 max_episode_steps: int = 2000,
                 fallback_intent: str = "Cooperate"):
        if substrate_name not in _SUBSTRATE_ASCII:
            raise ValueError(
                f"MP adapter scaffolded only for: {list(_SUBSTRATE_ASCII)}; "
                f"got {substrate_name!r}"
            )
        self.name = f"meltingpot:{substrate_name}"
        self.substrate_name = substrate_name
        self.n_players = 2
        self.world = WorldGrid.from_ascii(_SUBSTRATE_ASCII[substrate_name])
        self.max_steps_per_round = max_steps_per_round
        self.max_episode_steps = max_episode_steps
        self.fallback_intent = fallback_intent

        # Lazy state populated on reset()
        self._env = None
        self._timestep = None
        self._step_idx = 0
        self._round_idx = 0
        self._episode_done = False
        self._controllers: list[IntentController] = []
        # Cumulative reward by player (for prompt observability).
        self._cum_reward = [0.0, 0.0]
        # Last interaction summary per player (set by step()).
        self._last_interaction: list[Optional[dict]] = [None, None]

    # ── lifecycle ───────────────────────────────────────────────────
    def reset(self, seed: int | None = None) -> list[ObsView]:
        if self._env is None:
            self._build_env()
        else:
            # Re-init for a fresh episode.
            pass
        self._timestep = self._env.reset()
        self._step_idx = 0
        self._round_idx = 0
        self._episode_done = False
        self._cum_reward = [0.0, 0.0]
        self._last_interaction = [None, None]
        self._controllers = [
            IntentController(self.world, player_idx=0),
            IntentController(self.world, player_idx=1),
        ]
        return [self._render_obs(pid) for pid in range(self.n_players)]

    def step(self, joint_action: list[int]) -> StepResult:
        """Run the underlying substrate until the next interaction event.

        `joint_action` here is a list of intent indices (0=Cooperate, 1=Defect).
        Translation to primitive actions is done inside per-step.
        """
        if self._env is None:
            raise RuntimeError("call reset() before step()")
        if self._episode_done:
            # Episode is over; return a no-op.
            obs = [self._render_obs(pid) for pid in range(self.n_players)]
            return StepResult(rewards=[0.0, 0.0], done=True, obs=obs,
                              info={"reason": "episode_already_done"})

        intents = [
            _INTENT_NAMES[int(a)] if 0 <= int(a) < len(_INTENT_NAMES)
            else self.fallback_intent
            for a in joint_action
        ]
        # Map UI labels to controller intents.
        ctrl_intents = [_INTENT_MAP[i] for i in intents]

        round_rewards = [0.0, 0.0]
        interaction_observed = False
        steps_in_round = 0
        primitive_log: list[tuple[int, int]] = []
        # Latest INTERACTION_INVENTORIES we saw with non-sentinel values
        # (-1 means "no interaction this step"). Captured per player.
        interaction_snap: list[Optional[list]] = [None, None]

        while steps_in_round < self.max_steps_per_round:
            # Read positions for both players (or None if respawning).
            obs0 = self._timestep.observation[0]
            obs1 = self._timestep.observation[1]
            pos0 = self._safe_pos(obs0)
            pos1 = self._safe_pos(obs1)
            a0 = self._controllers[0].step(ctrl_intents[0], obs0, pos1)
            a1 = self._controllers[1].step(ctrl_intents[1], obs1, pos0)
            primitive_log.append((a0, a1))

            self._timestep = self._env.step([int(a0), int(a1)])
            self._step_idx += 1
            steps_in_round += 1

            r = self._timestep.reward
            if r is not None:
                round_rewards[0] += float(r[0])
                round_rewards[1] += float(r[1])
                self._cum_reward[0] += float(r[0])
                self._cum_reward[1] += float(r[1])

            # Capture INTERACTION_INVENTORIES whenever they leave the
            # sentinel state (-1, -1). These appear on the step the
            # FIRE_ZAP lands, BEFORE the reward is delivered (which can
            # arrive a few steps later).
            for pid in (0, 1):
                ii = self._timestep.observation[pid].get("INTERACTION_INVENTORIES")
                if ii is not None:
                    flat = np.asarray(ii).reshape(-1)
                    if not np.all(flat < 0):
                        interaction_snap[pid] = np.asarray(ii).tolist()

            if self._timestep.last() or self._step_idx >= self.max_episode_steps:
                self._episode_done = True
                break

            # Detect a finished interaction: any non-zero reward on this
            # MP step (the matrix payoff is delivered as the per-step reward).
            if r is not None and (abs(float(r[0])) > 1e-6 or abs(float(r[1])) > 1e-6):
                interaction_observed = True
                self._last_interaction[0] = self._build_interaction_record(
                    0, float(r[0]), interaction_snap[0]
                )
                self._last_interaction[1] = self._build_interaction_record(
                    1, float(r[1]), interaction_snap[1]
                )
                break

        self._round_idx += 1
        info = {
            "round": self._round_idx,
            "intents": intents,
            "primitive_steps": steps_in_round,
            "interaction": interaction_observed,
            "ep_step": self._step_idx,
            "ep_done": self._episode_done,
        }
        obs = [self._render_obs(pid) for pid in range(self.n_players)]
        return StepResult(
            rewards=round_rewards,
            done=self._episode_done,
            obs=obs,
            info=info,
        )

    # ── action space ────────────────────────────────────────────────
    def legal_action_names(self, player_id: int) -> list[str]:
        return list(_INTENT_NAMES)

    # ── prompt rendering ────────────────────────────────────────────
    def system_preamble(self, player_id: int) -> str:
        desc = _SUBSTRATE_DESCRIPTIONS.get(
            self.substrate_name,
            f"You are player {player_id} in Melting Pot substrate "
            f"'{self.substrate_name}'."
        )
        actions = ", ".join(self.legal_action_names(player_id))
        return (
            f"{desc}\n\n"
            f"You are player {player_id}. Each round you commit to one of "
            f"two stances: {actions}. Choosing 'Cooperate' tells your "
            f"low-level controller to collect green resources; 'Defect' "
            f"tells it to collect red. After collecting at least one "
            f"resource it will hunt the opponent and fire the interaction "
            f"beam. Whichever color dominates each player's inventory at "
            f"the moment of interaction selects their row/column of the "
            f"payoff matrix. After each interaction both inventories reset."
        )

    def is_cooperative_action(self, player_id: int, action_idx: int) -> bool:
        # Action 0 = "Cooperate" stance.
        return action_idx == 0

    # ── private helpers ─────────────────────────────────────────────
    def _build_env(self) -> None:
        try:
            from meltingpot import substrate  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "Melting Pot is not installed in the current Python env. "
                "Use the position-mp conda env: "
                "`conda activate position-mp` (Python 3.11). Underlying "
                f"error: {e}"
            ) from e
        roles = ("default", "default")
        self._env = substrate.build(self.substrate_name, roles=roles)

    @staticmethod
    def _safe_pos(obs) -> Optional[tuple[int, int]]:
        p = obs.get("POSITION")
        if p is None:
            return None
        try:
            return (int(p[0]), int(p[1]))
        except Exception:
            return None

    def _build_interaction_record(self, pid: int, reward: float,
                                   snap: Optional[list]) -> dict:
        if snap is None:
            return {
                "reward": reward,
                "self_at_interaction": None,
                "opp_at_interaction": None,
                "ep_step": self._step_idx,
            }
        # snap is shape (2, num_resources): [self_inv, opp_inv]
        return {
            "reward": reward,
            "self_at_interaction": list(snap[0]),
            "opp_at_interaction": list(snap[1]),
            "ep_step": self._step_idx,
        }

    def _render_obs(self, player_id: int) -> ObsView:
        obs = self._timestep.observation[player_id]
        inv = obs.get("INVENTORY", np.zeros(2))
        ready = float(obs.get("READY_TO_SHOOT", 0.0))
        last = self._last_interaction[player_id]
        last_str = (
            f"last interaction: you held [coop={last['self_at_interaction'][0]:.0f}, "
            f"defect={last['self_at_interaction'][1]:.0f}] vs. opp "
            f"[coop={last['opp_at_interaction'][0]:.0f}, "
            f"defect={last['opp_at_interaction'][1]:.0f}]; "
            f"reward delivered: {last['reward']:.1f}"
        ) if last and last["self_at_interaction"] is not None else "(no prior interaction this episode)"
        text = (
            f"[round {self._round_idx}, ep_step {self._step_idx}/{self.max_episode_steps}]\n"
            f"INVENTORY: cooperate={inv[0]:.0f}, defect={inv[1]:.0f}\n"
            f"READY_TO_SHOOT: {'yes' if ready > 0.5 else 'no'}\n"
            f"cumulative reward: {self._cum_reward[player_id]:.1f}\n"
            f"{last_str}"
        )
        return ObsView(
            player_id=player_id,
            text=text,
            legal_actions=self.legal_action_names(player_id),
            extras={
                "inventory": list(inv),
                "ready_to_shoot": ready,
                "cum_reward": self._cum_reward[player_id],
                "last_interaction": last,
            },
        )
