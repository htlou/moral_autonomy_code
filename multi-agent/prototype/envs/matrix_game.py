"""OpenSpiel matrix-game adapter.

Implements the EnvAdapter contract over `pyspiel.load_game("matrix_pd")`,
`matrix_sh`, etc. Iteration is handled by the runner — each `reset/step`
cycle is one one-shot matrix game; the runner repeats them N times.
"""

from __future__ import annotations

from typing import List, Tuple

import pyspiel

from .base import EnvAdapter, ObsView, StepResult


class MatrixGameAdapter(EnvAdapter):
    """OpenSpiel 2-player simultaneous-move matrix games."""

    def __init__(self, game_name: str = "matrix_pd"):
        self.name = game_name
        self.game = pyspiel.load_game(game_name)
        if self.game.num_players() != 2:
            raise ValueError(f"{game_name} is not 2-player")
        self.n_players = 2
        self.num_actions = self.game.num_distinct_actions()

        s0 = self.game.new_initial_state()
        self._action_names: List[str] = [
            s0.action_to_string(0, a) for a in range(self.num_actions)
        ]

        # Cache the full payoff matrix.
        # payoffs[a0][a1] = (u0, u1)
        self.payoffs: List[List[Tuple[float, float]]] = []
        for a0 in range(self.num_actions):
            row = []
            for a1 in range(self.num_actions):
                s = self.game.new_initial_state()
                s.apply_actions([a0, a1])
                u = s.returns()
                row.append((float(u[0]), float(u[1])))
            self.payoffs.append(row)

        self._done = False
        self._step_idx = 0

    # ── lifecycle ───────────────────────────────────────────────────
    def reset(self, seed: int | None = None) -> list[ObsView]:
        self._done = False
        self._step_idx = 0
        return [self._initial_obs(pid) for pid in range(self.n_players)]

    def step(self, joint_action: list[int]) -> StepResult:
        if len(joint_action) != self.n_players:
            raise ValueError(f"need {self.n_players} actions, got {len(joint_action)}")
        a0, a1 = int(joint_action[0]), int(joint_action[1])
        u0, u1 = self.payoffs[a0][a1]
        # Matrix games are one-shot: episode ends after a single joint action.
        self._done = True
        self._step_idx += 1

        obs = [
            self._post_step_obs(pid, joint_action, [u0, u1])
            for pid in range(self.n_players)
        ]
        info = {
            "joint_action": joint_action,
            "joint_action_names": [
                self._action_names[a0], self._action_names[a1]
            ],
            "payoffs": [u0, u1],
        }
        return StepResult(rewards=[u0, u1], done=True, obs=obs, info=info)

    # ── action space ────────────────────────────────────────────────
    def legal_action_names(self, player_id: int) -> list[str]:
        return list(self._action_names)

    # ── prompt rendering ────────────────────────────────────────────
    def system_preamble(self, player_id: int) -> str:
        return (
            f"Game: {self.name}. You are player {player_id} in a 2-player "
            f"simultaneous-move matrix game.\n\n"
            f"{self.payoff_table_text(player_id)}"
        )

    def payoff_table_text(self, player_id: int = 0) -> str:
        """Render the payoff matrix from the perspective of `player_id`.
        For player 0 the rows index *your* action; for player 1 we transpose
        so rows still index *your* action.
        """
        names = self._action_names
        if player_id == 0:
            header = "      " + "  ".join(f"{n[:8]:>10}" for n in names)
            lines = ["Payoffs as (you, opp):", header]
            for a_self, n_self in enumerate(names):
                cells = []
                for a_opp in range(self.num_actions):
                    u_self, u_opp = self.payoffs[a_self][a_opp]
                    cells.append(f"({u_self:>4.1f},{u_opp:>4.1f})")
                lines.append(f"{n_self[:6]:<6}" + "  ".join(f"{c:>10}" for c in cells))
        else:
            header = "      " + "  ".join(f"{n[:8]:>10}" for n in names)
            lines = ["Payoffs as (you, opp):", header]
            for a_self, n_self in enumerate(names):
                cells = []
                for a_opp in range(self.num_actions):
                    # Player 1's POV: opp is player 0.
                    u_opp, u_self = self.payoffs[a_opp][a_self]
                    cells.append(f"({u_self:>4.1f},{u_opp:>4.1f})")
                lines.append(f"{n_self[:6]:<6}" + "  ".join(f"{c:>10}" for c in cells))
        return "\n".join(lines)

    def is_cooperative_action(self, player_id: int, action_idx: int) -> bool:
        """For matrix_pd / matrix_sh action 0 ('Cooperate'/'Stag') is the
        prosocial choice. Override if you load a non-PD matrix game."""
        coop_keywords = ("cooperate", "stag", "coop")
        if 0 <= action_idx < self.num_actions:
            return any(k in self._action_names[action_idx].lower() for k in coop_keywords)
        return False

    # ── private helpers ─────────────────────────────────────────────
    def _initial_obs(self, player_id: int) -> ObsView:
        return ObsView(
            player_id=player_id,
            text=("(start of episode — no moves yet; choose your opening action)"),
            legal_actions=self.legal_action_names(player_id),
        )

    def _post_step_obs(self, player_id: int, joint_action: list[int],
                       payoffs: list[float]) -> ObsView:
        a_self, a_opp = joint_action[player_id], joint_action[1 - player_id]
        return ObsView(
            player_id=player_id,
            text=(
                f"You played {self._action_names[a_self]}; "
                f"opponent played {self._action_names[a_opp]}. "
                f"Payoff: you={payoffs[player_id]:g}, opp={payoffs[1-player_id]:g}."
            ),
            legal_actions=self.legal_action_names(player_id),
            extras={
                "self_action": a_self,
                "opp_action": a_opp,
                "self_payoff": payoffs[player_id],
                "opp_payoff": payoffs[1 - player_id],
            },
        )
