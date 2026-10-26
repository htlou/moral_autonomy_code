"""Low-level intent controller for *_in_the_matrix substrates.

One controller drives one player. It exposes a high-level *intent* API:

  intent ∈ { "collect_cooperative", "collect_defective" }

While an intent is active the controller emits one primitive action per
underlying MP step:
  - while not READY_TO_SHOOT (i.e. inventory is empty/just reset):
      pathfind to the nearest available resource of the preferred color
      and walk to it (collect-by-walking-onto-it).
  - once READY_TO_SHOOT and we have ≥1 resource of either kind:
      pathfind toward the opponent and FIRE_ZAP when we are adjacent and
      facing them.

The adapter calls `step(timestep, opponent_pos)` each underlying MP step
to get the next primitive action.

Limitations (intentional, MVP):
  - Doesn't track which tiles are currently in 'wait' state (regenerating).
    If we walk onto a tile that hasn't regrown, we just keep going to the
    next-nearest of that color.
  - Treats 'a' choice tiles as unknown — only navigates to the
    deterministic '1'/'2' tiles. There are plenty of those.
  - Doesn't handle FIRE_ZAP missing (e.g., wall in between). Retries.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from .mp_world import (
    BACKWARD, DIR_VEC, FORWARD, INTERACT, NOOP, STEP_LEFT,
    STEP_RIGHT, TURN_LEFT, TURN_RIGHT, WorldGrid, add, turn_action, vec_to_ori,
)

logger = logging.getLogger(__name__)


# Intent codes used in `step(intent, ...)`.
INTENT_COLLECT_COOP = "collect_cooperative"
INTENT_COLLECT_DEFECT = "collect_defective"


class IntentController:
    """Drives one player toward their current intent.

    A new intent should be set after each interaction event (handled by
    the adapter, not here).
    """

    def __init__(self, world: WorldGrid, player_idx: int,
                 zap_max_range: int = 3, recent_blocked_decay: int = 30):
        self.world = world
        self.player_idx = player_idx
        # Cooldown counter on resource tiles we walked onto but didn't pick up
        # (tile is in wait state). Keys: (col,row); values: ticks remaining.
        self._blocked: dict[tuple[int, int], int] = {}
        self._blocked_decay = recent_blocked_decay
        self.zap_max_range = zap_max_range
        # Track last position so we can detect "didn't move on FORWARD" =
        # blocked / wall in front.
        self._last_pos: Optional[tuple[int, int]] = None

    # ── helpers ─────────────────────────────────────────────────────
    @staticmethod
    def _pos(obs: dict) -> Optional[tuple[int, int]]:
        p = obs.get("POSITION")
        if p is None:
            return None
        try:
            return (int(p[0]), int(p[1]))
        except Exception:
            return None

    @staticmethod
    def _ori(obs: dict) -> int:
        o = obs.get("ORIENTATION")
        if o is None:
            return 0
        try:
            return int(o)
        except Exception:
            return 0

    @staticmethod
    def _ready(obs: dict) -> bool:
        return bool(float(obs.get("READY_TO_SHOOT", 0.0)) > 0.5)

    @staticmethod
    def _inventory(obs: dict) -> tuple[float, float]:
        inv = obs.get("INVENTORY", np.zeros(2))
        try:
            return float(inv[0]), float(inv[1])
        except Exception:
            return 0.0, 0.0

    def _tick_blocked(self) -> None:
        for k in list(self._blocked.keys()):
            self._blocked[k] -= 1
            if self._blocked[k] <= 0:
                self._blocked.pop(k)

    # ── main step ───────────────────────────────────────────────────
    def step(self, intent: str, obs: dict, opponent_pos: Optional[tuple[int, int]]) -> int:
        """Return the next primitive action index for this player."""
        self._tick_blocked()

        pos = self._pos(obs)
        ori = self._ori(obs)
        if pos is None:
            return NOOP  # respawning / frozen

        # Detect "didn't move on FORWARD" → tile in front blocked. Mark and
        # try a different tile next time.
        if self._last_pos is not None and pos == self._last_pos:
            front = add(pos, DIR_VEC[ori])
            self._blocked[front] = self._blocked_decay
        self._last_pos = pos

        ready = self._ready(obs)
        inv = self._inventory(obs)
        have_any = (inv[0] + inv[1]) >= 0.5

        # --- Phase 1: collect a resource of the preferred color ---
        # Always try to refill the preferred color while not READY_TO_SHOOT.
        # Once READY and have_any, switch to seeking the opponent.
        if not ready or not have_any:
            return self._step_collect(pos, ori, intent)

        # --- Phase 2: hunt opponent ---
        if opponent_pos is not None:
            return self._step_hunt(pos, ori, opponent_pos)

        # No info on opponent — wander forward.
        return self._safe_forward(pos, ori)

    # ── collect-resource phase ──────────────────────────────────────
    def _step_collect(self, pos: tuple[int, int], ori: int, intent: str) -> int:
        if intent == INTENT_COLLECT_COOP:
            candidates = self.world.coop_tiles
            fallback = self.world.defect_tiles
        else:
            candidates = self.world.defect_tiles
            fallback = self.world.coop_tiles
        # Try preferred color first; if everything is blocked, try the other.
        action = self._step_to_any(pos, ori, candidates)
        if action is not None:
            return action
        action = self._step_to_any(pos, ori, fallback)
        if action is not None:
            return action
        return self._safe_forward(pos, ori)

    def _step_to_any(self, pos, ori, candidates) -> Optional[int]:
        result = self.world.nearest_of(pos, candidates, blocked=set(self._blocked.keys()))
        if result is None:
            return None
        target, path = result
        if len(path) <= 1:
            # Already on a goal cell — no movement needed; the engine should
            # have given us the resource on the previous step. If not, mark
            # this tile blocked and try elsewhere next tick.
            self._blocked[target] = self._blocked_decay
            return self._safe_forward(pos, ori)
        # Next step is path[1]
        next_cell = path[1]
        return self._action_toward(pos, ori, next_cell)

    # ── hunt phase ──────────────────────────────────────────────────
    def _step_hunt(self, pos, ori, opponent_pos) -> int:
        # Already facing-aligned with clear line-of-sight → fire.
        if self._opponent_in_zap_line(pos, ori, opponent_pos):
            return INTERACT

        # If axis-aligned within zap range AND clear LoS in that
        # direction → orient & fire on next tick.
        target_ori = self._ori_toward(pos, opponent_pos)
        if target_ori is not None:
            # Check whether facing target_ori would give us a clean line
            # to opponent within zap range.
            if self._has_zap_line_in(pos, target_ori, opponent_pos):
                if target_ori != ori:
                    return turn_action(ori, target_ori)
                return INTERACT

        # Otherwise pathfind toward opponent. Their cell is on
        # upperPhysical (obstacle for us), so stop one cell short.
        path = self.world.bfs_path(pos, [opponent_pos])
        if path is None or len(path) <= 1:
            # Adjacent / co-located — face & fire even without LoS confirm.
            if target_ori is not None and target_ori != ori:
                return turn_action(ori, target_ori)
            return INTERACT
        if path[1] == opponent_pos:
            if target_ori is not None and target_ori != ori:
                return turn_action(ori, target_ori)
            return INTERACT
        return self._action_toward(pos, ori, path[1])

    def _has_zap_line_in(self, pos, target_ori, opponent_pos) -> bool:
        """Same as _opponent_in_zap_line but for a hypothetical orientation."""
        d = DIR_VEC[target_ori]
        cur = pos
        for _ in range(self.zap_max_range):
            cur = add(cur, d)
            if not self.world.in_bounds(cur):
                return False
            if cur in self.world.walls:
                return False
            if cur == opponent_pos:
                return True
        return False

    def _opponent_in_zap_line(self, pos, ori, opponent_pos) -> bool:
        """Crude: opponent is exactly N steps in front along current facing,
        with no walls in between, for N in [1, zap_max_range]."""
        d = DIR_VEC[ori]
        cur = pos
        for _ in range(self.zap_max_range):
            cur = add(cur, d)
            if not self.world.in_bounds(cur):
                return False
            if cur in self.world.walls:
                return False
            if cur == opponent_pos:
                return True
        return False

    # ── primitive choice ────────────────────────────────────────────
    def _action_toward(self, pos, ori, next_cell) -> int:
        """Pick a primitive action that moves toward `next_cell`."""
        d = (next_cell[0] - pos[0], next_cell[1] - pos[1])
        target_ori = vec_to_ori(d)
        if target_ori < 0:
            return self._safe_forward(pos, ori)

        if target_ori == ori:
            return FORWARD
        # Use STEP_LEFT/STEP_RIGHT for 90° relative steps without turning,
        # to save time. STEP_RIGHT moves us along ori+1; STEP_LEFT along ori-1.
        if target_ori == (ori + 1) % 4:
            return STEP_RIGHT
        if target_ori == (ori - 1) % 4:
            return STEP_LEFT
        # 180° → BACKWARD
        return BACKWARD

    def _ori_toward(self, pos, target) -> Optional[int]:
        d = (target[0] - pos[0], target[1] - pos[1])
        # Reduce to cardinal axis if aligned.
        if d[0] != 0 and d[1] != 0:
            return None
        if d == (0, 0):
            return None
        ux = (1 if d[0] > 0 else -1) if d[0] != 0 else 0
        uy = (1 if d[1] > 0 else -1) if d[1] != 0 else 0
        return vec_to_ori((ux, uy))

    def _safe_forward(self, pos, ori) -> int:
        """Move forward if not blocked; else turn arbitrarily."""
        front = add(pos, DIR_VEC[ori])
        if self.world.passable(front):
            return FORWARD
        return TURN_RIGHT
