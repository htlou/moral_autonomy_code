"""Static world model for `*_in_the_matrix` substrates.

Parses the ASCII map to extract walls and resource positions. Provides BFS
pathfinding so the high-level controller doesn't need to read pixels.

Convention from the live substrate:
  - POSITION is reported as [col, row], 0-indexed.
  - ORIENTATION is 0=N, 1=E, 2=S, 3=W.
  - TURN_LEFT decrements ORI mod 4; TURN_RIGHT increments.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable, Optional


# ── direction utilities ─────────────────────────────────────────────────────
# Stored as (dcol, drow). N is "up" (row decreases).
DIR_VEC = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}
DIR_NAME = {0: "N", 1: "E", 2: "S", 3: "W"}

# Action indices (must match prisoners_dilemma_in_the_matrix__repeated.ACTION_SET)
NOOP, FORWARD, BACKWARD, STEP_LEFT, STEP_RIGHT, TURN_LEFT, TURN_RIGHT, INTERACT = range(8)


def add(p, d):
    return (p[0] + d[0], p[1] + d[1])


def turn_action(curr_ori: int, target_ori: int) -> int:
    """Return one primitive action that rotates curr_ori toward target_ori,
    or FORWARD if already aligned."""
    if curr_ori == target_ori:
        return FORWARD
    diff = (target_ori - curr_ori) % 4
    if diff == 1:
        return TURN_RIGHT
    if diff == 3:
        return TURN_LEFT
    # 180° turn — pick TURN_RIGHT arbitrarily; controller will need 2 turns.
    return TURN_RIGHT


def vec_to_ori(d: tuple[int, int]) -> int:
    """(dcol, drow) -> ORI (0=N, 1=E, 2=S, 3=W). Returns -1 if not unit-axis."""
    for ori, vec in DIR_VEC.items():
        if vec == d:
            return ori
    return -1


# ── ASCII map parsing ───────────────────────────────────────────────────────

# This is the literal map string from the PD-in-the-matrix repeated config.
# Hand-copied so we don't depend on the substrate Python being importable.
ASCII_MAP_PD_REPEATED = """\
WWWWWWWWWWWWWWWWWWWWWWW
Wn         n         nW
W   WWW W  W  W WW    W
W    W 11a W a22 W    W
Wn  WW 11a W a22 WW  nW
W      11a   a22      W
W                     W
Wn WW  WW  n WW  WWW nW
W                     W
W      22a W a11      W
Wn   W 22a W a11 W   nW
W    W 22a W a11 WW   W
W  WWWW W  W  W WWW   W
Wn         n         nW
WWWWWWWWWWWWWWWWWWWWWWW
"""


@dataclass
class WorldGrid:
    """Static features of the substrate's map.

    Coordinates are (col, row), matching MP's POSITION reporting.
    """
    width: int
    height: int
    walls: set[tuple[int, int]]      # (col, row) blocked by wall
    coop_tiles: set[tuple[int, int]] # deterministic cooperate (resource_class1)
    defect_tiles: set[tuple[int, int]]  # deterministic defect (resource_class2)
    random_tiles: set[tuple[int, int]]  # 'a' choice tiles — color unknown a priori
    spawns: set[tuple[int, int]]

    @classmethod
    def from_ascii(cls, ascii_map: str) -> "WorldGrid":
        rows = [line for line in ascii_map.splitlines() if line]
        height = len(rows)
        width = max(len(r) for r in rows)
        walls = set()
        coop = set()
        defect = set()
        rand = set()
        spawns = set()
        for row_idx, line in enumerate(rows):
            for col_idx, ch in enumerate(line):
                pos = (col_idx, row_idx)
                if ch == "W":
                    walls.add(pos)
                elif ch == "1":
                    coop.add(pos)
                elif ch == "2":
                    defect.add(pos)
                elif ch == "a":
                    rand.add(pos)
                elif ch == "n":
                    spawns.add(pos)
        return cls(width, height, walls, coop, defect, rand, spawns)

    def in_bounds(self, p: tuple[int, int]) -> bool:
        return 0 <= p[0] < self.width and 0 <= p[1] < self.height

    def passable(self, p: tuple[int, int]) -> bool:
        """A square is passable if it is in-bounds and not a wall.

        Resource tiles are passable — they sit on a separate physical layer.
        """
        return self.in_bounds(p) and p not in self.walls

    # ── pathfinding ─────────────────────────────────────────────────
    def bfs_path(self, start: tuple[int, int],
                 goals: Iterable[tuple[int, int]],
                 max_distance: int = 200) -> Optional[list[tuple[int, int]]]:
        """Shortest path from `start` to any cell in `goals`.

        Returns the sequence of cells from start to goal (inclusive), or
        None if no path. Goals do NOT need to be passable themselves —
        we accept goals where the goal IS the destination tile, so we
        consider neighbors of goals too.
        """
        goals = set(goals)
        if not goals:
            return None
        if start in goals:
            return [start]
        visited = {start: None}
        q = deque([start])
        while q:
            cur = q.popleft()
            if visited.get(cur, -1) is None:
                pass  # depth tracker not stored; bound by max_distance via len
            for dvec in DIR_VEC.values():
                nxt = add(cur, dvec)
                if nxt in visited:
                    continue
                if not self.passable(nxt) and nxt not in goals:
                    continue
                visited[nxt] = cur
                if nxt in goals:
                    # reconstruct
                    path = [nxt]
                    while path[-1] != start:
                        path.append(visited[path[-1]])
                    path.reverse()
                    return path
                q.append(nxt)
            if len(visited) > max_distance * max_distance:
                return None
        return None

    def nearest_of(self, start: tuple[int, int],
                   candidates: Iterable[tuple[int, int]],
                   blocked: Optional[set[tuple[int, int]]] = None
                   ) -> Optional[tuple[tuple[int, int], list[tuple[int, int]]]]:
        """Return (nearest_pos, path) or None. `blocked` lets the caller
        exclude resource tiles known to be in their wait state."""
        cands = set(candidates)
        if blocked:
            cands -= blocked
        if not cands:
            return None
        path = self.bfs_path(start, cands)
        if path is None:
            return None
        return path[-1], path
