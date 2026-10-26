"""Minimal multi-agent constitution revision mechanism.

This is a compact sketch of the paper's practical-wisdom experiment:
an agent acts in a repeated interaction, then periodically reflects on
recent outcomes and edits a small constitution.

The file intentionally omits plotting, multi-seed orchestration, external
environments, and model-serving setup.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
import json
import random
import uuid


COOPERATE = "cooperate"
DEFECT = "defect"


@dataclass
class Principle:
    text: str
    confidence: float = 0.5
    evidence: list[str] = field(default_factory=list)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    version: int = 1
    active: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


class SmallConstitution:
    def __init__(self):
        self.entries: dict[str, Principle] = {}
        self.snapshots: list[dict] = []

    def add(self, text: str, confidence: float, evidence: str = "") -> str:
        entry = Principle(text=text, confidence=confidence, evidence=[evidence] if evidence else [])
        self.entries[entry.id] = entry
        return entry.id

    def update(self, entry_id: str, text: str, confidence: float, evidence: str = "") -> None:
        entry = self.entries[entry_id]
        entry.text = text
        entry.confidence = confidence
        entry.version += 1
        if evidence:
            entry.evidence.append(evidence)

    def retire(self, entry_id: str, evidence: str = "") -> None:
        entry = self.entries[entry_id]
        entry.active = False
        if evidence:
            entry.evidence.append(evidence)

    def active(self) -> list[Principle]:
        return [e for e in self.entries.values() if e.active]

    def summary(self) -> str:
        active = self.active()
        if not active:
            return "(no active principles)"
        return "\n".join(f"[{e.id} v{e.version}] {e.text}" for e in active)

    def snapshot(self, round_index: int) -> None:
        self.snapshots.append(
            {"round": round_index, "entries": [e.to_dict() for e in self.entries.values()]}
        )


class MatrixGame:
    """Two-action repeated game with prisoner-dilemma style payoffs."""

    def payoff(self, action0: str, action1: str) -> tuple[float, float]:
        if action0 == COOPERATE and action1 == COOPERATE:
            return 5.0, 5.0
        if action0 == DEFECT and action1 == COOPERATE:
            return 10.0, 0.0
        if action0 == COOPERATE and action1 == DEFECT:
            return 0.0, 10.0
        return 1.0, 1.0


def fixed_opponent(kind: str, history: list[dict], rng: random.Random) -> str:
    if kind == "always_cooperate":
        return COOPERATE
    if kind == "always_defect":
        return DEFECT
    if kind == "random":
        return rng.choice([COOPERATE, DEFECT])
    if kind == "tit_for_tat" and history:
        return history[-1]["self_action"]
    return COOPERATE


ACTION_PROMPT = """Environment history:
{history}

Your constitution:
{constitution}

Choose one action: cooperate or defect. Return JSON:
{{"reasoning": "...", "action": "cooperate|defect"}}
"""

REFLECTION_PROMPT = """Recent interaction block:
{recent}

Current constitution:
{constitution}

Return JSON:
{{
  "reasoning": "...",
  "deltas": [
    {{"action": "add", "principle": "...", "confidence": 0.8, "evidence": "..."}},
    {{"action": "update", "id": "...", "principle": "...", "confidence": 0.8, "evidence": "..."}},
    {{"action": "retire", "id": "...", "evidence": "..."}}
  ]
}}
"""


class IteratingAgent:
    def __init__(self, backend):
        self.backend = backend
        self.constitution = SmallConstitution()

    def act(self, history: list[dict]) -> str:
        raw = self.backend.generate(
            ACTION_PROMPT.format(
                history=history[-8:],
                constitution=self.constitution.summary(),
            )
        )
        try:
            action = json.loads(raw).get("action", COOPERATE)
        except json.JSONDecodeError:
            action = COOPERATE
        return action if action in {COOPERATE, DEFECT} else COOPERATE

    def reflect(self, history: list[dict], block_size: int) -> None:
        raw = self.backend.reflect(
            REFLECTION_PROMPT.format(
                recent=history[-block_size:],
                constitution=self.constitution.summary(),
            )
        )
        try:
            reflection = json.loads(raw)
        except json.JSONDecodeError:
            reflection = {"deltas": []}
        for delta in reflection.get("deltas", []):
            action = delta.get("action")
            if action == "add":
                self.constitution.add(
                    delta.get("principle", ""),
                    float(delta.get("confidence", 0.5)),
                    delta.get("evidence", ""),
                )
            elif action == "update" and delta.get("id") in self.constitution.entries:
                self.constitution.update(
                    delta["id"],
                    delta.get("principle", ""),
                    float(delta.get("confidence", 0.5)),
                    delta.get("evidence", ""),
                )
            elif action == "retire" and delta.get("id") in self.constitution.entries:
                self.constitution.retire(delta["id"], delta.get("evidence", ""))


def run_repeated_game(
    backend,
    opponent_kind: str,
    rounds: int = 100,
    reflect_every: int = 5,
    seed: int = 0,
) -> dict:
    rng = random.Random(seed)
    game = MatrixGame()
    agent = IteratingAgent(backend)
    history: list[dict] = []
    agent.constitution.snapshot(0)

    for round_index in range(rounds):
        self_action = agent.act(history)
        opponent_action = fixed_opponent(opponent_kind, history, rng)
        self_reward, opponent_reward = game.payoff(self_action, opponent_action)
        history.append(
            {
                "round": round_index,
                "self_action": self_action,
                "opponent_action": opponent_action,
                "self_reward": self_reward,
                "opponent_reward": opponent_reward,
            }
        )
        if (round_index + 1) % reflect_every == 0:
            agent.reflect(history, block_size=reflect_every)
            agent.constitution.snapshot(round_index + 1)

    return {
        "opponent": opponent_kind,
        "history": history,
        "constitution_snapshots": agent.constitution.snapshots,
        "final_constitution": [p.to_dict() for p in agent.constitution.active()],
    }
