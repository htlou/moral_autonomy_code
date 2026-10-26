"""Agents.

Two kinds:
  - LLMAgent          : uses an LLM client + a constitution store
  - FixedStrategyAgent: hand-coded baselines (always_cooperate / always_defect /
                        tit_for_tat / random) for matrix-style games

Both implement the same `pick_action` signature so the runner can mix them.
LLMAgent additionally exposes `reflect_and_update` (used only in the
iterating-constitution condition).
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .envs.base import EnvAdapter, ObsView, TransitionLog
from .llm import parse_json_lenient
from .prompts import ACTION_SYSTEM, ACTION_USER, REFLECTION_SYSTEM, REFLECTION_USER
from .store import ConstitutionEntry, ListConstitutionStore

logger = logging.getLogger(__name__)


# ── History formatting ──────────────────────────────────────────────────────

def format_history(transitions: List[TransitionLog], player_id: int,
                   window: int) -> str:
    if not transitions:
        return "(no history yet)"
    sub = transitions[-window:]
    lines = []
    for t in sub:
        my_a = t.action_names[player_id]
        my_r = t.rewards[player_id]
        # Render other players' moves compactly.
        others = ", ".join(
            f"p{i}={t.action_names[i]} (r={t.rewards[i]:.1f})"
            for i in range(len(t.action_names)) if i != player_id
        )
        lines.append(
            f"  step {t.step_idx}: you={my_a} (r={my_r:.1f}) | {others}"
        )
    return "\n".join(lines)


# ── Fixed-strategy opponents (matrix-style only) ────────────────────────────

class FixedStrategyAgent:
    """Hand-coded opponents for 2-action matrix games. action 0 = cooperate."""

    KINDS = {"always_cooperate", "always_defect", "tit_for_tat", "random"}

    def __init__(self, kind: str, env: EnvAdapter, player_id: int, seed: int = 0):
        if kind not in self.KINDS:
            raise ValueError(f"unknown strategy: {kind}")
        self.kind = kind
        self.env = env
        self.player_id = player_id
        self._rng = random.Random(seed)
        n = len(env.legal_action_names(player_id))
        self._coop = 0
        self._defect = min(1, n - 1)

    def pick_action(self, step_idx: int, transitions: List[TransitionLog],
                    observation: ObsView, total_steps: int) -> Tuple[int, dict]:
        if self.kind == "always_cooperate":
            a = self._coop
        elif self.kind == "always_defect":
            a = self._defect
        elif self.kind == "tit_for_tat":
            if not transitions:
                a = self._coop
            else:
                # Mirror the opponent's last action. With 2 players, the
                # "other" player's id is 1 - self.player_id.
                opp_id = 1 - self.player_id
                a = transitions[-1].actions[opp_id]
        elif self.kind == "random":
            a = self._rng.randrange(len(observation.legal_actions))
        else:
            raise RuntimeError(self.kind)
        return a, {"strategy": self.kind, "action": observation.legal_actions[a]}

    # No-op interfaces so the runner can treat LLM and fixed agents uniformly.
    def reflect_and_update(self, *args, **kwargs):
        return {"reasoning": "fixed strategy", "deltas": [], "applied": []}

    @property
    def store(self) -> Optional[ListConstitutionStore]:
        return None


# ── LLM agent ──────────────────────────────────────────────────────────────

@dataclass
class HysteresisConfig:
    """Damp constitution churn. Defaults are no-ops so behavior is unchanged
    when hysteresis is disabled.

    - retire_min_consecutive : a `retire` delta is buffered and only applied
        after the same principle has been retire-voted in this many
        consecutive reflections. Updates that drop confidence by more than
        `retire_conf_drop_counts` count as a retire vote.
    - retire_conf_drop_counts : threshold used to convert a confidence-drop
        update into a retire vote.
    - update_min_conf_delta : updates with |new_conf - old_conf| below this
        threshold AND no principle text change are treated as no-ops.
    - add_min_conf : new principles below this confidence are dropped.
    """
    retire_min_consecutive: int = 1
    retire_conf_drop_counts: float = 0.5
    update_min_conf_delta: float = 0.0
    add_min_conf: float = 0.0


def default_hysteresis() -> HysteresisConfig:
    """Recommended hysteresis policy when --hysteresis is enabled."""
    return HysteresisConfig(
        retire_min_consecutive=3,
        retire_conf_drop_counts=0.5,
        update_min_conf_delta=0.3,
        add_min_conf=0.5,
    )


@dataclass
class LLMAgentConfig:
    player_id: int
    history_window: int = 10
    action_temperature: float = 0.7
    reflect_temperature: float = 0.3
    max_tokens_action: int = 256
    max_tokens_reflect: int = 1024
    max_constitution_size: int = 20
    fallback_action_idx: int = 0
    hysteresis: HysteresisConfig = field(default_factory=HysteresisConfig)


class LLMAgent:
    """LLM-driven agent that reads/updates a constitution store."""

    def __init__(self, llm, env: EnvAdapter, store: ListConstitutionStore,
                 cfg: LLMAgentConfig, name: str | None = None):
        self.llm = llm
        self.env = env
        self.store = store
        self.cfg = cfg
        self.name = name or f"llm_p{cfg.player_id}"
        self.action_log: List[dict] = []
        self.reflection_log: List[dict] = []
        # Hysteresis state: number of consecutive retire votes per principle id.
        # Reset to 0 whenever a reflection does NOT propose retiring the id.
        self._retire_votes: dict[str, int] = {}

    # ── action ──────────────────────────────────────────────────────
    def pick_action(self, step_idx: int, transitions: List[TransitionLog],
                    observation: ObsView, total_steps: int) -> Tuple[int, dict]:
        legal = observation.legal_actions
        user = ACTION_USER.format(
            env_preamble=self.env.system_preamble(self.cfg.player_id),
            player_id=self.cfg.player_id,
            legal_actions=", ".join(legal),
            constitution=self.store.summary(),
            history=format_history(transitions, self.cfg.player_id,
                                   self.cfg.history_window),
            history_n=self.cfg.history_window,
            observation=observation.text,
            step_idx=step_idx,
            total_steps=total_steps,
        )
        raw = self.llm.complete(
            [{"role": "system", "content": ACTION_SYSTEM},
             {"role": "user", "content": user}],
            temperature=self.cfg.action_temperature,
            max_tokens=self.cfg.max_tokens_action,
        )
        parsed = parse_json_lenient(raw)
        action_name = (parsed or {}).get("action", "")
        a = self._resolve_action_idx(action_name, raw, legal)

        log_entry = {
            "step": step_idx,
            "agent": self.name,
            "raw": raw,
            "parsed": parsed,
            "chosen_idx": a,
            "chosen_action": legal[a] if 0 <= a < len(legal) else None,
        }
        self.action_log.append(log_entry)
        return a, log_entry

    def _resolve_action_idx(self, name: str, raw: str, legal: list[str]) -> int:
        if name:
            try:
                return self.env.action_index(self.cfg.player_id, name)
            except ValueError:
                pass
        # lexical scan: any legal name appearing in raw output
        low = raw.lower()
        for i, n in enumerate(legal):
            if n.lower() in low:
                return i
        return self.cfg.fallback_action_idx

    # ── reflection ──────────────────────────────────────────────────
    def reflect_and_update(self, transitions: List[TransitionLog],
                           start: int, end: int) -> dict:
        block = [t for t in transitions if start <= t.step_idx < end]
        if not block:
            return {"reasoning": "no rounds in block", "deltas": [], "applied": []}

        my_id = self.cfg.player_id
        cum = sum(t.rewards[my_id] for t in transitions)
        n = len(transitions)
        history_text = format_history(block, my_id, len(block))

        sys_prompt = REFLECTION_SYSTEM.format(max_size=self.cfg.max_constitution_size)
        user_prompt = REFLECTION_USER.format(
            env_preamble=self.env.system_preamble(my_id),
            player_id=my_id,
            constitution=self.store.summary(),
            start=start, end=end - 1,
            history=history_text,
            cum_payoff=cum, n_steps=n,
            mean_payoff=(cum / n) if n else 0.0,
        )
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ]
        raw = self.llm.complete(
            messages,
            temperature=self.cfg.reflect_temperature,
            max_tokens=self.cfg.max_tokens_reflect,
        )
        parsed = parse_json_lenient(raw) or {}
        deltas = parsed.get("deltas", []) if isinstance(parsed, dict) else []
        applied = self._apply_deltas(deltas, evidence_tag=f"steps_{start}_{end-1}")
        record = {
            "agent": self.name,
            "steps": (start, end - 1),
            "raw": raw,
            "parsed": parsed,
            "applied": applied,
            # Retain the exact pre-generation input for causal activation
            # extraction.  This is deliberately reflection-only so result
            # artifacts do not duplicate every action prompt.
            "messages": [dict(message) for message in messages],
            "input_prompt_version": "reflection_v1",
        }
        self.reflection_log.append(record)
        return record

    def _apply_deltas(self, deltas: list, evidence_tag: str) -> List[str]:
        applied: List[str] = []
        active_count = len(self.store.list_active())
        h = self.cfg.hysteresis

        # Track which principles received a retire signal *this* reflection
        # so we can decay vote counters for everything else.
        retire_voted_this_round: set[str] = set()

        for d in deltas or []:
            if not isinstance(d, dict):
                continue
            action = str(d.get("action", "")).lower()
            try:
                if action == "add":
                    if active_count >= self.cfg.max_constitution_size:
                        applied.append("SKIP_ADD (capacity)")
                        continue
                    p = (d.get("principle") or "").strip()
                    if not p:
                        continue
                    conf = float(d.get("confidence", 0.5))
                    if conf < h.add_min_conf:
                        applied.append(f"SKIP_ADD (conf {conf:.2f} < {h.add_min_conf:.2f})")
                        continue
                    entry = ConstitutionEntry(
                        principle=p,
                        confidence=conf,
                        evidence=[d.get("evidence", evidence_tag)],
                    )
                    eid = self.store.add(entry)
                    active_count += 1
                    applied.append(f"ADD {eid}: {p[:60]}")

                elif action == "update":
                    eid = d.get("id")
                    if not eid or eid not in self.store.index:
                        applied.append(f"SKIP_UPDATE {eid} (not found)")
                        continue
                    existing = self.store.get(eid)
                    if existing is None:
                        continue

                    new_principle = d.get("principle")
                    new_conf = (
                        float(d["confidence"]) if "confidence" in d else existing.confidence
                    )
                    text_changed = (
                        new_principle is not None
                        and new_principle.strip() != existing.principle.strip()
                    )
                    conf_delta = abs(new_conf - existing.confidence)

                    # Hysteresis: skip tiny no-op updates.
                    if not text_changed and conf_delta < h.update_min_conf_delta:
                        applied.append(
                            f"SKIP_UPDATE {eid} "
                            f"(Δconf {conf_delta:.2f} < {h.update_min_conf_delta:.2f}, no text change)"
                        )
                        # A confidence drop counts as a soft retire vote.
                        if (existing.confidence - new_conf) >= h.retire_conf_drop_counts:
                            self._retire_votes[eid] = self._retire_votes.get(eid, 0) + 1
                            retire_voted_this_round.add(eid)
                            applied.append(
                                f"RETIRE_VOTE {eid} "
                                f"({self._retire_votes[eid]}/{h.retire_min_consecutive})"
                            )
                        continue

                    updates: dict = {}
                    if new_principle is not None:
                        updates["principle"] = new_principle
                    if "confidence" in d:
                        updates["confidence"] = new_conf
                    updates["evidence"] = existing.evidence + [
                        d.get("evidence", evidence_tag)
                    ]
                    self.store.update(eid, updates)
                    applied.append(f"UPDATE {eid} (Δconf={conf_delta:.2f})")

                elif action == "retire":
                    eid = d.get("id")
                    if not eid or eid not in self.store.index:
                        applied.append(f"SKIP_RETIRE {eid} (not found)")
                        continue
                    # Buffer retire vote.
                    self._retire_votes[eid] = self._retire_votes.get(eid, 0) + 1
                    retire_voted_this_round.add(eid)
                    if self._retire_votes[eid] >= h.retire_min_consecutive:
                        self.store.retire(eid)
                        active_count = max(0, active_count - 1)
                        self._retire_votes.pop(eid, None)
                        applied.append(f"RETIRE {eid}")
                    else:
                        applied.append(
                            f"BUFFER_RETIRE {eid} "
                            f"({self._retire_votes[eid]}/{h.retire_min_consecutive})"
                        )
            except Exception as e:
                applied.append(f"ERR {action}: {e}")

        # Decay retire votes for principles NOT voted against this reflection.
        for eid in list(self._retire_votes.keys()):
            if eid not in retire_voted_this_round:
                self._retire_votes.pop(eid, None)

        return applied
