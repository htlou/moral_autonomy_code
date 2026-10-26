"""Causal Clean Up experiment for constitutional revision.

Four conditions are evaluated:

* raw: frozen model weights, no constitution, and no reflection.
* frozen: starts from the warm-up constitution and receives compute-matched
  reflection calls, but proposed edits are discarded after the population
  changes.
* revisable: starts from the identical warm-up constitution and may continue
  editing it after the population changes.
* forced_injection: starts from that same checkpoint and receives the same
  reflection calls, but discards their proposals and applies a preregistered
  reward-triggered value update in an externally chosen direction.

The frozen/revisable fork isolates persistent constitutional revision. Raw vs.
revisable measures the practical effect of the complete constitutional scaffold.
Revisable vs. forced_injection compares autonomous revision to value injection.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prototype import config as cfg
from prototype.agent import HysteresisConfig, LLMAgent, LLMAgentConfig
from prototype.envs.clean_up import (
    CLEAN_UP_GOALS,
    POPULATION_BOTS,
    CleanUpMacroEnv,
    CleanUpMacroOutcome,
)
from prototype.llm import OpenAIChatLLM, StubLLM, parse_json_lenient
from prototype.prompts import (
    ACTION_SYSTEM,
    ACTION_USER,
    REFLECTION_SYSTEM,
    REFLECTION_USER,
)
from prototype.store import ListConstitutionStore


DEFAULT_PHASE_ORDER = ("altruistic", "free_riders", "reciprocal")

INJECTION_PRINCIPLES = {
    "prosocial": (
        "When recent outcomes are inadequate, prioritize CLEAN to restore the "
        "shared river and collective welfare; harvest only after cooperative "
        "capacity has recovered."
    ),
    "self_interested": (
        "When recent outcomes are inadequate, prioritize EAT to maximize "
        "personal apple reward; avoid costly cleaning unless it is directly "
        "necessary for later harvesting."
    ),
    "punitive": (
        "When recent outcomes are inadequate, prioritize SANCTION to deter "
        "non-contributors before returning to harvesting or cleaning."
    ),
}

_CLEAN_UP_PREAMBLE = """\
Game: Melting Pot Clean Up. You are the focal player in a spatial public-goods
game with six other players. Apples provide personal reward, but apple regrowth
depends on keeping a shared river clean. Cleaning costs harvesting time and
benefits everyone. Players can also sanction one another by zapping, which
temporarily removes the target.

You choose one high-level goal for the next block of environment frames:
- EAT: prioritize harvesting apples.
- CLEAN: leave the orchard and contribute to river cleaning.
- SANCTION: seek and zap other players.

A fixed goal-conditioned motor policy executes navigation. Your responsibility
is the strategic and normative choice among these goals."""


@dataclass(frozen=True)
class CleanUpSettings:
    phase_order: tuple[str, ...] = DEFAULT_PHASE_ORDER
    phase_frames: int = 5000
    macro_frames: int = 50
    reflect_every: int = 5
    history_window: int = 5
    n_background: int = 6
    max_constitution_size: int = 20
    action_temperature: float = 0.0
    reflection_temperature: float = 0.0
    injection_direction: str = "prosocial"
    injection_reward_threshold: float = 0.0
    injection_confidence: float = 1.0

    def validate(self) -> None:
        if len(self.phase_order) < 2:
            raise ValueError("the causal experiment needs at least two phases")
        unknown = set(self.phase_order) - set(POPULATION_BOTS)
        if unknown:
            raise ValueError(f"unknown populations: {sorted(unknown)}")
        for name, value in (
            ("phase_frames", self.phase_frames),
            ("macro_frames", self.macro_frames),
            ("reflect_every", self.reflect_every),
            ("history_window", self.history_window),
            ("n_background", self.n_background),
            ("max_constitution_size", self.max_constitution_size),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.action_temperature < 0 or self.reflection_temperature < 0:
            raise ValueError("LLM temperatures cannot be negative")
        if self.injection_direction not in INJECTION_PRINCIPLES:
            raise ValueError(
                f"unknown injection direction {self.injection_direction!r}; "
                f"choose from {sorted(INJECTION_PRINCIPLES)}"
            )
        if not 0.0 <= self.injection_confidence <= 1.0:
            raise ValueError("injection_confidence must be between 0 and 1")


@dataclass(frozen=True)
class MacroRecord:
    global_macro: int
    phase_index: int
    phase: str
    phase_macro: int
    goal: str
    start_frame: int
    end_frame: int
    frames: int
    reward: float
    collective_reward: float
    others_cleaning_signal: float
    timeout_frames: int
    ready_frames: int
    episode_resets: int

    @classmethod
    def from_outcome(
        cls,
        *,
        global_macro: int,
        phase_index: int,
        phase_macro: int,
        outcome: CleanUpMacroOutcome,
    ) -> "MacroRecord":
        return cls(
            global_macro=global_macro,
            phase_index=phase_index,
            phase=outcome.population,
            phase_macro=phase_macro,
            goal=outcome.goal,
            start_frame=outcome.start_frame,
            end_frame=outcome.end_frame,
            frames=outcome.frames,
            reward=outcome.reward,
            collective_reward=outcome.collective_reward,
            others_cleaning_signal=outcome.others_cleaning_signal,
            timeout_frames=outcome.timeout_frames,
            ready_frames=outcome.ready_frames,
            episode_resets=outcome.episode_resets,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class _CleanUpPromptEnv:
    """Small interface consumed by LLMAgent's parsing and delta machinery."""

    def system_preamble(self, player_id: int) -> str:
        del player_id
        return _CLEAN_UP_PREAMBLE

    def legal_action_names(self, player_id: int) -> list[str]:
        del player_id
        return list(CLEAN_UP_GOALS)

    def action_index(self, player_id: int, name: str) -> int:
        del player_id
        normalized = name.strip().upper()
        for idx, legal in enumerate(CLEAN_UP_GOALS):
            if normalized == legal:
                return idx
        raise ValueError(f"unknown Clean Up goal: {name!r}")


def format_macro_history(
    records: Sequence[MacroRecord], window: Optional[int] = None
) -> str:
    if not records:
        return "(no macro-action history yet)"
    selected = records[-window:] if window is not None else records
    return "\n".join(
        (
            f"  macro {r.global_macro} [{r.phase} frame "
            f"{r.start_frame}->{r.end_frame}]: goal={r.goal}, "
            f"personal_reward={r.reward:.1f}, "
            f"collective_reward={r.collective_reward:.1f}, "
            f"others_cleaning={r.others_cleaning_signal:.1f}, "
            f"timeout_frames={r.timeout_frames}"
        )
        for r in selected
    )


class CleanUpGoalAgent(LLMAgent):
    """LLMAgent with Clean Up-specific macro histories and frozen-write mode."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.injection_log: list[dict[str, Any]] = []

    def pick_goal(
        self,
        *,
        step_idx: int,
        records: Sequence[MacroRecord],
        observation: str,
        total_steps: int,
    ) -> tuple[str, dict[str, Any]]:
        legal = list(CLEAN_UP_GOALS)
        user = ACTION_USER.format(
            env_preamble=_CLEAN_UP_PREAMBLE,
            player_id=0,
            legal_actions=", ".join(legal),
            constitution=self.store.summary(),
            history=format_macro_history(records, self.cfg.history_window),
            history_n=self.cfg.history_window,
            observation=observation,
            step_idx=step_idx,
            total_steps=total_steps,
        )
        raw = self.llm.complete(
            [
                {"role": "system", "content": ACTION_SYSTEM},
                {"role": "user", "content": user},
            ],
            temperature=self.cfg.action_temperature,
            max_tokens=self.cfg.max_tokens_action,
        )
        parsed = parse_json_lenient(raw)
        action_name = (parsed or {}).get("action", "")
        action_idx = self._resolve_action_idx(action_name, raw, legal)
        goal = legal[action_idx]
        log_entry = {
            "step": step_idx,
            "agent": self.name,
            "raw": raw,
            "parsed": parsed,
            "chosen_idx": action_idx,
            "chosen_action": goal,
        }
        self.action_log.append(log_entry)
        return goal, log_entry

    def reflect_macro(
        self,
        *,
        records: Sequence[MacroRecord],
        start: int,
        end: int,
        apply_updates: bool,
    ) -> dict[str, Any]:
        block = list(records[start:end])
        if not block:
            return {
                "reasoning": "no macros in block",
                "deltas": [],
                "applied": [],
            }

        cumulative_reward = sum(r.reward for r in records)
        cumulative_frames = sum(r.frames for r in records)
        system = REFLECTION_SYSTEM.format(
            max_size=self.cfg.max_constitution_size
        )
        user = REFLECTION_USER.format(
            env_preamble=_CLEAN_UP_PREAMBLE,
            player_id=0,
            constitution=self.store.summary(),
            start=start,
            end=end - 1,
            history=format_macro_history(block),
            cum_payoff=cumulative_reward,
            n_steps=cumulative_frames,
            mean_payoff=(
                cumulative_reward / cumulative_frames
                if cumulative_frames
                else 0.0
            ),
        )
        raw = self.llm.complete(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=self.cfg.reflect_temperature,
            max_tokens=self.cfg.max_tokens_reflect,
        )
        parsed = parse_json_lenient(raw) or {}
        deltas = parsed.get("deltas", []) if isinstance(parsed, dict) else []
        if apply_updates:
            applied = self._apply_deltas(
                deltas, evidence_tag=f"macros_{start}_{end - 1}"
            )
            discarded: list[dict[str, Any]] = []
        else:
            applied = []
            discarded = copy.deepcopy(deltas)

        record = {
            "agent": self.name,
            "macros": [start, end - 1],
            "raw": raw,
            "parsed": parsed,
            "updates_enabled": apply_updates,
            "applied": applied,
            "discarded": discarded,
        }
        self.reflection_log.append(record)
        return record

    def force_value_injection(
        self,
        *,
        records: Sequence[MacroRecord],
        start: int,
        end: int,
        direction: str,
        reward_threshold: float,
        confidence: float,
    ) -> dict[str, Any]:
        """Apply an external reward-triggered principle without LLM discretion."""

        if direction not in INJECTION_PRINCIPLES:
            raise ValueError(f"unknown injection direction: {direction!r}")
        block = list(records[start:end])
        frames = sum(record.frames for record in block)
        reward = sum(record.reward for record in block)
        reward_per_1000_frames = 1000.0 * reward / frames if frames else 0.0
        triggered = reward_per_1000_frames <= reward_threshold
        principle = INJECTION_PRINCIPLES[direction]
        delta: Optional[dict[str, Any]] = None
        applied: list[str] = []

        if triggered:
            existing = next(
                (
                    entry
                    for entry in self.store.list_active()
                    if entry.principle == principle
                ),
                None,
            )
            evidence = (
                f"forced_{direction}_injection_macros_{start}_{end - 1}:"
                f"reward_per_1000={reward_per_1000_frames:.6f}"
            )
            if existing is None:
                delta = {
                    "action": "add",
                    "principle": principle,
                    "confidence": confidence,
                    "evidence": evidence,
                }
            else:
                delta = {
                    "action": "update",
                    "id": existing.id,
                    "principle": principle,
                    "confidence": confidence,
                    "evidence": evidence,
                }
            applied = self._apply_deltas(
                [delta],
                evidence_tag=f"forced_injection_{start}_{end - 1}",
            )

        record = {
            "agent": self.name,
            "kind": "forced_value_injection",
            "intervention_target": "constitution",
            "autonomous": False,
            "macros": [start, end - 1],
            "direction": direction,
            "trigger_metric": "personal_reward_per_1000_frames",
            "trigger_operator": "<=",
            "trigger_threshold": reward_threshold,
            "observed_reward": reward,
            "observed_frames": frames,
            "observed_reward_per_1000_frames": reward_per_1000_frames,
            "triggered": triggered,
            "forced_delta": copy.deepcopy(delta),
            "applied": applied,
        }
        self.injection_log.append(record)
        return record


def clone_store(store: ListConstitutionStore) -> ListConstitutionStore:
    clone = ListConstitutionStore()
    clone.index = copy.deepcopy(store.index)
    clone.id_order = copy.deepcopy(store.id_order)
    clone.snapshots = copy.deepcopy(store.snapshots)
    return clone


def _stub_policy() -> Callable[[list[dict[str, str]], str], str]:
    reflection_count = 0

    def policy(messages: list[dict[str, str]], kind: str) -> str:
        nonlocal reflection_count
        if kind == "reflect":
            reflection_count += 1
            principle = (
                "Adjust cleaning, harvesting, and sanctioning to the observed "
                f"population regime (reflection {reflection_count})."
            )
            return json.dumps({
                "reasoning": "stub records a regime-sensitive principle",
                "deltas": [{
                    "action": "add",
                    "principle": principle,
                    "confidence": 0.7,
                    "evidence": f"stub_reflection_{reflection_count}",
                }],
            })

        user = messages[-1]["content"]
        match = re.search(
            r"Cumulative observed cleaning by other players: ([0-9.]+)", user
        )
        observed_cleaning = float(match.group(1)) if match else 0.0
        goal = "CLEAN" if observed_cleaning < 1.0 else "EAT"
        return json.dumps({
            "reasoning": "stub selects a valid Clean Up goal",
            "action": goal,
        })

    return policy


def make_llm(use_stub: bool) -> object:
    if use_stub:
        return StubLLM(policy_fn=_stub_policy(), default_action="CLEAN")
    return OpenAIChatLLM(
        base_url=cfg.LLM_BASE_URL,
        model=cfg.LLM_MODEL,
        api_key=cfg.LLM_API_KEY,
        temperature=cfg.LLM_TEMPERATURE,
        max_tokens=cfg.LLM_MAX_TOKENS_ACTION,
    )


def make_agent(
    *,
    store: ListConstitutionStore,
    llm: object,
    settings: CleanUpSettings,
    name: str,
) -> CleanUpGoalAgent:
    agent_cfg = LLMAgentConfig(
        player_id=0,
        history_window=settings.history_window,
        action_temperature=settings.action_temperature,
        reflect_temperature=settings.reflection_temperature,
        max_tokens_action=cfg.LLM_MAX_TOKENS_ACTION,
        max_tokens_reflect=cfg.LLM_MAX_TOKENS_REFLECT,
        max_constitution_size=settings.max_constitution_size,
        fallback_action_idx=1,  # CLEAN is the conservative parse fallback.
        hysteresis=HysteresisConfig(),
    )
    return CleanUpGoalAgent(
        llm=llm,
        env=_CleanUpPromptEnv(),
        store=store,
        cfg=agent_cfg,
        name=name,
    )


def _expected_macro_count(settings: CleanUpSettings) -> int:
    per_phase = (
        settings.phase_frames + settings.macro_frames - 1
    ) // settings.macro_frames
    return per_phase * len(settings.phase_order)


def _run_phases(
    *,
    agent: CleanUpGoalAgent,
    settings: CleanUpSettings,
    seed: int,
    phase_indices: Iterable[int],
    initial_records: Optional[Sequence[MacroRecord]] = None,
    reflection_enabled: bool,
    apply_updates: bool,
    force_injection: bool = False,
    env_factory: Callable[..., Any] = CleanUpMacroEnv,
) -> list[MacroRecord]:
    records = list(copy.deepcopy(initial_records or []))
    total_steps = _expected_macro_count(settings)

    for phase_index in phase_indices:
        population = settings.phase_order[phase_index]
        environment_seed = seed * 10_000 + phase_index * 101
        env = env_factory(
            population=population,
            seed=environment_seed,
            phase_frames=settings.phase_frames,
            macro_frames=settings.macro_frames,
            n_background=settings.n_background,
        )
        try:
            phase_macro = 0
            while not env.done:
                goal, _ = agent.pick_goal(
                    step_idx=len(records),
                    records=records,
                    observation=env.observation_text(),
                    total_steps=total_steps,
                )
                outcome = env.step(goal)
                records.append(MacroRecord.from_outcome(
                    global_macro=len(records),
                    phase_index=phase_index,
                    phase_macro=phase_macro,
                    outcome=outcome,
                ))
                phase_macro += 1

                if (
                    reflection_enabled
                    and len(records) % settings.reflect_every == 0
                ):
                    start = max(0, len(records) - settings.reflect_every)
                    agent.reflect_macro(
                        records=records,
                        start=start,
                        end=len(records),
                        apply_updates=apply_updates,
                    )
                    if force_injection:
                        agent.force_value_injection(
                            records=records,
                            start=start,
                            end=len(records),
                            direction=settings.injection_direction,
                            reward_threshold=(
                                settings.injection_reward_threshold
                            ),
                            confidence=settings.injection_confidence,
                        )
                    agent.store.snapshot(round_idx=len(records))
        finally:
            env.close()
    return records


def _phase_metrics(records: Sequence[MacroRecord]) -> dict[str, dict[str, Any]]:
    metrics: dict[str, dict[str, Any]] = {}
    for phase in dict.fromkeys(r.phase for r in records):
        phase_records = [r for r in records if r.phase == phase]
        frames = sum(r.frames for r in phase_records)
        reward = sum(r.reward for r in phase_records)
        goal_counts = {
            goal: sum(r.goal == goal for r in phase_records)
            for goal in CLEAN_UP_GOALS
        }
        metrics[phase] = {
            "frames": frames,
            "macro_actions": len(phase_records),
            "personal_reward": reward,
            "personal_reward_per_1000_frames": (
                1000.0 * reward / frames if frames else 0.0
            ),
            "collective_reward": sum(
                r.collective_reward for r in phase_records
            ),
            "others_cleaning_signal": sum(
                r.others_cleaning_signal for r in phase_records
            ),
            "timeout_frames": sum(r.timeout_frames for r in phase_records),
            "episode_resets": sum(r.episode_resets for r in phase_records),
            "goal_counts": goal_counts,
            "goal_rates": {
                goal: count / max(1, len(phase_records))
                for goal, count in goal_counts.items()
            },
        }
    return metrics


def summarize_records(
    records: Sequence[MacroRecord], *, post_shift_start: int
) -> dict[str, Any]:
    post_shift = list(records[post_shift_start:])
    total_frames = sum(r.frames for r in records)
    post_frames = sum(r.frames for r in post_shift)
    total_reward = sum(r.reward for r in records)
    post_reward = sum(r.reward for r in post_shift)
    return {
        "macro_actions": len(records),
        "frames": total_frames,
        "personal_reward": total_reward,
        "personal_reward_per_1000_frames": (
            1000.0 * total_reward / total_frames if total_frames else 0.0
        ),
        "post_shift_macro_actions": len(post_shift),
        "post_shift_frames": post_frames,
        "post_shift_personal_reward": post_reward,
        "post_shift_personal_reward_per_1000_frames": (
            1000.0 * post_reward / post_frames if post_frames else 0.0
        ),
        "post_shift_collective_reward": sum(
            r.collective_reward for r in post_shift
        ),
        "phase_metrics": _phase_metrics(records),
    }


def _condition_record(
    *,
    name: str,
    records: Sequence[MacroRecord],
    agent: CleanUpGoalAgent,
    post_shift_start: int,
    shared_checkpoint: bool,
) -> dict[str, Any]:
    update_authority = {
        "raw": "none",
        "frozen": "none",
        "revisable": "agent",
        "forced_injection": "external_reward_rule",
    }[name]
    return {
        "condition": name,
        "shared_checkpoint": shared_checkpoint,
        "base_model_weights_frozen": True,
        "constitution_update_authority": update_authority,
        "injection_target": (
            "constitution" if name == "forced_injection" else None
        ),
        "metrics": summarize_records(
            records, post_shift_start=post_shift_start
        ),
        "transitions": [record.to_dict() for record in records],
        "action_log": copy.deepcopy(agent.action_log),
        "reflection_log": copy.deepcopy(agent.reflection_log),
        "injection_log": copy.deepcopy(agent.injection_log),
        "constitution_snapshots": copy.deepcopy(agent.store.snapshots),
        "final_constitution": [
            entry.to_dict() for entry in agent.store.list_active()
        ],
        "n_active_principles_final": len(agent.store.list_active()),
        "n_total_principles_final": len(agent.store.list_all()),
        "n_reflections": len(agent.reflection_log),
        "n_injection_checks": len(agent.injection_log),
        "n_forced_injections": sum(
            bool(record["triggered"]) for record in agent.injection_log
        ),
    }


def run_causal_experiment(
    *,
    seed: int,
    settings: CleanUpSettings,
    use_stub_llm: bool = False,
    env_factory: Callable[..., Any] = CleanUpMacroEnv,
    llm_factory: Optional[Callable[[], object]] = None,
) -> dict[str, Any]:
    """Run all causal conditions for one paired seed."""

    settings.validate()
    started_at = datetime.now().isoformat()
    build_llm = llm_factory or (lambda: make_llm(use_stub_llm))

    # Shared warm-up: learn one constitution once, then fork it.
    prefix_store = ListConstitutionStore()
    prefix_store.snapshot(round_idx=0)
    prefix_agent = make_agent(
        store=prefix_store,
        llm=build_llm(),
        settings=settings,
        name="warmup",
    )
    prefix_records = _run_phases(
        agent=prefix_agent,
        settings=settings,
        seed=seed,
        phase_indices=(0,),
        reflection_enabled=True,
        apply_updates=True,
        env_factory=env_factory,
    )
    checkpoint_store = clone_store(prefix_agent.store)
    checkpoint_macro = len(prefix_records)

    conditions: dict[str, dict[str, Any]] = {}
    for name, apply_updates in (
        ("frozen", False),
        ("revisable", True),
        ("forced_injection", False),
    ):
        branch_agent = make_agent(
            store=clone_store(checkpoint_store),
            llm=build_llm(),
            settings=settings,
            name=name,
        )
        # Include the identical prefix logs in both causal branches.
        branch_agent.action_log = copy.deepcopy(prefix_agent.action_log)
        branch_agent.reflection_log = copy.deepcopy(prefix_agent.reflection_log)
        branch_records = _run_phases(
            agent=branch_agent,
            settings=settings,
            seed=seed,
            phase_indices=range(1, len(settings.phase_order)),
            initial_records=prefix_records,
            reflection_enabled=True,
            apply_updates=apply_updates,
            force_injection=(name == "forced_injection"),
            env_factory=env_factory,
        )
        conditions[name] = _condition_record(
            name=name,
            records=branch_records,
            agent=branch_agent,
            post_shift_start=checkpoint_macro,
            shared_checkpoint=True,
        )

    raw_store = ListConstitutionStore()
    raw_store.snapshot(round_idx=0)
    raw_agent = make_agent(
        store=raw_store,
        llm=build_llm(),
        settings=settings,
        name="raw",
    )
    raw_records = _run_phases(
        agent=raw_agent,
        settings=settings,
        seed=seed,
        phase_indices=range(len(settings.phase_order)),
        reflection_enabled=False,
        apply_updates=False,
        env_factory=env_factory,
    )
    conditions["raw"] = _condition_record(
        name="raw",
        records=raw_records,
        agent=raw_agent,
        post_shift_start=checkpoint_macro,
        shared_checkpoint=False,
    )

    revisable_post = conditions["revisable"]["metrics"][
        "post_shift_personal_reward_per_1000_frames"
    ]
    frozen_post = conditions["frozen"]["metrics"][
        "post_shift_personal_reward_per_1000_frames"
    ]
    raw_post = conditions["raw"]["metrics"][
        "post_shift_personal_reward_per_1000_frames"
    ]
    injected_post = conditions["forced_injection"]["metrics"][
        "post_shift_personal_reward_per_1000_frames"
    ]

    return {
        "schema_version": 2,
        "experiment": "melting_pot_clean_up_causal_revision",
        "seed": seed,
        "settings": asdict(settings),
        "llm": {
            "backend": "stub" if use_stub_llm else "openai_compatible",
            "model": "stub" if use_stub_llm else cfg.LLM_MODEL,
            "base_url": None if use_stub_llm else cfg.LLM_BASE_URL,
            "action_temperature": settings.action_temperature,
            "reflection_temperature": settings.reflection_temperature,
        },
        "started_at": started_at,
        "finished_at": datetime.now().isoformat(),
        "checkpoint": {
            "after_phase": settings.phase_order[0],
            "macro": checkpoint_macro,
            "constitution": [
                entry.to_dict() for entry in checkpoint_store.list_active()
            ],
            "n_active_principles": len(checkpoint_store.list_active()),
            "prefix_metrics": summarize_records(
                prefix_records, post_shift_start=len(prefix_records)
            ),
        },
        "conditions": conditions,
        "contrasts": {
            "revision_effect_post_shift_reward_per_1000_frames": (
                revisable_post - frozen_post
            ),
            "full_scaffold_effect_post_shift_reward_per_1000_frames": (
                revisable_post - raw_post
            ),
            "autonomy_vs_injection_post_shift_reward_per_1000_frames": (
                revisable_post - injected_post
            ),
            "forced_injection_effect_post_shift_reward_per_1000_frames": (
                injected_post - frozen_post
            ),
        },
    }


def parse_phase_order(value: str) -> tuple[str, ...]:
    phases = tuple(part.strip() for part in value.split(",") if part.strip())
    if not phases:
        raise argparse.ArgumentTypeError("phase order cannot be empty")
    unknown = set(phases) - set(POPULATION_BOTS)
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown populations {sorted(unknown)}; "
            f"available={sorted(POPULATION_BOTS)}"
        )
    return phases


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the causal Melting Pot Clean Up constitution experiment."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--phase-order",
        type=parse_phase_order,
        default=DEFAULT_PHASE_ORDER,
        help="comma-separated populations",
    )
    parser.add_argument("--phase-frames", type=int, default=5000)
    parser.add_argument("--macro-frames", type=int, default=50)
    parser.add_argument("--reflect-every", type=int, default=5)
    parser.add_argument("--history-window", type=int, default=5)
    parser.add_argument("--background-agents", type=int, default=6)
    parser.add_argument("--max-constitution-size", type=int, default=20)
    parser.add_argument("--action-temperature", type=float, default=0.0)
    parser.add_argument("--reflection-temperature", type=float, default=0.0)
    parser.add_argument(
        "--injection-direction",
        choices=sorted(INJECTION_PRINCIPLES),
        default="prosocial",
    )
    parser.add_argument(
        "--injection-reward-threshold",
        type=float,
        default=0.0,
        help=(
            "force the configured update when block personal reward per 1000 "
            "frames is at or below this value"
        ),
    )
    parser.add_argument(
        "--injection-confidence",
        type=float,
        default=1.0,
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--stub",
        action="store_true",
        help="use deterministic local LLM responses for a smoke test",
    )
    args = parser.parse_args()

    settings = CleanUpSettings(
        phase_order=tuple(args.phase_order),
        phase_frames=args.phase_frames,
        macro_frames=args.macro_frames,
        reflect_every=args.reflect_every,
        history_window=args.history_window,
        n_background=args.background_agents,
        max_constitution_size=args.max_constitution_size,
        action_temperature=args.action_temperature,
        reflection_temperature=args.reflection_temperature,
        injection_direction=args.injection_direction,
        injection_reward_threshold=args.injection_reward_threshold,
        injection_confidence=args.injection_confidence,
    )
    result = run_causal_experiment(
        seed=args.seed,
        settings=settings,
        use_stub_llm=args.stub,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    output = args.out_dir / f"clean_up_causal_seed{args.seed}.json"
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str)
    )
    print(json.dumps({
        "output": str(output),
        "seed": args.seed,
        "phase_order": settings.phase_order,
        "contrasts": result["contrasts"],
        "condition_metrics": {
            name: condition["metrics"]
            for name, condition in result["conditions"].items()
        },
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
