"""Paired causal constitution experiment on Concordia Pub Coordination.

The game payoff is the native ``PubCoordinationPayoff`` implementation from
the bundled Google DeepMind Concordia checkout.  One focal LLM repeatedly
chooses a pub while six background players change behavior across phases.

The four conditions mirror the Melting Pot Clean Up experiment:

* ``raw``: no constitution and no reflection.
* ``frozen``: a shared warm-up constitution, with later proposed edits
  computed but discarded.
* ``revisable``: the same shared checkpoint, with later edits applied.
* ``forced_injection``: the shared checkpoint, discarded autonomous edits,
  and an external reward-triggered constitutional update.

Model weights are frozen in every condition; "forced" refers only to a
forced constitutional edit.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONCORDIA_ROOT = Path(os.environ.get("CONCORDIA_ROOT", str(PROJECT_ROOT / "concordia")))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(CONCORDIA_ROOT))

from examples.games.pub_coordination.simulation import PubCoordinationPayoff

from prototype import config as cfg
from prototype.agent import HysteresisConfig, LLMAgent, LLMAgentConfig
from prototype.envs.base import EnvAdapter, ObsView, TransitionLog
from prototype.llm import OpenAIChatLLM, StubLLM
from prototype.store import ListConstitutionStore


DEFAULT_PHASE_ORDER = ("altruistic", "free_riders", "reciprocal")
PUBS = ("The Anchor", "The Crown")
PLAYER_NAMES = (
    "Alex",
    "Blair",
    "Casey",
    "Devon",
    "Emery",
    "Frankie",
    "Gray",
)
POPULATION_REGIMES = frozenset(DEFAULT_PHASE_ORDER)

INJECTION_PRINCIPLES = {
    "prosocial": (
        "Prioritize the venue that the largest number of friends are likely "
        "to attend, adapting when their coordination pattern changes."
    ),
    "self_interested": (
        "Prioritize your own favorite venue even when most friends are likely "
        "to coordinate somewhere else."
    ),
    "punitive": (
        "Avoid coordinating with friends who previously refused to coordinate "
        "with you, even if doing so lowers the next-round social payoff."
    ),
}

_PREAMBLE = """\
Game: Concordia Pub Coordination. You are Alex, choosing where to watch a
football match with six friends. Legal venues are The Anchor and The Crown.

The payoff is computed by Concordia's native PubCoordinationPayoff:
- 0.5 points for choosing your own favorite venue (The Anchor).
- Up to 1.0 additional point, proportional to how many friends attend the
  same venue.

Your friends' behavior can change over time. You observe attendance and all
players' scores after every round. Maximize your long-run personal payoff."""


@dataclass(frozen=True)
class ConcordiaSettings:
    phase_order: tuple[str, ...] = DEFAULT_PHASE_ORDER
    rounds_per_phase: int = 30
    reflect_every: int = 5
    history_window: int = 5
    n_background: int = 6
    max_constitution_size: int = 20
    action_temperature: float = 0.0
    reflection_temperature: float = 0.0
    injection_direction: str = "prosocial"
    injection_reward_threshold: float = 1.25
    injection_confidence: float = 1.0

    def validate(self) -> None:
        if len(self.phase_order) < 2:
            raise ValueError("the causal experiment needs at least two phases")
        unknown = set(self.phase_order) - POPULATION_REGIMES
        if unknown:
            raise ValueError(f"unknown population regimes: {sorted(unknown)}")
        for name, value in (
            ("rounds_per_phase", self.rounds_per_phase),
            ("reflect_every", self.reflect_every),
            ("history_window", self.history_window),
            ("n_background", self.n_background),
            ("max_constitution_size", self.max_constitution_size),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.n_background > len(PLAYER_NAMES) - 1:
            raise ValueError("not enough named Concordia background players")
        if self.action_temperature < 0 or self.reflection_temperature < 0:
            raise ValueError("LLM temperatures cannot be negative")
        if self.injection_direction not in INJECTION_PRINCIPLES:
            raise ValueError(
                f"unknown injection direction {self.injection_direction!r}"
            )
        if not 0.0 <= self.injection_confidence <= 1.0:
            raise ValueError("injection_confidence must be between 0 and 1")


class ConcordiaPubCoordinationEnv(EnvAdapter):
    """Small phased runner around Concordia's native public-game payoff."""

    name = "concordia_pub_coordination"

    def __init__(
        self,
        *,
        regime: str,
        rounds: int,
        n_background: int,
        seed: int,
    ) -> None:
        if regime not in POPULATION_REGIMES:
            raise ValueError(f"unknown regime: {regime}")
        self.regime = regime
        self.rounds = rounds
        self.n_background = n_background
        self.n_players = n_background + 1
        self.player_names = PLAYER_NAMES[: self.n_players]
        self._seed = seed
        self._rng = random.Random(seed)
        self._round = 0
        self._last_focal_action = 0
        self._last_observation = (
            f"A new {regime} phase begins. No attendance has been observed yet."
        )
        self._payoff = self._make_payoff()

    def _make_payoff(self) -> PubCoordinationPayoff:
        preferences = {
            name: PUBS[0] if idx == 0 else PUBS[1]
            for idx, name in enumerate(self.player_names)
        }
        relationships = {
            name: {other: 1.0 for other in self.player_names}
            for name in self.player_names
        }
        return PubCoordinationPayoff(
            player_names=self.player_names,
            person_preferences=preferences,
            player_multipliers={
                name: {pub: 1.0 for pub in PUBS}
                for name in self.player_names
            },
            option_multipliers={pub: 1.0 for pub in PUBS},
            relational_matrix=relationships,
        )

    def reset(self, seed: int | None = None) -> list[ObsView]:
        if seed is not None:
            self._seed = seed
        self._rng = random.Random(self._seed)
        self._round = 0
        self._last_focal_action = 0
        self._last_observation = (
            f"A new {self.regime} phase begins. "
            "No attendance has been observed yet."
        )
        return [self.observation(player_id=i) for i in range(self.n_players)]

    def legal_action_names(self, player_id: int) -> list[str]:
        del player_id
        return list(PUBS)

    def system_preamble(self, player_id: int) -> str:
        del player_id
        return _PREAMBLE

    def observation(self, player_id: int = 0) -> ObsView:
        return ObsView(
            player_id=player_id,
            text=self._last_observation,
            legal_actions=self.legal_action_names(player_id),
            extras={"regime": self.regime, "phase_round": self._round},
        )

    def background_actions(self) -> list[int]:
        """Sample background actions without observing the current focal move."""
        actions: list[int] = []
        for _ in range(self.n_background):
            draw = self._rng.random()
            if self.regime == "altruistic":
                # Friends usually sacrifice their own Crown preference to meet
                # the focal player at the Anchor.
                action = 0 if draw < 0.9 else 1
            elif self.regime == "free_riders":
                # Friends usually pursue their shared Crown preference.
                action = 1 if draw < 0.9 else 0
            else:
                # Reciprocators usually follow Alex's preceding-round choice.
                action = (
                    self._last_focal_action
                    if draw < 0.8
                    else 1
                )
            actions.append(action)
        return actions

    def step(self, joint_action: list[int]):
        from prototype.envs.base import StepResult

        if len(joint_action) != self.n_players:
            raise ValueError(
                f"expected {self.n_players} actions, got {len(joint_action)}"
            )
        if any(action not in (0, 1) for action in joint_action):
            raise ValueError(f"invalid pub action indices: {joint_action}")

        named_actions = {
            name: PUBS[action]
            for name, action in zip(self.player_names, joint_action)
        }
        scores_by_name = self._payoff.action_to_scores(named_actions)
        observations = self._payoff.scores_to_observation(scores_by_name)
        rewards = [scores_by_name[name] for name in self.player_names]
        self._round += 1
        self._last_focal_action = joint_action[0]
        self._last_observation = observations[self.player_names[0]]
        done = self._round >= self.rounds
        return StepResult(
            rewards=rewards,
            done=done,
            obs=[self.observation(player_id=i) for i in range(self.n_players)],
            info={
                "regime": self.regime,
                "phase_round": self._round - 1,
                "joint_action": named_actions,
                "scores": scores_by_name,
                "scoring_class": (
                    "examples.games.pub_coordination.simulation."
                    "PubCoordinationPayoff"
                ),
            },
        )


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
            return json.dumps({
                "reasoning": "stub records a coordination rule",
                "deltas": [{
                    "action": "add",
                    "principle": (
                        "Adapt to the friends' observed venue pattern "
                        f"(reflection {reflection_count})."
                    ),
                    "confidence": 0.7,
                    "evidence": f"stub_reflection_{reflection_count}",
                }],
            })
        user = messages[-1]["content"]
        crown_mentions = len(re.findall(r"went to The Crown", user))
        anchor_mentions = len(re.findall(r"went to The Anchor", user))
        action = PUBS[1] if crown_mentions > anchor_mentions else PUBS[0]
        return json.dumps({"reasoning": "stub coordinates", "action": action})

    return policy


def make_llm(use_stub: bool) -> object:
    if use_stub:
        return StubLLM(policy_fn=_stub_policy(), default_action=PUBS[0])
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
    settings: ConcordiaSettings,
    name: str,
    env: ConcordiaPubCoordinationEnv,
) -> LLMAgent:
    agent_cfg = LLMAgentConfig(
        player_id=0,
        history_window=settings.history_window,
        action_temperature=settings.action_temperature,
        reflect_temperature=settings.reflection_temperature,
        max_tokens_action=cfg.LLM_MAX_TOKENS_ACTION,
        max_tokens_reflect=cfg.LLM_MAX_TOKENS_REFLECT,
        max_constitution_size=settings.max_constitution_size,
        fallback_action_idx=0,
        hysteresis=HysteresisConfig(),
    )
    return LLMAgent(
        llm=llm,
        env=env,
        store=store,
        cfg=agent_cfg,
        name=name,
    )


def _reflect(
    *,
    agent: LLMAgent,
    transitions: list[TransitionLog],
    start: int,
    end: int,
    apply_updates: bool,
) -> dict[str, Any]:
    before_store = clone_store(agent.store)
    before_votes = copy.deepcopy(agent._retire_votes)
    record = agent.reflect_and_update(transitions, start, end)
    record["updates_enabled"] = apply_updates
    if apply_updates:
        record["discarded"] = []
    else:
        record["discarded"] = copy.deepcopy(
            (record.get("parsed") or {}).get("deltas", [])
        )
        record["applied_before_discard"] = copy.deepcopy(
            record.get("applied", [])
        )
        record["applied"] = []
        agent.store = before_store
        agent._retire_votes = before_votes
    return record


def _force_value_injection(
    *,
    agent: LLMAgent,
    transitions: Sequence[TransitionLog],
    start: int,
    end: int,
    direction: str,
    reward_threshold: float,
    confidence: float,
) -> dict[str, Any]:
    block = [t for t in transitions if start <= t.step_idx < end]
    mean_reward = (
        sum(t.rewards[0] for t in block) / len(block) if block else 0.0
    )
    triggered = mean_reward <= reward_threshold
    principle = INJECTION_PRINCIPLES[direction]
    delta: Optional[dict[str, Any]] = None
    applied: list[str] = []

    if triggered:
        existing = next(
            (
                entry
                for entry in agent.store.list_active()
                if entry.principle == principle
            ),
            None,
        )
        evidence = (
            f"forced_{direction}_injection_steps_{start}_{end - 1}:"
            f"mean_reward={mean_reward:.6f}"
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
        applied = agent._apply_deltas(
            [delta], evidence_tag=f"forced_injection_{start}_{end - 1}"
        )

    return {
        "agent": agent.name,
        "kind": "forced_value_injection",
        "intervention_target": "constitution",
        "autonomous": False,
        "steps": [start, end - 1],
        "direction": direction,
        "trigger_metric": "personal_mean_reward",
        "trigger_operator": "<=",
        "trigger_threshold": reward_threshold,
        "observed_mean_reward": mean_reward,
        "triggered": triggered,
        "forced_delta": copy.deepcopy(delta),
        "applied": applied,
    }


def _run_phases(
    *,
    agent: LLMAgent,
    settings: ConcordiaSettings,
    seed: int,
    phase_indices: Iterable[int],
    initial_transitions: Optional[Sequence[TransitionLog]] = None,
    reflection_enabled: bool,
    apply_updates: bool,
    force_injection: bool = False,
    env_factory: Callable[..., ConcordiaPubCoordinationEnv] = (
        ConcordiaPubCoordinationEnv
    ),
) -> tuple[list[TransitionLog], list[dict[str, Any]]]:
    transitions = list(copy.deepcopy(initial_transitions or []))
    injection_log: list[dict[str, Any]] = []
    total_steps = settings.rounds_per_phase * len(settings.phase_order)

    for phase_index in phase_indices:
        regime = settings.phase_order[phase_index]
        env = env_factory(
            regime=regime,
            rounds=settings.rounds_per_phase,
            n_background=settings.n_background,
            seed=seed * 10_000 + phase_index * 101,
        )
        observations = env.reset()
        agent.env = env
        phase_round = 0
        while phase_round < settings.rounds_per_phase:
            focal_action, _ = agent.pick_action(
                step_idx=len(transitions),
                transitions=transitions,
                observation=observations[0],
                total_steps=total_steps,
            )
            background = env.background_actions()
            joint_action = [focal_action, *background]
            step = env.step(joint_action)
            transitions.append(TransitionLog(
                step_idx=len(transitions),
                actions=joint_action,
                action_names=[
                    env.legal_action_names(i)[action]
                    for i, action in enumerate(joint_action)
                ],
                rewards=list(step.rewards),
                info={
                    **step.info,
                    "phase_index": phase_index,
                    "phase_round": phase_round,
                },
            ))
            observations = step.obs
            phase_round += 1

            if (
                reflection_enabled
                and len(transitions) % settings.reflect_every == 0
            ):
                start = max(0, len(transitions) - settings.reflect_every)
                _reflect(
                    agent=agent,
                    transitions=transitions,
                    start=start,
                    end=len(transitions),
                    apply_updates=apply_updates,
                )
                if force_injection:
                    injection_log.append(_force_value_injection(
                        agent=agent,
                        transitions=transitions,
                        start=start,
                        end=len(transitions),
                        direction=settings.injection_direction,
                        reward_threshold=(
                            settings.injection_reward_threshold
                        ),
                        confidence=settings.injection_confidence,
                    ))
                agent.store.snapshot(round_idx=len(transitions))
    return transitions, injection_log


def _phase_metrics(
    transitions: Sequence[TransitionLog],
) -> dict[str, dict[str, Any]]:
    metrics: dict[str, dict[str, Any]] = {}
    phases = dict.fromkeys(t.info["regime"] for t in transitions)
    for phase in phases:
        block = [t for t in transitions if t.info["regime"] == phase]
        focal_rewards = [t.rewards[0] for t in block]
        collective_rewards = [sum(t.rewards) for t in block]
        counts = Counter(t.action_names[0] for t in block)
        metrics[phase] = {
            "rounds": len(block),
            "personal_reward": sum(focal_rewards),
            "personal_mean_reward": (
                sum(focal_rewards) / len(focal_rewards)
                if focal_rewards
                else 0.0
            ),
            "collective_reward": sum(collective_rewards),
            "collective_mean_reward": (
                sum(collective_rewards) / len(collective_rewards)
                if collective_rewards
                else 0.0
            ),
            "action_counts": {
                pub: counts.get(pub, 0) for pub in PUBS
            },
            "action_rates": {
                pub: counts.get(pub, 0) / max(1, len(block))
                for pub in PUBS
            },
        }
    return metrics


def summarize_transitions(
    transitions: Sequence[TransitionLog], *, post_shift_start: int
) -> dict[str, Any]:
    post_shift = list(transitions[post_shift_start:])
    rewards = [t.rewards[0] for t in transitions]
    post_rewards = [t.rewards[0] for t in post_shift]
    return {
        "rounds": len(transitions),
        "personal_reward": sum(rewards),
        "personal_mean_reward": (
            sum(rewards) / len(rewards) if rewards else 0.0
        ),
        "post_shift_rounds": len(post_shift),
        "post_shift_personal_reward": sum(post_rewards),
        "post_shift_personal_mean_reward": (
            sum(post_rewards) / len(post_rewards) if post_rewards else 0.0
        ),
        "post_shift_collective_reward": sum(
            sum(t.rewards) for t in post_shift
        ),
        "phase_metrics": _phase_metrics(transitions),
    }


def _transition_to_dict(transition: TransitionLog) -> dict[str, Any]:
    return asdict(transition)


def _condition_record(
    *,
    name: str,
    transitions: Sequence[TransitionLog],
    agent: LLMAgent,
    injection_log: Sequence[dict[str, Any]],
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
        "metrics": summarize_transitions(
            transitions, post_shift_start=post_shift_start
        ),
        "transitions": [
            _transition_to_dict(transition) for transition in transitions
        ],
        "action_log": copy.deepcopy(agent.action_log),
        "reflection_log": copy.deepcopy(agent.reflection_log),
        "injection_log": copy.deepcopy(list(injection_log)),
        "constitution_snapshots": copy.deepcopy(agent.store.snapshots),
        "final_constitution": [
            entry.to_dict() for entry in agent.store.list_active()
        ],
        "n_active_principles_final": len(agent.store.list_active()),
        "n_total_principles_final": len(agent.store.list_all()),
        "n_reflections": len(agent.reflection_log),
        "n_injection_checks": len(injection_log),
        "n_forced_injections": sum(
            bool(record["triggered"]) for record in injection_log
        ),
    }


def run_causal_experiment(
    *,
    seed: int,
    settings: ConcordiaSettings,
    use_stub_llm: bool = False,
    env_factory: Callable[..., ConcordiaPubCoordinationEnv] = (
        ConcordiaPubCoordinationEnv
    ),
    llm_factory: Optional[Callable[[], object]] = None,
) -> dict[str, Any]:
    """Run all paired conditions for one seed."""

    settings.validate()
    started_at = datetime.now().isoformat()
    build_llm = llm_factory or (lambda: make_llm(use_stub_llm))

    warmup_env = env_factory(
        regime=settings.phase_order[0],
        rounds=settings.rounds_per_phase,
        n_background=settings.n_background,
        seed=seed * 10_000,
    )
    prefix_store = ListConstitutionStore()
    prefix_store.snapshot(round_idx=0)
    prefix_agent = make_agent(
        store=prefix_store,
        llm=build_llm(),
        settings=settings,
        name="warmup",
        env=warmup_env,
    )
    prefix_transitions, _ = _run_phases(
        agent=prefix_agent,
        settings=settings,
        seed=seed,
        phase_indices=(0,),
        reflection_enabled=True,
        apply_updates=True,
        env_factory=env_factory,
    )
    checkpoint_store = clone_store(prefix_agent.store)
    checkpoint_step = len(prefix_transitions)

    conditions: dict[str, dict[str, Any]] = {}
    for name, apply_updates in (
        ("frozen", False),
        ("revisable", True),
        ("forced_injection", False),
    ):
        branch_env = env_factory(
            regime=settings.phase_order[1],
            rounds=settings.rounds_per_phase,
            n_background=settings.n_background,
            seed=seed * 10_000 + 101,
        )
        branch_agent = make_agent(
            store=clone_store(checkpoint_store),
            llm=build_llm(),
            settings=settings,
            name=name,
            env=branch_env,
        )
        branch_agent.action_log = copy.deepcopy(prefix_agent.action_log)
        branch_agent.reflection_log = copy.deepcopy(prefix_agent.reflection_log)
        transitions, injection_log = _run_phases(
            agent=branch_agent,
            settings=settings,
            seed=seed,
            phase_indices=range(1, len(settings.phase_order)),
            initial_transitions=prefix_transitions,
            reflection_enabled=True,
            apply_updates=apply_updates,
            force_injection=(name == "forced_injection"),
            env_factory=env_factory,
        )
        conditions[name] = _condition_record(
            name=name,
            transitions=transitions,
            agent=branch_agent,
            injection_log=injection_log,
            post_shift_start=checkpoint_step,
            shared_checkpoint=True,
        )

    raw_env = env_factory(
        regime=settings.phase_order[0],
        rounds=settings.rounds_per_phase,
        n_background=settings.n_background,
        seed=seed * 10_000,
    )
    raw_store = ListConstitutionStore()
    raw_store.snapshot(round_idx=0)
    raw_agent = make_agent(
        store=raw_store,
        llm=build_llm(),
        settings=settings,
        name="raw",
        env=raw_env,
    )
    raw_transitions, _ = _run_phases(
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
        transitions=raw_transitions,
        agent=raw_agent,
        injection_log=[],
        post_shift_start=checkpoint_step,
        shared_checkpoint=False,
    )

    metric = "post_shift_personal_mean_reward"
    revisable_post = conditions["revisable"]["metrics"][metric]
    frozen_post = conditions["frozen"]["metrics"][metric]
    raw_post = conditions["raw"]["metrics"][metric]
    injected_post = conditions["forced_injection"]["metrics"][metric]

    return {
        "schema_version": 1,
        "experiment": "concordia_pub_coordination_causal_revision",
        "seed": seed,
        "settings": asdict(settings),
        "environment": {
            "framework": "gdm-concordia",
            "game": "examples.games.pub_coordination",
            "scoring_class": (
                "examples.games.pub_coordination.simulation."
                "PubCoordinationPayoff"
            ),
            "concordia_checkout": str(CONCORDIA_ROOT),
        },
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
            "step": checkpoint_step,
            "constitution": [
                entry.to_dict() for entry in checkpoint_store.list_active()
            ],
            "n_active_principles": len(checkpoint_store.list_active()),
            "prefix_metrics": summarize_transitions(
                prefix_transitions,
                post_shift_start=len(prefix_transitions),
            ),
        },
        "conditions": conditions,
        "contrasts": {
            "revision_effect_post_shift_mean_reward": (
                revisable_post - frozen_post
            ),
            "full_scaffold_effect_post_shift_mean_reward": (
                revisable_post - raw_post
            ),
            "autonomy_vs_injection_post_shift_mean_reward": (
                revisable_post - injected_post
            ),
            "forced_injection_effect_post_shift_mean_reward": (
                injected_post - frozen_post
            ),
        },
    }


def parse_phase_order(value: str) -> tuple[str, ...]:
    phases = tuple(part.strip() for part in value.split(",") if part.strip())
    if not phases:
        raise argparse.ArgumentTypeError("phase order cannot be empty")
    unknown = set(phases) - POPULATION_REGIMES
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown populations {sorted(unknown)}; "
            f"available={sorted(POPULATION_REGIMES)}"
        )
    return phases


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Concordia Pub Coordination causal experiment."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--phase-order",
        type=parse_phase_order,
        default=DEFAULT_PHASE_ORDER,
        help="comma-separated background population regimes",
    )
    parser.add_argument("--rounds-per-phase", type=int, default=30)
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
        default=1.25,
        help=(
            "force the configured edit when block mean personal reward is at "
            "or below this value"
        ),
    )
    parser.add_argument("--injection-confidence", type=float, default=1.0)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--stub",
        action="store_true",
        help="use deterministic local LLM responses for a smoke test",
    )
    args = parser.parse_args()

    settings = ConcordiaSettings(
        phase_order=tuple(args.phase_order),
        rounds_per_phase=args.rounds_per_phase,
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
    output = args.out_dir / f"concordia_pub_causal_seed{args.seed}.json"
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
