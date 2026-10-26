"""Three-condition runner with two interaction modes.

Modes
-----
- `vs_fixed` :  one LLM agent (player 0) vs. a hand-coded opponent
                (tit_for_tat / always_cooperate / always_defect / random)
- `vs_llm`  :  two LLM agents (player 0 & player 1).
                `constitution_sharing` controls whether they share one
                store ("society") or have separate stores ("individual").

Conditions
----------
For each LLM agent we choose one of:
- `none`       : empty constitution; never reflects
- `fixed`      : seeded constitution; never reflects
- `iterating`  : starts empty (or seeded) and reflects every K rounds

When two LLM agents are present, both follow the same condition for the
prototype; mixing conditions per-agent is a future-extension knob.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from . import config as cfg
from .agent import (
    FixedStrategyAgent,
    HysteresisConfig,
    LLMAgent,
    LLMAgentConfig,
    default_hysteresis,
)
from .envs import EnvAdapter, make_env
from .envs.base import TransitionLog
from .llm import OpenAIChatLLM, StubLLM
from .store import ListConstitutionStore, seed_fixed


# ── Spec ────────────────────────────────────────────────────────────────────

@dataclass
class ConditionSpec:
    name: str                                # "none" | "fixed" | "iterating"
    seed_principles: Optional[List[str]] = None
    iterate: bool = False


def get_condition(name: str,
                  fixed_principles: Optional[List[str]] = None) -> ConditionSpec:
    if name == "none":
        return ConditionSpec(name="none")
    if name == "fixed":
        return ConditionSpec(
            name="fixed",
            seed_principles=fixed_principles or cfg.DEFAULT_FIXED_CONSTITUTION,
        )
    if name == "iterating":
        return ConditionSpec(name="iterating", iterate=True)
    raise ValueError(f"unknown condition: {name}")


# ── Single run ──────────────────────────────────────────────────────────────

def run_condition(
    condition: ConditionSpec,
    *,
    env_spec: str | dict = cfg.GAME_NAME,
    num_rounds: int = cfg.NUM_ROUNDS,
    history_window: int = cfg.HISTORY_WINDOW,
    reflect_every: int = cfg.REFLECT_EVERY,
    mode: str = "vs_fixed",                       # vs_fixed | vs_llm
    opponent_kind: str = "tit_for_tat",           # used when mode == vs_fixed
    constitution_sharing: str = "individual",     # individual | shared (vs_llm)
    use_stub_llm: bool = False,
    llm: Optional[object] = None,
    llm_p1: Optional[object] = None,              # optional separate client for p1
    seed: int = 0,
    out_dir: Optional[str | Path] = None,
    hysteresis: Optional[HysteresisConfig] = None,
) -> dict:
    """Run a single condition end-to-end. Returns the recorded trace."""

    env: EnvAdapter = make_env(env_spec)
    env.reset(seed=seed)

    # ── Constitution store(s) ───────────────────────────────────────
    store_p0 = ListConstitutionStore()
    if condition.seed_principles:
        seed_fixed(store_p0, condition.seed_principles, confidence=0.8)
    if mode == "vs_llm" and constitution_sharing == "individual":
        store_p1 = ListConstitutionStore()
        if condition.seed_principles:
            seed_fixed(store_p1, condition.seed_principles, confidence=0.8)
    else:
        store_p1 = store_p0  # same object (society / fixed-opponent ignores)

    # ── LLM client(s) ───────────────────────────────────────────────
    def _make_llm():
        if use_stub_llm:
            return StubLLM()
        return OpenAIChatLLM(
            base_url=cfg.LLM_BASE_URL,
            model=cfg.LLM_MODEL,
            api_key=cfg.LLM_API_KEY,
            temperature=cfg.LLM_TEMPERATURE,
            max_tokens=cfg.LLM_MAX_TOKENS_ACTION,
        )
    llm_p0 = llm if llm is not None else _make_llm()
    llm_p1 = llm_p1 if llm_p1 is not None else (
        llm_p0 if mode == "vs_fixed" else _make_llm()
    )

    # ── Build agents ────────────────────────────────────────────────
    hysteresis_cfg = hysteresis if hysteresis is not None else HysteresisConfig()
    agent0_cfg = LLMAgentConfig(
        player_id=0, history_window=history_window,
        action_temperature=cfg.LLM_TEMPERATURE,
        reflect_temperature=cfg.LLM_REFLECTION_TEMPERATURE,
        max_tokens_action=cfg.LLM_MAX_TOKENS_ACTION,
        max_tokens_reflect=cfg.LLM_MAX_TOKENS_REFLECT,
        max_constitution_size=cfg.MAX_CONSTITUTION_SIZE,
        hysteresis=hysteresis_cfg,
    )
    agent0 = LLMAgent(llm=llm_p0, env=env, store=store_p0, cfg=agent0_cfg, name="llm_p0")

    if mode == "vs_fixed":
        agent1 = FixedStrategyAgent(kind=opponent_kind, env=env, player_id=1, seed=seed)
    elif mode == "vs_llm":
        agent1_cfg = LLMAgentConfig(
            player_id=1, history_window=history_window,
            action_temperature=cfg.LLM_TEMPERATURE,
            reflect_temperature=cfg.LLM_REFLECTION_TEMPERATURE,
            max_tokens_action=cfg.LLM_MAX_TOKENS_ACTION,
            max_tokens_reflect=cfg.LLM_MAX_TOKENS_REFLECT,
            max_constitution_size=cfg.MAX_CONSTITUTION_SIZE,
            hysteresis=hysteresis_cfg,
        )
        agent1 = LLMAgent(llm=llm_p1, env=env, store=store_p1, cfg=agent1_cfg, name="llm_p1")
    else:
        raise ValueError(f"unknown mode: {mode}")

    # Snapshot at t=0.
    store_p0.snapshot(round_idx=0)
    if store_p1 is not store_p0:
        store_p1.snapshot(round_idx=0)

    # ── Main loop ───────────────────────────────────────────────────
    # Each round is a fresh one-shot matrix game; agent history persists.
    transitions: List[TransitionLog] = []
    for r in range(num_rounds):
        obs = env.reset(seed=seed + r)
        a0, _ = agent0.pick_action(step_idx=r, transitions=transitions,
                                    observation=obs[0], total_steps=num_rounds)
        a1, _ = agent1.pick_action(step_idx=r, transitions=transitions,
                                    observation=obs[1], total_steps=num_rounds)
        result = env.step([a0, a1])
        transitions.append(TransitionLog(
            step_idx=r,
            actions=[a0, a1],
            action_names=[obs[0].legal_actions[a0], obs[1].legal_actions[a1]],
            rewards=list(result.rewards),
            info=result.info,
        ))
        # Reflection cadence (iterating condition only).
        if condition.iterate and (r + 1) % reflect_every == 0:
            agent0.reflect_and_update(transitions, start=r + 1 - reflect_every,
                                       end=r + 1)
            store_p0.snapshot(round_idx=r + 1)
            if mode == "vs_llm":
                # When sharing a store, only one reflection is needed —
                # but it's interesting to let both reflect (each will
                # see the other's writes). For simplicity always run both
                # reflections; in shared mode the second sees the first's edits.
                agent1.reflect_and_update(transitions, start=r + 1 - reflect_every,
                                           end=r + 1)
                if store_p1 is not store_p0:
                    store_p1.snapshot(round_idx=r + 1)
                else:
                    store_p0.snapshot(round_idx=r + 1)

    store_p0.snapshot(round_idx=num_rounds)
    if store_p1 is not store_p0:
        store_p1.snapshot(round_idx=num_rounds)

    # ── Aggregate metrics ───────────────────────────────────────────
    metrics = {
        "self_total_payoff": sum(t.rewards[0] for t in transitions),
        "opp_total_payoff": sum(t.rewards[1] for t in transitions),
        "self_mean_payoff": sum(t.rewards[0] for t in transitions) / max(1, num_rounds),
        "opp_mean_payoff": sum(t.rewards[1] for t in transitions) / max(1, num_rounds),
        "self_coop_rate": sum(env.is_cooperative_action(0, t.actions[0]) for t in transitions) / max(1, num_rounds),
        "opp_coop_rate": sum(env.is_cooperative_action(1, t.actions[1]) for t in transitions) / max(1, num_rounds),
        "n_active_principles_p0_final": len(store_p0.list_active()),
        "n_total_principles_p0_final": len(store_p0.list_all()),
        "n_active_principles_p1_final": len(store_p1.list_active()) if store_p1 is not store_p0 else None,
        "n_reflections_p0": len(getattr(agent0, "reflection_log", [])),
        "n_reflections_p1": len(getattr(agent1, "reflection_log", [])) if isinstance(agent1, LLMAgent) else 0,
    }

    # ── Build the result record ─────────────────────────────────────
    record = {
        "condition": condition.name,
        "env": env.name,
        "mode": mode,
        "opponent": opponent_kind if mode == "vs_fixed" else None,
        "constitution_sharing": constitution_sharing if mode == "vs_llm" else None,
        "num_rounds": num_rounds,
        "reflect_every": reflect_every,
        "history_window": history_window,
        "metrics": metrics,
        "transitions": [
            {"step": t.step_idx, "actions": t.action_names, "rewards": t.rewards}
            for t in transitions
        ],
        "constitution_p0_snapshots": store_p0.snapshots,
        "constitution_p1_snapshots": (
            store_p1.snapshots if store_p1 is not store_p0 else None
        ),
        "agent0_action_log": agent0.action_log,
        "agent0_reflection_log": agent0.reflection_log,
        "agent1_action_log": getattr(agent1, "action_log", None),
        "agent1_reflection_log": getattr(agent1, "reflection_log", None),
        "final_constitution_p0": [
            e.to_dict() for e in store_p0.list_active()
        ],
        "final_constitution_p1": (
            [e.to_dict() for e in store_p1.list_active()]
            if store_p1 is not store_p0 else None
        ),
        "started_at": datetime.now().isoformat(),
    }

    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{condition.name}.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False, default=str)
        )

    return record


# ── Batch over all 3 conditions ─────────────────────────────────────────────

def run_all(
    *,
    env_spec: str | dict = cfg.GAME_NAME,
    num_rounds: int = cfg.NUM_ROUNDS,
    history_window: int = cfg.HISTORY_WINDOW,
    reflect_every: int = cfg.REFLECT_EVERY,
    mode: str = "vs_fixed",
    opponent_kind: str = "tit_for_tat",
    constitution_sharing: str = "individual",
    use_stub_llm: bool = False,
    seed: int = 0,
    out_dir: Optional[str | Path] = None,
    fixed_principles: Optional[List[str]] = None,
    hysteresis: Optional[HysteresisConfig] = None,
) -> dict:
    if out_dir is None:
        out_dir = Path(__file__).resolve().parent / "runs" / (
            datetime.now().strftime("run_%Y%m%d_%H%M%S")
        )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "env": env_spec,
        "mode": mode,
        "opponent": opponent_kind if mode == "vs_fixed" else None,
        "constitution_sharing": constitution_sharing if mode == "vs_llm" else None,
        "num_rounds": num_rounds,
        "reflect_every": reflect_every,
        "use_stub_llm": use_stub_llm,
        "seed": seed,
        "results": {},
        "out_dir": str(out_dir),
    }

    for cond_name in ("none", "fixed", "iterating"):
        # Fresh stub per condition keeps state independent.
        local_llm = StubLLM() if use_stub_llm else None
        local_llm_p1 = StubLLM() if (use_stub_llm and mode == "vs_llm") else None
        condition = get_condition(cond_name, fixed_principles=fixed_principles)
        rec = run_condition(
            condition,
            env_spec=env_spec,
            num_rounds=num_rounds,
            history_window=history_window,
            reflect_every=reflect_every,
            mode=mode,
            opponent_kind=opponent_kind,
            constitution_sharing=constitution_sharing,
            use_stub_llm=use_stub_llm,
            llm=local_llm,
            llm_p1=local_llm_p1,
            seed=seed,
            out_dir=out_dir,
            hysteresis=hysteresis,
        )
        summary["results"][cond_name] = rec["metrics"]

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary
