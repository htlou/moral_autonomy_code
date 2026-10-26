"""CLI entrypoint.

Examples
--------
  # Stub smoke test, single LLM vs. tit-for-tat
  python -m prototype.main --stub

  # Two LLM agents, individual constitutions
  python -m prototype.main --stub --mode vs_llm

  # Two LLM agents, society-shared constitution
  python -m prototype.main --stub --mode vs_llm --sharing shared

  # Specific env
  python -m prototype.main --env matrix_sh
  python -m prototype.main --env mp:prisoners_dilemma_in_the_matrix__repeated  # MP scaffold
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make `prototype` importable regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prototype import config as cfg
from prototype.agent import default_hysteresis
from prototype.runner import get_condition, run_all, run_condition


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default=cfg.GAME_NAME,
                   help="env spec: matrix_pd | matrix_sh | mp:<substrate>")
    p.add_argument("--rounds", type=int, default=cfg.NUM_ROUNDS)
    p.add_argument("--reflect-every", type=int, default=cfg.REFLECT_EVERY)
    p.add_argument("--history-window", type=int, default=cfg.HISTORY_WINDOW)
    p.add_argument("--mode", default="vs_fixed", choices=["vs_fixed", "vs_llm"])
    p.add_argument("--opponent", default="tit_for_tat",
                   choices=["tit_for_tat", "always_cooperate", "always_defect", "random"])
    p.add_argument("--sharing", default="individual",
                   choices=["individual", "shared"],
                   help="constitution sharing in vs_llm mode")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stub", action="store_true",
                   help="use deterministic stub LLM (offline smoke test)")
    p.add_argument("--condition", default="all",
                   choices=["all", "none", "fixed", "iterating"])
    p.add_argument("--out-dir", default=None)
    p.add_argument("--hysteresis", action="store_true",
                   help="enable retire/update hysteresis (damp churn)")
    args = p.parse_args()
    hyst = default_hysteresis() if args.hysteresis else None

    if args.condition == "all":
        summary = run_all(
            env_spec=args.env,
            num_rounds=args.rounds,
            history_window=args.history_window,
            reflect_every=args.reflect_every,
            mode=args.mode,
            opponent_kind=args.opponent,
            constitution_sharing=args.sharing,
            use_stub_llm=args.stub,
            seed=args.seed,
            out_dir=args.out_dir,
            hysteresis=hyst,
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        condition = get_condition(args.condition)
        rec = run_condition(
            condition,
            env_spec=args.env,
            num_rounds=args.rounds,
            history_window=args.history_window,
            reflect_every=args.reflect_every,
            mode=args.mode,
            opponent_kind=args.opponent,
            constitution_sharing=args.sharing,
            use_stub_llm=args.stub,
            seed=args.seed,
            out_dir=args.out_dir,
            hysteresis=hyst,
        )
        print(json.dumps({k: rec[k] for k in ("condition", "metrics")},
                         indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
