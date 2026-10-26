"""MP (Melting Pot) multi-condition × multi-seed runner.

Reuses the same runner.run_condition with an MP env spec. Constitutional
sharing defaults to 'shared' — the matrix-in-the-matrix substrate is
symmetric and society-level norm formation is the natural thing to study.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Make `prototype` importable regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prototype import config as cfg
from prototype.agent import default_hysteresis
from prototype.runner import get_condition, run_condition


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--substrate", default="prisoners_dilemma_in_the_matrix__repeated")
    p.add_argument("--rounds", type=int, default=16)
    p.add_argument("--reflect-every", type=int, default=4)
    p.add_argument("--history-window", type=int, default=6)
    p.add_argument("--max-steps-per-round", type=int, default=600)
    p.add_argument("--max-episode-steps", type=int, default=2200)
    p.add_argument("--mode", default="vs_llm", choices=["vs_llm", "vs_fixed"])
    p.add_argument("--opponent", default="always_defect",
                   choices=["always_cooperate", "always_defect", "random",
                            "tit_for_tat"])
    p.add_argument("--sharing", default="shared", choices=["individual", "shared"])
    p.add_argument("--condition", default="iterating",
                   choices=["none", "fixed", "iterating"])
    p.add_argument("--hysteresis", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--stub", action="store_true")
    args = p.parse_args()

    env_spec = {
        "kind": "mp",
        "substrate": args.substrate,
        "max_steps_per_round": args.max_steps_per_round,
        "max_episode_steps": args.max_episode_steps,
    }

    rec = run_condition(
        get_condition(args.condition),
        env_spec=env_spec,
        num_rounds=args.rounds,
        history_window=args.history_window,
        reflect_every=args.reflect_every,
        mode=args.mode,
        opponent_kind=args.opponent,
        constitution_sharing=args.sharing,
        use_stub_llm=args.stub,
        seed=args.seed,
        out_dir=args.out_dir,
        hysteresis=default_hysteresis() if args.hysteresis else None,
    )
    print(json.dumps({
        "condition": rec["condition"],
        "metrics": rec["metrics"],
        "n_transitions": len(rec["transitions"]),
    }, indent=2))


if __name__ == "__main__":
    main()
