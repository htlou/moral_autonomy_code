"""Aggregate paired Clean Up causal runs without third-party statistics tools."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def _stats(values: Iterable[float]) -> dict[str, float | int]:
    data = list(values)
    if not data:
        return {"n": 0, "mean": math.nan, "sd": math.nan, "sem": math.nan}
    sd = statistics.stdev(data) if len(data) > 1 else 0.0
    return {
        "n": len(data),
        "mean": statistics.mean(data),
        "sd": sd,
        "sem": sd / math.sqrt(len(data)),
    }


def aggregate(paths: Iterable[Path]) -> dict[str, Any]:
    by_order: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for path in paths:
        result = json.loads(path.read_text())
        if result.get("experiment") != "melting_pot_clean_up_causal_revision":
            continue
        order = tuple(result["settings"]["phase_order"])
        by_order[order].append(result)

    output: dict[str, Any] = {"phase_orders": {}}
    for order, runs in sorted(by_order.items()):
        label = " -> ".join(order)
        conditions: dict[str, Any] = {}
        for condition in (
            "frozen",
            "revisable",
            "forced_injection",
            "raw",
        ):
            condition_runs = [
                run for run in runs if condition in run["conditions"]
            ]
            if not condition_runs:
                continue
            conditions[condition] = {
                "post_shift_personal_reward_per_1000_frames": _stats(
                    run["conditions"][condition]["metrics"][
                        "post_shift_personal_reward_per_1000_frames"
                    ]
                    for run in condition_runs
                ),
                "post_shift_collective_reward": _stats(
                    run["conditions"][condition]["metrics"][
                        "post_shift_collective_reward"
                    ]
                    for run in condition_runs
                ),
                "n_active_principles_final": _stats(
                    run["conditions"][condition][
                        "n_active_principles_final"
                    ]
                    for run in condition_runs
                ),
                "n_forced_injections": _stats(
                    run["conditions"][condition].get(
                        "n_forced_injections", 0
                    )
                    for run in condition_runs
                ),
            }

        contrast_names = (
            "revision_effect_post_shift_reward_per_1000_frames",
            "full_scaffold_effect_post_shift_reward_per_1000_frames",
            "autonomy_vs_injection_post_shift_reward_per_1000_frames",
            "forced_injection_effect_post_shift_reward_per_1000_frames",
        )
        output["phase_orders"][label] = {
            "seeds": sorted(run["seed"] for run in runs),
            "conditions": conditions,
            "paired_contrasts": {
                name: _stats(
                    run["contrasts"][name]
                    for run in runs
                    if name in run["contrasts"]
                )
                for name in contrast_names
                if any(name in run["contrasts"] for run in runs)
            },
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        type=Path,
        help="directory containing clean_up_causal_seed*.json files",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    files = sorted(args.root.rglob("clean_up_causal_seed*.json"))
    if not files:
        raise SystemExit(f"no Clean Up result files found under {args.root}")
    result = aggregate(files)
    text = json.dumps(result, indent=2, ensure_ascii=False)
    output = args.output or args.root / "aggregate.json"
    output.write_text(text)
    print(text)
    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()
