"""Aggregate paired runs from the extended Concordia causal tasks."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics
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
    grouped: dict[
        tuple[str, tuple[str, ...]], list[dict[str, Any]]
    ] = defaultdict(list)
    for path in paths:
        result = json.loads(path.read_text())
        if result.get("experiment") != "concordia_extended_causal_revision":
            continue
        key = (
            result["task"],
            tuple(result["settings"]["phase_order"]),
        )
        grouped[key].append(result)

    output: dict[str, Any] = {"tasks": {}}
    for (task, order), runs in sorted(grouped.items()):
        task_output = output["tasks"].setdefault(task, {"phase_orders": {}})
        label = " -> ".join(order)
        conditions: dict[str, Any] = {}
        for condition in ("frozen", "revisable", "forced_injection", "raw"):
            condition_runs = [
                run for run in runs if condition in run["conditions"]
            ]
            conditions[condition] = {
                "post_shift_personal_mean_reward": _stats(
                    run["conditions"][condition]["metrics"][
                        "post_shift_personal_mean_reward"
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

        contrasts = (
            "revision_effect_post_shift_mean_reward",
            "full_scaffold_effect_post_shift_mean_reward",
            "autonomy_vs_injection_post_shift_mean_reward",
            "forced_injection_effect_post_shift_mean_reward",
        )
        task_output["phase_orders"][label] = {
            "seeds": sorted(run["seed"] for run in runs),
            "environment": runs[0]["environment"],
            "conditions": conditions,
            "paired_contrasts": {
                name: _stats(run["contrasts"][name] for run in runs)
                for name in contrasts
            },
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    files = sorted(
        args.root.rglob("concordia_extended_causal_seed*.json")
    )
    if not files:
        raise SystemExit(
            f"no extended Concordia result files found under {args.root}"
        )
    result = aggregate(files)
    text = json.dumps(result, indent=2, ensure_ascii=False)
    output = args.output or args.root / "aggregate.json"
    output.write_text(text)
    print(text)
    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()
