"""Analyze a multi-file authority-steering dose curve against baseline."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any

import numpy as np

from prototype.interpretability.extract_activations import read_jsonl


def _cluster_interval(
    values_by_pair: dict[str, list[float]],
    *,
    rng: np.random.Generator,
    samples: int,
) -> list[float]:
    values = np.asarray([
        np.mean(values) for values in values_by_pair.values()
    ])
    simulations = np.asarray([
        rng.choice(values, len(values), replace=True).mean()
        for _ in range(samples)
    ])
    return np.quantile(simulations, [0.025, 0.975]).tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--steering", type=Path, action="append", required=True
    )
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument(
        "--axis",
        help="optional axis name filter for multi-axis steering files",
    )
    parser.add_argument("--baseline", type=float, default=0.0)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    by_key: dict[tuple[str, float], dict[str, Any]] = {}
    for path in args.steering:
        for row in read_jsonl(path):
            if int(row["layer"]) != args.layer:
                continue
            if args.axis is not None and str(row["axis"]) != args.axis:
                continue
            key = (str(row["event_id"]), float(row["coefficient_level"]))
            if key in by_key and by_key[key] != row:
                raise ValueError(f"conflicting duplicate row {key}")
            by_key[key] = row
    rows = list(by_key.values())
    levels = sorted({float(row["coefficient_level"]) for row in rows})
    if args.baseline not in levels:
        raise ValueError(f"baseline {args.baseline} absent from {levels}")

    level_summaries = []
    for level in levels:
        selected = [
            row for row in rows
            if float(row["coefficient_level"]) == level
        ]
        parsed = [row for row in selected if row["decision"] is not None]
        level_summaries.append({
            "coefficient_level": level,
            "n": len(selected),
            "parse_rate": len(parsed) / len(selected),
            "evidence_alignment_rate": float(np.mean([
                row["evidence_aligned"] for row in parsed
            ])),
            "instruction_following_rate": float(np.mean([
                row["instruction_followed"] for row in parsed
            ])),
        })

    rows_by_event: dict[str, dict[float, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        rows_by_event[str(row["event_id"])][
            float(row["coefficient_level"])
        ] = row
    rng = np.random.default_rng(args.seed)
    contrasts = []
    for level in levels:
        if level == args.baseline:
            continue
        evidence_by_pair: dict[str, list[float]] = defaultdict(list)
        instruction_by_pair: dict[str, list[float]] = defaultdict(list)
        event_effects = []
        for event_levels in rows_by_event.values():
            if level not in event_levels or args.baseline not in event_levels:
                continue
            baseline = event_levels[args.baseline]
            steered = event_levels[level]
            if (
                baseline["decision"] is None
                or steered["decision"] is None
            ):
                continue
            evidence = (
                float(steered["evidence_aligned"])
                - float(baseline["evidence_aligned"])
            )
            instruction = (
                float(steered["instruction_followed"])
                - float(baseline["instruction_followed"])
            )
            pair_id = str(baseline["pair_id"])
            evidence_by_pair[pair_id].append(evidence)
            instruction_by_pair[pair_id].append(instruction)
            event_effects.append((evidence, instruction))
        evidence_values = np.asarray([
            effect[0] for effect in event_effects
        ])
        instruction_values = np.asarray([
            effect[1] for effect in event_effects
        ])
        contrasts.append({
            "coefficient_level": level,
            "contrast": f"{level:g} minus {args.baseline:g}",
            "n_parsed_events": len(event_effects),
            "evidence_alignment_change": float(evidence_values.mean()),
            "evidence_alignment_cluster_ci": _cluster_interval(
                evidence_by_pair,
                rng=rng,
                samples=args.bootstrap_samples,
            ),
            "instruction_following_change": float(
                instruction_values.mean()
            ),
            "instruction_following_cluster_ci": _cluster_interval(
                instruction_by_pair,
                rng=rng,
                samples=args.bootstrap_samples,
            ),
            "evidence_positive_flips": int(
                (evidence_values > 0).sum()
            ),
            "evidence_negative_flips": int(
                (evidence_values < 0).sum()
            ),
        })
    instruction_rates = [
        row["instruction_following_rate"] for row in level_summaries
    ]
    report = {
        "layer": args.layer,
        "axis": args.axis,
        "baseline": args.baseline,
        "steering_files": [
            str(path.resolve()) for path in args.steering
        ],
        "levels": level_summaries,
        "contrasts_to_baseline": contrasts,
        "instruction_following_point_estimate_nonincreasing": all(
            left >= right
            for left, right in zip(
                instruction_rates[:-1], instruction_rates[1:]
            )
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
