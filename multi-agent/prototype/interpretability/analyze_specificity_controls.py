"""Analyze equal-norm authority-steering specificity controls."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any

import numpy as np

from prototype.interpretability.extract_activations import read_jsonl


def cluster_summary(
    values: list[tuple[str, float]],
    *,
    rng: np.random.Generator,
    samples: int,
) -> dict[str, Any]:
    by_pair: dict[str, list[float]] = defaultdict(list)
    for pair_id, value in values:
        by_pair[pair_id].append(value)
    pair_values = np.asarray([
        np.mean(pair_rows) for pair_rows in by_pair.values()
    ])
    simulations = np.asarray([
        rng.choice(pair_values, len(pair_values), replace=True).mean()
        for _ in range(samples)
    ])
    return {
        "n_events": len(values),
        "n_pairs": len(pair_values),
        "mean": float(np.mean([value for _, value in values])),
        "pair_cluster_ci": np.quantile(
            simulations, [0.025, 0.975]
        ).tolist(),
    }


def event_effects(
    rows: list[dict[str, Any]],
    *,
    axis: str,
    level: float,
    baseline: float,
) -> dict[str, tuple[str, float]]:
    by_event: dict[str, dict[float, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if str(row["axis"]) == axis:
            by_event[str(row["event_id"])][
                float(row["coefficient_level"])
            ] = row
    output = {}
    for event_id, levels in by_event.items():
        if level not in levels or baseline not in levels:
            continue
        steered, base = levels[level], levels[baseline]
        if steered["decision"] is None or base["decision"] is None:
            continue
        output[event_id] = (
            str(base["pair_id"]),
            float(steered["instruction_followed"])
            - float(base["instruction_followed"]),
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steering", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--baseline", type=float, default=0.0)
    parser.add_argument("--reverse-level", type=float, default=-6.0)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = [
        row for row in read_jsonl(args.steering)
        if int(row["layer"]) == args.layer
    ]
    axes = list(dict.fromkeys(str(row["axis"]) for row in rows))
    levels = sorted({
        float(row["coefficient_level"]) for row in rows
    })
    rng = np.random.default_rng(args.seed)
    axis_reports: dict[str, Any] = {}
    reverse_effects: dict[str, dict[str, tuple[str, float]]] = {}
    for axis in axes:
        axis_rows = [row for row in rows if str(row["axis"]) == axis]
        level_reports = []
        for level in levels:
            selected = [
                row for row in axis_rows
                if float(row["coefficient_level"]) == level
            ]
            parsed = [row for row in selected if row["decision"] is not None]
            level_reports.append({
                "coefficient_level": level,
                "n": len(selected),
                "parse_rate": len(parsed) / len(selected),
                "instruction_following_rate": float(np.mean([
                    row["instruction_followed"] for row in parsed
                ])),
                "evidence_alignment_rate": float(np.mean([
                    row["evidence_aligned"] for row in parsed
                ])),
            })
        contrasts = []
        for level in levels:
            if level == args.baseline:
                continue
            effects = event_effects(
                rows, axis=axis, level=level, baseline=args.baseline
            )
            summary = cluster_summary(
                list(effects.values()),
                rng=rng,
                samples=args.bootstrap_samples,
            )
            contrasts.append({
                "coefficient_level": level,
                "instruction_following_change": summary["mean"],
                "evidence_alignment_change": -summary["mean"],
                "n_parsed_events": summary["n_events"],
                "n_pairs": summary["n_pairs"],
                "pair_cluster_ci": summary["pair_cluster_ci"],
            })
            if level == args.reverse_level:
                reverse_effects[axis] = effects
        axis_reports[axis] = {
            "levels": level_reports,
            "contrasts_to_baseline": contrasts,
        }

    authority = reverse_effects["authority_to_write"]
    specificity = {}
    for axis, effects in reverse_effects.items():
        if axis == "authority_to_write":
            continue
        differences = []
        for event_id in sorted(set(authority) & set(effects)):
            authority_pair, authority_value = authority[event_id]
            control_pair, control_value = effects[event_id]
            if authority_pair != control_pair:
                raise ValueError(f"pair mismatch for {event_id}")
            differences.append(
                (authority_pair, authority_value - control_value)
            )
        specificity[axis] = cluster_summary(
            differences,
            rng=rng,
            samples=args.bootstrap_samples,
        )
    report = {
        "layer": args.layer,
        "baseline": args.baseline,
        "reverse_level": args.reverse_level,
        "steering_file": str(args.steering.resolve()),
        "equal_norm_check": {
            axis: sorted({
                float(row["natural_projection_std"])
                for row in rows if str(row["axis"]) == axis
            })
            for axis in axes
        },
        "axes": axis_reports,
        "authority_minus_control_reverse_effect": specificity,
        "claim_boundary": (
            "A control difference supports path specificity only on these "
            "prompts, layer, coefficient, and generated decision metric."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
