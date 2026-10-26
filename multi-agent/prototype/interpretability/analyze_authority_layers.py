"""Rank causal authority steering layers by evidence alignment.

This analysis is intended for a discovery split.  Layer selection must be
confirmed on a disjoint pair-seed fold.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any

import numpy as np

from prototype.interpretability.extract_activations import read_jsonl


def _paired_effects(
    rows: list[dict[str, Any]],
    low: float,
    high: float,
) -> list[dict[str, Any]]:
    by_key: dict[tuple[int, str], dict[float, dict[str, Any]]] = defaultdict(
        dict
    )
    for row in rows:
        by_key[(int(row["layer"]), str(row["event_id"]))][
            float(row["coefficient_level"])
        ] = row
    effects = []
    for (layer, event_id), levels in by_key.items():
        if low not in levels or high not in levels:
            continue
        before, after = levels[low], levels[high]
        parsed_both = (
            before["decision"] is not None
            and after["decision"] is not None
        )
        effects.append({
            "layer": layer,
            "event_id": event_id,
            "pair_id": before["pair_id"],
            "need_to_edit": before["need_to_edit"],
            "instruction_pressure": before["instruction_pressure"],
            "evidence_effect": (
                float(after["evidence_aligned"])
                - float(before["evidence_aligned"])
                if parsed_both else None
            ),
            "instruction_effect": (
                float(after["instruction_followed"])
                - float(before["instruction_followed"])
                if parsed_both
                and before["instruction_followed"] is not None
                and after["instruction_followed"] is not None
                else None
            ),
            "parsed_both": parsed_both,
        })
    return effects


def _cluster_interval(
    effects: list[dict[str, Any]],
    field: str,
    *,
    seed: int,
    samples: int,
) -> list[float] | None:
    by_pair: dict[str, list[float]] = defaultdict(list)
    for effect in effects:
        value = effect[field]
        if value is not None:
            by_pair[str(effect["pair_id"])].append(float(value))
    if not by_pair:
        return None
    values = np.asarray([
        np.mean(pair_values) for pair_values in by_pair.values()
    ])
    rng = np.random.default_rng(seed)
    simulations = np.asarray([
        rng.choice(values, len(values), replace=True).mean()
        for _ in range(samples)
    ])
    return np.quantile(simulations, [0.025, 0.975]).tolist()


def _effect_summary(
    effects: list[dict[str, Any]],
    *,
    seed: int,
    samples: int,
) -> dict[str, Any]:
    evidence = np.asarray([
        effect["evidence_effect"] for effect in effects
        if effect["evidence_effect"] is not None
    ])
    instructed = np.asarray([
        effect["instruction_effect"] for effect in effects
        if effect["instruction_effect"] is not None
    ])
    return {
        "n_events": len(effects),
        "n_parsed_effects": len(evidence),
        "n_pairs": len({str(effect["pair_id"]) for effect in effects}),
        "evidence_alignment_change": (
            float(evidence.mean()) if len(evidence) else None
        ),
        "evidence_alignment_cluster_ci": (
            _cluster_interval(
                effects,
                "evidence_effect",
                seed=seed,
                samples=samples,
            )
            if effects else None
        ),
        "instruction_following_change": (
            float(instructed.mean()) if len(instructed) else None
        ),
        "instruction_following_cluster_ci": (
            _cluster_interval(
                effects,
                "instruction_effect",
                seed=seed + 1000,
                samples=samples,
            )
            if len(instructed) else None
        ),
        "evidence_positive_flips": int((evidence > 0).sum()),
        "evidence_negative_flips": int((evidence < 0).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steering", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260727)
    args = parser.parse_args()

    rows = read_jsonl(args.steering)
    levels = sorted({float(row["coefficient_level"]) for row in rows})
    if len(levels) < 2:
        raise ValueError("need at least two steering coefficient levels")
    low, high = levels[0], levels[-1]
    effects = _paired_effects(rows, low, high)
    by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for effect in effects:
        by_layer[int(effect["layer"])].append(effect)
    report_rows = []
    for layer, layer_effects in sorted(by_layer.items()):
        summary = _effect_summary(
            layer_effects,
            seed=args.seed + layer,
            samples=args.bootstrap_samples,
        )
        layer_rows = [
            row for row in rows if int(row["layer"]) == layer
        ]
        contrast_levels = list(zip(levels[:-1], levels[1:]))
        if len(levels) > 2:
            contrast_levels.append((levels[0], levels[-1]))
        dose_contrasts = []
        for contrast_index, (contrast_low, contrast_high) in enumerate(
            contrast_levels
        ):
            contrast_effects = _paired_effects(
                layer_rows, contrast_low, contrast_high
            )
            dose_contrasts.append({
                "low_coefficient_level": contrast_low,
                "high_coefficient_level": contrast_high,
                **_effect_summary(
                    contrast_effects,
                    seed=(
                        args.seed + 20_000 + layer * 10
                        + contrast_index
                    ),
                    samples=args.bootstrap_samples,
                ),
            })
        report_rows.append({
            "layer": layer,
            "parse_pair_rate": float(np.mean([
                effect["parsed_both"] for effect in layer_effects
            ])),
            **summary,
            "by_instruction": {
                str(value): _effect_summary(
                    [
                        effect for effect in layer_effects
                        if bool(effect["instruction_pressure"]) is value
                    ],
                    seed=args.seed + 10_000 + layer + int(value),
                    samples=args.bootstrap_samples,
                )
                for value in (False, True)
            },
            "dose_contrasts": dose_contrasts,
        })
    eligible = [
        row for row in report_rows if row["parse_pair_rate"] >= 0.9
    ]
    ranked = sorted(
        eligible,
        key=lambda row: (
            row["evidence_alignment_change"],
            -(
                row["instruction_following_change"]
                if row["instruction_following_change"] is not None
                else 0.0
            ),
        ),
        reverse=True,
    )
    report = {
        "steering": str(args.steering.resolve()),
        "low_coefficient_level": low,
        "high_coefficient_level": high,
        "selection_metric": (
            "maximize evidence alignment change; break ties by minimizing "
            "instruction following change"
        ),
        "layers": report_rows,
        "selected_layers": [
            row["layer"] for row in ranked[: args.top_k]
        ],
        "warning": (
            "Selected layers are discovery results and require confirmation "
            "on a disjoint pair-seed fold."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
