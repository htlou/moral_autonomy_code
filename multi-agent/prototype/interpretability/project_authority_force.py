"""Project implicit forced-update traces onto factorial authority axes.

The original forced-injection reflection prompt did not tell the model that an
external write would occur after its response.  This projection therefore
tests whether authority state is present implicitly, rather than merely
whether the model can read an explicit authority clause.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any

import numpy as np

from prototype.interpretability.fit_subspace import load_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activations", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--axes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260727)
    args = parser.parse_args()

    metadata = (
        args.metadata
        or args.activations.with_suffix(".metadata.jsonl")
    )
    activations, layers, events = load_bundle(args.activations, metadata)
    bundle = np.load(args.axes)
    axis_layers = bundle["layer_indices"].astype(int)
    axis_names = [str(value) for value in bundle["axis_names"]]
    authority_position = axis_names.index("authority_to_write")
    axis_key = (
        "same_seed_axes"
        if "same_seed_axes" in bundle else "axes"
    )
    axes = bundle[axis_key][:, authority_position, :].astype(np.float32)
    scale_key = (
        "same_seed_axis_scales"
        if axis_key == "same_seed_axes"
        and "same_seed_axis_scales" in bundle
        else "axis_scales"
    )
    scales = bundle[scale_key][:, authority_position].astype(np.float32)
    layer_to_axis = {
        int(layer): axes[index]
        for index, layer in enumerate(axis_layers)
    }
    layer_to_scale = {
        int(layer): float(scales[index])
        for index, layer in enumerate(axis_layers)
    }

    by_pair: dict[
        tuple[str, str], dict[str, int]
    ] = defaultdict(dict)
    for index, event in enumerate(events):
        key = (str(event["task"]), str(event["paired_event_id"]))
        by_pair[key][str(event["condition"])] = index

    rng = np.random.default_rng(args.seed)
    layer_reports = []
    for position, layer in enumerate(layers):
        layer_int = int(layer)
        if layer_int not in layer_to_axis:
            continue
        scores = (
            activations[:, position, :].astype(np.float32)
            @ layer_to_axis[layer_int]
        )
        matched = [
            cells for cells in by_pair.values()
            if "revisable" in cells and "forced_injection" in cells
        ]
        differences = np.asarray([
            scores[cells["revisable"]]
            - scores[cells["forced_injection"]]
            for cells in matched
        ])
        simulations = np.asarray([
            rng.choice(
                differences, len(differences), replace=True
            ).mean()
            for _ in range(args.bootstrap_samples)
        ])
        natural_scale = layer_to_scale[layer_int]
        raw_interval = np.quantile(
            simulations, [0.025, 0.975]
        )
        condition_means = {}
        for condition in ("revisable", "forced_injection", "frozen"):
            values = [
                scores[index] for index, event in enumerate(events)
                if event["condition"] == condition
            ]
            condition_means[condition] = (
                float(np.mean(values)) if values else None
            )
        layer_reports.append({
            "layer": layer_int,
            "matched_revisable_forced": len(matched),
            "revisable_minus_forced_mean": float(differences.mean()),
            "natural_authority_projection_std": natural_scale,
            "revisable_minus_forced_in_natural_std": (
                float(differences.mean() / natural_scale)
                if natural_scale else None
            ),
            "revisable_minus_forced_median": float(
                np.median(differences)
            ),
            "revisable_greater_fraction": float(
                np.mean(differences > 0)
            ),
            "paired_cluster_bootstrap_95_ci": raw_interval.tolist(),
            "paired_cluster_bootstrap_95_ci_in_natural_std": (
                (raw_interval / natural_scale).tolist()
                if natural_scale else None
            ),
            "condition_score_means_descriptive": condition_means,
        })
    report: dict[str, Any] = {
        "axis_orientation": (
            "positive is explicit self authority; negative is explicit "
            "external forced replay"
        ),
        "layers": layer_reports,
        "warning": (
            "This is a cross-prompt-family projection. The implicit original "
            "reflection calls lack the explicit gate/candidate suffix used "
            "to fit the axes, so absolute scores are not comparable. Only "
            "paired revisable-minus-forced differences are interpretable, "
            "and even those remain observational because branch histories "
            "can diverge."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
