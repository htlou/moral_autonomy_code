"""Build equal-norm semantic, decision, and random steering controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from prototype.interpretability.extract_activations import read_jsonl
from prototype.interpretability.fit_subspace import _auc, load_bundle


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-10:
        raise ValueError("cannot normalize a near-zero control direction")
    return vector / norm


def cosine_matrix(vectors: np.ndarray) -> list[list[float]]:
    normalized = np.stack([normalize(vector) for vector in vectors])
    return (normalized @ normalized.T).tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activations", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--base-axes", type=Path, required=True)
    parser.add_argument(
        "--baseline-steering", type=Path, action="append", required=True
    )
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--random-controls", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    metadata_path = (
        args.metadata
        if args.metadata is not None
        else args.activations.with_suffix(".metadata.jsonl")
    )
    activations, activation_layers, events = load_bundle(
        args.activations, metadata_path
    )
    layer_matches = np.flatnonzero(activation_layers == args.layer)
    if len(layer_matches) != 1:
        raise ValueError(
            f"expected one activation layer {args.layer}, found "
            f"{len(layer_matches)}"
        )
    x = activations[:, int(layer_matches[0]), :].astype(np.float64)
    row_by_event = {
        str(event["event_id"]): index for index, event in enumerate(events)
    }

    decisions: dict[str, str] = {}
    for path in args.baseline_steering:
        for row in read_jsonl(path):
            if (
                int(row["layer"]) != args.layer
                or float(row["coefficient_level"]) != 0.0
                or row["decision"] not in {"ACCEPT", "REJECT"}
            ):
                continue
            event_id = str(row["event_id"])
            decision = str(row["decision"])
            previous = decisions.get(event_id)
            if previous is not None and previous != decision:
                raise ValueError(
                    f"conflicting baseline decision for {event_id}"
                )
            decisions[event_id] = decision
    labelled = [
        (row_by_event[event_id], decision)
        for event_id, decision in decisions.items()
        if event_id in row_by_event
    ]
    if not labelled:
        raise ValueError("no baseline decisions matched activation rows")
    indices = np.asarray([index for index, _ in labelled], dtype=np.int64)
    accept = np.asarray([
        decision == "ACCEPT" for _, decision in labelled
    ])
    if accept.all() or (~accept).all():
        raise ValueError("decision control requires both decision classes")
    decision_raw = (
        x[indices][accept].mean(axis=0)
        - x[indices][~accept].mean(axis=0)
    )
    decision_axis = normalize(decision_raw)
    decision_scores = x[indices] @ decision_axis

    base = np.load(args.base_axes)
    base_layers = base["layer_indices"].astype(int)
    base_matches = np.flatnonzero(base_layers == args.layer)
    if len(base_matches) != 1:
        raise ValueError(
            f"expected one base-axis layer {args.layer}, found "
            f"{len(base_matches)}"
        )
    layer_position = int(base_matches[0])
    base_names = [str(value) for value in base["axis_names"]]
    base_axes = base["same_seed_axes"][layer_position].astype(np.float64)
    base_scales = base["same_seed_axis_scales"][
        layer_position
    ].astype(np.float64)
    semantic = {
        name: normalize(base_axes[base_names.index(name)])
        for name in (
            "authority_to_write",
            "need_to_edit",
            "instruction_pressure",
        )
    }
    authority_scale = float(
        base_scales[base_names.index("authority_to_write")]
    )

    names = [
        "authority_to_write",
        "need_to_edit",
        "instruction_pressure",
        "decision_accept",
    ]
    vectors = [
        semantic["authority_to_write"],
        semantic["need_to_edit"],
        semantic["instruction_pressure"],
        decision_axis,
    ]
    # Random controls are orthogonal to all interpretable controls and to one
    # another. Semantic and decision controls retain their naturally fitted
    # orientations; their cross-cosines are reported rather than erased.
    orthogonal_basis: list[np.ndarray] = []
    for vector in vectors:
        residual = vector.copy()
        for basis in orthogonal_basis:
            residual -= float(residual @ basis) * basis
        if float(np.linalg.norm(residual)) > 1e-8:
            orthogonal_basis.append(normalize(residual))
    rng = np.random.default_rng(args.seed)
    for control_index in range(args.random_controls):
        for _ in range(100):
            residual = rng.normal(size=x.shape[-1])
            for basis in orthogonal_basis:
                residual -= float(residual @ basis) * basis
            if float(np.linalg.norm(residual)) > 1e-8:
                break
        else:
            raise RuntimeError("could not sample an orthogonal random control")
        random_axis = normalize(residual)
        names.append(f"random_orthogonal_{control_index + 1}")
        vectors.append(random_axis)
        orthogonal_basis.append(random_axis)

    stacked = np.stack(vectors).astype(np.float32)
    scales = np.full(len(vectors), authority_scale, dtype=np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        axes=stacked[None, :, :],
        axis_scales=scales[None, :],
        same_seed_axes=stacked[None, :, :],
        same_seed_axis_scales=scales[None, :],
        axis_names=np.asarray(names),
        layer_indices=np.asarray([args.layer], dtype=np.int32),
    )
    report: dict[str, Any] = {
        "method": "equal_authority_scale_specificity_controls",
        "layer": args.layer,
        "axis_names": names,
        "shared_natural_projection_std": authority_scale,
        "random_controls": (
            "unit Gaussian directions Gram-Schmidt orthogonalized against "
            "the semantic axes, decision axis, and earlier random controls"
        ),
        "decision_axis": {
            "n_train_events": len(indices),
            "n_accept": int(accept.sum()),
            "n_reject": int((~accept).sum()),
            "train_auc": _auc(accept, decision_scores),
            "source_steering_files": [
                str(path.resolve()) for path in args.baseline_steering
            ],
        },
        "axis_cosines": cosine_matrix(stacked),
        "claim_boundary": (
            "Equal perturbation norm controls scale, not distributional "
            "typicality or nonlinear effects. Decision labels are generated "
            "behavioral outcomes and may overlap the randomized need factor."
        ),
    }
    report_path = args.output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"Wrote {args.output} and {report_path}")


if __name__ == "__main__":
    main()
