"""Fit PCA, edit, or autonomy-interaction residual-stream subspaces.

``contrast`` estimates large-edit minus no-edit within the self-authorized
revisable branch.

``autonomy_interaction`` estimates the difference-in-differences

  (self large - self no edit) - (forced-branch large - forced-branch no edit)

after linearly residualizing configured nuisance variables.  This interaction
is a candidate update-gating direction, not by itself proof of autonomy.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def read_metadata(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def load_bundle(
    activation_path: Path, metadata_path: Path
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    bundle = np.load(activation_path)
    activations = bundle["activations"].astype(np.float32)
    layers = bundle["layer_indices"].astype(np.int32)
    event_ids = [str(value) for value in bundle["event_ids"]]
    metadata = read_metadata(metadata_path)
    if not all("activation_row" in event for event in metadata):
        raise ValueError("metadata is missing activation_row")
    ordered = sorted(metadata, key=lambda event: int(event["activation_row"]))
    if len(ordered) != activations.shape[0]:
        raise ValueError("metadata and activation row counts differ")
    ordered_ids = [str(event["event_id"]) for event in ordered]
    if ordered_ids != event_ids:
        mismatch = next(
            (
                index for index, (metadata_id, activation_id)
                in enumerate(zip(ordered_ids, event_ids))
                if metadata_id != activation_id
            ),
            None,
        )
        raise ValueError(
            "metadata rows do not align with activation event IDs"
            + (f" at row {mismatch}" if mismatch is not None else "")
        )
    return activations, layers, ordered


def _design_matrix(
    events: Sequence[dict[str, Any]],
    nuisance_fields: Sequence[str],
) -> tuple[np.ndarray, list[str]]:
    columns = [np.ones(len(events), dtype=np.float64)]
    names = ["intercept"]
    for field in nuisance_fields:
        values = [event.get(field) for event in events]
        numeric = all(
            value is None or isinstance(value, (int, float, bool))
            for value in values
        )
        if numeric:
            array = np.asarray(
                [0.0 if value is None else float(value) for value in values],
                dtype=np.float64,
            )
            std = float(array.std())
            if std > 1e-8:
                array = (array - float(array.mean())) / std
                columns.append(array)
                names.append(field)
            continue
        categories = sorted({str(value) for value in values})
        for category in categories[1:]:
            columns.append(np.asarray(
                [float(str(value) == category) for value in values]
            ))
            names.append(f"{field}={category}")
    return np.stack(columns, axis=1), names


def _residualize(
    train_x: np.ndarray,
    test_x: np.ndarray,
    train_z: np.ndarray,
    test_z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    coefficients = np.linalg.pinv(train_z) @ train_x
    return (
        train_x - train_z @ coefficients,
        test_x - test_z @ coefficients,
    )


def _cells(
    x: np.ndarray,
    events: Sequence[dict[str, Any]],
) -> dict[tuple[str, bool], np.ndarray]:
    result: dict[tuple[str, bool], np.ndarray] = {}
    conditions = np.asarray([event["condition"] for event in events])
    groups = np.asarray([event["edit_group"] for event in events])
    for condition in ("revisable", "forced_injection"):
        for label in (False, True):
            wanted = "large" if label else "none"
            mask = (conditions == condition) & (groups == wanted)
            if mask.any():
                result[(condition, label)] = x[mask]
    return result


def fit_direction(
    x: np.ndarray,
    events: Sequence[dict[str, Any]],
    method: str,
) -> tuple[np.ndarray, dict[str, int]]:
    cells = _cells(x, events)
    counts = {f"{condition}:{int(label)}": len(values)
              for (condition, label), values in cells.items()}
    if method == "contrast":
        required = (("revisable", False), ("revisable", True))
        missing = [cell for cell in required if cell not in cells]
        if missing:
            raise ValueError(f"missing contrast cells: {missing}")
        direction = (
            cells[("revisable", True)].mean(axis=0)
            - cells[("revisable", False)].mean(axis=0)
        )
    elif method == "autonomy_interaction":
        required = tuple(
            (condition, label)
            for condition in ("revisable", "forced_injection")
            for label in (False, True)
        )
        missing = [cell for cell in required if cell not in cells]
        if missing:
            raise ValueError(f"missing interaction cells: {missing}")
        self_edit = (
            cells[("revisable", True)].mean(axis=0)
            - cells[("revisable", False)].mean(axis=0)
        )
        forced_edit = (
            cells[("forced_injection", True)].mean(axis=0)
            - cells[("forced_injection", False)].mean(axis=0)
        )
        direction = self_edit - forced_edit
    else:
        raise ValueError(f"unknown direction method: {method}")
    norm = float(np.linalg.norm(direction))
    if norm < 1e-8:
        raise ValueError("estimated direction has near-zero norm")
    return direction / norm, counts


def _auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    positive = scores[labels]
    negative = scores[~labels]
    if not len(positive) or not len(negative):
        return None
    comparisons = (
        (positive[:, None] > negative[None, :]).mean()
        + 0.5 * (positive[:, None] == negative[None, :]).mean()
    )
    return float(comparisons)


def _diagnostics(
    x: np.ndarray,
    events: Sequence[dict[str, Any]],
    direction: np.ndarray,
) -> dict[str, Any]:
    scores = x @ direction
    condition = np.asarray([event["condition"] for event in events])
    large = np.asarray([
        bool(event["proposed_large_edit"]) for event in events
    ])
    no_edit = np.asarray([
        event["edit_group"] == "none" for event in events
    ])
    output: dict[str, Any] = {}
    for name in ("revisable", "forced_injection", "frozen"):
        mask = condition == name
        if not mask.any():
            continue
        evaluable = mask & (large | no_edit)
        output[name] = {
            "n": int(mask.sum()),
            "large_edit_rate": float(large[mask].mean()),
            "large_vs_no_edit_n": int(evaluable.sum()),
            "large_edit_auc": _auc(
                large[evaluable], scores[evaluable]
            ),
            "score_mean_large": (
                float(scores[mask & large].mean())
                if (mask & large).any() else None
            ),
            "score_mean_no_edit": (
                float(scores[mask & no_edit].mean())
                if (mask & no_edit).any() else None
            ),
        }
    return output


def _score_diagnostics(
    scores: np.ndarray,
    events: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    condition = np.asarray([event["condition"] for event in events])
    large = np.asarray([
        event["edit_group"] == "large" for event in events
    ])
    no_edit = np.asarray([
        event["edit_group"] == "none" for event in events
    ])
    output: dict[str, Any] = {}
    for name in ("revisable", "forced_injection", "frozen"):
        evaluable = (condition == name) & (large | no_edit)
        if evaluable.any():
            output[name] = {
                "n": int(evaluable.sum()),
                "large_edit_auc": _auc(
                    large[evaluable], scores[evaluable]
                ),
                "large_minus_no_score": (
                    float(scores[evaluable & large].mean())
                    - float(scores[evaluable & no_edit].mean())
                    if (evaluable & large).any()
                    and (evaluable & no_edit).any()
                    else None
                ),
            }
    return output


def cross_validated_scores(
    x: np.ndarray,
    z: np.ndarray,
    events: Sequence[dict[str, Any]],
    method: str,
    cv_group: str,
) -> tuple[np.ndarray, int]:
    if cv_group == "seed":
        assignments = np.asarray([
            int(event["seed"]) % 5 for event in events
        ])
    elif cv_group == "task":
        tasks = sorted({str(event["task"]) for event in events})
        by_task = {task: index for index, task in enumerate(tasks)}
        assignments = np.asarray([
            by_task[str(event["task"])] for event in events
        ])
    else:
        raise ValueError(f"unknown CV group: {cv_group}")
    scores = np.full(len(events), np.nan, dtype=np.float64)
    folds_used = 0
    for fold in sorted(set(assignments.tolist())):
        test = assignments == fold
        train = ~test
        train_events = [event for event, keep in zip(events, train) if keep]
        try:
            residual_train, residual_test = _residualize(
                x[train], x[test], z[train], z[test]
            )
            direction, _ = fit_direction(
                residual_train, train_events, method
            )
        except ValueError:
            continue
        scores[test] = residual_test @ direction
        folds_used += 1
    return scores, folds_used


def fit_pca(
    activations: np.ndarray, n_components: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Randomized low-rank PCA without materializing a full 8192x8192 SVD."""
    means = activations.mean(axis=0)
    components = []
    explained = []
    rng = np.random.default_rng(0)
    for layer_position in range(activations.shape[1]):
        centered = activations[:, layer_position, :] - means[layer_position]
        rank = min(
            n_components + 8,
            centered.shape[0],
            centered.shape[1],
        )
        projection = rng.standard_normal(
            (centered.shape[1], rank), dtype=np.float32
        )
        subspace = centered @ projection
        for _ in range(2):
            subspace = centered @ (centered.T @ subspace)
            subspace, _ = np.linalg.qr(subspace, mode="reduced")
        basis, _ = np.linalg.qr(subspace, mode="reduced")
        small = basis.T @ centered
        _u, singular, vt = np.linalg.svd(small, full_matrices=False)
        components.append(vt[:n_components])
        variance = singular ** 2
        total_variance = float(
            np.einsum("ij,ij->", centered, centered)
        )
        explained.append(
            variance[:n_components]
            / max(total_variance, 1e-12)
        )
    return (
        np.stack(components),
        means,
        np.stack(explained),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activations", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--method",
        choices=("pca", "contrast", "autonomy_interaction"),
        default="autonomy_interaction",
    )
    parser.add_argument("--n-components", type=int, default=8)
    parser.add_argument(
        "--nuisance-fields",
        default="task,phase,block_mean_reward,input_tokens",
    )
    parser.add_argument(
        "--cv-group", choices=("none", "seed", "task"), default="seed"
    )
    args = parser.parse_args()
    metadata_path = (
        args.metadata
        or args.activations.with_suffix(".metadata.jsonl")
    )
    activations, layer_indices, events = load_bundle(
        args.activations, metadata_path
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.method == "pca":
        components, means, explained = fit_pca(
            activations, args.n_components
        )
        np.savez_compressed(
            args.output,
            components=components.astype(np.float32),
            means=means.astype(np.float32),
            explained_variance_ratio=explained.astype(np.float32),
            layer_indices=layer_indices,
        )
        report = {
            "method": "pca",
            "n": int(activations.shape[0]),
            "n_components": args.n_components,
            "explained_variance_ratio": explained.tolist(),
        }
    else:
        nuisance_fields = [
            field for field in args.nuisance_fields.split(",") if field
        ]
        z, design_names = _design_matrix(events, nuisance_fields)
        directions = []
        diagnostics = []
        cv_diagnostics = []
        cell_counts: dict[str, int] | None = None
        for layer_position in range(activations.shape[1]):
            x = activations[:, layer_position, :].astype(np.float64)
            residual, _ = _residualize(x, x, z, z)
            direction, counts = fit_direction(
                residual, events, args.method
            )
            directions.append(direction)
            diagnostics.append(
                _diagnostics(residual, events, direction)
            )
            if args.cv_group != "none":
                cv_scores, folds_used = cross_validated_scores(
                    x, z, events, args.method, args.cv_group
                )
                valid = np.isfinite(cv_scores)
                cv_report = _score_diagnostics(
                    cv_scores[valid],
                    [event for event, keep in zip(events, valid) if keep],
                )
                cv_report["folds_used"] = folds_used
                cv_diagnostics.append(cv_report)
            cell_counts = counts
        directions_array = np.stack(directions).astype(np.float32)
        np.savez_compressed(
            args.output,
            directions=directions_array,
            layer_indices=layer_indices,
            method=np.asarray(args.method),
        )
        report = {
            "method": args.method,
            "n": int(activations.shape[0]),
            "cell_counts": cell_counts,
            "nuisance_design": design_names,
            "diagnostics_by_layer": {
                str(int(layer)): diagnostic
                for layer, diagnostic in zip(layer_indices, diagnostics)
            },
            "cross_validated_by_layer": (
                {
                    str(int(layer)): diagnostic
                    for layer, diagnostic in zip(
                        layer_indices, cv_diagnostics
                    )
                }
                if args.cv_group != "none" else {}
            ),
            "cv_group": args.cv_group,
            "warning": (
                "A separable or steerable direction is a candidate causal "
                "mechanism, not sufficient evidence of autonomy. Use matched "
                "forced replay and specificity controls."
            ),
        }
    report_path = args.output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"Wrote {args.output} and {report_path}")


if __name__ == "__main__":
    main()
