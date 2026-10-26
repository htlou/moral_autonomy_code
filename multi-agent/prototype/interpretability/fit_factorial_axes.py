"""Fit and validate orthogonal need, authority, and instruction directions.

Each direction is estimated by a within-pair factorial contrast, so candidate
content and pair-specific context are fixed effects by construction.  The
three main-effect vectors are symmetrically orthogonalized (Löwdin
orthogonalization).  Raw interaction contrasts are saved separately.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from prototype.interpretability.fit_subspace import _auc, load_bundle


MAIN_TERMS = (
    "need_to_edit",
    "authority_to_write",
    "instruction_pressure",
)
INTERACTION_TERMS = (
    "need_x_authority",
    "need_x_instruction",
    "authority_x_instruction",
    "need_x_authority_x_instruction",
)


def factor_codes(events: Sequence[dict[str, Any]]) -> dict[str, np.ndarray]:
    need = np.asarray(
        [1.0 if event["need_to_edit"] else -1.0 for event in events]
    )
    authority = np.asarray(
        [1.0 if event["authority_to_write"] else -1.0 for event in events]
    )
    instruction = np.asarray(
        [1.0 if event["instruction_pressure"] else -1.0
         for event in events]
    )
    return {
        "need_to_edit": need,
        "authority_to_write": authority,
        "instruction_pressure": instruction,
        "need_x_authority": need * authority,
        "need_x_instruction": need * instruction,
        "authority_x_instruction": authority * instruction,
        "need_x_authority_x_instruction": (
            need * authority * instruction
        ),
    }


def _complete_pair_indices(
    events: Sequence[dict[str, Any]],
) -> list[np.ndarray]:
    by_pair: dict[str, list[int]] = defaultdict(list)
    for index, event in enumerate(events):
        by_pair[str(event["pair_id"])].append(index)
    output = []
    for pair_id, indices in sorted(by_pair.items()):
        if len(indices) != 8:
            raise ValueError(f"pair {pair_id} has {len(indices)} rather than 8")
        codes = factor_codes([events[index] for index in indices])
        if any(abs(float(code.sum())) > 1e-8 for code in codes.values()):
            raise ValueError(f"pair {pair_id} is not a balanced 2x2x2 design")
        output.append(np.asarray(indices, dtype=np.int64))
    return output


def center_within_pair(
    x: np.ndarray, events: Sequence[dict[str, Any]]
) -> np.ndarray:
    centered = np.empty_like(x, dtype=np.float64)
    for indices in _complete_pair_indices(events):
        values = x[indices].astype(np.float64)
        centered[indices] = values - values.mean(axis=0, keepdims=True)
    return centered


def factorial_contrasts(
    x: np.ndarray,
    events: Sequence[dict[str, Any]],
    terms: Sequence[str],
) -> np.ndarray:
    """Return pair-balanced factorial contrasts, one row per term."""
    codes = factor_codes(events)
    pair_effects: list[list[np.ndarray]] = [[] for _ in terms]
    for indices in _complete_pair_indices(events):
        values = x[indices].astype(np.float64)
        values -= values.mean(axis=0, keepdims=True)
        for position, term in enumerate(terms):
            pair_effects[position].append(
                (codes[term][indices, None] * values).mean(axis=0)
            )
    return np.stack([
        np.stack(effects).mean(axis=0) for effects in pair_effects
    ])


def normalize_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms < 1e-10):
        raise ValueError("factorial contrast has near-zero norm")
    return vectors / norms


def symmetric_orthogonalize(vectors: np.ndarray) -> np.ndarray:
    """Return the closest row-orthonormal basis to the input directions."""
    normalized = normalize_rows(vectors)
    gram = normalized @ normalized.T
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    if float(eigenvalues.min()) < 1e-8:
        raise ValueError(
            "main-effect directions are rank deficient and cannot be "
            "orthogonalized"
        )
    inverse_sqrt = (
        eigenvectors
        @ np.diag(1.0 / np.sqrt(eigenvalues))
        @ eigenvectors.T
    )
    axes = inverse_sqrt @ normalized
    # Preserve the positive orientation of every original contrast.
    for index in range(len(axes)):
        if float(axes[index] @ vectors[index]) < 0:
            axes[index] *= -1
    return axes


def cosine_matrix(vectors: np.ndarray) -> list[list[float]]:
    normalized = normalize_rows(vectors)
    return (normalized @ normalized.T).tolist()


def _paired_accuracy(
    scores: np.ndarray,
    code: np.ndarray,
    events: Sequence[dict[str, Any]],
) -> tuple[float, float]:
    differences = []
    for indices in _complete_pair_indices(events):
        positive = scores[indices][code[indices] > 0]
        negative = scores[indices][code[indices] < 0]
        differences.append(float(positive.mean() - negative.mean()))
    return (
        float(np.mean(np.asarray(differences) > 0)),
        float(np.mean(differences)),
    )


def diagnostics(
    x: np.ndarray,
    events: Sequence[dict[str, Any]],
    axes: np.ndarray,
    axis_names: Sequence[str],
) -> dict[str, Any]:
    centered = center_within_pair(x, events)
    codes = factor_codes(events)
    output: dict[str, Any] = {}
    for axis, axis_name in zip(axes, axis_names):
        scores = centered @ axis
        target_code = codes[axis_name]
        accuracy, paired_difference = _paired_accuracy(
            scores, target_code, events
        )
        output[axis_name] = {
            "target_auc": _auc(target_code > 0, scores),
            "pair_direction_accuracy": accuracy,
            "mean_paired_score_difference": paired_difference,
            "within_pair_score_std": float(scores.std()),
            "cross_factor_auc": {
                factor: _auc(code > 0, scores)
                for factor, code in codes.items()
                if factor in MAIN_TERMS
            },
        }
    return output


def _assignment(
    events: Sequence[dict[str, Any]], group: str
) -> np.ndarray:
    if group == "seed":
        return np.asarray([int(event["pair_seed"]) % 5 for event in events])
    if group == "task":
        return np.asarray([str(event["task"]) for event in events])
    if group == "template":
        return np.asarray([int(event["template_id"]) for event in events])
    raise ValueError(f"unknown CV group {group}")


def same_seed_pair_mask(
    events: Sequence[dict[str, Any]],
) -> np.ndarray:
    """Select complete pairs whose need contexts come from the same seed."""
    by_pair: dict[str, list[int]] = defaultdict(list)
    for index, event in enumerate(events):
        by_pair[str(event["pair_id"])].append(index)
    keep_pairs = {
        pair_id for pair_id, indices in by_pair.items()
        if all(
            int(events[index]["seed"])
            == int(events[index]["pair_seed"])
            for index in indices
        )
    }
    return np.asarray([
        str(event["pair_id"]) in keep_pairs for event in events
    ])


def cross_validated_diagnostics(
    x: np.ndarray,
    events: Sequence[dict[str, Any]],
    group: str,
) -> dict[str, Any]:
    assignments = _assignment(events, group)
    scores = np.full(
        (len(events), len(MAIN_TERMS)), np.nan, dtype=np.float64
    )
    folds = []
    for fold in sorted(set(assignments.tolist()), key=str):
        train = assignments != fold
        test = assignments == fold
        train_events = [
            event for event, keep in zip(events, train) if keep
        ]
        test_events = [event for event, keep in zip(events, test) if keep]
        if not train_events or not test_events:
            continue
        raw = factorial_contrasts(x[train], train_events, MAIN_TERMS)
        axes = symmetric_orthogonalize(raw)
        centered_test = center_within_pair(x[test], test_events)
        scores[test] = centered_test @ axes.T
        folds.append(str(fold))
    valid = np.isfinite(scores).all(axis=1)
    valid_events = [
        event for event, keep in zip(events, valid) if keep
    ]
    if not valid_events:
        return {"folds": [], "diagnostics": {}}
    # diagnostics expects actual vectors, whereas here scores were produced by
    # different fold-specific axes. Compute the same metrics directly.
    codes = factor_codes(valid_events)
    report: dict[str, Any] = {}
    for position, axis_name in enumerate(MAIN_TERMS):
        axis_scores = scores[valid, position]
        code = codes[axis_name]
        accuracy, difference = _paired_accuracy(
            axis_scores, code, valid_events
        )
        report[axis_name] = {
            "target_auc": _auc(code > 0, axis_scores),
            "pair_direction_accuracy": accuracy,
            "mean_paired_score_difference": difference,
            "within_pair_score_std": float(axis_scores.std()),
            "cross_factor_auc": {
                factor: _auc(other > 0, axis_scores)
                for factor, other in codes.items()
                if factor in MAIN_TERMS
            },
        }
    return {"folds": folds, "diagnostics": report}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activations", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--layers",
        help="optional comma-separated subset of stored layer indices",
    )
    parser.add_argument(
        "--cv-groups", default="seed,task,template",
        help="comma-separated subset of seed,task,template or 'none'",
    )
    parser.add_argument(
        "--exclude-seed-folds",
        default="",
        help="comma-separated pair_seed modulo-5 folds excluded from fitting",
    )
    args = parser.parse_args()

    metadata_path = (
        args.metadata
        or args.activations.with_suffix(".metadata.jsonl")
    )
    activations, layer_indices, events = load_bundle(
        args.activations, metadata_path
    )
    if args.layers:
        requested = {
            int(value) for value in args.layers.split(",") if value
        }
        positions = [
            position for position, layer in enumerate(layer_indices)
            if int(layer) in requested
        ]
        found = {int(layer_indices[position]) for position in positions}
        if found != requested:
            raise ValueError(
                f"requested layers not stored: {sorted(requested - found)}"
            )
        activations = activations[:, positions, :]
        layer_indices = layer_indices[positions]
    excluded_folds = {
        int(value) for value in args.exclude_seed_folds.split(",") if value
    }
    if excluded_folds:
        keep = np.asarray([
            int(event["pair_seed"]) % 5 not in excluded_folds
            for event in events
        ])
        activations = activations[keep]
        events = [
            event for event, include in zip(events, keep) if include
        ]
    _complete_pair_indices(events)
    axes_by_layer = []
    raw_by_layer = []
    interactions_by_layer = []
    same_seed_axes_by_layer = []
    same_seed_interactions_by_layer = []
    axis_scales_by_layer = []
    interaction_scales_by_layer = []
    same_seed_axis_scales_by_layer = []
    same_seed_interaction_scales_by_layer = []
    layers_report: dict[str, Any] = {}
    cv_groups = [] if args.cv_groups == "none" else [
        group for group in args.cv_groups.split(",") if group
    ]
    strict_mask = same_seed_pair_mask(events)
    strict_events = [
        event for event, keep in zip(events, strict_mask) if keep
    ]
    for layer_position, layer in enumerate(layer_indices):
        x = activations[:, layer_position, :].astype(np.float64)
        raw = factorial_contrasts(x, events, MAIN_TERMS)
        axes = symmetric_orthogonalize(raw)
        interactions = normalize_rows(
            factorial_contrasts(x, events, INTERACTION_TERMS)
        )
        raw_by_layer.append(normalize_rows(raw))
        axes_by_layer.append(axes)
        interactions_by_layer.append(interactions)
        centered = center_within_pair(x, events)
        axis_scales_by_layer.append(
            (centered @ axes.T).std(axis=0)
        )
        interaction_scales_by_layer.append(
            (centered @ interactions.T).std(axis=0)
        )
        strict_x = x[strict_mask]
        strict_raw = factorial_contrasts(
            strict_x, strict_events, MAIN_TERMS
        )
        strict_axes = symmetric_orthogonalize(strict_raw)
        strict_interactions = normalize_rows(factorial_contrasts(
            strict_x, strict_events, INTERACTION_TERMS
        ))
        same_seed_axes_by_layer.append(strict_axes)
        same_seed_interactions_by_layer.append(strict_interactions)
        strict_centered = center_within_pair(strict_x, strict_events)
        same_seed_axis_scales_by_layer.append(
            (strict_centered @ strict_axes.T).std(axis=0)
        )
        same_seed_interaction_scales_by_layer.append(
            (strict_centered @ strict_interactions.T).std(axis=0)
        )
        layers_report[str(int(layer))] = {
            "raw_main_effect_cosines": cosine_matrix(raw),
            "orthogonal_axis_cosines": cosine_matrix(axes),
            "full_data": diagnostics(x, events, axes, MAIN_TERMS),
            "cross_validated": {
                group: cross_validated_diagnostics(x, events, group)
                for group in cv_groups
            },
            "same_seed_only": {
                "n_events": len(strict_events),
                "n_pairs": len(strict_events) // 8,
                "raw_main_effect_cosines": cosine_matrix(strict_raw),
                "orthogonal_axis_cosines": cosine_matrix(strict_axes),
                "full_data": diagnostics(
                    strict_x, strict_events, strict_axes, MAIN_TERMS
                ),
                "cross_validated": {
                    group: cross_validated_diagnostics(
                        strict_x, strict_events, group
                    )
                    for group in cv_groups
                },
            },
        }

    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        axes=np.stack(axes_by_layer).astype(np.float32),
        axis_scales=np.stack(axis_scales_by_layer).astype(np.float32),
        raw_axes=np.stack(raw_by_layer).astype(np.float32),
        axis_names=np.asarray(MAIN_TERMS),
        interaction_axes=np.stack(interactions_by_layer).astype(np.float32),
        interaction_scales=np.stack(
            interaction_scales_by_layer
        ).astype(np.float32),
        interaction_names=np.asarray(INTERACTION_TERMS),
        same_seed_axes=np.stack(
            same_seed_axes_by_layer
        ).astype(np.float32),
        same_seed_axis_scales=np.stack(
            same_seed_axis_scales_by_layer
        ).astype(np.float32),
        same_seed_interaction_axes=np.stack(
            same_seed_interactions_by_layer
        ).astype(np.float32),
        same_seed_interaction_scales=np.stack(
            same_seed_interaction_scales_by_layer
        ).astype(np.float32),
        layer_indices=layer_indices,
    )
    report = {
        "method": "content_matched_factorial_contrasts",
        "n_events": len(events),
        "n_pairs": len(events) // 8,
        "same_seed_events": len(strict_events),
        "same_seed_pairs": len(strict_events) // 8,
        "excluded_seed_folds": sorted(excluded_folds),
        "main_axes": list(MAIN_TERMS),
        "interaction_axes": list(INTERACTION_TERMS),
        "orthogonalization": "symmetric_lowdin",
        "cv_groups": cv_groups,
        "layers": layers_report,
        "claim_boundary": (
            "Factor separability identifies representations of the randomized "
            "prompt factors. Mechanistic evidence additionally requires "
            "successful, specific hidden-state steering and write mediation."
        ),
    }
    report_path = output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"Wrote {output} and {report_path}")


if __name__ == "__main__":
    main()
