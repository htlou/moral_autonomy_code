"""Merge sharded multi-site gate-localization artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from prototype.interpretability.localize_gate_interaction import (
    rank_candidates,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--heldout-fold", type=int, default=1)
    args = parser.parse_args()

    bundles = [np.load(path) for path in args.input]
    reference = bundles[0]
    for bundle in bundles[1:]:
        for key in ("term_names", "site_names", "layer_indices"):
            if not np.array_equal(bundle[key], reference[key]):
                raise ValueError(f"shards disagree on {key}")
    pair_effects = np.concatenate([
        bundle["pair_effects"] for bundle in bundles
    ])
    pair_ids = np.concatenate([bundle["pair_ids"] for bundle in bundles])
    pair_folds = np.concatenate([
        bundle["pair_folds"] for bundle in bundles
    ])
    counts = np.asarray([
        len(bundle["pair_ids"]) for bundle in bundles
    ], dtype=np.float64)
    term_means = sum(
        float(count) * bundle["term_means"]
        for count, bundle in zip(counts, bundles)
    ) / counts.sum()
    order = np.argsort(pair_ids)
    pair_effects = pair_effects[order]
    pair_ids = pair_ids[order]
    pair_folds = pair_folds[order]
    ranking = rank_candidates(
        pair_effects,
        pair_folds,
        heldout_fold=args.heldout_fold,
        layer_indices=reference["layer_indices"],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        pair_effects=pair_effects,
        term_means=term_means.astype(np.float32),
        term_names=reference["term_names"],
        site_names=reference["site_names"],
        layer_indices=reference["layer_indices"],
        pair_ids=pair_ids,
        pair_folds=pair_folds,
    )
    report = {
        "method": "merged_within_pair_need_x_authority_multisite_localization",
        "source_shards": [str(path.resolve()) for path in args.input],
        "n_pairs": len(pair_ids),
        "pair_fold_counts": {
            str(fold): int(np.sum(pair_folds == fold))
            for fold in sorted(set(pair_folds.tolist()))
        },
        "heldout_fold": args.heldout_fold,
        "selection_rule": (
            "descending minimum leave-one-discovery-fold cosine, then mean "
            "pair-direction accuracy; heldout metrics are report-only"
        ),
        "top_candidates": ranking[:30],
        "claim_boundary": (
            "Candidate ranking is outcome-free representation localization. "
            "Natural activation patching is required for causal mediation."
        ),
    }
    report_path = args.output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"Wrote {args.output} and {report_path}")


if __name__ == "__main__":
    main()
