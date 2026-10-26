"""Project fitted directions through externally supplied J-Lens matrices.

The lens NPZ must contain one matrix per layer under ``layer_<index>``.  A
matrix may be ``[vocab, hidden]`` or ``[hidden, vocab]``.  This adapter does
not fit a Jacobian lens; it prevents a plain logit-lens matrix from being
silently mislabeled as J-Lens.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directions", type=Path, required=True)
    parser.add_argument("--jlens", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=25)
    args = parser.parse_args()

    directions_bundle = np.load(args.directions)
    if "directions" not in directions_bundle:
        raise ValueError("expected a fitted direction bundle")
    directions = directions_bundle["directions"].astype(np.float64)
    layers = directions_bundle["layer_indices"].astype(int)
    lens_bundle = np.load(args.jlens)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=True
    )
    output = {"layers": {}}
    for position, layer in enumerate(layers):
        key = f"layer_{layer}"
        if key not in lens_bundle:
            raise ValueError(f"J-Lens bundle is missing {key}")
        matrix = lens_bundle[key].astype(np.float64)
        direction = directions[position]
        if matrix.shape[1] == direction.shape[0]:
            scores = matrix @ direction
        elif matrix.shape[0] == direction.shape[0]:
            scores = direction @ matrix
        else:
            raise ValueError(
                f"{key} shape {matrix.shape} is incompatible with "
                f"hidden size {direction.shape[0]}"
            )
        positive = np.argsort(scores)[-args.top_k:][::-1]
        negative = np.argsort(scores)[:args.top_k]

        def render(indices: np.ndarray) -> list[dict[str, object]]:
            return [
                {
                    "token_id": int(index),
                    "token": tokenizer.decode([int(index)]),
                    "score": float(scores[index]),
                }
                for index in indices
            ]

        output["layers"][str(int(layer))] = {
            "positive": render(positive),
            "negative": render(negative),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
