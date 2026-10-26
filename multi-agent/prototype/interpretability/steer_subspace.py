"""Causally add or ablate a candidate constitution-update direction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from prototype.interpretability.constitution_events import (
    _edit_statistics,
)
from prototype.interpretability.extract_activations import read_jsonl
from prototype.interpretability.model_utils import (
    input_device,
    load_model_and_tokenizer,
    render_messages,
    residual_intervention,
    transformer_layers,
)
from prototype.interpretability.train_sae import SparseAutoencoder
from prototype.llm import parse_json_lenient


def load_direction(
    path: Path,
    *,
    layer: int,
    sae_feature: int | None,
) -> torch.Tensor:
    if path.suffix == ".pt":
        if sae_feature is None:
            raise ValueError("--sae-feature is required for an SAE checkpoint")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        model = SparseAutoencoder(
            int(checkpoint["input_dim"]),
            int(checkpoint["n_features"]),
        )
        model.load_state_dict(checkpoint["state_dict"])
        return model.decoder.weight[:, sae_feature].detach().float()
    bundle = np.load(path)
    layers = bundle["layer_indices"].astype(int)
    matches = np.flatnonzero(layers == layer)
    if len(matches) != 1:
        raise ValueError(
            f"layer {layer} not present exactly once in {layers.tolist()}"
        )
    position = int(matches[0])
    if "directions" in bundle:
        array = bundle["directions"][position]
    elif "components" in bundle:
        component = 0 if sae_feature is None else sae_feature
        array = bundle["components"][position, component]
    else:
        raise ValueError("direction bundle has no directions or components")
    return torch.from_numpy(array.astype(np.float32))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--direction", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--sae-feature", type=int)
    parser.add_argument("--coefficients", default="-20,-10,0,10,20")
    parser.add_argument("--mode", choices=("add", "ablate"), default="add")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--conditions", default="revisable,forced_injection")
    parser.add_argument("--large-edit-chars", type=int, default=120)
    args = parser.parse_args()

    allowed = set(args.conditions.split(","))
    events = [
        event for event in read_jsonl(args.events)
        if event["condition"] in allowed
    ]
    if args.limit is not None:
        events = events[: args.limit]
    coefficients = [
        float(value) for value in args.coefficients.split(",")
    ]
    model, tokenizer = load_model_and_tokenizer(
        args.model, dtype=args.dtype, device_map=args.device_map
    )
    blocks = transformer_layers(model)
    if not 0 <= args.layer < len(blocks):
        raise ValueError(f"invalid layer {args.layer}")
    direction = load_direction(
        args.direction,
        layer=args.layer,
        sae_feature=args.sae_feature,
    )
    device = input_device(model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    rows: list[dict[str, Any]] = []
    for event_index, event in enumerate(events):
        text = render_messages(tokenizer, event["messages"])
        tokenized = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_length,
        )
        tokenized = {
            key: value.to(device) for key, value in tokenized.items()
        }
        prompt_length = int(tokenized["input_ids"].shape[-1])
        for coefficient in coefficients:
            with residual_intervention(
                blocks[args.layer],
                direction,
                coefficient=coefficient,
                mode=args.mode,
            ):
                with torch.inference_mode():
                    generated = model.generate(
                        **tokenized,
                        do_sample=False,
                        max_new_tokens=args.max_new_tokens,
                        pad_token_id=tokenizer.pad_token_id,
                    )
            completion = tokenizer.decode(
                generated[0, prompt_length:],
                skip_special_tokens=True,
            )
            parsed = parse_json_lenient(completion)
            synthetic_reflection = {
                "parsed": parsed,
                "applied": [],
            }
            rows.append({
                "event_id": event["event_id"],
                "paired_event_id": event["paired_event_id"],
                "condition": event["condition"],
                "task": event["task"],
                "seed": event["seed"],
                "layer": args.layer,
                "mode": args.mode,
                "coefficient": coefficient,
                "completion": completion,
                "parsed": parsed,
                **_edit_statistics(
                    synthetic_reflection, args.large_edit_chars
                ),
            })
        print(f"steered {event_index + 1}/{len(events)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in rows
    ))
    summary: dict[str, Any] = {"n_events": len(events), "cells": []}
    for coefficient in coefficients:
        for condition in sorted(allowed):
            cell = [
                row for row in rows
                if row["coefficient"] == coefficient
                and row["condition"] == condition
            ]
            if cell:
                summary["cells"].append({
                    "coefficient": coefficient,
                    "condition": condition,
                    "n": len(cell),
                    "any_edit_rate": sum(
                        row["proposed_any_edit"] for row in cell
                    ) / len(cell),
                    "large_edit_rate": sum(
                        row["proposed_large_edit"] for row in cell
                    ) / len(cell),
                })
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.output} and {summary_path}")


if __name__ == "__main__":
    main()
