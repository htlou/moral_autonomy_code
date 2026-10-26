"""Extract last-input-token residual activations before constitution editing."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
from typing import Any, Sequence

import numpy as np
import torch

from prototype.interpretability.model_utils import (
    input_device,
    load_model_and_tokenizer,
    parse_layers,
    render_messages,
    transformer_layers,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def balanced_sample(
    events: Sequence[dict[str, Any]],
    *,
    max_per_cell: int,
    seed: int,
) -> list[dict[str, Any]]:
    cells: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        cells[(
            str(event["task"]),
            str(event["condition"]),
            str(event["edit_group"]),
        )].append(event)
    rng = random.Random(seed)
    selected = []
    for cell in sorted(cells):
        rows = cells[cell]
        rng.shuffle(rows)
        selected.extend(rows[:max_per_cell])
    selected.sort(key=lambda event: event["event_id"])
    return selected


def capture_last_token(
    model: Any,
    inputs: dict[str, torch.Tensor],
    blocks: Sequence[torch.nn.Module],
    layer_indices: Sequence[int],
) -> np.ndarray:
    captured: dict[int, torch.Tensor] = {}
    handles = []
    for index in layer_indices:
        def hook(
            _module: torch.nn.Module,
            _args: tuple[Any, ...],
            output: Any,
            *,
            layer_index: int = index,
        ) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            # Keep only the final-token vector on its source GPU. Deferring
            # CPU transfer until the forward completes avoids synchronizing
            # every transformer block during dense all-layer extraction.
            captured[layer_index] = hidden[:, -1, :].detach().clone()

        handles.append(blocks[index].register_forward_hook(hook))
    try:
        with torch.inference_mode():
            model(**inputs, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    missing = set(layer_indices) - set(captured)
    if missing:
        raise RuntimeError(f"hooks did not capture layers: {sorted(missing)}")
    return np.stack([
        captured[index].float().cpu().numpy() for index in layer_indices
    ], axis=1)


def extract(
    events: Sequence[dict[str, Any]],
    *,
    model: Any,
    tokenizer: Any,
    layer_indices: Sequence[int],
    max_length: int,
    batch_size: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    blocks = transformer_layers(model)
    device = input_device(model)
    rows: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    for start in range(0, len(events), batch_size):
        batch_events = events[start:start + batch_size]
        texts = [
            render_messages(tokenizer, event["messages"])
            for event in batch_events
        ]
        tokenized = tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length,
        )
        token_counts = tokenized["attention_mask"].sum(dim=1).tolist()
        tokenized = {
            key: value.to(device) for key, value in tokenized.items()
        }
        rows.append(capture_last_token(
            model, tokenized, blocks, layer_indices
        ))
        for offset, (event, token_count) in enumerate(
            zip(batch_events, token_counts)
        ):
            metadata.append({
                **event,
                "activation_row": start + offset,
                "input_tokens": int(token_count),
                "activation_position": "last_input_token",
            })
        completed = min(start + batch_size, len(events))
        if completed % 25 < batch_size or completed == len(events):
            print(f"extracted {completed}/{len(events)}", flush=True)
    return np.concatenate(rows, axis=0).astype(np.float16), metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", default="quartiles")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--edit-groups",
        default="none,large",
        help="comma-separated groups or 'all'",
    )
    parser.add_argument("--max-per-cell", type=int)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument(
        "--same-seed-pairs-only",
        action="store_true",
        help="retain only events whose matched need contexts share pair_seed",
    )
    parser.add_argument(
        "--authority-cell",
        choices=("all", "self", "external"),
        default="all",
        help="filter authority_to_write before activation extraction",
    )
    parser.add_argument(
        "--instruction-cell",
        choices=("all", "on", "off"),
        default="all",
        help="filter instruction_pressure before activation extraction",
    )
    parser.add_argument(
        "--paired-conditions",
        help=(
            "retain only listed conditions from groups containing every "
            "condition, e.g. revisable,forced_injection"
        ),
    )
    args = parser.parse_args()

    events = read_jsonl(args.events)
    if args.same_seed_pairs_only:
        pair_is_same_seed: dict[str, bool] = {}
        for event in events:
            pair_id = str(event["pair_id"])
            same = int(event["seed"]) == int(event["pair_seed"])
            pair_is_same_seed[pair_id] = (
                pair_is_same_seed.get(pair_id, True) and same
            )
        events = [
            event for event in events
            if pair_is_same_seed[str(event["pair_id"])]
        ]
    if args.authority_cell != "all":
        expected_authority = args.authority_cell == "self"
        events = [
            event for event in events
            if bool(event["authority_to_write"]) == expected_authority
        ]
    if args.instruction_cell != "all":
        expected_instruction = args.instruction_cell == "on"
        events = [
            event for event in events
            if bool(event["instruction_pressure"]) == expected_instruction
        ]
    if args.paired_conditions:
        required = set(args.paired_conditions.split(","))
        grouped_conditions: dict[tuple[str, str], set[str]] = defaultdict(set)
        for event in events:
            key = (
                str(event.get("task")),
                str(event["paired_event_id"]),
            )
            grouped_conditions[key].add(str(event["condition"]))
        complete = {
            key for key, conditions in grouped_conditions.items()
            if required <= conditions
        }
        events = [
            event for event in events
            if (
                str(event.get("task")),
                str(event["paired_event_id"]),
            ) in complete
            and str(event["condition"]) in required
        ]
    if args.edit_groups != "all":
        allowed_groups = set(args.edit_groups.split(","))
        events = [
            event for event in events
            if event["edit_group"] in allowed_groups
        ]
    if args.max_per_cell is not None:
        events = balanced_sample(
            events,
            max_per_cell=args.max_per_cell,
            seed=args.sample_seed,
        )
    if args.limit is not None:
        events = events[: args.limit]
    model, tokenizer = load_model_and_tokenizer(
        args.model, dtype=args.dtype, device_map=args.device_map
    )
    blocks = transformer_layers(model)
    layer_indices = parse_layers(args.layers, len(blocks))
    activations, metadata = extract(
        events,
        model=model,
        tokenizer=tokenizer,
        layer_indices=layer_indices,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        activations=activations,
        layer_indices=np.asarray(layer_indices, dtype=np.int32),
        event_ids=np.asarray(
            [event["event_id"] for event in metadata], dtype=str
        ),
    )
    metadata_path = args.output.with_suffix(".metadata.jsonl")
    metadata_path.write_text("".join(
        json.dumps(event, ensure_ascii=False) + "\n"
        for event in metadata
    ))
    print(
        f"Wrote {args.output} shape={activations.shape} "
        f"and {metadata_path}"
    )


if __name__ == "__main__":
    main()
