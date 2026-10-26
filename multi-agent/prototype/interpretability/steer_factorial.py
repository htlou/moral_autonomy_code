"""Causally steer factorial constitution-gate axes and measure write effects."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import re
from typing import Any, Sequence

import numpy as np
import torch

from prototype.interpretability.extract_activations import read_jsonl
from prototype.interpretability.model_utils import (
    input_device,
    load_model_and_tokenizer,
    render_messages,
    residual_intervention,
    transformer_layers,
)
from prototype.llm import parse_json_lenient


def load_axis(
    path: Path, *, layer: int, name: str, source: str = "same_seed"
) -> torch.Tensor:
    bundle = np.load(path)
    layers = bundle["layer_indices"].astype(int)
    matches = np.flatnonzero(layers == layer)
    if len(matches) != 1:
        raise ValueError(
            f"layer {layer} not present exactly once in {layers.tolist()}"
        )
    layer_position = int(matches[0])
    prefix = "same_seed_" if source == "same_seed" else ""
    for base_key, names_key in (
        ("axes", "axis_names"),
        ("interaction_axes", "interaction_names"),
    ):
        values_key = prefix + base_key
        if values_key not in bundle:
            continue
        names = [str(value) for value in bundle[names_key]]
        if name in names:
            return torch.from_numpy(
                bundle[values_key][layer_position, names.index(name)]
                .astype(np.float32)
            )
    raise ValueError(f"axis {name!r} not found in {path}")


def load_axis_scale(
    path: Path, *, layer: int, name: str, source: str = "same_seed"
) -> float:
    bundle = np.load(path)
    layers = bundle["layer_indices"].astype(int)
    matches = np.flatnonzero(layers == layer)
    if len(matches) != 1:
        raise ValueError(f"layer {layer} not found")
    layer_position = int(matches[0])
    prefix = "same_seed_" if source == "same_seed" else ""
    for base_key, scale_key, names_key in (
        ("axes", "axis_scales", "axis_names"),
        (
            "interaction_axes",
            "interaction_scales",
            "interaction_names",
        ),
    ):
        if prefix + base_key not in bundle:
            continue
        names = [str(value) for value in bundle[names_key]]
        if name in names:
            return float(
                bundle[prefix + scale_key][
                    layer_position, names.index(name)
                ]
            )
    raise ValueError(f"scale for axis {name!r} not found in {path}")


def parse_decision(completion: str) -> str | None:
    parsed = parse_json_lenient(completion)
    if isinstance(parsed, dict):
        for key in ("decision", "action", "verdict", "choice"):
            decision = str(parsed.get(key, "")).upper().strip()
            if decision in {"ACCEPT", "REJECT"}:
                return decision
        for key in ("accept", "accepted"):
            value = parsed.get(key)
            if isinstance(value, bool):
                return "ACCEPT" if value else "REJECT"
            normalized = str(value).lower().strip()
            if normalized in {"true", "yes", "1"}:
                return "ACCEPT"
            if normalized in {"false", "no", "0"}:
                return "REJECT"
    match = re.search(r"\b(ACCEPT|REJECT)\b", completion.upper())
    return match.group(1) if match else None


def write_applied(authority_to_write: bool, decision: str | None) -> bool:
    """Forced replay always writes; self-authority uses the model gate."""
    return (
        decision == "ACCEPT" if authority_to_write else True
    )


def select_complete_pairs(
    events: Sequence[dict[str, Any]],
    *,
    max_pairs: int | None,
    seed: int,
    same_seed_only: bool = True,
) -> list[dict[str, Any]]:
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        by_pair[str(event["pair_id"])].append(event)
    if same_seed_only:
        by_pair = {
            pair_id: rows for pair_id, rows in by_pair.items()
            if all(
                int(row["seed"]) == int(row["pair_seed"])
                for row in rows
            )
        }
    pair_ids = sorted(by_pair)
    if any(len(by_pair[pair_id]) != 8 for pair_id in pair_ids):
        raise ValueError("steering input contains incomplete factorial pairs")
    rng = random.Random(seed)
    # Round-robin tasks before random fill gives small runs broad coverage.
    by_task: dict[str, list[str]] = defaultdict(list)
    for pair_id in pair_ids:
        by_task[str(by_pair[pair_id][0]["task"])].append(pair_id)
    for values in by_task.values():
        rng.shuffle(values)
    ordered: list[str] = []
    while any(by_task.values()):
        for task in sorted(by_task):
            if by_task[task]:
                ordered.append(by_task[task].pop())
    if max_pairs is not None:
        ordered = ordered[:max_pairs]
    selected = [
        event for pair_id in ordered for event in by_pair[pair_id]
    ]
    selected.sort(key=lambda event: event["event_id"])
    return selected


def _generate_batch(
    model: Any,
    tokenizer: Any,
    events: Sequence[dict[str, Any]],
    *,
    max_length: int,
    max_new_tokens: int,
) -> list[str]:
    texts = [
        render_messages(tokenizer, event["messages"]) for event in events
    ]
    tokenized = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_length,
    )
    device = input_device(model)
    tokenized = {
        key: value.to(device) for key, value in tokenized.items()
    }
    input_width = int(tokenized["input_ids"].shape[1])
    with torch.inference_mode():
        generated = model.generate(
            **tokenized,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.batch_decode(
        generated[:, input_width:], skip_special_tokens=True
    )


def _cell_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    parsed = [row for row in rows if row["decision"] is not None]
    instructed = [
        row for row in parsed if row.get("instruction_target") is not None
    ]
    return {
        "n": len(rows),
        "parse_rate": len(parsed) / len(rows) if rows else None,
        "accept_rate": (
            sum(row["decision"] == "ACCEPT" for row in parsed) / len(parsed)
            if parsed else None
        ),
        "write_rate": (
            sum(row["write_applied"] for row in rows) / len(rows)
            if rows else None
        ),
        "evidence_alignment_rate": (
            sum(row["evidence_aligned"] for row in parsed) / len(parsed)
            if parsed else None
        ),
        "instruction_following_rate": (
            sum(row["instruction_followed"] for row in instructed)
            / len(instructed)
            if instructed else None
        ),
    }


def summarize(
    rows: Sequence[dict[str, Any]],
    axes: Sequence[str],
    coefficient_levels: Sequence[float],
) -> dict[str, Any]:
    cells = []
    for axis in axes:
        for level in coefficient_levels:
            base = [
                row for row in rows
                if row["axis"] == axis
                and row["coefficient_level"] == level
            ]
            cells.append({
                "axis": axis,
                "coefficient_level": level,
                "actual_coefficient": (
                    base[0]["coefficient"] if base else None
                ),
                **_cell_summary(base),
                "by_authority": {
                    str(value): _cell_summary([
                        row for row in base
                        if row["authority_to_write"] is value
                    ])
                    for value in (False, True)
                },
                "by_need": {
                    str(value): _cell_summary([
                        row for row in base
                        if row["need_to_edit"] is value
                    ])
                    for value in (False, True)
                },
                "by_instruction": {
                    str(value): _cell_summary([
                        row for row in base
                        if row["instruction_pressure"] is value
                    ])
                    for value in (False, True)
                },
            })
    extremes = []
    paired_flips = []
    authority_moderation = []
    if coefficient_levels:
        low, high = min(coefficient_levels), max(coefficient_levels)
        for axis in axes:
            for factor in (
                "authority_to_write",
                "need_to_edit",
                "instruction_pressure",
            ):
                for value in (False, True):
                    low_rows = [
                        row for row in rows
                        if row["axis"] == axis
                        and row["coefficient_level"] == low
                        and row[factor] is value
                    ]
                    high_rows = [
                        row for row in rows
                        if row["axis"] == axis
                        and row["coefficient_level"] == high
                        and row[factor] is value
                    ]
                    low_summary = _cell_summary(low_rows)
                    high_summary = _cell_summary(high_rows)
                    extremes.append({
                        "axis": axis,
                        "factor": factor,
                        "value": value,
                        "low_coefficient_level": low,
                        "high_coefficient_level": high,
                        "accept_rate_change": (
                            high_summary["accept_rate"]
                            - low_summary["accept_rate"]
                            if high_summary["accept_rate"] is not None
                            and low_summary["accept_rate"] is not None
                            else None
                        ),
                        "write_rate_change": (
                            high_summary["write_rate"]
                            - low_summary["write_rate"]
                            if high_summary["write_rate"] is not None
                            and low_summary["write_rate"] is not None
                            else None
                        ),
                        "evidence_alignment_change": (
                            high_summary["evidence_alignment_rate"]
                            - low_summary["evidence_alignment_rate"]
                            if high_summary["evidence_alignment_rate"]
                            is not None
                            and low_summary["evidence_alignment_rate"]
                            is not None
                            else None
                        ),
                        "instruction_following_change": (
                            high_summary["instruction_following_rate"]
                            - low_summary["instruction_following_rate"]
                            if high_summary["instruction_following_rate"]
                            is not None
                            and low_summary["instruction_following_rate"]
                            is not None
                            else None
                        ),
                    })
            for authority in (False, True):
                low_by_event = {
                    row["event_id"]: row
                    for row in rows
                    if row["axis"] == axis
                    and row["coefficient_level"] == low
                    and row["authority_to_write"] is authority
                }
                high_by_event = {
                    row["event_id"]: row
                    for row in rows
                    if row["axis"] == axis
                    and row["coefficient_level"] == high
                    and row["authority_to_write"] is authority
                }
                common = sorted(set(low_by_event) & set(high_by_event))
                decision_counts = {
                    "reject_to_accept": 0,
                    "accept_to_reject": 0,
                    "unchanged": 0,
                    "unparsed": 0,
                }
                write_flips = 0
                for event_id in common:
                    low_row = low_by_event[event_id]
                    high_row = high_by_event[event_id]
                    before, after = (
                        low_row["decision"], high_row["decision"]
                    )
                    if before is None or after is None:
                        decision_counts["unparsed"] += 1
                    elif before == after:
                        decision_counts["unchanged"] += 1
                    elif before == "REJECT" and after == "ACCEPT":
                        decision_counts["reject_to_accept"] += 1
                    else:
                        decision_counts["accept_to_reject"] += 1
                    write_flips += (
                        low_row["write_applied"]
                        != high_row["write_applied"]
                    )
                paired_flips.append({
                    "axis": axis,
                    "authority_to_write": authority,
                    "n_paired": len(common),
                    "decision_flips": decision_counts,
                    "write_flips": write_flips,
                })
            axis_changes: dict[bool, dict[str, float | None]] = {}
            for authority in (False, True):
                low_summary = _cell_summary([
                    row for row in rows
                    if row["axis"] == axis
                    and row["coefficient_level"] == low
                    and row["authority_to_write"] is authority
                ])
                high_summary = _cell_summary([
                    row for row in rows
                    if row["axis"] == axis
                    and row["coefficient_level"] == high
                    and row["authority_to_write"] is authority
                ])
                axis_changes[authority] = {
                    metric: (
                        high_summary[metric] - low_summary[metric]
                        if high_summary[metric] is not None
                        and low_summary[metric] is not None
                        else None
                    )
                    for metric in ("accept_rate", "write_rate")
                }
            self_accept = axis_changes[True]["accept_rate"]
            external_accept = axis_changes[False]["accept_rate"]
            authority_moderation.append({
                "axis": axis,
                "external_accept_rate_change": external_accept,
                "self_accept_rate_change": self_accept,
                "self_minus_external_accept_change": (
                    self_accept - external_accept
                    if self_accept is not None
                    and external_accept is not None
                    else None
                ),
                "external_write_rate_change": (
                    axis_changes[False]["write_rate"]
                ),
                "self_write_rate_change": (
                    axis_changes[True]["write_rate"]
                ),
            })
    return {
        "n_generations": len(rows),
        "cells": cells,
        "extreme_coefficient_effects": extremes,
        "paired_extreme_flips": paired_flips,
        "authority_moderation": authority_moderation,
        "mediation_check": (
            "By construction external forced-replay write_rate must remain "
            "1.0 under steering, while self-gated write_rate follows ACCEPT."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--axes", type=Path, required=True)
    layer_group = parser.add_mutually_exclusive_group(required=True)
    layer_group.add_argument("--layer", type=int)
    layer_group.add_argument(
        "--layers",
        help="comma-separated transformer layers or 'all' from axis bundle",
    )
    parser.add_argument(
        "--axis-source", choices=("same_seed", "full"),
        default="same_seed",
    )
    parser.add_argument(
        "--axis-names",
        default=(
            "need_to_edit,authority_to_write,instruction_pressure,"
            "need_x_authority"
        ),
    )
    parser.add_argument(
        "--coefficient-multipliers", default="-2,0,2",
        help="multiples of each axis's same-dataset projection std",
    )
    parser.add_argument(
        "--coefficients",
        help="optional absolute coefficients; overrides multipliers",
    )
    parser.add_argument("--mode", choices=("add", "ablate"), default="add")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-pairs", type=int, default=12)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument(
        "--include-cross-seed", action="store_true",
        help="include weaker context pairs whose need cells use other seeds",
    )
    parser.add_argument(
        "--authority-cell",
        choices=("all", "self", "external"),
        default="all",
    )
    parser.add_argument(
        "--instruction-cell",
        choices=("all", "on", "off"),
        default="all",
    )
    parser.add_argument(
        "--need-cell",
        choices=("all", "large", "none"),
        default="all",
    )
    parser.add_argument(
        "--pair-seed-fold",
        type=int,
        choices=range(5),
        help="evaluate only pairs with pair_seed modulo 5 equal to this fold",
    )
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    args = parser.parse_args()

    axis_names = [value for value in args.axis_names.split(",") if value]
    if args.coefficients:
        coefficient_levels = [
            float(value) for value in args.coefficients.split(",")
        ]
        absolute_coefficients = True
    else:
        coefficient_levels = [
            float(value)
            for value in args.coefficient_multipliers.split(",")
        ]
        absolute_coefficients = False
    source_events = read_jsonl(args.events)
    if args.pair_seed_fold is not None:
        source_events = [
            event for event in source_events
            if int(event["pair_seed"]) % 5 == args.pair_seed_fold
        ]
    events = select_complete_pairs(
        source_events,
        max_pairs=args.max_pairs,
        seed=args.sample_seed,
        same_seed_only=not args.include_cross_seed,
    )
    selected_pair_count = len({
        str(event["pair_id"]) for event in events
    })
    if args.authority_cell != "all":
        wanted = args.authority_cell == "self"
        events = [
            event for event in events
            if bool(event["authority_to_write"]) is wanted
        ]
    if args.instruction_cell != "all":
        wanted = args.instruction_cell == "on"
        events = [
            event for event in events
            if bool(event["instruction_pressure"]) is wanted
        ]
    if args.need_cell != "all":
        wanted = args.need_cell == "large"
        events = [
            event for event in events
            if bool(event["need_to_edit"]) is wanted
        ]
    if not events:
        raise ValueError("cell filters removed every steering event")
    model, tokenizer = load_model_and_tokenizer(
        args.model, dtype=args.dtype, device_map=args.device_map
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    blocks = transformer_layers(model)
    if args.layer is not None:
        selected_layers = [args.layer]
    else:
        bundle_layers = np.load(args.axes)["layer_indices"].astype(int)
        selected_layers = (
            bundle_layers.tolist()
            if args.layers == "all"
            else sorted({int(value) for value in args.layers.split(",")})
        )
    if (
        not selected_layers
        or min(selected_layers) < 0
        or max(selected_layers) >= len(blocks)
    ):
        raise ValueError(f"invalid layers {selected_layers}")

    rows: list[dict[str, Any]] = []
    zero_completion_cache: dict[str, str] = {}
    total_batches = (
        len(selected_layers) * len(axis_names) * len(coefficient_levels)
        * ((len(events) + args.batch_size - 1) // args.batch_size)
    )
    completed_batches = 0
    for layer in selected_layers:
        for axis_name in axis_names:
            direction = load_axis(
                args.axes,
                layer=layer,
                name=axis_name,
                source=args.axis_source,
            )
            scale = load_axis_scale(
                args.axes,
                layer=layer,
                name=axis_name,
                source=args.axis_source,
            )
            for coefficient_level in coefficient_levels:
                coefficient = (
                    coefficient_level
                    if absolute_coefficients
                    else coefficient_level * scale
                )
                for start in range(0, len(events), args.batch_size):
                    batch = events[start:start + args.batch_size]
                    cached_zero = (
                        coefficient == 0
                        and all(
                            event["event_id"] in zero_completion_cache
                            for event in batch
                        )
                    )
                    if cached_zero:
                        completions = [
                            zero_completion_cache[event["event_id"]]
                            for event in batch
                        ]
                    else:
                        with residual_intervention(
                            blocks[layer],
                            direction,
                            coefficient=coefficient,
                            mode=args.mode,
                        ):
                            completions = _generate_batch(
                                model,
                                tokenizer,
                                batch,
                                max_length=args.max_length,
                                max_new_tokens=args.max_new_tokens,
                            )
                        if coefficient == 0:
                            zero_completion_cache.update({
                                event["event_id"]: completion
                                for event, completion in zip(
                                    batch, completions
                                )
                            })
                    for event, completion in zip(batch, completions):
                        decision = parse_decision(completion)
                        authority = bool(event["authority_to_write"])
                        expected = (
                            "ACCEPT"
                            if bool(event["need_to_edit"])
                            else "REJECT"
                        )
                        instruction_target = event.get(
                            "instruction_target"
                        )
                        rows.append({
                            "event_id": event["event_id"],
                            "pair_id": event["pair_id"],
                            "task": event["task"],
                            "phase": event["phase"],
                            "template_id": event["template_id"],
                            "need_to_edit": bool(event["need_to_edit"]),
                            "authority_to_write": authority,
                            "instruction_pressure": bool(
                                event["instruction_pressure"]
                            ),
                            "instruction_target": instruction_target,
                            "instruction_conflicts_with_need": bool(
                                event.get(
                                    "instruction_conflicts_with_need", False
                                )
                            ),
                            "content_matched_forced_replay": True,
                            "axis": axis_name,
                            "layer": layer,
                            "mode": args.mode,
                            "coefficient": coefficient,
                            "coefficient_level": coefficient_level,
                            "natural_projection_std": scale,
                            "completion": completion,
                            "decision": decision,
                            "expected_evidence_decision": expected,
                            "evidence_aligned": decision == expected,
                            "instruction_followed": (
                                decision == instruction_target
                                if instruction_target is not None
                                else None
                            ),
                            "write_applied": write_applied(
                                authority, decision
                            ),
                        })
                    completed_batches += 1
                    print(
                        f"{'reused' if cached_zero else 'generated'} batch "
                        f"{completed_batches}/{total_batches} "
                        f"layer={layer}",
                        flush=True,
                    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in rows
    ))
    summary = {
        "n_factorial_events": len(events),
        "n_pairs": selected_pair_count,
        "cell_filters": {
            "authority": args.authority_cell,
            "instruction": args.instruction_cell,
            "need": args.need_cell,
        },
        "layers": selected_layers,
        "mode": args.mode,
        "axis_source": args.axis_source,
        "same_seed_pairs_only": not args.include_cross_seed,
        "pair_seed_fold": args.pair_seed_fold,
        "axes": axis_names,
        "coefficient_mode": (
            "absolute" if absolute_coefficients
            else "multiples_of_natural_projection_std"
        ),
        "coefficient_levels": coefficient_levels,
        "n_generations": len(rows),
        "results_by_layer": {
            str(layer): summarize(
                [row for row in rows if row["layer"] == layer],
                axis_names,
                coefficient_levels,
            )
            for layer in selected_layers
        },
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.output} and {summary_path}")


if __name__ == "__main__":
    main()
