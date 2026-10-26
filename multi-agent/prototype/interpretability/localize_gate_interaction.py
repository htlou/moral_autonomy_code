"""Localize a content-matched authority-by-need interaction across tokens.

The extractor processes one complete 2x2x2 pair at a time and stores the
within-pair need x authority factorial effect at every transformer layer and
five prompt positions. Candidate ranking uses only representation consistency,
not generated ACCEPT/REJECT outcomes.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from prototype.interpretability.extract_activations import read_jsonl
from prototype.interpretability.fit_factorial_axes import (
    INTERACTION_TERMS,
    MAIN_TERMS,
    factor_codes,
)
from prototype.interpretability.model_utils import (
    input_device,
    load_model_and_tokenizer,
    render_messages,
    transformer_layers,
)


SITE_NAMES = (
    "evidence_end",
    "candidate_end",
    "authority_end",
    "final_prompt",
    "verdict_cue",
)
PROMPT_SITE_NAMES = SITE_NAMES[:-1]
TERMS = MAIN_TERMS + INTERACTION_TERMS
MARKERS = {
    "evidence_end": "=== FIXED CANDIDATE AMENDMENT ===",
    "final_prompt": None,
}


def complete_same_seed_pairs(
    events: Sequence[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        grouped[str(event["pair_id"])].append(event)
    output = []
    for pair_id, rows in sorted(grouped.items()):
        if len(rows) != 8:
            continue
        if not all(
            int(row["seed"]) == int(row["pair_seed"]) for row in rows
        ):
            continue
        cells = {
            (
                bool(row["need_to_edit"]),
                bool(row["authority_to_write"]),
                bool(row["instruction_pressure"]),
            )
            for row in rows
        }
        if len(cells) != 8:
            raise ValueError(f"pair {pair_id} is not a complete 2x2x2")
        output.append(sorted(
            rows,
            key=lambda row: (
                bool(row["need_to_edit"]),
                bool(row["authority_to_write"]),
                bool(row["instruction_pressure"]),
            ),
        ))
    return output


def prompt_positions(
    tokenizer: Any,
    texts: Sequence[str],
    input_ids: torch.Tensor,
    events: Sequence[dict[str, Any]],
    *,
    max_length: int,
) -> torch.Tensor:
    positions = []
    padded_length = int(input_ids.shape[1])
    for text, ids_tensor, event in zip(texts, input_ids, events):
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True,
        )
        ids = list(encoded["input_ids"])
        offsets = list(encoded["offset_mapping"])
        first = padded_length - len(ids)
        if ids_tensor[first:].tolist() != ids:
            raise ValueError("per-event tokenization does not match batch")
        event_positions = []
        for site in PROMPT_SITE_NAMES:
            if site == "final_prompt":
                token_position = len(ids) - 1
            else:
                candidate_marker = MARKERS["evidence_end"]
                marker_occurrences = [
                    index
                    for index in range(len(text))
                    if text.startswith(candidate_marker, index)
                ]
                if len(marker_occurrences) != 1:
                    raise ValueError(
                        "expected one fixed-candidate marker, found "
                        f"{len(marker_occurrences)}"
                    )
                if site == "evidence_end":
                    boundary = marker_occurrences[0]
                else:
                    candidate_break = text.find(
                        "\n\n",
                        marker_occurrences[0] + len(candidate_marker),
                    )
                    if candidate_break < 0:
                        raise ValueError("candidate section has no terminator")
                    if site == "candidate_end":
                        boundary = candidate_break
                    else:
                        authority_start = candidate_break + 2
                        authority_break = text.find(
                            "\n\n", authority_start
                        )
                        if authority_break < 0:
                            raise ValueError(
                                "authority section has no terminator"
                            )
                        boundary = authority_break
                before = [
                    index for index, (start, end) in enumerate(offsets)
                    if end <= boundary and end > start
                ]
                if not before:
                    raise ValueError(f"{site}: no token before marker")
                token_position = before[-1]
            if token_position < 0:
                raise ValueError(f"invalid position for {site}")
            event_positions.append(first + token_position)
        positions.append(event_positions)
    return torch.as_tensor(positions, dtype=torch.long)


def capture_sites(
    base_model: Any,
    blocks: Sequence[torch.nn.Module],
    inputs: dict[str, torch.Tensor],
    positions: torch.Tensor,
) -> np.ndarray:
    captured: dict[int, torch.Tensor] = {}
    handles = []
    for layer_index, block in enumerate(blocks):
        def hook(
            _module: torch.nn.Module,
            _args: tuple[Any, ...],
            output: Any,
            *,
            index: int = layer_index,
        ) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            local_positions = positions.to(hidden.device)
            batch_indices = torch.arange(
                hidden.shape[0], device=hidden.device
            )[:, None].expand_as(local_positions)
            captured[index] = hidden[
                batch_indices, local_positions
            ].detach().clone()

        handles.append(block.register_forward_hook(hook))
    try:
        with torch.inference_mode():
            base_model(**inputs, use_cache=False, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()
    if len(captured) != len(blocks):
        missing = sorted(set(range(len(blocks))) - set(captured))
        raise RuntimeError(f"missing captured layers {missing}")
    return np.stack([
        captured[index].float().cpu().numpy()
        for index in range(len(blocks))
    ], axis=1)


def tokenize(
    tokenizer: Any,
    texts: Sequence[str],
    *,
    device: torch.device,
    max_length: int,
) -> dict[str, torch.Tensor]:
    encoded = tokenizer(
        list(texts),
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_length,
    )
    return {key: value.to(device) for key, value in encoded.items()}


def rank_candidates(
    pair_effects: np.ndarray,
    pair_folds: np.ndarray,
    *,
    heldout_fold: int,
    layer_indices: np.ndarray,
) -> list[dict[str, Any]]:
    discovery_folds = sorted(
        set(pair_folds.tolist()) - {heldout_fold}
    )
    candidates = []
    effects = pair_effects.astype(np.float32)
    for layer_position, layer in enumerate(layer_indices):
        for site_position, site in enumerate(SITE_NAMES):
            x = effects[:, layer_position, site_position, :]
            discovery = pair_folds != heldout_fold
            discovery_mean = x[discovery].mean(axis=0)
            discovery_norm = float(np.linalg.norm(discovery_mean))
            fold_cosines = []
            fold_accuracies = []
            for fold in discovery_folds:
                train = discovery & (pair_folds != fold)
                test = pair_folds == fold
                train_mean = x[train].mean(axis=0)
                train_norm = float(np.linalg.norm(train_mean))
                test_mean = x[test].mean(axis=0)
                test_norm = float(np.linalg.norm(test_mean))
                cosine = float(
                    train_mean @ test_mean
                    / max(train_norm * test_norm, 1e-12)
                )
                fold_cosines.append(cosine)
                axis = train_mean / max(train_norm, 1e-12)
                fold_accuracies.append(float(np.mean(x[test] @ axis > 0)))
            heldout = pair_folds == heldout_fold
            axis = discovery_mean / max(discovery_norm, 1e-12)
            heldout_scores = x[heldout] @ axis
            heldout_mean = x[heldout].mean(axis=0)
            heldout_cosine = float(
                discovery_mean @ heldout_mean
                / max(
                    discovery_norm
                    * float(np.linalg.norm(heldout_mean)),
                    1e-12,
                )
            )
            candidates.append({
                "layer": int(layer),
                "site": site,
                "discovery_effect_norm": discovery_norm,
                "discovery_fold_cosines": fold_cosines,
                "minimum_discovery_fold_cosine": float(
                    min(fold_cosines)
                ),
                "mean_discovery_fold_cosine": float(
                    np.mean(fold_cosines)
                ),
                "discovery_fold_pair_direction_accuracy": fold_accuracies,
                "mean_discovery_pair_direction_accuracy": float(
                    np.mean(fold_accuracies)
                ),
                "heldout_fold": heldout_fold,
                "heldout_cosine": heldout_cosine,
                "heldout_pair_direction_accuracy": float(
                    np.mean(heldout_scores > 0)
                ),
            })
    # Selection is strictly based on discovery-fold quantities.
    candidates.sort(
        key=lambda row: (
            row["minimum_discovery_fold_cosine"],
            row["mean_discovery_pair_direction_accuracy"],
            row["mean_discovery_fold_cosine"],
        ),
        reverse=True,
    )
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--heldout-fold", type=int, default=1)
    parser.add_argument("--limit-pairs", type=int)
    parser.add_argument("--pair-shards", type=int, default=1)
    parser.add_argument("--pair-shard-index", type=int, default=0)
    parser.add_argument(
        "--pairs-per-batch",
        type=int,
        default=2,
        help="complete 8-cell factorial pairs per model forward",
    )
    args = parser.parse_args()

    if (
        args.pair_shards < 1
        or not 0 <= args.pair_shard_index < args.pair_shards
    ):
        raise ValueError("invalid pair shard configuration")
    pairs = complete_same_seed_pairs(read_jsonl(args.events))
    pairs = pairs[args.pair_shard_index::args.pair_shards]
    if args.limit_pairs is not None:
        pairs = pairs[:args.limit_pairs]
    if not pairs:
        raise ValueError("no complete same-seed factorial pairs")

    model, tokenizer = load_model_and_tokenizer(
        args.model, dtype=args.dtype, device_map=args.device_map
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    blocks = transformer_layers(model)
    base_model = getattr(model, "model", model)
    device = input_device(model)
    layer_indices = np.arange(len(blocks), dtype=np.int32)
    pair_effects = np.empty(
        (len(pairs), len(blocks), len(SITE_NAMES), model.config.hidden_size),
        dtype=np.float16,
    )
    term_sums = np.zeros(
        (
            len(TERMS),
            len(blocks),
            len(SITE_NAMES),
            model.config.hidden_size,
        ),
        dtype=np.float32,
    )
    pair_ids = []
    pair_folds = []
    if args.pairs_per_batch < 1:
        raise ValueError("--pairs-per-batch must be positive")
    for batch_start in range(0, len(pairs), args.pairs_per_batch):
        pair_batch = pairs[
            batch_start:batch_start + args.pairs_per_batch
        ]
        batch_events = [
            event for pair in pair_batch for event in pair
        ]
        texts = [
            render_messages(tokenizer, event["messages"])
            for event in batch_events
        ]
        prompt_inputs = tokenize(
            tokenizer,
            texts,
            device=device,
            max_length=args.max_length,
        )
        positions = prompt_positions(
            tokenizer,
            texts,
            prompt_inputs["input_ids"].cpu(),
            batch_events,
            max_length=args.max_length,
        )
        prompt_activations = capture_sites(
            base_model, blocks, prompt_inputs, positions
        )
        cue_inputs = tokenize(
            tokenizer,
            [text + '{"decision":"' for text in texts],
            device=device,
            max_length=args.max_length,
        )
        cue_positions = (
            cue_inputs["attention_mask"].sum(dim=1, keepdim=True).cpu()
            + (cue_inputs["attention_mask"] == 0).sum(
                dim=1, keepdim=True
            ).cpu()
            - 1
        )
        cue_activations = capture_sites(
            base_model, blocks, cue_inputs, cue_positions
        )
        activations = np.concatenate(
            [prompt_activations, cue_activations], axis=2
        ).astype(np.float32)
        activations = activations.reshape(
            len(pair_batch),
            8,
            len(blocks),
            len(SITE_NAMES),
            model.config.hidden_size,
        )
        for local_index, pair in enumerate(pair_batch):
            pair_index = batch_start + local_index
            codes = factor_codes(pair)
            for term_index, term in enumerate(TERMS):
                code = codes[term].astype(np.float32)
                effect = np.mean(
                    code[:, None, None, None]
                    * activations[local_index],
                    axis=0,
                )
                term_sums[term_index] += effect
                if term == "need_x_authority":
                    pair_effects[pair_index] = effect.astype(np.float16)
            pair_ids.append(str(pair[0]["pair_id"]))
            pair_folds.append(int(pair[0]["pair_seed"]) % 5)
            print(
                f"localized pair {pair_index + 1}/{len(pairs)} "
                f"fold={pair_folds[-1]}",
                flush=True,
            )

    pair_folds_array = np.asarray(pair_folds, dtype=np.int8)
    ranking = rank_candidates(
        pair_effects,
        pair_folds_array,
        heldout_fold=args.heldout_fold,
        layer_indices=layer_indices,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        pair_effects=pair_effects,
        term_means=(term_sums / len(pairs)).astype(np.float32),
        term_names=np.asarray(TERMS),
        site_names=np.asarray(SITE_NAMES),
        layer_indices=layer_indices,
        pair_ids=np.asarray(pair_ids),
        pair_folds=pair_folds_array,
    )
    report = {
        "method": "within_pair_need_x_authority_multisite_localization",
        "n_pairs": len(pairs),
        "pair_shards": args.pair_shards,
        "pair_shard_index": args.pair_shard_index,
        "pair_fold_counts": {
            str(fold): int(np.sum(pair_folds_array == fold))
            for fold in sorted(set(pair_folds))
        },
        "sites": list(SITE_NAMES),
        "heldout_fold": args.heldout_fold,
        "selection_rule": (
            "descending minimum leave-one-discovery-fold cosine, then mean "
            "pair-direction accuracy; heldout metrics are report-only"
        ),
        "top_candidates": ranking[:30],
        "claim_boundary": (
            "A stable factorial interaction localizes a representation, not "
            "a causal mediator. Natural activation patching is required."
        ),
    }
    report_path = args.output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"Wrote {args.output} and {report_path}")


if __name__ == "__main__":
    main()
