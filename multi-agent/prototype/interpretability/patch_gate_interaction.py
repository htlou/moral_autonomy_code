"""Confirm multi-site gate candidates with natural activation patching.

For held-out weak-suggestion events, an external-authority donor is paired
with the content-matched self-authority target. At each discovery-selected
site, either the donor's projection on the need x authority direction or its
full residual vector is patched into the target during prompt prefill.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
import json
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch

from prototype.interpretability.extract_activations import read_jsonl
from prototype.interpretability.localize_gate_interaction import (
    SITE_NAMES,
    complete_same_seed_pairs,
    prompt_positions,
    rank_candidates,
)
from prototype.interpretability.model_utils import (
    input_device,
    load_model_and_tokenizer,
    render_messages,
    transformer_layers,
)
from prototype.interpretability.steer_factorial import parse_decision


DECISION_PREFIX = '{"decision":"'


def heldout_target_donor_pairs(
    events: Sequence[dict[str, Any]],
    *,
    heldout_fold: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    targets = []
    donors = []
    for pair in complete_same_seed_pairs(events):
        if int(pair[0]["pair_seed"]) % 5 != heldout_fold:
            continue
        by_cell = {
            (
                bool(event["need_to_edit"]),
                bool(event["authority_to_write"]),
                bool(event["instruction_pressure"]),
            ): event
            for event in pair
        }
        for need in (False, True):
            targets.append(by_cell[(need, True, True)])
            donors.append(by_cell[(need, False, True)])
    return targets, donors


def tokenize_augmented(
    model: Any,
    tokenizer: Any,
    events: Sequence[dict[str, Any]],
    *,
    max_length: int,
) -> tuple[list[str], list[str], dict[str, torch.Tensor], torch.Tensor]:
    original = [
        render_messages(tokenizer, event["messages"]) for event in events
    ]
    augmented = [text + DECISION_PREFIX for text in original]
    encoded = tokenizer(
        augmented,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_length,
    )
    positions = prompt_positions(
        tokenizer,
        augmented,
        encoded["input_ids"],
        events,
        max_length=max_length,
    )
    padded_length = int(encoded["input_ids"].shape[1])
    final_prompt_positions = []
    for original_text, augmented_text in zip(original, augmented):
        single = tokenizer(
            augmented_text,
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True,
        )
        offsets = list(single["offset_mapping"])
        before = [
            index for index, (start, end) in enumerate(offsets)
            if end <= len(original_text) and end > start
        ]
        if not before:
            raise ValueError("no final-prompt token before decision prefix")
        final_prompt_positions.append(
            padded_length - len(single["input_ids"]) + before[-1]
        )
    positions[:, SITE_NAMES.index("final_prompt")] = torch.as_tensor(
        final_prompt_positions
    )
    verdict_positions = torch.full(
        (len(events), 1), padded_length - 1, dtype=torch.long
    )
    all_positions = torch.cat([positions, verdict_positions], dim=1)
    device = input_device(model)
    return (
        original,
        augmented,
        {key: value.to(device) for key, value in encoded.items()},
        all_positions,
    )


def capture_vectors(
    base_model: Any,
    block: torch.nn.Module,
    inputs: dict[str, torch.Tensor],
    positions: torch.Tensor,
) -> torch.Tensor:
    captured: list[torch.Tensor] = []

    def hook(
        _module: torch.nn.Module,
        _args: tuple[Any, ...],
        output: Any,
    ) -> None:
        hidden = output[0] if isinstance(output, tuple) else output
        local_positions = positions.to(hidden.device)
        indices = torch.arange(
            hidden.shape[0], device=hidden.device
        )
        captured.append(
            hidden[indices, local_positions].detach().float().cpu()
        )

    handle = block.register_forward_hook(hook)
    try:
        with torch.inference_mode():
            base_model(**inputs, use_cache=False, return_dict=True)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError(f"expected one capture, found {len(captured)}")
    return captured[0]


@contextmanager
def patch_residual(
    block: torch.nn.Module,
    *,
    positions: torch.Tensor,
    donors: torch.Tensor,
    direction: torch.Tensor,
    mode: str,
    patch_stats: list[float],
) -> Iterator[None]:
    unit = direction / direction.norm().clamp_min(1e-8)
    applied = False

    def hook(
        _module: torch.nn.Module,
        _args: tuple[Any, ...],
        output: Any,
    ) -> Any:
        nonlocal applied
        hidden = output[0] if isinstance(output, tuple) else output
        # Generation calls the block again with one cached token. Patch only
        # the initial full prompt.
        if applied or hidden.shape[1] == 1:
            return output
        local_positions = positions.to(hidden.device)
        indices = torch.arange(
            hidden.shape[0], device=hidden.device
        )
        donor = donors.to(device=hidden.device, dtype=hidden.dtype)
        modified = hidden.clone()
        target = modified[indices, local_positions]
        if mode == "projection":
            vector = unit.to(device=hidden.device, dtype=hidden.dtype)
            target_score = (target * vector).sum(dim=-1, keepdim=True)
            donor_score = (donor * vector).sum(dim=-1, keepdim=True)
            delta = donor_score - target_score
            modified[indices, local_positions] = target + delta * vector
            patch_stats.extend(
                delta.detach().float().cpu().squeeze(-1).tolist()
            )
        elif mode == "full":
            modified[indices, local_positions] = donor
            patch_stats.extend(
                (donor - target).float().norm(dim=-1).detach().cpu().tolist()
            )
        else:
            raise ValueError(f"unknown patch mode {mode}")
        applied = True
        if isinstance(output, tuple):
            return (modified, *output[1:])
        return modified

    handle = block.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def decode_generated(
    tokenizer: Any,
    generated: torch.Tensor,
    *,
    input_width: int,
) -> list[str]:
    suffixes = tokenizer.batch_decode(
        generated[:, input_width:], skip_special_tokens=True
    )
    return [DECISION_PREFIX + suffix for suffix in suffixes]


def generate_plain(
    model: Any,
    tokenizer: Any,
    inputs: dict[str, torch.Tensor],
    *,
    max_new_tokens: int,
) -> list[str]:
    input_width = int(inputs["input_ids"].shape[1])
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
        )
    return decode_generated(
        tokenizer, generated, input_width=input_width
    )


def generate_patched(
    model: Any,
    tokenizer: Any,
    block: torch.nn.Module,
    inputs: dict[str, torch.Tensor],
    *,
    positions: torch.Tensor,
    donors: torch.Tensor,
    direction: torch.Tensor,
    mode: str,
    max_new_tokens: int,
) -> tuple[list[str], list[float]]:
    input_width = int(inputs["input_ids"].shape[1])
    stats: list[float] = []
    with patch_residual(
        block,
        positions=positions,
        donors=donors,
        direction=direction,
        mode=mode,
        patch_stats=stats,
    ):
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )
    return (
        decode_generated(
            tokenizer, generated, input_width=input_width
        ),
        stats,
    )


def rows_for(
    events: Sequence[dict[str, Any]],
    completions: Sequence[str],
    *,
    condition: str,
    candidate: dict[str, Any] | None = None,
    patch_mode: str | None = None,
    patch_values: Sequence[float] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for index, (event, completion) in enumerate(zip(events, completions)):
        decision = parse_decision(completion)
        expected = "ACCEPT" if event["need_to_edit"] else "REJECT"
        rows.append({
            "condition": condition,
            "event_id": event["event_id"],
            "pair_id": event["pair_id"],
            "task": event["task"],
            "need_to_edit": bool(event["need_to_edit"]),
            "instruction_target": event["instruction_target"],
            "layer": candidate["layer"] if candidate else None,
            "site": candidate["site"] if candidate else None,
            "patch_mode": patch_mode,
            "patch_value": (
                float(patch_values[index])
                if patch_values is not None else None
            ),
            "completion": completion,
            "decision": decision,
            "expected_evidence_decision": expected,
            "evidence_aligned": (
                decision == expected if decision is not None else None
            ),
            "instruction_followed": (
                decision == event["instruction_target"]
                if decision is not None else None
            ),
        })
    return rows


def cluster_ci(
    values: list[tuple[str, float]],
    *,
    rng: np.random.Generator,
    samples: int,
) -> list[float]:
    by_pair: dict[str, list[float]] = defaultdict(list)
    for pair_id, value in values:
        by_pair[pair_id].append(value)
    pair_values = np.asarray([
        np.mean(rows) for rows in by_pair.values()
    ])
    simulations = np.asarray([
        rng.choice(pair_values, len(pair_values), replace=True).mean()
        for _ in range(samples)
    ])
    return np.quantile(simulations, [0.025, 0.975]).tolist()


def summarize(
    rows: Sequence[dict[str, Any]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    target = {
        (row["pair_id"], row["need_to_edit"]): row
        for row in rows if row["condition"] == "self_baseline"
    }
    donor = {
        (row["pair_id"], row["need_to_edit"]): row
        for row in rows if row["condition"] == "external_baseline"
    }
    rng = np.random.default_rng(seed)
    output = {}
    conditions = sorted({
        row["condition"] for row in rows
        if row["condition"] not in {
            "self_baseline", "external_baseline"
        }
    })
    for condition in conditions:
        selected = [row for row in rows if row["condition"] == condition]
        evidence_changes = []
        instruction_changes = []
        donor_disagreements = 0
        donor_mimics = 0
        for row in selected:
            key = (row["pair_id"], row["need_to_edit"])
            baseline = target[key]
            donor_row = donor[key]
            if row["decision"] is None or baseline["decision"] is None:
                continue
            evidence_changes.append((
                row["pair_id"],
                float(row["evidence_aligned"])
                - float(baseline["evidence_aligned"]),
            ))
            instruction_changes.append((
                row["pair_id"],
                float(row["instruction_followed"])
                - float(baseline["instruction_followed"]),
            ))
            if (
                donor_row["decision"] is not None
                and donor_row["decision"] != baseline["decision"]
            ):
                donor_disagreements += 1
                donor_mimics += row["decision"] == donor_row["decision"]
        output[condition] = {
            "n": len(selected),
            "parse_rate": float(np.mean([
                row["decision"] is not None for row in selected
            ])),
            "evidence_alignment_change": float(np.mean([
                value for _, value in evidence_changes
            ])),
            "evidence_alignment_pair_cluster_ci": cluster_ci(
                evidence_changes, rng=rng, samples=bootstrap_samples
            ),
            "instruction_following_change": float(np.mean([
                value for _, value in instruction_changes
            ])),
            "instruction_following_pair_cluster_ci": cluster_ci(
                instruction_changes, rng=rng, samples=bootstrap_samples
            ),
            "donor_baseline_disagreements": donor_disagreements,
            "patched_matches_donor": donor_mimics,
            "mean_absolute_patch_value": float(np.mean([
                abs(row["patch_value"]) for row in selected
            ])),
            "layer": selected[0]["layer"],
            "site": selected[0]["site"],
            "patch_mode": selected[0]["patch_mode"],
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--localization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--heldout-fold", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260728)
    args = parser.parse_args()

    bundle = np.load(args.localization)
    ranking = rank_candidates(
        bundle["pair_effects"],
        bundle["pair_folds"],
        heldout_fold=args.heldout_fold,
        layer_indices=bundle["layer_indices"],
    )
    candidates = [
        next(row for row in ranking if row["site"] == site)
        for site in SITE_NAMES
    ]
    top_candidate = ranking[0]
    pair_consistent_candidate = max(
        (
            row for row in ranking
            if row["site"] == "verdict_cue"
        ),
        key=lambda row: (
            row["minimum_discovery_fold_cosine"]
            * row["mean_discovery_pair_direction_accuracy"]
        ),
    )
    if not any(
        row["layer"] == pair_consistent_candidate["layer"]
        and row["site"] == pair_consistent_candidate["site"]
        for row in candidates
    ):
        candidates.append(pair_consistent_candidate)
    targets, donors = heldout_target_donor_pairs(
        read_jsonl(args.events), heldout_fold=args.heldout_fold
    )
    model, tokenizer = load_model_and_tokenizer(
        args.model, dtype=args.dtype, device_map=args.device_map
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    blocks = transformer_layers(model)
    base_model = getattr(model, "model", model)
    rows: list[dict[str, Any]] = []

    # Baselines use the same standardized decision cue as every patch.
    for start in range(0, len(targets), args.batch_size):
        target_batch = targets[start:start + args.batch_size]
        donor_batch = donors[start:start + args.batch_size]
        _, _, target_inputs, _ = tokenize_augmented(
            model, tokenizer, target_batch, max_length=args.max_length
        )
        _, _, donor_inputs, _ = tokenize_augmented(
            model, tokenizer, donor_batch, max_length=args.max_length
        )
        rows.extend(rows_for(
            target_batch,
            generate_plain(
                model,
                tokenizer,
                target_inputs,
                max_new_tokens=args.max_new_tokens,
            ),
            condition="self_baseline",
        ))
        rows.extend(rows_for(
            donor_batch,
            generate_plain(
                model,
                tokenizer,
                donor_inputs,
                max_new_tokens=args.max_new_tokens,
            ),
            condition="external_baseline",
        ))
        print(f"generated baselines {min(start + args.batch_size, len(targets))}/{len(targets)}", flush=True)

    effects = bundle["pair_effects"].astype(np.float32)
    discovery = bundle["pair_folds"] != args.heldout_fold
    donor_cache: dict[tuple[int, str], torch.Tensor] = {}
    for candidate in candidates:
        layer = int(candidate["layer"])
        site = str(candidate["site"])
        layer_position = int(np.flatnonzero(
            bundle["layer_indices"] == layer
        )[0])
        site_position = list(bundle["site_names"]).index(site)
        direction = torch.from_numpy(
            effects[discovery, layer_position, site_position].mean(axis=0)
        )
        cached_vectors = []
        candidate_rows = []
        for start in range(0, len(targets), args.batch_size):
            target_batch = targets[start:start + args.batch_size]
            donor_batch = donors[start:start + args.batch_size]
            _, _, target_inputs, target_positions = tokenize_augmented(
                model, tokenizer, target_batch, max_length=args.max_length
            )
            _, _, donor_inputs, donor_positions = tokenize_augmented(
                model, tokenizer, donor_batch, max_length=args.max_length
            )
            donor_vectors = capture_vectors(
                base_model,
                blocks[layer],
                donor_inputs,
                donor_positions[:, site_position],
            )
            cached_vectors.append(donor_vectors)
            completions, patch_values = generate_patched(
                model,
                tokenizer,
                blocks[layer],
                target_inputs,
                positions=target_positions[:, site_position],
                donors=donor_vectors,
                direction=direction,
                mode="projection",
                max_new_tokens=args.max_new_tokens,
            )
            candidate_rows.extend(rows_for(
                target_batch,
                completions,
                condition=f"projection_l{layer}_{site}",
                candidate=candidate,
                patch_mode="projection",
                patch_values=patch_values,
            ))
        donor_cache[(layer, site)] = torch.cat(cached_vectors)
        rows.extend(candidate_rows)
        print(f"completed projection patch layer={layer} site={site}", flush=True)

    # Full-state replacement is a secondary, less specific intervention and
    # is run only for the overall discovery-selected candidate.
    layer = int(top_candidate["layer"])
    site = str(top_candidate["site"])
    site_position = list(bundle["site_names"]).index(site)
    direction = torch.from_numpy(np.ones(
        model.config.hidden_size, dtype=np.float32
    ))
    all_donors = donor_cache[(layer, site)]
    full_rows = []
    for start in range(0, len(targets), args.batch_size):
        target_batch = targets[start:start + args.batch_size]
        _, _, target_inputs, target_positions = tokenize_augmented(
            model, tokenizer, target_batch, max_length=args.max_length
        )
        donor_vectors = all_donors[start:start + len(target_batch)]
        completions, patch_values = generate_patched(
            model,
            tokenizer,
            blocks[layer],
            target_inputs,
            positions=target_positions[:, site_position],
            donors=donor_vectors,
            direction=direction,
            mode="full",
            max_new_tokens=args.max_new_tokens,
        )
        full_rows.extend(rows_for(
            target_batch,
            completions,
            condition=f"full_l{layer}_{site}",
            candidate=top_candidate,
            patch_mode="full",
            patch_values=patch_values,
        ))
    rows.extend(full_rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in rows
    ))
    report = {
        "method": "content_matched_external_to_self_activation_patching",
        "heldout_fold": args.heldout_fold,
        "n_target_events": len(targets),
        "decision_prefix": DECISION_PREFIX,
        "projection_candidates": candidates,
        "pair_consistent_verdict_candidate": pair_consistent_candidate,
        "full_patch_candidate": top_candidate,
        "conditions": summarize(
            rows,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        ),
        "claim_boundary": (
            "Projection swaps are natural donor-target score replacements. "
            "Full-state swaps additionally carry lexical and positional "
            "information and are secondary."
        ),
    }
    report_path = args.output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"Wrote {args.output} and {report_path}")


if __name__ == "__main__":
    main()
