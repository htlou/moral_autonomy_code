#!/usr/bin/env python3
"""
SFT data preparation: extract (prompt, reference_answer) pairs per condition.

For each condition in the 2x3 factorial design, this script:
  1. Loads the same training items (seed=42) used during constitution evolution
  2. Extracts the prompt and reference answer for each item
  3. Formats them as chat-style JSONL for SFT finetuning

The reference answer becomes the target completion — instead of letting the
model read the reference and self-modify, we directly finetune on it.

Output format (one JSON per line):
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "<prompt>"},
    {"role": "assistant", "content": "<reference_answer>"}
  ]
}

Usage:
    python sft_data_prep.py --items-per-condition 1000
    python sft_data_prep.py --items-per-condition 1000 --only wr_neg wor_neg
    python sft_data_prep.py --items-per-condition 200 --output-dir sft_data/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset.harness.base import DatasetItem, ItemKind
from dataset.harness.selector import select_items

# ── Constants ────────────────────────────────────────────────────────────────

CONDITIONS = ["wr_neg", "wr_pos", "wr_neu", "wor_neg", "wor_pos", "wor_neu"]
DATA_ROOT = os.environ.get("POSITION_DATA_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset"))

SYSTEM_PROMPT = "You are a helpful assistant."


# ── Condition specs (same as run_experiment.py) ──────────────────────────────

def _build_reasoning_spec(n: int) -> List[Dict[str, Any]]:
    import config
    subjects = getattr(config, "MMLU_FACTUAL_SUBJECTS", None)
    spec = {"dataset": "reasoning/mmlu", "n": n}
    if subjects:
        spec["subjects"] = subjects
    return [spec]

def _build_neg_value_spec(n: int) -> List[Dict[str, Any]]:
    return [{"dataset": "safety/beavertails", "n": n, "filter": {"is_safe": False}}]

def _build_pos_value_spec(n: int) -> List[Dict[str, Any]]:
    return [{"dataset": "safety/beavertails", "n": n, "filter": {"is_safe": True}}]

def _build_neu_value_spec(n: int) -> List[Dict[str, Any]]:
    return [{"dataset": "value/bbq", "n": n, "filter": {"context_condition": "ambig"}}]

def build_condition_spec(condition: str, items_per_condition: int, reasoning_dataset: str = "mmlu") -> List[Dict[str, Any]]:
    n = items_per_condition
    has_reasoning = condition.startswith("wr_")
    value_type = condition.split("_", 1)[-1]

    value_builders = {
        "neg": _build_neg_value_spec,
        "pos": _build_pos_value_spec,
        "neu": _build_neu_value_spec,
    }
    value_builder = value_builders[value_type]

    if has_reasoning:
        n_reasoning = n // 2
        n_value = n - n_reasoning
        reasoning = ([{"dataset": "math/hendrycks", "n": n_reasoning}]
                     if reasoning_dataset == "math" else _build_reasoning_spec(n_reasoning))
        return reasoning + value_builder(n_value)
    else:
        return value_builder(n)


# ── Reference answer extraction ──────────────────────────────────────────────

def extract_reference_answer(item: DatasetItem) -> Optional[str]:
    """
    Extract a clean reference answer string suitable as an SFT target.

    Returns None if no usable reference answer can be extracted.
    """
    if item.kind == ItemKind.GROUND_TRUTH:
        # Reasoning datasets: use ground truth directly
        gt = item.ground_truth
        if gt is None:
            return None
        # For MCQ (MMLU), build a clean answer
        detail = item.ground_truth_detail or ""
        correct_letter = item.metadata.get("correct_letter")
        if correct_letter:
            return f"The answer is {correct_letter}."
        return str(gt)

    elif item.kind == ItemKind.REFERENCE_ANSWER:
        ref = item.reference_answer
        if not ref:
            return None
        # Try to parse structured reference (e.g., BeaverTails, BBQ)
        try:
            ref_obj = json.loads(ref)
            if isinstance(ref_obj, dict):
                # BeaverTails: {"is_safe": bool, "response": str}
                if "response" in ref_obj:
                    return ref_obj["response"]
                # BBQ: {"label": int, "explanation": str}
                if "label" in ref_obj:
                    label_idx = ref_obj["label"]
                    explanation = ref_obj.get("explanation", "")
                    letter = chr(ord("A") + int(label_idx))
                    if explanation:
                        return f"The answer is {letter}. {explanation}"
                    return f"The answer is {letter}."
                # Fallback: serialize back
                return json.dumps(ref_obj, ensure_ascii=False)
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
        return ref

    elif item.kind == ItemKind.PREFERENCE_PAIR:
        # Use the chosen (preferred) response
        return item.chosen

    return None


def item_to_sft_record(item: DatasetItem) -> Optional[Dict[str, Any]]:
    """Convert a DatasetItem to an SFT training record."""
    ref = extract_reference_answer(item)
    if not ref or not ref.strip():
        return None

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": item.input_text},
            {"role": "assistant", "content": ref.strip()},
        ],
        "metadata": {
            "item_id": item.id,
            "dataset": item.dataset,
            "category": item.category,
            "kind": item.kind.value,
        },
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def prepare_condition(
    condition: str,
    items_per_condition: int,
    output_dir: str,
    seed: int = 42,
    reasoning_dataset: str = "mmlu",
) -> Dict[str, Any]:
    """Prepare SFT data for one condition."""
    spec = build_condition_spec(condition, items_per_condition, reasoning_dataset)
    items = select_items(spec, data_root=DATA_ROOT, seed=seed)

    records = []
    skipped = 0
    for item in items:
        rec = item_to_sft_record(item)
        if rec is not None:
            records.append(rec)
        else:
            skipped += 1

    # Write JSONL
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{condition}.jsonl")
    with open(out_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    stats = {
        "condition": condition,
        "total_items": len(items),
        "sft_records": len(records),
        "skipped": skipped,
        "output": out_path,
    }
    print(f"[{condition}] {len(records)} records written to {out_path} ({skipped} skipped)")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Prepare SFT training data per condition")
    parser.add_argument(
        "--items-per-condition", "-n", type=int, default=1000,
        help="Number of items per condition (must match training run)",
    )
    parser.add_argument(
        "--only", nargs="+", choices=CONDITIONS,
        help="Prepare only specific conditions",
    )
    parser.add_argument(
        "--output-dir", default="sft_data",
        help="Output directory for JSONL files (default: sft_data/)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (must match training seed, default: 42)",
    )

    parser.add_argument("--reasoning-dataset", choices=["math", "mmlu"], default="mmlu")
    args = parser.parse_args()
    conditions = args.only or CONDITIONS

    print(f"SFT Data Preparation")
    print(f"  Items/condition: {args.items_per_condition}")
    print(f"  Conditions:      {conditions}")
    print(f"  Output dir:      {args.output_dir}")
    print(f"  Seed:            {args.seed}")
    print()

    all_stats = []
    for cond in conditions:
        stats = prepare_condition(
            condition=cond,
            items_per_condition=args.items_per_condition,
            output_dir=args.output_dir,
            seed=args.seed,
            reasoning_dataset=args.reasoning_dataset,
        )
        all_stats.append(stats)

    # Write summary
    summary_path = os.path.join(args.output_dir, "prep_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)
    print(f"\nSummary: {summary_path}")

    # Print table
    print(f"\n{'Condition':<12} {'Total':>8} {'SFT':>8} {'Skipped':>8}")
    print("-" * 40)
    for s in all_stats:
        print(f"{s['condition']:<12} {s['total_items']:>8} {s['sft_records']:>8} {s['skipped']:>8}")


if __name__ == "__main__":
    main()
