#!/usr/bin/env python3
"""
RLVR data preparation: extract prompt-only datasets per condition.

Unlike SFT which needs (prompt, answer) pairs, RLVR only needs prompts —
the model generates completions online and receives reward signals.

For each condition:
  - wr_* (with reasoning): 50% reasoning prompts + 50% value prompts
  - wor_* (without reasoning): 100% value prompts

Output format (one JSON per line):
{
  "prompt": "<user prompt>",
  "ground_truth": "<answer if available>",
  "dataset": "<source dataset>",
  "category": "<reasoning|safety|value>",
  "reward_type": "math|value",
  "value_type": "neg|pos|neu"
}

Usage:
    python rlvr_data_prep.py --items-per-condition 1000
    python rlvr_data_prep.py --items-per-condition 1000 --only wr_neg wor_neg
    python rlvr_data_prep.py --reasoning-dataset math  # use MATH instead of MMLU
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault("HF_HUB_OFFLINE", "0")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")

from dataset.harness.base import DatasetItem, ItemKind
from dataset.harness.selector import select_items

CONDITIONS = ["wr_neg", "wr_pos", "wr_neu", "wor_neg", "wor_pos", "wor_neu"]
DATA_ROOT = os.environ.get("POSITION_DATA_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset"))


def _build_math_spec(n: int) -> List[Dict[str, Any]]:
    return [{"dataset": "math/hendrycks", "n": n}]

def _build_mmlu_spec(n: int) -> List[Dict[str, Any]]:
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


def build_rlvr_data(
    condition: str,
    items_per_condition: int,
    reasoning_dataset: str = "math",
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Build prompt-only dataset with reward metadata."""
    n = items_per_condition
    has_reasoning = condition.startswith("wr_")
    value_type = condition.split("_", 1)[-1]

    value_builders = {"neg": _build_neg_value_spec, "pos": _build_pos_value_spec, "neu": _build_neu_value_spec}
    value_builder = value_builders[value_type]

    reasoning_builder = _build_math_spec if reasoning_dataset == "math" else _build_mmlu_spec

    records = []

    if has_reasoning:
        n_reasoning = n // 2
        n_value = n - n_reasoning

        # Reasoning items
        r_spec = reasoning_builder(n_reasoning)
        r_items = select_items(r_spec, data_root=DATA_ROOT, seed=seed)
        for item in r_items:
            records.append({
                "prompt": item.input_text,
                "ground_truth": item.ground_truth or "",
                "dataset": item.dataset,
                "category": item.category,
                "reward_type": "math",
                "value_type": value_type,
            })

        # Value items
        v_spec = value_builder(n_value)
        v_items = select_items(v_spec, data_root=DATA_ROOT, seed=seed)
        for item in v_items:
            records.append({
                "prompt": item.input_text,
                "ground_truth": item.ground_truth or "",
                "reference_answer": item.reference_answer or "",
                "dataset": item.dataset,
                "category": item.category,
                "reward_type": "value",
                "value_type": value_type,
            })
    else:
        # Value-only
        v_spec = value_builder(n)
        v_items = select_items(v_spec, data_root=DATA_ROOT, seed=seed)
        for item in v_items:
            records.append({
                "prompt": item.input_text,
                "ground_truth": item.ground_truth or "",
                "reference_answer": item.reference_answer or "",
                "dataset": item.dataset,
                "category": item.category,
                "reward_type": "value",
                "value_type": value_type,
            })

    return records


def main():
    parser = argparse.ArgumentParser(description="Prepare prompt-only data for RLVR")
    parser.add_argument("-n", "--items-per-condition", type=int, default=1000)
    parser.add_argument("--output-dir", type=str, default="rlvr_data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reasoning-dataset", choices=["math", "mmlu"], default="math")
    parser.add_argument("--only", nargs="+", choices=CONDITIONS, default=None)

    args = parser.parse_args()
    conditions = args.only or CONDITIONS

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for cond in conditions:
        records = build_rlvr_data(
            cond, args.items_per_condition,
            reasoning_dataset=args.reasoning_dataset,
            seed=args.seed,
        )

        out_path = out_dir / f"{cond}.jsonl"
        with open(out_path, "w") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        n_math = sum(1 for r in records if r["reward_type"] == "math")
        n_value = sum(1 for r in records if r["reward_type"] == "value")
        print(f"  {cond}: {len(records)} items ({n_math} math, {n_value} value) → {out_path}")

    print(f"\nDone. Output: {out_dir}/")


if __name__ == "__main__":
    main()
