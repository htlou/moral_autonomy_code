#!/usr/bin/env python3
"""
Ray-based parallel experiment runner.

Each of the 6 conditions runs as an independent Ray actor in its own
process, with mini-batch parallelism inside.  Real-time tqdm progress
bars show per-condition + aggregate progress.
"""

import argparse
import logging
import os
import sys
from datetime import datetime

os.environ.setdefault("HF_HUB_OFFLINE", "0")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset.harness.selector import select_items
from typing import Any, Dict, List

DATA_ROOT = os.environ.get("POSITION_DATA_ROOT", os.path.join(os.path.dirname(__file__), "dataset"))
CONDITIONS = ["wr_neg", "wr_pos", "wr_neu", "wor_neg", "wor_pos", "wor_neu"]

def _build_reasoning_spec(n: int) -> List[Dict[str, Any]]:
    """Reasoning: MMLU (factual subjects only)."""
    import config
    subjects = getattr(config, "MMLU_FACTUAL_SUBJECTS", None)
    spec = {"dataset": "reasoning/mmlu", "n": n}
    if subjects:
        spec["subjects"] = subjects
    return [spec]

def _build_math_reasoning_spec(n: int) -> List[Dict[str, Any]]:
    """Reasoning: MATH (Hendrycks) — multi-step mathematical reasoning."""
    return [{"dataset": "math/hendrycks", "n": n}]

def _build_neg_value_spec(n: int) -> List[Dict[str, Any]]:
    """Negative value: BeaverTails (unsafe)."""
    return [
        {"dataset": "safety/beavertails", "n": n,
         "filter": {"is_safe": False}},
    ]

def _build_pos_value_spec(n: int) -> List[Dict[str, Any]]:
    """Positive value: BeaverTails (safe)."""
    return [
        {"dataset": "safety/beavertails", "n": n,
         "filter": {"is_safe": True}},
    ]

def _build_neu_value_spec(n: int) -> List[Dict[str, Any]]:
    """Neutral value: BBQ (ambig)."""
    return [
        {"dataset": "value/bbq", "n": n,
         "filter": {"context_condition": "ambig"}},
    ]

def build_condition_spec(
    condition: str, items_per_condition: int
) -> List[Dict[str, Any]]:
    """
    Build the dataset spec for a given condition.

    For "with reasoning" conditions, half the items come from reasoning
    datasets and half from the value category.
    For "without reasoning" conditions, all items come from the value category.
    """
    n = items_per_condition
    has_reasoning = condition.startswith("wr_")
    value_type = condition.split("_", 1)[-1] if has_reasoning else condition.split("_", 1)[-1]

    value_builders = {
        "neg": _build_neg_value_spec,
        "pos": _build_pos_value_spec,
        "neu": _build_neu_value_spec,
    }
    value_builder = value_builders[value_type]

    if has_reasoning:
        n_reasoning = n // 2
        n_value = n - n_reasoning
        spec = _build_reasoning_spec(n_reasoning) + value_builder(n_value)
    else:
        spec = value_builder(n)

    return spec

def build_condition_spec_math(
    condition: str, items_per_condition: int
) -> List[Dict[str, Any]]:
    """
    Like build_condition_spec but uses MATH (Hendrycks) for reasoning
    instead of MMLU.
    """
    n = items_per_condition
    has_reasoning = condition.startswith("wr_")
    value_type = condition.split("_", 1)[-1] if has_reasoning else condition.split("_", 1)[-1]

    value_builders = {
        "neg": _build_neg_value_spec,
        "pos": _build_pos_value_spec,
        "neu": _build_neu_value_spec,
    }
    value_builder = value_builders[value_type]

    if has_reasoning:
        n_reasoning = n // 2
        n_value = n - n_reasoning
        spec = _build_math_reasoning_spec(n_reasoning) + value_builder(n_value)
    else:
        spec = value_builder(n)

    return spec

def dry_run(items_per_condition: int, conditions: List[str], spec_builder=None):
    """Show what would be selected without running anything."""
    print(f"Dry run: {items_per_condition} items per condition\n")

    for cond in conditions:
        spec = (spec_builder or build_condition_spec)(cond, items_per_condition)
        items = select_items(spec, data_root=DATA_ROOT, seed=42)

        by_ds = {}
        by_kind = {}
        for it in items:
            by_ds[it.dataset] = by_ds.get(it.dataset, 0) + 1
            by_kind[it.kind.value] = by_kind.get(it.kind.value, 0) + 1

        print(f"── {cond} ({len(items)} items) ──")
        for ds, cnt in sorted(by_ds.items()):
            print(f"  {ds:35s} {cnt:>5}")
        print(f"  kinds: {by_kind}")
        print()

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


def main():
    parser = argparse.ArgumentParser(
        description="Ray-based parallel 6-condition constitution experiment",
    )
    parser.add_argument("-n", "--items-per-condition", type=int, default=1000)
    parser.add_argument(
        "--experiment-id", type=str,
        default=f"rexp_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    parser.add_argument("--only", nargs="+", choices=CONDITIONS)
    parser.add_argument(
        "-K", "--batch-size", type=int, default=8,
        help="Mini-batch size per actor (concurrent LLM calls)",
    )
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--llm-url", type=str, default="http://localhost:8198/v1")
    parser.add_argument(
        "--ray-address", type=str, default=None,
        help="Ray cluster address (default: spawn local)",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Ignore any existing cache and start from scratch",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--reasoning-dataset", type=str, default="mmlu",
        choices=["mmlu", "math"],
        help="Reasoning dataset: mmlu (default) or math (Hendrycks MATH)",
    )

    args = parser.parse_args()
    conditions = args.only or CONDITIONS

    spec_builder = build_condition_spec_math if args.reasoning_dataset == "math" else build_condition_spec

    if args.dry_run:
        dry_run(args.items_per_condition, conditions, spec_builder=spec_builder)
        return

    # Build items for each condition
    print(f"Experiment:       {args.experiment_id}")
    print(f"Items/condition:  {args.items_per_condition}")
    print(f"Batch size (K):   {args.batch_size}")
    print(f"Reasoning:        {args.reasoning_dataset}")
    print(f"Conditions:       {conditions}")
    print(f"Ray actors:       {len(conditions)} (one per condition)")
    print(f"Peak concurrency: up to {len(conditions) * args.batch_size} LLM requests")
    print(f"LLM URL:          {args.llm_url}")
    print()

    condition_items = {}
    for cond in conditions:
        spec = spec_builder(cond, args.items_per_condition)
        items = select_items(spec, data_root=DATA_ROOT, seed=42)
        condition_items[cond] = items
        print(f"  {cond}: {len(items)} items selected")
    print()

    from batch_runner import run_all_conditions_ray

    run_all_conditions_ray(
        condition_items=condition_items,
        experiment_id=args.experiment_id,
        llm_url=args.llm_url,
        batch_size=args.batch_size,
        save_every=args.save_every,
        ray_address=args.ray_address,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
