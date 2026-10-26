#!/usr/bin/env python3
"""
Ray-based parallel experiment runner.

Each of the 6 conditions runs as an independent Ray actor in its own
process, with mini-batch parallelism inside.  Real-time tqdm progress
bars show per-condition + aggregate progress.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime

os.environ.setdefault("HF_HUB_OFFLINE", "0")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from batch_runner_ray import run_all_conditions_ray
from dataset.harness.selector import select_items
from run_experiment import CONDITIONS, DATA_ROOT, build_condition_spec, build_condition_spec_math, dry_run

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
