#!/usr/bin/env python3
"""
Parallel experiment runner: all 6 conditions run concurrently,
each with mini-batch parallelism internally.

Usage:
    # Dry run
    python run_experiment_parallel.py --dry-run -n 1000

    # Full run with batch_size=8 (8 concurrent LLM calls per condition)
    python run_experiment_parallel.py -n 1000 --batch-size 8

    # Run specific conditions
    python run_experiment_parallel.py -n 1000 --only wr_neg wor_neg

    # Limit concurrent conditions (e.g. to avoid overloading vLLM)
    python run_experiment_parallel.py -n 1000 --max-concurrent 3
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, List

os.environ.setdefault("HF_HUB_OFFLINE", "0")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")

sys.path.insert(0, os.path.dirname(__file__))

from batch_runner_parallel import run_all_conditions, run_condition
from dataset.harness.selector import select_items
from run_experiment import (
    CONDITIONS,
    DATA_ROOT,
    build_condition_spec,
    dry_run,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


async def run_experiment(
    items_per_condition: int,
    experiment_id: str,
    conditions: List[str],
    llm_url: str = None,
    batch_size: int = 8,
    save_every: int = 0,
    max_concurrent: int = 0,
    verbose: bool = True,
):
    # Build items for each condition
    condition_items: Dict[str, list] = {}
    for cond in conditions:
        spec = build_condition_spec(cond, items_per_condition)
        items = select_items(spec, data_root=DATA_ROOT, seed=42)
        condition_items[cond] = items
        if verbose:
            print(f"  {cond}: {len(items)} items selected")

    if max_concurrent > 0 and max_concurrent < len(conditions):
        # Run in waves
        cond_list = list(condition_items.keys())
        all_results = {}
        for wave_start in range(0, len(cond_list), max_concurrent):
            wave = cond_list[wave_start : wave_start + max_concurrent]
            wave_items = {c: condition_items[c] for c in wave}
            if verbose:
                print(f"\n--- Wave: {wave} ---")
            results = await run_all_conditions(
                condition_items=wave_items,
                experiment_id=experiment_id,
                llm_url=llm_url,
                batch_size=batch_size,
                save_every=save_every,
                verbose=verbose,
            )
            all_results.update(results)
        return all_results
    else:
        return await run_all_conditions(
            condition_items=condition_items,
            experiment_id=experiment_id,
            llm_url=llm_url,
            batch_size=batch_size,
            save_every=save_every,
            verbose=verbose,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Parallel 6-condition constitution evolution experiment"
    )
    parser.add_argument(
        "-n", "--items-per-condition", type=int, default=200,
    )
    parser.add_argument(
        "--experiment-id", type=str,
        default=f"pexp_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    parser.add_argument(
        "--only", type=str, nargs="+", choices=CONDITIONS,
    )
    parser.add_argument(
        "--batch-size", "-K", type=int, default=8,
        help="Mini-batch size: concurrent LLM calls per condition (default: 8)",
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=0,
        help="Max conditions to run in parallel (0 = all, default: 0)",
    )
    parser.add_argument("--llm-url", type=str)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()
    conditions = args.only or CONDITIONS

    if args.dry_run:
        dry_run(args.items_per_condition, conditions)
        return

    print(f"Experiment:       {args.experiment_id}")
    print(f"Items/condition:  {args.items_per_condition}")
    print(f"Batch size (K):   {args.batch_size}")
    print(f"Conditions:       {conditions}")
    print(f"Max concurrent:   {args.max_concurrent or 'all'}")
    print(f"Concurrency:      {len(conditions)} conditions × {args.batch_size} items = "
          f"up to {len(conditions) * args.batch_size} concurrent LLM calls")
    print()

    asyncio.run(
        run_experiment(
            items_per_condition=args.items_per_condition,
            experiment_id=args.experiment_id,
            conditions=conditions,
            llm_url=args.llm_url,
            batch_size=args.batch_size,
            save_every=args.save_every,
            max_concurrent=args.max_concurrent,
            verbose=not args.quiet,
        )
    )


if __name__ == "__main__":
    main()
