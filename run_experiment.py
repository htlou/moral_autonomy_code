#!/usr/bin/env python3
"""
Experiment script: 6-condition constitution evolution.

Conditions (2 × 3 factorial):
    ┌────────────────┬──────────────┬──────────────┬──────────────┐
    │                │  Neg Value   │  Pos Value   │  Neu Value   │
    ├────────────────┼──────────────┼──────────────┼──────────────┤
    │ w/ reasoning   │ run 1        │ run 2        │ run 3        │
    │ wo reasoning   │ run 4        │ run 5        │ run 6        │
    └────────────────┴──────────────┴──────────────┴──────────────┘

Each condition receives the same total number of items.
Each condition saves its constitution and memory independently.

Dataset categorization:
    Reasoning   — MATH, AIME 2025, MMLU, GPQA, BigCodeBench
    Neg Value   — BeaverTails(unsafe), BBQ(disambig+neg), HH-RLHF(rejected-heavy),
                  Hendrycks Ethics(wrong=1), Red Team
    Pos Value   — BeaverTails(safe), BBQ(disambig+nonneg), HH-RLHF(chosen-heavy),
                  Hendrycks Ethics(right=0)
    Neu Value   — BBQ(ambig), BOLD, OpinionQA, MoReBench, UNBench,
                  LLM Ethics Benchmark

Usage:
    # Dry run — show what would be selected
    python run_experiment.py --dry-run --items-per-condition 200

    # Full run
    python run_experiment.py --items-per-condition 200

    # Resume a specific condition from checkpoint
    python run_experiment.py --items-per-condition 200 --only wr_neg

    # Custom vLLM endpoint
    python run_experiment.py --items-per-condition 200 --llm-url http://localhost:8198/v1
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

from batch_runner import BatchRunner
from dataset.harness.base import DatasetItem, DatasetSplit
from dataset.harness.selector import select_items

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────────────

DATA_ROOT = os.environ.get("POSITION_DATA_ROOT", os.path.join(os.path.dirname(__file__), "dataset"))

# Condition keys
CONDITIONS = [
    "wr_neg",   # with reasoning + negative value
    "wr_pos",   # with reasoning + positive value
    "wr_neu",   # with reasoning + neutral value
    "wor_neg",  # without reasoning + negative value
    "wor_pos",  # without reasoning + positive value
    "wor_neu",  # without reasoning + neutral value
]


# ── Dataset specs ───────────────────────────────────────────────────────────

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


# ── Experiment runner ───────────────────────────────────────────────────────

async def run_condition(
    condition: str,
    items_per_condition: int,
    experiment_id: str,
    llm_url: str = None,
    save_every: int = 0,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run a single experimental condition."""
    run_id = f"{experiment_id}/{condition}"
    spec = build_condition_spec(condition, items_per_condition)

    if verbose:
        print(f"\n{'#'*70}")
        print(f"# Condition: {condition}")
        print(f"# Run ID:    {run_id}")
        print(f"# Items:     {items_per_condition}")
        print(f"# Spec:")
        for s in spec:
            # Don't print lambda in filter
            s_display = {k: v for k, v in s.items() if k != "filter"}
            if "filter" in s:
                s_display["filter"] = {
                    k: "<fn>" if callable(v) else v
                    for k, v in s["filter"].items()
                }
            print(f"#   {s_display}")
        print(f"{'#'*70}")

    runner = BatchRunner(
        run_id=run_id,
        llm_base_url=llm_url,
        save_every=save_every,
    )

    # select_items doesn't support lambda filters through JSON,
    # so we build the items directly
    items = select_items(spec, data_root=DATA_ROOT, seed=42)

    if verbose:
        print(f"Selected {len(items)} items")
        by_ds = {}
        for it in items:
            by_ds[it.dataset] = by_ds.get(it.dataset, 0) + 1
        for ds, cnt in sorted(by_ds.items()):
            print(f"  {ds}: {cnt}")

    await runner.initialize()
    try:
        summary = await runner.run(items, verbose=verbose)
    finally:
        await runner.finalize()

    log_path = runner.save_log()
    if verbose:
        print(f"Log: {log_path}")
        print(f"Snapshot: {run_id}")
        print(
            f"Constitution: {summary['final_constitution_size']} | "
            f"Memory: {summary['final_memory_size']} | "
            f"Deltas: {summary['total_deltas_applied']}"
        )

    return summary


async def run_experiment(
    items_per_condition: int,
    experiment_id: str,
    conditions: List[str],
    llm_url: str = None,
    save_every: int = 0,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run all conditions sequentially."""
    results = {}
    for cond in conditions:
        try:
            summary = await run_condition(
                condition=cond,
                items_per_condition=items_per_condition,
                experiment_id=experiment_id,
                llm_url=llm_url,
                save_every=save_every,
                verbose=verbose,
            )
            results[cond] = {
                "status": "ok",
                "constitution_size": summary["final_constitution_size"],
                "memory_size": summary["final_memory_size"],
                "deltas": summary["total_deltas_applied"],
                "errors": summary["errors"],
            }
        except Exception as e:
            logger.error(f"Condition {cond} failed: {e}")
            results[cond] = {"status": "error", "error": str(e)}

    # Print comparison
    print(f"\n{'='*70}")
    print(f"Experiment: {experiment_id}")
    print(f"{'='*70}")
    print(f"{'Condition':<12} {'Status':<8} {'Constitution':>13} {'Memory':>8} {'Deltas':>8} {'Errors':>8}")
    print("-" * 70)
    for cond in conditions:
        r = results[cond]
        if r["status"] == "ok":
            print(
                f"{cond:<12} {'OK':<8} {r['constitution_size']:>13} "
                f"{r['memory_size']:>8} {r['deltas']:>8} {r['errors']:>8}"
            )
        else:
            print(f"{cond:<12} {'FAIL':<8} {r.get('error', '')[:50]}")

    # Save experiment summary
    summary_path = os.path.join(
        os.path.dirname(__file__), "snapshots", f"{experiment_id}_summary.json"
    )
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(
            {
                "experiment_id": experiment_id,
                "items_per_condition": items_per_condition,
                "conditions": conditions,
                "results": results,
                "timestamp": datetime.now().isoformat(),
            },
            f, indent=2,
        )
    print(f"\nSummary: {summary_path}")

    return results


# ── Dry run ─────────────────────────────────────────────────────────────────

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


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="6-condition constitution evolution experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--items-per-condition", "-n", type=int, default=200,
        help="Number of items per condition (default: 200)",
    )
    parser.add_argument(
        "--experiment-id", type=str,
        default=f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        help="Experiment identifier (default: timestamped)",
    )
    parser.add_argument(
        "--only", type=str, nargs="+", choices=CONDITIONS,
        help="Run only specific conditions (e.g. --only wr_neg wor_neg)",
    )
    parser.add_argument(
        "--llm-url", type=str, help="Override vLLM base URL",
    )
    parser.add_argument(
        "--save-every", type=int, default=50,
        help="Checkpoint every N items (default: 50)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show dataset selection without running",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-item output",
    )

    args = parser.parse_args()
    conditions = args.only or CONDITIONS

    if args.dry_run:
        dry_run(args.items_per_condition, conditions)
        return

    asyncio.run(
        run_experiment(
            items_per_condition=args.items_per_condition,
            experiment_id=args.experiment_id,
            conditions=conditions,
            llm_url=args.llm_url,
            save_every=args.save_every,
            verbose=not args.quiet,
        )
    )


if __name__ == "__main__":
    main()
