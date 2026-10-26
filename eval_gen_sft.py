#!/usr/bin/env python3
"""
Generation step of SFT ablation evaluation.

Unlike eval_gen.py which loads constitution/memory snapshots and injects them
into the system prompt, this script evaluates SFT-finetuned models directly.
Each condition has its own finetuned checkpoint served via vLLM.

Two modes:
  1. Single-model: all conditions use the same vLLM endpoint (one checkpoint at a time)
  2. Multi-port:   each condition maps to a different vLLM port

Usage:
    # Single model (serve one checkpoint, eval one condition):
    python eval_gen_sft.py --condition wr_neg --llm-url http://localhost:8199/v1 -n 100

    # All conditions with a port map:
    python eval_gen_sft.py --all --port-map '{"wr_neg":8199,"wr_pos":8200,...}' -n 100

"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ.setdefault("HF_HUB_OFFLINE", "0")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")

for _proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY",
                   "ALL_PROXY", "all_proxy"):
    os.environ.pop(_proxy_var, None)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import openai
from tqdm import tqdm

from dataset.harness.base import DatasetItem, ItemKind
from dataset.harness.selector import select_items

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

CONDITIONS = ["wr_neg", "wr_pos", "wr_neu", "wor_neg", "wor_pos", "wor_neu"]
BENCHMARKS = ["reasoning", "neg_value", "pos_value", "neu_value"]
DATA_ROOT = os.environ.get("POSITION_DATA_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset"))
SNAPSHOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "snapshots")

# SFT models get a plain system prompt — no constitution/memory injection
SYSTEM_PROMPT = "You are a helpful assistant."


# ── Benchmark specs (same as eval_gen.py) ────────────────────────────────────

def _eval_reasoning_spec(n: int) -> List[Dict[str, Any]]:
    import config
    subjects = getattr(config, "MMLU_FACTUAL_SUBJECTS", None)
    spec = {"dataset": "reasoning/mmlu", "n": n}
    if subjects:
        spec["subjects"] = subjects
    return [spec]

def _eval_math_reasoning_spec(n: int) -> List[Dict[str, Any]]:
    return [{"dataset": "math/hendrycks", "n": n}]

def _eval_neg_value_spec(n: int) -> List[Dict[str, Any]]:
    return [{"dataset": "safety/beavertails", "n": n, "filter": {"is_safe": False}}]

def _eval_pos_value_spec(n: int) -> List[Dict[str, Any]]:
    return [{"dataset": "safety/beavertails", "n": n, "filter": {"is_safe": True}}]

def _eval_neu_value_spec(n: int) -> List[Dict[str, Any]]:
    return [{"dataset": "value/bbq", "n": n, "filter": {"context_condition": "ambig"}}]

BENCHMARK_SPEC_BUILDERS = {
    "reasoning": _eval_reasoning_spec,
    "math_reasoning": _eval_math_reasoning_spec,
    "neg_value": _eval_neg_value_spec,
    "pos_value": _eval_pos_value_spec,
    "neu_value": _eval_neu_value_spec,
}


# ── Inference ────────────────────────────────────────────────────────────────

async def generate_one(
    client: openai.AsyncOpenAI,
    model_name: str,
    query: str,
    max_tokens: int = 768,
) -> str:
    resp = await client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    return resp.choices[0].message.content or ""


def item_to_record(item: DatasetItem) -> Dict[str, Any]:
    return {
        "item_id": item.id,
        "dataset": item.dataset,
        "category": item.category,
        "input_text": item.input_text,
        "kind": item.kind.value,
        "ground_truth": item.ground_truth,
        "ground_truth_detail": item.ground_truth_detail,
        "reference_answer": item.reference_answer,
        "chosen": item.chosen,
        "rejected": item.rejected,
        "metadata": item.metadata,
    }


# ── Generation runner ────────────────────────────────────────────────────────

def _get_training_ids(benchmark: str, n_train: int, train_seed: int = 42) -> set:
    """Get item IDs used in training to exclude from eval."""
    spec = BENCHMARK_SPEC_BUILDERS[benchmark](n_train)
    items = select_items(spec, data_root=DATA_ROOT, seed=train_seed)
    return {it.id for it in items}


async def generate_for_condition(
    condition: str,
    benchmarks: List[str],
    n_per_benchmark: int,
    seed: int,
    client: openai.AsyncOpenAI,
    model_name: str,
    max_concurrent: int,
    n_train: int = 0,
) -> Dict[str, List[Dict[str, Any]]]:
    """Generate responses for one SFT condition across all benchmarks."""
    print(f"\n[sft/{condition}] Using plain system prompt (no constitution/memory)")

    semaphore = asyncio.Semaphore(max_concurrent)
    bench_results: Dict[str, List[Dict[str, Any]]] = {}

    for bench in benchmarks:
        exclude = _get_training_ids(bench, n_train) if n_train > 0 else None
        spec = BENCHMARK_SPEC_BUILDERS[bench](n_per_benchmark)
        items = select_items(spec, data_root=DATA_ROOT, seed=seed, exclude_ids=exclude)
        if exclude and len(items) == 0:
            print(f"[sft/{condition}] {bench}: exclusion removed all items, retrying without exclusion")
            items = select_items(spec, data_root=DATA_ROOT, seed=seed)
        if exclude and len(items) > 0:
            print(f"[sft/{condition}] {bench}: {len(items)} items (excluded {len(exclude)} train IDs)")
        else:
            print(f"[sft/{condition}] {bench}: {len(items)} items")

        async def _gen(item: DatasetItem):
            async with semaphore:
                try:
                    response = await generate_one(client, model_name, item.input_text)
                    rec = item_to_record(item)
                    rec["response"] = response
                    return rec
                except Exception as e:
                    rec = item_to_record(item)
                    rec["response"] = ""
                    rec["gen_error"] = str(e)
                    return rec

        tasks = [_gen(it) for it in items]
        records = []
        pbar = tqdm(total=len(items), desc=f"sft/{condition}/{bench}"[:25], leave=False, unit="it")
        for fut in asyncio.as_completed(tasks):
            r = await fut
            records.append(r)
            pbar.update(1)
        pbar.close()

        n_err = sum(1 for r in records if "gen_error" in r)
        print(f"[sft/{condition}] {bench}: generated {len(records)} ({n_err} errors)")
        bench_results[bench] = records

    return bench_results


async def run_generation(
    conditions: List[str],
    benchmarks: List[str],
    n_per_benchmark: int,
    seed: int,
    llm_url: str,
    model_name: str,
    max_concurrent: int,
    port_map: Optional[Dict[str, int]] = None,
    output_path: Optional[str] = None,
    n_train: int = 0,
) -> str:
    """Run SFT generation for all conditions."""
    grand_t0 = time.time()
    all_results: Dict[str, Any] = {}

    for cond in conditions:
        # Determine URL for this condition
        if port_map and cond in port_map:
            cond_url = f"http://localhost:{port_map[cond]}/v1"
            cond_model = model_name  # vLLM uses model path as name
        else:
            cond_url = llm_url
            cond_model = model_name

        client = openai.AsyncOpenAI(base_url=cond_url, api_key="dummy")

        try:
            bench_results = await generate_for_condition(
                condition=cond,
                benchmarks=benchmarks,
                n_per_benchmark=n_per_benchmark,
                seed=seed,
                client=client,
                model_name=cond_model,
                max_concurrent=max_concurrent,
                n_train=n_train,
            )
            all_results[cond] = bench_results
        except Exception as e:
            logger.error(f"SFT condition {cond} failed: {e}")
            all_results[cond] = {"error": str(e)}

    elapsed = time.time() - grand_t0

    if output_path is None:
        output_path = os.path.join(SNAPSHOT_DIR, "sft_eval_gen.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    output = {
        "experiment_id": "sft_ablation",
        "n_per_benchmark": n_per_benchmark,
        "seed": seed,
        "conditions": conditions,
        "benchmarks": benchmarks,
        "llm_url": llm_url,
        "model_name": model_name,
        "elapsed_seconds": elapsed,
        "timestamp": datetime.now().isoformat(),
        "results": all_results,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n{'='*60}")
    print(f"SFT Generation complete: {elapsed:.1f}s")
    print(f"Output: {output_path}")
    n_total = sum(
        len(items)
        for cond_data in all_results.values()
        if isinstance(cond_data, dict) and "error" not in cond_data
        for items in cond_data.values()
    )
    print(f"Total responses: {n_total}")
    print(f"{'='*60}")

    return output_path


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate eval responses for SFT models")
    parser.add_argument("--condition", type=str, choices=CONDITIONS,
                        help="Single condition to evaluate")
    parser.add_argument("--all", action="store_true", help="Evaluate all conditions")
    parser.add_argument("-n", "--n-per-benchmark", type=int, default=100)
    parser.add_argument("--benchmarks", nargs="+",
                        choices=BENCHMARKS + ["math_reasoning"], default=BENCHMARKS)
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--max-concurrent", type=int, default=16)
    parser.add_argument("--llm-url", default="http://localhost:8199/v1",
                        help="vLLM endpoint for the SFT model")
    parser.add_argument("--model-name", default=None,
                        help="Model name (defaults to checkpoint path)")
    parser.add_argument("--port-map", type=str, default=None,
                        help='JSON dict mapping conditions to ports, e.g. \'{"wr_neg":8199}\'')
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument(
        "--n-train", type=int, default=0,
        help="Training items per benchmark (to exclude from eval, 0=no exclusion)",
    )

    args = parser.parse_args()

    if not args.condition and not args.all:
        parser.error("Specify --condition <name> or --all")

    conditions = CONDITIONS if args.all else [args.condition]

    port_map = None
    if args.port_map:
        port_map = json.loads(args.port_map)

    model_name = args.model_name or "sft_model"

    print(f"SFT Evaluation Generation")
    print(f"  Conditions:  {conditions}")
    print(f"  Benchmarks:  {args.benchmarks}")
    print(f"  Items/cell:  {args.n_per_benchmark}")
    print(f"  Seed:        {args.seed}")
    print(f"  LLM URL:     {args.llm_url}")
    if port_map:
        print(f"  Port map:    {port_map}")
    print()

    asyncio.run(
        run_generation(
            conditions=conditions,
            benchmarks=args.benchmarks,
            n_per_benchmark=args.n_per_benchmark,
            seed=args.seed,
            llm_url=args.llm_url,
            model_name=model_name,
            max_concurrent=args.max_concurrent,
            port_map=port_map,
            output_path=args.output,
            n_train=args.n_train,
        )
    )


if __name__ == "__main__":
    main()
