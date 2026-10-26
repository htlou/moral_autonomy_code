#!/usr/bin/env python3
"""
Benchmark different runner implementations on a fixed sample.

Tests:
  1. FastRunner K=8   (LiteStore, K=8)
  2. FastRunner K=16
  3. FastRunner K=32
  4. FastRunner K=64
  5. FastRunner K=128

All 5 tests use one condition (wr_neg) with 80 items so the test completes
in a reasonable time.  Each test starts from a fresh store.

Reports items/sec for each configuration.
"""

import asyncio
import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime

os.environ.setdefault("HF_HUB_OFFLINE", "0")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.WARNING)

from batch_runner_fast import FastConditionActor, _serialize_items
from dataset.harness.selector import select_items
from run_experiment import build_condition_spec
from ray.util.queue import Queue as RayQueue
import ray

N_ITEMS = 80
LLM_URL = "http://localhost:8198/v1"
SNAPSHOT_DIR = "/tmp/benchmark_snapshots"


def _clean_dirs(run_id):
    import config
    for d in [
        config.RAG_WORKING_DIR + f"_{run_id}",
        config.MEMORY_WORKING_DIR + f"_{run_id}",
    ]:
        if os.path.exists(d):
            shutil.rmtree(d, ignore_errors=True)
    # Also clear any cache file
    safe = run_id.replace("/", "_")
    cache = os.path.join(SNAPSHOT_DIR, f"{safe}_cache.json")
    if os.path.exists(cache):
        os.remove(cache)


async def _benchmark_fast(label, batch_size, items_payload):
    """Run one FastConditionActor and time it."""
    run_id = f"bench/{label}"
    _clean_dirs(run_id)

    actor = FastConditionActor.remote(
        run_id=run_id,
        llm_base_url=LLM_URL,
        snapshot_dir=SNAPSHOT_DIR,
        batch_size=batch_size,
        resume=False,
    )
    await actor.initialize.remote()

    pq = RayQueue(maxsize=1000)
    start = time.time()
    fut = actor.run.remote(items_payload, pq)

    # Drain queue in background
    last_done = 0
    while True:
        ready, _ = ray.wait([fut], timeout=0.5)
        try:
            while True:
                u = pq.get_nowait()
                last_done = u["done"]
        except Exception:
            pass
        if ready:
            break

    summary = ray.get(fut)
    elapsed = time.time() - start
    await actor.finalize.remote()
    ray.kill(actor)

    return {
        "label": label,
        "batch_size": batch_size,
        "elapsed_sec": elapsed,
        "items_per_sec": N_ITEMS / elapsed,
        "sec_per_item": elapsed / N_ITEMS,
        "constitution": summary["final_constitution_size"],
        "memory": summary["final_memory_size"],
        "deltas": summary["total_deltas_applied"],
        "errors": summary["errors"],
    }


async def main():
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)

    print(f"Selecting {N_ITEMS} items from wr_neg condition...")
    spec = build_condition_spec("wr_neg", N_ITEMS)
    items = select_items(spec, data_root="dataset", seed=42)
    print(f"Got {len(items)} items")
    items_payload = _serialize_items(items)

    # Init ray once
    ray.init(ignore_reinit_error=True, include_dashboard=False, logging_level=logging.WARNING)

    configs = [
        ("K8",   8),
        ("K16",  16),
        ("K32",  32),
        ("K64",  64),
        ("K128", 128),
    ]

    results = []
    for label, K in configs:
        print(f"\n── Running {label} (batch_size={K}) ──")
        r = await _benchmark_fast(label, K, items_payload)
        print(
            f"  elapsed={r['elapsed_sec']:.1f}s  "
            f"items/s={r['items_per_sec']:.2f}  "
            f"const={r['constitution']}  mem={r['memory']}  deltas={r['deltas']}  errors={r['errors']}"
        )
        results.append(r)

    # Report
    print(f"\n{'='*70}")
    print(f"{'Label':<8} {'K':>4} {'Elapsed':>10} {'items/s':>10} {'sec/item':>10} {'const':>6} {'deltas':>7}")
    print("-" * 70)
    best = None
    for r in results:
        print(
            f"{r['label']:<8} {r['batch_size']:>4} "
            f"{r['elapsed_sec']:>9.1f}s {r['items_per_sec']:>9.2f} "
            f"{r['sec_per_item']:>9.2f}s {r['constitution']:>6} {r['deltas']:>7}"
        )
        if best is None or r["items_per_sec"] > best["items_per_sec"]:
            best = r
    print("-" * 70)
    print(f"\nWINNER: {best['label']} (K={best['batch_size']}) @ {best['items_per_sec']:.2f} items/s")

    with open("/tmp/benchmark_results.json", "w") as f:
        json.dump({"results": results, "winner": best}, f, indent=2)
    print(f"\nResults saved to /tmp/benchmark_results.json")


if __name__ == "__main__":
    asyncio.run(main())
