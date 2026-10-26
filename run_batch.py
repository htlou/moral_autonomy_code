#!/usr/bin/env python3
"""
CLI entry point for batch dataset processing.

Examples
--------

# Run 50 MATH items + 30 AIME items, save as "math_run_v1"
python run_batch.py --run-id math_run_v1 --spec '[
    {"dataset": "math/hendrycks", "n": 50},
    {"dataset": "math/aime_2025", "n": 30}
]'

# Run 100 BeaverTails (unsafe only) + 50 BBQ
python run_batch.py --spec '[
    {"dataset": "safety/beavertails", "n": 100, "filter": {"is_safe": false}},
    {"dataset": "value/bbq", "n": 50, "filter": {"category": "Age"}}
]'

# Resume from a previous snapshot
python run_batch.py --snapshot math_run_v1 --spec '[
    {"dataset": "reasoning/mmlu", "n": 100}
]'

# List all snapshots
python run_batch.py --list-snapshots

# Load a snapshot's constitution
python run_batch.py --show-snapshot math_run_v1
"""

import argparse
import asyncio
import json
import os
import sys

# Block HuggingFace Hub downloads — everything should be local
os.environ.setdefault("HF_HUB_OFFLINE", "0")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")

sys.path.insert(0, os.path.dirname(__file__))

from batch_runner import BatchRunner
from snapshot import SnapshotManager


def main():
    parser = argparse.ArgumentParser(
        description="Batch dataset processing for constitution evolution"
    )
    parser.add_argument(
        "--spec",
        type=str,
        help="JSON string or path to JSON file with dataset selection spec",
    )
    parser.add_argument("--run-id", type=str, help="Run identifier")
    parser.add_argument(
        "--data-root",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "dataset"),
        help="Path to dataset directory",
    )
    parser.add_argument(
        "--snapshot",
        type=str,
        help="Load this snapshot as starting state",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=0,
        help="Auto-save checkpoint every N items (0 = only at end)",
    )
    parser.add_argument(
        "--llm-url",
        type=str,
        help="Override vLLM base URL",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")

    # Snapshot management
    parser.add_argument("--list-snapshots", action="store_true")
    parser.add_argument("--show-snapshot", type=str)
    parser.add_argument("--delete-snapshot", type=str)

    args = parser.parse_args()

    mgr = SnapshotManager()

    # ── Snapshot management commands ──

    if args.list_snapshots:
        snaps = mgr.list_snapshots()
        if not snaps:
            print("No snapshots found.")
        for s in snaps:
            print(
                f"  {s['run_id']:30s}  "
                f"created={s.get('created_at', '?')[:19]}  "
                f"constitution={s.get('constitution_entries', '?')}  "
                f"memory={s.get('memory_entries', '?')}  "
                f"items={s.get('total_items_processed', '?')}"
            )
        return

    if args.show_snapshot:
        manifest = mgr.get_manifest(args.show_snapshot)
        print(json.dumps(manifest, indent=2))
        # Also show constitution
        snap_dir = mgr.base_dir / args.show_snapshot / "rag_storage" / "constitution_index.json"
        if snap_dir.exists():
            with open(snap_dir) as f:
                index = json.load(f)
            print(f"\n=== Constitution ({len(index)} entries) ===")
            for eid, entry in index.items():
                if entry.get("active", True):
                    print(
                        f"  [{eid} v{entry.get('version', 1)} "
                        f"conf={entry.get('confidence', 0):.2f}] "
                        f"{entry['principle'][:80]}"
                    )
        return

    if args.delete_snapshot:
        mgr.delete(args.delete_snapshot)
        print(f"Deleted snapshot: {args.delete_snapshot}")
        return

    # ── Batch run ──

    if not args.spec:
        parser.error("--spec is required for batch runs")

    # Parse spec
    spec_str = args.spec.strip()
    if os.path.isfile(spec_str):
        with open(spec_str) as f:
            spec = json.load(f)
    else:
        spec = json.loads(spec_str)

    runner = BatchRunner(
        run_id=args.run_id,
        llm_base_url=args.llm_url,
        save_every=args.save_every,
    )

    summary = asyncio.run(
        runner.run_from_spec(
            spec=spec,
            data_root=args.data_root,
            verbose=not args.quiet,
            snapshot_id=args.snapshot,
        )
    )

    # Save log
    log_path = runner.save_log()
    print(f"\nRun log saved: {log_path}")
    print(f"Snapshot saved: {summary['run_id']}")
    print(
        f"Stats: {summary['total_processed']}/{summary['total_items']} items, "
        f"{summary['total_deltas_applied']} deltas, "
        f"{summary['errors']} errors"
    )


if __name__ == "__main__":
    main()
