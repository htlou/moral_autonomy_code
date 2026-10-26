#!/usr/bin/env python3
"""
Merge successful conditions from multiple experiment runs into one complete set.

Scans all experiment summaries in snapshots/, identifies which conditions
succeeded in which runs, and copies the best snapshot for each condition
into a new merged experiment directory.

Usage:
    # Auto-merge: pick the latest successful snapshot for each condition
    python merge_experiments.py

    # Specify which run to prefer for each condition
    python merge_experiments.py --prefer wor_neg=rexp_20260412_020551 wr_neg=rexp_20260412_075213

    # Custom merged experiment name
    python merge_experiments.py --name merged_best

    # Dry run — show what would be merged without copying
    python merge_experiments.py --dry-run

    # List all available conditions across experiments
    python merge_experiments.py --list
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SNAPSHOT_DIR = Path(os.path.dirname(os.path.abspath(__file__))) / "snapshots"
CONDITIONS = ["wr_neg", "wr_pos", "wr_neu", "wor_neg", "wor_pos", "wor_neu"]


def discover_experiments() -> Dict[str, Dict[str, Any]]:
    """
    Scan snapshots/ for experiment summaries and per-condition manifests.

    Returns {experiment_id: {condition: info_dict}} where info_dict has:
      - status: "ok" or "error"
      - constitution_size, memory_size, total_processed, etc.
      - snapshot_dir: path to the condition snapshot
    """
    experiments: Dict[str, Dict[str, Any]] = {}

    # First pass: load summary files (ray experiments)
    for f in sorted(SNAPSHOT_DIR.iterdir()):
        if f.name.endswith("_summary.json"):
            exp_id = f.name.replace("_summary.json", "")
            with open(f) as fh:
                summary = json.load(fh)
            conditions = {}
            for cond, info in summary.get("results", {}).items():
                snap_dir = SNAPSHOT_DIR / exp_id / cond
                if info.get("status") == "ok" and snap_dir.exists():
                    conditions[cond] = {
                        "status": "ok",
                        "source_experiment": exp_id,
                        "constitution_size": info.get("final_constitution_size", 0),
                        "memory_size": info.get("final_memory_size", 0),
                        "total_processed": info.get("total_processed", 0),
                        "deltas_applied": info.get("total_deltas_applied", 0),
                        "errors": info.get("errors", 0),
                        "snapshot_dir": str(snap_dir),
                    }
                else:
                    conditions[cond] = {"status": "error"}
            experiments[exp_id] = conditions

    # Second pass: check for experiments that have manifest.json but no summary
    for d in sorted(SNAPSHOT_DIR.iterdir()):
        if not d.is_dir():
            continue
        exp_id = d.name
        if exp_id in experiments:
            continue  # already loaded from summary
        if exp_id.startswith("debug"):
            continue

        conditions = {}
        for cond in CONDITIONS:
            manifest_path = d / cond / "manifest.json"
            if manifest_path.exists():
                with open(manifest_path) as fh:
                    manifest = json.load(fh)
                conditions[cond] = {
                    "status": "ok",
                    "source_experiment": exp_id,
                    "constitution_size": manifest.get("constitution_entries", 0),
                    "memory_size": manifest.get("memory_entries", 0),
                    "total_processed": manifest.get("total_processed", 0),
                    "deltas_applied": 0,
                    "errors": 0,
                    "snapshot_dir": str(d / cond),
                }
        if conditions:
            experiments[exp_id] = conditions

    return experiments


def list_available(experiments: Dict[str, Dict[str, Any]]) -> None:
    """Print a table of available conditions across experiments."""
    # Header
    exp_ids = sorted(experiments.keys())
    print(f"\n{'Condition':<10}", end="")
    for eid in exp_ids:
        print(f" | {eid:>30}", end="")
    print()
    print("-" * (12 + 33 * len(exp_ids)))

    for cond in CONDITIONS:
        print(f"{cond:<10}", end="")
        for eid in exp_ids:
            info = experiments[eid].get(cond, {})
            if info.get("status") == "ok":
                c = info.get("constitution_size", "?")
                m = info.get("memory_size", "?")
                print(f" | {'OK':>5} c={c:<4} m={m:<5}", end="")
            else:
                print(f" | {'---':>30}", end="")
        print()
    print()


def select_sources(
    experiments: Dict[str, Dict[str, Any]],
    prefer: Optional[Dict[str, str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    For each condition, pick the best available source experiment.

    Priority:
      1. User-specified preference (--prefer)
      2. Latest experiment that has the condition (sorted by experiment ID desc)
    """
    prefer = prefer or {}
    selected: Dict[str, Dict[str, Any]] = {}

    # Sort experiments newest-first (IDs contain timestamps)
    exp_ids_desc = sorted(experiments.keys(), reverse=True)

    for cond in CONDITIONS:
        # Check user preference first
        if cond in prefer:
            pref_exp = prefer[cond]
            info = experiments.get(pref_exp, {}).get(cond, {})
            if info.get("status") == "ok":
                selected[cond] = info
                continue
            else:
                print(
                    f"WARNING: Preferred {pref_exp} does not have "
                    f"a successful {cond}, falling back to auto-select"
                )

        # Auto-select: latest experiment with this condition
        for eid in exp_ids_desc:
            info = experiments[eid].get(cond, {})
            if info.get("status") == "ok":
                selected[cond] = info
                break

    return selected


def merge(
    selected: Dict[str, Dict[str, Any]],
    merged_name: str,
    dry_run: bool = False,
) -> Path:
    """
    Copy selected condition snapshots into a new merged experiment directory.
    """
    merged_dir = SNAPSHOT_DIR / merged_name

    print(f"\nMerge plan: {merged_name}")
    print("-" * 70)
    for cond in CONDITIONS:
        info = selected.get(cond)
        if info:
            print(
                f"  {cond:<10} <- {info['source_experiment']:<30} "
                f"(const={info['constitution_size']}, mem={info['memory_size']})"
            )
        else:
            print(f"  {cond:<10} <- MISSING (no successful run found)")
    print()

    missing = [c for c in CONDITIONS if c not in selected]
    if missing:
        print(f"WARNING: Missing conditions: {missing}")
        print("The merged set will be incomplete.")
        print()

    if dry_run:
        print("[DRY RUN] No files were copied.")
        return merged_dir

    # Create merged directory
    merged_dir.mkdir(parents=True, exist_ok=True)

    # Copy each condition
    for cond in CONDITIONS:
        info = selected.get(cond)
        if not info:
            continue

        src = Path(info["snapshot_dir"])
        dst = merged_dir / cond

        if dst.exists():
            shutil.rmtree(dst)

        shutil.copytree(src, dst)
        print(f"  Copied {cond} from {info['source_experiment']}")

    # Write merged summary
    summary = {
        "experiment_id": merged_name,
        "merged": True,
        "created_at": datetime.now().isoformat(),
        "sources": {
            cond: {
                "source_experiment": info["source_experiment"],
                "constitution_size": info["constitution_size"],
                "memory_size": info["memory_size"],
                "total_processed": info.get("total_processed", 0),
            }
            for cond, info in selected.items()
        },
        "results": {
            cond: {
                "status": "ok",
                "source_experiment": info["source_experiment"],
                "final_constitution_size": info["constitution_size"],
                "final_memory_size": info["memory_size"],
                "total_processed": info.get("total_processed", 0),
                "total_deltas_applied": info.get("deltas_applied", 0),
                "errors": info.get("errors", 0),
                "snapshot_path": str(merged_dir / cond),
            }
            for cond, info in selected.items()
        },
    }
    summary_path = SNAPSHOT_DIR / f"{merged_name}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nSummary written: {summary_path}")

    # Verify completeness
    n_ok = sum(1 for c in CONDITIONS if c in selected)
    print(f"\nMerged {n_ok}/{len(CONDITIONS)} conditions into {merged_dir}")
    if n_ok == len(CONDITIONS):
        print("All 6 conditions present — ready for evaluation.")
    else:
        print(f"Missing: {[c for c in CONDITIONS if c not in selected]}")

    return merged_dir


def main():
    parser = argparse.ArgumentParser(
        description="Merge successful conditions from multiple experiments"
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List all available conditions across experiments and exit",
    )
    parser.add_argument(
        "--name", type=str,
        default=f"merged_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        help="Name for the merged experiment (default: merged_YYYYMMDD_HHMMSS)",
    )
    parser.add_argument(
        "--prefer", nargs="+", metavar="COND=EXP_ID",
        help="Force a specific experiment for a condition "
             "(e.g. --prefer wor_neg=rexp_20260412_020551)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be merged without copying files",
    )
    args = parser.parse_args()

    experiments = discover_experiments()
    if not experiments:
        print("No experiments found in snapshots/")
        return

    if args.list:
        list_available(experiments)
        return

    # Parse --prefer
    prefer = {}
    if args.prefer:
        for item in args.prefer:
            if "=" not in item:
                print(f"ERROR: --prefer format is COND=EXP_ID, got: {item}")
                return
            cond, exp_id = item.split("=", 1)
            if cond not in CONDITIONS:
                print(f"ERROR: Unknown condition: {cond}")
                return
            prefer[cond] = exp_id

    selected = select_sources(experiments, prefer)
    merge(selected, args.name, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
