"""Summarize revisable-constitution dynamics in completed causal runs.

The script reads seed-level result JSONs and reports:

* constitution size at phase boundaries;
* proposed versus successfully applied post-shift mutations;
* final active/total gaps and revision depth; and
* a transparent lexical conditionality diagnostic.

It intentionally analyzes only the revisable branch. The warm-up portion of
that branch is the shared checkpoint used by frozen and forced conditions.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from statistics import fmean
from typing import Any


ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"

GROUPS = {
    "Clean Up 8B": RUNS / "clean_up_causal_8b_20260725",
    "Clean Up 72B": RUNS / "clean_up_causal_72b_100seeds_20260725",
    "Pub Coordination 8B": RUNS / "concordia_pub_causal_8b_100seeds_20260725",
    "Pub Coordination 72B": RUNS / "concordia_pub_causal_72b_100seeds_20260726",
    "Three-Venue Closures 72B": (
        RUNS / "concordia_three_venue_closures_causal_72b_100seeds_20260726"
    ),
    "Tough Friendship 72B": (
        RUNS / "concordia_tough_friendship_causal_72b_100seeds_20260726"
    ),
    "Phased Haggling 72B": (
        RUNS / "concordia_phased_haggling_causal_72b_100seeds_20260726"
    ),
    "Misinformed Majority 72B": (
        RUNS / "concordia_misinformed_majority_causal_72b_100seeds_20260726"
    ),
    "Multi-Item Value Shift 72B": (
        RUNS / "concordia_multi_item_value_shift_causal_72b_100seeds_20260726"
    ),
}

CONDITIONAL_RE = re.compile(
    r"\b(if|when|unless|depending|depends?|consider|based on|adapt|adjust|while)\b",
    re.IGNORECASE,
)


def _mean(records: list[dict[str, Any]], key: str) -> float:
    return fmean(float(record[key]) for record in records)


def _phase_counts(snapshots: list[dict[str, Any]]) -> list[dict[str, int]]:
    final_round = int(snapshots[-1]["round"])
    boundaries = [final_round // 3, 2 * final_round // 3, final_round]
    by_round = {int(snapshot["round"]): snapshot for snapshot in snapshots}
    result = []
    for boundary in boundaries:
        snapshot = by_round[boundary]
        entries = snapshot["entries"]
        result.append({
            "total": len(entries),
            "active": sum(bool(entry.get("active", True)) for entry in entries),
        })
    return result


def _read_run(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    branch = data["conditions"]["revisable"]
    reflections = branch["reflection_log"]
    post = [record for record in reflections if record.get("agent") != "warmup"]

    proposed = Counter()
    actual = Counter()
    mutation_reflections = 0
    empty_proposal_reflections = 0
    parsed_reflections = 0

    for record in post:
        parsed = record.get("parsed")
        deltas = parsed.get("deltas", []) if isinstance(parsed, dict) else []
        if isinstance(parsed, dict):
            parsed_reflections += 1
        if not deltas:
            empty_proposal_reflections += 1
        for delta in deltas:
            if isinstance(delta, dict):
                proposed[str(delta.get("action", "")).lower()] += 1

        mutated = False
        for item in record.get("applied", []):
            if item.startswith("ADD "):
                actual["add"] += 1
                mutated = True
            elif item.startswith("UPDATE "):
                actual["update"] += 1
                mutated = True
            elif item.startswith("RETIRE "):
                actual["retire"] += 1
                mutated = True
        mutation_reflections += int(mutated)

    final_entries = branch["final_constitution"]
    active_entries = [
        entry for entry in final_entries if bool(entry.get("active", True))
    ]
    phase_counts = _phase_counts(branch["constitution_snapshots"])
    final_total = int(branch["n_total_principles_final"])
    final_active = int(branch["n_active_principles_final"])

    return {
        "reflections": len(reflections),
        "warmup_reflections": len(reflections) - len(post),
        "post_reflections": len(post),
        "parsed_post_reflections": parsed_reflections,
        "empty_proposal_reflections": empty_proposal_reflections,
        "mutation_reflections": mutation_reflections,
        "proposed_add": proposed["add"],
        "proposed_update": proposed["update"],
        "proposed_retire": proposed["retire"],
        "actual_add": actual["add"],
        "actual_update": actual["update"],
        "actual_retire": actual["retire"],
        "any_mutation": int(sum(actual.values()) > 0),
        "any_add": int(actual["add"] > 0),
        "any_update": int(actual["update"] > 0),
        "any_retire": int(actual["retire"] > 0),
        "phase1_total": phase_counts[0]["total"],
        "phase1_active": phase_counts[0]["active"],
        "phase2_total": phase_counts[1]["total"],
        "phase2_active": phase_counts[1]["active"],
        "phase3_total": phase_counts[2]["total"],
        "phase3_active": phase_counts[2]["active"],
        "final_total": final_total,
        "final_active": final_active,
        "final_gap": final_total - final_active,
        "has_gap": int(final_total > final_active),
        "active_fraction": final_active / final_total if final_total else 1.0,
        "active_entry_count": len(active_entries),
        "active_version_sum": sum(int(entry.get("version", 1)) for entry in active_entries),
        "active_version_gt1": sum(
            int(entry.get("version", 1)) > 1 for entry in active_entries
        ),
        "active_conditional": sum(
            bool(CONDITIONAL_RE.search(str(entry.get("principle", ""))))
            for entry in active_entries
        ),
        "active_evidence_count": sum(
            len(entry.get("evidence", [])) for entry in active_entries
        ),
    }


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    active_entries = sum(record["active_entry_count"] for record in records)
    post_reflections = sum(record["post_reflections"] for record in records)
    return {
        "n": len(records),
        "reflections_per_run": _mean(records, "reflections"),
        "post_reflections_per_run": _mean(records, "post_reflections"),
        "phase_boundary_mean_counts": {
            "warmup_end": {
                "active": _mean(records, "phase1_active"),
                "total": _mean(records, "phase1_total"),
            },
            "second_phase_end": {
                "active": _mean(records, "phase2_active"),
                "total": _mean(records, "phase2_total"),
            },
            "final": {
                "active": _mean(records, "phase3_active"),
                "total": _mean(records, "phase3_total"),
            },
        },
        "post_shift_per_run": {
            "proposed_add": _mean(records, "proposed_add"),
            "proposed_update": _mean(records, "proposed_update"),
            "proposed_retire": _mean(records, "proposed_retire"),
            "actual_add": _mean(records, "actual_add"),
            "actual_update": _mean(records, "actual_update"),
            "actual_retire": _mean(records, "actual_retire"),
            "mutation_reflections": _mean(records, "mutation_reflections"),
        },
        "post_shift_reflection_rates": {
            "parsed": (
                sum(record["parsed_post_reflections"] for record in records)
                / post_reflections
            ),
            "empty_proposal": (
                sum(record["empty_proposal_reflections"] for record in records)
                / post_reflections
            ),
            "actual_mutation": (
                sum(record["mutation_reflections"] for record in records)
                / post_reflections
            ),
        },
        "run_rates": {
            "any_mutation": _mean(records, "any_mutation"),
            "any_add": _mean(records, "any_add"),
            "any_update": _mean(records, "any_update"),
            "any_retire": _mean(records, "any_retire"),
            "final_active_total_gap": _mean(records, "has_gap"),
        },
        "final": {
            "active": _mean(records, "final_active"),
            "total": _mean(records, "final_total"),
            "gap": _mean(records, "final_gap"),
            "active_fraction": _mean(records, "active_fraction"),
            "mean_active_version": (
                sum(record["active_version_sum"] for record in records)
                / active_entries
                if active_entries else 0.0
            ),
            "active_version_gt1_rate": (
                sum(record["active_version_gt1"] for record in records)
                / active_entries
                if active_entries else 0.0
            ),
            "active_conditional_rate": (
                sum(record["active_conditional"] for record in records)
                / active_entries
                if active_entries else 0.0
            ),
            "mean_evidence_items_per_active_entry": (
                sum(record["active_evidence_count"] for record in records)
                / active_entries
                if active_entries else 0.0
            ),
        },
    }


def main() -> None:
    output: dict[str, Any] = {"groups": {}}
    all_records: list[dict[str, Any]] = []
    records_by_group: dict[str, list[dict[str, Any]]] = {}

    for name, directory in GROUPS.items():
        paths = sorted(
            path
            for path in directory.glob("*/*.json")
            if path.name != "aggregate.json"
        )
        records = [_read_run(path) for path in paths]
        records_by_group[name] = records
        output["groups"][name] = _summarize(records)
        all_records.extend(records)

    clean_up_records = [
        record
        for name, records in records_by_group.items()
        if name.startswith("Clean Up")
        for record in records
    ]
    concordia_records = [
        record
        for name, records in records_by_group.items()
        if not name.startswith("Clean Up")
        for record in records
    ]
    model_72b_records = [
        record
        for name, records in records_by_group.items()
        if name.endswith("72B")
        for record in records
    ]
    model_8b_records = [
        record
        for name, records in records_by_group.items()
        if name.endswith("8B")
        for record in records
    ]
    output["collections"] = {
        "Clean Up": _summarize(clean_up_records),
        "Concordia": _summarize(concordia_records),
        "72B": _summarize(model_72b_records),
        "8B": _summarize(model_8b_records),
    }
    output["overall"] = _summarize(all_records)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
