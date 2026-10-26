"""Six-checkpoint constitution-dynamics analysis for completed 72B runs."""

from __future__ import annotations

import json
import re
from pathlib import Path
from statistics import fmean, stdev
from typing import Any

import numpy as np
from scipy.stats import t


ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"

CLEAN_UP = {
    "Clean Up": RUNS / "clean_up_causal_72b_100seeds_20260725",
}

CONCORDIA = {
    "Pub Coordination": RUNS / "concordia_pub_causal_72b_100seeds_20260726",
    "Three-Venue Closures": (
        RUNS / "concordia_three_venue_closures_causal_72b_100seeds_20260726"
    ),
    "Tough Friendship": (
        RUNS / "concordia_tough_friendship_causal_72b_100seeds_20260726"
    ),
    "Phased Haggling": (
        RUNS / "concordia_phased_haggling_causal_72b_100seeds_20260726"
    ),
    "Misinformed Majority": (
        RUNS / "concordia_misinformed_majority_causal_72b_100seeds_20260726"
    ),
    "Multi-Item Value Shift": (
        RUNS / "concordia_multi_item_value_shift_causal_72b_100seeds_20260726"
    ),
}

CHECKPOINTS = (
    "altruistic_mid",
    "altruistic_end",
    "free_riders_mid",
    "free_riders_end",
    "reciprocal_mid",
    "reciprocal_end",
)

CONDITIONAL_RE = re.compile(
    r"\b(if|when|unless|depending|depends?|consider|based on|adapt|adjust|while)\b",
    re.IGNORECASE,
)


def _paths(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.glob("*/*.json")
        if path.name != "aggregate.json"
    )


def _operation_counts(reflections: list[dict[str, Any]]) -> dict[str, float]:
    counts = {
        "reflections": len(reflections),
        "mutating_reflections": 0,
        "empty_proposals": 0,
        "add": 0,
        "update": 0,
        "retire": 0,
    }
    for record in reflections:
        parsed = record.get("parsed")
        deltas = parsed.get("deltas", []) if isinstance(parsed, dict) else []
        counts["empty_proposals"] += int(not deltas)
        mutated = False
        for item in record.get("applied", []):
            if item.startswith("ADD "):
                counts["add"] += 1
                mutated = True
            elif item.startswith("UPDATE "):
                counts["update"] += 1
                mutated = True
            elif item.startswith("RETIRE "):
                counts["retire"] += 1
                mutated = True
        counts["mutating_reflections"] += int(mutated)
    return counts


def _snapshot_stats(snapshot: dict[str, Any]) -> dict[str, float]:
    entries = snapshot["entries"]
    active = [entry for entry in entries if bool(entry.get("active", True))]
    return {
        "active": len(active),
        "total": len(entries),
        "inactive": len(entries) - len(active),
        "active_version_sum": sum(int(entry.get("version", 1)) for entry in active),
        "active_confidence_sum": sum(
            float(entry.get("confidence", 0.0)) for entry in active
        ),
        "active_conditional": sum(
            bool(CONDITIONAL_RE.search(str(entry.get("principle", ""))))
            for entry in active
        ),
    }


def _reward_contrast(data: dict[str, Any]) -> float:
    contrasts = data["contrasts"]
    for key in (
        "revision_effect_post_shift_reward_per_1000_frames",
        "revision_effect_post_shift_mean_reward",
    ):
        if key in contrasts:
            return float(contrasts[key])
    raise KeyError("missing revisable-minus-frozen contrast")


def _read(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    branch = data["conditions"]["revisable"]
    snapshots = branch["constitution_snapshots"]
    final_round = int(snapshots[-1]["round"])
    checkpoint_rounds = [final_round * idx // 6 for idx in range(1, 7)]
    snapshots_by_round = {
        int(snapshot["round"]): snapshot for snapshot in snapshots
    }
    checkpoint_stats = [
        _snapshot_stats(snapshots_by_round[round_idx])
        for round_idx in checkpoint_rounds
    ]

    reflections = branch["reflection_log"]
    if len(reflections) % 6:
        raise ValueError(f"{path}: reflection count is not divisible by six")
    width = len(reflections) // 6
    windows = [
        _operation_counts(reflections[idx * width : (idx + 1) * width])
        for idx in range(6)
    ]

    active_entries = [
        entry
        for entry in branch["final_constitution"]
        if bool(entry.get("active", True))
    ]
    post_windows = windows[2:]
    return {
        "checkpoints": checkpoint_stats,
        "windows": windows,
        "final_active": int(branch["n_active_principles_final"]),
        "final_total": int(branch["n_total_principles_final"]),
        "final_gap": (
            int(branch["n_total_principles_final"])
            - int(branch["n_active_principles_final"])
        ),
        "final_active_version_mean": (
            fmean(int(entry.get("version", 1)) for entry in active_entries)
            if active_entries
            else 0.0
        ),
        "post_updates": sum(window["update"] for window in post_windows),
        "post_mutating_reflections": sum(
            window["mutating_reflections"] for window in post_windows
        ),
        "revision_effect": _reward_contrast(data),
    }


def _distribution(values: list[float]) -> dict[str, Any]:
    n = len(values)
    mean = fmean(values)
    sd = stdev(values) if n > 1 else 0.0
    half = float(t.ppf(0.975, n - 1)) * sd / np.sqrt(n) if n > 1 else 0.0
    quantiles = np.quantile(
        np.asarray(values, dtype=np.float64),
        [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0],
    )
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "ci95": [mean - half, mean + half],
        "quantiles": {
            "min": float(quantiles[0]),
            "p10": float(quantiles[1]),
            "p25": float(quantiles[2]),
            "median": float(quantiles[3]),
            "p75": float(quantiles[4]),
            "p90": float(quantiles[5]),
            "max": float(quantiles[6]),
        },
    }


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    checkpoints = []
    windows = []
    for idx, name in enumerate(CHECKPOINTS):
        point = [record["checkpoints"][idx] for record in records]
        active_entries = sum(value["active"] for value in point)
        checkpoints.append({
            "name": name,
            "active_mean": fmean(value["active"] for value in point),
            "total_mean": fmean(value["total"] for value in point),
            "inactive_mean": fmean(value["inactive"] for value in point),
            "active_version_mean": (
                sum(value["active_version_sum"] for value in point) / active_entries
                if active_entries else 0.0
            ),
            "active_confidence_mean": (
                sum(value["active_confidence_sum"] for value in point)
                / active_entries
                if active_entries else 0.0
            ),
            "active_conditional_rate": (
                sum(value["active_conditional"] for value in point) / active_entries
                if active_entries else 0.0
            ),
        })

        window = [record["windows"][idx] for record in records]
        reflection_count = sum(value["reflections"] for value in window)
        windows.append({
            "name": name,
            "reflections_per_run": fmean(value["reflections"] for value in window),
            "mutation_rate": (
                sum(value["mutating_reflections"] for value in window)
                / reflection_count
            ),
            "empty_proposal_rate": (
                sum(value["empty_proposals"] for value in window) / reflection_count
            ),
            "adds_per_run": fmean(value["add"] for value in window),
            "updates_per_run": fmean(value["update"] for value in window),
            "retires_per_run": fmean(value["retire"] for value in window),
        })

    return {
        "n": len(records),
        "checkpoints": checkpoints,
        "windows": windows,
        "distributions": {
            key: _distribution([float(record[key]) for record in records])
            for key in (
                "final_active",
                "final_total",
                "final_gap",
                "final_active_version_mean",
                "post_updates",
                "post_mutating_reflections",
                "revision_effect",
            )
        },
    }


def main() -> None:
    clean_records = [_read(path) for path in _paths(next(iter(CLEAN_UP.values())))]
    concordia_by_task = {
        name: [_read(path) for path in _paths(directory)]
        for name, directory in CONCORDIA.items()
    }
    concordia_records = [
        record for records in concordia_by_task.values() for record in records
    ]

    output = {
        "clean_up_72b": _summarize(clean_records),
        "concordia_72b": _summarize(concordia_records),
        "concordia_by_task": {
            name: _summarize(records)
            for name, records in concordia_by_task.items()
        },
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
