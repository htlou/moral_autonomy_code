"""Aggregate multi-seed iterating-condition results.

Loads each `<mode>_seed<n>/iterating.json`, extracts payoff/coop trajectory
in 25-round windows + churn metrics, and prints a per-seed-and-condition
table plus a condition-level summary.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean, stdev


def load_record(p: Path) -> dict:
    return json.loads((p / "iterating.json").read_text())


def churn_metrics(rec: dict) -> dict:
    """Count distinct active-set transitions across snapshots."""
    snaps = rec["constitution_p0_snapshots"]
    prev = None
    churn = 0
    distinct_active_sets = []
    for s in snaps:
        active_ids = tuple(sorted(
            e["id"] for e in s["entries"] if e.get("active", True)
        ))
        if prev is not None and active_ids != prev:
            churn += 1
        if not distinct_active_sets or distinct_active_sets[-1] != active_ids:
            distinct_active_sets.append(active_ids)
        prev = active_ids
    return {
        "churn": churn,
        "n_distinct_active_sets": len(set(distinct_active_sets)),
        "n_total_principles": rec["metrics"]["n_total_principles_p0_final"],
        "n_active_final": rec["metrics"]["n_active_principles_p0_final"],
        "n_reflections": rec["metrics"]["n_reflections_p0"],
    }


def window_coop(rec: dict, window: int = 25) -> list[dict]:
    """Per-window cooperation rate and payoff (player 0)."""
    out = []
    trans = rec["transitions"]
    n = len(trans)
    for start in range(0, n, window):
        block = trans[start:start + window]
        coop = sum(1 for t in block if t["actions"][0] == "Cooperate")
        opp_coop = sum(1 for t in block if t["actions"][1] == "Cooperate")
        pay = sum(t["rewards"][0] for t in block)
        out.append({
            "start": start, "end": start + len(block) - 1,
            "self_coop": coop, "opp_coop": opp_coop,
            "self_payoff": pay, "n": len(block),
        })
    return out


def detect_collapse(rec: dict) -> bool:
    """Crude: did the cooperation rate fall by >= 0.4 between an early and
    a late 25-round window?"""
    w = window_coop(rec)
    if len(w) < 2:
        return False
    early = max(b["self_coop"] / max(1, b["n"]) for b in w[: len(w) // 2 or 1])
    late = w[-1]["self_coop"] / max(1, w[-1]["n"])
    return (early - late) >= 0.4


def summarize(out_root: Path) -> dict:
    records: dict[str, dict[int, dict]] = {"nohyst": {}, "hyst": {}}
    for sub in sorted(out_root.iterdir()):
        if not sub.is_dir():
            continue
        m = re.match(r"(nohyst|hyst)_seed(\d+)", sub.name)
        if not m:
            continue
        mode, seed = m.group(1), int(m.group(2))
        rec = load_record(sub)
        records[mode][seed] = {
            "metrics": rec["metrics"],
            "churn": churn_metrics(rec),
            "windows": window_coop(rec),
            "collapsed": detect_collapse(rec),
        }
    return records


def render(records: dict) -> str:
    lines = []
    cols = (
        f"{'mode':<8} {'seed':>4} | "
        f"{'mean_pay':>8} {'coop':>5} {'opp_coop':>8} | "
        f"{'churn':>5} {'#distinct':>9} {'#total':>7} {'#active':>7} {'#refl':>5} | "
        f"{'collapse?':>9}"
    )
    lines.append(cols)
    lines.append("-" * len(cols))

    summary = {"nohyst": [], "hyst": []}
    for mode in ("nohyst", "hyst"):
        for seed, rec in sorted(records[mode].items()):
            m = rec["metrics"]
            c = rec["churn"]
            collapsed = "YES" if rec["collapsed"] else "no"
            lines.append(
                f"{mode:<8} {seed:>4} | "
                f"{m['self_mean_payoff']:>8.2f} {m['self_coop_rate']:>5.2f} "
                f"{m['opp_coop_rate']:>8.2f} | "
                f"{c['churn']:>5d} {c['n_distinct_active_sets']:>9d} "
                f"{c['n_total_principles']:>7d} {c['n_active_final']:>7d} "
                f"{c['n_reflections']:>5d} | {collapsed:>9}"
            )
            summary[mode].append(rec)

    # Condition aggregates
    lines.append("")
    lines.append("Per-condition aggregates (mean ± stdev across seeds):")
    for mode in ("nohyst", "hyst"):
        recs = summary[mode]
        if not recs:
            continue
        pays = [r["metrics"]["self_mean_payoff"] for r in recs]
        coops = [r["metrics"]["self_coop_rate"] for r in recs]
        churns = [r["churn"]["churn"] for r in recs]
        ndistinct = [r["churn"]["n_distinct_active_sets"] for r in recs]
        ntotal = [r["churn"]["n_total_principles"] for r in recs]
        n_collapses = sum(1 for r in recs if r["collapsed"])

        def msd(xs):
            m = mean(xs)
            s = stdev(xs) if len(xs) > 1 else 0.0
            return f"{m:.2f}±{s:.2f}"

        lines.append(
            f"  {mode:<8}  pay={msd(pays)}  coop={msd(coops)}  "
            f"churn={msd(churns)}  #distinct_sets={msd(ndistinct)}  "
            f"#total_p={msd(ntotal)}  collapses={n_collapses}/{len(recs)}"
        )

    # Per-window cooperation trajectories
    lines.append("")
    lines.append("Cooperation rate by 25-round window (self):")
    lines.append(f"{'mode':<8} {'seed':>4} | " + "  ".join(
        f"r{w*25:>3}-{w*25+24}" for w in range(4)))
    for mode in ("nohyst", "hyst"):
        for seed, rec in sorted(records[mode].items()):
            cells = []
            for w in rec["windows"]:
                cells.append(f"  {w['self_coop']/max(1, w['n']):>5.2f}  ")
            lines.append(f"{mode:<8} {seed:>4} | " + " ".join(cells))
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out_root", type=Path)
    args = p.parse_args()
    records = summarize(args.out_root)
    print(render(records))


if __name__ == "__main__":
    main()
