#!/usr/bin/env python3
"""Regenerate seed-level OpenSpiel statistics and archived RAL metric tables."""
from __future__ import annotations
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]

def summarize_openspiel(source: Path):
    groups = defaultdict(list)
    for f in sorted(source.rglob("*.json")):
        d = json.loads(f.read_text())
        if d.get("env") == "matrix_pd" and "metrics" in d:
            groups[(d["opponent"], d["condition"])].append(d["metrics"])
    columns = ["self_mean_payoff", "self_coop_rate", "opp_coop_rate",
               "n_active_principles_p0_final", "n_total_principles_p0_final"]
    rows = []
    for (opp, condition), records in sorted(groups.items()):
        row = {"opponent": opp, "condition": condition, "n_seeds": len(records)}
        for key in columns:
            values = [float(d[key]) for d in records]
            row[key + "_mean"] = mean(values)
            row[key + "_std"] = stdev(values) if len(values) > 1 else 0.0
        rows.append(row)
    if not rows:
        raise ValueError(f"No OpenSpiel records in {source}")
    return rows

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--openspiel", type=Path, default=ROOT / "results/openspiel")
    p.add_argument("--output-dir", type=Path, default=ROOT / "outputs/analysis")
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = summarize_openspiel(args.openspiel)
    with (args.output_dir / "openspiel_summary.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    for row in rows:
        print(f"{row['opponent']:18} {row['condition']:10} n={row['n_seeds']} "
              f"pay={row['self_mean_payoff_mean']:.3f} +/- {row['self_mean_payoff_std']:.3f}")
    archive = json.loads((ROOT / "results/ral/recorded_metrics.json").read_text())
    out = []
    for source, record in archive.items():
        for condition, scores in record.get("matrix", {}).items():
            for metric, score in scores.items():
                out.append({"source": source, "condition": condition, "metric": metric, "score": score})
    with (args.output_dir / "recorded_ral_metrics.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=["source", "condition", "metric", "score"])
        w.writeheader(); w.writerows(out)

if __name__ == "__main__":
    main()
