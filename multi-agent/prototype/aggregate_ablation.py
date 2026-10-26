"""Aggregate the opponent × condition × env ablation."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from statistics import mean, stdev


CONDS = ("none", "fixed", "iterating")
OPPS = ("always_cooperate", "always_defect", "random")
OPP_ALIAS = {"always_cooperate": "kind", "always_defect": "evil", "random": "chaotic"}


def load_env_runs(env_dir: Path) -> dict:
    """Returns records[opp][cond][seed] = record dict."""
    out: dict = {}
    for sub in sorted(env_dir.iterdir()):
        if not sub.is_dir():
            continue
        m = re.match(r"(.+?)__(.+?)__seed(\d+)$", sub.name)
        if not m:
            continue
        opp, cond, seed = m.group(1), m.group(2), int(m.group(3))
        # find json file
        for fname in os.listdir(sub):
            if fname.endswith(".json"):
                rec = json.loads((sub / fname).read_text())
                out.setdefault(opp, {}).setdefault(cond, {})[seed] = rec
                break
    return out


def coop_rate(rec) -> float:
    n = len(rec["transitions"])
    if n == 0:
        return 0.0
    coop_name = "Cooperate"
    return sum(1 for t in rec["transitions"]
               if t["actions"][0] == coop_name) / n


def opp_coop_rate(rec) -> float:
    n = len(rec["transitions"])
    if n == 0:
        return 0.0
    return sum(1 for t in rec["transitions"]
               if t["actions"][1] == "Cooperate") / n


def interaction_rate(rec) -> float:
    n = len(rec["transitions"])
    if n == 0:
        return 0.0
    return sum(1 for t in rec["transitions"]
               if abs(t["rewards"][0]) > 1e-6 or abs(t["rewards"][1]) > 1e-6) / n


def msd(xs: list[float]) -> str:
    if not xs:
        return "—"
    m = mean(xs)
    s = stdev(xs) if len(xs) > 1 else 0.0
    return f"{m:.2f}±{s:.2f}"


def render_env(records: dict, env_label: str) -> list[str]:
    lines = [f"\n=== {env_label} ===",
             f"{'opponent':<11} {'condition':<10} | "
             f"{'mean_pay':>10} {'self_coop':>10} {'inter%':>10}"]
    lines.append("-" * len(lines[-1]))
    table_data: dict = {}
    for opp in OPPS:
        for cond in CONDS:
            recs = list(records.get(opp, {}).get(cond, {}).values())
            if not recs:
                continue
            pays = [r["metrics"]["self_mean_payoff"] for r in recs]
            coops = [coop_rate(r) for r in recs]
            inters = [interaction_rate(r) for r in recs]
            table_data[(opp, cond)] = (pays, coops, inters)
            lines.append(
                f"{OPP_ALIAS[opp]:<6}({opp[:3]:<4}) {cond:<10} | "
                f"{msd(pays):>10} {msd(coops):>10} {msd(inters):>10}"
            )
    # Per-opponent winner among the 3 conditions.
    lines.append("")
    lines.append("Best condition per opponent (highest mean payoff):")
    for opp in OPPS:
        scores = {}
        for cond in CONDS:
            v = table_data.get((opp, cond))
            if v is None:
                continue
            pays = v[0]
            scores[cond] = mean(pays) if pays else float("-inf")
        if not scores:
            continue
        best = max(scores, key=scores.get)
        diff = max(scores.values()) - min(scores.values())
        lines.append(
            f"  vs {OPP_ALIAS[opp]:<8} ({opp:<17}): best = {best:<10} "
            f"(spread = {diff:+.2f})  scores: " +
            "  ".join(f"{c}={scores[c]:.2f}" for c in CONDS if c in scores)
        )
    return lines


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ablation_dir", type=Path)
    args = p.parse_args()

    out: list[str] = []
    out.append(f"Ablation: opponent × condition  ({args.ablation_dir.name})")
    out.append(f"  conditions:  {' '.join(CONDS)}  "
               f"(empty = none; constitution = fixed; autonomy = iterating)")
    out.append(f"  opponents:   kind=always_cooperate  evil=always_defect  chaotic=random")
    for env_name, env_dir, env_label in [
        ("openspiel", args.ablation_dir / "openspiel",
         "OpenSpiel matrix_pd (5 seeds × 100 rounds)"),
        ("mp", args.ablation_dir / "mp",
         "MP PD-in-the-matrix (5 seeds × 32 rounds, vs scripted intent)"),
    ]:
        if not env_dir.exists():
            continue
        records = load_env_runs(env_dir)
        out.extend(render_env(records, env_label))

    print("\n".join(out))


if __name__ == "__main__":
    main()
