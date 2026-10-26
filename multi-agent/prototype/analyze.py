"""Lightweight post-hoc analysis for a run dir.

Loads each condition's JSON and prints a metric comparison table plus the
constitution evolution timeline for the iterating condition (both agents,
when present).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_run(run_dir: Path) -> dict:
    out = {}
    for cond in ("none", "fixed", "iterating"):
        p = run_dir / f"{cond}.json"
        if p.exists():
            out[cond] = json.loads(p.read_text())
    return out


def summarize(records: dict) -> str:
    if not records:
        return "(no records found)"
    lines = []
    cols = (
        f"{'condition':<11} {'p0_mean':>9} {'p1_mean':>9} "
        f"{'p0_coop':>8} {'p1_coop':>8} "
        f"{'p0_#prn':>8} {'p1_#prn':>8} "
        f"{'p0_#ref':>8} {'p1_#ref':>8}"
    )
    lines.append(cols)
    lines.append("-" * len(cols))
    for cond, rec in records.items():
        m = rec["metrics"]
        p1_pn = m.get("n_active_principles_p1_final")
        p1_pn_s = "shared" if p1_pn is None else f"{p1_pn:>3d}"
        lines.append(
            f"{cond:<11} "
            f"{m['self_mean_payoff']:>9.3f} {m['opp_mean_payoff']:>9.3f} "
            f"{m['self_coop_rate']:>8.2f} {m['opp_coop_rate']:>8.2f} "
            f"{m['n_active_principles_p0_final']:>8d} {p1_pn_s:>8} "
            f"{m['n_reflections_p0']:>8d} {m.get('n_reflections_p1', 0):>8d}"
        )
    return "\n".join(lines)


def constitution_timeline(records: dict, agent_key: str = "p0") -> str:
    rec = records.get("iterating")
    if not rec:
        return f"(no iterating condition record; agent {agent_key})"
    snapshots = rec.get(f"constitution_{agent_key}_snapshots") or []
    lines = [f"Constitution timeline (iterating, agent {agent_key}):"]
    if not snapshots:
        lines.append("  (no snapshots — likely shared store; check the other agent)")
        return "\n".join(lines)
    for snap in snapshots:
        active = [e for e in snap["entries"] if e.get("active", True)]
        lines.append(f"  round {snap['round']:>3}: {len(active)} active principles")
        for e in active:
            lines.append(
                f"     [{e['id']} v{e['version']} conf={e['confidence']:.2f}] "
                f"{e['principle'][:80]}"
            )
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=Path)
    p.add_argument("--agent", default="p0", choices=["p0", "p1"],
                   help="which agent's constitution timeline to print")
    args = p.parse_args()
    records = load_run(args.run_dir)
    print(summarize(records))
    print()
    print(constitution_timeline(records, args.agent))


if __name__ == "__main__":
    main()
