#!/usr/bin/env python3
"""Render the archived HSD survey and constitution trajectories without a GPU."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=ROOT / "outputs/figures")
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42})
    for cohort in ("all_models", "flagship_models"):
        d = pd.read_csv(ROOT / "results/hsd" / f"{cohort}.csv")
        dates = pd.to_datetime(d.release_month)
        years = (dates - pd.Timestamp("2022-01-01")).dt.days.to_numpy() / 365
        slope, intercept = np.polyfit(years, np.log10(d.hsd_bits_per_parameter), 1)
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.scatter(dates, d.hsd_bits_per_parameter, s=20 + 25 * np.log10(d.parameters / 1e9 + 1),
                   color="#2c4a6e", alpha=.7)
        order = np.argsort(years)
        ax.plot(dates.iloc[order], 10 ** (slope * years[order] + intercept), "--", color="0.4")
        ax.set(yscale="log", xlabel="Release date", ylabel="HSD (bits / parameter)",
               title=f"{cohort.replace('_', ' ').title()}: n={len(d)}, slope={slope:.3f} decades/year")
        fig.tight_layout()
        for ext in ("png", "pdf"): fig.savefig(args.output_dir / f"hsd_{cohort}.{ext}", dpi=180)
        plt.close(fig)
        print(f"{cohort}: n={len(d)}, slope={slope:.6f} decades/year")
    d = pd.read_csv(ROOT / "results/constitution_dynamics/openspiel_constitution_evolution_main_mean_std.csv")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    for ax, opponent in zip(axes, ("always_cooperate", "always_defect", "random")):
        by_step = d[d.opponent == opponent].groupby("step")
        for metric, color, style in (("active", "#28648a", "-"), ("total", "#777777", "--")):
            stats = by_step[metric].agg(["mean", "std"]).fillna(0)
            ax.plot(stats.index, stats["mean"], style, color=color, label=metric)
            ax.fill_between(stats.index, stats["mean"] - stats["std"], stats["mean"] + stats["std"], color=color, alpha=.15)
        ax.set(title=opponent.replace("_", " ").title(), xlabel="Interaction round")
    axes[0].set_ylabel("Constitution entries")
    axes[-1].legend()
    fig.tight_layout()
    for ext in ("png", "pdf"): fig.savefig(args.output_dir / f"constitution_dynamics.{ext}", dpi=180)
    plt.close(fig)

if __name__ == "__main__":
    main()
