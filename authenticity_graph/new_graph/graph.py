from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, LogFormatterMathtext

# ---- Wire up data import ---------------------------------------------------
_HERE = Path(__file__).resolve().parent
_PARENT = _HERE.parent
sys.path.insert(0, str(_PARENT))
sys.path.insert(0, str(_HERE))

from plot_hsd_split import all_models, flagship_models  # noqa: E402

OUT = str(_HERE)


# =============================================================================
# Visual language
# =============================================================================

INK = "#1f1f1f"
MUTED = "#6b6b6b"
GRID = "#cfcfcf"
TREND = "#2a2a2a"
POINT = "#2c4a6e"


# =============================================================================
# Data prep
# =============================================================================

@dataclass
class Row:
    name: str
    date: datetime
    hsd: float
    params: float
    org: str
    level: int


def prepare_rows(models) -> list[Row]:
    rows = []
    for name, date_str, hsd, old_level, params, org in models:
        rows.append(Row(
            name=name,
            date=datetime.strptime(date_str, "%Y-%m"),
            hsd=float(hsd),
            params=float(params),
            org=org,
            level=int(old_level),
        ))
    return rows


def point_size(params: float, lo: float, hi: float, base=42, gain=160) -> float:
    """Marker size scales with log10(parameter count)."""
    logp = np.log10(params)
    t = (logp - lo) / max(hi - lo, 1e-9)
    return base + gain * t


# =============================================================================
# Trend
# =============================================================================

def fit_loglinear(rows: list[Row]):
    ref = datetime(2022, 1, 1)
    days = np.array([(r.date - ref).days for r in rows], dtype=float)
    log_hsd = np.log10([r.hsd for r in rows])

    slope, intercept = np.polyfit(days, log_hsd, 1)
    return ref, slope, intercept, days, log_hsd


def trend_line_xy(rows: list[Row], n=300, pad_days=(45, 90)):
    ref, slope, intercept, days, _ = fit_loglinear(rows)

    x = np.linspace(days.min() - pad_days[0], days.max() + pad_days[1], n)
    dates = [ref + timedelta(days=float(v)) for v in x]
    y = 10 ** (slope * x + intercept)

    return dates, y, slope


# =============================================================================
# Figure
# =============================================================================

def make_figure(
    models,
    *,
    filename_base: str,
    figsize=(13.5, 8.0),
):
    rows = prepare_rows(models)
    rows.sort(key=lambda r: r.date)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 13,
        "axes.labelsize": 15,
        "axes.titlesize": 17,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 12.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
        "axes.edgecolor": "#888",
        "axes.linewidth": 0.8,
    })

    fig = plt.figure(figsize=figsize, facecolor="white")

    # Leave a little top margin for the figure-level slope box.
    ax = fig.add_axes([0.085, 0.115, 0.875, 0.82])
    ax.set_facecolor("#fafafa")

    # ---- Axes config -------------------------------------------------------
    dates = [r.date for r in rows]
    hsds = [r.hsd for r in rows]

    ax.set_yscale("log")
    ax.set_xlim(min(dates) - timedelta(days=120), max(dates) + timedelta(days=140))

    log_lo, log_hi = np.log10(min(hsds)), np.log10(max(hsds))
    ax.set_ylim(10 ** (log_lo - 0.55), 10 ** (log_hi + 0.85))

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))

    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=8))
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax.yaxis.set_minor_locator(
        LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=20)
    )

    ax.grid(
        True, which="major",
        color=GRID, lw=0.6, alpha=0.55,
        linestyle="-", zorder=0
    )
    ax.grid(
        True, which="minor",
        color=GRID, lw=0.4, alpha=0.25,
        linestyle="-", zorder=0
    )
    ax.set_axisbelow(True)

    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#888")

    ax.tick_params(colors=MUTED, length=4, width=0.8)
    ax.set_xlabel("Release year", color=INK, labelpad=8)
    ax.set_ylabel(
        "Human specification density  (bits / parameter)",
        color=INK,
        labelpad=10,
    )

    # ---- Trend line --------------------------------------------------------
    t_dates, t_y, slope_per_day = trend_line_xy(rows)
    slope_per_year = slope_per_day * 365.25

    ax.plot(
        t_dates,
        t_y,
        color=TREND,
        linewidth=2.0,
        linestyle=(0, (7, 4)),
        alpha=0.7,
        zorder=2,
        solid_capstyle="round",
    )

    # ---- Slope annotation: whole-figure top-left ---------------------------
    slope_text = f"log-linear trend\n{slope_per_year:+.2f} decades / year"

    fig.text(
        0.035,
        0.965,
        slope_text,
        ha="left",
        va="top",
        fontsize=14,
        color=INK,
        fontweight="medium",
        linespacing=1.25,
        bbox=dict(
            boxstyle="round,pad=0.50",
            fc="white",
            ec=TREND,
            lw=1.0,
            alpha=0.97,
        ),
        zorder=20,
    )

    # ---- Draw points: no model names --------------------------------------
    logps = [np.log10(r.params) for r in rows]
    lo_lp, hi_lp = min(logps), max(logps)

    for r in rows:
        s = point_size(r.params, lo_lp, hi_lp)

        ax.scatter(
            r.date,
            r.hsd,
            s=s,
            facecolor=POINT,
            edgecolor="white",
            linewidth=1.1,
            alpha=0.38,
            zorder=3,
        )

    # ---- Legend: parameter size + trend -----------------------------------
    size_ref_params = [
        (10e9, "10B"),
        (100e9, "100B"),
        (1e12, "1T"),
    ]

    size_handles = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="",
            markerfacecolor=POINT,
            markeredgecolor="white",
            markeredgewidth=1.0,
            markersize=np.sqrt(point_size(p, lo_lp, hi_lp)),
            label=label,
            alpha=0.75,
        )
        for p, label in size_ref_params
    ]

    size_handles.append(
        Line2D(
            [0], [0],
            color=TREND,
            linewidth=2.2,
            linestyle=(0, (7, 4)),
            label="Log-linear trend",
        )
    )

    size_leg = ax.legend(
        handles=size_handles,
        loc="lower left",
        frameon=True,
        facecolor="white",
        edgecolor="#cfcfcf",
        framealpha=0.95,
        borderpad=1.0,
        labelspacing=1.3,
        handlelength=2.0,
        handletextpad=1.1,
        title="Parameters",
        title_fontsize=13.5,
        fontsize=12.5,
    )

    size_leg.get_title().set_color(INK)
    size_leg.get_title().set_fontweight("semibold")

    # ---- Save --------------------------------------------------------------
    pdf = os.path.join(OUT, f"{filename_base}.pdf")
    png = os.path.join(OUT, f"{filename_base}.png")

    fig.savefig(pdf)
    fig.savefig(png, dpi=300, facecolor="white")
    plt.close(fig)

    print(f"Saved: {pdf}")
    print(f"Saved: {png}")


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    make_figure(
        flagship_models,
        filename_base="hsd_flagship_clean",
        figsize=(13.5, 8.0),
    )

    make_figure(
        all_models,
        filename_base="hsd_all_clean",
        figsize=(15.5, 9.5),
    )
