import os
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, LogFormatterMathtext

# Reuse the model lists from the existing script in the parent directory.
_HERE = Path(__file__).resolve().parent
_PARENT = _HERE.parent
sys.path.insert(0, str(_PARENT))
from plot_hsd_split import all_models, flagship_models  # noqa: E402

OUT = str(_HERE)

try:
    from adjustText import adjust_text
except Exception:
    adjust_text = None


# =========================
# Style configuration
# =========================

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.labelsize": 8.5,
    "axes.titlesize": 10,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 13,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.unicode_minus": False,
})

LEVEL_COLORS = {
    1: "#c94c43",  # muted red
    2: "#c88a2d",  # muted amber
    3: "#2f8f6b",  # teal
    4: "#2f7fa8",  # blue
    5: "#6f62c9",  # purple
}

LEVEL_NAMES = {
    1: "L1: Scaling",
    2: "L2: Temporal",
    3: "L3: Reflection",
    4: "L4: Co-learning",
    5: "L5: Real-world",
}

ORG_MARKERS = {
    "OpenAI": "o",
    "Anthropic": "s",
    "Meta": "^",
    "Google": "P",
    "DeepSeek": "D",
    "Alibaba": "p",
    "Microsoft": "X",
    "NVIDIA": "h",
    "HuggingFace": "v",
    "AI2": "*",
    "Baichuan": "8",
    "Shanghai AI": "d",
    "Cohere": ">",
    "CMU": "<",
    "IBM": "H",
    "Berkeley": "v",
    "OpenBMB": "o",
    "Tsinghua": "o",
    "01.AI": "p",
    "Zhipu AI": "d",
    "ByteDance": "o",
    "MiniMax": "o",
    "Moonshot": "o",
    "Mistral": "o",
    "StepFun": "o",
}

REFLECTION_METHODS = (
    "Dromedary",
    "SALMON",
    "Self-Rewarding",
    "Meta-Rewarding",
    "DeepSeek-R1-Zero",
    "GLM-Z1",
)

USE_CORRECTED_LADDER = True


def ladder_level(name, old_level):
    if USE_CORRECTED_LADDER:
        if any(key in name for key in REFLECTION_METHODS):
            return 3
        return 1
    return int(np.clip(np.ceil(float(old_level)), 1, 5))


SHORT_LABELS = {
    "InstructGPT (175B)": "InstructGPT\n175B",
    "Constitutional AI (52B)": "Constitutional AI\n52B",
    "Llama 2 (70B)": "Llama 2\n70B",
    "Baichuan 2 (13B)": "Baichuan 2\n13B",
    "Llama 3.1 (405B)": "Llama 3.1\n405B",
    "Llama 4 Maverick (400B)": "Llama 4\n400B",
    "Qwen3 (235B)": "Qwen3\n235B",
    "Qwen3-Coder (80B)": "Qwen3-Coder\n80B",
    "DeepSeek-V3 (671B)": "DS-V3\n671B",
    "DeepSeek-R1 (671B)": "DS-R1\n671B",
    "DeepSeek-R1-Zero (671B)": "DS-R1-Zero\n671B",
    "DeepSeek-V3.1 (671B)": "DS-V3.1\n671B",
    "Gemma 3 (27B)": "Gemma 3\n27B",
    "Gemma 4 (31B)": "Gemma 4\n31B",
    "Gemma 4 MoE (25B)": "Gemma 4 MoE\n25B",
    "Nemotron-Ultra (253B)": "Nemotron-Ultra\n253B",
    "Mistral Large 3 (675B)": "Mistral L3\n675B",
    "MiniMax-M1 (456B)": "MiniMax-M1\n456B",
    "MiniMax-M2.5 (230B)": "MiniMax-M2.5\n230B",
    "Kimi K2.5 (1T)": "Kimi K2.5\n1T",
    "GLM-5 (744B)": "GLM-5\n744B",
    "gpt-oss-120b (117B)": "gpt-oss\n117B",
    "Seed-Coder (8B)": "Seed-Coder\n8B",
    "Command A (111B)": "Command A\n111B",
    "Granite 3.3 (8B)": "Granite 3.3\n8B",
}


def short_label(name):
    if name in SHORT_LABELS:
        return SHORT_LABELS[name]
    m = re.match(r"(.+?)\s*\((.*?)\)$", name)
    if m:
        return f"{m.group(1)}\n{m.group(2)}"
    return name


def prepare_rows(models):
    rows = []
    for name, date_str, hsd, old_level, params, org in models:
        rows.append({
            "name": name,
            "label": short_label(name),
            "date": datetime.strptime(date_str, "%Y-%m"),
            "hsd": float(hsd),
            "params": float(params),
            "org": org,
            "level": ladder_level(name, old_level),
        })
    return rows


def point_size(params, logp_min, logp_max):
    logp = np.log10(params)
    return 38 + 58 * (logp - logp_min) / max(logp_max - logp_min, 1e-9)


def style_axis(ax, rows, xlim=None, ylim=None, right_pad_days=340):
    dates = [r["date"] for r in rows]
    hsds = [r["hsd"] for r in rows]

    ax.set_yscale("log")
    ax.set_axisbelow(True)

    if xlim is None:
        ax.set_xlim(
            min(dates) - timedelta(days=110),
            max(dates) + timedelta(days=right_pad_days),
        )
    else:
        ax.set_xlim(
            datetime.strptime(xlim[0], "%Y-%m"),
            datetime.strptime(xlim[1], "%Y-%m"),
        )

    if ylim is None:
        log_lo = np.log10(min(hsds))
        log_hi = np.log10(max(hsds))
        ax.set_ylim(10 ** (log_lo - 0.7), 10 ** (log_hi + 0.7))
    else:
        ax.set_ylim(*ylim)

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))

    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=7))
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=20))

    ax.grid(True, which="major", linewidth=0.45, alpha=0.22, linestyle="--")
    ax.grid(True, which="minor", linewidth=0.25, alpha=0.08, linestyle=":")

    ax.set_xlabel("Release year")
    ax.set_ylabel("Human specification density (bits / parameter)")

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ax.spines["left"].set_color("0.35")
    ax.spines["bottom"].set_color("0.35")
    ax.tick_params(axis="both", colors="0.25", width=0.6)


def add_trend(
    ax,
    rows,
    label_frac=0.90,
    x_label_pad_days=45,
    y_label_factor=1.45,
    line_color="0.22",
    line_width=2.4,
    label_fontsize=15,
):
    ref = datetime(2022, 1, 1)

    days = np.array([(r["date"] - ref).days for r in rows], dtype=float)
    log_hsd = np.log10([r["hsd"] for r in rows])

    z = np.polyfit(days, log_hsd, 1)
    p = np.poly1d(z)

    slope_per_year = z[0] * 365.25

    x = np.linspace(days.min() - 30, days.max() + 220, 300)
    trend_dates = [ref + timedelta(days=float(v)) for v in x]
    trend_y = 10 ** p(x)

    ax.plot(
        trend_dates,
        trend_y,
        color=line_color,
        linestyle=(0, (7, 3)),
        alpha=0.85,
        linewidth=line_width,
        zorder=2,
    )

    idx = int(np.clip(label_frac, 0.05, 0.96) * (len(x) - 1))
    anchor_date = trend_dates[idx]
    anchor_y = trend_y[idx]

    text_date = anchor_date + timedelta(days=x_label_pad_days)
    text_y = anchor_y * y_label_factor

    ylo, yhi = ax.get_ylim()
    if text_y > yhi / 1.15:
        text_y = anchor_y / y_label_factor
        va = "top"
    else:
        va = "bottom"

    ax.annotate(
        f"slope = {slope_per_year:.2f} decades/year",
        xy=(anchor_date, anchor_y),
        xytext=(text_date, text_y),
        textcoords="data",
        ha="left",
        va=va,
        fontsize=label_fontsize,
        fontweight="semibold",
        color="0.16",
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="white",
            edgecolor="0.55",
            linewidth=0.9,
            alpha=0.96,
        ),
        arrowprops=dict(
            arrowstyle="-",
            color="0.25",
            lw=1.2,
            alpha=0.75,
            shrinkA=4,
            shrinkB=4,
        ),
        zorder=12,
    )

    return slope_per_year


def add_legends(
    ax,
    rows,
    encode_org=False,
    show_org_legend=False,
    show_all_levels=False,
    legend_fontsize=14,
    legend_title_fontsize=14,
    legend_marker_size=11.5,
):
    present_levels = sorted({r["level"] for r in rows})
    levels = [1, 2, 3, 4, 5] if show_all_levels else present_levels

    level_handles = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="",
            markerfacecolor=LEVEL_COLORS[k],
            markeredgecolor="0.2",
            markersize=legend_marker_size,
            label=LEVEL_NAMES[k],
        )
        for k in levels
    ]

    level_handles.append(
        Line2D(
            [0], [0],
            linestyle=(0, (7, 3)),
            color="0.22",
            alpha=0.85,
            linewidth=2.4,
            label="Trend",
        )
    )

    leg1 = ax.legend(
        handles=level_handles,
        loc="lower left",
        frameon=True,
        framealpha=0.96,
        facecolor="white",
        edgecolor="0.75",
        title="Color = autonomy ladder",
        fontsize=legend_fontsize,
        title_fontsize=legend_title_fontsize,
        ncol=1,
        labelspacing=0.62,
        handlelength=2.3,
        handletextpad=0.75,
        borderpad=0.80,
    )
    ax.add_artist(leg1)

    if encode_org and show_org_legend:
        orgs = sorted({r["org"] for r in rows})
        org_handles = [
            Line2D(
                [0], [0],
                marker=ORG_MARKERS.get(org, "o"),
                linestyle="",
                markerfacecolor="0.35",
                markeredgecolor="0.15",
                markersize=legend_marker_size,
                label=org,
            )
            for org in orgs
        ]

        ax.legend(
            handles=org_handles,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            frameon=True,
            framealpha=0.96,
            facecolor="white",
            edgecolor="0.75",
            title="Shape = organization",
            fontsize=max(legend_fontsize - 1, 10),
            title_fontsize=legend_title_fontsize,
            ncol=1,
            borderpad=0.80,
            labelspacing=0.58,
            handletextpad=0.70,
        )


def preferred_offset_points(row, x_mid_num, y_mid_log):
    """
    Give each label a sensible initial offset in points before adjustText runs.
    This helps reduce early pile-ups.
    """
    x_num = mdates.date2num(row["date"])
    y_log = np.log10(row["hsd"])

    dx = 12 if x_num <= x_mid_num else -12
    dy = 10 if y_log <= y_mid_log else -10

    # Small deterministic variation by name to avoid too many identical offsets.
    name_hash = sum(ord(c) for c in row["name"]) % 4
    if name_hash == 1:
        dy += 8
    elif name_hash == 2:
        dx += 8 if dx > 0 else -8
    elif name_hash == 3:
        dy -= 8

    return dx, dy


def add_connectors_for_labels(
    ax,
    texts,
    labeled_rows,
    connector_color="0.45",
    connector_lw=0.65,
    connector_alpha=0.50,
    min_connector_px=28,
):
    """
    Draw subtle connector lines only if the label moved far enough away from the point.
    This avoids the ugly forest of unnecessary lines.
    """
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    for text, row in zip(texts, labeled_rows):
        text_bbox = text.get_window_extent(renderer=renderer)

        point_disp = ax.transData.transform(
            (mdates.date2num(row["date"]), row["hsd"])
        )
        text_center_disp = np.array([
            (text_bbox.x0 + text_bbox.x1) / 2.0,
            (text_bbox.y0 + text_bbox.y1) / 2.0,
        ])

        dist_px = np.linalg.norm(text_center_disp - point_disp)
        if dist_px < min_connector_px:
            continue

        text_x, text_y = text.get_position()

        ax.annotate(
            "",
            xy=(row["date"], row["hsd"]),
            xytext=(text_x, text_y),
            textcoords="data",
            arrowprops=dict(
                arrowstyle="-",
                color=connector_color,
                lw=connector_lw,
                alpha=connector_alpha,
                shrinkA=8,
                shrinkB=6,
                capstyle="round",
                joinstyle="round",
            ),
            zorder=7,
        )


def draw_points_and_labels(
    ax,
    rows,
    label_names=None,
    label_all=False,
    encode_org=False,
    label_fontsize=6.0,
    point_scale=1.0,
    background_alpha=0.42,
    connector_lines=True,
):
    label_names = set(label_names or [])

    logps = [np.log10(r["params"]) for r in rows]
    logp_min, logp_max = min(logps), max(logps)

    dates_num = [mdates.date2num(r["date"]) for r in rows]
    hsds_log = [np.log10(r["hsd"]) for r in rows]
    x_mid_num = np.median(dates_num)
    y_mid_log = np.median(hsds_log)

    # Draw points first.
    for r in rows:
        is_labeled = label_all or r["name"] in label_names
        marker = ORG_MARKERS.get(r["org"], "o") if encode_org else "o"
        color = LEVEL_COLORS.get(r["level"], "0.65")
        base_size = point_size(r["params"], logp_min, logp_max) * point_scale

        ax.scatter(
            r["date"],
            r["hsd"],
            s=base_size * (1.15 if is_labeled else 1.0),
            c=color,
            marker=marker,
            edgecolors="0.16",
            linewidths=0.70 if is_labeled else 0.45,
            alpha=0.95 if is_labeled else background_alpha,
            zorder=4 if is_labeled else 3,
        )

    labeled_rows = [r for r in rows if label_all or r["name"] in label_names]
    texts = []

    # Create offset labels instead of placing boxes directly on points.
    for r in labeled_rows:
        dx, dy = preferred_offset_points(r, x_mid_num, y_mid_log)

        ann = ax.annotate(
            r["label"],
            xy=(r["date"], r["hsd"]),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=label_fontsize,
            linespacing=0.92,
            bbox=dict(
                boxstyle="round,pad=0.20",
                facecolor="white",
                edgecolor="0.82",
                linewidth=0.35,
                alpha=0.95,
            ),
            zorder=8,
        )
        texts.append(ann)

    # Then use adjustText to further untangle them.
    if adjust_text is not None and texts:
        adjust_text(
            texts,
            ax=ax,
            expand=(1.35, 1.60),
            force_text=(0.95, 1.15),
            force_static=(0.35, 0.55),
            force_pull=(0.08, 0.12),
            only_move={"text": "xy", "static": "xy", "explode": "xy", "pull": "xy"},
            avoid_self=True,
            ensure_inside_axes=True,
            arrowprops=None,   # important: do not let adjustText draw ugly arrows
            time_lim=20,
        )

    if connector_lines and texts:
        add_connectors_for_labels(
            ax,
            texts,
            labeled_rows,
            connector_color="0.45",
            connector_lw=0.60,
            connector_alpha=0.48,
            min_connector_px=30,
        )

    return texts


def plot_hsd(
    models,
    title,
    subtitle,
    filename_base,
    label_names=None,
    label_all=False,
    encode_org=False,
    show_org_legend=False,
    show_all_levels=False,
    figure_size=(7.2, 4.35),
    label_fontsize=5.8,
    point_scale=1.0,
    xlim=None,
    ylim=None,
    trend_label_frac=0.90,
    trend_label_fontsize=15,
    legend_fontsize=14,
    legend_title_fontsize=14,
    legend_marker_size=11.5,
    title_fontsize=17,
    subtitle_fontsize=11,
    connector_lines=True,
):
    rows = prepare_rows(models)

    fig, ax = plt.subplots(figsize=figure_size, constrained_layout=False)

    right_margin = 0.82 if (encode_org and show_org_legend) else 0.985
    fig.subplots_adjust(
        left=0.070,
        right=right_margin,
        bottom=0.080,
        top=0.900,
    )

    # Figure-level title/subtitle to avoid upper-left overlap.
    fig.text(
        0.070,
        0.970,
        title,
        ha="left",
        va="top",
        fontsize=title_fontsize,
        fontweight="semibold",
        color="0.05",
    )

    if subtitle:
        fig.text(
            0.070,
            0.943,
            subtitle,
            ha="left",
            va="top",
            fontsize=subtitle_fontsize,
            color="0.35",
        )

    style_axis(
        ax,
        rows,
        xlim=xlim,
        ylim=ylim,
        right_pad_days=340 if not show_org_legend else 260,
    )

    add_trend(
        ax,
        rows,
        label_frac=trend_label_frac,
        label_fontsize=trend_label_fontsize,
    )

    draw_points_and_labels(
        ax,
        rows,
        label_names=label_names,
        label_all=label_all,
        encode_org=encode_org,
        label_fontsize=label_fontsize,
        point_scale=point_scale,
        background_alpha=0.40 if not label_all else 0.72,
        connector_lines=connector_lines,
    )

    add_legends(
        ax,
        rows,
        encode_org=encode_org,
        show_org_legend=show_org_legend,
        show_all_levels=show_all_levels,
        legend_fontsize=legend_fontsize,
        legend_title_fontsize=legend_title_fontsize,
        legend_marker_size=legend_marker_size,
    )

    pdf_path = os.path.join(OUT, f"{filename_base}.pdf")
    png_path = os.path.join(OUT, f"{filename_base}.png")

    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=600)
    plt.close(fig)

    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    # Main-paper version: all models
    plot_hsd(
        all_models,
        title="Human specification density over time: all verified models",
        subtitle=f"N = {len(all_models)} models, 2022–2026; Wave 7 corrected HSD",
        filename_base="graph8a_hsd_all_clean",
        label_all=True,
        encode_org=False,
        show_org_legend=False,
        show_all_levels=False,
        figure_size=(22.0, 13.0),
        label_fontsize=9.1,
        point_scale=1.25,
        trend_label_frac=0.91,
        trend_label_fontsize=16,
        legend_fontsize=15,
        legend_title_fontsize=15,
        legend_marker_size=12.5,
        title_fontsize=18,
        subtitle_fontsize=11.5,
        connector_lines=True,
    )

    # Main-paper version: flagship only
    plot_hsd(
        flagship_models,
        title="Human specification density over time: flagship industry models",
        subtitle=f"N = {len(flagship_models)} models; largest per series, industry only",
        filename_base="graph8b_hsd_flagship_clean",
        label_all=True,
        encode_org=False,
        show_org_legend=False,
        show_all_levels=False,
        figure_size=(18.0, 11.0),
        label_fontsize=10.3,
        point_scale=1.30,
        trend_label_frac=0.92,
        trend_label_fontsize=16,
        legend_fontsize=15,
        legend_title_fontsize=15,
        legend_marker_size=12.5,
        title_fontsize=17,
        subtitle_fontsize=11.0,
        connector_lines=True,
    )

    # Appendix version: all labels + organization shapes
    plot_hsd(
        all_models,
        title="Human specification density over time: all verified models",
        subtitle=f"N = {len(all_models)} models, 2022–2026; all labels shown",
        filename_base="graph8a_hsd_all_appendix",
        label_all=True,
        encode_org=True,
        show_org_legend=True,
        show_all_levels=True,
        figure_size=(18.0, 10.5),
        label_fontsize=6.2,
        point_scale=1.10,
        trend_label_frac=0.91,
        trend_label_fontsize=12,
        legend_fontsize=11,
        legend_title_fontsize=11,
        legend_marker_size=9.5,
        title_fontsize=15.5,
        subtitle_fontsize=10.2,
        connector_lines=True,
    )

    # Appendix version: flagship + organization shapes
    plot_hsd(
        flagship_models,
        title="Human specification density over time: flagship industry models",
        subtitle=f"N = {len(flagship_models)} models; all labels shown",
        filename_base="graph8b_hsd_flagship_appendix",
        label_all=True,
        encode_org=True,
        show_org_legend=True,
        show_all_levels=True,
        figure_size=(15.0, 8.5),
        label_fontsize=7.1,
        point_scale=1.15,
        trend_label_frac=0.92,
        trend_label_fontsize=12,
        legend_fontsize=11,
        legend_title_fontsize=11,
        legend_marker_size=9.5,
        title_fontsize=15,
        subtitle_fontsize=10,
        connector_lines=True,
    )
