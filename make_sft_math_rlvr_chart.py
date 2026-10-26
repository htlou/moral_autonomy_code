#!/usr/bin/env python3
"""Build a markdown chart for SFT Math vs RLVR Main evaluation outputs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


CONDITIONS = ["wr_neg", "wr_pos", "wr_neu", "wor_neg", "wor_pos", "wor_neu"]
BENCHMARKS = ["math_reasoning", "neg_value", "pos_value", "neu_value"]
EXPERIMENTS = [
    ("sft_math", "SFT Math"),
    ("rlvr_main", "RLVR Main"),
]


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def fmt_score(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def fmt_int(value: Any) -> str:
    if value is None:
        return ""
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return str(value)


def aggregate_n(data: dict[str, Any], cond: str, bench: str, key: str) -> Any:
    return (
        data.get("details", {})
        .get(cond, {})
        .get(bench, {})
        .get("aggregate", {})
        .get(key)
    )


def rel(path: Path) -> str:
    try:
        return os.path.relpath(path, Path.cwd())
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    manifest = load_json(run_dir / "manifest.json") or {}
    judge = manifest.get("judge", {})

    lines: list[str] = []
    lines.append("# SFT Math vs RLVR Main Evaluation Chart")
    lines.append("")
    lines.append(f"- Run directory: `{run_dir}`")
    lines.append(f"- Benchmarks: `{', '.join(BENCHMARKS)}`")
    lines.append(
        f"- Judge for `neg_value`/`pos_value`: `{judge.get('model', '')}` via `{judge.get('backend', '')}` on port `{judge.get('port', '')}`"
    )
    lines.append("- RLVR neutral value rows use the `external_neutral_value_experiment/wr_neu` and `external_neutral_value_experiment/wor_neu` checkpoints.")
    lines.append("- SFT Math rows reuse existing 1000-item vLLM generation files and are rescored in this run.")
    lines.append("")

    header = [
        "Experiment",
        "Condition",
        "Model path",
        "Math",
        "Neg value",
        "Pos value",
        "Neu value",
        "Neu bias",
        "n math",
        "n neg",
        "n pos",
        "n neu",
        "Eval file",
    ]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")

    for exp_key, exp_label in EXPERIMENTS:
        for cond in CONDITIONS:
            eval_path = run_dir / f"{exp_key}_{cond}_eval.json"
            data = load_json(eval_path)
            model_path = manifest.get("experiments", {}).get(exp_key, {}).get(cond, "")
            if data is None:
                row = [exp_label, cond, f"`{model_path}`", "", "", "", "", "", "", "", "", "", f"`{rel(eval_path)}`"]
                lines.append("| " + " | ".join(row) + " |")
                continue

            matrix = data.get("matrix", {}).get(cond, {})
            bias = data.get("bias_scores", {}).get(cond)
            row = [
                exp_label,
                cond,
                f"`{model_path}`",
                fmt_score(matrix.get("math_reasoning")),
                fmt_score(matrix.get("neg_value")),
                fmt_score(matrix.get("pos_value")),
                fmt_score(matrix.get("neu_value")),
                fmt_score(bias),
                fmt_int(aggregate_n(data, cond, "math_reasoning", "n_scored")),
                fmt_int(aggregate_n(data, cond, "neg_value", "n_scored")),
                fmt_int(aggregate_n(data, cond, "pos_value", "n_scored")),
                fmt_int(aggregate_n(data, cond, "neu_value", "n_scored")),
                f"`{rel(eval_path)}`",
            ]
            lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append("## Source Checkpoints")
    lines.append("")
    for exp_key, exp_label in EXPERIMENTS:
        lines.append(f"### {exp_label}")
        lines.append("")
        for cond in CONDITIONS:
            model_path = manifest.get("experiments", {}).get(exp_key, {}).get(cond, "")
            lines.append(f"- `{cond}`: `{model_path}`")
        lines.append("")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
