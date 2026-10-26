"""Reparse saved steering completions and rebuild their summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prototype.interpretability.extract_activations import read_jsonl
from prototype.interpretability.steer_factorial import (
    parse_decision,
    summarize,
    write_applied,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    changed = 0
    for row in rows:
        decision = parse_decision(str(row["completion"]))
        if decision != row.get("decision"):
            changed += 1
        row["decision"] = decision
        row["evidence_aligned"] = (
            decision == row["expected_evidence_decision"]
        )
        target = row.get("instruction_target")
        row["instruction_followed"] = (
            decision == target if target is not None else None
        )
        row["write_applied"] = write_applied(
            bool(row["authority_to_write"]), decision
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in rows
    ))

    input_summary_path = args.input.with_suffix(".summary.json")
    if input_summary_path.exists():
        report = json.loads(input_summary_path.read_text())
        layers = [int(layer) for layer in report["layers"]]
        axes = [str(axis) for axis in report["axes"]]
        levels = [
            float(level) for level in report["coefficient_levels"]
        ]
        report["n_generations"] = len(rows)
        report["results_by_layer"] = {
            str(layer): summarize(
                [row for row in rows if int(row["layer"]) == layer],
                axes,
                levels,
            )
            for layer in layers
        }
        report["decision_parser_recomputed"] = True
        report["decision_rows_changed"] = changed
        args.output.with_suffix(".summary.json").write_text(
            json.dumps(report, indent=2)
        )
    print(f"Wrote {args.output}; changed decisions={changed}")


if __name__ == "__main__":
    main()
