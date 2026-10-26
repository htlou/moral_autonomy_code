#!/usr/bin/env bash
# Paper setting: 3 opponents x 3 conditions x 5 seeds, 100 rounds.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}/multi-agent:${PYTHONPATH:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/outputs/openspiel}"
STUB_ARGS=()
if [[ "${STUB:-0}" == 1 ]]; then STUB_ARGS+=(--stub); fi
for seed in ${SEEDS:-0 1 2 3 4}; do
  for opponent in always_cooperate always_defect random; do
    for condition in none fixed iterating; do
      "$PYTHON_BIN" -m prototype.main --env matrix_pd \
        --rounds "${ROUNDS:-100}" --reflect-every 5 --history-window 10 \
        --mode vs_fixed --opponent "$opponent" --condition "$condition" \
        --seed "$seed" --out-dir "$OUT_ROOT/${opponent}__${condition}__seed${seed}" \
        "${STUB_ARGS[@]}"
    done
  done
done
