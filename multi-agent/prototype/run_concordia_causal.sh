#!/usr/bin/env bash
# Paired multi-seed Concordia Pub Coordination experiment.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname -- "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/.venv-concordia/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-prototype/runs/concordia_pub_causal}"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
PHASE_ORDERS="${PHASE_ORDERS:-altruistic,free_riders,reciprocal}"
ROUNDS_PER_PHASE="${ROUNDS_PER_PHASE:-30}"
REFLECT_EVERY="${REFLECT_EVERY:-5}"
HISTORY_WINDOW="${HISTORY_WINDOW:-5}"
BACKGROUND_AGENTS="${BACKGROUND_AGENTS:-6}"
MAX_CONSTITUTION_SIZE="${MAX_CONSTITUTION_SIZE:-20}"
ACTION_TEMPERATURE="${ACTION_TEMPERATURE:-0}"
REFLECTION_TEMPERATURE="${REFLECTION_TEMPERATURE:-0}"
INJECTION_DIRECTION="${INJECTION_DIRECTION:-prosocial}"
INJECTION_REWARD_THRESHOLD="${INJECTION_REWARD_THRESHOLD:-1.25}"
INJECTION_CONFIDENCE="${INJECTION_CONFIDENCE:-1}"
MAX_PARALLEL="${MAX_PARALLEL:-1}"

export CUDA_VISIBLE_DEVICES=""
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1}"

if (( MAX_PARALLEL < 1 )); then
  echo "MAX_PARALLEL must be at least 1" >&2
  exit 2
fi

IFS=';' read -r -a order_array <<< "$PHASE_ORDERS"
read -r -a seed_array <<< "$SEEDS"

pids=()
labels=()
status=0

wait_oldest() {
  local pid="${pids[0]}"
  local label="${labels[0]}"
  if ! wait "$pid"; then
    echo "FAILED: $label" >&2
    status=1
  fi
  pids=("${pids[@]:1}")
  labels=("${labels[@]:1}")
}

for phase_order in "${order_array[@]}"; do
  order_label="${phase_order//,/_}"
  order_dir="$OUTPUT_ROOT/$order_label"
  for seed in "${seed_array[@]}"; do
    label="$order_label seed=$seed"
    echo "START: $label"
    "$PYTHON_BIN" -m prototype.concordia_pub_coordination_experiment \
      --seed "$seed" \
      --phase-order "$phase_order" \
      --rounds-per-phase "$ROUNDS_PER_PHASE" \
      --reflect-every "$REFLECT_EVERY" \
      --history-window "$HISTORY_WINDOW" \
      --background-agents "$BACKGROUND_AGENTS" \
      --max-constitution-size "$MAX_CONSTITUTION_SIZE" \
      --action-temperature "$ACTION_TEMPERATURE" \
      --reflection-temperature "$REFLECTION_TEMPERATURE" \
      --injection-direction "$INJECTION_DIRECTION" \
      --injection-reward-threshold "$INJECTION_REWARD_THRESHOLD" \
      --injection-confidence "$INJECTION_CONFIDENCE" \
      --out-dir "$order_dir" \
      "$@" &
    pids+=("$!")
    labels+=("$label")

    if (( ${#pids[@]} >= MAX_PARALLEL )); then
      wait_oldest
    fi
  done
done

while (( ${#pids[@]} )); do
  wait_oldest
done

if (( status != 0 )); then
  exit "$status"
fi

"$PYTHON_BIN" -m prototype.aggregate_concordia "$OUTPUT_ROOT"
