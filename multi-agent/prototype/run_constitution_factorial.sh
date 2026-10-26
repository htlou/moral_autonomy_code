#!/usr/bin/env bash
# Content-matched factorial representation and causal-steering experiment.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname -- "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:-cognitivecomputations/dolphin-2.9.2-qwen2-72b}"
SOURCE_EVENTS="${SOURCE_EVENTS:-prototype/runs/constitution_interpretability_72b_20260726/events.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-prototype/runs/constitution_factorial_gate_72b_20260727}"
LAYERS="${LAYERS:-19,39,59,79}"
STEERING_LAYER="${STEERING_LAYER:-59}"
MAX_PAIRS_PER_TASK_PHASE="${MAX_PAIRS_PER_TASK_PHASE:-15}"
STEERING_MAX_PAIRS="${STEERING_MAX_PAIRS:-12}"
BATCH_SIZE="${BATCH_SIZE:-4}"
STEERING_BATCH_SIZE="${STEERING_BATCH_SIZE:-8}"
STEERING_MAX_NEW_TOKENS="${STEERING_MAX_NEW_TOKENS:-32}"
RUN_STEERING="${RUN_STEERING:-1}"

mkdir -p "$OUTPUT_ROOT"

"$PYTHON_BIN" -m prototype.interpretability.factorial_gate_events \
  --events "$SOURCE_EVENTS" \
  --max-pairs-per-task-phase "$MAX_PAIRS_PER_TASK_PHASE" \
  --output "$OUTPUT_ROOT/events.jsonl"

CUDA_VISIBLE_DEVICES=0,1,2,3 \
"$PYTHON_BIN" -m prototype.interpretability.extract_activations \
  --events "$OUTPUT_ROOT/events.jsonl" \
  --model "$MODEL_PATH" \
  --layers "$LAYERS" \
  --edit-groups all \
  --batch-size "$BATCH_SIZE" \
  --output "$OUTPUT_ROOT/activations.npz"

"$PYTHON_BIN" -m prototype.interpretability.fit_factorial_axes \
  --activations "$OUTPUT_ROOT/activations.npz" \
  --cv-groups seed,task,template \
  --output "$OUTPUT_ROOT/factorial_axes.npz"

if [[ "$RUN_STEERING" == 1 ]]; then
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  "$PYTHON_BIN" -m prototype.interpretability.steer_factorial \
    --events "$OUTPUT_ROOT/events.jsonl" \
    --model "$MODEL_PATH" \
    --axes "$OUTPUT_ROOT/factorial_axes.npz" \
    --layer "$STEERING_LAYER" \
    --axis-source same_seed \
    --axis-names \
      need_to_edit,authority_to_write,instruction_pressure,need_x_authority \
    --coefficient-multipliers=-2,0,2 \
    --batch-size "$STEERING_BATCH_SIZE" \
    --max-new-tokens "$STEERING_MAX_NEW_TOKENS" \
    --max-pairs "$STEERING_MAX_PAIRS" \
    --output "$OUTPUT_ROOT/steering.jsonl"
fi

echo "Factorial constitution-gate pipeline complete: $OUTPUT_ROOT"
