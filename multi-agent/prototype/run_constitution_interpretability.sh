#!/usr/bin/env bash
# End-to-end activation/subspace pipeline. Run only after the vLLM job exits.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname -- "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/.venv-concordia/bin/python}"
MODEL_PATH="${MODEL_PATH:-cognitivecomputations/dolphin-2.9.2-qwen2-72b}"
OUTPUT_ROOT="${OUTPUT_ROOT:-prototype/runs/constitution_interpretability}"
LAYERS="${LAYERS:-19,39,59,79}"
MAX_PER_CELL="${MAX_PER_CELL:-300}"
BATCH_SIZE="${BATCH_SIZE:-2}"
SAE_LAYER="${SAE_LAYER:-59}"
RUN_SAE="${RUN_SAE:-0}"
RUN_STEERING="${RUN_STEERING:-0}"
ALLOW_ACTIVE_VLLM="${ALLOW_ACTIVE_VLLM:-0}"

if [[ "$ALLOW_ACTIVE_VLLM" != 1 ]] \
  && curl -sf http://127.0.0.1:8198/metrics >/dev/null 2>&1; then
  echo "Refusing to load a second 72B model while vLLM :8198 is active." >&2
  echo "Wait for the experiment queue, or set ALLOW_ACTIVE_VLLM=1." >&2
  exit 1
fi

if (( $# == 0 )); then
  set -- \
    prototype/runs/concordia_pub_causal_72b_100seeds_20260726 \
    prototype/runs/concordia_three_venue_closures_causal_72b_100seeds_20260726 \
    prototype/runs/concordia_tough_friendship_causal_72b_100seeds_20260726 \
    prototype/runs/concordia_phased_haggling_causal_72b_100seeds_20260726 \
    prototype/runs/concordia_misinformed_majority_causal_72b_100seeds_20260726 \
    prototype/runs/concordia_multi_item_value_shift_causal_72b_100seeds_20260726
fi

mkdir -p "$OUTPUT_ROOT"
"$PYTHON_BIN" -m prototype.interpretability.constitution_events \
  "$@" --output "$OUTPUT_ROOT/events.jsonl"

"$PYTHON_BIN" -m prototype.interpretability.extract_activations \
  --events "$OUTPUT_ROOT/events.jsonl" \
  --model "$MODEL_PATH" \
  --layers "$LAYERS" \
  --edit-groups none,large \
  --max-per-cell "$MAX_PER_CELL" \
  --batch-size "$BATCH_SIZE" \
  --output "$OUTPUT_ROOT/activations.npz"

"$PYTHON_BIN" -m prototype.interpretability.fit_subspace \
  --activations "$OUTPUT_ROOT/activations.npz" \
  --method pca \
  --n-components 8 \
  --output "$OUTPUT_ROOT/pca.npz"

"$PYTHON_BIN" -m prototype.interpretability.fit_subspace \
  --activations "$OUTPUT_ROOT/activations.npz" \
  --method autonomy_interaction \
  --nuisance-fields task,phase,block_mean_reward,input_tokens \
  --cv-group seed \
  --output "$OUTPUT_ROOT/autonomy_interaction.npz"

if [[ "$RUN_SAE" == 1 ]]; then
  "$PYTHON_BIN" -m prototype.interpretability.train_sae \
    --activations "$OUTPUT_ROOT/activations.npz" \
    --layer "$SAE_LAYER" \
    --output "$OUTPUT_ROOT/layer${SAE_LAYER}_sae.pt"
fi

if [[ "$RUN_STEERING" == 1 ]]; then
  "$PYTHON_BIN" -m prototype.interpretability.steer_subspace \
    --events "$OUTPUT_ROOT/events.jsonl" \
    --model "$MODEL_PATH" \
    --direction "$OUTPUT_ROOT/autonomy_interaction.npz" \
    --layer "$SAE_LAYER" \
    --coefficients=-20,-10,0,10,20 \
    --limit 200 \
    --output "$OUTPUT_ROOT/steering.jsonl"
fi

echo "Interpretability pipeline complete: $OUTPUT_ROOT"
