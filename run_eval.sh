#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_eval.sh — Two-step evaluation: generate responses, then score them.
#
# Usage:
#   bash run_eval.sh --experiment-id merged_best -n 100
#   bash run_eval.sh --experiment-id merged_best -n 10 --only wr_neg wr_pos
#   bash run_eval.sh --experiment-id merged_best -n 100 --vllm
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── Defaults ────────────────────────────────────────────────────────────────

EXPERIMENT_ID=""
N_PER_BENCHMARK=100
SEED=999
MAX_CONCURRENT=16
VLLM_PORT=8198
MODEL_PATH="cognitivecomputations/dolphin-2.9.2-qwen2-72b"
START_VLLM=false
ONLY_CONDITIONS=()
BENCHMARKS=()
GEN_OUTPUT=""
SCORE_ONLY=""

# ── Parse args ──────────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case "$1" in
        --experiment-id)     EXPERIMENT_ID="$2"; shift 2 ;;
        -n|--n-per-benchmark) N_PER_BENCHMARK="$2"; shift 2 ;;
        --seed)              SEED="$2"; shift 2 ;;
        --max-concurrent)    MAX_CONCURRENT="$2"; shift 2 ;;
        --vllm)              START_VLLM=true; shift ;;
        --vllm-port)         VLLM_PORT="$2"; shift 2 ;;
        --model)             MODEL_PATH="$2"; shift 2 ;;
        --score-only)        SCORE_ONLY="$2"; shift 2 ;;
        --only)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                ONLY_CONDITIONS+=("$1"); shift
            done ;;
        --benchmarks)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                BENCHMARKS+=("$1"); shift
            done ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$EXPERIMENT_ID" && -z "$SCORE_ONLY" ]]; then
    echo "Usage: bash run_eval.sh --experiment-id <ID> -n <N>"
    echo "       bash run_eval.sh --score-only <gen_file.json>"
    exit 1
fi

# ── Environment ─────────────────────────────────────────────────────────────

# Activate your Python environment before running this script.

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy 2>/dev/null || true

# ── vLLM ────────────────────────────────────────────────────────────────────

if $START_VLLM; then
    if curl -s "http://localhost:${VLLM_PORT}/v1/models" > /dev/null 2>&1; then
        echo "[OK] vLLM already running on port ${VLLM_PORT}"
    else
        echo "[..] Starting vLLM on port ${VLLM_PORT}..."
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m vllm.entrypoints.openai.api_server \
            --model "$MODEL_PATH" \
            --port "$VLLM_PORT" \
            --max-model-len 32768 \
            --tensor-parallel-size 8 \
            --gpu-memory-utilization 0.9 \
            --dtype float16 &
        VLLM_PID=$!
        echo "[..] Waiting for vLLM (PID ${VLLM_PID})..."
        until curl -s "http://localhost:${VLLM_PORT}/v1/models" > /dev/null 2>&1; do
            sleep 2
        done
        echo "[OK] vLLM ready"
    fi
else
    if ! curl -s "http://localhost:${VLLM_PORT}/v1/models" > /dev/null 2>&1; then
        echo "[ERROR] vLLM not running on port ${VLLM_PORT}."
        echo "        Start it with:  bash run_eval.sh --vllm ..."
        exit 1
    fi
    echo "[OK] vLLM running on port ${VLLM_PORT}"
fi

LLM_URL="http://localhost:${VLLM_PORT}/v1"

# ── Score-only mode ─────────────────────────────────────────────────────────

if [[ -n "$SCORE_ONLY" ]]; then
    echo ""
    echo "═══════════════════════════════════════════════════════════════════"
    echo "  Score-only mode"
    echo "  Gen file: ${SCORE_ONLY}"
    echo "═══════════════════════════════════════════════════════════════════"
    echo ""
    python eval_score.py \
        --gen-file "$SCORE_ONLY" \
        --llm-url "$LLM_URL" \
        --model-name "$MODEL_PATH" \
        --max-concurrent "$MAX_CONCURRENT"
    exit 0
fi

# ── Build gen command ───────────────────────────────────────────────────────

GEN_FILE="snapshots/${EXPERIMENT_ID}_gen.json"

GEN_CMD=(
    python eval_gen.py
    --experiment-id "$EXPERIMENT_ID"
    -n "$N_PER_BENCHMARK"
    --seed "$SEED"
    --max-concurrent "$MAX_CONCURRENT"
    --llm-url "$LLM_URL"
    --model-name "$MODEL_PATH"
    --output "$GEN_FILE"
)
[[ ${#ONLY_CONDITIONS[@]} -gt 0 ]] && GEN_CMD+=(--only "${ONLY_CONDITIONS[@]}")
[[ ${#BENCHMARKS[@]} -gt 0 ]] && GEN_CMD+=(--benchmarks "${BENCHMARKS[@]}")

# ── Print plan ──────────────────────────────────────────────────────────────

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "  Experiment:     ${EXPERIMENT_ID}"
echo "  Items/cell:     ${N_PER_BENCHMARK}"
echo "  Seed:           ${SEED}"
echo "  Conditions:     ${ONLY_CONDITIONS[*]:-wr_neg wr_pos wr_neu wor_neg wor_pos wor_neu none}"
echo "  Benchmarks:     ${BENCHMARKS[*]:-reasoning neg_value pos_value neu_value}"
echo "  Concurrency:    ${MAX_CONCURRENT}"
echo "  LLM URL:        ${LLM_URL}"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""

# ── Step 1: Generate ────────────────────────────────────────────────────────

echo "======== STEP 1: Generation ========"
"${GEN_CMD[@]}"

echo ""

# ── Step 2: Score ───────────────────────────────────────────────────────────

echo "======== STEP 2: Scoring ========"
python eval_score.py \
    --gen-file "$GEN_FILE" \
    --llm-url "$LLM_URL" \
    --model-name "$MODEL_PATH" \
    --max-concurrent "$MAX_CONCURRENT"

echo ""
echo "Done."
