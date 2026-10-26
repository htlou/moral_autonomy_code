#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_eval_sft.sh — Evaluate SFT ablation models.
#
# For each condition, this script:
#   1. Starts vLLM with the finetuned checkpoint
#   2. Runs eval generation (no constitution/memory, just the SFT model)
#   3. Scores with the same eval_score.py used for constitution models
#   4. Stops vLLM
#
# Usage:
#   bash run_eval_sft.sh --checkpoint-dir sft_checkpoints --condition wr_neg
#   bash run_eval_sft.sh --checkpoint-dir sft_checkpoints --all
#   bash run_eval_sft.sh --checkpoint-dir sft_checkpoints --all -n 100
#
# If vLLM is already running with the right model, use --no-vllm:
#   bash run_eval_sft.sh --condition wr_neg --no-vllm --llm-url http://localhost:8199/v1
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── Defaults ────────────────────────────────────────────────────────────────

CHECKPOINT_DIR="sft_checkpoints"
CONDITIONS=()
ALL_CONDITIONS=false
N_PER_BENCHMARK=100
SEED=999
MAX_CONCURRENT=16
VLLM_PORT=8199
BASE_VLLM_PORT=8198  # for scoring (BeaverTails judge uses base model)
NUM_GPUS=8
GPU_IDS="0,1,2,3,4,5,6,7"
NO_VLLM=false
LLM_URL=""
MODEL_NAME=""

# ── Parse args ──────────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint-dir)  CHECKPOINT_DIR="$2"; shift 2 ;;
        --condition)       CONDITIONS+=("$2"); shift 2 ;;
        --all)             ALL_CONDITIONS=true; shift ;;
        -n)                N_PER_BENCHMARK="$2"; shift 2 ;;
        --seed)            SEED="$2"; shift 2 ;;
        --max-concurrent)  MAX_CONCURRENT="$2"; shift 2 ;;
        --vllm-port)       VLLM_PORT="$2"; shift 2 ;;
        --base-vllm-port)  BASE_VLLM_PORT="$2"; shift 2 ;;
        --num-gpus)        NUM_GPUS="$2"; shift 2 ;;
        --gpu-ids)         GPU_IDS="$2"; shift 2 ;;
        --no-vllm)         NO_VLLM=true; shift ;;
        --llm-url)         LLM_URL="$2"; shift 2 ;;
        --model-name)      MODEL_NAME="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if $ALL_CONDITIONS; then
    CONDITIONS=("wr_neg" "wr_pos" "wr_neu" "wor_neg" "wor_pos" "wor_neu")
fi

if [[ ${#CONDITIONS[@]} -eq 0 ]]; then
    echo "Usage: bash run_eval_sft.sh --checkpoint-dir <dir> --condition <cond> | --all"
    exit 1
fi

# ── Environment ─────────────────────────────────────────────────────────────

# Activate your Python environment before running this script.

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy 2>/dev/null || true

# ── Print plan ──────────────────────────────────────────────────────────────

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "  SFT Evaluation"
echo "  Checkpoint dir: ${CHECKPOINT_DIR}"
echo "  Conditions:     ${CONDITIONS[*]}"
echo "  Items/cell:     ${N_PER_BENCHMARK}"
echo "  Seed:           ${SEED}"
echo "  vLLM port:      ${VLLM_PORT} (SFT model)"
echo "  Base vLLM port: ${BASE_VLLM_PORT} (for BeaverTails judge)"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""

# Check that base vLLM is running (needed for BeaverTails scoring judge)
if ! curl -s "http://localhost:${BASE_VLLM_PORT}/v1/models" > /dev/null 2>&1; then
    echo "[WARN] Base vLLM not running on port ${BASE_VLLM_PORT}."
    echo "       BeaverTails scoring will fail without a judge model."
    echo "       Start it: bash run_eval.sh --vllm ..."
fi

# ── Evaluate each condition ────────────────────────────────────────────────

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

for cond in "${CONDITIONS[@]}"; do
    CKPT_PATH="${CHECKPOINT_DIR}/${cond}"
    GEN_FILE="snapshots/sft_${cond}_gen.json"
    EVAL_FILE="snapshots/sft_${cond}_eval.json"

    echo "======== SFT EVAL: ${cond} ========"

    if [[ ! -d "$CKPT_PATH" ]] && ! $NO_VLLM; then
        echo "[ERROR] Checkpoint not found: ${CKPT_PATH}"
        echo "        Run: bash run_sft.sh --condition ${cond}"
        continue
    fi

    # ── Start vLLM for this checkpoint ──
    VLLM_PID=""
    if ! $NO_VLLM; then
        echo "[..] Starting vLLM for ${cond} on port ${VLLM_PORT}..."

        # Kill any existing vLLM on this port

        sleep 1

        python -m vllm.entrypoints.openai.api_server \
            --model "$CKPT_PATH" \
            --port "$VLLM_PORT" \
            --max-model-len 4096 \
            --tensor-parallel-size "$NUM_GPUS" \
            --gpu-memory-utilization 0.9 \
            --dtype float16 &
        VLLM_PID=$!

        echo "[..] Waiting for vLLM (PID ${VLLM_PID})..."
        until curl -s "http://localhost:${VLLM_PORT}/v1/models" > /dev/null 2>&1; do
            sleep 2
        done
        echo "[OK] vLLM ready for ${cond}"

        COND_LLM_URL="http://localhost:${VLLM_PORT}/v1"
        COND_MODEL_NAME="${CKPT_PATH}"
    else
        COND_LLM_URL="${LLM_URL:-http://localhost:${VLLM_PORT}/v1}"
        COND_MODEL_NAME="${MODEL_NAME:-sft_model}"
    fi

    # ── Step 1: Generate ──
    echo ""
    echo "---- Generation ----"
    python eval_gen_sft.py \
        --condition "$cond" \
        -n "$N_PER_BENCHMARK" \
        --seed "$SEED" \
        --max-concurrent "$MAX_CONCURRENT" \
        --llm-url "$COND_LLM_URL" \
        --model-name "$COND_MODEL_NAME" \
        --output "$GEN_FILE"

    # ── Step 2: Score ──
    echo ""
    echo "---- Scoring ----"
    python eval_score.py \
        --gen-file "$GEN_FILE" \
        --llm-url "http://localhost:${BASE_VLLM_PORT}/v1" \
        --model-name "cognitivecomputations/dolphin-2.9.2-qwen2-72b" \
        --max-concurrent "$MAX_CONCURRENT" \
        --output "$EVAL_FILE"

    # ── Stop vLLM ──
    if [[ -n "$VLLM_PID" ]]; then
        echo "[..] Stopping vLLM (PID ${VLLM_PID})..."
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
        echo "[OK] vLLM stopped"
    fi

    echo ""
    echo "[OK] ${cond} eval complete → ${EVAL_FILE}"
    echo ""
done

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "  SFT Evaluation Complete"
echo "  Results: snapshots/sft_<condition>_eval.json"
echo "═══════════════════════════════════════════════════════════════════════"
