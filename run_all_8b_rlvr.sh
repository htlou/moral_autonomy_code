#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_all_8b_rlvr.sh — RLVR (GRPO) ablation pipeline using dolphin-2.9-llama3-8b.
#
# Counterpart to run_all_8b.sh (constitution) and run_all_8b_math.sh (SFT):
#   Phase 1: Data preparation (prompt-only, with reward metadata)
#   Phase 2: GRPO training (6 conditions, multi-GPU)
#   Phase 3: Evaluation (same benchmarks as other pipelines)
#
# Reward functions:
#   - Reasoning (MATH): verifiable logic matching (\\boxed{} extraction)
#   - Value (neg/pos/neu): Claude Sonnet 4.6 judge scoring
#
# Usage:
#   bash run_all_8b_rlvr.sh                          # full pipeline
#   bash run_all_8b_rlvr.sh --phase 1                # data prep only
#   bash run_all_8b_rlvr.sh --phase 2                # training only
#   bash run_all_8b_rlvr.sh --phase 3                # evaluation only
#   bash run_all_8b_rlvr.sh --only wr_neg wor_neg    # specific conditions
#   bash run_all_8b_rlvr.sh --dry-run                # preview
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_PATH="${POSITION_MODEL_DIR:-cognitivecomputations/dolphin-2.9-llama3-8b}"
MODEL_TAG="8b_rlvr"

ITEMS_PER_CONDITION=5000               # fewer than SFT — GRPO is more expensive
EVAL_ITEMS=1000
SEED_TRAIN=42
SEED_EVAL=999

NUM_GPUS=8
GPU_IDS="0,1,2,3,4,5,6,7"
VLLM_PORT=8198
MAX_MODEL_LEN=8192
VLLM_DTYPE="float16"
GPU_MEM_UTIL=0.6
MAX_CONCURRENT=64

# GRPO training defaults
RLVR_EPOCHS=1
RLVR_LR="5e-6"
RLVR_BATCH_SIZE=2                      # per-GPU micro batch
RLVR_GRAD_ACCUM=2                      # effective batch = 2 * 2 * 8 GPUs = 32
RLVR_MAX_COMPLETION=512
RLVR_NUM_GENERATIONS=4                 # completions per prompt for GRPO
RLVR_BETA=0.1                          # KL penalty coefficient (higher = more conservative)
RLVR_JUDGE_CONCURRENCY=32              # concurrent Claude API calls
RLVR_ZERO_STAGE=2

# Eval vLLM port for finetuned models
EVAL_VLLM_PORT=8199

DRY_RUN=false
PHASES="1,2,3"
ONLY_CONDITIONS=()
EXPERIMENT_ID=""
REASONING_DATASET="math"               # "math" or "mmlu"

# ═══════════════════════════════════════════════════════════════════════════════
# Parse args
# ═══════════════════════════════════════════════════════════════════════════════

while [[ $# -gt 0 ]]; do
    case "$1" in
        -n|--items)              ITEMS_PER_CONDITION="$2"; shift 2 ;;
        --eval-items)            EVAL_ITEMS="$2"; shift 2 ;;
        --experiment-id)         EXPERIMENT_ID="$2"; shift 2 ;;
        --model)                 MODEL_PATH="$2"; shift 2 ;;
        --phase)                 PHASES="$2"; shift 2 ;;
        --vllm-port)             VLLM_PORT="$2"; shift 2 ;;
        --num-gpus)              NUM_GPUS="$2"; shift 2 ;;
        --gpu-ids)               GPU_IDS="$2"; shift 2 ;;
        --gpu-mem-util)          GPU_MEM_UTIL="$2"; shift 2 ;;
        --max-concurrent)        MAX_CONCURRENT="$2"; shift 2 ;;
        --epochs)                RLVR_EPOCHS="$2"; shift 2 ;;
        --lr)                    RLVR_LR="$2"; shift 2 ;;
        --batch-size)            RLVR_BATCH_SIZE="$2"; shift 2 ;;
        --grad-accum)            RLVR_GRAD_ACCUM="$2"; shift 2 ;;
        --num-generations)       RLVR_NUM_GENERATIONS="$2"; shift 2 ;;
        --beta)                  RLVR_BETA="$2"; shift 2 ;;
        --judge-concurrency)     RLVR_JUDGE_CONCURRENCY="$2"; shift 2 ;;
        --reasoning-dataset)     REASONING_DATASET="$2"; shift 2 ;;
        --dry-run)               DRY_RUN=true; shift ;;
        --only)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                ONLY_CONDITIONS+=("$1"); shift
            done ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$EXPERIMENT_ID" ]]; then
    EXPERIMENT_ID="rlvr_${MODEL_TAG}_$(date +%Y%m%d_%H%M%S)"
fi

RLVR_DATA_DIR="rlvr_data_${MODEL_TAG}"
RLVR_CKPT_DIR="rlvr_checkpoints_${MODEL_TAG}"

ALL_6_CONDITIONS=("wr_neg" "wr_pos" "wr_neu" "wor_neg" "wor_pos" "wor_neu")
if [[ ${#ONLY_CONDITIONS[@]} -gt 0 ]]; then
    TRAIN_CONDITIONS=("${ONLY_CONDITIONS[@]}")
else
    TRAIN_CONDITIONS=("${ALL_6_CONDITIONS[@]}")
fi

IFS=',' read -ra PHASE_ARRAY <<< "$PHASES"

phase_enabled() {
    local target="$1"
    for p in "${PHASE_ARRAY[@]}"; do
        [[ "$p" == "$target" ]] && return 0
    done
    return 1
}

# Determine eval benchmarks based on reasoning dataset
if [[ "$REASONING_DATASET" == "math" ]]; then
    EVAL_BENCHMARKS="math_reasoning neg_value pos_value neu_value"
else
    EVAL_BENCHMARKS="reasoning neg_value pos_value neu_value"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# Environment
# ═══════════════════════════════════════════════════════════════════════════════

# Activate your Python environment before running this script.

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy 2>/dev/null || true
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

export POSITION_MODEL_DIR="${MODEL_PATH}"

# Memory + NCCL config for GRPO with external API calls
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_TIMEOUT=3600
export NCCL_BLOCKING_WAIT=1

# Wandb
export WANDB_PROJECT="position-rlvr-8b"

# Local judge model (Llama-3.1-8B-Instruct via vLLM on dedicated GPU)
JUDGE_MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"
JUDGE_VLLM_PORT=8197
JUDGE_GPU=7                             # dedicated GPU for judge
TRAIN_GPUS=7                            # remaining GPUs for DeepSpeed (0-6)
TRAIN_GPU_IDS="0,1,2,3,4,5,6"

# ═══════════════════════════════════════════════════════════════════════════════
# Print plan
# ═══════════════════════════════════════════════════════════════════════════════

echo ""
echo "╔═══════════════════════════════════════════════════════════════════════╗"
echo "║            RLVR (GRPO) EXPERIMENT — 8B MODEL                       ║"
echo "╠═══════════════════════════════════════════════════════════════════════╣"
echo "║  Model:          ${MODEL_PATH}"
echo "║  Experiment ID:  ${EXPERIMENT_ID}"
echo "║  Reasoning:      ${REASONING_DATASET}"
echo "║  Phases:         ${PHASES}"
echo "║  Items/cond:     ${ITEMS_PER_CONDITION} (train) / ${EVAL_ITEMS} (eval)"
echo "║  Conditions:     ${TRAIN_CONDITIONS[*]}"
echo "║  GPUs:           ${NUM_GPUS} (${GPU_IDS})"
echo "╠═══════════════════════════════════════════════════════════════════════╣"
echo "║  GRPO: epochs=${RLVR_EPOCHS} lr=${RLVR_LR} beta=${RLVR_BETA}"
echo "║  Batch: micro=${RLVR_BATCH_SIZE}x${RLVR_GRAD_ACCUM}x${NUM_GPUS}gpu"
echo "║  Generations/prompt: ${RLVR_NUM_GENERATIONS}  max_completion: ${RLVR_MAX_COMPLETION}"
echo "║  Rewards: math=logic_match  value=claude-sonnet-4-6"
echo "║  Judge concurrency: ${RLVR_JUDGE_CONCURRENCY}"
echo "║  Data dir:       ${RLVR_DATA_DIR}/"
echo "║  Checkpoints:    ${RLVR_CKPT_DIR}/"
echo "╚═══════════════════════════════════════════════════════════════════════╝"
echo ""

# ═══════════════════════════════════════════════════════════════════════════════
# Helper: vLLM management
# ═══════════════════════════════════════════════════════════════════════════════

VLLM_PID=""

start_vllm() {
    local model="$1"
    local port="$2"
    local max_len="${3:-$MAX_MODEL_LEN}"

    if curl -s "http://localhost:${port}/v1/models" > /dev/null 2>&1; then
        echo "[OK] vLLM already running on port ${port}"
        return
    fi

    local tp=1
    local dp=$NUM_GPUS

    echo "[..] Starting vLLM: model=$(basename $model), port=${port}, TP=${tp}, DP=${dp}..."
    python -m vllm.entrypoints.openai.api_server \
        --model "$model" \
        --port "$port" \
        --max-model-len "$max_len" \
        --tensor-parallel-size "$tp" \
        --data-parallel-size "$dp" \
        --gpu-memory-utilization "$GPU_MEM_UTIL" \
        --dtype "$VLLM_DTYPE" &
    VLLM_PID=$!

    echo "[..] Waiting for vLLM (PID ${VLLM_PID})..."
    local waited=0
    until curl -s "http://localhost:${port}/v1/models" > /dev/null 2>&1; do
        sleep 2
        waited=$((waited + 2))
        if [[ $waited -gt 300 ]]; then
            echo "[ERROR] vLLM failed to start after 300s"
            kill "$VLLM_PID" 2>/dev/null || true
            exit 1
        fi
    done
    echo "[OK] vLLM ready on port ${port}"
}

stop_vllm() {
    local port="$1"
    if [[ -n "$VLLM_PID" ]]; then
        echo "[..] Stopping vLLM (PID ${VLLM_PID})..."
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
        VLLM_PID=""
    fi

    sleep 1
}

LLM_URL="http://localhost:${VLLM_PORT}/v1"

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1: Data Preparation
# ═══════════════════════════════════════════════════════════════════════════════

if phase_enabled 1; then
    echo ""
    echo "###################################################################"
    echo "#  PHASE 1: RLVR Data Preparation                                #"
    echo "###################################################################"
    echo ""

    ONLY_ARGS=""
    for c in "${TRAIN_CONDITIONS[@]}"; do
        ONLY_ARGS="${ONLY_ARGS} ${c}"
    done

    python rlvr_data_prep.py \
        --items-per-condition "$ITEMS_PER_CONDITION" \
        --output-dir "$RLVR_DATA_DIR" \
        --seed "$SEED_TRAIN" \
        --reasoning-dataset "$REASONING_DATASET" \
        --only ${ONLY_ARGS}

    echo ""
    echo "[OK] Phase 1 complete. Data: ${RLVR_DATA_DIR}/"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2: GRPO Training
# ═══════════════════════════════════════════════════════════════════════════════

if phase_enabled 2; then
    echo ""
    echo "###################################################################"
    echo "#  PHASE 2: GRPO Training                                        #"
    echo "###################################################################"
    echo ""

    if $DRY_RUN; then
        echo "[DRY RUN] Would train: ${TRAIN_CONDITIONS[*]}"
        echo "[DRY RUN] Epochs: ${RLVR_EPOCHS}, LR: ${RLVR_LR}, Beta: ${RLVR_BETA}"
    else
        # Start local judge model on dedicated GPU
        JUDGE_PID=""
        if ! curl -s "http://localhost:${JUDGE_VLLM_PORT}/v1/models" > /dev/null 2>&1; then
            echo "[..] Starting local judge: $(basename $JUDGE_MODEL_PATH) on GPU ${JUDGE_GPU}, port ${JUDGE_VLLM_PORT}..."
            CUDA_VISIBLE_DEVICES="${JUDGE_GPU}" python -m vllm.entrypoints.openai.api_server \
                --model "$JUDGE_MODEL_PATH" \
                --port "$JUDGE_VLLM_PORT" \
                --max-model-len 4096 \
                --gpu-memory-utilization 0.85 \
                --dtype float16 &
            JUDGE_PID=$!
            echo "[..] Waiting for judge vLLM (PID ${JUDGE_PID})..."
            waited=0
            until curl -s "http://localhost:${JUDGE_VLLM_PORT}/v1/models" > /dev/null 2>&1; do
                sleep 2
                waited=$((waited + 2))
                if [[ $waited -gt 180 ]]; then
                    echo "[ERROR] Judge vLLM failed to start after 180s"
                    kill "$JUDGE_PID" 2>/dev/null || true
                    exit 1
                fi
            done
            echo "[OK] Local judge ready on port ${JUDGE_VLLM_PORT}"
        else
            echo "[OK] Judge vLLM already running on port ${JUDGE_VLLM_PORT}"
        fi

        # Point reward function at local judge
        export POSITION_JUDGE_URL="http://localhost:${JUDGE_VLLM_PORT}/v1"
        export POSITION_JUDGE_MODEL="$JUDGE_MODEL_PATH"
        export POSITION_JUDGE_API_KEY="dummy"

        # Train on remaining GPUs (exclude judge GPU)
        export CUDA_VISIBLE_DEVICES="${TRAIN_GPU_IDS}"

        for cond in "${TRAIN_CONDITIONS[@]}"; do
            echo ""
            echo "======== RLVR: ${cond} ========"

            deepspeed --num_gpus "$TRAIN_GPUS" rlvr_train.py \
                --condition "$cond" \
                --data-dir "$RLVR_DATA_DIR" \
                --output-dir "$RLVR_CKPT_DIR" \
                --model-path "$MODEL_PATH" \
                --epochs "$RLVR_EPOCHS" \
                --lr "$RLVR_LR" \
                --batch-size "$RLVR_BATCH_SIZE" \
                --grad-accum "$RLVR_GRAD_ACCUM" \
                --max-completion-length "$RLVR_MAX_COMPLETION" \
                --num-generations "$RLVR_NUM_GENERATIONS" \
                --beta "$RLVR_BETA" \
                --judge-concurrency "$RLVR_JUDGE_CONCURRENCY" \
                --deepspeed ds_config_zero2_rlvr.json \
                --bf16

            echo "[OK] ${cond} → ${RLVR_CKPT_DIR}/${cond}/"
        done

        # Stop judge
        if [[ -n "$JUDGE_PID" ]]; then
            echo "[..] Stopping judge vLLM..."
            kill "$JUDGE_PID" 2>/dev/null || true
            wait "$JUDGE_PID" 2>/dev/null || true
        fi


        # Restore CUDA_VISIBLE_DEVICES for eval
        export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

        echo ""
        echo "[OK] Phase 2 complete. Checkpoints: ${RLVR_CKPT_DIR}/"
    fi
fi

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3: Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

if phase_enabled 3; then
    echo ""
    echo "###################################################################"
    echo "#  PHASE 3: RLVR Model Evaluation                                #"
    echo "###################################################################"
    echo ""

    if $DRY_RUN; then
        echo "[DRY RUN] Would evaluate RLVR models: ${TRAIN_CONDITIONS[*]}"
    else
        # Start base model for judge scoring
        start_vllm "$MODEL_PATH" "$VLLM_PORT"

        for cond in "${TRAIN_CONDITIONS[@]}"; do
            CKPT_PATH="${RLVR_CKPT_DIR}/${cond}"
            RLVR_GEN_FILE="snapshots/rlvr_${MODEL_TAG}_${cond}_gen.json"
            RLVR_EVAL_FILE="snapshots/rlvr_${MODEL_TAG}_${cond}_eval.json"

            echo ""
            echo "======== RLVR EVAL: ${cond} ========"

            if [[ ! -d "$CKPT_PATH" ]]; then
                echo "[ERROR] Checkpoint not found: ${CKPT_PATH} — skipping"
                continue
            fi

            # Start RLVR model on separate port
            stop_vllm "$EVAL_VLLM_PORT" 2>/dev/null || true
            start_vllm "$CKPT_PATH" "$EVAL_VLLM_PORT" 4096

            echo "---- Generation ----"
            python eval_gen_sft.py \
                --condition "$cond" \
                -n "$EVAL_ITEMS" \
                --seed "$SEED_EVAL" \
                --max-concurrent "$MAX_CONCURRENT" \
                --llm-url "http://localhost:${EVAL_VLLM_PORT}/v1" \
                --model-name "$CKPT_PATH" \
                --benchmarks ${EVAL_BENCHMARKS} \
                --n-train "$ITEMS_PER_CONDITION" \
                --output "$RLVR_GEN_FILE"

            echo "---- Scoring ----"
            python eval_score.py \
                --gen-file "$RLVR_GEN_FILE" \
                --llm-url "$LLM_URL" \
                --model-name "$MODEL_PATH" \
                --max-concurrent "$MAX_CONCURRENT" \
                --output "$RLVR_EVAL_FILE"

            stop_vllm "$EVAL_VLLM_PORT"

            echo "[OK] ${cond} → ${RLVR_EVAL_FILE}"
        done

        echo ""
        echo "[OK] Phase 3 complete."
    fi
fi

# ═══════════════════════════════════════════════════════════════════════════════
# Cleanup & Summary
# ═══════════════════════════════════════════════════════════════════════════════

stop_vllm "$VLLM_PORT" 2>/dev/null || true
stop_vllm "$EVAL_VLLM_PORT" 2>/dev/null || true

echo ""
echo "╔═══════════════════════════════════════════════════════════════════════╗"
echo "║                     RLVR EXPERIMENT COMPLETE                        ║"
echo "╠═══════════════════════════════════════════════════════════════════════╣"
echo "║  Experiment:  ${EXPERIMENT_ID}"
echo "║  Model:       $(basename $MODEL_PATH)"
echo "║  Reasoning:   ${REASONING_DATASET}"
echo "║  Method:      GRPO (RLVR)"
echo "║  Rewards:     math=logic_match  value=claude-sonnet-4-6"
echo "║  Phases run:  ${PHASES}"
echo "╠═══════════════════════════════════════════════════════════════════════╣"
echo "║  Data:         ${RLVR_DATA_DIR}/"
echo "║  Checkpoints:  ${RLVR_CKPT_DIR}/"
echo "║  Eval:         snapshots/rlvr_${MODEL_TAG}_*_eval.json"
echo "╚═══════════════════════════════════════════════════════════════════════╝"
