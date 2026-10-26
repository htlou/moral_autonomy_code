#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_all_8b.sh — Full experiment pipeline using dolphin-2.9-llama3-8b.
#
# Runs everything end-to-end:
#   Phase 1: Constitution evolution (6 conditions + "none" baseline)
#   Phase 2: Evaluation of constitution models (7 conditions incl. "none")
#   Phase 3: SFT data preparation
#   Phase 4: SFT finetuning (6 conditions, DeepSpeed)
#   Phase 5: SFT evaluation
#
# Usage:
#   bash run_all_8b.sh                          # full pipeline
#   bash run_all_8b.sh --phase 1                # constitution evolution only
#   bash run_all_8b.sh --phase 2                # constitution eval only
#   bash run_all_8b.sh --phase 3                # SFT data prep only
#   bash run_all_8b.sh --phase 4                # SFT finetuning only
#   bash run_all_8b.sh --phase 5                # SFT eval only
#   bash run_all_8b.sh --phase 1,2              # multiple phases
#   bash run_all_8b.sh --dry-run                # preview dataset selection
#   bash run_all_8b.sh -n 200                   # fewer items (for testing)
#   bash run_all_8b.sh --only wr_neg wor_neg    # specific conditions
#   bash run_all_8b.sh --num-gpus 4              # fewer GPUs
#
# Parallelism strategy (8B on 8 GPUs):
#   vLLM inference:  TP=1, DP=8  (8 model replicas, one per GPU)
#   SFT training:    DeepSpeed ZeRO-2 DDP (data-parallel across all GPUs)
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_PATH="${POSITION_MODEL_DIR:-cognitivecomputations/dolphin-2.9-llama3-8b}"
MODEL_TAG="8b"                          # used in experiment/output naming

ITEMS_PER_CONDITION=10000
EVAL_ITEMS=1000
SAVE_EVERY=500
SEED_TRAIN=42
SEED_EVAL=999

VLLM_PORT=8198
NUM_GPUS=8                              # use all 8 GPUs
GPU_IDS="0,1,2,3,4,5,6,7"
MAX_MODEL_LEN=8192                      # llama3 context
VLLM_DTYPE="float16"
GPU_MEM_UTIL=0.6                        # fraction of GPU memory for vLLM
RAY_BATCH_SIZE=8                        # mini-batch size per Ray actor (concurrent LLM calls)
MAX_CONCURRENT=64                       # max concurrent requests for eval phases

# Constitution / Memory size limits (1x of training dataset size)
MAX_CONSTITUTION=10000
MAX_MEMORY=10000

# SFT defaults — 8B across 8 GPUs with DeepSpeed ZeRO-2 (DDP + optimizer sharding)
SFT_EPOCHS=3
SFT_LR="2e-5"
SFT_BATCH_SIZE=4                        # per-GPU micro batch (8B fits easily per GPU)
SFT_GRAD_ACCUM=2                        # effective batch = 4 * 2 * 8 GPUs = 64
SFT_MAX_SEQ_LENGTH=2048
SFT_ZERO_STAGE=2                        # ZeRO-2: shard optimizer+gradients across 8 GPUs
SFT_VLLM_PORT=8199

DRY_RUN=false
PHASES="1,2,3,4,5"
ONLY_CONDITIONS=()
EXPERIMENT_ID=""

# ═══════════════════════════════════════════════════════════════════════════════
# Parse args
# ═══════════════════════════════════════════════════════════════════════════════

while [[ $# -gt 0 ]]; do
    case "$1" in
        -n|--items)          ITEMS_PER_CONDITION="$2"; shift 2 ;;
        --eval-items)        EVAL_ITEMS="$2"; shift 2 ;;
        --save-every)        SAVE_EVERY="$2"; shift 2 ;;
        --experiment-id)     EXPERIMENT_ID="$2"; shift 2 ;;
        --model)             MODEL_PATH="$2"; shift 2 ;;
        --phase)             PHASES="$2"; shift 2 ;;
        --vllm-port)         VLLM_PORT="$2"; shift 2 ;;
        --num-gpus)          NUM_GPUS="$2"; shift 2 ;;
        --gpu-ids)           GPU_IDS="$2"; shift 2 ;;
        --max-concurrent)    MAX_CONCURRENT="$2"; shift 2 ;;
        --sft-epochs)        SFT_EPOCHS="$2"; shift 2 ;;
        --sft-lr)            SFT_LR="$2"; shift 2 ;;
        --sft-batch-size)    SFT_BATCH_SIZE="$2"; shift 2 ;;
        --sft-zero)          SFT_ZERO_STAGE="$2"; shift 2 ;;
        --max-constitution)  MAX_CONSTITUTION="$2"; shift 2 ;;
        --max-memory)        MAX_MEMORY="$2"; shift 2 ;;
        --gpu-mem-util)      GPU_MEM_UTIL="$2"; shift 2 ;;
        --ray-batch-size|-K) RAY_BATCH_SIZE="$2"; shift 2 ;;
        --dry-run)           DRY_RUN=true; shift ;;
        --only)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                ONLY_CONDITIONS+=("$1"); shift
            done ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Generate experiment ID if not provided
if [[ -z "$EXPERIMENT_ID" ]]; then
    EXPERIMENT_ID="rexp_${MODEL_TAG}_$(date +%Y%m%d_%H%M%S)"
fi

# Derived paths
SFT_DATA_DIR="sft_data_${MODEL_TAG}"
SFT_CKPT_DIR="sft_checkpoints_${MODEL_TAG}"

# Build condition list
ALL_6_CONDITIONS=("wr_neg" "wr_pos" "wr_neu" "wor_neg" "wor_pos" "wor_neu")
if [[ ${#ONLY_CONDITIONS[@]} -gt 0 ]]; then
    TRAIN_CONDITIONS=("${ONLY_CONDITIONS[@]}")
else
    TRAIN_CONDITIONS=("${ALL_6_CONDITIONS[@]}")
fi
# For eval, include "none" baseline
EVAL_CONDITIONS=("${TRAIN_CONDITIONS[@]}" "none")

# Parse phases into an array
IFS=',' read -ra PHASE_ARRAY <<< "$PHASES"

phase_enabled() {
    local target="$1"
    for p in "${PHASE_ARRAY[@]}"; do
        [[ "$p" == "$target" ]] && return 0
    done
    return 1
}

# ═══════════════════════════════════════════════════════════════════════════════
# Environment
# ═══════════════════════════════════════════════════════════════════════════════

# Activate your Python environment before running this script.

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy 2>/dev/null || true
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

# Override config.py via environment — 8B model + larger constitution/memory bounds
export POSITION_MODEL_DIR="${MODEL_PATH}"
export POSITION_VLLM_URL="http://localhost:${VLLM_PORT}/v1"
export POSITION_VLLM_MODEL="${MODEL_PATH}"
export POSITION_MAX_CONSTITUTION="${MAX_CONSTITUTION}"
export POSITION_MAX_MEMORY="${MAX_MEMORY}"

# Judge API (gpt-4o-mini via proxy) for eval scoring
export POSITION_JUDGE_URL="${POSITION_JUDGE_URL:-http://localhost:8197/v1}"
export POSITION_JUDGE_MODEL="${POSITION_JUDGE_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
export POSITION_JUDGE_API_KEY="${POSITION_JUDGE_API_KEY:-dummy}"

# ═══════════════════════════════════════════════════════════════════════════════
# Print plan
# ═══════════════════════════════════════════════════════════════════════════════

echo ""
echo "╔═══════════════════════════════════════════════════════════════════════╗"
echo "║                   FULL EXPERIMENT — 8B MODEL                        ║"
echo "╠═══════════════════════════════════════════════════════════════════════╣"
echo "║  Model:          ${MODEL_PATH}"
echo "║  Experiment ID:  ${EXPERIMENT_ID}"
echo "║  Phases:         ${PHASES}"
echo "║  Items/cond:     ${ITEMS_PER_CONDITION} (train) / ${EVAL_ITEMS} (eval)"
echo "║  Train conds:    ${TRAIN_CONDITIONS[*]}"
echo "║  Eval conds:     ${EVAL_CONDITIONS[*]}"
echo "║  GPUs:           ${NUM_GPUS} (${GPU_IDS})"
echo "║  vLLM:           TP=1, DP=${NUM_GPUS} on port ${VLLM_PORT}"
echo "║  Ray batch (K):  ${RAY_BATCH_SIZE} (peak: ${#TRAIN_CONDITIONS[@]}×${RAY_BATCH_SIZE} concurrent)"
echo "║  Max model len:  ${MAX_MODEL_LEN}"
echo "║  Max constitution: ${MAX_CONSTITUTION}  |  Max memory: ${MAX_MEMORY}"
if $DRY_RUN; then
echo "║  Mode:           DRY RUN"
fi
echo "╠═══════════════════════════════════════════════════════════════════════╣"
echo "║  SFT: epochs=${SFT_EPOCHS} lr=${SFT_LR} micro=${SFT_BATCH_SIZE}x${SFT_GRAD_ACCUM}x${NUM_GPUS}gpu ZeRO-${SFT_ZERO_STAGE}"
echo "║  SFT data:       ${SFT_DATA_DIR}/"
echo "║  SFT checkpoints: ${SFT_CKPT_DIR}/"
echo "╚═══════════════════════════════════════════════════════════════════════╝"
echo ""

# ═══════════════════════════════════════════════════════════════════════════════
# Helper: ensure vLLM is running
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

    # 8B model: TP=1, DP=NUM_GPUS for max throughput (one replica per GPU)
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
    # Also kill anything else on that port

    sleep 1
}

LLM_URL="http://localhost:${VLLM_PORT}/v1"

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1: Constitution Evolution (training)
# ═══════════════════════════════════════════════════════════════════════════════

if phase_enabled 1; then
    echo ""
    echo "###################################################################"
    echo "#  PHASE 1: Constitution Evolution                                #"
    echo "###################################################################"
    echo ""

    if $DRY_RUN; then
        python run_experiment_ray.py \
            --items-per-condition "$ITEMS_PER_CONDITION" \
            --experiment-id "$EXPERIMENT_ID" \
            --only "${TRAIN_CONDITIONS[@]}" \
            --dry-run
    else
        start_vllm "$MODEL_PATH" "$VLLM_PORT"

        python run_experiment_ray.py \
            --items-per-condition "$ITEMS_PER_CONDITION" \
            --experiment-id "$EXPERIMENT_ID" \
            --save-every "$SAVE_EVERY" \
            --llm-url "$LLM_URL" \
            -K "$RAY_BATCH_SIZE" \
            --only "${TRAIN_CONDITIONS[@]}"

        echo ""
        echo "[OK] Phase 1 complete. Snapshots: snapshots/${EXPERIMENT_ID}/"
    fi

    # Create "none" snapshot for this experiment
    NONE_DIR="snapshots/${EXPERIMENT_ID}/none"
    mkdir -p "${NONE_DIR}/rag_storage" "${NONE_DIR}/memory_storage"
    echo '{}' > "${NONE_DIR}/rag_storage/constitution_index.json"
    echo '{}' > "${NONE_DIR}/memory_storage/memory_index.json"
    cat > "${NONE_DIR}/manifest.json" <<MANIFEST
{
  "run_id": "none",
  "created_at": "$(date -Iseconds)",
  "constitution_entries": 0,
  "memory_entries": 0,
  "total_processed": 0,
  "final_constitution_size": 0,
  "final_memory_size": 0,
  "description": "Ablation baseline: empty constitution and memory"
}
MANIFEST
    echo "[OK] Created 'none' baseline snapshot"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2: Constitution Model Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

if phase_enabled 2; then
    echo ""
    echo "###################################################################"
    echo "#  PHASE 2: Constitution Model Evaluation                         #"
    echo "###################################################################"
    echo ""

    if $DRY_RUN; then
        echo "[DRY RUN] Would evaluate: ${EVAL_CONDITIONS[*]}"
        echo "[DRY RUN] Items per benchmark: ${EVAL_ITEMS}"
    else
        start_vllm "$MODEL_PATH" "$VLLM_PORT"

        GEN_FILE="snapshots/${EXPERIMENT_ID}_gen.json"
        EVAL_FILE="snapshots/${EXPERIMENT_ID}_eval.json"

        # Build --only args including "none"
        ONLY_ARGS=()
        for c in "${EVAL_CONDITIONS[@]}"; do
            ONLY_ARGS+=("$c")
        done

        echo "---- Step 2a: Generation ----"
        python eval_gen.py \
            --experiment-id "$EXPERIMENT_ID" \
            -n "$EVAL_ITEMS" \
            --seed "$SEED_EVAL" \
            --max-concurrent "$MAX_CONCURRENT" \
            --llm-url "$LLM_URL" \
            --model-name "$MODEL_PATH" \
            --only "${ONLY_ARGS[@]}" \
            --n-train "$ITEMS_PER_CONDITION" \
            --output "$GEN_FILE"

        echo ""
        echo "---- Step 2b: Scoring ----"
        python eval_score.py \
            --gen-file "$GEN_FILE" \
            --llm-url "$LLM_URL" \
            --model-name "$MODEL_PATH" \
            --max-concurrent "$MAX_CONCURRENT" \
            --output "$EVAL_FILE"

        echo ""
        echo "[OK] Phase 2 complete. Results: ${EVAL_FILE}"
    fi
fi

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3: SFT Data Preparation
# ═══════════════════════════════════════════════════════════════════════════════

if phase_enabled 3; then
    echo ""
    echo "###################################################################"
    echo "#  PHASE 3: SFT Data Preparation                                 #"
    echo "###################################################################"
    echo ""

    ONLY_SFT=""
    for c in "${TRAIN_CONDITIONS[@]}"; do
        ONLY_SFT="${ONLY_SFT} ${c}"
    done

    python sft_data_prep.py \
        --items-per-condition "$ITEMS_PER_CONDITION" \
        --output-dir "$SFT_DATA_DIR" \
        --seed "$SEED_TRAIN" \
        --only ${ONLY_SFT}

    echo ""
    echo "[OK] Phase 3 complete. Data: ${SFT_DATA_DIR}/"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 4: SFT Finetuning
# ═══════════════════════════════════════════════════════════════════════════════

if phase_enabled 4; then
    echo ""
    echo "###################################################################"
    echo "#  PHASE 4: SFT Finetuning (DeepSpeed ZeRO-${SFT_ZERO_STAGE})              #"
    echo "###################################################################"
    echo ""

    if $DRY_RUN; then
        echo "[DRY RUN] Would finetune: ${TRAIN_CONDITIONS[*]}"
        echo "[DRY RUN] Epochs: ${SFT_EPOCHS}, LR: ${SFT_LR}"
    else
        # Stop vLLM to free GPU memory for training
        stop_vllm "$VLLM_PORT"

        DS_CONFIG="ds_config_zero${SFT_ZERO_STAGE}.json"

        for cond in "${TRAIN_CONDITIONS[@]}"; do
            echo ""
            echo "======== SFT: ${cond} ========"

            deepspeed --num_gpus "$NUM_GPUS" sft_train.py \
                --condition "$cond" \
                --data-dir "$SFT_DATA_DIR" \
                --output-dir "$SFT_CKPT_DIR" \
                --model-path "$MODEL_PATH" \
                --epochs "$SFT_EPOCHS" \
                --lr "$SFT_LR" \
                --batch-size "$SFT_BATCH_SIZE" \
                --grad-accum "$SFT_GRAD_ACCUM" \
                --max-seq-length "$SFT_MAX_SEQ_LENGTH" \
                --deepspeed "$DS_CONFIG" \
                --bf16

            echo "[OK] ${cond} → ${SFT_CKPT_DIR}/${cond}/"
        done

        echo ""
        echo "[OK] Phase 4 complete. Checkpoints: ${SFT_CKPT_DIR}/"
    fi
fi

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 5: SFT Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

if phase_enabled 5; then
    echo ""
    echo "###################################################################"
    echo "#  PHASE 5: SFT Model Evaluation                                 #"
    echo "###################################################################"
    echo ""

    if $DRY_RUN; then
        echo "[DRY RUN] Would evaluate SFT models: ${TRAIN_CONDITIONS[*]}"
    else
        # Need base model for BeaverTails judge scoring
        start_vllm "$MODEL_PATH" "$VLLM_PORT"

        for cond in "${TRAIN_CONDITIONS[@]}"; do
            CKPT_PATH="${SFT_CKPT_DIR}/${cond}"
            SFT_GEN_FILE="snapshots/sft_${MODEL_TAG}_${cond}_gen.json"
            SFT_EVAL_FILE="snapshots/sft_${MODEL_TAG}_${cond}_eval.json"

            echo ""
            echo "======== SFT EVAL: ${cond} ========"

            if [[ ! -d "$CKPT_PATH" ]]; then
                echo "[ERROR] Checkpoint not found: ${CKPT_PATH} — skipping"
                continue
            fi

            # Start SFT model on a separate port
            stop_vllm "$SFT_VLLM_PORT" 2>/dev/null || true
            start_vllm "$CKPT_PATH" "$SFT_VLLM_PORT" 4096

            echo "---- Generation ----"
            python eval_gen_sft.py \
                --condition "$cond" \
                -n "$EVAL_ITEMS" \
                --seed "$SEED_EVAL" \
                --max-concurrent "$MAX_CONCURRENT" \
                --llm-url "http://localhost:${SFT_VLLM_PORT}/v1" \
                --model-name "$CKPT_PATH" \
                --n-train "$ITEMS_PER_CONDITION" \
                --output "$SFT_GEN_FILE"

            echo "---- Scoring ----"
            python eval_score.py \
                --gen-file "$SFT_GEN_FILE" \
                --llm-url "$LLM_URL" \
                --model-name "$MODEL_PATH" \
                --max-concurrent "$MAX_CONCURRENT" \
                --output "$SFT_EVAL_FILE"

            stop_vllm "$SFT_VLLM_PORT"

            echo "[OK] ${cond} → ${SFT_EVAL_FILE}"
        done

        echo ""
        echo "[OK] Phase 5 complete."
    fi
fi

# ═══════════════════════════════════════════════════════════════════════════════
# Cleanup & Summary
# ═══════════════════════════════════════════════════════════════════════════════

# Stop any remaining vLLM
stop_vllm "$VLLM_PORT" 2>/dev/null || true
stop_vllm "$SFT_VLLM_PORT" 2>/dev/null || true

echo ""
echo "╔═══════════════════════════════════════════════════════════════════════╗"
echo "║                     EXPERIMENT COMPLETE                             ║"
echo "╠═══════════════════════════════════════════════════════════════════════╣"
echo "║  Experiment:  ${EXPERIMENT_ID}"
echo "║  Model:       $(basename $MODEL_PATH)"
echo "║  Phases run:  ${PHASES}"
echo "╠═══════════════════════════════════════════════════════════════════════╣"
echo "║  Constitution snapshots:  snapshots/${EXPERIMENT_ID}/"
echo "║  Constitution eval:       snapshots/${EXPERIMENT_ID}_eval.json"
echo "║  SFT data:                ${SFT_DATA_DIR}/"
echo "║  SFT checkpoints:         ${SFT_CKPT_DIR}/"
echo "║  SFT eval:                snapshots/sft_${MODEL_TAG}_*_eval.json"
echo "╚═══════════════════════════════════════════════════════════════════════╝"
