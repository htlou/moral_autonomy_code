#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_sft.sh — SFT ablation: finetune per condition using DeepSpeed.
#
# Steps:
#   1. Prepare SFT data (if not already done)
#   2. Finetune one model per condition using DeepSpeed ZeRO-3
#
# Usage:
#   bash run_sft.sh --condition wr_neg              # single condition
#   bash run_sft.sh --all                           # all 6 conditions
#   bash run_sft.sh --all --prep-only               # only prepare data
#   bash run_sft.sh --condition wr_neg --epochs 5   # custom epochs
#   bash run_sft.sh --condition wr_neg --zero 2     # use ZeRO-2 instead
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── Defaults ────────────────────────────────────────────────────────────────

CONDITIONS=()
ALL_CONDITIONS=false
ITEMS_PER_CONDITION=1000
DATA_DIR="sft_data"
OUTPUT_DIR="sft_checkpoints"
MODEL_PATH="cognitivecomputations/dolphin-2.9.2-qwen2-72b"
EPOCHS=3
LR="2e-5"
BATCH_SIZE=1
GRAD_ACCUM=8
MAX_SEQ_LENGTH=2048
ZERO_STAGE=3
NUM_GPUS=8
GPU_IDS="0,1,2,3,4,5,6,7"
PREP_ONLY=false

# ── Parse args ──────────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case "$1" in
        --condition)     CONDITIONS+=("$2"); shift 2 ;;
        --all)           ALL_CONDITIONS=true; shift ;;
        -n|--items)      ITEMS_PER_CONDITION="$2"; shift 2 ;;
        --data-dir)      DATA_DIR="$2"; shift 2 ;;
        --output-dir)    OUTPUT_DIR="$2"; shift 2 ;;
        --model)         MODEL_PATH="$2"; shift 2 ;;
        --epochs)        EPOCHS="$2"; shift 2 ;;
        --lr)            LR="$2"; shift 2 ;;
        --batch-size)    BATCH_SIZE="$2"; shift 2 ;;
        --grad-accum)    GRAD_ACCUM="$2"; shift 2 ;;
        --max-seq-length) MAX_SEQ_LENGTH="$2"; shift 2 ;;
        --zero)          ZERO_STAGE="$2"; shift 2 ;;
        --num-gpus)      NUM_GPUS="$2"; shift 2 ;;
        --gpu-ids)       GPU_IDS="$2"; shift 2 ;;
        --prep-only)     PREP_ONLY=true; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if $ALL_CONDITIONS; then
    CONDITIONS=("wr_neg" "wr_pos" "wr_neu" "wor_neg" "wor_pos" "wor_neu")
fi

if [[ ${#CONDITIONS[@]} -eq 0 ]]; then
    echo "Usage: bash run_sft.sh --condition <cond> | --all"
    echo "Conditions: wr_neg wr_pos wr_neu wor_neg wor_pos wor_neu"
    exit 1
fi

# Select DeepSpeed config
DS_CONFIG="ds_config_zero${ZERO_STAGE}.json"
if [[ ! -f "$DS_CONFIG" ]]; then
    echo "[ERROR] DeepSpeed config not found: $DS_CONFIG"
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
echo "  SFT Ablation"
echo "  Conditions:    ${CONDITIONS[*]}"
echo "  Items/cond:    ${ITEMS_PER_CONDITION}"
echo "  Model:         ${MODEL_PATH}"
echo "  DeepSpeed:     ZeRO-${ZERO_STAGE} (${DS_CONFIG})"
echo "  GPUs:          ${NUM_GPUS} (${GPU_IDS})"
echo "  Epochs:        ${EPOCHS}"
echo "  LR:            ${LR}"
echo "  Batch:         ${BATCH_SIZE} x ${GRAD_ACCUM} accum"
echo "  Max seq len:   ${MAX_SEQ_LENGTH}"
echo "  Data dir:      ${DATA_DIR}"
echo "  Output dir:    ${OUTPUT_DIR}"
if $PREP_ONLY; then
echo "  Mode:          DATA PREP ONLY"
fi
echo "═══════════════════════════════════════════════════════════════════════"
echo ""

# ── Step 1: Data preparation ───────────────────────────────────────────────

ONLY_ARGS=""
for cond in "${CONDITIONS[@]}"; do
    ONLY_ARGS="${ONLY_ARGS} ${cond}"
done

# Check if data already exists for all conditions
NEED_PREP=false
for cond in "${CONDITIONS[@]}"; do
    if [[ ! -f "${DATA_DIR}/${cond}.jsonl" ]]; then
        NEED_PREP=true
        break
    fi
done

if $NEED_PREP; then
    echo "======== STEP 1: Data Preparation ========"
    python sft_data_prep.py \
        --items-per-condition "$ITEMS_PER_CONDITION" \
        --output-dir "$DATA_DIR" \
        --only ${ONLY_ARGS}
    echo ""
else
    echo "[OK] SFT data already exists for all conditions in ${DATA_DIR}/"
    echo ""
fi

if $PREP_ONLY; then
    echo "Data preparation complete. Exiting (--prep-only)."
    exit 0
fi

# ── Step 2: Finetune per condition ─────────────────────────────────────────

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

for cond in "${CONDITIONS[@]}"; do
    echo "======== FINETUNE: ${cond} ========"
    echo ""

    deepspeed --num_gpus "$NUM_GPUS" sft_train.py \
        --condition "$cond" \
        --data-dir "$DATA_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --model-path "$MODEL_PATH" \
        --epochs "$EPOCHS" \
        --lr "$LR" \
        --batch-size "$BATCH_SIZE" \
        --grad-accum "$GRAD_ACCUM" \
        --max-seq-length "$MAX_SEQ_LENGTH" \
        --deepspeed "$DS_CONFIG" \
        --bf16

    echo ""
    echo "[OK] ${cond} finetuning complete → ${OUTPUT_DIR}/${cond}/"
    echo ""
done

# ── Done ────────────────────────────────────────────────────────────────────

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "  SFT Ablation Complete"
echo "  Checkpoints: ${OUTPUT_DIR}/<condition>/"
echo ""
echo "  To evaluate, serve each checkpoint with vLLM:"
echo "    python -m vllm.entrypoints.openai.api_server \\"
echo "        --model ${OUTPUT_DIR}/wr_neg --port 8199 \\"
echo "        --tensor-parallel-size 8 --dtype float16"
echo ""
echo "  Then run eval:"
echo "    bash run_eval_sft.sh --checkpoint-dir ${OUTPUT_DIR} --condition wr_neg"
echo "═══════════════════════════════════════════════════════════════════════"
