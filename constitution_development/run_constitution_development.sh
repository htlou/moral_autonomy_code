#!/usr/bin/env bash
set -euo pipefail

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${POSITION_REPO_ROOT:-$(cd "${EXP_DIR}/.." && pwd)}"
PYTHON="${PYTHON:-python}"

BACKEND="${BACKEND:-8b}"
EXPERIMENT_ID="${EXPERIMENT_ID:-constitution_dev_$(date '+%Y%m%d_%H%M%S')}"
ITEMS_PER_STAGE="${ITEMS_PER_STAGE:-10000}"
SAVE_EVERY="${SAVE_EVERY:-500}"
BATCH_SIZE="${BATCH_SIZE:-8}"
SEED="${SEED:-42}"
PORT="${PORT:-8198}"

if [[ "${BACKEND}" == "8b" ]]; then
  MODEL_DIR="${MODEL_DIR:-meta-llama/Llama-3.1-8B-Instruct}"
  GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
  TP_SIZE="${TP_SIZE:-1}"
elif [[ "${BACKEND}" == "72b" ]]; then
  MODEL_DIR="${MODEL_DIR:-cognitivecomputations/dolphin-2.9.2-qwen2-72b}"
  GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
  TP_SIZE="${TP_SIZE:-8}"
else
  echo "BACKEND must be 8b or 72b, got ${BACKEND}" >&2
  exit 2
fi

MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.70}"
DTYPE="${DTYPE:-float16}"
if [[ -z "${DP_SIZE:-}" ]]; then
  if [[ "${BACKEND}" == "72b" ]]; then
    DP_SIZE=1
  else
    DP_SIZE="$(awk -F, '{print NF}' <<< "${GPU_IDS}")"
  fi
fi
LOG_DIR="${EXP_DIR}/logs/${EXPERIMENT_ID}/${BACKEND}"
mkdir -p "${LOG_DIR}"

export POSITION_REPO_ROOT="${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export POSITION_MODEL_DIR="${MODEL_DIR}"
export POSITION_VLLM_MODEL="${MODEL_DIR}"
export POSITION_VLLM_URL="http://127.0.0.1:${PORT}/v1"
export POSITION_MAX_CONSTITUTION="${POSITION_MAX_CONSTITUTION:-1000}"
export POSITION_MAX_MEMORY="${POSITION_MAX_MEMORY:-200000}"

port_ready() {
  curl -fsS --max-time 5 "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1
}

VLLM_PID=""
cleanup() {
  if [[ -n "${VLLM_PID}" ]]; then
    kill "${VLLM_PID}" >/dev/null 2>&1 || true
    wait "${VLLM_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

if port_ready; then
  echo "[run] reusing existing OpenAI-compatible server on :${PORT}"
else
  echo "[run] starting vLLM backend=${BACKEND} model=${MODEL_DIR} gpus=${GPU_IDS} tp=${TP_SIZE} dp=${DP_SIZE}"
  VLLM_ARGS=(
    -m vllm.entrypoints.openai.api_server
    --model "${MODEL_DIR}"
    --served-model-name "${MODEL_DIR}"
    --port "${PORT}"
    --tensor-parallel-size "${TP_SIZE}"
    --max-model-len "${MAX_MODEL_LEN}"
    --gpu-memory-utilization "${GPU_MEMORY_UTIL}"
    --dtype "${DTYPE}"
  )
  if (( DP_SIZE > 1 )); then
    VLLM_ARGS+=(--data-parallel-size "${DP_SIZE}")
  fi
  CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${PYTHON}" "${VLLM_ARGS[@]}" > "${LOG_DIR}/vllm.log" 2>&1 &
  VLLM_PID=$!

  waited=0
  until port_ready; do
    sleep 5
    waited=$((waited + 5))
    if (( waited > 600 )); then
      echo "vLLM did not become ready; tail follows:" >&2
      tail -120 "${LOG_DIR}/vllm.log" >&2 || true
      exit 4
    fi
  done
  echo "[run] vLLM ready on :${PORT}"
fi

"${PYTHON}" "${EXP_DIR}/constitution_development_experiment.py" \
  --experiment-id "${EXPERIMENT_ID}" \
  --backend "${BACKEND}" \
  --output-root "${EXP_DIR}" \
  --llm-url "http://127.0.0.1:${PORT}/v1" \
  --items-per-stage "${ITEMS_PER_STAGE}" \
  --save-every "${SAVE_EVERY}" \
  --batch-size "${BATCH_SIZE}" \
  --seed "${SEED}" \
  "$@"
