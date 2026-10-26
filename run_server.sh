#!/bin/bash
# Launch the constitution self-modification portal.
# Usage:
#   bash run_server.sh            # start web server only (assumes vLLM already running)
#   bash run_server.sh --vllm     # also start vLLM if not running

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Activate conda env
# Activate your Python environment before running this script.

# Disable proxy — tiktoken and vLLM calls are all local/direct
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

SERVER_PORT="${SERVER_PORT:-8200}"
VLLM_PORT=8198
MODEL_PATH="cognitivecomputations/dolphin-2.9.2-qwen2-72b"

# Optionally start vLLM
if [[ "$1" == "--vllm" ]]; then
    if curl -s "http://localhost:${VLLM_PORT}/v1/models" > /dev/null 2>&1; then
        echo "[OK] vLLM already running on port ${VLLM_PORT}"
    else
        echo "[..] Starting vLLM on all GPUs (TP=8), port ${VLLM_PORT}..."
        python -m vllm.entrypoints.openai.api_server \
            --model "$MODEL_PATH" \
            --port "$VLLM_PORT" \
            --tensor-parallel-size 8 \
            --max-model-len 4096 \
            --dtype float16 &
        VLLM_PID=$!
        echo "[..] Waiting for vLLM (PID ${VLLM_PID}) to be ready..."
        until curl -s "http://localhost:${VLLM_PORT}/v1/models" > /dev/null 2>&1; do
            sleep 2
        done
        echo "[OK] vLLM ready."
    fi
fi

echo ""
echo "========================================="
echo "  Constitution Portal"
echo "  http://0.0.0.0:${SERVER_PORT}"
echo "========================================="
echo ""

python -m uvicorn server.app:app --host 0.0.0.0 --port "$SERVER_PORT"
