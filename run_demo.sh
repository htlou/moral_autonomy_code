#!/bin/bash
# Launch the Gradio demo.
# Usage:
#   bash run_demo.sh              # start demo only (assumes vLLM already running)
#   bash run_demo.sh --vllm       # also start vLLM if not running
#   bash run_demo.sh --share      # create public Gradio share link

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Activate conda env
# Activate your Python environment before running this script.

# Keep HTTP proxy for external access (Gradio share tunnel) but:
# - unset SOCKS proxy (all_proxy) which breaks httpx without socksio
# - ensure local calls (vLLM, tiktoken) bypass the proxy via no_proxy
unset all_proxy ALL_PROXY
export no_proxy="localhost,127.0.0.1,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16"
export NO_PROXY="$no_proxy"

VLLM_PORT=8198
MODEL_PATH="cognitivecomputations/dolphin-2.9.2-qwen2-72b"

SHARE_FLAG=""
START_VLLM=false

for arg in "$@"; do
    case "$arg" in
        --vllm)   START_VLLM=true ;;
        --share)  SHARE_FLAG="--share" ;;
    esac
done

# Optionally start vLLM
if $START_VLLM; then
    if curl -s "http://localhost:${VLLM_PORT}/v1/models" > /dev/null 2>&1; then
        echo "[OK] vLLM already running on port ${VLLM_PORT}"
    else
        echo "[..] Starting vLLM on GPU ${VLLM_GPU}, port ${VLLM_PORT}..."
        python -m vllm.entrypoints.openai.api_server \
            --model "$MODEL_PATH" \
            --port "$VLLM_PORT" \
            --tensor-parallel-size 8 \
            --max-model-len 16384 \
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
echo "  Constitution Demo (Gradio)"
echo "========================================="
echo ""

python -u demo.py $SHARE_FLAG &
DEMO_PID=$!

# If --share, start frpc tunnel manually (needs HTTP_PROXY to reach gradio-live.com)
if [ -n "$SHARE_FLAG" ]; then
    echo "[..] Waiting for Gradio to start on port 7860..."
    until curl -s http://localhost:7860 > /dev/null 2>&1; do
        sleep 2
    done
    echo "[OK] Gradio is up locally."

    FRPC_BIN="${GRADIO_FRPC_BIN:-}"
    if [ -x "$FRPC_BIN" ]; then
        TUNNEL_NAME="demo_$(date +%s)"
        echo "[..] Starting share tunnel..."
        HTTP_PROXY=http://127.0.0.1:17890 \
        HTTPS_PROXY=http://127.0.0.1:17890 \
        "$FRPC_BIN" http \
            -n "$TUNNEL_NAME" -l 7860 -i 0.0.0.0 \
            --uc --sd random --ue \
            --server_addr gradio-live.com:7000 \
            --tls_enable --tls_trusted_ca_file .gradio/certificate.pem &
        FRPC_PID=$!
        sleep 3
        echo "[OK] Share tunnel started (PID ${FRPC_PID})."
        echo "     Check output above for the https://xxx.gradio.live URL."
    else
        echo "[WARN] frpc not found. No public share link available."
        echo "       Local access: http://0.0.0.0:7860"
    fi
fi

wait $DEMO_PID
