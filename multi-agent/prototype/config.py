"""Prototype configuration. Override via env vars or CLI."""

import os

# ── LLM (reuse the project's vLLM endpoint by default) ──────────────────────
LLM_BASE_URL = os.environ.get("PROTO_LLM_URL", "http://localhost:8198/v1")
LLM_MODEL = os.environ.get(
    "PROTO_LLM_MODEL",
    "cognitivecomputations/dolphin-2.9-llama3-8b",
)
LLM_API_KEY = os.environ.get("PROTO_LLM_KEY", "dummy")
LLM_TEMPERATURE = float(os.environ.get("PROTO_LLM_TEMP", "0.7"))
LLM_REFLECTION_TEMPERATURE = float(os.environ.get("PROTO_LLM_REFLECT_TEMP", "0.3"))
LLM_MAX_TOKENS_ACTION = 256
LLM_MAX_TOKENS_REFLECT = 1024

# ── Game ────────────────────────────────────────────────────────────────────
GAME_NAME = os.environ.get("PROTO_GAME", "matrix_pd")  # matrix_pd | matrix_sh
NUM_ROUNDS = int(os.environ.get("PROTO_ROUNDS", "30"))
HISTORY_WINDOW = int(os.environ.get("PROTO_HISTORY_WINDOW", "10"))
REFLECT_EVERY = int(os.environ.get("PROTO_REFLECT_EVERY", "5"))

# ── Constitution ────────────────────────────────────────────────────────────
MAX_CONSTITUTION_SIZE = int(os.environ.get("PROTO_MAX_CONST", "20"))

DEFAULT_FIXED_CONSTITUTION = [
    "Cooperate by default; trust is the foundation of mutual gain.",
    "If the other player defects repeatedly, respond in kind to deter exploitation.",
    "Forgive occasional defection; do not let one slip destroy long-term cooperation.",
]
