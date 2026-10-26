"""Global configuration for the self-modifying RAG project."""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

# ── Model ────────────────────────────────────────────────────────────────────
# Override via environment: POSITION_MODEL_DIR, POSITION_VLLM_URL, etc.

MODEL_DIR = os.environ.get(
    "POSITION_MODEL_DIR",
    "cognitivecomputations/dolphin-2.9-llama3-8b",
)
VLLM_BASE_URL = os.environ.get("POSITION_VLLM_URL", "http://localhost:8198/v1")
VLLM_MODEL_NAME = os.environ.get("POSITION_VLLM_MODEL", MODEL_DIR)
VLLM_GPU = os.environ.get("POSITION_VLLM_GPU", "7")

# Embedding config — use small GPT2 instead of the main model.
# GPT2 (~124M params, 768 dim) runs in milliseconds per embedding.
EMBEDDING_MODEL_PATH = os.environ.get("POSITION_EMBEDDING_MODEL", "gpt2")
EMBEDDING_DIM = 768  # gpt2 hidden_size
EMBEDDING_DEVICE = "cpu"  # GPU occupied by vLLM; CPU is fine for GPT2
EMBEDDING_BATCH_SIZE = 16
EMBEDDING_MAX_LENGTH = 512  # max tokens per text for embedding

# Use LightRAG "naive" query mode — skips entity/KG extraction, which
# would otherwise trigger extra LLM calls per insert/query.
RAG_QUERY_MODE = "naive"

RAG_WORKING_DIR = os.environ.get("POSITION_RAG_DIR", str(REPO_ROOT / "rag_storage"))
MEMORY_WORKING_DIR = os.environ.get("POSITION_MEMORY_DIR", str(REPO_ROOT / "memory_storage"))

# ── Constitution / Memory limits ─────────────────────────────────────────────
# Override via environment for different experiment sizes.

MAX_CONSTITUTION_SIZE = int(os.environ.get("POSITION_MAX_CONSTITUTION", "1000"))
MAX_MEMORY_SIZE = int(os.environ.get("POSITION_MAX_MEMORY", "1000"))
CONFIDENCE_THRESHOLD = 0.3  # min confidence to keep a principle
EVIDENCE_REQUIRED = 2  # interactions needed before principle becomes active
SIMILARITY_MERGE_THRESHOLD = 0.85  # merge principles above this cosine sim

# ── MMLU factual subjects ────────────────────────────────────────────────────
# Subjects focusing on factual knowledge (excluding math/logic/CS/physics).

MMLU_FACTUAL_SUBJECTS = [
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_medicine",
    "global_facts",
    "high_school_biology",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_microeconomics",
    "high_school_psychology",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]
