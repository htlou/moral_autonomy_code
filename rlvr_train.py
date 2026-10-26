#!/usr/bin/env python3
"""
RLVR training script: GRPO finetuning per condition.

Uses TRL's GRPOTrainer with two reward functions:
  - math_reward: verifiable logic matching (\\boxed{} extraction)
  - value_reward: Claude Sonnet 4.6 judge (safety/value alignment)

Each training item carries a reward_type tag ("math" or "value") that
routes it to the appropriate reward function.

Usage:
    python rlvr_train.py --condition wr_neg --data-dir rlvr_data/
    python rlvr_train.py --condition wor_neg --data-dir rlvr_data/ --epochs 1

    # Via shell wrapper (recommended):
    bash run_all_8b_rlvr.sh
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Bump NCCL collective timeout to 30 min BEFORE torch is imported.
# Default is 600s; one slow vLLM rollout can stall a rank past that and trip
# the watchdog → ALLREDUCE timeout → whole run dies.
os.environ.setdefault("TORCH_NCCL_DEFAULT_PG_TIMEOUT_MS", "1800000")
os.environ.setdefault("NCCL_TIMEOUT", "1800")

import torch
import torch.distributed as _dist

# Backstop: HF/accelerate/deepspeed may pass explicit timeouts that override
# the env var, OR create sub-groups (PG 1, PG 2…) that don't inherit our
# long timeout. We saw `[PG ID 1] Timeout(ms)=600000` repeatedly. Patch every
# entry point and override the torch module-level constants too.
_LONG_TIMEOUT = datetime.timedelta(minutes=30)

# Override torch.distributed module-level defaults so anything reading them
# at import time picks up 30 min.
try:
    import torch.distributed.constants as _dist_constants
    _dist_constants.default_pg_nccl_timeout = _LONG_TIMEOUT
    _dist_constants.default_pg_timeout = _LONG_TIMEOUT
except Exception:
    pass
try:
    from torch.distributed import distributed_c10d as _c10d
    if hasattr(_c10d, "_DEFAULT_PG_TIMEOUT"):
        _c10d._DEFAULT_PG_TIMEOUT = _LONG_TIMEOUT
    if hasattr(_c10d, "_DEFAULT_PG_NCCL_TIMEOUT"):
        _c10d._DEFAULT_PG_NCCL_TIMEOUT = _LONG_TIMEOUT
except Exception:
    pass

# Wrap init_process_group AND new_group to force >= 30 min timeout.
def _wrap_with_long_timeout(orig):
    def _wrapped(*args, **kwargs):
        t = kwargs.get("timeout")
        if t is None or t < _LONG_TIMEOUT:
            kwargs["timeout"] = _LONG_TIMEOUT
        return orig(*args, **kwargs)
    return _wrapped

_dist.init_process_group = _wrap_with_long_timeout(_dist.init_process_group)
_dist.new_group = _wrap_with_long_timeout(_dist.new_group)
# Some callers go through the c10d module directly:
try:
    _c10d.init_process_group = _wrap_with_long_timeout(_c10d.init_process_group)
    _c10d.new_group = _wrap_with_long_timeout(_c10d.new_group)
except Exception:
    pass

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rlvr_reward import math_reward, value_reward_batch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = os.environ.get(
    "POSITION_MODEL_DIR",
    "cognitivecomputations/dolphin-2.9-llama3-8b",
)
DEFAULT_OUTPUT_DIR = "rlvr_checkpoints"


# ── Dataset ─────────────────────────────────────────────────────────────────

def load_rlvr_dataset(data_dir: str, condition: str) -> Dataset:
    """Load RLVR prompt dataset from JSONL."""
    path = Path(data_dir) / f"{condition}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"RLVR data not found: {path}")

    records = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            records.append(rec)

    # GRPOTrainer expects a "prompt" column (list of messages or string)
    dataset_dict = {
        "prompt": [],
        "ground_truth": [],
        "reference_answer": [],
        "reward_type": [],
        "value_type": [],
        "original_prompt": [],  # raw prompt text for judge
    }

    for rec in records:
        # Format as chat messages for the model
        dataset_dict["prompt"].append([
            {"role": "user", "content": rec["prompt"]},
        ])
        dataset_dict["ground_truth"].append(rec.get("ground_truth", ""))
        dataset_dict["reference_answer"].append(rec.get("reference_answer", ""))
        dataset_dict["reward_type"].append(rec.get("reward_type", "value"))
        dataset_dict["value_type"].append(rec.get("value_type", "pos"))
        dataset_dict["original_prompt"].append(rec["prompt"])

    return Dataset.from_dict(dataset_dict)


# ── Reward function wrapper ─────────────────────────────────────────────────

def make_combined_reward_fn(max_concurrent: int = 8):
    """
    Create a reward function that routes to math_reward or value_reward
    based on each sample's reward_type metadata.

    Only rank 0 calls the Claude API; rewards are broadcast via torch.distributed.
    """
    def combined_reward(completions, **kwargs):
        """
        Combined reward function for GRPOTrainer.

        GRPOTrainer calls this with completions and additional columns
        from the dataset as kwargs.
        """
        ground_truths = kwargs.get("ground_truth", [""] * len(completions))
        reward_types = kwargs.get("reward_type", ["value"] * len(completions))
        value_types = kwargs.get("value_type", ["pos"] * len(completions))
        reference_answers = kwargs.get("reference_answer", [""] * len(completions))
        original_prompts = kwargs.get("original_prompt", [""] * len(completions))

        # Extract text from completions (may be list of message dicts)
        comp_texts = []
        for c in completions:
            if isinstance(c, list):
                text = c[-1]["content"] if c and isinstance(c[-1], dict) else str(c)
            elif isinstance(c, dict):
                text = c.get("content", str(c))
            else:
                text = str(c)
            comp_texts.append(text)

        # Determine rank for distributed training
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        is_distributed = torch.distributed.is_initialized()

        rewards = [0.0] * len(comp_texts)

        # Math rewards — deterministic, safe on all ranks
        math_indices = [i for i, rt in enumerate(reward_types) if rt == "math"]
        if math_indices:
            math_completions = [comp_texts[i] for i in math_indices]
            math_gts = [ground_truths[i] for i in math_indices]
            math_rewards = math_reward(math_completions, math_gts)
            for idx, r in zip(math_indices, math_rewards):
                rewards[idx] = r

        # Value rewards — Claude API on rank 0 only, then broadcast
        value_indices = [i for i, rt in enumerate(reward_types) if rt == "value"]
        if value_indices and local_rank == 0:
            val_completions = [comp_texts[i] for i in value_indices]
            val_prompts = [original_prompts[i] for i in value_indices]
            val_refs = [reference_answers[i] for i in value_indices]
            val_type = value_types[value_indices[0]] if value_indices else "pos"
            val_rewards = value_reward_batch(
                val_completions, val_prompts,
                reference_answers=val_refs,
                value_type=val_type,
                max_concurrent=max_concurrent,
            )
            for idx, r in zip(value_indices, val_rewards):
                rewards[idx] = r

        # Synchronize rewards across all ranks
        if is_distributed:
            rewards_tensor = torch.tensor(rewards, dtype=torch.float32).cuda()
            torch.distributed.broadcast(rewards_tensor, src=0)
            rewards = rewards_tensor.tolist()

        return rewards

    return combined_reward


# ── Training ────────────────────────────────────────────────────────────────

def train(
    condition: str,
    data_dir: str,
    output_dir: str,
    model_path: str,
    epochs: int = 1,
    lr: float = 5e-6,
    batch_size: int = 4,
    grad_accum: int = 2,
    max_completion_length: int = 768,
    num_generations: int = 4,
    beta: float = 0.04,
    judge_concurrency: int = 8,
    use_vllm: bool = False,
    deepspeed_config: Optional[str] = None,
    bf16: bool = True,
    fp16: bool = False,
    resume_from_checkpoint: bool = True,
    cast_lm_head_to_fp32: bool = False,
    local_rank: int = -1,
    max_steps: int = -1,
):
    """Run GRPO training for one condition."""
    logger.info(f"RLVR Training: condition={condition}")
    logger.info(f"  model: {model_path}")
    logger.info(f"  data: {data_dir}/{condition}.jsonl")
    logger.info(f"  epochs={epochs}, lr={lr}, beta={beta}")
    logger.info(f"  num_generations={num_generations}, max_completion_length={max_completion_length}")

    # Load dataset
    dataset = load_rlvr_dataset(data_dir, condition)
    logger.info(f"  dataset: {len(dataset)} items")

    n_math = sum(1 for rt in dataset["reward_type"] if rt == "math")
    n_value = sum(1 for rt in dataset["reward_type"] if rt == "value")
    logger.info(f"  math items: {n_math}, value items: {n_value}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Output path
    cond_output = Path(output_dir) / condition
    cond_output.mkdir(parents=True, exist_ok=True)

    if fp16 and bf16:
        raise ValueError("Choose only one precision mode: fp16 or bf16")
    if (fp16 or bf16) and cast_lm_head_to_fp32:
        logger.warning("Disabling cast_lm_head_to_fp32 for mixed precision to avoid lm_head dtype mismatch during generation.")
        cast_lm_head_to_fp32 = False
    model_dtype = torch.float16 if fp16 else torch.bfloat16 if bf16 else torch.float32

    # GRPO config
    training_args = GRPOConfig(
        output_dir=str(cond_output),
        num_train_epochs=epochs,
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=1,
        bf16=bf16,
        fp16=fp16,
        disable_dropout=True,
        cast_lm_head_to_fp32=cast_lm_head_to_fp32,
        gradient_checkpointing=True,
        report_to="wandb",
        run_name=f"rlvr_{condition}",
        # GRPO-specific
        num_generations=num_generations,
        max_completion_length=max_completion_length,
        temperature=0.6,
        top_p=0.95,
        beta=beta,
        loss_type="grpo",
        scale_rewards="group",
        mask_truncated_completions=True,
        # Conservative for neutral conditions which NaN-out on GRPO
        # almost immediately at step 16. Was 0.5 → 0.1 → 0.05.
        max_grad_norm=0.05,
        epsilon_high=0.2,
        # HF generate path. vLLM server hung on weight sync, vLLM colocate
        # OOM'd on KV cache. For fresh training (not resume from a checkpoint
        # on the numerical edge), HF generate has been the most stable —
        # fresh wr_neg got to 85% before NaN.
        use_vllm=use_vllm,
        generation_kwargs={
            "remove_invalid_values": True,
            "renormalize_logits": True,
        },
        # DeepSpeed
        deepspeed=deepspeed_config,
    )

    # Build reward function
    reward_fn = make_combined_reward_fn(max_concurrent=judge_concurrency)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=model_dtype,
        trust_remote_code=True,
    )

    # Create trainer
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    # Auto-resume if a prior checkpoint exists in this condition's output dir.
    resume = resume_from_checkpoint and any(cond_output.glob("checkpoint-*"))
    if resume:
        latest = max(cond_output.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
        logger.info(f"Resuming GRPO training from {latest}")
    else:
        logger.info("Starting GRPO training...")
    trainer.train(resume_from_checkpoint=resume)

    # Save final model
    logger.info(f"Saving model to {cond_output}")
    trainer.save_model(str(cond_output))
    tokenizer.save_pretrained(str(cond_output))

    logger.info(f"RLVR training complete: {condition} → {cond_output}")


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RLVR (GRPO) training per condition")
    parser.add_argument("--condition", required=True)
    parser.add_argument("--data-dir", default="rlvr_data")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=-1, help="Override epoch-based training length; paper target: 4500")
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-completion-length", type=int, default=768)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--judge-concurrency", type=int, default=8)
    parser.add_argument("--use-vllm", action="store_true")
    parser.add_argument("--deepspeed", type=str, default=None)
    parser.add_argument("--bf16", dest="bf16", action="store_true", default=True)
    parser.add_argument("--no-bf16", dest="bf16", action="store_false")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--no-resume", dest="resume_from_checkpoint", action="store_false", default=True)
    parser.add_argument("--cast-lm-head-to-fp32", dest="cast_lm_head_to_fp32", action="store_true", default=False)
    parser.add_argument("--no-cast-lm-head-to-fp32", dest="cast_lm_head_to_fp32", action="store_false")
    parser.add_argument("--local_rank", "--local-rank", type=int, default=-1)

    args = parser.parse_args()
    if args.fp16:
        args.bf16 = False

    train(
        condition=args.condition,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_path=args.model_path,
        epochs=args.epochs,
        max_steps=args.max_steps,
        lr=args.lr,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        beta=args.beta,
        judge_concurrency=args.judge_concurrency,
        use_vllm=args.use_vllm,
        deepspeed_config=args.deepspeed,
        bf16=args.bf16,
        fp16=args.fp16,
        resume_from_checkpoint=args.resume_from_checkpoint,
        cast_lm_head_to_fp32=args.cast_lm_head_to_fp32,
        local_rank=args.local_rank,
    )


if __name__ == "__main__":
    main()
