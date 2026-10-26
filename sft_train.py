#!/usr/bin/env python3
"""
SFT finetuning script: finetune Dolphin-Qwen2-72B per condition using DeepSpeed.

For each condition, this script finetunes the base model on the (prompt, reference_answer)
pairs extracted by sft_data_prep.py. This is the SFT ablation counterpart to the
constitution evolution approach.

Uses HuggingFace Trainer + DeepSpeed ZeRO-3 for multi-GPU training.

Usage:
    python sft_train.py --condition wr_neg --data-dir sft_data/
    python sft_train.py --condition wr_neg --data-dir sft_data/ --epochs 3 --lr 2e-5

    # Via shell wrapper (recommended):
    bash run_sft.sh --condition wr_neg
    bash run_sft.sh --all
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

MODEL_PATH = os.environ.get(
    "POSITION_MODEL_DIR",
    "cognitivecomputations/dolphin-2.9.2-qwen2-72b",
)
DEFAULT_OUTPUT_DIR = "sft_checkpoints"
DEFAULT_DATA_DIR = "sft_data"
MAX_SEQ_LENGTH = 2048

CONDITIONS = ["wr_neg", "wr_pos", "wr_neu", "wor_neg", "wor_pos", "wor_neu"]


# ── Dataset ──────────────────────────────────────────────────────────────────

class SFTDataset(Dataset):
    """Chat-format SFT dataset from JSONL files."""

    def __init__(self, data_path: str, tokenizer, max_length: int = MAX_SEQ_LENGTH):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        with open(data_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                self.examples.append(record["messages"])

        logger.info(f"Loaded {len(self.examples)} examples from {data_path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        messages = self.examples[idx]

        # Use the tokenizer's chat template to format messages
        # This handles system/user/assistant roles properly for Qwen2
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )

        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]

        # Create labels: mask everything before the assistant's response
        # so we only compute loss on the reference answer
        labels = input_ids.copy()

        # Find where the assistant response starts by tokenizing
        # just the system+user portion
        prefix_messages = [m for m in messages if m["role"] != "assistant"]
        prefix_text = self.tokenizer.apply_chat_template(
            prefix_messages, tokenize=False, add_generation_prompt=True
        )
        prefix_ids = self.tokenizer(
            prefix_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )["input_ids"]

        # Mask prefix tokens in labels (set to -100)
        prefix_len = len(prefix_ids)
        labels[:prefix_len] = [-100] * prefix_len

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# ── Training ─────────────────────────────────────────────────────────────────

def train_condition(
    condition: str,
    data_dir: str,
    output_dir: str,
    model_path: str,
    epochs: int,
    lr: float,
    batch_size: int,
    grad_accum: int,
    max_seq_length: int,
    deepspeed_config: Optional[str],
    local_rank: int,
    bf16: bool,
):
    """Finetune the model for one condition."""
    data_path = os.path.join(data_dir, f"{condition}.jsonl")
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"SFT data not found: {data_path}\n"
            f"Run: python sft_data_prep.py --items-per-condition 1000"
        )

    cond_output_dir = os.path.join(output_dir, condition)
    os.makedirs(cond_output_dir, exist_ok=True)

    logger.info(f"Loading tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Loading dataset from {data_path}")
    dataset = SFTDataset(data_path, tokenizer, max_length=max_seq_length)

    logger.info(f"Loading model from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if bf16 else torch.float16,
        trust_remote_code=True,
    )
    model.config.use_cache = False  # Required for gradient checkpointing

    # Training arguments
    training_args = TrainingArguments(
        output_dir=cond_output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="no",
        bf16=bf16,
        fp16=not bf16,
        gradient_checkpointing=True,
        deepspeed=deepspeed_config,
        local_rank=local_rank,
        report_to="none",
        dataloader_pin_memory=True,
        remove_unused_columns=False,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        max_length=max_seq_length,
        pad_to_multiple_of=8,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    logger.info(f"Starting training for condition: {condition}")
    logger.info(f"  Examples: {len(dataset)}")
    logger.info(f"  Epochs: {epochs}")
    logger.info(f"  LR: {lr}")
    logger.info(f"  Batch size: {batch_size} x {grad_accum} accum")
    logger.info(f"  Output: {cond_output_dir}")

    trainer.train()
    trainer.save_model(cond_output_dir)
    tokenizer.save_pretrained(cond_output_dir)

    logger.info(f"Model saved to {cond_output_dir}")
    return cond_output_dir


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SFT finetuning per condition")
    parser.add_argument(
        "--condition", type=str, required=True, choices=CONDITIONS,
        help="Which condition to finetune",
    )
    parser.add_argument(
        "--data-dir", type=str, default=DEFAULT_DATA_DIR,
        help="Directory with JSONL files from sft_data_prep.py",
    )
    parser.add_argument(
        "--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--model-path", type=str, default=MODEL_PATH,
        help="Path to base model",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-seq-length", type=int, default=MAX_SEQ_LENGTH)
    parser.add_argument("--deepspeed", type=str, default=None,
                        help="Path to DeepSpeed config JSON")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--no-bf16", dest="bf16", action="store_false")

    args = parser.parse_args()

    train_condition(
        condition=args.condition,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_path=args.model_path,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        max_seq_length=args.max_seq_length,
        deepspeed_config=args.deepspeed,
        local_rank=args.local_rank,
        bf16=args.bf16,
    )


if __name__ == "__main__":
    main()
