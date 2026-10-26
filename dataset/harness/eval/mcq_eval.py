"""
Local evaluators for multiple-choice datasets: MMLU and GPQA.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from ..base import BaseEvaluator, DatasetItem


# ── helpers ─────────────────────────────────────────────────────────────────

LETTER_MAP = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}
LETTER_TO_IDX = {v: k for k, v in LETTER_MAP.items()}


def extract_mcq_choice(text: str) -> Optional[str]:
    """
    Extract the chosen option letter from model output.

    Handles common patterns:
      - "The answer is (B)"
      - "Answer: C"
      - "B)"
      - "**B**"
      - Just the letter on a line by itself
    """
    text = text.strip()

    # Pattern 1: "answer is (X)" or "answer is X"
    m = re.search(
        r"(?:the\s+)?answer\s+is\s*[:\s]*\(?([A-E])\)?",
        text,
        re.IGNORECASE,
    )
    if m:
        return m.group(1).upper()

    # Pattern 2: "Answer: X"
    m = re.search(r"answer\s*:\s*\(?([A-E])\)?", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # Pattern 3: bold letter **X**
    m = re.search(r"\*\*([A-E])\*\*", text)
    if m:
        return m.group(1).upper()

    # Pattern 4: letter followed by ) or . at the start of a line
    m = re.search(r"^([A-E])[).]", text, re.MULTILINE)
    if m:
        return m.group(1).upper()

    # Pattern 5: single letter (last resort — take the last standalone letter)
    matches = re.findall(r"\b([A-E])\b", text)
    if matches:
        return matches[-1].upper()

    return None


# ── MMLU ────────────────────────────────────────────────────────────────────

class MMLUEvaluator(BaseEvaluator):
    """
    MMLU ground truth is an integer index (0-3) into the ``choices`` list.
    The model is expected to respond with a letter A–D.
    """

    def judge(self, item: DatasetItem, model_output: str) -> Dict[str, Any]:
        gt_idx = item.ground_truth  # stored as string of the index
        try:
            gt_idx_int = int(gt_idx)
        except (ValueError, TypeError):
            return {
                "correct": False,
                "extracted_answer": None,
                "detail": f"Invalid ground truth index: {gt_idx!r}",
            }

        gt_letter = LETTER_MAP.get(gt_idx_int, "?")
        pred_letter = extract_mcq_choice(model_output)

        if pred_letter is None:
            return {
                "correct": False,
                "extracted_answer": None,
                "detail": "Could not extract letter choice from model output.",
            }

        correct = pred_letter == gt_letter
        return {
            "correct": correct,
            "extracted_answer": pred_letter,
            "detail": f"gt={gt_letter} (idx {gt_idx_int}), pred={pred_letter}",
        }


# ── GPQA ────────────────────────────────────────────────────────────────────

class GPQAEvaluator(BaseEvaluator):
    """
    GPQA stores correct + 3 incorrect answers as separate strings.
    At prompt time we shuffle them into A/B/C/D; the mapping is stored
    in ``item.metadata["correct_letter"]``.
    """

    def judge(self, item: DatasetItem, model_output: str) -> Dict[str, Any]:
        gt_letter = item.metadata.get("correct_letter")
        if not gt_letter:
            return {
                "correct": False,
                "extracted_answer": None,
                "detail": "Missing correct_letter in metadata.",
            }

        pred_letter = extract_mcq_choice(model_output)
        if pred_letter is None:
            return {
                "correct": False,
                "extracted_answer": None,
                "detail": "Could not extract letter choice from model output.",
            }

        correct = pred_letter == gt_letter
        return {
            "correct": correct,
            "extracted_answer": pred_letter,
            "detail": f"gt={gt_letter}, pred={pred_letter}",
        }
