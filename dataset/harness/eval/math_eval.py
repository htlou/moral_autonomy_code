"""
Local evaluator for MATH (Hendrycks).

MATH answers are extracted from \\boxed{} in both the ground truth and
the model output.
"""


from __future__ import annotations


import re


from typing import Any, Dict, Optional


from ..base import BaseEvaluator, DatasetItem


def _strip_latex(s: str) -> str:
    """Lightly normalise a LaTeX string for comparison."""
    s = s.strip()
    # remove \text{...}, \textbf{...}, \mathrm{...}
    s = re.sub(r"\\(?:text|textbf|mathrm|mathbf)\{([^}]*)\}", r"\1", s)
    # remove \left / \right
    s = s.replace("\\left", "").replace("\\right", "")
    # remove \, \; \! \  spacing commands
    s = re.sub(r"\\[,;! ]", "", s)
    # collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def extract_boxed(text: str) -> Optional[str]:
    """
    Extract the *last* \\boxed{…} from ``text``, handling nested braces.
    Returns None if no \\boxed found.
    """
    # find the last occurrence of \boxed{
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    start = idx + len("\\boxed{")
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return text[start : i - 1]


def extract_last_number(text: str) -> Optional[str]:
    """Fallback: grab the last number-like token from the text."""
    nums = re.findall(r"-?\d+(?:\.\d+)?(?:/\d+)?", text)
    return nums[-1] if nums else None


def normalise_answer(answer: str) -> str:
    """
    Best-effort normalisation so that '\\frac{1}{2}' == '0.5', etc.
    Falls back to stripped string comparison.
    """
    a = _strip_latex(answer)
    # try to evaluate simple fractions
    m = re.fullmatch(r"\\frac\{([^}]+)\}\{([^}]+)\}", a)
    if m:
        try:
            return str(float(m.group(1)) / float(m.group(2)))
        except (ValueError, ZeroDivisionError):
            pass
    # try plain float
    try:
        return str(float(a))
    except ValueError:
        pass
    return a.lower()


class MathEvaluator(BaseEvaluator):
    """
    Judges MATH (Hendrycks) responses.

    Extraction pipeline:
      1. Look for \\boxed{…} in the model output.
      2. Fallback: take the last number.
      3. Normalise and compare to the ground-truth \\boxed{…}.
    """

    def judge(self, item: DatasetItem, model_output: str) -> Dict[str, Any]:
        gt_raw = item.ground_truth or ""
        gt_boxed = extract_boxed(gt_raw)
        if gt_boxed is None:
            # ground truth might already be just the answer string
            gt_boxed = gt_raw.strip()

        pred_boxed = extract_boxed(model_output)
        if pred_boxed is None:
            pred_boxed = extract_last_number(model_output)

        if pred_boxed is None:
            return {
                "correct": False,
                "extracted_answer": None,
                "detail": "Could not extract answer from model output.",
            }

        gt_norm = normalise_answer(gt_boxed)
        pred_norm = normalise_answer(pred_boxed)

        correct = gt_norm == pred_norm
        return {
            "correct": correct,
            "extracted_answer": pred_boxed,
            "detail": f"gt={gt_boxed!r} -> {gt_norm!r}; pred={pred_boxed!r} -> {pred_norm!r}",
        }
