"""
Base classes for the dataset harness.

Terminology (intentional):
  - "ground_truth"     → used ONLY for reasoning/code datasets where there is
                          an objectively verifiable correct answer.
  - "reference_answer" → used for safety / value datasets where the label
                          reflects one perspective and we stay value-neutral.
"""

from __future__ import annotations

import abc
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union


# ── Canonical item ──────────────────────────────────────────────────────────

class ItemKind(str, Enum):
    """How the label should be interpreted downstream."""
    GROUND_TRUTH = "ground_truth"          # objectively verifiable
    REFERENCE_ANSWER = "reference_answer"  # value-laden / perspective-dependent
    PREFERENCE_PAIR = "preference_pair"    # chosen vs rejected
    PROMPT_ONLY = "prompt_only"            # no label; measure generation


@dataclass
class DatasetItem:
    """
    A single example in the unified harness format.

    Fields
    ------
    id          : unique identifier within the dataset
    dataset     : dataset name, e.g. "math/hendrycks"
    category    : top-level category (code | math | reasoning | safety | value)
    input_text  : the prompt that will be sent to the LM
    kind        : how the label should be interpreted (see ItemKind)

    # For GROUND_TRUTH items
    ground_truth       : the objectively correct answer (string)
    ground_truth_detail: extra info (e.g. full step-by-step solution)

    # For REFERENCE_ANSWER items
    reference_answer   : the label from the dataset (not necessarily "correct")

    # For PREFERENCE_PAIR items
    chosen             : the preferred response
    rejected           : the less-preferred response

    # Metadata bag — anything dataset-specific goes here
    metadata           : dict
    """
    id: str
    dataset: str
    category: str
    input_text: str
    kind: ItemKind

    ground_truth: Optional[str] = None
    ground_truth_detail: Optional[str] = None

    reference_answer: Optional[str] = None

    chosen: Optional[str] = None
    rejected: Optional[str] = None

    metadata: Dict[str, Any] = field(default_factory=dict)

    # ── convenience ──

    def to_lm_payload(self) -> Dict[str, Any]:
        """
        Build the dict that gets sent to the deployed LM.

        The LM always sees:
          - prompt
          - label_kind  (so it knows how to treat the label)
          - label       (ground_truth | reference_answer | preference pair)
          - metadata

        The LM can then decide to modify its RAG retrieval or
        constitution based on the label_kind.
        """
        payload: Dict[str, Any] = {
            "id": self.id,
            "dataset": self.dataset,
            "category": self.category,
            "prompt": self.input_text,
            "label_kind": self.kind.value,
            "metadata": self.metadata,
        }

        if self.kind == ItemKind.GROUND_TRUTH:
            payload["ground_truth"] = self.ground_truth
            if self.ground_truth_detail:
                payload["ground_truth_detail"] = self.ground_truth_detail
        elif self.kind == ItemKind.REFERENCE_ANSWER:
            payload["reference_answer"] = self.reference_answer
        elif self.kind == ItemKind.PREFERENCE_PAIR:
            payload["chosen"] = self.chosen
            payload["rejected"] = self.rejected
        # PROMPT_ONLY → no label field

        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_lm_payload(), ensure_ascii=False)


# ── Dataset split ───────────────────────────────────────────────────────────

class DatasetSplit(str, Enum):
    TRAIN = "train"
    DEV = "dev"
    TEST = "test"
    ALL = "all"


# ── Abstract loader ─────────────────────────────────────────────────────────

class BaseDatasetLoader(abc.ABC):
    """Every loader subclass must implement ``load``."""

    name: str                    # e.g. "math/hendrycks"
    category: str                # e.g. "math"
    kind: ItemKind               # default item kind

    def __init__(self, data_root: Union[str, Path]):
        self.data_root = Path(data_root)

    @abc.abstractmethod
    def load(self, split: DatasetSplit = DatasetSplit.TEST) -> List[DatasetItem]:
        ...

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r} root={self.data_root}>"


class ReasoningDataset(BaseDatasetLoader):
    """
    A dataset with objectively verifiable answers.
    Subclasses may attach an ``evaluator`` that judges LM output locally.
    """
    kind = ItemKind.GROUND_TRUTH

    @abc.abstractmethod
    def get_evaluator(self) -> "BaseEvaluator":
        ...


class NonReasoningDataset(BaseDatasetLoader):
    """
    Safety / value datasets.  Labels are reference answers, not ground truth.
    """
    kind = ItemKind.REFERENCE_ANSWER


# ── Abstract evaluator ──────────────────────────────────────────────────────

class BaseEvaluator(abc.ABC):
    """
    Judges whether an LM response is correct for a reasoning dataset.
    Returns True/False per item, plus optional detail.
    """

    @abc.abstractmethod
    def judge(self, item: DatasetItem, model_output: str) -> Dict[str, Any]:
        """
        Returns
        -------
        {
            "correct": bool,
            "extracted_answer": str | None,
            "detail": str | None,
        }
        """
        ...

    def judge_batch(
        self, items: Sequence[DatasetItem], outputs: Sequence[str]
    ) -> List[Dict[str, Any]]:
        return [self.judge(it, out) for it, out in zip(items, outputs)]

    def accuracy(
        self, items: Sequence[DatasetItem], outputs: Sequence[str]
    ) -> float:
        results = self.judge_batch(items, outputs)
        if not results:
            return 0.0
        return sum(r["correct"] for r in results) / len(results)
