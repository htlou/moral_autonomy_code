"""
Loaders for MMLU and GPQA.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import List, Optional

import pandas as pd

from ..base import (
    DatasetItem,
    DatasetSplit,
    ItemKind,
    ReasoningDataset,
)
from ..eval.mcq_eval import GPQAEvaluator, MMLUEvaluator, LETTER_MAP


# ── MMLU ────────────────────────────────────────────────────────────────────

class MMLULoader(ReasoningDataset):
    name = "reasoning/mmlu"
    category = "reasoning"

    def __init__(self, data_root: str | Path):
        super().__init__(data_root)
        self._eval = MMLUEvaluator()

    def get_evaluator(self) -> MMLUEvaluator:
        return self._eval

    def load(
        self,
        split: DatasetSplit = DatasetSplit.TEST,
        subjects: Optional[List[str]] = None,
    ) -> List[DatasetItem]:
        mmlu_dir = self.data_root / "general_reasoning" / "mmlu"
        split_file = {
            DatasetSplit.TRAIN: "dev-00000-of-00001.parquet",   # MMLU "dev" = few-shot
            DatasetSplit.DEV: "validation-00000-of-00001.parquet",
            DatasetSplit.TEST: "test-00000-of-00001.parquet",
        }
        fname = split_file.get(split)
        if fname is None:
            raise ValueError(f"MMLU has train/dev/test splits, got {split}")

        # discover subjects
        if subjects is None:
            subjects = sorted(
                d.name
                for d in mmlu_dir.iterdir()
                if d.is_dir() and d.name not in {"all", "auxiliary_train"}
            )

        items: List[DatasetItem] = []
        for subj in subjects:
            pq = mmlu_dir / subj / fname
            if not pq.exists():
                continue
            df = pd.read_parquet(pq)
            for idx, row in df.iterrows():
                choices = list(row["choices"])  # convert ndarray to list
                answer_idx = int(row["answer"])
                items.append(
                    DatasetItem(
                        id=f"mmlu/{subj}/{split.value}/{idx}",
                        dataset=self.name,
                        category=self.category,
                        input_text=self._format_prompt(
                            row["question"], choices, subj
                        ),
                        kind=ItemKind.GROUND_TRUTH,
                        ground_truth=str(answer_idx),
                        metadata={
                            "subject": subj,
                            "choices": choices,
                            "correct_letter": LETTER_MAP[answer_idx],
                        },
                    )
                )
        return items

    @staticmethod
    def _format_prompt(question: str, choices: list, subject: str) -> str:
        subj_pretty = subject.replace("_", " ").title()
        options = "\n".join(
            f"  {LETTER_MAP[i]}. {c}" for i, c in enumerate(choices)
        )
        return (
            f"Subject: {subj_pretty}\n\n"
            f"Question: {question}\n\n"
            f"Options:\n{options}\n\n"
            "Answer with the letter (A, B, C, or D) of the correct option."
        )


# ── GPQA ────────────────────────────────────────────────────────────────────

class GPQALoader(ReasoningDataset):
    name = "reasoning/gpqa"
    category = "reasoning"

    def __init__(
        self,
        data_root: str | Path,
        variant: str = "diamond",
        seed: int = 42,
    ):
        super().__init__(data_root)
        self.variant = variant  # diamond | main | extended
        self.seed = seed
        self._eval = GPQAEvaluator()

    def get_evaluator(self) -> GPQAEvaluator:
        return self._eval

    def load(self, split: DatasetSplit = DatasetSplit.TEST) -> List[DatasetItem]:
        csv_path = (
            self.data_root
            / "general_reasoning"
            / "gpqa"
            / "dataset"
            / f"gpqa_{self.variant}.csv"
        )
        df = pd.read_csv(csv_path)
        rng = random.Random(self.seed)

        items: List[DatasetItem] = []
        for idx, row in df.iterrows():
            correct = row["Correct Answer"]
            incorrects = [
                row["Incorrect Answer 1"],
                row["Incorrect Answer 2"],
                row["Incorrect Answer 3"],
            ]
            # shuffle choices deterministically
            choices = [correct] + incorrects
            order = list(range(4))
            rng.shuffle(order)
            shuffled = [choices[i] for i in order]
            correct_letter = LETTER_MAP[order.index(0)]

            items.append(
                DatasetItem(
                    id=f"gpqa/{self.variant}/{idx}",
                    dataset=self.name,
                    category=self.category,
                    input_text=self._format_prompt(
                        row["Question"], shuffled, row.get("Subdomain", "")
                    ),
                    kind=ItemKind.GROUND_TRUTH,
                    ground_truth=correct,
                    metadata={
                        "correct_letter": correct_letter,
                        "choices": shuffled,
                        "subdomain": row.get("Subdomain", ""),
                        "domain": row.get("High-level domain", ""),
                        "record_id": row.get("Record ID", ""),
                    },
                )
            )
        return items

    @staticmethod
    def _format_prompt(
        question: str, choices: list, subdomain: str
    ) -> str:
        options = "\n".join(
            f"  {LETTER_MAP[i]}. {c}" for i, c in enumerate(choices)
        )
        header = f"Domain: {subdomain}\n\n" if subdomain else ""
        return (
            f"{header}"
            f"Question: {question}\n\n"
            f"Options:\n{options}\n\n"
            "Answer with the letter (A, B, C, or D) of the correct option."
        )
