"""MMLU dataset loader."""


from __future__ import annotations


from pathlib import Path


from typing import List, Optional


import pandas as pd


from ..base import (
    DatasetItem,
    DatasetSplit,
    ItemKind,
    ReasoningDataset,
)


from ..eval.mcq_eval import MMLUEvaluator, LETTER_MAP


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
