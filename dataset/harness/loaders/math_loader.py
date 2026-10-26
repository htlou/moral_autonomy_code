"""MATH dataset loader."""


from __future__ import annotations


from pathlib import Path


from typing import List


import pandas as pd


from ..base import (
    DatasetItem,
    DatasetSplit,
    ItemKind,
    ReasoningDataset,
)


from ..eval.math_eval import MathEvaluator


MATH_SUBTOPICS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]


SPLIT_FILE_MAP = {
    DatasetSplit.TRAIN: "train-00000-of-00001.parquet",
    DatasetSplit.TEST: "test-00000-of-00001.parquet",
}


class MATHLoader(ReasoningDataset):
    name = "math/hendrycks"
    category = "math"

    def __init__(self, data_root: str | Path):
        super().__init__(data_root)
        self._eval = MathEvaluator()

    def get_evaluator(self) -> MathEvaluator:
        return self._eval

    def load(
        self,
        split: DatasetSplit = DatasetSplit.TEST,
        subtopics: List[str] | None = None,
    ) -> List[DatasetItem]:
        subtopics = subtopics or MATH_SUBTOPICS
        fname = SPLIT_FILE_MAP.get(split)
        if fname is None:
            raise ValueError(f"MATH only has train/test splits, got {split}")

        items: List[DatasetItem] = []
        for topic in subtopics:
            pq_path = self.data_root / "math" / "hendrycks_math" / topic / fname
            if not pq_path.exists():
                continue
            df = pd.read_parquet(pq_path)
            for idx, row in df.iterrows():
                items.append(
                    DatasetItem(
                        id=f"math_hendrycks/{topic}/{split.value}/{idx}",
                        dataset=self.name,
                        category=self.category,
                        input_text=self._format_prompt(row["problem"]),
                        kind=ItemKind.GROUND_TRUTH,
                        ground_truth=self._extract_answer(row["solution"]),
                        ground_truth_detail=row["solution"],
                        metadata={
                            "level": row.get("level", ""),
                            "type": row.get("type", topic),
                            "subtopic": topic,
                        },
                    )
                )
        return items

    # ── prompt formatting ──

    @staticmethod
    def _format_prompt(problem: str) -> str:
        return (
            "Solve the following math problem. "
            "Show your work step by step, and put your final answer "
            "inside \\boxed{}.\n\n"
            f"Problem: {problem}"
        )

    @staticmethod
    def _extract_answer(solution: str) -> str:
        """Pull \\boxed{} from the solution; fall back to full string."""
        from ..eval.math_eval import extract_boxed

        boxed = extract_boxed(solution)
        return boxed if boxed is not None else solution.strip()
