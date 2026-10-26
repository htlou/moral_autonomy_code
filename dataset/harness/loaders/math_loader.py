"""
Loaders for MATH (Hendrycks), AIME 1983-2024, and AIME 2025.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import List

import pandas as pd

from ..base import (
    DatasetItem,
    DatasetSplit,
    ItemKind,
    ReasoningDataset,
)
from ..eval.math_eval import AIMEEvaluator, MathEvaluator


# ── MATH (Hendrycks) ───────────────────────────────────────────────────────

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


# ── AIME 1983-2024 ─────────────────────────────────────────────────────────

class AIME1983Loader(ReasoningDataset):
    name = "math/aime_1983_2024"
    category = "math"

    def __init__(self, data_root: str | Path):
        super().__init__(data_root)
        self._eval = AIMEEvaluator()

    def get_evaluator(self) -> AIMEEvaluator:
        return self._eval

    def load(self, split: DatasetSplit = DatasetSplit.TEST) -> List[DatasetItem]:
        csv_path = (
            self.data_root
            / "math"
            / "aime_1983_2024"
            / "AIME_Dataset_1983_2024.csv"
        )
        df = pd.read_csv(csv_path)
        items: List[DatasetItem] = []
        for _, row in df.iterrows():
            qid = str(row["ID"])
            # some answers are ambiguous, e.g. "080 or 081"; take first int
            raw_ans = str(row["Answer"]).strip()
            try:
                answer = str(int(float(raw_ans)))
            except ValueError:
                import re
                nums = re.findall(r"\d+", raw_ans)
                answer = nums[0] if nums else raw_ans
            items.append(
                DatasetItem(
                    id=f"aime_1983_2024/{qid}",
                    dataset=self.name,
                    category=self.category,
                    input_text=self._format_prompt(row["Question"]),
                    kind=ItemKind.GROUND_TRUTH,
                    ground_truth=answer,
                    metadata={
                        "year": int(row["Year"]),
                        "problem_number": int(row["Problem Number"]),
                        "part": str(row.get("Part", "")),
                    },
                )
            )
        return items

    @staticmethod
    def _format_prompt(question: str) -> str:
        return (
            "Solve the following AIME problem. The answer is a "
            "non-negative integer from 0 to 999. Show your work step by "
            "step, and put your final answer inside \\boxed{}.\n\n"
            f"Problem: {question}"
        )


# ── AIME 2025 ──────────────────────────────────────────────────────────────

class AIME2025Loader(ReasoningDataset):
    name = "math/aime_2025"
    category = "math"

    def __init__(self, data_root: str | Path):
        super().__init__(data_root)
        self._eval = AIMEEvaluator()

    def get_evaluator(self) -> AIMEEvaluator:
        return self._eval

    def load(self, split: DatasetSplit = DatasetSplit.TEST) -> List[DatasetItem]:
        jsonl_path = self.data_root / "math" / "aime_2025" / "test.jsonl"
        items: List[DatasetItem] = []
        with open(jsonl_path) as f:
            for line in f:
                row = json.loads(line)
                items.append(
                    DatasetItem(
                        id=f"aime_2025/{row['id']}",
                        dataset=self.name,
                        category=self.category,
                        input_text=AIME1983Loader._format_prompt(row["problem"]),
                        kind=ItemKind.GROUND_TRUTH,
                        ground_truth=str(row["answer"]),
                        metadata={"id": row["id"]},
                    )
                )
        return items
