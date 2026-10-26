"""
Loaders for BigCodeBench, SWE-bench, and SWE-bench Verified.
"""

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
from ..eval.code_eval import BigCodeBenchEvaluator, SWEBenchEvaluator


# ── BigCodeBench ────────────────────────────────────────────────────────────

class BigCodeBenchLoader(ReasoningDataset):
    name = "code/bigcodebench"
    category = "code"

    def __init__(self, data_root: str | Path, version: str = "v0.1.4"):
        super().__init__(data_root)
        self.version = version
        self._eval = BigCodeBenchEvaluator()

    def get_evaluator(self) -> BigCodeBenchEvaluator:
        return self._eval

    def load(self, split: DatasetSplit = DatasetSplit.TEST) -> List[DatasetItem]:
        data_dir = self.data_root / "code" / "bigcodebench" / "data"
        # Pick the versioned parquet; filenames are like v0.1.4-00000-of-00001.parquet
        pq_path = data_dir / f"{self.version}-00000-of-00001.parquet"
        if not pq_path.exists():
            pq_path = data_dir / "train-00000-of-00001.parquet"

        df = pd.read_parquet(pq_path)
        items: List[DatasetItem] = []
        for _, row in df.iterrows():
            task_id = row["task_id"]
            items.append(
                DatasetItem(
                    id=f"bigcodebench/{task_id}",
                    dataset=self.name,
                    category=self.category,
                    input_text=self._format_prompt(
                        row["instruct_prompt"], task_id
                    ),
                    kind=ItemKind.GROUND_TRUTH,
                    ground_truth=row["canonical_solution"],
                    metadata={
                        "code_prompt": row["code_prompt"],
                        "complete_prompt": row["complete_prompt"],
                        "test_code": row["test"],
                        "entry_point": row["entry_point"],
                        "libs": row["libs"],
                    },
                )
            )
        return items

    @staticmethod
    def _format_prompt(instruct_prompt: str, task_id: str) -> str:
        return (
            f"# Task: {task_id}\n\n"
            "Write a Python function that satisfies the following "
            "requirements. Provide only the function body (the `def` "
            "line and imports are already provided).\n\n"
            f"{instruct_prompt}"
        )


# ── SWE-bench ───────────────────────────────────────────────────────────────

class SWEBenchLoader(ReasoningDataset):
    name = "code/swe_bench"
    category = "code"

    def __init__(self, data_root: str | Path):
        super().__init__(data_root)
        self._eval = SWEBenchEvaluator()

    def get_evaluator(self) -> SWEBenchEvaluator:
        return self._eval

    def load(self, split: DatasetSplit = DatasetSplit.TEST) -> List[DatasetItem]:
        swe_dir = self.data_root / "code" / "swe_bench"
        split_map = {
            DatasetSplit.TEST: "test.parquet",
            DatasetSplit.DEV: "dev.parquet",
        }
        fname = split_map.get(split, "test.parquet")
        df = pd.read_parquet(swe_dir / fname)
        return self._df_to_items(df)

    def _df_to_items(self, df: pd.DataFrame) -> List[DatasetItem]:
        items: List[DatasetItem] = []
        for _, row in df.iterrows():
            instance_id = row["instance_id"]
            items.append(
                DatasetItem(
                    id=f"swe_bench/{instance_id}",
                    dataset=self.name,
                    category=self.category,
                    input_text=self._format_prompt(row),
                    kind=ItemKind.GROUND_TRUTH,
                    ground_truth=row["patch"],
                    metadata={
                        "repo": row["repo"],
                        "base_commit": row["base_commit"],
                        "test_patch": row["test_patch"],
                        "hints_text": row.get("hints_text", ""),
                        "fail_to_pass": row["FAIL_TO_PASS"],
                        "pass_to_pass": row["PASS_TO_PASS"],
                        "version": row.get("version", ""),
                    },
                )
            )
        return items

    @staticmethod
    def _format_prompt(row: pd.Series) -> str:
        return (
            f"# Repository: {row['repo']}\n"
            f"# Base commit: {row['base_commit']}\n\n"
            "The following issue was reported. Write a patch (unified diff "
            "format) that fixes it.\n\n"
            f"## Issue\n\n{row['problem_statement']}\n\n"
            + (
                f"## Hints\n\n{row['hints_text']}\n\n"
                if row.get("hints_text")
                else ""
            )
            + "Provide your fix as a unified diff (```diff\\n...\\n```)."
        )


# ── SWE-bench Verified ──────────────────────────────────────────────────────

class SWEBenchVerifiedLoader(SWEBenchLoader):
    name = "code/swe_bench_verified"

    def load(self, split: DatasetSplit = DatasetSplit.TEST) -> List[DatasetItem]:
        pq_path = (
            self.data_root
            / "code"
            / "swe_bench_verified"
            / "data"
            / "test-00000-of-00001.parquet"
        )
        df = pd.read_parquet(pq_path)
        items = self._df_to_items(df)
        # inject difficulty into metadata
        for item, (_, row) in zip(items, df.iterrows()):
            item.metadata["difficulty"] = row.get("difficulty", "")
            item.id = item.id.replace("swe_bench/", "swe_bench_verified/")
            item.dataset = self.name
        return items
