"""
Loaders for safety datasets: BeaverTails, Anthropic Red Team, HH-RLHF.

All labels are treated as *reference answers* (not ground truth) to stay
value-neutral.  The harness surfaces them so the deployed LM can
condition its RAG / constitution on the safety signal.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import List

from ..base import (
    DatasetItem,
    DatasetSplit,
    ItemKind,
    NonReasoningDataset,
)


# ── BeaverTails ─────────────────────────────────────────────────────────────

class BeaverTailsLoader(NonReasoningDataset):
    name = "safety/beavertails"
    category = "safety"

    def load(
        self,
        split: DatasetSplit = DatasetSplit.TEST,
        size: str = "330k",
    ) -> List[DatasetItem]:
        """
        Parameters
        ----------
        size : str
            "30k" for the 30k subset (plain jsonl), or
            "330k" for the full 330k set (xz-compressed under round0/330k/).
        """
        bt_dir = self.data_root / "safety" / "beavertails"

        if size == "330k":
            split_fname = {
                DatasetSplit.TRAIN: "train.jsonl.xz",
                DatasetSplit.TEST: "test.jsonl.xz",
            }.get(split, "test.jsonl.xz")
            fpath = bt_dir / "round0" / "330k" / split_fname
            opener = lambda p: __import__("lzma").open(p, "rt")
        else:
            split_fname = {
                DatasetSplit.TRAIN: "train_30k.jsonl",
                DatasetSplit.TEST: "test_30k.jsonl",
            }.get(split, "test_30k.jsonl")
            fpath = bt_dir / split_fname
            opener = lambda p: open(p)

        items: List[DatasetItem] = []
        with opener(fpath) as f:
            for i, line in enumerate(f):
                row = json.loads(line)
                is_safe = row["is_safe"]
                ref = {
                    "is_safe": is_safe,
                    "categories": row["category"],
                    "response": row["response"],
                }
                items.append(
                    DatasetItem(
                        id=f"beavertails/{size}/{split.value}/{i}",
                        dataset=self.name,
                        category=self.category,
                        input_text=row["prompt"],
                        kind=ItemKind.REFERENCE_ANSWER,
                        reference_answer=json.dumps(ref),
                        metadata={
                            "prompt": row["prompt"],
                            "response": row["response"],
                            "is_safe": is_safe,
                            "category": row["category"],
                        },
                    )
                )
        return items


# ── Anthropic Red Team ──────────────────────────────────────────────────────

class AnthropicRedTeamLoader(NonReasoningDataset):
    name = "safety/anthropic_red_team"
    category = "safety"

    def load(self, split: DatasetSplit = DatasetSplit.TEST) -> List[DatasetItem]:
        jsonl_path = (
            self.data_root
            / "safety"
            / "anthropic_red_team"
            / "red_team_attempts.jsonl"
        )
        items: List[DatasetItem] = []
        with open(jsonl_path) as f:
            for i, line in enumerate(f):
                row = json.loads(line)
                items.append(
                    DatasetItem(
                        id=f"red_team/{i}",
                        dataset=self.name,
                        category=self.category,
                        input_text=row["transcript"],
                        kind=ItemKind.REFERENCE_ANSWER,
                        reference_answer=json.dumps({
                            "harmlessness_score": row[
                                "min_harmlessness_score_transcript"
                            ],
                            "rating": row["rating"],
                        }),
                        metadata={
                            "task_description": row.get("task_description", ""),
                            "model_type": row.get("model_type", ""),
                            "num_params": row.get("num_params", ""),
                            "tags": row.get("tags"),
                            "red_team_member_id": row.get(
                                "red_team_member_id"
                            ),
                        },
                    )
                )
        return items


# ── HH-RLHF (preference pairs) ─────────────────────────────────────────────

class HHRLHFLoader(NonReasoningDataset):
    name = "safety/hh_rlhf"
    category = "safety"

    SUBSETS = [
        "harmless-base",
        "helpful-base",
        "helpful-online",
        "helpful-rejection-sampled",
    ]

    def load(
        self,
        split: DatasetSplit = DatasetSplit.TEST,
        subsets: list[str] | None = None,
    ) -> List[DatasetItem]:
        subsets = subsets or self.SUBSETS
        hh_dir = self.data_root / "safety" / "anthropic_hh_rlhf"
        split_fname = {
            DatasetSplit.TRAIN: "train.jsonl.gz",
            DatasetSplit.TEST: "test.jsonl.gz",
        }.get(split, "test.jsonl.gz")

        items: List[DatasetItem] = []
        for subset in subsets:
            gz_path = hh_dir / subset / split_fname
            if not gz_path.exists():
                continue
            with gzip.open(gz_path, "rt") as f:
                for i, line in enumerate(f):
                    row = json.loads(line)
                    items.append(
                        DatasetItem(
                            id=f"hh_rlhf/{subset}/{split.value}/{i}",
                            dataset=self.name,
                            category=self.category,
                            input_text=self._extract_prompt(row["chosen"]),
                            kind=ItemKind.PREFERENCE_PAIR,
                            chosen=row["chosen"],
                            rejected=row["rejected"],
                            metadata={"subset": subset},
                        )
                    )
        return items

    @staticmethod
    def _extract_prompt(conversation: str) -> str:
        """
        Extract just the human turns from the conversation to use as the
        'input' — the full chosen/rejected conversations are in their
        respective fields.
        """
        parts = conversation.split("\n\nAssistant:")
        # keep only the first human turn as the prompt
        if parts:
            first = parts[0]
            if first.startswith("\n\nHuman:"):
                first = first[len("\n\nHuman:") :].strip()
            return first.strip()
        return conversation.strip()
