"""BeaverTails dataset loader."""


from __future__ import annotations


import json


from typing import List


from ..base import (
    DatasetItem,
    DatasetSplit,
    ItemKind,
    NonReasoningDataset,
)


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
