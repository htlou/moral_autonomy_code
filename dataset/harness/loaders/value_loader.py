"""BBQ dataset loader."""


from __future__ import annotations


import json


from typing import List, Optional


from ..base import (
    DatasetItem,
    DatasetSplit,
    ItemKind,
    NonReasoningDataset,
)


class BBQLoader(NonReasoningDataset):
    name = "value/bbq"
    category = "value"

    CATEGORIES = [
        "Age",
        "Disability_status",
        "Gender_identity",
        "Nationality",
        "Physical_appearance",
        "Race_ethnicity",
        "Race_x_gender",
        "Race_x_SES",
        "Religion",
        "SES",
        "Sexual_orientation",
    ]

    def load(
        self,
        split: DatasetSplit = DatasetSplit.TEST,
        categories: Optional[List[str]] = None,
    ) -> List[DatasetItem]:
        categories = categories or self.CATEGORIES
        bbq_dir = self.data_root / "value" / "bbq_github" / "data"
        items: List[DatasetItem] = []

        for cat in categories:
            jsonl_path = bbq_dir / f"{cat}.jsonl"
            if not jsonl_path.exists():
                continue
            with open(jsonl_path) as f:
                for line in f:
                    row = json.loads(line)
                    eid = row["example_id"]
                    choices = [row["ans0"], row["ans1"], row["ans2"]]
                    label = row["label"]
                    items.append(
                        DatasetItem(
                            id=f"bbq/{cat}/{eid}",
                            dataset=self.name,
                            category=self.category,
                            input_text=self._format_prompt(
                                row["context"],
                                row["question"],
                                choices,
                            ),
                            kind=ItemKind.REFERENCE_ANSWER,
                            reference_answer=json.dumps({
                                "label": label,
                                "correct_answer": choices[label],
                            }),
                            metadata={
                                "category": cat,
                                "question_polarity": row.get(
                                    "question_polarity", ""
                                ),
                                "context_condition": row.get(
                                    "context_condition", ""
                                ),
                                "answer_info": row.get("answer_info", {}),
                                "choices": choices,
                            },
                        )
                    )
        return items

    @staticmethod
    def _format_prompt(context: str, question: str, choices: list) -> str:
        opts = "\n".join(
            f"  {chr(65 + i)}. {c}" for i, c in enumerate(choices)
        )
        return (
            f"Context: {context}\n\n"
            f"Question: {question}\n\n"
            f"Options:\n{opts}\n\n"
            "Choose the most appropriate answer (A, B, or C)."
        )
