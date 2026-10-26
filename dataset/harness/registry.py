"""
Central registry: maps dataset name → loader class.

Usage::

    from harness import load_dataset

    items = load_dataset("math/hendrycks", data_root="/path/to/dataset")
    items = load_dataset("all", data_root="/path/to/dataset")  # load everything
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Type, Union

from .base import BaseDatasetLoader, DatasetItem, DatasetSplit
from .loaders import (
    AIME1983Loader,
    AIME2025Loader,
    AnthropicRedTeamLoader,
    BBQLoader,
    BeaverTailsLoader,
    BigCodeBenchLoader,
    BOLDLoader,
    GPQALoader,
    HendrycksEthicsLoader,
    HHRLHFLoader,
    LLMEthicsBenchLoader,
    MATHLoader,
    MMLULoader,
    MoReBenchLoader,
    OpinionQALoader,
    SWEBenchLoader,
    SWEBenchVerifiedLoader,
    UNBenchLoader,
)

# name → loader class
REGISTRY: Dict[str, Type[BaseDatasetLoader]] = {
    # ── reasoning (ground truth) ──
    "math/hendrycks": MATHLoader,
    "math/aime_1983_2024": AIME1983Loader,
    "math/aime_2025": AIME2025Loader,
    "code/bigcodebench": BigCodeBenchLoader,
    "code/swe_bench": SWEBenchLoader,
    "code/swe_bench_verified": SWEBenchVerifiedLoader,
    "reasoning/mmlu": MMLULoader,
    "reasoning/gpqa": GPQALoader,
    # ── safety (reference answers) ──
    "safety/beavertails": BeaverTailsLoader,
    "safety/anthropic_red_team": AnthropicRedTeamLoader,
    "safety/hh_rlhf": HHRLHFLoader,
    # ── value (reference answers) ──
    "value/bbq": BBQLoader,
    "value/bold": BOLDLoader,
    "value/opinionqa": OpinionQALoader,
    "value/unbench": UNBenchLoader,
    "value/morebench": MoReBenchLoader,
    "value/hendrycks_ethics": HendrycksEthicsLoader,
    "value/llm_ethics_benchmark": LLMEthicsBenchLoader,
}

# Convenience groups
GROUPS: Dict[str, List[str]] = {
    "math": ["math/hendrycks", "math/aime_1983_2024", "math/aime_2025"],
    "code": ["code/bigcodebench", "code/swe_bench", "code/swe_bench_verified"],
    "reasoning": ["reasoning/mmlu", "reasoning/gpqa"],
    "safety": ["safety/beavertails", "safety/anthropic_red_team", "safety/hh_rlhf"],
    "value": [
        "value/bbq",
        "value/bold",
        "value/opinionqa",
        "value/unbench",
        "value/morebench",
        "value/hendrycks_ethics",
        "value/llm_ethics_benchmark",
    ],
}
GROUPS["all"] = [name for names in GROUPS.values() for name in names]


def load_dataset(
    name: str,
    data_root: Union[str, Path] = ".",
    split: DatasetSplit = DatasetSplit.TEST,
    **kwargs,
) -> List[DatasetItem]:
    """
    Load a dataset (or group) by name.

    Parameters
    ----------
    name : str
        A dataset name (e.g. "math/hendrycks") or a group name
        (e.g. "math", "safety", "all").
    data_root : str | Path
        Path to the top-level ``dataset/`` directory.
    split : DatasetSplit
        Which split to load.
    **kwargs
        Forwarded to the loader's ``load()`` method.
    """
    data_root = Path(data_root)

    if name in GROUPS:
        items: List[DatasetItem] = []
        for ds_name in GROUPS[name]:
            items.extend(load_dataset(ds_name, data_root, split, **kwargs))
        return items

    if name not in REGISTRY:
        raise KeyError(
            f"Unknown dataset {name!r}. "
            f"Available: {sorted(REGISTRY)} + groups {sorted(GROUPS)}"
        )

    loader_cls = REGISTRY[name]
    loader = loader_cls(data_root)
    return loader.load(split=split, **kwargs)
