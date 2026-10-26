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
from .loaders import MATHLoader, MMLULoader, BeaverTailsLoader, BBQLoader

REGISTRY: Dict[str, Type[BaseDatasetLoader]] = {
    "math/hendrycks": MATHLoader,
    "reasoning/mmlu": MMLULoader,
    "safety/beavertails": BeaverTailsLoader,
    "value/bbq": BBQLoader,
}
GROUPS = {
    "math": ["math/hendrycks"],
    "reasoning": ["reasoning/mmlu"],
    "safety": ["safety/beavertails"],
    "value": ["value/bbq"],
}
GROUPS["all"] = list(REGISTRY)


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
