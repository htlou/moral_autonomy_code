"""
Dataset selector: flexible filtering and sampling for batch runs.

Usage::

    spec = [
        {"dataset": "math/hendrycks", "n": 100},
        {"dataset": "safety/beavertails", "fraction": 0.5,
         "filter": {"is_safe": False}},
        {"dataset": "reasoning/mmlu", "n": 200, "subjects": ["abstract_algebra", "anatomy"]},
        {"dataset": "value/bbq", "fraction": 0.1, "filter": {"category": "Age"}},
    ]
    items = select_items(spec, data_root="/path/to/dataset")
"""

from __future__ import annotations

import random
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from .base import DatasetItem, DatasetSplit
from .registry import load_dataset


def _match_filter(item: DatasetItem, filt: Dict[str, Any]) -> bool:
    """
    Check if an item matches a filter dict.

    Filter keys are checked against ``item.metadata``.  Supports:
      - Exact match:      {"is_safe": False}
      - Substring match:  {"category": "Age"}  (for string values)
      - List membership:  {"subtask": ["justice", "virtue"]}
      - Callable:         {"custom": lambda item: item.metadata["year"] > 2000}
    """
    for key, value in filt.items():
        if key == "custom" and callable(value):
            if not value(item):
                return False
            continue

        meta_val = item.metadata.get(key)

        if isinstance(value, list):
            if meta_val not in value:
                return False
        elif isinstance(value, bool):
            if meta_val != value:
                return False
        elif isinstance(meta_val, str) and isinstance(value, str):
            if value != meta_val:
                return False
        else:
            if meta_val != value:
                return False

    return True


def _sample(
    items: List[DatasetItem],
    n: Optional[int] = None,
    fraction: Optional[float] = None,
    seed: int = 42,
) -> List[DatasetItem]:
    """Sub-sample items by count or fraction."""
    if n is None and fraction is None:
        return items

    rng = random.Random(seed)

    if fraction is not None:
        n_target = max(1, int(len(items) * fraction))
    else:
        n_target = n

    if n_target >= len(items):
        return items

    return rng.sample(items, n_target)


def select_from_spec(
    spec_entry: Dict[str, Any],
    data_root: str,
    split: DatasetSplit = DatasetSplit.TEST,
    seed: int = 42,
) -> List[DatasetItem]:
    """
    Load and filter a single dataset spec entry.

    Parameters
    ----------
    spec_entry : dict
        Must contain ``"dataset"`` (str).  Optional keys:
        - ``"n"``: exact number of items to sample
        - ``"fraction"``: fraction of items to sample (0-1)
        - ``"filter"``: dict of metadata filters
        - ``"split"``: override the default split
        - any other key is forwarded to the loader's ``load()``
    data_root : str
        Path to the dataset directory.
    split : DatasetSplit
        Default split (overridden by spec_entry["split"]).
    seed : int
        Random seed for sampling.
    """
    entry = dict(spec_entry)  # copy so we can pop
    ds_name = entry.pop("dataset")
    n = entry.pop("n", None)
    fraction = entry.pop("fraction", None)
    filt = entry.pop("filter", None)
    entry_split = entry.pop("split", None)

    use_split = DatasetSplit(entry_split) if entry_split else split

    # remaining keys are loader kwargs (e.g. subjects=, subtopics=, variant=)
    items = load_dataset(ds_name, data_root=data_root, split=use_split, **entry)
    if not items:
        raise FileNotFoundError(
            f"No items loaded for {ds_name!r} under {data_root}. "
            "Prepare datasets with scripts/prepare_data.py; see docs/REPRODUCING.md."
        )

    # apply filter
    if filt:
        items = [it for it in items if _match_filter(it, filt)]

    # sample
    items = _sample(items, n=n, fraction=fraction, seed=seed)

    return items


def select_items(
    spec: List[Dict[str, Any]],
    data_root: str,
    split: DatasetSplit = DatasetSplit.TEST,
    seed: int = 42,
    shuffle: bool = True,
    exclude_ids: Optional[set] = None,
) -> List[DatasetItem]:
    """
    Build a combined item list from multiple dataset specs.

    Parameters
    ----------
    spec : list of dict
        Each entry describes one dataset selection (see ``select_from_spec``).
    data_root : str
        Path to the dataset directory.
    split : DatasetSplit
        Default split for all datasets.
    seed : int
        Random seed.
    shuffle : bool
        Whether to shuffle the final combined list.
    exclude_ids : set, optional
        Item IDs to exclude *before* sampling (e.g. training IDs to avoid
        train/eval overlap). When set, each spec entry loads all matching
        items, removes excluded IDs, then samples ``n`` from the remainder.

    Returns
    -------
    List of DatasetItem, ready for batch processing.
    """
    all_items: List[DatasetItem] = []
    for entry in spec:
        if exclude_ids:
            # Load all matching items first, exclude, then sample
            entry_copy = dict(entry)
            n = entry_copy.pop("n", None)
            fraction = entry_copy.pop("fraction", None)
            # select_from_spec without n/fraction → returns all matching items
            items = select_from_spec(entry_copy, data_root, split, seed)
            items = [it for it in items if it.id not in exclude_ids]
            items = _sample(items, n=n, fraction=fraction, seed=seed)
        else:
            items = select_from_spec(entry, data_root, split, seed)
        all_items.extend(items)

    if shuffle:
        random.Random(seed).shuffle(all_items)

    return all_items
