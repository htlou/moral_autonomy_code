"""In-memory constitution store for the prototype.

Compatible-shaped subset of the project's `LiteConstitutionStore` — same
ConstitutionEntry schema, same add/update/retire/list_active API — but
with no embedding and no RAG retrieval. The constitution stays small
(<= MAX_CONSTITUTION_SIZE), so we just dump all active principles each
round.
"""

from __future__ import annotations

import copy
import json
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import uuid

# Reuse the existing schema if available; fall back to a local copy.
try:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from constitution.schema import ConstitutionEntry  # type: ignore
except Exception:
    @dataclass
    class ConstitutionEntry:  # type: ignore
        principle: str
        confidence: float = 0.5
        evidence: list = field(default_factory=list)
        version: int = 1
        id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
        created_at: str = field(default_factory=lambda: datetime.now().isoformat())
        supersedes: Optional[str] = None
        active: bool = True

        def to_dict(self):
            return asdict(self)

        @classmethod
        def from_dict(cls, d):
            return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class ListConstitutionStore:
    """Minimal list-backed store. All entries kept in memory; no embeddings."""

    def __init__(self):
        self.index: dict = {}      # id -> dict
        self.id_order: List[str] = []
        self.snapshots: List[dict] = []  # list of {round, entries}

    def add(self, entry: ConstitutionEntry) -> str:
        self.index[entry.id] = entry.to_dict()
        self.id_order.append(entry.id)
        return entry.id

    def get(self, entry_id: str) -> Optional[ConstitutionEntry]:
        d = self.index.get(entry_id)
        return ConstitutionEntry.from_dict(d) if d else None

    def update(self, entry_id: str, updates: dict) -> Optional[ConstitutionEntry]:
        if entry_id not in self.index:
            return None
        old = ConstitutionEntry.from_dict(self.index[entry_id])
        new_data = old.to_dict()
        new_data.update(updates)
        new_data["version"] = old.version + 1
        new_data["supersedes"] = entry_id
        new_entry = ConstitutionEntry.from_dict(new_data)
        new_entry.id = entry_id  # keep stable id; track lineage via version+supersedes
        self.index[entry_id] = new_entry.to_dict()
        return new_entry

    def retire(self, entry_id: str) -> bool:
        if entry_id not in self.index:
            return False
        self.index[entry_id]["active"] = False
        return True

    def list_active(self) -> List[ConstitutionEntry]:
        return [
            ConstitutionEntry.from_dict(d)
            for d in self.index.values()
            if d.get("active", True)
        ]

    def list_all(self) -> List[ConstitutionEntry]:
        return [ConstitutionEntry.from_dict(d) for d in self.index.values()]

    def summary(self) -> str:
        active = self.list_active()
        if not active:
            return "(no active principles)"
        return "\n".join(
            f"[{e.id} v{e.version} conf={e.confidence:.2f}] {e.principle}"
            for e in sorted(active, key=lambda x: x.confidence, reverse=True)
        )

    # ── Snapshots for evolution analysis ──────────────────────────────

    def snapshot(self, round_idx: int) -> None:
        self.snapshots.append({
            "round": round_idx,
            "entries": copy.deepcopy(list(self.index.values())),
        })

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({
            "final_index": self.index,
            "id_order": self.id_order,
            "snapshots": self.snapshots,
        }, indent=2, ensure_ascii=False))


def seed_fixed(store: ListConstitutionStore, principles: list[str], confidence: float = 0.8) -> None:
    """Pre-load a fixed constitution from a list of strings."""
    for p in principles:
        store.add(ConstitutionEntry(principle=p, confidence=confidence,
                                    evidence=["seed"]))
