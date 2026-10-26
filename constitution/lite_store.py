"""
LiteStore: a minimal constitution/memory store that bypasses LightRAG.

Uses a flat JSON index + numpy cosine similarity for retrieval.
Inserts take <10ms instead of LightRAG's 4s/insert.

API is intentionally compatible with ConstitutionStore / MemoryStore so
it can be dropped in as a replacement:
  - initialize() / finalize()
  - add / get / update / retire / delete (ConstitutionEntry)
  - query(question, top_k)
  - list_active() / list_all() / summary()
  - remember / forget / recall (MemoryEntry)
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from constitution.schema import ConstitutionEntry
from constitution.memory_store import MemoryEntry
from constitution.store import ModelEmbedder

import config

logger = logging.getLogger(__name__)


class LiteConstitutionStore:
    """
    Flat-file constitution store.  No LightRAG, no entity extraction,
    no graph storage.  Just a JSON index + numpy vector DB.
    """

    def __init__(self, working_dir: str, embedder: Optional[ModelEmbedder] = None):
        self.working_dir = working_dir
        os.makedirs(working_dir, exist_ok=True)
        self.index_path = os.path.join(working_dir, "constitution_index.json")
        self.vecs_path = os.path.join(working_dir, "constitution_vecs.npy")

        self.embedder = embedder or ModelEmbedder()
        self.index: Dict[str, dict] = {}
        self.vectors: Optional[np.ndarray] = None   # (N, D)
        self.id_order: List[str] = []               # parallel to vectors rows
        self._lock = threading.Lock()

    # ── Lifecycle ──────────────────────────────────────────────────────

    async def initialize(self):
        if os.path.exists(self.index_path):
            with open(self.index_path) as f:
                self.index = json.load(f)
        if os.path.exists(self.vecs_path):
            self.vectors = np.load(self.vecs_path)
            self.id_order = list(self.index.keys())
            if self.vectors.shape[0] != len(self.id_order):
                logger.warning(
                    f"Vector/index length mismatch: {self.vectors.shape[0]} vs {len(self.id_order)}"
                )
                self.vectors = None
                self.id_order = []

    async def finalize(self):
        self._flush()

    def _flush(self):
        with self._lock:
            with open(self.index_path, "w") as f:
                json.dump(self.index, f, indent=2, ensure_ascii=False)
            if self.vectors is not None and len(self.id_order) > 0:
                np.save(self.vecs_path, self.vectors)

    # ── Embedding helper ──────────────────────────────────────────────

    def _embed(self, text: str) -> np.ndarray:
        vec = self.embedder.embed([text])[0]  # (D,)
        # ensure unit-normalized for cosine sim
        n = np.linalg.norm(vec)
        if n > 0:
            vec = vec / n
        return vec.astype(np.float32)

    def _embed_batch(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, config.EMBEDDING_DIM), dtype=np.float32)
        vecs = self.embedder.embed(texts)  # (N, D)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return (vecs / norms).astype(np.float32)

    # ── CRUD ───────────────────────────────────────────────────────────

    async def add(self, entry: ConstitutionEntry) -> str:
        vec = self._embed(entry.principle)
        with self._lock:
            self.index[entry.id] = entry.to_dict()
            self.id_order.append(entry.id)
            if self.vectors is None:
                self.vectors = vec[None, :]
            else:
                self.vectors = np.vstack([self.vectors, vec[None, :]])
        self._flush()
        return entry.id

    async def get(self, entry_id: str) -> Optional[ConstitutionEntry]:
        d = self.index.get(entry_id)
        return ConstitutionEntry.from_dict(d) if d else None

    async def update(self, entry_id: str, updates: dict) -> Optional[ConstitutionEntry]:
        if entry_id not in self.index:
            return None
        old = ConstitutionEntry.from_dict(self.index[entry_id])
        new_data = old.to_dict()
        new_data.update(updates)
        new_data["version"] = old.version + 1
        new_data["supersedes"] = entry_id
        new_entry = ConstitutionEntry.from_dict(new_data)
        new_entry.id = entry_id

        # Re-embed if principle text changed
        if "principle" in updates:
            vec = self._embed(new_entry.principle)
            with self._lock:
                if entry_id in self.id_order:
                    idx = self.id_order.index(entry_id)
                    self.vectors[idx] = vec
                self.index[entry_id] = new_entry.to_dict()
        else:
            with self._lock:
                self.index[entry_id] = new_entry.to_dict()

        self._flush()
        return new_entry

    async def retire(self, entry_id: str) -> bool:
        if entry_id not in self.index:
            return False
        with self._lock:
            self.index[entry_id]["active"] = False
        self._flush()
        return True

    async def delete(self, entry_id: str) -> bool:
        if entry_id not in self.index:
            return False
        with self._lock:
            del self.index[entry_id]
            if entry_id in self.id_order:
                idx = self.id_order.index(entry_id)
                self.id_order.pop(idx)
                if self.vectors is not None:
                    self.vectors = np.delete(self.vectors, idx, axis=0)
        self._flush()
        return True

    # ── Query ──────────────────────────────────────────────────────────

    async def query(self, question: str, top_k: int = 5) -> str:
        """Return top-k relevant active principles as a concatenated string."""
        active_ids = [eid for eid, d in self.index.items() if d.get("active", True)]
        if not active_ids or self.vectors is None:
            return "(No relevant principles)"

        q_vec = self._embed(question)

        # Subset vectors to active only
        active_mask = np.array(
            [self.id_order[i] in active_ids for i in range(len(self.id_order))]
        )
        active_vecs = self.vectors[active_mask]
        active_order = [self.id_order[i] for i in range(len(self.id_order)) if active_mask[i]]

        sims = active_vecs @ q_vec  # cosine sim since unit-normalized
        top = np.argsort(-sims)[:top_k]

        lines = []
        for rank, i in enumerate(top, 1):
            eid = active_order[i]
            d = self.index[eid]
            lines.append(
                f"[{rank}] ({d['id']} conf={d.get('confidence', 0):.2f}) "
                f"{d['principle']}"
            )
        return "\n".join(lines)

    # ── Listing ────────────────────────────────────────────────────────

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
            return "(No active constitution principles)"
        lines = [
            f"[{e.id} v{e.version} conf={e.confidence:.2f}] {e.principle}"
            for e in sorted(active, key=lambda x: x.confidence, reverse=True)
        ]
        return "\n".join(lines)


class LiteMemoryStore:
    """
    Flat-file memory store — mirror of LiteConstitutionStore for memory.
    """

    def __init__(self, working_dir: str, embedder: Optional[ModelEmbedder] = None):
        self.working_dir = working_dir
        os.makedirs(working_dir, exist_ok=True)
        self.index_path = os.path.join(working_dir, "memory_index.json")
        self.vecs_path = os.path.join(working_dir, "memory_vecs.npy")

        self.embedder = embedder or ModelEmbedder()
        self.index: Dict[str, dict] = {}
        self.vectors: Optional[np.ndarray] = None
        self.id_order: List[str] = []
        self._lock = threading.Lock()

    async def initialize(self):
        if os.path.exists(self.index_path):
            with open(self.index_path) as f:
                self.index = json.load(f)
        if os.path.exists(self.vecs_path):
            self.vectors = np.load(self.vecs_path)
            self.id_order = list(self.index.keys())
            if self.vectors.shape[0] != len(self.id_order):
                self.vectors = None
                self.id_order = []

    async def finalize(self):
        self._flush()

    def _flush(self):
        with self._lock:
            with open(self.index_path, "w") as f:
                json.dump(self.index, f, indent=2, ensure_ascii=False)
            if self.vectors is not None and len(self.id_order) > 0:
                np.save(self.vecs_path, self.vectors)

    def _embed(self, text: str) -> np.ndarray:
        vec = self.embedder.embed([text])[0]
        n = np.linalg.norm(vec)
        if n > 0:
            vec = vec / n
        return vec.astype(np.float32)

    async def remember(
        self, content: str, source: str = "", tags: Optional[List[str]] = None
    ) -> str:
        entry = MemoryEntry(content=content, source=source, tags=tags)
        vec = self._embed(content)
        with self._lock:
            self.index[entry.id] = entry.to_dict()
            self.id_order.append(entry.id)
            if self.vectors is None:
                self.vectors = vec[None, :]
            else:
                self.vectors = np.vstack([self.vectors, vec[None, :]])
        self._flush()
        return entry.id

    async def forget(self, memory_id: str) -> bool:
        if memory_id not in self.index:
            return False
        with self._lock:
            del self.index[memory_id]
            if memory_id in self.id_order:
                idx = self.id_order.index(memory_id)
                self.id_order.pop(idx)
                if self.vectors is not None:
                    self.vectors = np.delete(self.vectors, idx, axis=0)
        self._flush()
        return True

    async def recall(self, question: str, top_k: int = 5) -> str:
        if not self.index or self.vectors is None:
            return "(No memories stored)"
        q_vec = self._embed(question)
        sims = self.vectors @ q_vec
        top = np.argsort(-sims)[:top_k]
        lines = []
        for rank, i in enumerate(top, 1):
            mid = self.id_order[i]
            d = self.index[mid]
            tags = ", ".join(d.get("tags", [])) or "general"
            lines.append(f"[{rank}] ({mid}, {tags}) {d['content']}")
        return "\n".join(lines)

    def list_all(self) -> List[MemoryEntry]:
        return [MemoryEntry.from_dict(d) for d in self.index.values()]

    def summary(self) -> str:
        if not self.index:
            return "(No memories stored)"
        lines = []
        for mid, d in self.index.items():
            tags = ", ".join(d.get("tags", [])) or "general"
            lines.append(f"[{mid}] ({tags}) {d['content'][:80]}")
        return "\n".join(lines)
