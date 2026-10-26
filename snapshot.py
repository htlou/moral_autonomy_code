"""
Snapshot manager: save and load constitution + memory state.

Each snapshot is identified by a ``run_id`` and stored under
``snapshots/<run_id>/``.  It contains:

    constitution_index.json   — the full constitution index
    memory_index.json         — the full memory index
    rag_storage/              — LightRAG files for constitution
    memory_storage/           — LightRAG files for memory
    manifest.json             — run metadata (spec, timestamp, stats)

Snapshots can be loaded back into a live Orchestrator to resume
from a previous state.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_SNAPSHOT_DIR = os.path.join(os.path.dirname(__file__), "snapshots")


class SnapshotManager:
    """Manages saving and loading of constitution/memory snapshots."""

    def __init__(self, base_dir: str = DEFAULT_SNAPSHOT_DIR):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    # ── Save ────────────────────────────────────────────────────────────

    def save(
        self,
        run_id: str,
        constitution_working_dir: str,
        memory_working_dir: str,
        manifest_extra: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Snapshot the current constitution + memory state.

        Parameters
        ----------
        run_id : str
            Unique identifier for this snapshot.
        constitution_working_dir : str
            Path to the live ConstitutionStore working directory.
        memory_working_dir : str
            Path to the live MemoryStore working directory.
        manifest_extra : dict, optional
            Additional metadata to store in manifest.json
            (e.g. dataset spec, stats, notes).

        Returns
        -------
        Path to the snapshot directory.
        """
        snap_dir = self.base_dir / run_id
        if snap_dir.exists():
            logger.warning(f"Snapshot {run_id} already exists, overwriting.")
            shutil.rmtree(snap_dir)
        snap_dir.mkdir(parents=True)

        # Copy RAG storage directories
        const_src = Path(constitution_working_dir)
        mem_src = Path(memory_working_dir)

        if const_src.exists():
            shutil.copytree(const_src, snap_dir / "rag_storage")
        if mem_src.exists():
            shutil.copytree(mem_src, snap_dir / "memory_storage")

        # Write manifest
        manifest = {
            "run_id": run_id,
            "created_at": datetime.now().isoformat(),
            "constitution_entries": self._count_entries(
                const_src / "constitution_index.json"
            ),
            "memory_entries": self._count_entries(
                mem_src / "memory_index.json"
            ),
        }
        if manifest_extra:
            manifest.update(manifest_extra)

        with open(snap_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False, default=str)

        logger.info(
            f"Snapshot saved: {run_id} "
            f"({manifest['constitution_entries']} constitution, "
            f"{manifest['memory_entries']} memory)"
        )
        return snap_dir

    # ── Load ────────────────────────────────────────────────────────────

    def load(
        self,
        run_id: str,
        constitution_working_dir: str,
        memory_working_dir: str,
    ) -> Dict[str, Any]:
        """
        Restore a snapshot into live working directories.

        This **overwrites** the contents of the target directories.

        Parameters
        ----------
        run_id : str
            Snapshot to load.
        constitution_working_dir : str
            Destination for constitution data.
        memory_working_dir : str
            Destination for memory data.

        Returns
        -------
        The manifest dict.
        """
        snap_dir = self.base_dir / run_id
        if not snap_dir.exists():
            raise FileNotFoundError(f"Snapshot {run_id} not found at {snap_dir}")

        # Restore constitution RAG storage
        const_snap = snap_dir / "rag_storage"
        if const_snap.exists():
            const_dst = Path(constitution_working_dir)
            if const_dst.exists():
                shutil.rmtree(const_dst)
            shutil.copytree(const_snap, const_dst)

        # Restore memory storage
        mem_snap = snap_dir / "memory_storage"
        if mem_snap.exists():
            mem_dst = Path(memory_working_dir)
            if mem_dst.exists():
                shutil.rmtree(mem_dst)
            shutil.copytree(mem_snap, mem_dst)

        manifest = self.get_manifest(run_id)
        logger.info(f"Snapshot loaded: {run_id}")
        return manifest

    # ── List / Inspect ──────────────────────────────────────────────────

    def list_snapshots(self) -> List[Dict[str, Any]]:
        """Return manifests of all saved snapshots, sorted by creation time."""
        manifests = []
        for d in sorted(self.base_dir.iterdir()):
            if d.is_dir() and (d / "manifest.json").exists():
                manifests.append(self.get_manifest(d.name))
        return manifests

    def get_manifest(self, run_id: str) -> Dict[str, Any]:
        """Read a snapshot's manifest."""
        manifest_path = self.base_dir / run_id / "manifest.json"
        with open(manifest_path) as f:
            return json.load(f)

    def delete(self, run_id: str) -> bool:
        """Delete a snapshot."""
        snap_dir = self.base_dir / run_id
        if snap_dir.exists():
            shutil.rmtree(snap_dir)
            logger.info(f"Snapshot deleted: {run_id}")
            return True
        return False

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _count_entries(index_path: Path) -> int:
        if index_path.exists():
            with open(index_path) as f:
                return len(json.load(f))
        return 0
