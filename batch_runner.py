"""
Batch runner: feeds dataset items to the LM for constitution/RAG evolution.

For every item, the LM sees:
  1. The dataset prompt (the actual question/scenario)
  2. A **reference answer** and optionally a **reference judge** output
     — both framed neutrally (not as "correct" or "ground truth")
  3. The LM then reflects and may modify its constitution or memory

The runner never tells the LM whether the reference answer is correct.
The LM decides autonomously what to learn from the reference signal.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from constitution.store import ConstitutionStore
from constitution.memory_store import MemoryStore
from pipeline.orchestrator import Orchestrator
from snapshot import SnapshotManager
from dataset.harness.base import DatasetItem, ItemKind, ReasoningDataset
from dataset.harness.selector import select_items
from dataset.harness.registry import REGISTRY

import config

logger = logging.getLogger(__name__)


# ── Neutral formatting ──────────────────────────────────────────────────────

def _format_reference_signal(item: DatasetItem) -> str:
    """
    Build the reference signal shown to the LM.

    Framing is strictly neutral — the LM sees "reference answer" and
    "reference judge", never "correct answer" or "ground truth".
    """
    parts = []

    if item.kind == ItemKind.GROUND_TRUTH:
        # reasoning dataset — show reference answer + optional judge
        ref = item.ground_truth or ""
        parts.append(f"[Reference Answer]\n{ref}")

        if item.ground_truth_detail and item.ground_truth_detail != ref:
            parts.append(
                f"[Reference Reasoning]\n{item.ground_truth_detail}"
            )

        # Run the reference judge if available
        judge_output = _run_reference_judge(item)
        if judge_output:
            parts.append(f"[Reference Judge]\n{judge_output}")

    elif item.kind == ItemKind.REFERENCE_ANSWER:
        ref = item.reference_answer or ""
        try:
            ref_obj = __import__('json').loads(ref)
            if isinstance(ref_obj, dict) and "response" in ref_obj:
                parts.append(f"[Reference]\n{ref_obj['response']}")
            else:
                parts.append(f"[Reference Answer]\n{ref}")
        except (ValueError, TypeError):
            parts.append(f"[Reference Answer]\n{ref}")

    elif item.kind == ItemKind.PREFERENCE_PAIR:
        parts.append(f"[Reference Preferred Response]\n{item.chosen}")
        parts.append(f"[Reference Non-Preferred Response]\n{item.rejected}")

    elif item.kind == ItemKind.PROMPT_ONLY:
        parts.append(
            "[Note: This is a prompt-only item. No reference answer is "
            "provided. Evaluate the prompt and its potential implications.]"
        )

    # Add dataset context
    parts.append(
        f"[Dataset: {item.dataset} | Category: {item.category} | "
        f"Label Kind: {item.kind.value}]"
    )

    return "\n\n".join(parts)


def _run_reference_judge(item: DatasetItem) -> Optional[str]:
    """
    For reasoning datasets, produce a judge explanation based on the
    reference answer.  This does NOT evaluate the LM's output — it
    simply describes what the reference judge would check.
    """
    ds_name = item.dataset
    loader_cls = REGISTRY.get(ds_name)
    if loader_cls is None or not issubclass(loader_cls, ReasoningDataset):
        return None

    if "math" in ds_name or "aime" in ds_name:
        return (
            f"The reference judge extracts the final answer (from \\boxed{{}} "
            f"if present) and compares it to the reference: {item.ground_truth}"
        )
    elif "mmlu" in ds_name or "gpqa" in ds_name:
        letter = item.metadata.get("correct_letter", "?")
        return (
            f"The reference judge extracts the chosen letter (A-D) and "
            f"compares it to the reference letter: {letter}"
        )
    elif "bigcodebench" in ds_name:
        return (
            "The reference judge composes the solution with the provided "
            "unit tests and runs them in a subprocess. All tests must pass."
        )
    elif "swe_bench" in ds_name:
        return (
            "The reference judge applies the predicted patch to the "
            "repository and runs the FAIL_TO_PASS test suite. All "
            "failing tests must become passing."
        )
    return None


# ── Batch runner ────────────────────────────────────────────────────────────

class BatchRunner:
    """
    Feeds dataset items through the Orchestrator's generate → reflect → delta
    loop, exposing reference answers neutrally.

    Parameters
    ----------
    run_id : str
        Unique run identifier.  Used for logging and snapshot naming.
    rag_working_dir : str
        Working directory for constitution RAG.
    memory_working_dir : str
        Working directory for memory RAG.
    llm_base_url : str
        vLLM (or any OpenAI-compatible) endpoint.
    snapshot_dir : str
        Where to save snapshots.
    save_every : int
        Auto-save a snapshot every N items (0 = only at end).
    """

    def __init__(
        self,
        run_id: str = None,
        rag_working_dir: str = None,
        memory_working_dir: str = None,
        llm_base_url: str = None,
        snapshot_dir: str = None,
        save_every: int = 0,
    ):
        self.run_id = run_id or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.rag_working_dir = rag_working_dir or os.path.join(
            config.RAG_WORKING_DIR + f"_{self.run_id}"
        )
        self.memory_working_dir = memory_working_dir or os.path.join(
            config.MEMORY_WORKING_DIR + f"_{self.run_id}"
        )
        self.llm_base_url = llm_base_url or config.VLLM_BASE_URL
        self.snapshot_mgr = SnapshotManager(
            snapshot_dir or os.path.join(os.path.dirname(__file__), "snapshots")
        )
        self.save_every = save_every

        self.store: Optional[ConstitutionStore] = None
        self.memory: Optional[MemoryStore] = None
        self.orchestrator: Optional[Orchestrator] = None

        # Run log
        self.results: List[Dict[str, Any]] = []
        self.spec_used: Optional[List[Dict]] = None

    # ── Lifecycle ───────────────────────────────────────────────────────

    async def initialize(self, snapshot_id: str = None):
        """
        Set up stores and orchestrator.

        Parameters
        ----------
        snapshot_id : str, optional
            If provided, load this snapshot as starting state.
        """
        if snapshot_id:
            self.snapshot_mgr.load(
                snapshot_id,
                self.rag_working_dir,
                self.memory_working_dir,
            )
            logger.info(f"Loaded snapshot {snapshot_id} as starting state.")

        self.store = ConstitutionStore(
            working_dir=self.rag_working_dir,
            llm_base_url=self.llm_base_url,
        )
        await self.store.initialize()

        self.memory = MemoryStore(
            working_dir=self.memory_working_dir,
            embedder=self.store.embedder,
            llm_base_url=self.llm_base_url,
        )
        await self.memory.initialize()

        self.orchestrator = Orchestrator(
            store=self.store,
            memory=self.memory,
            llm_base_url=self.llm_base_url,
        )

    async def finalize(self):
        """Save final snapshot and tear down."""
        if self.store:
            await self.store.finalize()
        if self.memory:
            await self.memory.finalize()

    # ── Core loop ───────────────────────────────────────────────────────

    async def run(
        self,
        items: List[DatasetItem],
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Process a list of DatasetItems through the constitution evolution loop.

        For each item:
          1. Format the prompt (original dataset input)
          2. Build the reference signal (neutral framing)
          3. Feed both to the Orchestrator as query + feedback
          4. The LM generates a response, reflects, and applies deltas

        Returns
        -------
        Summary dict with stats and the full run log.
        """
        if not self.orchestrator:
            raise RuntimeError("Call initialize() before run().")

        total = len(items)
        for i, item in enumerate(items):
            step_id = f"{self.run_id}/step_{i:05d}"

            # The "query" is the dataset prompt
            query = item.input_text

            # The "feedback" carries the reference signal (neutrally framed)
            feedback = _format_reference_signal(item)

            if verbose:
                print(f"\n{'='*70}")
                print(f"[{i+1}/{total}] {item.dataset} | {item.id}")
                print(f"  Kind: {item.kind.value}")

            try:
                result = await self.orchestrator.step(
                    query=query, feedback=feedback
                )

                record = {
                    "step": i,
                    "step_id": step_id,
                    "item_id": item.id,
                    "dataset": item.dataset,
                    "category": item.category,
                    "kind": item.kind.value,
                    "response": result.get("response", ""),
                    "reflection": result.get("reflection", {}),
                    "applied": result.get("applied", []),
                    "active_principles": len(self.store.list_active()),
                    "total_memories": len(self.memory.index),
                }
                self.results.append(record)

                if verbose:
                    print(f"  Response: {record['response'][:100]}...")
                    reason = record["reflection"].get("reasoning", "")
                    print(f"  Reflection: {reason[:100]}...")
                    print(f"  Applied: {record['applied']}")
                    print(
                        f"  Constitution: {record['active_principles']} active | "
                        f"Memory: {record['total_memories']}"
                    )

            except Exception as e:
                logger.error(f"Step {i} failed: {e}")
                self.results.append({
                    "step": i,
                    "step_id": step_id,
                    "item_id": item.id,
                    "dataset": item.dataset,
                    "error": str(e),
                })
                if verbose:
                    print(f"  ERROR: {e}")

            # Auto-save checkpoint
            if self.save_every > 0 and (i + 1) % self.save_every == 0:
                ckpt_id = f"{self.run_id}_ckpt_{i+1:05d}"
                self._save_snapshot(ckpt_id, items[:i+1])
                if verbose:
                    print(f"  [Checkpoint saved: {ckpt_id}]")

        # Final save
        self._save_snapshot(self.run_id, items)
        if verbose:
            print(f"\n{'='*70}")
            print(f"Run complete. Snapshot: {self.run_id}")
            print(f"  Constitution: {len(self.store.list_active())} active principles")
            print(f"  Memory: {len(self.memory.index)} entries")

        return self._build_summary(items)

    # ── Convenience: run from spec ──────────────────────────────────────

    async def run_from_spec(
        self,
        spec: List[Dict[str, Any]],
        data_root: str,
        verbose: bool = True,
        snapshot_id: str = None,
    ) -> Dict[str, Any]:
        """
        End-to-end: select items from spec → initialize → run → finalize.

        Parameters
        ----------
        spec : list of dict
            Dataset selection spec (see selector.select_items).
        data_root : str
            Path to the dataset directory.
        verbose : bool
            Print progress.
        snapshot_id : str, optional
            Load a previous snapshot as starting state.
        """
        self.spec_used = spec

        # Select items
        items = select_items(spec, data_root=data_root)
        if verbose:
            print(f"Selected {len(items)} items from {len(spec)} dataset(s)")
            for s in spec:
                print(f"  - {s}")

        # Initialize
        await self.initialize(snapshot_id=snapshot_id)

        try:
            summary = await self.run(items, verbose=verbose)
        finally:
            await self.finalize()

        return summary

    # ── Internal ────────────────────────────────────────────────────────

    def _save_snapshot(
        self, snap_id: str, items: List[DatasetItem]
    ) -> Path:
        """Save constitution + memory + manifest."""
        manifest_extra = {
            "spec": self.spec_used,
            "total_items_processed": len(self.results),
            "datasets_seen": list(
                set(r.get("dataset", "") for r in self.results)
            ),
            "final_constitution_size": len(self.store.list_active()),
            "final_memory_size": len(self.memory.index),
            "errors": sum(1 for r in self.results if "error" in r),
        }
        return self.snapshot_mgr.save(
            run_id=snap_id,
            constitution_working_dir=self.rag_working_dir,
            memory_working_dir=self.memory_working_dir,
            manifest_extra=manifest_extra,
        )

    def _build_summary(self, items: List[DatasetItem]) -> Dict[str, Any]:
        """Aggregate run stats."""
        by_dataset: Dict[str, int] = {}
        by_kind: Dict[str, int] = {}
        by_category: Dict[str, int] = {}
        total_deltas = 0
        errors = 0

        for r in self.results:
            ds = r.get("dataset", "unknown")
            by_dataset[ds] = by_dataset.get(ds, 0) + 1
            kind = r.get("kind", "unknown")
            by_kind[kind] = by_kind.get(kind, 0) + 1
            cat = r.get("category", "unknown")
            by_category[cat] = by_category.get(cat, 0) + 1
            total_deltas += len(r.get("applied", []))
            if "error" in r:
                errors += 1

        return {
            "run_id": self.run_id,
            "total_items": len(items),
            "total_processed": len(self.results),
            "errors": errors,
            "total_deltas_applied": total_deltas,
            "final_constitution_size": len(self.store.list_active()),
            "final_memory_size": len(self.memory.index),
            "by_dataset": by_dataset,
            "by_kind": by_kind,
            "by_category": by_category,
            "results": self.results,
        }

    # ── Save run log ────────────────────────────────────────────────────

    def save_log(self, path: str = None) -> str:
        """Save the full run log to a JSON file."""
        path = path or os.path.join(
            os.path.dirname(__file__), "snapshots", f"{self.run_id}_log.json"
        )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False, default=str)
        return path
