"""
Parallel batch runner: asyncio-based concurrent constitution evolution.

Parallelism strategy:
  - Inter-condition: all 6 conditions run concurrently (independent stores)
  - Intra-condition: items processed in mini-batches of K
    1. Snapshot constitution/memory state (read)
    2. Fire K generate calls concurrently
    3. Fire K reflect calls concurrently
    4. Collect all deltas, apply sequentially (write)

This avoids sync issues: all K items in a mini-batch see the same
constitution snapshot; writes happen in a single serial pass after
the batch completes.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import openai
from json_repair import repair_json

from constitution.schema import ConstitutionEntry
from constitution.store import ConstitutionStore
from constitution.memory_store import MemoryStore
from constitution.prompts import REFLECTION_SYSTEM, REFLECTION_USER, GENERATE_SYSTEM
from snapshot import SnapshotManager
from dataset.harness.base import DatasetItem, ItemKind, ReasoningDataset
from dataset.harness.selector import select_items
from dataset.harness.registry import REGISTRY

import config

logger = logging.getLogger(__name__)


# ── Neutral reference signal (same as batch_runner.py) ──────────────────────

def _format_reference_signal(item: DatasetItem) -> str:
    parts = []

    if item.kind == ItemKind.GROUND_TRUTH:
        ref = item.ground_truth or ""
        parts.append(f"[Reference Answer]\n{ref}")
        if item.ground_truth_detail and item.ground_truth_detail != ref:
            parts.append(f"[Reference Reasoning]\n{item.ground_truth_detail}")
        judge_output = _run_reference_judge(item)
        if judge_output:
            parts.append(f"[Reference Judge]\n{judge_output}")
    elif item.kind == ItemKind.REFERENCE_ANSWER:
        ref_str = item.reference_answer or ''
        try:
            ref_obj = __import__('json').loads(ref_str)
            if isinstance(ref_obj, dict) and "response" in ref_obj:
                parts.append(f"[Reference]\n{ref_obj['response']}")
            else:
                parts.append(f"[Reference Answer]\n{ref_str}")
        except (ValueError, TypeError):
            parts.append(f"[Reference Answer]\n{ref_str}")
    elif item.kind == ItemKind.PREFERENCE_PAIR:
        parts.append(f"[Reference Preferred Response]\n{item.chosen}")
        parts.append(f"[Reference Non-Preferred Response]\n{item.rejected}")
    elif item.kind == ItemKind.PROMPT_ONLY:
        parts.append(
            "[Note: This is a prompt-only item. No reference answer is "
            "provided. Evaluate the prompt and its potential implications.]"
        )

    parts.append(
        f"[Dataset: {item.dataset} | Category: {item.category} | "
        f"Label Kind: {item.kind.value}]"
    )
    return "\n\n".join(parts)


def _run_reference_judge(item: DatasetItem) -> Optional[str]:
    ds_name = item.dataset
    loader_cls = REGISTRY.get(ds_name)
    if loader_cls is None or not issubclass(loader_cls, ReasoningDataset):
        return None
    if "math" in ds_name or "aime" in ds_name:
        return f"The reference judge extracts the final answer (from \\boxed{{}} if present) and compares it to the reference: {item.ground_truth}"
    elif "mmlu" in ds_name or "gpqa" in ds_name:
        return f"The reference judge extracts the chosen letter (A-D) and compares it to the reference letter: {item.metadata.get('correct_letter', '?')}"
    elif "bigcodebench" in ds_name:
        return "The reference judge composes the solution with the provided unit tests and runs them in a subprocess. All tests must pass."
    elif "swe_bench" in ds_name:
        return "The reference judge applies the predicted patch to the repository and runs the FAIL_TO_PASS test suite."
    return None


# ── Parallel worker for a single condition ──────────────────────────────────

class ParallelConditionRunner:
    """
    Processes one condition's items with mini-batch parallelism.

    Within each mini-batch of K items:
      1. Read constitution + memory snapshot (shared across batch)
      2. Concurrently generate K responses
      3. Concurrently reflect on K (query, response, feedback) tuples
      4. Sequentially apply all collected deltas
    """

    def __init__(
        self,
        run_id: str,
        llm_base_url: str = None,
        snapshot_dir: str = None,
        save_every: int = 0,
        batch_size: int = 8,
    ):
        self.run_id = run_id
        self.llm_base_url = llm_base_url or config.VLLM_BASE_URL
        self.batch_size = batch_size
        self.save_every = save_every

        self.rag_working_dir = config.RAG_WORKING_DIR + f"_{run_id}"
        self.memory_working_dir = config.MEMORY_WORKING_DIR + f"_{run_id}"
        self.snapshot_mgr = SnapshotManager(
            snapshot_dir or os.path.join(os.path.dirname(__file__), "snapshots")
        )

        self.store: Optional[ConstitutionStore] = None
        self.memory: Optional[MemoryStore] = None
        self.client: Optional[openai.AsyncOpenAI] = None

        self.results: List[Dict[str, Any]] = []

    # ── Lifecycle ───────────────────────────────────────────────────────

    async def initialize(self, snapshot_id: str = None):
        if snapshot_id:
            self.snapshot_mgr.load(
                snapshot_id, self.rag_working_dir, self.memory_working_dir
            )

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

        self.client = openai.AsyncOpenAI(
            base_url=self.llm_base_url, api_key="dummy"
        )

    async def finalize(self):
        if self.store:
            await self.store.finalize()
        if self.memory:
            await self.memory.finalize()

    # ── Single-item LLM calls (no state mutation) ──────────────────────

    async def _generate_one(
        self, query: str, constitution_text: str, memory_text: str
    ) -> str:
        """Generate a response. Pure LLM call, no writes."""
        system_prompt = GENERATE_SYSTEM.format(
            constitution=constitution_text, memory=memory_text
        )
        resp = await self.client.chat.completions.create(
            model=config.VLLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
            max_tokens=512,
            temperature=0.7,
        )
        return resp.choices[0].message.content

    async def _reflect_one(
        self,
        query: str,
        response: str,
        feedback: str,
        constitution_text: str,
        memory_text: str,
    ) -> dict:
        """Reflect on one interaction. Pure LLM call, no writes."""
        user_prompt = REFLECTION_USER.format(
            constitution=constitution_text,
            memory=memory_text,
            query=query,
            response=response,
            feedback=feedback,
        )
        resp = await self.client.chat.completions.create(
            model=config.VLLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": REFLECTION_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=1024,
            temperature=0.3,
        )
        raw = resp.choices[0].message.content
        try:
            return json.loads(repair_json(raw))
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Failed to parse reflection: {e}")
            return {"reasoning": raw, "deltas": []}

    # ── Delta application (sequential, mutates state) ──────────────────

    async def _apply_deltas(
        self, deltas: list[dict], interaction_id: str
    ) -> list[str]:
        """Apply deltas to constitution/memory. Same logic as Orchestrator."""
        applied = []
        for delta in deltas:
            action = delta.get("action", "").lower()
            try:
                if action == "add":
                    entry = ConstitutionEntry(
                        principle=delta["principle"],
                        confidence=float(delta.get("confidence", 0.5)),
                        evidence=[delta.get("evidence", interaction_id)],
                    )
                    eid = await self.store.add(entry)
                    applied.append(f"ADD {eid}: {entry.principle[:60]}...")
                elif action == "update":
                    eid = delta.get("id")
                    if not eid or eid not in self.store.index:
                        continue
                    updates = {}
                    if "principle" in delta:
                        updates["principle"] = delta["principle"]
                    if "confidence" in delta:
                        updates["confidence"] = float(delta["confidence"])
                    existing = await self.store.get(eid)
                    if existing:
                        updates["evidence"] = existing.evidence + [
                            delta.get("evidence", interaction_id)
                        ]
                    await self.store.update(eid, updates)
                    applied.append(f"UPDATE {eid}")
                elif action == "retire":
                    eid = delta.get("id")
                    if not eid or eid not in self.store.index:
                        continue
                    await self.store.retire(eid)
                    applied.append(f"RETIRE {eid}")
                elif action == "remember":
                    if not self.memory:
                        continue
                    content = delta.get("content", "")
                    if not content:
                        continue
                    mid = await self.memory.remember(
                        content=content,
                        source=interaction_id,
                        tags=delta.get("tags", []),
                    )
                    applied.append(f"REMEMBER {mid}: {content[:60]}...")
                elif action == "forget":
                    if not self.memory:
                        continue
                    mid = delta.get("id", "")
                    if not mid or mid not in self.memory.index:
                        continue
                    await self.memory.forget(mid)
                    applied.append(f"FORGET {mid}")
            except Exception as e:
                logger.error(f"Delta failed: {delta} — {e}")
        return applied

    # ── Mini-batch processing ──────────────────────────────────────────

    async def _process_mini_batch(
        self,
        batch: List[DatasetItem],
        batch_offset: int,
        total: int,
        verbose: bool,
    ) -> List[Dict[str, Any]]:
        """
        Process a mini-batch of K items:
          1. Snapshot constitution + memory (read)
          2. Concurrently generate K responses
          3. Concurrently reflect on K items
          4. Sequentially apply all deltas (write)
        """
        # 1. Snapshot current state (all items in this batch see the same state)
        constitution_text = self.store.summary()
        memory_text = self.memory.summary() if self.memory else "(No memories stored)"

        queries = [item.input_text for item in batch]
        feedbacks = [_format_reference_signal(item) for item in batch]

        # 2. Concurrent generate
        gen_tasks = [
            self._generate_one(q, constitution_text, memory_text)
            for q in queries
        ]
        responses = await asyncio.gather(*gen_tasks, return_exceptions=True)

        # 3. Concurrent reflect
        reflect_tasks = []
        for i, (item, resp) in enumerate(zip(batch, responses)):
            if isinstance(resp, Exception):
                reflect_tasks.append(asyncio.coroutine(lambda: {"reasoning": str(resp), "deltas": []})())
                continue
            reflect_tasks.append(
                self._reflect_one(
                    queries[i], resp, feedbacks[i],
                    constitution_text, memory_text,
                )
            )
        reflections = await asyncio.gather(*reflect_tasks, return_exceptions=True)

        # 4. Collect all deltas, apply sequentially
        batch_results = []
        all_deltas = []
        for i, (item, resp, refl) in enumerate(zip(batch, responses, reflections)):
            global_idx = batch_offset + i

            if isinstance(resp, Exception):
                record = {
                    "step": global_idx,
                    "item_id": item.id,
                    "dataset": item.dataset,
                    "error": str(resp),
                }
                batch_results.append(record)
                continue

            if isinstance(refl, Exception):
                refl = {"reasoning": str(refl), "deltas": []}

            deltas = refl.get("deltas", [])
            all_deltas.append((global_idx, item, deltas))

            record = {
                "step": global_idx,
                "item_id": item.id,
                "dataset": item.dataset,
                "category": item.category,
                "kind": item.kind.value,
                "response": resp if isinstance(resp, str) else "",
                "reflection": refl,
                "deltas_proposed": len(deltas),
            }
            batch_results.append(record)

        # Apply all deltas from this mini-batch sequentially
        total_applied = []
        for global_idx, item, deltas in all_deltas:
            interaction_id = f"{self.run_id}/step_{global_idx:05d}"
            applied = await self._apply_deltas(deltas, interaction_id)
            total_applied.extend(applied)
            # Attach applied info to the record
            for rec in batch_results:
                if rec.get("step") == global_idx:
                    rec["applied"] = applied

        if verbose:
            end_idx = batch_offset + len(batch)
            n_errors = sum(1 for r in batch_results if "error" in r)
            print(
                f"  [{batch_offset+1}-{end_idx}/{total}] "
                f"deltas={len(total_applied)} "
                f"errors={n_errors} "
                f"constitution={len(self.store.list_active())} "
                f"memory={len(self.memory.index)}"
            )

        return batch_results

    # ── Main run loop ──────────────────────────────────────────────────

    async def run(
        self,
        items: List[DatasetItem],
        verbose: bool = True,
    ) -> Dict[str, Any]:
        if not self.store:
            raise RuntimeError("Call initialize() before run().")

        total = len(items)
        K = self.batch_size

        if verbose:
            print(
                f"  Processing {total} items in mini-batches of {K} "
                f"({(total + K - 1) // K} rounds)"
            )

        for batch_start in range(0, total, K):
            batch = items[batch_start : batch_start + K]
            results = await self._process_mini_batch(
                batch, batch_start, total, verbose
            )
            self.results.extend(results)

            # Checkpoint
            if self.save_every > 0 and (batch_start + K) % self.save_every < K:
                ckpt_id = f"{self.run_id}_ckpt_{batch_start + len(batch):05d}"
                self._save_snapshot(ckpt_id)

        # Final snapshot
        self._save_snapshot(self.run_id)

        return self._build_summary(total)

    def _save_snapshot(self, snap_id: str):
        self.snapshot_mgr.save(
            run_id=snap_id,
            constitution_working_dir=self.rag_working_dir,
            memory_working_dir=self.memory_working_dir,
            manifest_extra={
                "total_items_processed": len(self.results),
                "final_constitution_size": len(self.store.list_active()),
                "final_memory_size": len(self.memory.index),
                "errors": sum(1 for r in self.results if "error" in r),
            },
        )

    def _build_summary(self, total: int) -> Dict[str, Any]:
        by_dataset: Dict[str, int] = {}
        total_deltas = 0
        errors = 0
        for r in self.results:
            ds = r.get("dataset", "unknown")
            by_dataset[ds] = by_dataset.get(ds, 0) + 1
            total_deltas += len(r.get("applied", []))
            if "error" in r:
                errors += 1

        return {
            "run_id": self.run_id,
            "total_items": total,
            "total_processed": len(self.results),
            "errors": errors,
            "total_deltas_applied": total_deltas,
            "final_constitution_size": len(self.store.list_active()),
            "final_memory_size": len(self.memory.index),
            "by_dataset": by_dataset,
        }

    def save_log(self, path: str = None) -> str:
        path = path or os.path.join(
            os.path.dirname(__file__), "snapshots", f"{self.run_id}_log.json"
        )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False, default=str)
        return path


# ── Top-level experiment runner (all conditions concurrent) ─────────────────

async def run_condition(
    condition: str,
    items: List[DatasetItem],
    experiment_id: str,
    llm_url: str = None,
    batch_size: int = 8,
    save_every: int = 0,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run a single condition end-to-end."""
    run_id = f"{experiment_id}/{condition}"

    if verbose:
        print(f"\n[{condition}] Starting — {len(items)} items, batch_size={batch_size}")

    runner = ParallelConditionRunner(
        run_id=run_id,
        llm_base_url=llm_url,
        batch_size=batch_size,
        save_every=save_every,
    )

    await runner.initialize()
    try:
        summary = await runner.run(items, verbose=verbose)
    finally:
        await runner.finalize()

    runner.save_log()

    if verbose:
        print(
            f"[{condition}] Done — "
            f"constitution={summary['final_constitution_size']} "
            f"memory={summary['final_memory_size']} "
            f"deltas={summary['total_deltas_applied']} "
            f"errors={summary['errors']}"
        )

    return {"condition": condition, **summary}


async def run_all_conditions(
    condition_items: Dict[str, List[DatasetItem]],
    experiment_id: str,
    llm_url: str = None,
    batch_size: int = 8,
    save_every: int = 0,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run all conditions concurrently.

    Parameters
    ----------
    condition_items : dict
        {condition_name: list_of_DatasetItem}
    """
    tasks = [
        run_condition(
            condition=cond,
            items=items,
            experiment_id=experiment_id,
            llm_url=llm_url,
            batch_size=batch_size,
            save_every=save_every,
            verbose=verbose,
        )
        for cond, items in condition_items.items()
    ]

    results_list = await asyncio.gather(*tasks, return_exceptions=True)

    results = {}
    for cond, res in zip(condition_items.keys(), results_list):
        if isinstance(res, Exception):
            logger.error(f"Condition {cond} failed: {res}")
            results[cond] = {"status": "error", "error": str(res)}
        else:
            results[cond] = {"status": "ok", **res}

    # Print comparison table
    print(f"\n{'='*70}")
    print(f"Experiment: {experiment_id}")
    print(f"{'='*70}")
    print(f"{'Condition':<12} {'Status':<8} {'Constitution':>13} {'Memory':>8} {'Deltas':>8} {'Errors':>8}")
    print("-" * 70)
    for cond in condition_items:
        r = results[cond]
        if r["status"] == "ok":
            print(
                f"{cond:<12} {'OK':<8} {r['final_constitution_size']:>13} "
                f"{r['final_memory_size']:>8} {r['total_deltas_applied']:>8} "
                f"{r['errors']:>8}"
            )
        else:
            print(f"{cond:<12} {'FAIL':<8} {r.get('error', '')[:50]}")

    # Save summary
    summary_path = os.path.join(
        os.path.dirname(__file__), "snapshots", f"{experiment_id}_summary.json"
    )
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    return results
