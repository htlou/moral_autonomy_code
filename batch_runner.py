"""
Ray-based batch runner: each condition runs as an independent Ray actor
in its own process, with its own ConstitutionStore and MemoryStore.

Features:
  - Process-level isolation (each condition has its own memory space)
  - Mini-batch parallelism within each actor (K concurrent LLM calls)
  - tqdm progress bars (one per condition + aggregate)
  - Real-time progress reporting via ray Queue
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import ray
from ray.util.queue import Queue as RayQueue
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ── Ray Actor: one condition per process ────────────────────────────────────

@ray.remote
class ConditionActor:
    """
    A single condition runs here.  Instantiated once per condition.
    Owns its own ConstitutionStore and MemoryStore.
    """

    def __init__(
        self,
        run_id: str,
        llm_base_url: str,
        snapshot_dir: str,
        batch_size: int = 8,
        save_every: int = 0,
        resume: bool = True,
    ):
        # Heavy imports happen inside the actor process
        os.environ.setdefault("HF_HUB_OFFLINE", "0")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")

        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

        self.run_id = run_id
        self.llm_base_url = llm_base_url
        self.snapshot_dir = snapshot_dir
        self.batch_size = batch_size
        self.save_every = save_every
        self.resume = resume

        import config
        self.rag_working_dir = config.RAG_WORKING_DIR + f"_{run_id}"
        self.memory_working_dir = config.MEMORY_WORKING_DIR + f"_{run_id}"

        # Cache file tracks progress so interrupted runs can resume.
        safe_id = run_id.replace("/", "_")
        self.cache_path = os.path.join(snapshot_dir, f"{safe_id}_cache.json")

        self.store = None
        self.memory = None
        self.client = None
        self.results: List[Dict[str, Any]] = []
        self.resume_from: int = 0  # step index to start from

    async def initialize(self, snapshot_id: str = None) -> Dict[str, Any]:
        """
        Initialize stores and LLM client.

        If ``self.resume`` is True and a cache file exists for this run_id,
        restore the working dirs (they persist across runs) and the
        in-memory results list, then set ``self.resume_from`` so ``run``
        can skip already-processed items.

        Returns a dict with {"run_id": ..., "resume_from": ..., "prior_results": n}.
        """
        import openai
        from constitution.store import ConstitutionStore
        from constitution.memory_store import MemoryStore
        from snapshot import SnapshotManager

        if snapshot_id:
            mgr = SnapshotManager(self.snapshot_dir)
            mgr.load(snapshot_id, self.rag_working_dir, self.memory_working_dir)
        elif self.resume:
            self._try_load_cache()

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
        return {
            "run_id": self.run_id,
            "resume_from": self.resume_from,
            "prior_results": len(self.results),
        }

    # ── Cache (resume mechanism) ──────────────────────────────────────

    def _try_load_cache(self):
        """Load the progress cache for this run_id if it exists."""
        if not os.path.exists(self.cache_path):
            return
        try:
            with open(self.cache_path) as f:
                cache = json.load(f)
            self.resume_from = cache.get("next_step", 0)
            self.results = cache.get("results", [])
            logger.info(
                f"[{self.run_id}] Resuming from step {self.resume_from} "
                f"({len(self.results)} prior records)"
            )
        except Exception as e:
            logger.warning(f"Failed to load cache {self.cache_path}: {e}")

    def _save_cache(self, next_step: int, total_deltas: int, errors: int):
        """Write progress cache atomically."""
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        cache = {
            "run_id": self.run_id,
            "next_step": next_step,
            "total_deltas": total_deltas,
            "errors": errors,
            "results": self.results,
            "updated_at": datetime.now().isoformat(),
        }
        tmp_path = self.cache_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(cache, f, ensure_ascii=False, default=str)
        os.replace(tmp_path, self.cache_path)

    def _clear_cache(self):
        """Remove the cache file after a successful completion."""
        try:
            if os.path.exists(self.cache_path):
                os.remove(self.cache_path)
        except Exception as e:
            logger.warning(f"Failed to clear cache: {e}")

    async def finalize(self):
        if self.store:
            await self.store.finalize()
        if self.memory:
            await self.memory.finalize()

    # ── LLM calls (no state mutation) ──

    async def _generate_one(self, query, constitution_text, memory_text):
        from constitution.prompts import GENERATE_SYSTEM
        import config
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
        self, query, response, feedback, constitution_text, memory_text
    ):
        from constitution.prompts import REFLECTION_SYSTEM, REFLECTION_USER
        from json_repair import repair_json
        import config
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
            max_tokens=768,
            temperature=0.3,
        )
        raw = resp.choices[0].message.content
        try:
            return json.loads(repair_json(raw))
        except Exception as e:
            return {"reasoning": raw, "deltas": []}

    # ── Delta application (serial) ──

    async def _apply_deltas(self, deltas, interaction_id):
        from constitution.schema import ConstitutionEntry
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
                    applied.append(f"ADD {eid}")
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
                    applied.append(f"REMEMBER {mid}")
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

    # ── Main loop: process items in mini-batches ──

    async def run(
        self,
        items_payload: List[Dict],
        progress_queue: RayQueue,
    ) -> Dict[str, Any]:
        """
        Process items. items_payload is a list of dicts (since Ray serializes
        across processes, we can't pass DatasetItem objects directly).

        Pushes progress updates to progress_queue as:
            {"run_id": ..., "done": n, "constitution": c, "memory": m,
             "deltas": d, "errors": e}
        """
        from constitution.prompts import format_reference_signal
        from dataset.harness.base import DatasetItem, ItemKind

        # Rehydrate DatasetItems
        items = [
            DatasetItem(
                id=p["id"],
                dataset=p["dataset"],
                category=p["category"],
                input_text=p["input_text"],
                kind=ItemKind(p["kind"]),
                ground_truth=p.get("ground_truth"),
                ground_truth_detail=p.get("ground_truth_detail"),
                reference_answer=p.get("reference_answer"),
                chosen=p.get("chosen"),
                rejected=p.get("rejected"),
                metadata=p.get("metadata", {}),
            )
            for p in items_payload
        ]

        total = len(items)
        K = self.batch_size
        # Rebuild running tallies from cached results
        total_deltas = sum(len(r.get("applied", [])) for r in self.results)
        errors = sum(1 for r in self.results if "error" in r)

        # If resuming, push an initial progress update so the tqdm bar
        # reflects the cached state.
        if self.resume_from > 0:
            try:
                progress_queue.put_nowait({
                    "run_id": self.run_id,
                    "done": self.resume_from,
                    "total": total,
                    "constitution": 0,  # will be filled once store is loaded
                    "memory": 0,
                    "deltas": total_deltas,
                    "errors": errors,
                })
            except Exception:
                pass

        # Snap resume_from to a K-aligned boundary so mini-batches stay aligned
        start_offset = (self.resume_from // K) * K

        for batch_start in range(start_offset, total, K):
            batch = items[batch_start : batch_start + K]

            # Read snapshot
            constitution_text = self.store.summary()
            memory_text = (
                self.memory.summary() if self.memory else "(No memories stored)"
            )

            queries = [it.input_text for it in batch]
            feedbacks = [format_reference_signal(it) for it in batch]

            # Concurrent generate
            gen_tasks = [
                self._generate_one(q, constitution_text, memory_text)
                for q in queries
            ]
            responses = await asyncio.gather(*gen_tasks, return_exceptions=True)

            # Concurrent reflect
            reflect_tasks = []
            for i, (it, resp) in enumerate(zip(batch, responses)):
                if isinstance(resp, Exception):
                    async def _err(e=resp):
                        return {"reasoning": str(e), "deltas": []}
                    reflect_tasks.append(_err())
                else:
                    reflect_tasks.append(
                        self._reflect_one(
                            queries[i], resp, feedbacks[i],
                            constitution_text, memory_text,
                        )
                    )
            reflections = await asyncio.gather(*reflect_tasks, return_exceptions=True)

            # Apply deltas serially
            for i, (it, resp, refl) in enumerate(zip(batch, responses, reflections)):
                global_idx = batch_start + i
                if isinstance(resp, Exception):
                    errors += 1
                    self.results.append({
                        "step": global_idx,
                        "item_id": it.id,
                        "dataset": it.dataset,
                        "error": str(resp),
                    })
                    continue
                if isinstance(refl, Exception):
                    refl = {"reasoning": str(refl), "deltas": []}

                deltas = refl.get("deltas", [])
                interaction_id = f"{self.run_id}/step_{global_idx:05d}"
                applied = await self._apply_deltas(deltas, interaction_id)
                total_deltas += len(applied)

                self.results.append({
                    "step": global_idx,
                    "item_id": it.id,
                    "dataset": it.dataset,
                    "category": it.category,
                    "kind": it.kind.value,
                    "response": resp,
                    "reflection": refl,
                    "applied": applied,
                })

            done_now = batch_start + len(batch)

            # Push progress update
            try:
                progress_queue.put_nowait({
                    "run_id": self.run_id,
                    "done": done_now,
                    "total": total,
                    "constitution": len(self.store.list_active()),
                    "memory": len(self.memory.index),
                    "deltas": total_deltas,
                    "errors": errors,
                })
            except Exception:
                pass

            # Persist cache after every mini-batch — lightweight so cheap
            self._save_cache(done_now, total_deltas, errors)

            # Heavy checkpoint (full snapshot with RAG dirs)
            if self.save_every > 0:
                if done_now // self.save_every > batch_start // self.save_every:
                    await self._save_checkpoint(done_now)

        # Final snapshot
        snap_path = await self._save_final_snapshot()

        # Clear the resume cache — this run is complete
        self._clear_cache()

        return {
            "run_id": self.run_id,
            "total_processed": len(self.results),
            "final_constitution_size": len(self.store.list_active()),
            "final_memory_size": len(self.memory.index),
            "total_deltas_applied": total_deltas,
            "errors": errors,
            "snapshot_path": str(snap_path),
        }

    async def _save_checkpoint(self, done: int):
        from snapshot import SnapshotManager
        mgr = SnapshotManager(self.snapshot_dir)
        ckpt_id = f"{self.run_id}_ckpt_{done:05d}"
        mgr.save(
            run_id=ckpt_id,
            constitution_working_dir=self.rag_working_dir,
            memory_working_dir=self.memory_working_dir,
            manifest_extra={"processed": done},
        )

    async def _save_final_snapshot(self):
        from snapshot import SnapshotManager
        mgr = SnapshotManager(self.snapshot_dir)
        return mgr.save(
            run_id=self.run_id,
            constitution_working_dir=self.rag_working_dir,
            memory_working_dir=self.memory_working_dir,
            manifest_extra={
                "total_processed": len(self.results),
                "final_constitution_size": len(self.store.list_active()),
                "final_memory_size": len(self.memory.index),
            },
        )

    def save_log(self) -> str:
        path = os.path.join(
            self.snapshot_dir, f"{self.run_id.replace('/', '_')}_log.json"
        )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False, default=str)
        return path


# ── Main orchestration with tqdm ────────────────────────────────────────────

def _serialize_items(items) -> List[Dict]:
    """Convert DatasetItem list to plain dicts for Ray serialization."""
    return [
        {
            "id": it.id,
            "dataset": it.dataset,
            "category": it.category,
            "input_text": it.input_text,
            "kind": it.kind.value,
            "ground_truth": it.ground_truth,
            "ground_truth_detail": it.ground_truth_detail,
            "reference_answer": it.reference_answer,
            "chosen": it.chosen,
            "rejected": it.rejected,
            "metadata": it.metadata,
        }
        for it in items
    ]


def run_all_conditions_ray(
    condition_items: Dict[str, list],
    experiment_id: str,
    llm_url: str,
    batch_size: int = 8,
    save_every: int = 0,
    snapshot_dir: Optional[str] = None,
    ray_address: Optional[str] = None,
    resume: bool = True,
) -> Dict[str, Any]:
    """
    Launch all conditions as Ray actors and display real-time tqdm progress.
    """
    if not ray.is_initialized():
        ray.init(
            address=ray_address,
            ignore_reinit_error=True,
            include_dashboard=False,
            logging_level=logging.WARNING,
        )

    snapshot_dir = snapshot_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "snapshots"
    )
    os.makedirs(snapshot_dir, exist_ok=True)

    # Create a progress queue shared across actors
    progress_queue = RayQueue(maxsize=1000)

    # Create actors
    actors = {}
    init_futures = []
    for cond, items in condition_items.items():
        run_id = f"{experiment_id}/{cond}"
        actor = ConditionActor.remote(
            run_id=run_id,
            llm_base_url=llm_url,
            snapshot_dir=snapshot_dir,
            batch_size=batch_size,
            save_every=save_every,
            resume=resume,
        )
        actors[cond] = actor
        init_futures.append(actor.initialize.remote())

    # Wait for all actors to initialize
    print(f"[Ray] Initializing {len(actors)} actors...")
    init_results = ray.get(init_futures)
    for cond, info in zip(condition_items.keys(), init_results):
        if info.get("resume_from", 0) > 0:
            print(
                f"[Ray] {cond} resuming from step {info['resume_from']} "
                f"({info['prior_results']} prior records)"
            )
    print(f"[Ray] All actors ready.")

    # Launch run tasks
    run_futures = {}
    for cond, actor in actors.items():
        items = condition_items[cond]
        serialized = _serialize_items(items)
        run_futures[cond] = actor.run.remote(serialized, progress_queue)

    # Create tqdm bars: one per condition + one aggregate
    cond_list = list(condition_items.keys())
    totals = {cond: len(condition_items[cond]) for cond in cond_list}
    grand_total = sum(totals.values())

    bars = {}
    for i, cond in enumerate(cond_list):
        bars[cond] = tqdm(
            total=totals[cond],
            desc=f"{cond:8s}",
            position=i,
            leave=True,
            unit="it",
        )
    agg_bar = tqdm(
        total=grand_total,
        desc="TOTAL   ",
        position=len(cond_list),
        leave=True,
        unit="it",
        colour="green",
    )

    # Poll progress queue + check completion
    done_counts = {cond: 0 for cond in cond_list}
    remaining = set(cond_list)
    start_time = time.time()

    while remaining:
        # Drain progress queue
        try:
            while True:
                update = progress_queue.get_nowait()
                cond = update["run_id"].split("/")[-1]
                if cond in bars:
                    delta = update["done"] - done_counts[cond]
                    if delta > 0:
                        bars[cond].update(delta)
                        agg_bar.update(delta)
                        done_counts[cond] = update["done"]
                    bars[cond].set_postfix(
                        const=update["constitution"],
                        mem=update["memory"],
                        δ=update["deltas"],
                        err=update["errors"],
                    )
        except Exception:
            pass  # queue empty

        # Check for finished actors
        finished_conds = []
        for cond in list(remaining):
            fut = run_futures[cond]
            ready, _ = ray.wait([fut], timeout=0)
            if ready:
                finished_conds.append(cond)
        for cond in finished_conds:
            remaining.discard(cond)
            # Top off the bar if needed
            if done_counts[cond] < totals[cond]:
                bars[cond].update(totals[cond] - done_counts[cond])
                agg_bar.update(totals[cond] - done_counts[cond])

        if remaining:
            time.sleep(0.2)

    # Collect final results
    results = {}
    for cond, fut in run_futures.items():
        try:
            summary = ray.get(fut)
            results[cond] = {"status": "ok", **summary}
        except Exception as e:
            results[cond] = {"status": "error", "error": str(e)}

    # Save logs and finalize actors
    for cond, actor in actors.items():
        try:
            log_path = ray.get(actor.save_log.remote())
            results[cond]["log_path"] = log_path
        except Exception as e:
            logger.warning(f"save_log failed for {cond}: {e}")
        try:
            ray.get(actor.finalize.remote())
        except Exception as e:
            logger.warning(f"finalize failed for {cond}: {e}")

    # Close progress bars
    for b in bars.values():
        b.close()
    agg_bar.close()

    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"Experiment: {experiment_id}  (elapsed: {elapsed:.1f}s)")
    print(f"{'='*70}")
    print(f"{'Condition':<12} {'Status':<8} {'Constitution':>13} {'Memory':>8} {'Deltas':>8} {'Errors':>8}")
    print("-" * 70)
    for cond in cond_list:
        r = results[cond]
        if r["status"] == "ok":
            print(
                f"{cond:<12} {'OK':<8} {r['final_constitution_size']:>13} "
                f"{r['final_memory_size']:>8} {r['total_deltas_applied']:>8} "
                f"{r['errors']:>8}"
            )
        else:
            print(f"{cond:<12} {'FAIL':<8} {r.get('error','')[:50]}")

    # Save summary
    summary_path = os.path.join(
        snapshot_dir, f"{experiment_id}_summary.json"
    )
    with open(summary_path, "w") as f:
        json.dump(
            {
                "experiment_id": experiment_id,
                "elapsed_seconds": elapsed,
                "results": results,
            },
            f, indent=2, ensure_ascii=False, default=str,
        )
    print(f"\nSummary saved: {summary_path}")

    return results
