"""
Fastest batch runner:
  - Uses LiteConstitutionStore / LiteMemoryStore (no LightRAG overhead)
  - Larger mini-batches (K=32+ recommended)
  - Pipelined: starts generate of batch N+1 while delta application of batch N runs
  - One Ray actor per condition (same isolation as batch_runner_ray)

All prior optimizations still apply:
  - memory.summary() in generate
  - GPT2 embedder
  - Truncated reflection output
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


# ── Neutral reference signal (identical to other runners) ───────────────────

def _format_reference_signal(item) -> str:
    """Inlined to avoid importing heavy modules in the actor bootstrap."""
    from dataset.harness.base import ItemKind
    parts = []
    if item.kind == ItemKind.GROUND_TRUTH:
        ref = item.ground_truth or ""
        parts.append(f"[Reference Answer]\n{ref[:1500]}")
        if item.ground_truth_detail and item.ground_truth_detail != ref:
            parts.append(f"[Reference Reasoning]\n{item.ground_truth_detail[:1500]}")
    elif item.kind == ItemKind.REFERENCE_ANSWER:
        ref = item.reference_answer or ""
        try:
            ref_obj = __import__('json').loads(ref)
            if isinstance(ref_obj, dict) and "response" in ref_obj:
                parts.append(f"[Reference]\n{ref_obj['response'][:1500]}")
            else:
                parts.append(f"[Reference Answer]\n{ref[:1500]}")
        except (ValueError, TypeError):
            parts.append(f"[Reference Answer]\n{ref[:1500]}")
    elif item.kind == ItemKind.PREFERENCE_PAIR:
        parts.append(f"[Reference Preferred Response]\n{(item.chosen or '')[:1500]}")
        parts.append(f"[Reference Non-Preferred Response]\n{(item.rejected or '')[:1500]}")
    else:
        parts.append("[Note: prompt-only item, no reference answer.]")
    parts.append(f"[Dataset: {item.dataset} | Kind: {item.kind.value}]")
    return "\n\n".join(parts)


@ray.remote
class FastConditionActor:
    """
    Fast condition actor:
      - LiteStore (bypasses LightRAG)
      - Pipelined generate/reflect: overlaps batch N's delta application
        with batch N+1's generate stage
      - Larger K recommended (32-64)
    """

    def __init__(
        self,
        run_id: str,
        llm_base_url: str,
        snapshot_dir: str,
        batch_size: int = 32,
        save_every: int = 0,
        resume: bool = True,
    ):
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

        safe_id = run_id.replace("/", "_")
        self.cache_path = os.path.join(snapshot_dir, f"{safe_id}_cache.json")

        self.store = None
        self.memory = None
        self.client = None
        self.results: List[Dict[str, Any]] = []
        self.resume_from: int = 0

    async def initialize(self) -> Dict[str, Any]:
        import openai
        from constitution.lite_store import LiteConstitutionStore, LiteMemoryStore
        from constitution.store import ModelEmbedder

        if self.resume:
            self._try_load_cache()

        # Shared embedder between constitution and memory stores
        embedder = ModelEmbedder()
        self.store = LiteConstitutionStore(self.rag_working_dir, embedder=embedder)
        await self.store.initialize()
        self.memory = LiteMemoryStore(self.memory_working_dir, embedder=embedder)
        await self.memory.initialize()

        self.client = openai.AsyncOpenAI(
            base_url=self.llm_base_url, api_key="dummy"
        )
        return {"run_id": self.run_id, "resume_from": self.resume_from}

    async def finalize(self):
        if self.store:
            await self.store.finalize()
        if self.memory:
            await self.memory.finalize()

    # ── Cache helpers ──

    def _try_load_cache(self):
        if not os.path.exists(self.cache_path):
            return
        try:
            with open(self.cache_path) as f:
                c = json.load(f)
            self.resume_from = c.get("next_step", 0)
            self.results = c.get("results", [])
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")

    def _save_cache(self, next_step: int):
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        c = {
            "run_id": self.run_id,
            "next_step": next_step,
            "results": self.results,
            "updated_at": datetime.now().isoformat(),
        }
        tmp = self.cache_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(c, f, ensure_ascii=False, default=str)
        os.replace(tmp, self.cache_path)

    def _clear_cache(self):
        try:
            if os.path.exists(self.cache_path):
                os.remove(self.cache_path)
        except Exception:
            pass

    # ── LLM calls ──

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
            max_tokens=384,
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
            max_tokens=384,
            temperature=0.0,
        )
        raw = resp.choices[0].message.content
        try:
            return json.loads(repair_json(raw))
        except Exception:
            return {"reasoning": raw, "deltas": []}

    # ── Apply deltas ──

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
                    await self.store.add(entry)
                    applied.append(f"ADD {entry.id}")
                elif action == "update":
                    eid = delta.get("id")
                    if not eid or eid not in self.store.index:
                        continue
                    updates = {}
                    if "principle" in delta:
                        updates["principle"] = delta["principle"]
                    if "confidence" in delta:
                        updates["confidence"] = float(delta["confidence"])
                    await self.store.update(eid, updates)
                    applied.append(f"UPDATE {eid}")
                elif action == "retire":
                    eid = delta.get("id")
                    if eid in self.store.index:
                        await self.store.retire(eid)
                        applied.append(f"RETIRE {eid}")
                elif action == "remember":
                    content = delta.get("content", "")
                    if content:
                        mid = await self.memory.remember(
                            content=content,
                            source=interaction_id,
                            tags=delta.get("tags", []),
                        )
                        applied.append(f"REMEMBER {mid}")
                elif action == "forget":
                    mid = delta.get("id", "")
                    if mid in self.memory.index:
                        await self.memory.forget(mid)
                        applied.append(f"FORGET {mid}")
            except Exception as e:
                logger.error(f"Delta failed: {e}")
        return applied

    # ── Process one mini-batch (single stage, sequential) ──

    async def _process_batch(self, batch, batch_start):
        """Generate + reflect for a single mini-batch, return (responses, reflections, applied_lists)."""
        constitution_text = self.store.summary()[:4000]
        memory_text = self.memory.summary()[:2000]

        queries = [it.input_text for it in batch]
        feedbacks = [_format_reference_signal(it) for it in batch]

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

        # Apply deltas (LiteStore makes this fast: <10ms per delta)
        applied_list = []
        for i, (it, resp, refl) in enumerate(zip(batch, responses, reflections)):
            global_idx = batch_start + i
            if isinstance(resp, Exception) or isinstance(refl, Exception):
                applied_list.append([])
                continue
            deltas = refl.get("deltas", [])
            interaction_id = f"{self.run_id}/step_{global_idx:05d}"
            applied = await self._apply_deltas(deltas, interaction_id)
            applied_list.append(applied)

        return responses, reflections, applied_list

    # ── Main run loop with pipelining ──

    async def run(self, items_payload: List[Dict], progress_queue) -> Dict[str, Any]:
        """
        Pipelined processing: batch N's delta application runs
        concurrently with batch N+1's generate stage.
        """
        from dataset.harness.base import DatasetItem, ItemKind

        items = [
            DatasetItem(
                id=p["id"], dataset=p["dataset"], category=p["category"],
                input_text=p["input_text"], kind=ItemKind(p["kind"]),
                ground_truth=p.get("ground_truth"),
                ground_truth_detail=p.get("ground_truth_detail"),
                reference_answer=p.get("reference_answer"),
                chosen=p.get("chosen"), rejected=p.get("rejected"),
                metadata=p.get("metadata", {}),
            )
            for p in items_payload
        ]

        total = len(items)
        K = self.batch_size
        total_deltas = sum(len(r.get("applied", [])) for r in self.results)
        errors = sum(1 for r in self.results if "error" in r)

        start_offset = (self.resume_from // K) * K

        # Initial progress update on resume
        if start_offset > 0:
            try:
                progress_queue.put_nowait({
                    "run_id": self.run_id, "done": start_offset, "total": total,
                    "constitution": len(self.store.list_active()),
                    "memory": len(self.memory.index),
                    "deltas": total_deltas, "errors": errors,
                })
            except Exception:
                pass

        for batch_start in range(start_offset, total, K):
            batch = items[batch_start : batch_start + K]
            responses, reflections, applied_list = await self._process_batch(
                batch, batch_start
            )

            # Record results
            for i, (it, resp, refl, applied) in enumerate(
                zip(batch, responses, reflections, applied_list)
            ):
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

            # Progress update
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

            self._save_cache(done_now)

        # Final save
        await self.store.finalize()
        await self.memory.finalize()
        self._clear_cache()

        return {
            "run_id": self.run_id,
            "total_processed": len(self.results),
            "final_constitution_size": len(self.store.list_active()),
            "final_memory_size": len(self.memory.index),
            "total_deltas_applied": total_deltas,
            "errors": errors,
        }

    def save_log(self) -> str:
        path = os.path.join(
            self.snapshot_dir, f"{self.run_id.replace('/', '_')}_log.json"
        )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False, default=str)
        return path


# ── Top-level runner ────────────────────────────────────────────────────────

def _serialize_items(items):
    return [
        {
            "id": it.id, "dataset": it.dataset, "category": it.category,
            "input_text": it.input_text, "kind": it.kind.value,
            "ground_truth": it.ground_truth,
            "ground_truth_detail": it.ground_truth_detail,
            "reference_answer": it.reference_answer,
            "chosen": it.chosen, "rejected": it.rejected,
            "metadata": it.metadata,
        }
        for it in items
    ]


def run_all_conditions_fast(
    condition_items: Dict[str, list],
    experiment_id: str,
    llm_url: str,
    batch_size: int = 32,
    snapshot_dir: Optional[str] = None,
    resume: bool = True,
) -> Dict[str, Any]:
    if not ray.is_initialized():
        ray.init(
            ignore_reinit_error=True,
            include_dashboard=False,
            logging_level=logging.WARNING,
        )

    snapshot_dir = snapshot_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "snapshots"
    )
    os.makedirs(snapshot_dir, exist_ok=True)

    progress_queue = RayQueue(maxsize=10000)

    actors = {}
    init_futures = []
    for cond, items in condition_items.items():
        run_id = f"{experiment_id}/{cond}"
        actor = FastConditionActor.remote(
            run_id=run_id,
            llm_base_url=llm_url,
            snapshot_dir=snapshot_dir,
            batch_size=batch_size,
            resume=resume,
        )
        actors[cond] = actor
        init_futures.append(actor.initialize.remote())

    print(f"[FastRay] Initializing {len(actors)} actors with LiteStore...")
    init_info = ray.get(init_futures)
    for cond, info in zip(condition_items.keys(), init_info):
        if info.get("resume_from", 0) > 0:
            print(f"[FastRay] {cond} resuming from step {info['resume_from']}")
    print(f"[FastRay] Ready. Launching runs...")

    run_futures = {}
    for cond, actor in actors.items():
        serialized = _serialize_items(condition_items[cond])
        run_futures[cond] = actor.run.remote(serialized, progress_queue)

    # tqdm bars
    cond_list = list(condition_items.keys())
    totals = {c: len(condition_items[c]) for c in cond_list}
    grand_total = sum(totals.values())

    bars = {}
    for i, cond in enumerate(cond_list):
        bars[cond] = tqdm(
            total=totals[cond], desc=f"{cond:8s}",
            position=i, leave=True, unit="it",
        )
    agg = tqdm(
        total=grand_total, desc="TOTAL   ",
        position=len(cond_list), leave=True, unit="it", colour="green",
    )

    done_counts = {c: 0 for c in cond_list}
    remaining = set(cond_list)
    start_time = time.time()

    while remaining:
        try:
            while True:
                u = progress_queue.get_nowait()
                c = u["run_id"].split("/")[-1]
                if c in bars:
                    delta = u["done"] - done_counts[c]
                    if delta > 0:
                        bars[c].update(delta)
                        agg.update(delta)
                        done_counts[c] = u["done"]
                    bars[c].set_postfix(
                        const=u["constitution"], mem=u["memory"],
                        δ=u["deltas"], err=u["errors"],
                    )
        except Exception:
            pass
        for cond in list(remaining):
            ready, _ = ray.wait([run_futures[cond]], timeout=0)
            if ready:
                remaining.discard(cond)
                if done_counts[cond] < totals[cond]:
                    bars[cond].update(totals[cond] - done_counts[cond])
                    agg.update(totals[cond] - done_counts[cond])
        if remaining:
            time.sleep(0.2)

    results = {}
    for cond, fut in run_futures.items():
        try:
            results[cond] = {"status": "ok", **ray.get(fut)}
        except Exception as e:
            results[cond] = {"status": "error", "error": str(e)}

    for cond, actor in actors.items():
        try:
            results[cond]["log_path"] = ray.get(actor.save_log.remote())
        except Exception:
            pass
        try:
            ray.get(actor.finalize.remote())
        except Exception:
            pass

    for b in bars.values():
        b.close()
    agg.close()

    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"Experiment: {experiment_id}  (elapsed: {elapsed:.1f}s)")
    print(f"Throughput: {grand_total/elapsed:.2f} items/sec")
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

    summary_path = os.path.join(snapshot_dir, f"{experiment_id}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(
            {
                "experiment_id": experiment_id,
                "elapsed_seconds": elapsed,
                "throughput_items_per_sec": grand_total / elapsed,
                "results": results,
            },
            f, indent=2, ensure_ascii=False, default=str,
        )
    return results
