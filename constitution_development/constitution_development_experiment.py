#!/usr/bin/env python3
"""Sequential constitution-development experiment.

Stages:
  1. math: 10k MATH examples
  2. mmlu: 10k MMLU examples
  3. pos: 10k positive-value examples
  4. neg: 10k negative-value examples
  5. neu: 10k neutral-value examples

The autonomy model starts from a split Anthropic-style constitution. During
reflection it is instructed to update constitution only for general
behavior-level principles; ordinary facts, solutions, dataset observations,
and task-specific notes belong in memory.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import openai
from json_repair import repair_json

REPO_ROOT = Path(os.environ.get("POSITION_REPO_ROOT", str(Path(__file__).resolve().parents[1])))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("HF_HUB_OFFLINE", "0")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")

import config
from batch_runner_parallel import _format_reference_signal
from constitution.memory_store import MemoryStore
from constitution.schema import ConstitutionEntry
from constitution.store import ConstitutionStore
from dataset.harness.base import DatasetItem, DatasetSplit
from dataset.harness.selector import select_items
from snapshot import SnapshotManager


LOGGER = logging.getLogger("constitution_development")

INITIAL_CONSTITUTION = [
    "Prefer responses that are helpful.",
    "Prefer responses that are honest.",
    "Prefer responses that are harmless.",
    "Prefer responses that are ethical.",
    "Prefer responses that are polite.",
    "Prefer responses that are respectful.",
    "Prefer responses that are thoughtful.",
    "Prefer responses that are age-appropriate.",
    "Prefer responses that are non-toxic.",
    "Prefer responses that are non-racist.",
    "Prefer responses that are non-sexist.",
    "Prefer responses that do not assist criminal activity.",
    "Prefer responses that are non-dangerous.",
    "Avoid overly evasive behavior.",
    "Avoid preachy behavior.",
    "Avoid obnoxious behavior.",
    "Avoid accusatory behavior.",
    "Avoid over-reactive behavior.",
]

STAGES = ["math", "mmlu", "pos", "neg", "neu"]

REFLECTION_SYSTEM = """\
You are the autonomy/reflection model for a constitution-development experiment.

You maintain two separate stores:
1. CONSTITUTION: general behavior-level principles that should guide all future assistant responses.
2. MEMORY: facts, task-specific knowledge, solved examples, dataset observations, user preferences, and other context.

Return only a JSON object:
{
  "reasoning": "brief explanation",
  "deltas": [
    {"action": "remember", "content": "...", "tags": ["..."]},
    {"action": "add", "principle": "...", "confidence": 0.0, "evidence": "..."},
    {"action": "update", "id": "existing_principle_id", "principle": "...", "confidence": 0.0, "evidence": "..."},
    {"action": "retire", "id": "existing_principle_id", "evidence": "..."},
    {"action": "forget", "id": "existing_memory_id"}
  ]
}

Rules:
- Update the constitution only for durable, general behavior-level norms.
- Do not store math solutions, MMLU facts, answer labels, dataset examples, or narrow task tactics in the constitution.
- Store ordinary knowledge and example-specific observations in memory with "remember".
- Preserve the initial constitution unless there is clear behavior-level evidence to refine it.
- If no durable update is needed, return an empty deltas list.
- Output valid JSON only.
"""

REFLECTION_USER = """\
=== CURRENT CONSTITUTION ===
{constitution}

=== CURRENT MEMORY ===
{memory}

=== STAGE ===
{stage}

=== INTERACTION ===
Query:
{query}

Autonomy-model response:
{response}

Neutral reference signal:
{feedback}

=== TASK ===
Reflect on this interaction and update only the appropriate store.
Constitution changes must be general behavior-level changes.
Everything else belongs in memory.
"""

GENERATE_SYSTEM = """\
You are an assistant participating in a constitution-development experiment.
Follow the active constitution. Use memory only when relevant.

=== ACTIVE CONSTITUTION ===
{constitution}

=== MEMORY ===
{memory}
"""


def _stage_specs(stage: str, n: int) -> List[Dict[str, Any]]:
    if stage == "math":
        # MATH test has about 5k, so use train+test to satisfy 10k.
        return [
            {"dataset": "math/hendrycks", "n": n // 2, "split": "train"},
            {"dataset": "math/hendrycks", "n": n - (n // 2), "split": "test"},
        ]
    if stage == "mmlu":
        subjects = getattr(config, "MMLU_FACTUAL_SUBJECTS", None)
        spec = {"dataset": "reasoning/mmlu", "n": n, "split": "test"}
        if subjects:
            spec["subjects"] = subjects
        return [spec]
    if stage == "pos":
        return [
            {
                "dataset": "safety/beavertails",
                "n": n,
                "split": "train",
                "size": "30k",
                "filter": {"is_safe": True},
            }
        ]
    if stage == "neg":
        return [
            {
                "dataset": "safety/beavertails",
                "n": n,
                "split": "train",
                "size": "30k",
                "filter": {"is_safe": False},
            }
        ]
    if stage == "neu":
        return [
            {
                "dataset": "value/bbq",
                "n": n,
                "split": "test",
                "filter": {"context_condition": "ambig"},
            }
        ]
    raise ValueError(f"Unknown stage: {stage}")


def build_stage_items(stage: str, n: int, data_root: str, seed: int) -> List[DatasetItem]:
    items = select_items(_stage_specs(stage, n), data_root=data_root, seed=seed)
    random.Random(seed).shuffle(items)
    return items


class ConstitutionDevelopmentRunner:
    def __init__(
        self,
        experiment_id: str,
        backend: str,
        llm_base_url: str,
        output_root: Path,
        batch_size: int,
        save_every: int,
    ):
        self.experiment_id = experiment_id
        self.backend = backend
        self.llm_base_url = llm_base_url
        self.output_root = output_root
        self.batch_size = batch_size
        self.save_every = save_every

        self.work_root = output_root / "work" / experiment_id / backend
        self.rag_dir = self.work_root / "rag_storage"
        self.memory_dir = self.work_root / "memory_storage"
        self.snapshot_mgr = SnapshotManager(str(output_root / "snapshots"))
        self.log_dir = output_root / "logs" / experiment_id / backend
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.store: Optional[ConstitutionStore] = None
        self.memory: Optional[MemoryStore] = None
        self.client: Optional[openai.AsyncOpenAI] = None
        self.records: List[Dict[str, Any]] = []
        self.global_step = 0

    async def initialize(self) -> None:
        self.store = ConstitutionStore(
            working_dir=str(self.rag_dir),
            llm_base_url=self.llm_base_url,
        )
        await self.store.initialize()
        self.memory = MemoryStore(
            working_dir=str(self.memory_dir),
            embedder=self.store.embedder,
            llm_base_url=self.llm_base_url,
        )
        await self.memory.initialize()
        self.client = openai.AsyncOpenAI(base_url=self.llm_base_url, api_key="dummy")
        await self._seed_initial_constitution()

    async def finalize(self) -> None:
        if self.store:
            await self.store.finalize()
        if self.memory:
            await self.memory.finalize()

    async def _seed_initial_constitution(self) -> None:
        assert self.store is not None
        existing = {entry.principle for entry in self.store.list_all()}
        for principle in INITIAL_CONSTITUTION:
            if principle in existing:
                continue
            entry = ConstitutionEntry(
                principle=principle,
                confidence=1.0,
                evidence=["initial_anthropic_constitution_split"],
            )
            await self.store.add(entry)

    async def _generate_one(self, query: str, constitution_text: str, memory_text: str) -> str:
        assert self.client is not None
        resp = await self.client.chat.completions.create(
            model=config.VLLM_MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": GENERATE_SYSTEM.format(
                        constitution=constitution_text,
                        memory=memory_text,
                    ),
                },
                {"role": "user", "content": query},
            ],
            max_tokens=512,
            temperature=0.7,
        )
        return resp.choices[0].message.content or ""

    async def _memory_context(self, query: str, top_k: int = 8) -> str:
        assert self.memory is not None
        if not self.memory.index:
            return "(No memories stored)"
        try:
            return await self.memory.recall(query, top_k=top_k)
        except Exception as exc:
            LOGGER.warning("Memory recall failed, falling back to no-memory context: %s", exc)
            return "(Memory recall failed for this item)"

    async def _reflect_one(
        self,
        stage: str,
        query: str,
        response: str,
        feedback: str,
        constitution_text: str,
        memory_text: str,
    ) -> Dict[str, Any]:
        assert self.client is not None
        prompt = REFLECTION_USER.format(
            constitution=constitution_text,
            memory=memory_text,
            stage=stage,
            query=query,
            response=response,
            feedback=feedback,
        )
        resp = await self.client.chat.completions.create(
            model=config.VLLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": REFLECTION_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            max_tokens=1024,
            temperature=0.2,
        )
        raw = resp.choices[0].message.content or "{}"
        try:
            parsed = json.loads(repair_json(raw))
            if not isinstance(parsed, dict):
                return {"reasoning": raw, "deltas": []}
            if not isinstance(parsed.get("deltas", []), list):
                parsed["deltas"] = []
            return parsed
        except Exception as exc:
            return {"reasoning": f"parse_failed: {exc}; raw={raw}", "deltas": []}

    async def _apply_deltas(self, deltas: Iterable[Dict[str, Any]], interaction_id: str) -> List[str]:
        assert self.store is not None
        assert self.memory is not None
        applied: List[str] = []
        for delta in deltas:
            action = str(delta.get("action", "")).lower()
            try:
                if action == "add":
                    principle = str(delta.get("principle", "")).strip()
                    if not principle:
                        continue
                    entry = ConstitutionEntry(
                        principle=principle,
                        confidence=float(delta.get("confidence", 0.5)),
                        evidence=[str(delta.get("evidence", interaction_id))],
                    )
                    eid = await self.store.add(entry)
                    applied.append(f"ADD {eid}: {principle[:80]}")
                elif action == "update":
                    eid = str(delta.get("id", ""))
                    if not eid or eid not in self.store.index:
                        continue
                    updates: Dict[str, Any] = {}
                    if "principle" in delta:
                        updates["principle"] = str(delta["principle"])
                    if "confidence" in delta:
                        updates["confidence"] = float(delta["confidence"])
                    existing = await self.store.get(eid)
                    if existing:
                        updates["evidence"] = existing.evidence + [
                            str(delta.get("evidence", interaction_id))
                        ]
                    await self.store.update(eid, updates)
                    applied.append(f"UPDATE {eid}")
                elif action == "retire":
                    eid = str(delta.get("id", ""))
                    if eid and eid in self.store.index:
                        await self.store.retire(eid)
                        applied.append(f"RETIRE {eid}")
                elif action == "remember":
                    content = str(delta.get("content", "")).strip()
                    if not content:
                        continue
                    mid = await self.memory.remember(
                        content=content,
                        source=interaction_id,
                        tags=[str(t) for t in delta.get("tags", [])],
                    )
                    applied.append(f"REMEMBER {mid}: {content[:80]}")
                elif action == "forget":
                    mid = str(delta.get("id", ""))
                    if mid and mid in self.memory.index:
                        await self.memory.forget(mid)
                        applied.append(f"FORGET {mid}")
            except Exception as exc:
                LOGGER.warning("Delta failed at %s: %s (%s)", interaction_id, delta, exc)
        return applied

    async def run_stage(self, stage: str, items: List[DatasetItem]) -> Dict[str, Any]:
        assert self.store is not None
        assert self.memory is not None
        total = len(items)
        stage_start = self.global_step
        print(f"\n[{self.backend}:{stage}] {total} items, batch_size={self.batch_size}")

        for batch_start in range(0, total, self.batch_size):
            batch = items[batch_start : batch_start + self.batch_size]
            constitution_text = self.store.summary()
            queries = [item.input_text for item in batch]
            feedbacks = [_format_reference_signal(item) for item in batch]
            memory_contexts = await asyncio.gather(
                *[self._memory_context(query) for query in queries],
                return_exceptions=True,
            )
            memory_texts = [
                "(Memory recall failed for this item)" if isinstance(ctx, Exception) else str(ctx)
                for ctx in memory_contexts
            ]

            gen_tasks = [
                self._generate_one(query, constitution_text, memory_text)
                for query, memory_text in zip(queries, memory_texts)
            ]
            responses = await asyncio.gather(*gen_tasks, return_exceptions=True)

            reflect_tasks = []
            for query, feedback, response in zip(queries, feedbacks, responses):
                if isinstance(response, Exception):
                    async def _failed(exc=response):
                        return {"reasoning": str(exc), "deltas": []}
                    reflect_tasks.append(_failed())
                else:
                    reflect_tasks.append(
                        self._reflect_one(
                            stage,
                            query,
                            str(response),
                            feedback,
                            constitution_text,
                            memory_texts[len(reflect_tasks)],
                        )
                    )
            reflections = await asyncio.gather(*reflect_tasks, return_exceptions=True)

            for item, response, reflection in zip(batch, responses, reflections):
                interaction_id = (
                    f"{self.experiment_id}/{self.backend}/"
                    f"{stage}/step_{self.global_step + 1:05d}"
                )
                if isinstance(reflection, Exception):
                    reflection = {"reasoning": str(reflection), "deltas": []}
                if isinstance(response, Exception):
                    response_text = ""
                    error = str(response)
                    deltas = []
                else:
                    response_text = str(response)
                    error = ""
                    deltas = reflection.get("deltas", [])

                applied = await self._apply_deltas(deltas, interaction_id)
                self.global_step += 1

                record = {
                    "global_step": self.global_step,
                    "stage": stage,
                    "item_id": item.id,
                    "dataset": item.dataset,
                    "category": item.category,
                    "kind": item.kind.value,
                    "response": response_text,
                    "reflection": reflection,
                    "applied": applied,
                }
                if error:
                    record["error"] = error
                self.records.append(record)

                if self.save_every > 0 and self.global_step % self.save_every == 0:
                    self.save_snapshot(f"ckpt_{self.global_step:05d}", stage=stage)
                    self.save_log()

            n_errors = sum(1 for r in self.records[-len(batch):] if "error" in r)
            print(
                f"  [{stage} {batch_start + len(batch):05d}/{total:05d}] "
                f"global={self.global_step:05d} "
                f"constitution={len(self.store.list_active())} "
                f"memory={len(self.memory.index)} "
                f"errors={n_errors}"
            )

        self.save_snapshot(f"{stage}_end", stage=stage)
        self.save_log()
        return {
            "stage": stage,
            "items": total,
            "global_start": stage_start + 1,
            "global_end": self.global_step,
            "constitution": len(self.store.list_active()),
            "memory": len(self.memory.index),
        }

    def save_snapshot(self, suffix: str, stage: str) -> Path:
        assert self.store is not None
        assert self.memory is not None
        run_id = f"{self.experiment_id}/{self.backend}/{suffix}"
        return self.snapshot_mgr.save(
            run_id=run_id,
            constitution_working_dir=str(self.rag_dir),
            memory_working_dir=str(self.memory_dir),
            manifest_extra={
                "experiment_id": self.experiment_id,
                "backend": self.backend,
                "stage": stage,
                "global_step": self.global_step,
                "total_items_processed": len(self.records),
                "active_constitution": len(self.store.list_active()),
                "memory_entries": len(self.memory.index),
            },
        )

    def save_log(self) -> Path:
        path = self.log_dir / "run_log.json"
        with open(path, "w") as f:
            json.dump(self.records, f, indent=2, ensure_ascii=False, default=str)
        return path


async def amain(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    data_root = str(Path(args.data_root).resolve())

    if args.dry_run:
        for stage in args.stages:
            items = build_stage_items(stage, args.items_per_stage, data_root, args.seed)
            by_dataset: Dict[str, int] = {}
            for item in items:
                by_dataset[item.dataset] = by_dataset.get(item.dataset, 0) + 1
            if len(items) < args.items_per_stage:
                print(
                    f"[warn] stage {stage} requested {args.items_per_stage}, "
                    f"available {len(items)} after filtering"
                )
            print(f"{stage}: {len(items)} items {by_dataset}")
        return

    runner = ConstitutionDevelopmentRunner(
        experiment_id=args.experiment_id,
        backend=args.backend,
        llm_base_url=args.llm_url,
        output_root=output_root,
        batch_size=args.batch_size,
        save_every=args.save_every,
    )
    await runner.initialize()
    stage_summaries = []
    try:
        for stage in args.stages:
            items = build_stage_items(stage, args.items_per_stage, data_root, args.seed)
            if len(items) < args.items_per_stage:
                print(
                    f"[warn] stage {stage} requested {args.items_per_stage}, "
                    f"available {len(items)} after filtering"
                )
            stage_summaries.append(await runner.run_stage(stage, items))
        runner.save_snapshot("final", stage=args.stages[-1])
    finally:
        await runner.finalize()

    summary = {
        "experiment_id": args.experiment_id,
        "backend": args.backend,
        "items_per_stage": args.items_per_stage,
        "stages": args.stages,
        "stage_summaries": stage_summaries,
        "created_at": datetime.now().isoformat(),
    }
    summary_path = output_root / "logs" / args.experiment_id / args.backend / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSummary: {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", default=f"constitution_dev_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--backend", choices=["8b", "72b"], default=os.environ.get("CONSTITUTION_BACKEND", "8b"))
    parser.add_argument("--llm-url", default=os.environ.get("POSITION_VLLM_URL", "http://localhost:8198/v1"))
    parser.add_argument("--data-root", default=str(REPO_ROOT / "dataset"))
    parser.add_argument("--output-root", default=str(REPO_ROOT / "constitution_development"))
    parser.add_argument("--items-per-stage", type=int, default=10_000)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=STAGES)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    asyncio.run(amain(parse_args()))


if __name__ == "__main__":
    main()
