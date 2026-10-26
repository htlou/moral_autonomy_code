"""Orchestrator: the core loop for self-modifying constitution via LLM reflection."""

import asyncio
import json
import logging
from datetime import datetime
from typing import Optional

import openai
from json_repair import repair_json

from constitution.schema import ConstitutionEntry
from constitution.store import ConstitutionStore
from constitution.memory_store import MemoryStore
from constitution.prompts import REFLECTION_SYSTEM, REFLECTION_USER, GENERATE_SYSTEM
import config

logger = logging.getLogger(__name__)


class Orchestrator:
    """Runs the generate → reflect → update loop."""

    def __init__(self, store: ConstitutionStore, memory: MemoryStore = None, llm_base_url: str = None):
        self.store = store
        self.memory = memory
        self.client = openai.AsyncOpenAI(
            base_url=llm_base_url or config.VLLM_BASE_URL,
            api_key="dummy",
        )
        self.history: list[dict] = []  # interaction log

    async def generate(self, query: str) -> str:
        """Generate a response using the current constitution + memory as context."""
        constitution_text = self.store.summary()

        # Use recall (RAG query) to fetch only relevant memories instead of all.
        memory_text = (await self.memory.recall(query, top_k=10)) if self.memory else "(No memories stored)"

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

    async def reflect(self, query: str, response: str, feedback: str) -> dict:
        """Ask the LLM to reflect on the interaction and propose changes to constitution and/or memory."""
        constitution_text = self.store.summary()
        memory_text = (await self.memory.recall(query, top_k=10)) if self.memory else "(No memory store)"
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

        # Parse JSON (with repair for common LLM formatting issues)
        try:
            result = json.loads(repair_json(raw))
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Failed to parse reflection JSON: {e}\nRaw: {raw}")
            result = {"reasoning": raw, "deltas": []}

        return result

    async def apply_deltas(self, deltas: list[dict], interaction_id: str) -> list[str]:
        """Apply constitution modifications proposed by the LLM."""
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
                    logger.info(f"Added principle {eid}")

                elif action == "update":
                    eid = delta.get("id")
                    if not eid or eid not in self.store.index:
                        logger.warning(f"Update skipped: ID {eid} not found")
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
                    logger.info(f"Updated principle {eid}")

                elif action == "retire":
                    eid = delta.get("id")
                    if not eid or eid not in self.store.index:
                        logger.warning(f"Retire skipped: ID {eid} not found")
                        continue
                    await self.store.retire(eid)
                    applied.append(f"RETIRE {eid}")
                    logger.info(f"Retired principle {eid}")

                elif action == "remember":
                    if not self.memory:
                        logger.warning("Remember skipped: no memory store")
                        continue
                    content = delta.get("content", "")
                    if not content:
                        continue
                    tags = delta.get("tags", [])
                    mid = await self.memory.remember(
                        content=content,
                        source=interaction_id,
                        tags=tags,
                    )
                    applied.append(f"REMEMBER {mid}: {content[:60]}...")
                    logger.info(f"Stored memory {mid}")

                elif action == "forget":
                    if not self.memory:
                        logger.warning("Forget skipped: no memory store")
                        continue
                    mid = delta.get("id", "")
                    if not mid or mid not in self.memory.index:
                        logger.warning(f"Forget skipped: memory ID {mid} not found")
                        continue
                    await self.memory.forget(mid)
                    applied.append(f"FORGET {mid}")
                    logger.info(f"Forgot memory {mid}")

            except Exception as e:
                logger.error(f"Failed to apply delta {delta}: {e}")

        return applied

    async def step(self, query: str, feedback: str = "") -> dict:
        """Run one full cycle: generate → reflect → apply deltas."""
        interaction_id = f"step_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # 1. Generate response with current constitution
        response = await self.generate(query)

        # 2. Reflect on the interaction
        reflection = await self.reflect(query, response, feedback)

        # 3. Apply proposed deltas
        deltas = reflection.get("deltas", [])
        applied = await self.apply_deltas(deltas, interaction_id)

        # Log
        record = {
            "id": interaction_id,
            "query": query,
            "response": response,
            "feedback": feedback,
            "reflection": reflection,
            "applied": applied,
            "constitution_snapshot": self.store.summary(),
        }
        self.history.append(record)

        return record

    async def run_batch(self, items: list[dict], verbose: bool = True) -> list[dict]:
        """Run the loop over a batch of items.

        Each item should have: {"query": str, "feedback": str}
        """
        results = []
        for i, item in enumerate(items):
            if verbose:
                print(f"\n{'='*60}")
                print(f"Step {i+1}/{len(items)}")
                print(f"Query: {item['query'][:80]}...")

            result = await self.step(
                query=item["query"],
                feedback=item.get("feedback", ""),
            )

            if verbose:
                print(f"Response: {result['response'][:120]}...")
                print(f"Reflection: {result['reflection'].get('reasoning', '')[:120]}...")
                print(f"Applied: {result['applied']}")
                print(f"Active principles: {len(self.store.list_active())}")

            results.append(result)

        return results
