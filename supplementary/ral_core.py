"""Minimal Reflective Autonomy Learning mechanism.

This file omits model serving, dataset loading, evaluation, checkpointing,
and all experiment-specific paths. A caller supplies a backend with two
methods:

    generate(system: str, user: str) -> str
    reflect(system: str, user: str) -> str  # returns JSON-like text

The core mechanism is:
    current constitution + memory -> generate response
    interaction + reference signal -> reflect
    JSON deltas -> update constitution and memory
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
import json
import uuid


@dataclass
class ConstitutionEntry:
    principle: str
    confidence: float = 0.5
    evidence: list[str] = field(default_factory=list)
    version: int = 1
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    active: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


class ConstitutionStore:
    def __init__(self, initial_principles: list[str] | None = None):
        self.entries: dict[str, ConstitutionEntry] = {}
        for principle in initial_principles or []:
            self.add(principle, confidence=1.0, evidence="initial")

    def add(self, principle: str, confidence: float = 0.5, evidence: str = "") -> str:
        entry = ConstitutionEntry(principle, confidence, [evidence] if evidence else [])
        self.entries[entry.id] = entry
        return entry.id

    def update(self, entry_id: str, principle: str, confidence: float, evidence: str = "") -> None:
        entry = self.entries[entry_id]
        entry.principle = principle
        entry.confidence = confidence
        entry.version += 1
        if evidence:
            entry.evidence.append(evidence)

    def retire(self, entry_id: str, evidence: str = "") -> None:
        entry = self.entries[entry_id]
        entry.active = False
        if evidence:
            entry.evidence.append(evidence)

    def summary(self) -> str:
        active = [e for e in self.entries.values() if e.active]
        if not active:
            return "(no active principles)"
        return "\n".join(
            f"[{e.id} v{e.version} conf={e.confidence:.2f}] {e.principle}"
            for e in active
        )


class MemoryStore:
    def __init__(self):
        self.items: dict[str, dict] = {}

    def remember(self, content: str, tags: list[str] | None = None) -> str:
        memory_id = uuid.uuid4().hex[:8]
        self.items[memory_id] = {"content": content, "tags": tags or []}
        return memory_id

    def forget(self, memory_id: str) -> None:
        self.items.pop(memory_id, None)

    def recall(self, query: str, limit: int = 8) -> str:
        del query  # Minimal sketch: real implementation may use retrieval.
        selected = list(self.items.values())[:limit]
        if not selected:
            return "(no relevant memories)"
        return "\n".join(f"- {m['content']}" for m in selected)


GENERATE_SYSTEM = """You are an assistant guided by your own constitution.

Constitution:
{constitution}

Relevant memory:
{memory}
"""

REFLECT_SYSTEM = """You revise two stores after each interaction.

Use CONSTITUTION only for durable behavioral principles.
Use MEMORY for facts, examples, task observations, and local context.

Return JSON:
{
  "reasoning": "...",
  "deltas": [
    {"action": "add", "principle": "...", "confidence": 0.8, "evidence": "..."},
    {"action": "update", "id": "...", "principle": "...", "confidence": 0.8, "evidence": "..."},
    {"action": "retire", "id": "...", "evidence": "..."},
    {"action": "remember", "content": "...", "tags": ["..."]},
    {"action": "forget", "id": "..."}
  ]
}
"""

REFLECT_USER = """Current constitution:
{constitution}

Current memory:
{memory}

Query:
{query}

Response:
{response}

Reference signal:
{reference}

Decide whether to modify constitution or memory.
"""


class RALAgent:
    def __init__(self, backend, initial_principles: list[str] | None = None):
        self.backend = backend
        self.constitution = ConstitutionStore(initial_principles)
        self.memory = MemoryStore()
        self.trace: list[dict] = []

    def generate(self, query: str) -> str:
        system = GENERATE_SYSTEM.format(
            constitution=self.constitution.summary(),
            memory=self.memory.recall(query),
        )
        return self.backend.generate(system=system, user=query)

    def reflect(self, query: str, response: str, reference: str) -> dict:
        user = REFLECT_USER.format(
            constitution=self.constitution.summary(),
            memory=self.memory.recall(query),
            query=query,
            response=response,
            reference=reference,
        )
        raw = self.backend.reflect(system=REFLECT_SYSTEM, user=user)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"reasoning": raw, "deltas": []}

    def apply(self, reflection: dict) -> list[str]:
        applied: list[str] = []
        for delta in reflection.get("deltas", []):
            action = delta.get("action")
            if action == "add":
                entry_id = self.constitution.add(
                    delta["principle"],
                    float(delta.get("confidence", 0.5)),
                    delta.get("evidence", ""),
                )
                applied.append(f"add:{entry_id}")
            elif action == "update" and delta.get("id") in self.constitution.entries:
                self.constitution.update(
                    delta["id"],
                    delta["principle"],
                    float(delta.get("confidence", 0.5)),
                    delta.get("evidence", ""),
                )
                applied.append(f"update:{delta['id']}")
            elif action == "retire" and delta.get("id") in self.constitution.entries:
                self.constitution.retire(delta["id"], delta.get("evidence", ""))
                applied.append(f"retire:{delta['id']}")
            elif action == "remember":
                memory_id = self.memory.remember(delta.get("content", ""), delta.get("tags", []))
                applied.append(f"remember:{memory_id}")
            elif action == "forget":
                self.memory.forget(delta.get("id", ""))
                applied.append(f"forget:{delta.get('id', '')}")
        return applied

    def step(self, query: str, reference: str = "") -> dict:
        response = self.generate(query)
        reflection = self.reflect(query, response, reference)
        applied = self.apply(reflection)
        record = {
            "query": query,
            "response": response,
            "reference": reference,
            "reflection": reflection,
            "applied": applied,
            "constitution": self.constitution.summary(),
        }
        self.trace.append(record)
        return record
