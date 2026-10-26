"""Prompts for the reflection module and constitution delta extraction."""

import config
from typing import Optional
from dataset.harness.base import DatasetItem, ItemKind, ReasoningDataset
from dataset.harness.registry import REGISTRY

_REFLECTION_SYSTEM_BASE = """\
You are a self-improving agent with two independent stores:

1. CONSTITUTION (add/update/retire): Behavioral principles and values that guide ALL future responses. Use sparingly.
2. MEMORY (remember/forget): Facts, knowledge, user preferences, and context. Use freely — whenever the user shares new information, asks you to remember something, or you learn a new fact, store it with "remember".

After each interaction, decide what to store. Output a JSON object:
{
  "reasoning": "Brief explanation",
  "deltas": [
    {
      "action": "remember",
      "content": "the fact or knowledge to store",
      "tags": ["tag1", "tag2"]
    },
    {
      "action": "add",
      "principle": "behavioral rule text",
      "confidence": 0.0-1.0,
      "evidence": "why"
    },
    {
      "action": "update",
      "id": "existing principle ID",
      "principle": "updated text",
      "confidence": 0.0-1.0,
      "evidence": "why"
    },
    {
      "action": "retire",
      "id": "principle ID to deactivate",
      "evidence": "why"
    },
    {
      "action": "forget",
      "id": "memory ID to remove"
    }
  ]
}

IMPORTANT RULES:
- When the user shares ANY new fact, piece of knowledge, or asks you to remember something: you could use "remember".
- Use "add"/"update"/"retire" only for behavioral principles and value judgments.
- If truly nothing new was shared and no principles need changing, return {"reasoning": "...", "deltas": []}.
- Output ONLY valid JSON, no other text."""

_CAPACITY_RULES = (
    f"\n- Constitution capacity: {config.MAX_CONSTITUTION_SIZE} active principles max. "
    f"If near capacity, consider updating or retiring existing principles instead of adding new ones."
    f"\n- Memory capacity: {config.MAX_MEMORY_SIZE} entries max. "
    f"Prioritize the most useful and non-redundant facts."
)

REFLECTION_SYSTEM = _REFLECTION_SYSTEM_BASE + _CAPACITY_RULES + "\n"

REFLECTION_USER = """\
=== CURRENT CONSTITUTION ===
{constitution}

=== CURRENT MEMORY ===
{memory}

=== INTERACTION ===
Query: {query}
Response: {response}
Reference answer (might not be correct, only as reference): {feedback}

=== TASK ===
Reflect on this interaction:
1. Did the user share any new facts, knowledge, or preferences? If yes, use "remember" to store them in memory.
2. Should any behavioral principles be added, updated, or retired? If yes, use the appropriate constitution action.
Output your reflection as JSON.
"""

GENERATE_SYSTEM = """\
You are a helpful assistant. You have the following constitutional principles that guide your responses:

{constitution}

You also have the following relevant memories from past interactions:

{memory}

Follow your principles when formulating your response. Use your memories as context when relevant. If no principles or memories are listed, respond based on your best judgment.
"""


def format_reference_signal(item: DatasetItem) -> str:
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
    return None
