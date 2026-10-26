"""LLM client wrappers.

Two backends:
  - OpenAIChatLLM:  hits a chat-completions endpoint (vLLM-compatible)
  - StubLLM:        deterministic canned responses for smoke tests

Both expose the same `complete(messages, ...)` interface returning a
plain string.
"""

from __future__ import annotations

import json
import re
from typing import Callable, List, Optional

try:
    import openai  # noqa: F401
    _HAS_OPENAI = True
except ImportError:
    _HAS_OPENAI = False


def parse_json_lenient(text: str) -> dict:
    """Best-effort JSON extraction from an LLM response."""
    # Try direct parse.
    try:
        return json.loads(text)
    except Exception:
        pass
    # Strip code fences.
    cleaned = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    # Find the first balanced {...} block.
    start = cleaned.find("{")
    if start >= 0:
        depth = 0
        for i in range(start, len(cleaned)):
            if cleaned[i] == "{":
                depth += 1
            elif cleaned[i] == "}":
                depth -= 1
                if depth == 0:
                    snippet = cleaned[start : i + 1]
                    try:
                        return json.loads(snippet)
                    except Exception:
                        break
    # Try the project's json_repair if installed.
    try:
        from json_repair import repair_json  # type: ignore
        return json.loads(repair_json(text))
    except Exception:
        return {}


class OpenAIChatLLM:
    def __init__(self, base_url: str, model: str, api_key: str = "dummy",
                 temperature: float = 0.7, max_tokens: int = 256):
        if not _HAS_OPENAI:
            raise RuntimeError("openai package not available")
        self.client = openai.OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def complete(self, messages: List[dict], temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=self.max_tokens if max_tokens is None else max_tokens,
        )
        return resp.choices[0].message.content or ""


class StubLLM:
    """Deterministic stub.

    `policy_fn(messages, kind)` -> str, where `kind` is "action" or "reflect".
    Defaults to always-cooperate / no-deltas.
    """

    def __init__(self, policy_fn: Optional[Callable[[List[dict], str], str]] = None,
                 default_action: str = "Cooperate"):
        self.policy_fn = policy_fn
        self.default_action = default_action

    def _classify(self, messages: List[dict]) -> str:
        sys = next((m["content"] for m in messages if m["role"] == "system"), "")
        return "reflect" if "CONSTITUTION" in sys and "deltas" in sys else "action"

    def complete(self, messages: List[dict], temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None) -> str:
        kind = self._classify(messages)
        if self.policy_fn is not None:
            return self.policy_fn(messages, kind)
        if kind == "reflect":
            return json.dumps({"reasoning": "stub: no change", "deltas": []})
        return json.dumps({"reasoning": "stub", "action": self.default_action})
