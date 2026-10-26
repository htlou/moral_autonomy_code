"""Independent RAG memory store — decoupled from constitution principles.

Stores factual knowledge, user preferences, conversation context, etc.
The model can read/write here without touching the constitution index.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional
import uuid

import numpy as np
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc

import config

logger = logging.getLogger(__name__)


class MemoryEntry:
    """Lightweight memory record (no constitution overhead)."""

    def __init__(
        self,
        content: str,
        source: str = "",
        tags: list[str] = None,
        memory_id: str = None,
        created_at: str = None,
    ):
        self.id = memory_id or str(uuid.uuid4())[:8]
        self.content = content
        self.source = source  # e.g. "step_20260401_120000"
        self.tags = tags or []
        self.created_at = created_at or datetime.now().isoformat()

    def to_document(self) -> str:
        tag_str = ", ".join(self.tags) if self.tags else "general"
        return (
            f"[Memory {self.id}] ({tag_str})\n"
            f"{self.content}\n"
            f"[Source: {self.source} | Created: {self.created_at}]"
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content": self.content,
            "source": self.source,
            "tags": self.tags,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MemoryEntry":
        return cls(
            content=d["content"],
            source=d.get("source", ""),
            tags=d.get("tags", []),
            memory_id=d.get("id"),
            created_at=d.get("created_at"),
        )


class MemoryStore:
    """Standalone RAG store for non-constitutional knowledge.

    Shares the same embedder instance as ConstitutionStore (passed in)
    to avoid loading a second copy of the model.
    """

    def __init__(self, working_dir: str, embedder, llm_base_url: str = None):
        self.working_dir = working_dir
        self.index_path = os.path.join(working_dir, "memory_index.json")
        os.makedirs(working_dir, exist_ok=True)

        self.embedder = embedder  # shared ModelEmbedder
        self.llm_base_url = llm_base_url or config.VLLM_BASE_URL
        self.index: dict[str, dict] = {}
        self.rag: Optional[LightRAG] = None

    # --- persistence ---

    def _load_index(self):
        if os.path.exists(self.index_path):
            with open(self.index_path, "r") as f:
                self.index = json.load(f)
        else:
            self.index = {}

    def _save_index(self):
        with open(self.index_path, "w") as f:
            json.dump(self.index, f, indent=2, ensure_ascii=False)

    # --- init ---

    async def initialize(self):
        self._load_index()

        self.rag = LightRAG(
            working_dir=self.working_dir,
            llm_model_func=self._llm_func,
            llm_model_name=config.VLLM_MODEL_NAME,
            embedding_func=EmbeddingFunc(
                embedding_dim=config.EMBEDDING_DIM,
                max_token_size=8192,
                func=self._embedding_func,
            ),
        )
        await self.rag.initialize_storages()

        # Disable entity/relationship extraction — we only need vector search.
        async def _noop_extract(*args, **kwargs):
            return []
        self.rag._process_extract_entities = _noop_extract

        logger.info(f"MemoryStore initialized with {len(self.index)} memories.")

    async def finalize(self):
        if self.rag:
            await self.rag.finalize_storages()

    async def _llm_func(self, prompt, **kwargs) -> str:
        import openai

        client = openai.AsyncOpenAI(base_url=self.llm_base_url, api_key="dummy")
        messages = [{"role": "user", "content": prompt}]
        if "system_prompt" in kwargs and kwargs["system_prompt"]:
            messages.insert(0, {"role": "system", "content": kwargs["system_prompt"]})
        resp = await client.chat.completions.create(
            model=config.VLLM_MODEL_NAME,
            messages=messages,
            max_tokens=kwargs.get("max_tokens", 512),
            temperature=kwargs.get("temperature", 0.0),
        )
        return resp.choices[0].message.content

    async def _embedding_func(self, texts: list[str]) -> np.ndarray:
        return self.embedder.embed(texts)

    # --- CRUD ---

    async def remember(self, content: str, source: str = "", tags: list[str] = None) -> str:
        """Store a new memory, pruning oldest if at capacity. Returns memory ID."""
        # Enforce max memory size
        max_size = getattr(config, "MAX_MEMORY_SIZE", 0)
        if max_size > 0 and len(self.index) >= max_size:
            # Prune oldest memory
            oldest_id = min(
                self.index,
                key=lambda mid: self.index[mid].get("created_at", ""),
            )
            logger.info(f"Memory at capacity ({max_size}), forgetting oldest: {oldest_id}")
            await self.forget(oldest_id)

        entry = MemoryEntry(content=content, source=source, tags=tags)
        doc = entry.to_document()
        await self.rag.ainsert(doc, ids=[entry.id])
        self.index[entry.id] = entry.to_dict()
        self._save_index()
        logger.info(f"Stored memory {entry.id}: {content[:60]}...")
        return entry.id

    async def forget(self, memory_id: str) -> bool:
        """Remove a memory by ID."""
        if memory_id not in self.index:
            return False
        try:
            await self.rag.adelete_by_doc_id(memory_id)
        except Exception as e:
            logger.warning(f"RAG delete failed for memory {memory_id}: {e}")
        del self.index[memory_id]
        self._save_index()
        logger.info(f"Forgot memory {memory_id}")
        return True

    async def recall(self, question: str, top_k: int = 5) -> str:
        """Query the memory store for relevant knowledge."""
        if not self.index:
            return "(No memories stored)"
        result = await self.rag.aquery(
            question,
            param=QueryParam(
                mode=getattr(config, "RAG_QUERY_MODE", "naive"),
                top_k=top_k,
                only_need_context=True,
            ),
        )
        return result

    def list_all(self) -> list[MemoryEntry]:
        return [MemoryEntry.from_dict(d) for d in self.index.values()]

    def summary(self) -> str:
        if not self.index:
            return "(No memories stored)"
        lines = []
        for mid, d in self.index.items():
            tags = ", ".join(d.get("tags", [])) or "general"
            lines.append(f"[{mid}] ({tags}) {d['content'][:80]}")
        return "\n".join(lines)
