"""Constitution CRUD layer over LightRAG + local JSON index."""

import asyncio
import json
import logging
import os

logger = logging.getLogger(__name__)
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc

from constitution.schema import ConstitutionEntry
import config


class ModelEmbedder:
    """Loads a causal LM and extracts mean-pooled hidden states as embeddings."""

    def __init__(self, model_path: str = None, device: str = None):
        self.model_path = model_path or config.EMBEDDING_MODEL_PATH
        self.device = device or config.EMBEDDING_DEVICE
        self.tokenizer = None
        self.model = None

    def load(self):
        if self.model is not None:
            return
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModel.from_pretrained(
            self.model_path,
            dtype=torch.float32 if self.device == "cpu" else torch.float16,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def embed(self, texts: list[str]) -> np.ndarray:
        self.load()
        all_embeddings = []
        for i in range(0, len(texts), config.EMBEDDING_BATCH_SIZE):
            batch = texts[i : i + config.EMBEDDING_BATCH_SIZE]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=config.EMBEDDING_MAX_LENGTH,
                return_tensors="pt",
            ).to(self.device)
            outputs = self.model(**inputs)
            # Mean pool over non-padding tokens
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            hidden = outputs.last_hidden_state * mask
            pooled = hidden.sum(dim=1) / mask.sum(dim=1)
            # Normalize
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            all_embeddings.append(pooled.cpu().float().numpy())
        return np.concatenate(all_embeddings, axis=0)

    def unload(self):
        """Free GPU memory so vLLM can use it."""
        if self.model is not None:
            del self.model
            self.model = None
            torch.cuda.empty_cache()


class ConstitutionStore:
    """Manages constitution entries with LightRAG for semantic search
    and a local JSON index for fast ID-based CRUD."""

    def __init__(
        self,
        working_dir: str = None,
        llm_base_url: str = None,
        embedding_model_path: str = None,
    ):
        self.working_dir = working_dir or config.RAG_WORKING_DIR
        self.index_path = os.path.join(self.working_dir, "constitution_index.json")
        os.makedirs(self.working_dir, exist_ok=True)

        self.llm_base_url = llm_base_url or config.VLLM_BASE_URL
        self.embedder = ModelEmbedder(model_path=embedding_model_path)
        self.index: dict[str, dict] = {}
        self.rag: Optional[LightRAG] = None

    def _load_index(self):
        if os.path.exists(self.index_path):
            with open(self.index_path, "r") as f:
                self.index = json.load(f)
        else:
            self.index = {}

    def _save_index(self):
        with open(self.index_path, "w") as f:
            json.dump(self.index, f, indent=2, ensure_ascii=False)

    async def initialize(self):
        """Initialize LightRAG and load index."""
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
        # This skips the per-insert LLM calls that extract KG entities.
        async def _noop_extract(*args, **kwargs):
            return []
        self.rag._process_extract_entities = _noop_extract

    async def finalize(self):
        if self.rag:
            await self.rag.finalize_storages()

    async def _llm_func(self, prompt, **kwargs) -> str:
        """Call vLLM via OpenAI-compatible API."""
        import openai

        client = openai.AsyncOpenAI(
            base_url=self.llm_base_url, api_key="dummy"
        )
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
        """Use the local model for embeddings via mean-pooled hidden states."""
        return self.embedder.embed(texts)

    # === CRUD Operations ===

    async def add(self, entry: ConstitutionEntry) -> str:
        """Insert a new constitution entry, pruning lowest-confidence if at capacity."""
        # Enforce max constitution size
        max_size = getattr(config, "MAX_CONSTITUTION_SIZE", 0)
        if max_size > 0:
            active = self.list_active()
            if len(active) >= max_size:
                # Prune lowest-confidence active principle
                worst = min(active, key=lambda e: e.confidence)
                logger.info(
                    f"Constitution at capacity ({max_size}), retiring "
                    f"{worst.id} (conf={worst.confidence:.2f})"
                )
                await self.retire(worst.id)

        doc_text = entry.to_document()
        await self.rag.ainsert(doc_text, ids=[entry.id])
        self.index[entry.id] = entry.to_dict()
        self._save_index()
        return entry.id

    async def get(self, entry_id: str) -> Optional[ConstitutionEntry]:
        """Get a constitution entry by ID."""
        data = self.index.get(entry_id)
        if data:
            return ConstitutionEntry.from_dict(data)
        return None

    async def update(self, entry_id: str, updates: dict) -> Optional[ConstitutionEntry]:
        """Update an existing constitution entry."""
        if entry_id not in self.index:
            return None

        old_entry = ConstitutionEntry.from_dict(self.index[entry_id])

        # Create new version
        new_data = old_entry.to_dict()
        new_data.update(updates)
        new_data["version"] = old_entry.version + 1
        new_data["supersedes"] = entry_id

        new_entry = ConstitutionEntry.from_dict(new_data)
        new_entry.id = entry_id  # keep same ID

        # Delete old doc from RAG and insert updated
        try:
            await self.rag.adelete_by_doc_id(entry_id)
        except Exception as e:
            logger.warning(f"RAG delete failed for {entry_id}: {e}")
        await self.rag.ainsert(new_entry.to_document(), ids=[entry_id])

        self.index[entry_id] = new_entry.to_dict()
        self._save_index()
        return new_entry

    async def retire(self, entry_id: str) -> bool:
        """Mark a constitution entry as inactive."""
        if entry_id not in self.index:
            return False
        self.index[entry_id]["active"] = False
        self._save_index()
        try:
            await self.rag.adelete_by_doc_id(entry_id)
        except Exception as e:
            logger.warning(f"RAG delete failed for {entry_id} (retire): {e}")
        return True

    async def delete(self, entry_id: str) -> bool:
        """Permanently delete a constitution entry."""
        if entry_id not in self.index:
            return False
        del self.index[entry_id]
        self._save_index()
        try:
            await self.rag.adelete_by_doc_id(entry_id)
        except Exception as e:
            logger.warning(f"RAG delete failed for {entry_id} (delete): {e}")
        return True

    async def query(self, question: str, top_k: int = 5) -> str:
        """Query the constitution for relevant principles."""
        result = await self.rag.aquery(
            question,
            param=QueryParam(
                mode=getattr(config, "RAG_QUERY_MODE", "naive"),
                top_k=top_k,
                only_need_context=True,
            ),
        )
        return result

    def list_active(self) -> list[ConstitutionEntry]:
        """List all active constitution entries."""
        return [
            ConstitutionEntry.from_dict(d)
            for d in self.index.values()
            if d.get("active", True)
        ]

    def list_all(self) -> list[ConstitutionEntry]:
        """List all constitution entries."""
        return [ConstitutionEntry.from_dict(d) for d in self.index.values()]

    def summary(self) -> str:
        """Return a human-readable summary of the constitution."""
        active = self.list_active()
        if not active:
            return "(No active constitution principles)"
        lines = []
        for e in sorted(active, key=lambda x: x.confidence, reverse=True):
            lines.append(
                f"[{e.id} v{e.version} conf={e.confidence:.2f}] {e.principle}"
            )
        return "\n".join(lines)
