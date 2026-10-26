"""Multi-user portal for the self-modifying constitution system."""

import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from constitution.store import ConstitutionStore
from constitution.memory_store import MemoryStore
from pipeline.orchestrator import Orchestrator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SERVER_RAG_DIR = str(Path(config.RAG_WORKING_DIR).parent / "rag_storage_server")
SERVER_MEMORY_DIR = str(Path(config.MEMORY_WORKING_DIR).parent / "memory_storage_server")


# --- Pydantic models ---

class InteractionRequest(BaseModel):
    query: str
    feedback: str = ""
    user_id: str = "anonymous"


class PrincipleOut(BaseModel):
    id: str
    principle: str
    confidence: float
    version: int
    active: bool
    evidence_count: int


class InteractionResponse(BaseModel):
    id: str
    query: str
    response: str
    reflection: dict
    applied: list[str]
    constitution_snapshot: str
    user_id: str


# --- App state ---

store: ConstitutionStore = None
memory: MemoryStore = None
orchestrator: Orchestrator = None
constitution_lock: asyncio.Lock = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global store, memory, orchestrator, constitution_lock
    constitution_lock = asyncio.Lock()

    store = ConstitutionStore(working_dir=SERVER_RAG_DIR)
    await store.initialize()
    # Pre-load embedding model so first request isn't slow
    logger.info("Pre-loading embedding model (this may take a minute)...")
    store.embedder.load()
    logger.info("Embedding model loaded.")

    # Initialize independent memory store (shares embedder with constitution)
    memory = MemoryStore(
        working_dir=SERVER_MEMORY_DIR,
        embedder=store.embedder,
    )
    await memory.initialize()
    logger.info(f"Memory store initialized with {len(memory.index)} memories.")

    orchestrator = Orchestrator(store, memory=memory)
    logger.info(
        f"Server ready. Constitution: {len(store.list_active())} principles, "
        f"Memory: {len(memory.index)} entries."
    )
    yield

    # Cleanup
    store.embedder.unload()
    await memory.finalize()
    await store.finalize()
    logger.info("Server shut down.")


app = FastAPI(title="Constitution Self-Modification Portal", lifespan=lifespan)

static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
async def index():
    return FileResponse(str(static_dir / "index.html"))


@app.get("/api/health")
async def health():
    vllm_ok = False
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(f"{config.VLLM_BASE_URL}/models", timeout=3)
            vllm_ok = r.status_code == 200
    except Exception:
        pass
    return {
        "status": "ok",
        "vllm_reachable": vllm_ok,
        "active_principles": len(store.list_active()),
        "total_memories": len(memory.index),
        "total_interactions": len(orchestrator.history),
        "lock_held": constitution_lock.locked(),
    }


@app.post("/api/interact", response_model=InteractionResponse)
async def interact(req: InteractionRequest):
    if not req.query.strip():
        raise HTTPException(400, "Query cannot be empty")

    async with constitution_lock:
        result = await orchestrator.step(query=req.query, feedback=req.feedback)
    result["user_id"] = req.user_id

    return InteractionResponse(
        id=result["id"],
        query=result["query"],
        response=result["response"],
        reflection=result["reflection"],
        applied=result["applied"],
        constitution_snapshot=result["constitution_snapshot"],
        user_id=req.user_id,
    )


@app.get("/api/constitution")
async def get_constitution():
    active = store.list_active()
    all_entries = store.list_all()
    return {
        "summary": store.summary(),
        "total_active": len(active),
        "total_all": len(all_entries),
        "principles": [
            PrincipleOut(
                id=e.id,
                principle=e.principle,
                confidence=e.confidence,
                version=e.version,
                active=e.active,
                evidence_count=len(e.evidence),
            ).model_dump()
            for e in sorted(active, key=lambda x: x.confidence, reverse=True)
        ],
    }


@app.get("/api/memory")
async def get_memory():
    entries = memory.list_all()
    return {
        "summary": memory.summary(),
        "total": len(entries),
        "memories": [e.to_dict() for e in entries],
    }


@app.get("/api/memory/recall")
async def recall_memory(q: str = ""):
    if not q.strip():
        raise HTTPException(400, "Query parameter 'q' is required")
    result = await memory.recall(q)
    return {"query": q, "result": result}


@app.get("/api/history")
async def get_history():
    return {
        "total": len(orchestrator.history),
        "interactions": orchestrator.history,
    }
