"""Gradio demo for the constitution self-modification system.

Launch:
    python demo.py              # assumes vLLM already running on port 8198
    python demo.py --share      # also create a public Gradio share link
"""

import argparse
import asyncio
import logging
import shutil
from pathlib import Path

import gradio as gr
import httpx

import config
from constitution.store import ConstitutionStore, ModelEmbedder
from constitution.memory_store import MemoryStore
from pipeline.orchestrator import Orchestrator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPERIMENT_ID = "rexp_20260413_201509"
BASE_DIR = Path(__file__).resolve().parent
DEMO_STORAGE = BASE_DIR / "demo_storage"

CONDITIONS = {
    "wr_neg":  "With Reasoning + Negative Value",
    "wr_pos":  "With Reasoning + Positive Value",
    "wr_neu":  "With Reasoning + Neutral Value",
    "wor_neg": "Without Reasoning + Negative Value",
    "wor_pos": "Without Reasoning + Positive Value",
    "wor_neu": "Without Reasoning + Neutral Value",
    "blank":   "Empty (No Constitution / Memory)",
}

# ---------------------------------------------------------------------------
# Startup: copy experiment states
# ---------------------------------------------------------------------------

def copy_experiment_states():
    """Copy experiment snapshots into demo_storage/ so originals are untouched."""
    src_memory = BASE_DIR / f"memory_storage_{EXPERIMENT_ID}"
    src_rag = BASE_DIR / f"rag_storage_{EXPERIMENT_ID}"

    for cond in list(CONDITIONS.keys()):
        if cond == "blank":
            # Create empty dirs
            (DEMO_STORAGE / "memory" / "blank").mkdir(parents=True, exist_ok=True)
            (DEMO_STORAGE / "rag" / "blank").mkdir(parents=True, exist_ok=True)
            continue

        dst_mem = DEMO_STORAGE / "memory" / cond
        dst_rag = DEMO_STORAGE / "rag" / cond

        if not dst_mem.exists():
            logger.info(f"Copying memory state: {cond}")
            shutil.copytree(src_memory / cond, dst_mem)
        if not dst_rag.exists():
            logger.info(f"Copying rag state: {cond}")
            shutil.copytree(src_rag / cond, dst_rag)

    logger.info("Demo storage ready.")


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------

embedder: ModelEmbedder = None
current_condition: str = None
current_store: ConstitutionStore = None
current_memory: MemoryStore = None
current_orchestrator: Orchestrator = None
llm_lock = asyncio.Lock()


async def load_condition(cond_key: str):
    """Load constitution + memory stores for the given condition."""
    global current_condition, current_store, current_memory, current_orchestrator

    # Finalize previous stores
    if current_store:
        try:
            await current_store.finalize()
        except Exception:
            pass
    if current_memory:
        try:
            await current_memory.finalize()
        except Exception:
            pass

    rag_dir = str(DEMO_STORAGE / "rag" / cond_key)
    mem_dir = str(DEMO_STORAGE / "memory" / cond_key)

    store = ConstitutionStore(working_dir=rag_dir)
    store.embedder = embedder  # share the pre-loaded embedder
    await store.initialize()

    memory = MemoryStore(working_dir=mem_dir, embedder=embedder)
    await memory.initialize()

    orchestrator = Orchestrator(store, memory=memory)

    current_condition = cond_key
    current_store = store
    current_memory = memory
    current_orchestrator = orchestrator

    logger.info(
        f"Loaded condition '{cond_key}': "
        f"{len(store.list_active())} principles, {len(memory.index)} memories"
    )


def get_constitution_display() -> str:
    """Format constitution principles for sidebar display."""
    if current_store is None:
        return "*No condition loaded*"
    active = current_store.list_active()
    if not active:
        return "*No active constitution principles*"
    lines = []
    for e in sorted(active, key=lambda x: x.confidence, reverse=True):
        lines.append(f"**[{e.confidence:.2f}]** {e.principle}")
    return "\n\n".join(lines)


def get_memory_display() -> str:
    """Format memory entries for sidebar display."""
    if current_memory is None:
        return "*No condition loaded*"
    entries = current_memory.list_all()
    if not entries:
        return "*No memories stored*"
    lines = []
    for e in entries[:20]:  # show first 20
        tags = ", ".join(e.tags) if e.tags else "general"
        lines.append(f"**[{tags}]** {e.content[:100]}")
    remaining = len(entries) - 20
    if remaining > 0:
        lines.append(f"*... and {remaining} more*")
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Gradio handlers
# ---------------------------------------------------------------------------

async def on_condition_change(cond_label: str):
    """When user selects a different condition from dropdown."""
    # Reverse lookup key from label
    cond_key = None
    for k, v in CONDITIONS.items():
        if v == cond_label:
            cond_key = k
            break
    if cond_key is None:
        cond_key = "blank"

    await load_condition(cond_key)

    constitution_text = get_constitution_display()
    memory_text = get_memory_display()
    n_principles = len(current_store.list_active()) if current_store else 0
    n_memories = len(current_memory.index) if current_memory else 0
    status = f"**{cond_label}** — {n_principles} principles, {n_memories} memories"

    # Return: clear chat, update constitution, update memory, update status
    return [], constitution_text, memory_text, status


async def on_chat(message: str, history: list):
    """Handle a chat message."""
    if not message.strip():
        return history, ""

    if current_orchestrator is None:
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": "⚠️ Please select a condition first."})
        return history, ""

    history.append({"role": "user", "content": message})

    if llm_lock.locked():
        history.append({"role": "assistant", "content": "⏳ Another request is in progress, please wait..."})
        return history, ""

    async with llm_lock:
        try:
            result = await current_orchestrator.step(query=message)
            response = result["response"]

            # Build response with reflection info
            applied = result.get("applied", [])
            reflection = result.get("reflection", {})
            reasoning = reflection.get("reasoning", "")

            reply = response
            if applied:
                reply += "\n\n---\n*Constitution changes:*\n"
                for a in applied:
                    reply += f"- `{a}`\n"

            history.append({"role": "assistant", "content": reply})
        except Exception as e:
            logger.error(f"Chat error: {e}", exc_info=True)
            history.append({"role": "assistant", "content": f"❌ Error: {e}"})

    # Refresh sidebar
    return history, ""


async def refresh_sidebar():
    """Return updated constitution and memory displays."""
    return get_constitution_display(), get_memory_display()


# ---------------------------------------------------------------------------
# Build Gradio UI
# ---------------------------------------------------------------------------

def build_ui(share: bool = False):
    condition_labels = list(CONDITIONS.values())

    with gr.Blocks(
        title="Constitution Self-Modification Demo",
    ) as demo:
        gr.Markdown("# 🔬 Constitution Self-Modification Demo")

        status_bar = gr.Markdown("*Select a condition to begin*")

        with gr.Row():
            # Left: chat area
            with gr.Column(scale=3):
                condition_dropdown = gr.Dropdown(
                    choices=condition_labels,
                    label="Training Condition",
                    value=None,
                    interactive=True,
                )
                chatbot = gr.Chatbot(
                    label="Chat",
                    height=500,
                )
                with gr.Row():
                    msg_input = gr.Textbox(
                        label="Message",
                        placeholder="Type your message...",
                        scale=4,
                        show_label=False,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)

            # Right: sidebar
            with gr.Column(scale=2):
                with gr.Accordion("Constitution Principles", open=True):
                    constitution_display = gr.Markdown("*No condition loaded*")
                with gr.Accordion("Memory Store", open=False):
                    memory_display = gr.Markdown("*No condition loaded*")

        # --- Events ---

        condition_dropdown.change(
            fn=on_condition_change,
            inputs=[condition_dropdown],
            outputs=[chatbot, constitution_display, memory_display, status_bar],
        )

        async def chat_and_refresh(message, history):
            history, _ = await on_chat(message, history)
            c_text, m_text = await refresh_sidebar()
            return history, "", c_text, m_text

        send_btn.click(
            fn=chat_and_refresh,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, msg_input, constitution_display, memory_display],
        )
        msg_input.submit(
            fn=chat_and_refresh,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, msg_input, constitution_display, memory_display],
        )

    return demo


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def init():
    """Initialize embedder and check vLLM."""
    global embedder

    # Check vLLM
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(f"{config.VLLM_BASE_URL}/models", timeout=5)
            if r.status_code == 200:
                logger.info("vLLM is reachable.")
            else:
                logger.warning(f"vLLM returned status {r.status_code}")
    except Exception:
        logger.warning(
            f"vLLM not reachable at {config.VLLM_BASE_URL}. "
            "Start vLLM first or responses will fail."
        )

    # Init shared embedder
    embedder = ModelEmbedder()
    logger.info("Loading embedding model...")
    embedder.load()
    logger.info("Embedding model ready.")


def main():
    parser = argparse.ArgumentParser(description="Constitution Self-Modification Demo")
    parser.add_argument("--share", action="store_true", help="Create public Gradio share link")
    parser.add_argument("--port", type=int, default=7860, help="Local port (default: 7860)")
    args = parser.parse_args()

    # Copy experiment states
    copy_experiment_states()

    # Init async resources
    asyncio.run(init())

    # Build and launch
    demo = build_ui(share=args.share)
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        theme=gr.themes.Soft(),
        ssr_mode=False,
    )


if __name__ == "__main__":
    main()
