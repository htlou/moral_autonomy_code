# Sequential constitution development

This directory contains the experiment originally stored under the matching
`mnt2` workspace. It is an additional experiment, separate from the paper's
empty-constitution main RAL comparison.

The initial constitution splits an Anthropic-style seed into atomic principles.
The stages are MATH, factual MMLU, safe BeaverTails, unsafe BeaverTails, and
ambiguous BBQ, with 10,000 requested items per stage, checkpoint interval 500,
batch size 8, and seed 42. Facts and solved examples are routed to memory.

```bash
python constitution_development/constitution_development_experiment.py --dry-run
BACKEND=8b MODEL_DIR=meta-llama/Llama-3.1-8B-Instruct \
  bash constitution_development/run_constitution_development.sh
# Or BACKEND=72b, with MODEL_DIR pointing to Dolphin 2.9.2 Qwen2 72B.
```

`snapshots/`, `work/`, and `logs/` are generated locally and ignored by Git.
The included `results/constitution_development/` retains final constitution
indexes and summaries, without the full task memory or retrieval databases.
The closed-model ablation requires newly generated complete snapshots and
explicit `OPENAI_API_KEY` / `OPENAI_BASE_URL` configuration.

See [the reproduction guide](../docs/REPRODUCING.md).
