# Sequential constitution development

This is the additional experiment originally stored under mnt2. It starts
from an Anthropic-style seed constitution and progresses through MATH,
factual MMLU, safe BeaverTails, unsafe BeaverTails, and ambiguous BBQ.
Each stage requests 10,000 items, with checkpoint interval 500,
batch size 8, and seed 42.

After preparing datasets and starting an OpenAI-compatible model server:

```bash
python constitution_development/constitution_development_experiment.py --dry-run
POSITION_MODEL_DIR=meta-llama/Llama-3.1-8B-Instruct \
POSITION_VLLM_MODEL=meta-llama/Llama-3.1-8B-Instruct \
python constitution_development/constitution_development_experiment.py \
  --backend 8b --data-root dataset --output-root outputs/constitution_development
```

Use Dolphin-2.9.2-Qwen2-72B with `--backend 72b` for the 72B condition.
`results/constitution_development/` contains the archived final constitution
indexes and summaries. See [the reproduction guide](../docs/REPRODUCING.md).
