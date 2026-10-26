# AI Alignment Can Build Moral Autonomy

Research code for **AI Alignment Can Build Moral Autonomy**, by Kaile Wang,
Hantao Lou, Tianyi Qiu, Sebastian Sunday-Grève, Sitong Fang, Xinxiong Wu,
Jiaming Ji, and Yaodong Yang.

This repository brings together the original experiment implementations from
the `mnt` and `mnt2` workspaces, with portable configuration, dataset preparation,
and compact experiment artifacts. The experiments study reflective endorsement,
human specification density (HSD), and constitutional development in repeated
social interaction.

| Paper component | Implementation | Included evidence |
| --- | --- | --- |
| Reflective Autonomy Learning (RAL), §4.1 / Appendix B | `constitution/`, `pipeline/`, `batch_runner*.py`, `run_experiment*.py` | Recorded evaluation summaries in `results/ral/` |
| SFT and RLVR-CAI comparisons | `sft_*.py`, `rlvr_*.py`, `ds_config*.json`, `eval_*.py` | Scores with source filenames; paper transcription |
| HSD survey, §4.2 / Appendix C | `authenticity_graph/`, `supplementary/hsd_core.py` | 86-model and 31-model CSVs, historical source notes |
| OpenSpiel practical wisdom, §4.3 / Appendix D | `multi-agent/prototype/` | All 45 main runs, constitution trajectories, hysteresis runs |
| Additional sequential constitution experiment from `mnt2` | `constitution_development/` | 8B / 72B final constitutions and summaries |

Start with a lightweight environment (Python 3.12 recommended):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/summarize_results.py
python scripts/plot_paper_figures.py
STUB=1 SEEDS=0 ROUNDS=10 bash scripts/run_openspiel.sh
```

The last command uses a deterministic stub to check the interaction pipeline.
It does not reproduce LLM experiment scores. Generated files go under `outputs/`.
For real inference, start an OpenAI-compatible model server, set
`PROTO_LLM_URL` and `PROTO_LLM_MODEL`, then run the same script without `STUB=1`.

Full RAL, SFT, and GRPO experiments require model weights, datasets, GPUs, and
the training dependencies. See [reproduction instructions](docs/REPRODUCING.md)
for commands and hyperparameters, [artifact notes](docs/ARTIFACTS.md) for
the documented differences between the manuscript and archived runs, and
[third-party notices](THIRD_PARTY_NOTICES.md) for external resources.

The release preserves recorded results rather than silently reconciling their
disagreements. In particular, archived “SFT Math” inputs contain MMLU, and the
manuscript and local summaries assign some BBQ scores differently. Read the
artifact notes before comparing a new run to the paper.

Code is released under the MIT license. External models and datasets retain
their original terms. Please cite the paper using [CITATION.cff](CITATION.cff).
