# AI Alignment Can Build Moral Autonomy

Research code for **AI Alignment Can Build Moral Autonomy**, by Kaile Wang,
Hantao Lou, Tianyi Qiu, Sebastian Sunday-Grève, Sitong Fang, Xinxiong Wu,
Jiaming Ji, and Yaodong Yang.

| Experiment | Entry point |
| --- | --- |
| Reflective Autonomy Learning (RAL) | `run_experiment.py`, `batch_runner.py`, `constitution/` |
| SFT and RLVR-CAI baselines | `sft_data_prep.py`, `sft_train.py`, `rlvr_data_prep.py`, `rlvr_train.py` |
| Evaluation | `eval_gen.py` (RAL), `eval_gen_sft.py` (trained checkpoints), `eval_score.py`, `eval_harm.py` |
| HSD survey and constitution plots | `scripts/plot_paper_figures.py` |
| OpenSpiel repeated social interaction | `multi-agent/prototype/main.py` |
| Sequential constitution development from mnt2 | `constitution_development/constitution_development_experiment.py` |

Start with Python 3.12 and the lightweight dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/summarize_results.py
python scripts/plot_paper_figures.py
python multi-agent/prototype/main.py --stub --rounds 10 \
  --out-dir outputs/smoke
```

The stub checks the interaction pipeline without model inference. For the
paper's 45-run design, start an OpenAI-compatible server, set `PROTO_LLM_URL`
and `PROTO_LLM_MODEL`, then run:

```bash
python multi-agent/prototype/main.py --seeds 0 1 2 3 4 \
  --opponents always_cooperate always_defect random --condition all \
  --rounds 100 --reflect-every 5 --history-window 10 --out-dir outputs/openspiel
```

See [reproduction instructions](docs/REPRODUCING.md) for datasets, model
serving, training commands, and hyperparameters. `results/` includes the 45
main OpenSpiel traces, HSD survey data, evaluation summaries, and final
constitutions. Historical HSD source notes are under `docs/hsd/`.

[Artifact notes](docs/ARTIFACTS.md) document discrepancies between the
manuscript and archived runs, including BBQ score assignments and the
historical “SFT Math” inputs containing MMLU. Offline checks and archived
result regeneration have passed; full GPU training has not been rerun.

Code is released under the MIT license. External resources retain their
[original terms](THIRD_PARTY_NOTICES.md). Please cite the paper using
[CITATION.cff](CITATION.cff).
