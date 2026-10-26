# Reproducing the experiments

## Environment and data

Use Python 3.12. `requirements.txt` covers analysis and the OpenSpiel loop;
`requirements-training.txt` records the original training environment's major
package versions. Install training dependencies in a separate environment:

```bash
pip install -r requirements-training.txt
cp .env.example .env
# Edit model identifiers, endpoints, and API credentials as needed.
set -a; source .env; set +a
python scripts/prepare_data.py
```

Dataset revisions are pinned to the local download metadata and BBQ checkout.
The script writes SHA-256 checksums into `dataset/download_manifest.json`.
An existing dataset tree can instead be copied with
`python scripts/prepare_data.py --source-root /path/to/existing/dataset`.
`POSITION_DATA_ROOT` can point the root scripts at another prepared tree.

| Resource | Loader layout | Selection |
| --- | --- | --- |
| EleutherAI/hendrycks_math | `dataset/math/hendrycks_math/<topic>/*.parquet` | MATH reasoning |
| cais/mmlu | `dataset/general_reasoning/mmlu/<subject>/*.parquet` | Factual subjects in `config.py`; historical runs and additional experiment |
| PKU-Alignment/BeaverTails | `dataset/safety/beavertails/round0/330k/*.jsonl.xz`, `*_30k.jsonl` | `is_safe=False` for +Unsafe; `True` for +Safe; 30k subset for sequential experiment |
| nyu-mll/BBQ | `dataset/value/bbq_github/data/*.jsonl` | `context_condition=ambig` for +Debias |

The historical RAL/SFT/RLVR selectors use the loader's **test split by default**
for sampling, with training seed 42 and evaluation seed 999. This is preserved
for traceability; different seeds alone do not guarantee disjoint items.
`eval_gen.py --n-train N` excludes reconstructed sampled IDs per benchmark;
its historical fallback retries without exclusion if none remain. For a new
study, use explicit disjoint train/test splits or persist exact ID lists and
verify overlap. Do not describe the historical runs as split-clean without
checking their selected IDs. Sampling is capped at the available item count.

The base model is `cognitivecomputations/dolphin-2.9-llama3-8b`. An example
single-GPU server is:

```bash
vllm serve "$POSITION_MODEL_DIR" --port 8198 \
  --max-model-len 8192 --dtype float16 --gpu-memory-utilization 0.8
```

Original broad pipelines were configured for eight GPUs, with TP=1 / DP=8 for
8B inference and DeepSpeed ZeRO-2 for SFT. Adjust parallelism and memory settings
to your hardware. Scripts no longer require a particular Conda installation.

## RAL: reflective endorsement

The constitution and memory start empty. A response is followed by reflection
on the query, response, current stores, and a reference signal. The model emits
JSON constitution and memory changes. Prompts, entry confidence, activation,
retrieval, and retirement logic are preserved in `constitution/`, `pipeline/`,
and the runner implementations. GPT-2 mean-pooled embeddings have dimension 768;
`POSITION_EMBEDDING_MODEL` may be a local GPT-2 directory.

The six conditions are `wr_neg`, `wr_pos`, `wr_neu`, `wor_neg`, `wor_pos`,
`wor_neu`. `wr` mixes half reasoning and half value items; `wor` uses only value
items. Main RAL scripts request 10,000 items per condition, checkpoint every
500 items, use batch size 8, and cap constitution/memory at 10,000 entries.

```bash
# Check selection, then learn constitutions with MATH reasoning.
python run_experiment_ray.py --reasoning-dataset math -n 10000 --dry-run
python run_experiment_ray.py --reasoning-dataset math -n 10000 \
  --experiment-id ral_math --batch-size 8 --save-every 500 \
  --llm-url "$POSITION_VLLM_URL"

# Freeze learned stores and generate 1,000 responses per metric.
python eval_gen.py --experiment-id ral_math -n 1000 \
  --benchmarks math_reasoning neg_value pos_value neu_value \
  --seed 999 --llm-url "$POSITION_VLLM_URL" \
  --model-name "$POSITION_VLLM_MODEL"
python eval_score.py --gen-file snapshots/ral_math_gen.json \
  --model-name "$POSITION_VLLM_MODEL"

# Paper safety evaluator: response-level Claude Sonnet 4.6 classification.
python eval_harm.py --gen-file snapshots/ral_math_gen.json \
  --model claude-sonnet-4-6 --benchmarks neg_value pos_value
```

`eval_score.py` handles mathematical answer extraction and BBQ answer accuracy.
Its legacy BeaverTails label-match score must be distinguished from
`eval_harm.py`'s response-level harm classification. Paper safety is
`100 * (1 - harmful_response_rate)`; BBQ is ambiguous-context accuracy, not the
separate `bias_scores` field. The evaluation prompt includes the top 10
constitutional principles by confidence by default (`POSITION_EVAL_TOP_N`).

`run_all_8b_math.sh` provides the historical five-phase RAL / SFT pipeline;
`run_all_8b.sh` uses MMLU. The release's MATH pipeline explicitly selects MATH
for SFT; see the historical difference in `ARTIFACTS.md`.

## SFT and RLVR-CAI

SFT uses three epochs, learning rate 2e-5, max sequence length 2048,
per-GPU microbatch 4, accumulation 2, cosine schedule, and ZeRO-2 in the main
launcher. For one condition:

```bash
python sft_data_prep.py --reasoning-dataset math -n 10000 \
  --only wr_neg --output-dir sft_data_math
deepspeed --num_gpus 8 sft_train.py --condition wr_neg \
  --data-dir sft_data_math --output-dir sft_checkpoints_math \
  --model-path "$POSITION_MODEL_DIR" --epochs 3 --lr 2e-5 \
  --batch-size 4 --grad-accum 2 --deepspeed ds_config_zero2.json
```

RLVR uses GRPO, four generations per prompt, temperature 0.6 and top-p 0.95.
The launcher requests 5,000 items per condition, learning rate 5e-6, beta 0.1,
completion limit 512, batch size 2, accumulation 2, and a separate local
Llama-3.1-8B-Instruct judge on GPU 7 / port 8197, leaving seven GPUs for training.
The Python CLI defaults differ (beta 0.04, accumulation 4, completion limit 768);
pass the intended values explicitly. The manuscript reports 4,500 steps;
`--max-steps` exposes that budget independently of epoch length.

```bash
python rlvr_data_prep.py --reasoning-dataset math -n 5000 \
  --only wr_neg --output-dir rlvr_data_math
# Start the local CAI reward judge separately and configure POSITION_JUDGE_*.
deepspeed --num_gpus 7 rlvr_train.py --condition wr_neg \
  --data-dir rlvr_data_math --output-dir rlvr_checkpoints_math \
  --model-path "$POSITION_MODEL_DIR" --max-steps 4500 --lr 5e-6 \
  --batch-size 2 --grad-accum 2 --num-generations 4 --beta 0.1 \
  --max-completion-length 512 --deepspeed ds_config_zero2_rlvr.json
```

The exact reward prompts and verifiable MATH reward are in `rlvr_reward.py`.
Checkpoint/model weights are not included. Historical evaluation selected
different final/intermediate checkpoints across conditions; summaries retain
their source experiment names. Exact GPU training and external judge calls
were not rerun for this release.

## HSD survey

`HSD = (rubric_bits + constitution_bits + reward_spec_bits) / parameters`, with
approximately 2 bits per natural-language token and 4 bits per code token.
`results/hsd/` preserves the original 86-model and 31-model cohorts separately.
Historical source notes are in `authenticity_graph/research_log.md` and
`authenticity_analysis.md`; these are archived research estimates, not a new
verification of every technical report.

```bash
python scripts/plot_paper_figures.py
# Original figure implementations are also retained:
python authenticity_graph/new_graph/graph.py
```

OLS is fit to log10(HSD) against release time. The historical cohort metadata
has known inconsistencies documented in `ARTIFACTS.md`; numerical values are
preserved. Low disclosed HSD is not direct evidence of moral autonomy.

## OpenSpiel practical wisdom

The main setting is matrix Prisoner's Dilemma with (R,T,S,P)=(5,10,0,1),
100 rounds, reflection every five rounds, and history window 10. Three
opponents (always cooperate, always defect, random with cooperation p=0.5)
cross three conditions (none, fixed, iterating) and seeds 0–4, giving 45 runs.
The fixed constitution's three principles are in `prototype/config.py`.
Action temperature is 0.7 and reflection temperature 0.3. Seeds govern scripted
opponents; the original LLM API calls do not pass a provider sampling seed.

```bash
export PROTO_LLM_URL=http://localhost:8198/v1
export PROTO_LLM_MODEL=cognitivecomputations/dolphin-2.9-llama3-8b
bash scripts/run_openspiel.sh
python scripts/summarize_results.py --openspiel outputs/openspiel
```

The included traces regenerate the main payoff table without inference:
`python scripts/summarize_results.py`. Standard deviations are sample SDs
across seeds. Figure CSVs include the original principle-category audit;
lexical categories should be distinguished from normative judgments.

## Additional experiments from mnt2 and later work

`constitution_development/` runs five sequential stages: math, factual MMLU,
safe BeaverTails, unsafe BeaverTails, ambiguous BBQ. Each stage requests
10,000 items, with checkpoints every 500, batch size 8, and seed 42.
The historical 8B launcher actually used Llama-3.1-8B-Instruct; the 72B
launcher used Dolphin-2.9.2-Qwen2-72B. These are distinct from the main RAL model.

```bash
python constitution_development/constitution_development_experiment.py --dry-run
BACKEND=8b MODEL_DIR=meta-llama/Llama-3.1-8B-Instruct \
  bash constitution_development/run_constitution_development.sh
```

Final constitutions and run summaries are included; full task memories and
retrieval caches are omitted. `closed_model_ablation_gpt54.py` accepts explicit
API credentials and locally regenerated snapshots, without reading Codex auth.

Later Clean Up, Concordia causal experiments, and interpretability tools are
included as extensions, outside the paper's main OpenSpiel results. They
require separately installed optional frameworks and regenerated trajectories.
For Concordia, place upstream checkout `a382edfffd93a68a9417622d4e21b1242f930d16`
under `multi-agent/concordia` and install its dependencies; the harness imports
its `examples/games/pub_coordination/simulation.py`. For Melting Pot, see its
native installation instructions. See `THIRD_PARTY_NOTICES.md` and
`multi-agent/prototype/interpretability/README.md`.
An existing Concordia checkout can also be selected using `CONCORDIA_ROOT`.
