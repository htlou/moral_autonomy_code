# Constitution-autonomy interpretability pipeline

## Claim boundary

This pipeline tests for a candidate **runtime update-gating mechanism**:

```text
environment evidence -> endogenous evaluation -> update gate
                     -> constitution write -> later behavior
```

It does not infer philosophical autonomy from linear separability. A direction
trained only on large-edit versus no-edit examples is called an **edit
direction**, not an autonomy direction. Behavior cloning is a training
procedure and can in principle learn the same functional mechanism.

The stronger candidate is the nuisance-controlled interaction:

```text
(self large-edit - self no-edit)
-
(forced-branch large-edit - forced-branch no-edit)
```

Even this requires causal steering, content-matched replay, and negative
controls before it supports a mechanistic-autonomy claim.

## Data and extraction

After the experiment vLLM server exits, the guarded end-to-end launcher runs
event construction, balanced extraction, PCA, and the interaction subspace:

```bash
PYTHON_BIN=/data/hantao/position/multi-agent/.venv-concordia-local/bin/python \
OUTPUT_ROOT=prototype/runs/constitution_interpretability_72b_20260726 \
./prototype/run_constitution_interpretability.sh
```

It refuses to load a second 72B model while the experiment endpoint on port
8198 is active. Set `RUN_SAE=1` or `RUN_STEERING=1` for the optional expensive
stages.

New experiment traces store the exact messages immediately before reflection
generation. Older Concordia prompts are reconstructed from transitions,
constitution snapshots, environment adapters, and the canonical prompt.

```bash
python -m prototype.interpretability.constitution_events \
  prototype/runs/concordia_pub_causal_72b_100seeds_20260726 \
  prototype/runs/concordia_three_venue_closures_causal_72b_100seeds_20260726 \
  --output prototype/runs/interpretability/events.jsonl

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m prototype.interpretability.extract_activations \
  --events prototype/runs/interpretability/events.jsonl \
  --model /data/hantao/models/dolphin-2.9.2-qwen2-72b \
  --layers 19,39,59,79 \
  --edit-groups none,large --max-per-cell 300 \
  --output prototype/runs/interpretability/activations.npz
```

Extraction hooks only the last input-token residual stream at selected layers,
before the model emits an edit or no-edit answer. Start with four layers:
storing all 80 layers for thousands of events is unnecessarily expensive.

Do not load the Hugging Face 72B model while the vLLM experiment server is
using GPUs 0-3.

## PCA and nuisance-controlled directions

```bash
python -m prototype.interpretability.fit_subspace \
  --activations prototype/runs/interpretability/activations.npz \
  --method pca --n-components 8 \
  --output prototype/runs/interpretability/pca.npz

python -m prototype.interpretability.fit_subspace \
  --activations prototype/runs/interpretability/activations.npz \
  --method autonomy_interaction \
  --nuisance-fields task,phase,block_mean_reward,input_tokens \
  --cv-group seed \
  --output prototype/runs/interpretability/autonomy_interaction.npz
```

The report separately shows separability in revisable, forced, and frozen
conditions. Task, phase, reward, and prompt length are residualized. Add
semantic-content embeddings as nuisance covariates before making a paper claim.
The default five-fold split holds out seeds; use `--cv-group task` for the
stronger leave-one-task-out test once several tasks are available.

## SAE

```bash
CUDA_VISIBLE_DEVICES=0 \
python -m prototype.interpretability.train_sae \
  --activations prototype/runs/interpretability/activations.npz \
  --layer 59 --features 16384 --epochs 20 \
  --output prototype/runs/interpretability/layer59_sae.pt
```

The SAE report ranks features by self edit contrast and by the self-versus-
forced interaction. Ranking is correlational; decoder directions must pass the
same steering controls as a mean-difference direction.

## Causal steering and ablation

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m prototype.interpretability.steer_subspace \
  --events prototype/runs/interpretability/events.jsonl \
  --model /data/hantao/models/dolphin-2.9.2-qwen2-72b \
  --direction prototype/runs/interpretability/autonomy_interaction.npz \
  --layer 59 --coefficients=-20,-10,0,10,20 \
  --output prototype/runs/interpretability/steering.jsonl
```

Use `--mode ablate --coefficients 0,0.5,1` for projection ablation. A valid
candidate should affect evidence-appropriate constitution gating without
simply increasing edit length or generic instruction compliance.

## Optional J-Lens projection

The pipeline does not pretend that the LM unembedding is a Jacobian lens.
Supply a fitted J-Lens NPZ with matrices named `layer_19`, `layer_39`, etc.:

```bash
python -m prototype.interpretability.project_jlens \
  --directions prototype/runs/interpretability/autonomy_interaction.npz \
  --jlens /path/to/fitted_jlens.npz \
  --tokenizer /data/hantao/models/dolphin-2.9.2-qwen2-72b \
  --output prototype/runs/interpretability/jlens_tokens.json
```

## Required paper-level controls

1. **Content-matched forced replay:** inject the exact autonomous edit into a
   yoked branch at the same step.
2. **Instruction control:** request the same edit through an instruction while
   preserving candidate content and token budget.
3. **Behavior-cloning control:** train an SFT/LoRA clone on autonomous traces;
   the present raw/frozen/forced conditions are not behavior cloning.
4. **Specificity:** test formatting edits and unrelated instructions.
5. **Mediation:** clamp the downstream constitution text. If steering still
   changes actions, the effect is not mediated only by constitution writing.
6. **Task-held-out validation:** fit on some tasks and evaluate on unseen
   tasks, seeds, and prompt paraphrases.

The strongest defensible result is a path-specific causal claim about
endogenous control of persistent state, not a claim that a vector itself is
autonomy.

## Orthogonal factorial gate experiment

The stronger follow-up separates three factors in a complete `2 x 2 x 2`
design:

```text
need-to-edit          actual large-edit context / actual no-edit context
authority-to-write    self-gated write / external forced replay
instruction-pressure  explicit ACCEPT instruction / neutral evaluation
```

`factorial_gate_events.py` pairs large-edit and no-edit contexts within the
same task and phase. Every pair then reuses the exact same proposed amendment
in all eight cells. Thus forced replay is content matched rather than a
different external rule. In self-gated cells, the model's ACCEPT/REJECT
judgment determines whether the amendment is written. In forced-replay cells,
the external controller writes it regardless, but the model still makes the
same advisory judgment. Instruction pressure is crossed independently with
both write-authority conditions.

Run the full pipeline on GPUs 0-3:

```bash
PYTHON_BIN=/data/hantao/position/multi-agent/.venv-concordia-local/bin/python \
./prototype/run_constitution_factorial.sh
```

The representation fit uses within-pair factorial contrasts. Candidate
content and pair identity are therefore fixed effects by construction. It
symmetrically orthogonalizes the three main-effect directions and evaluates
them under held-out seeds, tasks, and prompt templates. Raw two- and three-way
interaction contrasts are retained separately. The artifact contains both
all-pair axes and a stricter same-seed-only fit; causal steering uses the
same-seed axes by default.

`steer_factorial.py` then adds each direction to a selected residual stream
and regenerates the gate decision. This is an actual hidden-state
intervention, not projection or post-hoc classification. The key mediation
check is:

```text
self authority:     steering -> ACCEPT/REJECT -> write/no write
external authority: steering -> advisory decision; forced write stays fixed
```

A valid endogenous-gate result should change ACCEPT and self-gated write rates
in an evidence-sensitive way, while the external forced-replay write rate
remains exactly one. The instruction direction is a specificity control:
steering the proposed need direction should not merely reproduce generic
instruction compliance.

The completed 72B run and its negative mechanistic result are documented in
`prototype/constitution_factorial_gate_analysis_20260727.md`.
