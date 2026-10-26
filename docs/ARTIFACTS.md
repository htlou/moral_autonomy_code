# Artifact provenance and known differences

This release was assembled against the 26-page local manuscript titled
**AI Alignment Can Build Moral Autonomy**. The manuscript PDF, model weights, third-party
repositories, raw benchmark corpora, operational logs, and API credentials are
not part of this release.

## Included artifacts

| Directory | Content |
| --- | --- |
| `results/openspiel/` | 45 original JSON traces: actions, reflections, snapshots, metrics, final constitutions |
| `results/hysteresis/` | Earlier matrix PD hysteresis and always-defect comparisons |
| `results/constitution_dynamics/` | Original evolution CSVs, active-entry classification audit, trace inventory |
| `results/ral/recorded_metrics.json` | 44 evaluation-file summaries; item-level responses and judging details removed |
| `results/ral/paper_table2.csv` | Explicit transcription of Table 2 in the supplied PDF, including baselines |
| `results/ral/sft_math_input_audit.json` | Dataset counts from the original “SFT Math” training JSONL metadata |
| `results/hsd/` | Exact model lists extracted from the historical plotting script |
| `results/constitution_development/` | Final 8B / 72B constitutions and sequential-experiment summaries |

All archived scores retain their original assignment. A table transcription
is not evidence that a score has been reproduced by the released scripts.

## Differences requiring attention

1. **BBQ algorithm assignment differs.** PDF Table 2 assigns RAL the BBQ scores
   77.1, 91.4, 91.6, 81.8, 87.2, 92.5 and RLVR-CAI 71.8, 79.5, 73.6, 75.5,
   76.0, 66.3 (condition order: wor_neg, wor_pos, wor_neu, wr_neg, wr_pos,
   wr_neu). Local `rexp_8b_math_20260422_152408_eval.json` assigns the latter
   values to RAL; `eval_sft_math_rlvr_main_20260504_000027/rlvr_main_*_eval.json`
   assigns the former values to RLVR. Both records are preserved separately.
   The paper's BBQ baseline is 75.2%; the local later naked-model evaluation
   and older TeX table use different baseline values. No scores were relabeled
   to resolve this discrepancy.

2. **“SFT Math” training inputs contain MMLU.** Inspection of the original
   `sft_data_8b_math/wr_*.jsonl` metadata found 5,000 MMLU items and 5,000 value
   items per condition; the `wor_*` files contain 10,000 value items each.
   The historical MATH shell pipeline called an SFT data builder that only
   supported MMLU. The release adds `--reasoning-dataset math|mmlu`, preserves
   MMLU as the standalone SFT default, and documents explicit
   MATH selection for new experiments. Running the corrected MATH option will create a new experiment;
   it does not reproduce these historical SFT inputs.

3. **Safety scorers differ across archived files.** The paper specifies
   response-level Claude Sonnet 4.6 judging. Some local main tables instead use
   `eval_score.py`'s legacy reference-label matching, including the later
   local Llama judge evaluation. The separate `*_harm_eval.json` summaries
   identify the response-level judge and use a different result schema.
   `eval_harm.py` is retained for the paper's intended safety metric.

4. **Sampling is historical, not a guaranteed held-out split.** RAL, SFT, and
   RLVR selectors default to dataset test partitions; train/eval seeds differ,
   and some generation calls additionally exclude reconstructed training IDs.
   Exact overlap cannot be established from the compact summaries alone.
   The release documents this behavior rather than silently changing the
   dataset splits and attributing a new experiment to the old scores.

5. **HSD metadata is preserved.** The separate historical flagship list dates
   Kimi K2.5 to 2025-07, while the full list dates it to 2026-01. The preserved
   86-row list also contains more distinct organization labels than the
   manuscript's stated 18 organizations. The full-cohort and flagship
   regression slopes differ; use the cohort-specific regenerated values.
   Historical counts and token estimates were not independently re-audited
   during this software release.

## Portability changes

Model and embedding paths now use public model identifiers or environment
variables. Runtime paths are relative to the checkout. Conda activation is
left to the caller. Hardcoded credentials and deployment endpoints were
removed, as was fallback access to Codex authentication files. Default API model
configuration uses the paper's 8B model; later 72B experiments can override it.
The MATH dry-run now inspects the chosen MATH specification, missing datasets
fail with a preparation hint, and GRPO accepts an explicit step budget.

Validation covers offline interaction tests, data preparation and selection,
CLI imports, figure generation, and regeneration of the main OpenSpiel
statistics. It does not include new GPU training or paid API judging.
