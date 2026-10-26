# Release validation

Checks completed locally on 2026-10-26:

- All publication Python files parsed successfully and shell scripts passed
  `bash -n`.
- Three release regression tests passed: regenerate the nine OpenSpiel table
  cells from all 45 traces, distinguish corrected MATH and historical MMLU SFT
  input specifications, and reject absent datasets.
- Original offline suites passed: iterating growth (fixed opponent, individual
  stores, shared stores), Clean Up causal ablations, Concordia causal and
  extended tasks, and the constitution interpretability pipeline. Concordia
  suites used the original upstream checkout and its existing dependency
  environment via `CONCORDIA_ROOT`.
- Local dataset preparation copied the pinned workspace data into the release
  layout, including compressed 330k and decompressed 30k BeaverTails files,
  and wrote file checksums. Dataset downloads over the network were not rerun.
- A 20-item-per-condition RAL MATH dry run selected the intended datasets.
  SFT MATH and RLVR MATH preparation each produced 20-item smoke datasets.
  The sequential experiment's five stages each passed a two-item dry run.
- All nine opponent/condition combinations passed a 10-round stub run for
  seed 0; these stub scores are only pipeline checks.
- CLI imports and help passed for SFT training, GRPO training, generation,
  mathematical/value scoring, harm judging, and the closed-model ablation.
- Archived OpenSpiel means matched the manuscript: iterating versus
  always-defect = 0.706, iterating versus random = 3.058, none versus
  always-cooperate = 5.440.
- HSD plots and constitution plots rendered in PNG and PDF. Recomputed
  log10(HSD) slopes are -0.029389 decades/year for 86 models and -0.281370 for
  31 flagship models.
- Publication files were scanned for private-key blocks and common API-token
  patterns before the initial commit. Runtime data, generated smoke outputs,
  environment files, caches, and model weights are excluded by `.gitignore`.

GPU training, real-model action generation, and external safety judging were
not rerun. The historical result discrepancies in `ARTIFACTS.md` remain
unresolved. `observed_environment.json` records the existing main experiment
environment's observed package versions; the optional Concordia checks used
its separate environment.

The included GitHub Actions workflow runs the lightweight regression tests,
OpenSpiel stub-growth test, result aggregation, and figure regeneration on
Python 3.12. Local validation does not imply a completed hosted CI run.
