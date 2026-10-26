# OpenSpiel experiments

`prototype/` implements the paper's repeated matrix games, constitutional
reflection, and none / fixed / iterating conditions.

From the repository root:

```bash
python multi-agent/prototype/main.py --stub --rounds 10 --out-dir outputs/smoke
PYTHONPATH=multi-agent python -m prototype.tests.test_iterating_grows
python multi-agent/prototype/main.py --seeds 0 1 2 3 4 \
  --opponents always_cooperate always_defect random --condition all \
  --rounds 100 --reflect-every 5 --history-window 10 --out-dir outputs/openspiel
```

For inference, configure `PROTO_LLM_URL` and `PROTO_LLM_MODEL` for a running
OpenAI-compatible server. The last command runs all 45 paper conditions;
add `--stub` to exercise the same design without inference.

Original traces are in `results/openspiel/`. Regenerate their statistics with
`python scripts/summarize_results.py`, or pass `--openspiel outputs/openspiel`
to summarize new runs. See [the reproduction guide](../docs/REPRODUCING.md).
