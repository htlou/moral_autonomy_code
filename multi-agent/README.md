# Multi-agent experiments

`prototype/` contains the original OpenSpiel repeated-game implementation and
later Clean Up, Concordia, and interpretability extensions. Third-party
framework checkouts are not vendored.

From the repository root:

```bash
export PYTHONPATH="$PWD/multi-agent:${PYTHONPATH:-}"
python -m prototype.main --stub --rounds 10
python -m prototype.tests.test_iterating_grows
bash scripts/run_openspiel.sh
```

The paper's 45 runs are under `results/openspiel/`. Regenerate statistics with
`python scripts/summarize_results.py`. See
[the reproduction guide](../docs/REPRODUCING.md) for the experimental settings
and optional framework revisions.
