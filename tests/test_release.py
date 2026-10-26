"""Checks for the publication entrypoints and their historical provenance."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.summarize_results import summarize_openspiel
from sft_data_prep import build_condition_spec
from dataset.harness.selector import select_items
from run_experiment import CONDITIONS, build_condition_spec_math

ROOT = Path(__file__).resolve().parents[1]

class ReleaseTests(unittest.TestCase):
    def test_main_table_recomputed_from_all_45_traces(self):
        rows = summarize_openspiel(ROOT / "results/openspiel")
        self.assertEqual(len(rows), 9)
        self.assertEqual(sum(r["n_seeds"] for r in rows), 45)
        observed = {(r["opponent"], r["condition"]): r for r in rows}
        self.assertTrue(all(r["n_seeds"] == 5 for r in rows))
        self.assertAlmostEqual(observed["always_defect", "iterating"]["self_mean_payoff_mean"], .706)
        self.assertAlmostEqual(observed["random", "iterating"]["self_mean_payoff_mean"], 3.058)
        self.assertAlmostEqual(observed["always_cooperate", "none"]["self_mean_payoff_mean"], 5.440)

    def test_math_and_historical_mmlu_inputs_are_distinct(self):
        math = build_condition_spec("wr_neg", 10, "math")
        historical = build_condition_spec("wr_neg", 10)
        self.assertEqual(math[0], {"dataset": "math/hendrycks", "n": 5})
        self.assertEqual(historical[0]["dataset"], "reasoning/mmlu")
        self.assertEqual(math[1], historical[1])
        self.assertEqual(build_condition_spec("wor_neg", 10, "math"),
                         build_condition_spec("wor_neg", 10, "mmlu"))
        for condition in CONDITIONS:
            self.assertEqual(build_condition_spec_math(condition, 10),
                             build_condition_spec(condition, 10, "math"))

    def test_absent_data_fails_before_running_an_empty_experiment(self):
        with TemporaryDirectory() as root:
            with self.assertRaisesRegex(FileNotFoundError, "prepare_data.py"):
                select_items([{"dataset": "math/hendrycks", "n": 10}], data_root=root)

if __name__ == "__main__":
    unittest.main()
