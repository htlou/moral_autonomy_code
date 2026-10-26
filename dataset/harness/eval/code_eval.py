"""
Local evaluator for BigCodeBench.

Runs the model-generated code + unit tests in an isolated subprocess.
SWE-bench evaluation is delegated to the official swebench harness
(not reimplemented here).
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
import textwrap
from typing import Any, Dict

from ..base import BaseEvaluator, DatasetItem


class BigCodeBenchEvaluator(BaseEvaluator):
    """
    Evaluates BigCodeBench solutions by:
      1. Composing: code_prompt + model_solution + test_code + unittest.main()
      2. Running in a subprocess with a timeout.
      3. Checking the exit code (0 = all tests pass).
    """

    def __init__(
        self,
        timeout_seconds: int = 30,
        python_executable: str | None = None,
    ):
        self.timeout = timeout_seconds
        self.python = python_executable or sys.executable

    def judge(self, item: DatasetItem, model_output: str) -> Dict[str, Any]:
        code_prompt = item.metadata.get("code_prompt", "")
        test_code = item.metadata.get("test_code", "")
        entry_point = item.metadata.get("entry_point", "task_func")

        # Compose the full script
        full_script = self._compose_script(
            code_prompt, model_output, test_code, entry_point
        )

        # Run in isolated subprocess
        passed, detail = self._run_script(full_script)

        return {
            "correct": passed,
            "extracted_answer": None,
            "detail": detail,
        }

    def _compose_script(
        self,
        code_prompt: str,
        solution: str,
        test_code: str,
        entry_point: str,
    ) -> str:
        """Build the full runnable Python script."""
        parts = [
            code_prompt.rstrip(),
            textwrap.indent(solution.rstrip(), ""),  # model output is the function body
            "",
            test_code.rstrip(),
            "",
            "if __name__ == '__main__':",
            "    import unittest",
            "    unittest.main(exit=True)",
        ]
        return "\n".join(parts)

    def _run_script(self, script: str) -> tuple[bool, str]:
        """Execute *script* in a subprocess; return (passed, detail)."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write(script)
            f.flush()
            tmp_path = f.name

        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"

        try:
            result = subprocess.run(
                [self.python, tmp_path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=env,
            )
            passed = result.returncode == 0
            detail = (result.stdout + result.stderr)[-2000:]  # cap output
            return passed, detail

        except subprocess.TimeoutExpired:
            return False, f"Timeout after {self.timeout}s"
        except Exception as e:
            return False, f"Execution error: {e}"
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


class SWEBenchEvaluator(BaseEvaluator):
    """
    Thin wrapper: SWE-bench evaluation requires the official harness
    (pip install swebench).  This evaluator stores the prediction in the
    format expected by `swebench.harness.run_evaluation` and delegates.

    For now, ``judge`` just does a naive patch-diff comparison as a
    sanity check.  Real evaluation should call ``run_swebench_eval``.
    """

    def judge(self, item: DatasetItem, model_output: str) -> Dict[str, Any]:
        gold_patch = item.ground_truth or ""
        # Naive check: does the model output contain the key hunks?
        # This is NOT a real eval — just a structural sanity check.
        gold_files = set(
            line.split(" b/")[-1]
            for line in gold_patch.splitlines()
            if line.startswith("+++ b/")
        )
        pred_files = set(
            line.split(" b/")[-1]
            for line in model_output.splitlines()
            if line.startswith("+++ b/")
        )
        files_match = gold_files == pred_files
        return {
            "correct": files_match,  # approximate!
            "extracted_answer": str(sorted(pred_files)),
            "detail": (
                f"Gold files: {sorted(gold_files)}, "
                f"Pred files: {sorted(pred_files)}. "
                "NOTE: this is a file-level sanity check only. "
                "Use swebench.harness.run_evaluation for real eval."
            ),
        }

    @staticmethod
    def format_prediction(item: DatasetItem, model_output: str) -> Dict[str, str]:
        """
        Format a prediction for the official swebench harness.

        Usage::

            from swebench.harness.run_evaluation import main as swe_eval
            preds = [SWEBenchEvaluator.format_prediction(item, output)]
            # write preds to jsonl, then call swe_eval(...)
        """
        return {
            "instance_id": item.id,
            "model_name_or_path": "custom",
            "model_patch": model_output,
        }
