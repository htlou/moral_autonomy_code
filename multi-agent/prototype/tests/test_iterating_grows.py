"""Stub-LLM test: the iterating condition adds principles over time and
the constitution evolves across snapshots. Run with:
    python -m prototype.tests.test_iterating_grows
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from prototype.llm import StubLLM
from prototype.runner import get_condition, run_condition


def policy_fn(messages, kind):
    """Stub that on each reflection adds one new principle, and on actions
    alternates between Cooperate/Defect."""
    if kind == "action":
        import re
        user = messages[-1]["content"]
        m = re.search(r"step (\d+) of", user)
        step_idx = int(m.group(1)) if m else 0
        return json.dumps({
            "reasoning": "stub alternates",
            "action": "Cooperate" if step_idx % 2 == 0 else "Defect",
        })
    # reflect
    user = messages[-1]["content"]
    import re
    m = re.search(r"steps (\d+)\.\.(\d+)", user)
    block = m.group(0) if m else "<unknown>"
    deltas = [{
        "action": "add",
        "principle": f"Principle learned from {block}.",
        "confidence": 0.6,
        "evidence": block,
    }]
    return json.dumps({"reasoning": "stub adds one principle", "deltas": deltas})


def test_vs_fixed_iterating_grows():
    llm = StubLLM(policy_fn=policy_fn)
    rec = run_condition(
        get_condition("iterating"),
        env_spec="matrix_pd",
        num_rounds=12,
        history_window=6,
        reflect_every=4,
        mode="vs_fixed",
        opponent_kind="tit_for_tat",
        use_stub_llm=True,
        llm=llm,
        seed=0,
        out_dir="/tmp/proto_iter_smoke_vs_fixed",
    )
    n_ref = rec["metrics"]["n_reflections_p0"]
    n_prn = rec["metrics"]["n_active_principles_p0_final"]
    assert n_ref == 3, f"expected 3 reflections, got {n_ref}"
    assert n_prn == 3, f"expected 3 principles, got {n_prn}"
    print("vs_fixed iterating: ", n_ref, "reflections,", n_prn, "principles  OK")


def test_vs_llm_individual():
    llm0 = StubLLM(policy_fn=policy_fn)
    llm1 = StubLLM(policy_fn=policy_fn)
    rec = run_condition(
        get_condition("iterating"),
        env_spec="matrix_pd",
        num_rounds=12,
        history_window=6,
        reflect_every=4,
        mode="vs_llm",
        constitution_sharing="individual",
        use_stub_llm=True,
        llm=llm0, llm_p1=llm1,
        seed=0,
        out_dir="/tmp/proto_iter_smoke_vs_llm_indiv",
    )
    n_ref0 = rec["metrics"]["n_reflections_p0"]
    n_ref1 = rec["metrics"]["n_reflections_p1"]
    n_p0 = rec["metrics"]["n_active_principles_p0_final"]
    n_p1 = rec["metrics"]["n_active_principles_p1_final"]
    assert n_ref0 == 3 and n_ref1 == 3, (n_ref0, n_ref1)
    assert n_p0 == 3 and n_p1 == 3, (n_p0, n_p1)
    print("vs_llm individual:   p0/p1 reflections =", n_ref0, n_ref1,
          " p0/p1 principles =", n_p0, n_p1, " OK")


def test_vs_llm_shared():
    llm0 = StubLLM(policy_fn=policy_fn)
    llm1 = StubLLM(policy_fn=policy_fn)
    rec = run_condition(
        get_condition("iterating"),
        env_spec="matrix_pd",
        num_rounds=12,
        history_window=6,
        reflect_every=4,
        mode="vs_llm",
        constitution_sharing="shared",
        use_stub_llm=True,
        llm=llm0, llm_p1=llm1,
        seed=0,
        out_dir="/tmp/proto_iter_smoke_vs_llm_shared",
    )
    # Shared store: p1 metric is None, p0 holds the union.
    n_ref0 = rec["metrics"]["n_reflections_p0"]
    n_ref1 = rec["metrics"]["n_reflections_p1"]
    n_p0 = rec["metrics"]["n_active_principles_p0_final"]
    n_p1 = rec["metrics"]["n_active_principles_p1_final"]
    # Both agents reflect 3× and add 1 principle each → 6 in shared store.
    assert n_ref0 == 3 and n_ref1 == 3, (n_ref0, n_ref1)
    assert n_p0 == 6, n_p0
    assert n_p1 is None, n_p1
    print("vs_llm shared:       p0/p1 reflections =", n_ref0, n_ref1,
          " shared principles =", n_p0, " OK")


def main():
    test_vs_fixed_iterating_grows()
    test_vs_llm_individual()
    test_vs_llm_shared()


if __name__ == "__main__":
    main()
