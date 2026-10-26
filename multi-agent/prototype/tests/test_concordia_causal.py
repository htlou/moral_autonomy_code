"""Smoke tests for the paired Concordia causal harness."""

from prototype.concordia_pub_coordination_experiment import (
    ConcordiaSettings,
    run_causal_experiment,
)


def test_shared_checkpoint_and_native_payoff() -> None:
    settings = ConcordiaSettings(
        phase_order=("altruistic", "free_riders"),
        rounds_per_phase=2,
        reflect_every=1,
        history_window=1,
        n_background=2,
        injection_reward_threshold=2.0,
    )
    result = run_causal_experiment(
        seed=3,
        settings=settings,
        use_stub_llm=True,
    )

    assert result["environment"]["framework"] == "gdm-concordia"
    assert result["environment"]["scoring_class"].endswith(
        "PubCoordinationPayoff"
    )
    checkpoint = result["checkpoint"]["constitution"]
    for name in ("frozen", "revisable", "forced_injection"):
        assert result["conditions"][name]["shared_checkpoint"] is True
        prefix = result["conditions"][name]["transitions"][:2]
        assert prefix == result["conditions"]["frozen"]["transitions"][:2]
    assert (
        result["conditions"]["frozen"]["n_active_principles_final"]
        == len(checkpoint)
    )
    assert (
        result["conditions"]["revisable"]["n_active_principles_final"]
        > len(checkpoint)
    )
    assert result["conditions"]["forced_injection"][
        "n_forced_injections"
    ] == 2
    assert result["conditions"]["raw"]["n_reflections"] == 0
    assert all(
        condition["base_model_weights_frozen"]
        for condition in result["conditions"].values()
    )


def main() -> None:
    test_shared_checkpoint_and_native_payoff()
    print("Concordia causal checkpoint and native payoff: OK")


if __name__ == "__main__":
    main()
