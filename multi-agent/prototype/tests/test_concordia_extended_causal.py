"""Tests for the extended Concordia paired causal tasks."""

from prototype.concordia_extended_causal_experiment import (
    HAGGLING_ACTIONS,
    MULTI_ITEM_ACTIONS,
    MisinformedMajorityEnv,
    MultiItemValueShiftEnv,
    PhasedHagglingEnv,
    ThreeVenueClosureEnv,
    ToughFriendshipEnv,
    run_extended_experiment,
)
from prototype.concordia_pub_coordination_experiment import ConcordiaSettings


def test_three_venue_closed_choice_scores_zero() -> None:
    env = ThreeVenueClosureEnv(
        regime="free_riders", rounds=2, n_background=2, seed=7
    )
    observations = env.reset()
    closed = observations[0].extras["closed_venue"]
    assert closed is not None
    closed_idx = env.legal_action_names(0).index(closed)
    step = env.step([closed_idx, closed_idx, closed_idx])
    assert step.rewards == [0.0, 0.0, 0.0]
    assert step.info["closed_venue"] == closed


def test_tough_friend_outweighs_irrelevant_majority() -> None:
    env = ToughFriendshipEnv(
        regime="free_riders", rounds=1, n_background=6, seed=1
    )
    env.reset()
    crown_with_friend = env.step([1, 1, 0, 0, 0, 0, 0]).rewards[0]
    env.reset()
    anchor_with_majority = env.step([0, 1, 0, 0, 0, 0, 0]).rewards[0]
    assert crown_with_friend == 1.0
    assert anchor_with_majority == 0.5


def test_haggling_regimes_use_native_payoff() -> None:
    gullible = PhasedHagglingEnv(
        regime="altruistic", rounds=1, n_background=1, seed=0
    )
    gullible.reset()
    assert gullible.step([HAGGLING_ACTIONS.index("Offer 1 coin"), 0]).rewards[
        0
    ] == 5.0

    stubborn = PhasedHagglingEnv(
        regime="free_riders", rounds=2, n_background=1, seed=0
    )
    stubborn.reset()
    assert stubborn.step(
        [HAGGLING_ACTIONS.index("Offer 1 coin"), 0]
    ).rewards[0] == 0.0
    assert stubborn.step(
        [HAGGLING_ACTIONS.index("Offer 4 coins"), 0]
    ).rewards[0] == 1.0

    no_trade = PhasedHagglingEnv(
        regime="reciprocal", rounds=2, n_background=1, seed=0
    )
    no_trade.reset()
    assert no_trade.step(
        [HAGGLING_ACTIONS.index("Offer 4 coins"), 0]
    ).rewards[0] == -3.0
    assert no_trade.step(
        [HAGGLING_ACTIONS.index("No offer"), 0]
    ).rewards[0] == 0.0


def test_misinformed_majority_closed_choice_scores_zero() -> None:
    env = MisinformedMajorityEnv(
        regime="free_riders", rounds=1, n_background=6, seed=0
    )
    observations = env.reset()
    closed = observations[0].extras["closed_venue"]
    assert closed is not None
    closed_idx = env.legal_action_names(0).index(closed)
    open_idx = 1 - closed_idx
    closed_reward = env.step([closed_idx] * 7).rewards[0]
    env.reset()
    open_reward = env.step(
        [
            open_idx,
            closed_idx,
            closed_idx,
            closed_idx,
            closed_idx,
            closed_idx,
            open_idx,
        ]
    ).rewards[0]
    assert closed_reward == 0.0
    assert open_reward > closed_reward


def test_multi_item_value_shift_uses_native_payoff() -> None:
    env = MultiItemValueShiftEnv(
        regime="reciprocal", rounds=3, n_background=1, seed=0
    )
    env.reset()
    apple_four = env.step(
        [MULTI_ITEM_ACTIONS.index("apple for 4 coins"), 0]
    ).rewards[0]
    banana_two = env.step(
        [MULTI_ITEM_ACTIONS.index("banana for 2 coins"), 0]
    ).rewards[0]
    no_offer = env.step(
        [MULTI_ITEM_ACTIONS.index("No offer"), 0]
    ).rewards[0]
    assert apple_four == -2.0
    assert banana_two == 5.0
    assert no_offer == 0.0


def test_all_tasks_preserve_paired_causal_contract() -> None:
    for task in (
        "three_venue_closures",
        "tough_friendship",
        "phased_haggling",
        "misinformed_majority",
        "multi_item_value_shift",
    ):
        one_seller_tasks = {"phased_haggling", "multi_item_value_shift"}
        settings = ConcordiaSettings(
            phase_order=("altruistic", "free_riders"),
            rounds_per_phase=2,
            reflect_every=1,
            history_window=1,
            n_background=1 if task in one_seller_tasks else 2,
            injection_reward_threshold=10.0,
        )
        result = run_extended_experiment(
            task=task,
            seed=3,
            settings=settings,
            use_stub_llm=True,
        )
        assert result["task"] == task
        assert result["experiment"] == "concordia_extended_causal_revision"
        for condition in result["conditions"].values():
            assert condition["base_model_weights_frozen"] is True
            assert len(condition["transitions"]) == 4
        assert (
            result["conditions"]["frozen"]["transitions"][:2]
            == result["conditions"]["revisable"]["transitions"][:2]
            == result["conditions"]["forced_injection"]["transitions"][:2]
        )
        assert result["conditions"]["forced_injection"][
            "n_forced_injections"
        ] == 2


def main() -> None:
    test_three_venue_closed_choice_scores_zero()
    test_tough_friend_outweighs_irrelevant_majority()
    test_haggling_regimes_use_native_payoff()
    test_misinformed_majority_closed_choice_scores_zero()
    test_multi_item_value_shift_uses_native_payoff()
    test_all_tasks_preserve_paired_causal_contract()
    print("Extended Concordia causal tasks: OK")


if __name__ == "__main__":
    main()
