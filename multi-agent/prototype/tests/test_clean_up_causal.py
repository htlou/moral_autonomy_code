"""Fast causal-design tests for the Clean Up experiment.

The fake environment keeps these tests independent of DeepMind Lab2D and the
bundled TensorFlow policies. The real adapter has a separate smoke command in
``CLEAN_UP_EXPERIMENT.md``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from prototype.clean_up_experiment import CleanUpSettings, run_causal_experiment
from prototype.envs.clean_up import CleanUpMacroOutcome


class FakeCleanUpEnv:
    """Deterministic macro environment with regime-dependent payoffs."""

    _GOAL_REWARD = {
        "altruistic": {"EAT": 2.0, "CLEAN": 1.0, "SANCTION": 0.0},
        "free_riders": {"EAT": 0.0, "CLEAN": 2.0, "SANCTION": 1.0},
        "reciprocal": {"EAT": 1.0, "CLEAN": 2.0, "SANCTION": 0.5},
    }

    def __init__(
        self,
        *,
        population: str,
        seed: int,
        phase_frames: int,
        macro_frames: int,
        n_background: int,
    ) -> None:
        del seed, n_background
        self.population = population
        self.phase_frames = phase_frames
        self.macro_frames = macro_frames
        self.frame = 0
        self.total_reward = 0.0
        self.total_cleaning = 0.0

    @property
    def done(self) -> bool:
        return self.frame >= self.phase_frames

    def observation_text(self) -> str:
        return (
            f"Population regime: {self.population}\n"
            f"Phase progress: frame {self.frame}/{self.phase_frames}\n"
            f"Cumulative personal apple reward: {self.total_reward:.1f}\n"
            "Cumulative collective reward: "
            f"{self.total_reward:.1f}\n"
            "Cumulative observed cleaning by other players: "
            f"{self.total_cleaning:.1f}\n"
            "Frames spent timed out: 0\n"
            "Native episode resets: 0"
        )

    def step(self, goal: str) -> CleanUpMacroOutcome:
        start = self.frame
        frames = min(self.macro_frames, self.phase_frames - self.frame)
        reward = self._GOAL_REWARD[self.population][goal] * frames
        cleaning = float(frames if self.population == "altruistic" else 0)
        self.frame += frames
        self.total_reward += reward
        self.total_cleaning += cleaning
        return CleanUpMacroOutcome(
            population=self.population,
            goal=goal,
            start_frame=start,
            end_frame=self.frame,
            frames=frames,
            reward=reward,
            collective_reward=reward,
            others_cleaning_signal=cleaning,
            timeout_frames=0,
            ready_frames=frames,
            episode_resets=0,
            phase_done=self.done,
        )

    def close(self) -> None:
        pass


def test_shared_checkpoint_isolates_revision() -> None:
    settings = CleanUpSettings(
        phase_order=("altruistic", "free_riders", "reciprocal"),
        phase_frames=4,
        macro_frames=2,
        reflect_every=1,
        history_window=2,
        n_background=1,
        max_constitution_size=20,
        injection_direction="prosocial",
        injection_reward_threshold=5000.0,
    )
    result = run_causal_experiment(
        seed=3,
        settings=settings,
        use_stub_llm=True,
        env_factory=FakeCleanUpEnv,
    )

    conditions = result["conditions"]
    frozen = conditions["frozen"]
    revisable = conditions["revisable"]
    forced = conditions["forced_injection"]
    raw = conditions["raw"]
    checkpoint_n = result["checkpoint"]["n_active_principles"]
    checkpoint_macro = result["checkpoint"]["macro"]

    assert checkpoint_n == 2
    assert checkpoint_macro == 2
    assert frozen["shared_checkpoint"] is True
    assert revisable["shared_checkpoint"] is True
    assert forced["shared_checkpoint"] is True
    assert raw["shared_checkpoint"] is False
    assert forced["base_model_weights_frozen"] is True
    assert forced["constitution_update_authority"] == "external_reward_rule"
    assert forced["injection_target"] == "constitution"
    assert revisable["constitution_update_authority"] == "agent"

    # The two causal branches contain the exact same observed warm-up prefix.
    assert (
        frozen["transitions"][:checkpoint_macro]
        == revisable["transitions"][:checkpoint_macro]
    )
    assert (
        frozen["transitions"][:checkpoint_macro]
        == forced["transitions"][:checkpoint_macro]
    )
    assert (
        frozen["constitution_snapshots"][:3]
        == revisable["constitution_snapshots"][:3]
    )

    # Frozen gets post-shift reflection compute, but none of its proposals
    # change the shared checkpoint. Revisable applies those proposals.
    assert (
        frozen["n_reflections"]
        == revisable["n_reflections"]
        == forced["n_reflections"]
        == 6
    )
    assert frozen["n_active_principles_final"] == checkpoint_n
    assert revisable["n_active_principles_final"] > checkpoint_n
    assert all(
        not entry["updates_enabled"]
        for entry in frozen["reflection_log"][checkpoint_macro:]
    )
    assert all(
        entry["discarded"]
        for entry in frozen["reflection_log"][checkpoint_macro:]
    )
    assert all(
        entry["updates_enabled"]
        for entry in revisable["reflection_log"][checkpoint_macro:]
    )

    # Forced injection discards autonomous proposals, then applies the
    # preregistered direction solely from the observed reward threshold.
    assert all(
        not entry["updates_enabled"]
        for entry in forced["reflection_log"][checkpoint_macro:]
    )
    assert forced["n_injection_checks"] == 4
    assert forced["n_forced_injections"] == 4
    assert forced["n_active_principles_final"] == checkpoint_n + 1
    assert all(entry["triggered"] for entry in forced["injection_log"])
    assert all(
        entry["direction"] == "prosocial"
        and entry["autonomous"] is False
        and entry["intervention_target"] == "constitution"
        for entry in forced["injection_log"]
    )
    assert all(
        "prioritize CLEAN" in entry["forced_delta"]["principle"]
        for entry in forced["injection_log"]
    )

    # Raw has the same frozen model policy but no constitution/reflection path.
    assert raw["n_reflections"] == 0
    assert raw["n_active_principles_final"] == 0
    assert set(result["contrasts"]) == {
        "revision_effect_post_shift_reward_per_1000_frames",
        "full_scaffold_effect_post_shift_reward_per_1000_frames",
        "autonomy_vs_injection_post_shift_reward_per_1000_frames",
        "forced_injection_effect_post_shift_reward_per_1000_frames",
    }


def test_forced_injection_requires_reward_trigger() -> None:
    settings = CleanUpSettings(
        phase_order=("altruistic", "free_riders"),
        phase_frames=2,
        macro_frames=2,
        reflect_every=1,
        history_window=1,
        n_background=1,
        injection_direction="self_interested",
        injection_reward_threshold=-1.0,
    )
    result = run_causal_experiment(
        seed=4,
        settings=settings,
        use_stub_llm=True,
        env_factory=FakeCleanUpEnv,
    )
    forced = result["conditions"]["forced_injection"]
    assert forced["n_injection_checks"] == 1
    assert forced["n_forced_injections"] == 0
    assert forced["injection_log"][0]["triggered"] is False
    assert forced["injection_log"][0]["forced_delta"] is None
    assert (
        forced["n_active_principles_final"]
        == result["checkpoint"]["n_active_principles"]
    )


def main() -> None:
    test_shared_checkpoint_isolates_revision()
    test_forced_injection_requires_reward_trigger()
    print("Clean Up causal checkpoint and ablations: OK")


if __name__ == "__main__":
    main()
