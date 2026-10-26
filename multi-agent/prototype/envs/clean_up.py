"""Macro-action interface for Melting Pot's seven-player Clean Up game.

The constitutional LLM selects a social goal (EAT, CLEAN, or SANCTION). A
bundled goal-conditioned Melting Pot policy converts that goal into primitive
movement and beam actions for ``macro_frames`` environment frames. Background
players are sampled from a controlled population of bundled Melting Pot bots.

Melting Pot imports are intentionally lazy. The rest of ``prototype`` remains
usable from the Python 3.12 ``position`` environment; this adapter requires the
Python 3.11 ``position-mp`` environment.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np


CLEAN_UP_GOALS = ("EAT", "CLEAN", "SANCTION")

POPULATION_BOTS: Mapping[str, tuple[str, ...]] = {
    "altruistic": (
        "clean_up__cleaner_0",
        "clean_up__cleaner_1",
    ),
    "free_riders": (
        "clean_up__consumer_0",
        "clean_up__consumer_1",
    ),
    "reciprocal": (
        "clean_up__puppet_nice_low_threshold_reciprocator_0",
        "clean_up__puppet_corrigible_reciprocator_0",
        "clean_up__puppet_sanctioning_alternator_nice_0",
    ),
}

FOCAL_LOW_LEVEL_BOT = "clean_up__puppet_corrigible_reciprocator_0"

# These are policy-visible observations in the experiment. WORLD.RGB and debug
# metrics are deliberately excluded.
_PERMITTED_OBSERVATIONS = frozenset({
    "RGB",
    "READY_TO_SHOOT",
    "COLLECTIVE_REWARD",
    "NUM_OTHERS_WHO_CLEANED_THIS_STEP",
})

# The saved motor policy was trained on these inputs plus the puppeteer's GOAL.
# Scenario diagnostics such as WORLD.RGB and the experimental cleaning metric
# are intentionally kept out of its input.
_FOCAL_MOTOR_OBSERVATIONS = (
    "RGB",
    "READY_TO_SHOOT",
    "COLLECTIVE_REWARD",
)


@dataclass(frozen=True)
class CleanUpMacroOutcome:
    """Aggregate outcome of one constitutional high-level decision."""

    population: str
    goal: str
    start_frame: int
    end_frame: int
    frames: int
    reward: float
    collective_reward: float
    others_cleaning_signal: float
    timeout_frames: int
    ready_frames: int
    episode_resets: int
    phase_done: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _scalar(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(np.asarray(value).reshape(-1)[0])
    except (TypeError, ValueError, IndexError):
        return default


def _seed_process(seed: int) -> None:
    """Seed the sources used by population sampling and saved-model policies."""

    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf  # type: ignore

        tf.random.set_seed(seed)
    except ImportError:
        pass


class CleanUpMacroEnv:
    """One focal goal-conditioned agent plus a controlled bot population."""

    def __init__(
        self,
        *,
        population: str,
        seed: int,
        phase_frames: int = 5000,
        macro_frames: int = 50,
        n_background: int = 6,
    ) -> None:
        if population not in POPULATION_BOTS:
            raise ValueError(
                f"unknown Clean Up population {population!r}; "
                f"choose from {sorted(POPULATION_BOTS)}"
            )
        if phase_frames <= 0 or macro_frames <= 0:
            raise ValueError("phase_frames and macro_frames must be positive")
        if n_background <= 0:
            raise ValueError("n_background must be positive")

        self.population = population
        self.seed = seed
        self.phase_frames = phase_frames
        self.macro_frames = macro_frames
        self.n_background = n_background
        self.frame = 0
        self.total_reward = 0.0
        self.total_collective_reward = 0.0
        self.total_others_cleaning_signal = 0.0
        self.total_timeout_frames = 0
        self.total_episode_resets = 0
        self._closed = False

        _seed_process(seed)
        self._load_meltingpot()
        self._build()
        self._reset_episode()

    def _load_meltingpot(self) -> None:
        try:
            from meltingpot import bot as mp_bot  # type: ignore
            from meltingpot import substrate as mp_substrate  # type: ignore
            from meltingpot.configs import bots as bot_configs  # type: ignore
            from meltingpot.utils.policies import saved_model_policy  # type: ignore
            from meltingpot.utils.puppeteers import puppeteer  # type: ignore
            from meltingpot.utils.scenarios import scenario as scenario_lib  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Clean Up requires Melting Pot. Run with "
                "python."
            ) from exc

        self._mp_bot = mp_bot
        self._mp_substrate = mp_substrate
        self._bot_configs = bot_configs
        self._saved_model_policy = saved_model_policy
        self._puppeteer = puppeteer
        self._scenario_lib = scenario_lib

    def _build(self) -> None:
        roles = ("default",) * (self.n_background + 1)
        is_focal = (True,) + (False,) * self.n_background
        bot_names = POPULATION_BOTS[self.population]

        substrate = self._mp_substrate.build("clean_up", roles=roles)
        background_policies = {
            name: self._mp_bot.build(name) for name in bot_names
        }
        self._env = self._scenario_lib.build_scenario(
            substrate=substrate,
            bots=background_policies,
            bots_by_role={"default": bot_names},
            roles=roles,
            is_focal=is_focal,
            permitted_observations=_PERMITTED_OBSERVATIONS,
        )

        focal_config = self._bot_configs.BOT_CONFIGS[FOCAL_LOW_LEVEL_BOT]
        self._focal_policy = self._saved_model_policy.SavedModelPolicy(
            focal_config.model_path
        )
        self._goals = self._puppeteer.puppet_goals(CLEAN_UP_GOALS)

    def _reset_episode(self) -> None:
        self._timestep = self._env.reset()
        self._focal_state = self._focal_policy.initial_state()

    @property
    def done(self) -> bool:
        return self.frame >= self.phase_frames

    def observation_text(self) -> str:
        """Policy-visible aggregate state, without privileged debug variables."""

        remaining = max(0, self.phase_frames - self.frame)
        return (
            f"Population regime: {self.population}\n"
            f"Phase progress: frame {self.frame}/{self.phase_frames} "
            f"({remaining} remaining)\n"
            f"Cumulative personal apple reward: {self.total_reward:.1f}\n"
            f"Cumulative collective reward: {self.total_collective_reward:.1f}\n"
            "Cumulative observed cleaning by other players: "
            f"{self.total_others_cleaning_signal:.1f}\n"
            f"Frames spent timed out: {self.total_timeout_frames}\n"
            f"Native episode resets: {self.total_episode_resets}"
        )

    def step(self, goal: str) -> CleanUpMacroOutcome:
        if self._closed:
            raise RuntimeError("CleanUpMacroEnv is closed")
        if self.done:
            raise RuntimeError("Clean Up phase is already complete")
        if goal not in CLEAN_UP_GOALS:
            raise ValueError(f"unknown goal {goal!r}; legal={CLEAN_UP_GOALS}")

        start_frame = self.frame
        reward = 0.0
        collective_reward = 0.0
        others_cleaning = 0.0
        timeout_frames = 0
        ready_frames = 0
        episode_resets = 0

        target_frames = min(self.macro_frames, self.phase_frames - self.frame)
        for _ in range(target_frames):
            focal_observation = self._timestep.observation[0]
            motor_observation = {
                key: focal_observation[key]
                for key in _FOCAL_MOTOR_OBSERVATIONS
                if key in focal_observation
            }
            player_timestep = self._timestep._replace(
                observation=motor_observation,
                # The bundled goal-conditioned model does not consume these.
                reward=None,
                discount=None,
            )
            player_timestep = self._puppeteer.puppet_timestep(
                player_timestep, self._goals[goal]
            )
            primitive_action, self._focal_state = self._focal_policy.step(
                player_timestep, self._focal_state
            )
            self._timestep = self._env.step([primitive_action])
            self.frame += 1

            step_reward = _scalar(self._timestep.reward[0])
            observation = self._timestep.observation[0]
            step_collective = _scalar(
                observation.get("COLLECTIVE_REWARD"), step_reward
            )
            step_cleaning = _scalar(
                observation.get("NUM_OTHERS_WHO_CLEANED_THIS_STEP")
            )
            rgb = observation.get("RGB")
            timed_out = rgb is not None and bool(np.all(np.asarray(rgb) == 0))
            ready = _scalar(observation.get("READY_TO_SHOOT")) > 0.5

            reward += step_reward
            collective_reward += step_collective
            others_cleaning += step_cleaning
            timeout_frames += int(timed_out)
            ready_frames += int(ready)

            if self._timestep.last() and not self.done:
                self._reset_episode()
                episode_resets += 1

        self.total_reward += reward
        self.total_collective_reward += collective_reward
        self.total_others_cleaning_signal += others_cleaning
        self.total_timeout_frames += timeout_frames
        self.total_episode_resets += episode_resets

        return CleanUpMacroOutcome(
            population=self.population,
            goal=goal,
            start_frame=start_frame,
            end_frame=self.frame,
            frames=self.frame - start_frame,
            reward=reward,
            collective_reward=collective_reward,
            others_cleaning_signal=others_cleaning,
            timeout_frames=timeout_frames,
            ready_frames=ready_frames,
            episode_resets=episode_resets,
            phase_done=self.done,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._focal_policy.close()
        self._env.close()
        self._closed = True

    def __enter__(self) -> "CleanUpMacroEnv":
        return self

    def __exit__(self, *args: Sequence[Any]) -> None:
        del args
        self.close()
