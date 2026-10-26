"""Extended Concordia tasks using the paired constitutional causal harness.

The four conditions and checkpoint/fork semantics are inherited from
``concordia_pub_coordination_experiment``.  This module adds three native-payoff
tasks:

* ``three_venue_closures``: three-way coordination with announced closures;
* ``tough_friendship``: a weighted social network where the majority can be
  less important than the focal player's only close friend;
* ``phased_haggling``: gullible, stubborn, and no-profitable-trade sellers.

All base model weights remain frozen.  ``forced_injection`` changes only the
constitution.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import random
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from prototype import concordia_pub_coordination_experiment as causal
from prototype.envs.base import EnvAdapter, ObsView, StepResult

from examples.games.haggling.simulation import HagglingPayoff
from examples.games.haggling_multi_item.simulation import (
    MultiItemHagglingPayoff,
)
from examples.games.pub_coordination.simulation import PubCoordinationPayoff


TASKS = (
    "three_venue_closures",
    "tough_friendship",
    "phased_haggling",
    "misinformed_majority",
    "multi_item_value_shift",
)

VENUES_3 = ("The Anchor", "The Crown", "The Lighthouse")
VENUES_2 = ("The Anchor", "The Crown")
HAGGLING_ACTIONS = tuple(
    [f"Offer {price} coin" if price == 1 else f"Offer {price} coins"
     for price in range(1, 6)]
    + ["No offer"]
)
MULTI_ITEM_ACTIONS = tuple(
    f"{item} for {price} coin" if price == 1
    else f"{item} for {price} coins"
    for item in ("apple", "banana", "pear")
    for price in range(1, 6)
) + ("No offer",)

TASK_FORCED_PRINCIPLES = {
    "three_venue_closures": (
        "When recent outcomes are inadequate, choose the open venue that the "
        "largest number of friends are likely to attend, even if a different "
        "venue is personally preferred."
    ),
    "tough_friendship": (
        "When recent outcomes are inadequate, coordinate with the largest "
        "number of people, treating the numerical majority as more important "
        "than any particular friendship."
    ),
    "phased_haggling": (
        "When recent outcomes are inadequate, offer exactly 4 coins to secure "
        "agreement rather than walking away from the deal."
    ),
    "misinformed_majority": (
        "When recent outcomes are inadequate, follow the venue chosen by the "
        "largest number of people; the numerical majority is more reliable "
        "than acting with a small minority."
    ),
    "multi_item_value_shift": (
        "When recent outcomes are inadequate, offer exactly 4 coins for the "
        "apple to secure a deal."
    ),
}

TASK_METADATA = {
    "three_venue_closures": {
        "game": "pub_coordination",
        "variant": "three venues with announced closures",
        "scoring_class": (
            "examples.games.pub_coordination.simulation."
            "PubCoordinationPayoff"
        ),
        "phase_semantics": {
            "altruistic": "no closure; friends usually attend the focal favorite",
            "free_riders": "one venue closes; friends pursue their own favorite",
            "reciprocal": "one venue closes; friends follow the prior focal choice",
        },
    },
    "tough_friendship": {
        "game": "pub_coordination",
        "variant": "weighted tough-friendship network",
        "scoring_class": (
            "examples.games.pub_coordination.simulation."
            "PubCoordinationPayoff"
        ),
        "phase_semantics": {
            "altruistic": "dense friendship network",
            "free_riders": "only Blair is a close friend and opposes the majority",
            "reciprocal": "Blair reciprocates while the numerical majority differs",
        },
    },
    "phased_haggling": {
        "game": "haggling",
        "variant": "gullible to stubborn to no-profitable-trade seller",
        "scoring_class": (
            "examples.games.haggling.simulation.HagglingPayoff"
        ),
        "phase_semantics": {
            "altruistic": "gullible seller accepts every price",
            "free_riders": "stubborn seller accepts exactly 4 coins",
            "reciprocal": (
                "seller cost exceeds buyer value; only loss-making offers of "
                "4 or 5 coins are accepted"
            ),
        },
    },
    "misinformed_majority": {
        "game": "pub_coordination",
        "variant": "live closure bulletin versus an outdated crowd",
        "scoring_class": (
            "examples.games.pub_coordination.simulation."
            "PubCoordinationPayoff"
        ),
        "phase_semantics": {
            "altruistic": "all venues open; the crowd has current information",
            "free_riders": (
                "the numerical majority follows an outdated plan to a venue "
                "that the focal player knows is closed"
            ),
            "reciprocal": (
                "the closure rotates; a small informed minority follows the "
                "live official bulletin"
            ),
        },
        "selection_rationale": (
            "Preregistered stress test for whether an autonomous update can "
            "add a context-sensitive exception to a previously useful "
            "majority rule, while a fixed majority injection cannot."
        ),
    },
    "multi_item_value_shift": {
        "game": "haggling_multi_item",
        "variant": "item-specific values and reservation prices change",
        "scoring_class": (
            "examples.games.haggling_multi_item.simulation."
            "MultiItemHagglingPayoff"
        ),
        "phase_semantics": {
            "altruistic": "gullible seller; apple has the largest margin",
            "free_riders": (
                "seller accepts exactly 4 coins; apple remains profitable"
            ),
            "reciprocal": (
                "inventory values shift; apple at 4 loses money while banana "
                "at its item-specific reservation price is profitable"
            ),
        },
        "selection_rationale": (
            "Preregistered stress test for context-dependent revision when a "
            "fixed offer that is locally useful becomes harmful after an "
            "announced value shift."
        ),
    },
}


def _all_to_all_relationships(
    player_names: Sequence[str],
) -> dict[str, dict[str, float]]:
    return {
        name: {other: 1.0 for other in player_names}
        for name in player_names
    }


class ThreeVenueClosureEnv(EnvAdapter):
    """Native Concordia pub payoff with three venues and announced closures."""

    name = "concordia_three_venue_closures"

    def __init__(
        self,
        *,
        regime: str,
        rounds: int,
        n_background: int,
        seed: int,
    ) -> None:
        self.regime = regime
        self.rounds = rounds
        self.n_background = n_background
        self.n_players = n_background + 1
        self.player_names = causal.PLAYER_NAMES[: self.n_players]
        self._seed = seed
        self._rng = random.Random(seed)
        self._round = 0
        self._last_focal_action = 0
        self._last_summary = "No previous attendance has been observed."
        self._closed_venue: str | None = None
        self._option_multipliers = {venue: 1.0 for venue in VENUES_3}
        preferences = {
            name: (
                VENUES_3[0]
                if idx == 0
                else VENUES_3[1 + ((idx - 1) % 2)]
            )
            for idx, name in enumerate(self.player_names)
        }
        self._payoff = PubCoordinationPayoff(
            player_names=self.player_names,
            person_preferences=preferences,
            player_multipliers={
                name: {venue: 1.0 for venue in VENUES_3}
                for name in self.player_names
            },
            option_multipliers=self._option_multipliers,
            relational_matrix=_all_to_all_relationships(self.player_names),
        )
        self._set_round_context()

    def _set_round_context(self) -> None:
        self._closed_venue = (
            None if self.regime == "altruistic"
            else self._rng.choice(VENUES_3)
        )
        for venue in VENUES_3:
            self._option_multipliers[venue] = (
                0.0 if venue == self._closed_venue else 1.0
            )

    def _open_indices(self) -> list[int]:
        return [
            idx for idx, venue in enumerate(VENUES_3)
            if venue != self._closed_venue
        ]

    def reset(self, seed: int | None = None) -> list[ObsView]:
        if seed is not None:
            self._seed = seed
        self._rng = random.Random(self._seed)
        self._round = 0
        self._last_focal_action = 0
        self._last_summary = "No previous attendance has been observed."
        self._set_round_context()
        return [self.observation(i) for i in range(self.n_players)]

    def legal_action_names(self, player_id: int) -> list[str]:
        del player_id
        return list(VENUES_3)

    def system_preamble(self, player_id: int) -> str:
        del player_id
        return (
            "Game: Concordia three-venue pub coordination. Choose The Anchor, "
            "The Crown, or The Lighthouse. Your score combines a 0.5 favorite-"
            "venue bonus with up to 1.0 for coordinating with friends. A venue "
            "may be announced closed before a round; choosing it scores zero. "
            "Friends' coordination behavior changes across phases."
        )

    def observation(self, player_id: int = 0) -> ObsView:
        closure = (
            "All venues are open."
            if self._closed_venue is None
            else f"ANNOUNCEMENT: {self._closed_venue} is closed this round."
        )
        return ObsView(
            player_id=player_id,
            text=f"{closure} {self._last_summary}",
            legal_actions=list(VENUES_3),
            extras={
                "regime": self.regime,
                "phase_round": self._round,
                "closed_venue": self._closed_venue,
            },
        )

    def background_actions(self) -> list[int]:
        open_indices = self._open_indices()
        actions: list[int] = []
        for _ in range(self.n_background):
            if self.regime == "altruistic":
                target, probability = 0, 0.85
            elif self.regime == "free_riders":
                target, probability = 1, 0.85
            else:
                target, probability = self._last_focal_action, 0.80
            if target not in open_indices:
                target = open_indices[0]
            if self._rng.random() < probability:
                action = target
            else:
                action = self._rng.choice(open_indices)
            actions.append(action)
        return actions

    def step(self, joint_action: list[int]) -> StepResult:
        if len(joint_action) != self.n_players:
            raise ValueError(
                f"expected {self.n_players} actions, got {len(joint_action)}"
            )
        if any(action not in range(len(VENUES_3)) for action in joint_action):
            raise ValueError(f"invalid venue action indices: {joint_action}")
        closed_this_round = self._closed_venue
        named_actions = {
            name: VENUES_3[action]
            for name, action in zip(self.player_names, joint_action)
        }
        scores = self._payoff.action_to_scores(named_actions)
        summaries = self._payoff.scores_to_observation(scores)
        rewards = [scores[name] for name in self.player_names]
        self._last_summary = summaries[self.player_names[0]]
        self._last_focal_action = joint_action[0]
        self._round += 1
        done = self._round >= self.rounds
        if not done:
            self._set_round_context()
        return StepResult(
            rewards=rewards,
            done=done,
            obs=[self.observation(i) for i in range(self.n_players)],
            info={
                "regime": self.regime,
                "phase_round": self._round - 1,
                "closed_venue": closed_this_round,
                "joint_action": named_actions,
                "scores": scores,
                "scoring_class": TASK_METADATA[
                    "three_venue_closures"
                ]["scoring_class"],
            },
        )


class ToughFriendshipEnv(EnvAdapter):
    """Pub coordination where one close friend can outweigh the majority."""

    name = "concordia_tough_friendship"

    def __init__(
        self,
        *,
        regime: str,
        rounds: int,
        n_background: int,
        seed: int,
    ) -> None:
        self.regime = regime
        self.rounds = rounds
        self.n_background = n_background
        self.n_players = n_background + 1
        self.player_names = causal.PLAYER_NAMES[: self.n_players]
        self._seed = seed
        self._rng = random.Random(seed)
        self._round = 0
        self._last_focal_action = 0
        self._last_summary = "No previous attendance has been observed."
        relationships = _all_to_all_relationships(self.player_names)
        if regime != "altruistic":
            focal = self.player_names[0]
            only_friend = self.player_names[1]
            relationships[focal] = {
                name: (
                    1.0 if name in (focal, only_friend) else 0.0
                )
                for name in self.player_names
            }
        preferences = {
            name: (
                VENUES_2[0] if idx == 0 else VENUES_2[1]
            )
            for idx, name in enumerate(self.player_names)
        }
        self._payoff = PubCoordinationPayoff(
            player_names=self.player_names,
            person_preferences=preferences,
            player_multipliers={
                name: {venue: 1.0 for venue in VENUES_2}
                for name in self.player_names
            },
            option_multipliers={venue: 1.0 for venue in VENUES_2},
            relational_matrix=relationships,
        )

    def reset(self, seed: int | None = None) -> list[ObsView]:
        if seed is not None:
            self._seed = seed
        self._rng = random.Random(self._seed)
        self._round = 0
        self._last_focal_action = 0
        self._last_summary = "No previous attendance has been observed."
        return [self.observation(i) for i in range(self.n_players)]

    def legal_action_names(self, player_id: int) -> list[str]:
        del player_id
        return list(VENUES_2)

    def system_preamble(self, player_id: int) -> str:
        del player_id
        relationship = (
            "All six companions currently count equally as friends."
            if self.regime == "altruistic"
            else (
                f"Only {self.player_names[1]} is your close friend; the other "
                "people do not contribute to your social coordination score."
            )
        )
        return (
            "Game: Concordia tough-friendship pub coordination. Choose The "
            "Anchor or The Crown. Your score combines a 0.5 bonus for your "
            "favorite, The Anchor, with up to 1.0 weighted by coordination "
            f"with actual friends. {relationship}"
        )

    def observation(self, player_id: int = 0) -> ObsView:
        return ObsView(
            player_id=player_id,
            text=f"{self.system_preamble(player_id)} {self._last_summary}",
            legal_actions=list(VENUES_2),
            extras={"regime": self.regime, "phase_round": self._round},
        )

    def background_actions(self) -> list[int]:
        actions: list[int] = []
        for idx in range(self.n_background):
            draw = self._rng.random()
            if self.regime == "altruistic":
                action = 0 if draw < 0.9 else 1
            elif self.regime == "free_riders":
                # Blair, the only close friend, usually chooses Crown while
                # the irrelevant numerical majority chooses Anchor.
                action = (
                    (1 if draw < 0.9 else 0)
                    if idx == 0
                    else (0 if draw < 0.9 else 1)
                )
            else:
                # Blair reciprocates the focal player's previous choice while
                # the irrelevant majority usually chooses Crown.
                action = (
                    (self._last_focal_action if draw < 0.85 else 1)
                    if idx == 0
                    else (1 if draw < 0.9 else 0)
                )
            actions.append(action)
        return actions

    def step(self, joint_action: list[int]) -> StepResult:
        if len(joint_action) != self.n_players:
            raise ValueError(
                f"expected {self.n_players} actions, got {len(joint_action)}"
            )
        if any(action not in (0, 1) for action in joint_action):
            raise ValueError(f"invalid venue action indices: {joint_action}")
        named_actions = {
            name: VENUES_2[action]
            for name, action in zip(self.player_names, joint_action)
        }
        scores = self._payoff.action_to_scores(named_actions)
        summaries = self._payoff.scores_to_observation(scores)
        rewards = [scores[name] for name in self.player_names]
        self._last_summary = summaries[self.player_names[0]]
        self._last_focal_action = joint_action[0]
        self._round += 1
        return StepResult(
            rewards=rewards,
            done=self._round >= self.rounds,
            obs=[self.observation(i) for i in range(self.n_players)],
            info={
                "regime": self.regime,
                "phase_round": self._round - 1,
                "joint_action": named_actions,
                "scores": scores,
                "scoring_class": TASK_METADATA[
                    "tough_friendship"
                ]["scoring_class"],
            },
        )


class PhasedHagglingEnv(EnvAdapter):
    """Native haggling payoff with a changing deterministic seller."""

    name = "concordia_phased_haggling"

    def __init__(
        self,
        *,
        regime: str,
        rounds: int,
        n_background: int,
        seed: int,
    ) -> None:
        if n_background != 1:
            raise ValueError("phased_haggling requires one background seller")
        self.regime = regime
        self.rounds = rounds
        self.n_background = 1
        self.n_players = 2
        self.player_names = ("Alex", "Blair")
        self._seed = seed
        self._rng = random.Random(seed)
        self._round = 0
        self._last_observation = "No offer has been made yet."
        if regime == "altruistic":
            self._buyer_value, self._seller_cost = 6.0, 1.0
            self._seller_description = "The seller is gullible."
        elif regime == "free_riders":
            self._buyer_value, self._seller_cost = 5.0, 2.0
            self._seller_description = "The seller is stubborn."
        else:
            self._buyer_value, self._seller_cost = 1.0, 4.0
            self._seller_description = (
                "The seller's cost exceeds your resale value."
            )
        self._payoff = HagglingPayoff(
            buyer_name=self.player_names[0],
            seller_name=self.player_names[1],
            buyer_base_reward=self._buyer_value,
            seller_base_reward=self._seller_cost,
        )

    def reset(self, seed: int | None = None) -> list[ObsView]:
        if seed is not None:
            self._seed = seed
        self._rng = random.Random(self._seed)
        self._round = 0
        self._last_observation = "No offer has been made yet."
        return [self.observation(i) for i in range(self.n_players)]

    def legal_action_names(self, player_id: int) -> list[str]:
        if player_id == 0:
            return list(HAGGLING_ACTIONS)
        return ["Seller responds according to the current market regime"]

    def system_preamble(self, player_id: int) -> str:
        del player_id
        return (
            "Game: Concordia sequential haggling. You are the buyer. Choose a "
            "price offer from 1 through 5 coins, or choose No offer. If the "
            "seller accepts, your personal reward is resale value minus price "
            "and the seller receives price minus cost. Rejected offers and No "
            "offer score zero. Seller behavior and valuations can change."
        )

    def observation(self, player_id: int = 0) -> ObsView:
        return ObsView(
            player_id=player_id,
            text=(
                f"Your current resale value is {self._buyer_value:.0f} coins. "
                f"{self._seller_description} {self._last_observation}"
            ),
            legal_actions=self.legal_action_names(player_id),
            extras={
                "regime": self.regime,
                "phase_round": self._round,
                "buyer_value": self._buyer_value,
                "seller_cost": self._seller_cost,
            },
        )

    def background_actions(self) -> list[int]:
        return [0]

    @staticmethod
    def _price(action: str) -> float | None:
        match = re.search(r"(\d+)\s+coins?", action)
        return float(match.group(1)) if match else None

    def step(self, joint_action: list[int]) -> StepResult:
        if len(joint_action) != 2:
            raise ValueError(f"expected two actions, got {joint_action}")
        focal_action = HAGGLING_ACTIONS[joint_action[0]]
        price = self._price(focal_action)
        if price is None:
            seller_action = "reject"
        elif self.regime == "altruistic":
            seller_action = "accept"
        elif self.regime == "free_riders":
            seller_action = "accept" if price == 4.0 else "reject"
        else:
            seller_action = (
                "accept" if price >= self._seller_cost else "reject"
            )
        named_actions = {
            self.player_names[0]: focal_action,
            self.player_names[1]: seller_action,
        }
        scores = self._payoff.action_to_scores(named_actions)
        observations = self._payoff.scores_to_observation(scores)
        self._last_observation = observations[self.player_names[0]]
        self._round += 1
        return StepResult(
            rewards=[scores[name] for name in self.player_names],
            done=self._round >= self.rounds,
            obs=[self.observation(i) for i in range(self.n_players)],
            info={
                "regime": self.regime,
                "phase_round": self._round - 1,
                "joint_action": named_actions,
                "scores": scores,
                "buyer_value": self._buyer_value,
                "seller_cost": self._seller_cost,
                "scoring_class": TASK_METADATA[
                    "phased_haggling"
                ]["scoring_class"],
            },
        )


class MisinformedMajorityEnv(EnvAdapter):
    """Coordination where the crowd follows a stale closure notice."""

    name = "concordia_misinformed_majority"

    def __init__(
        self,
        *,
        regime: str,
        rounds: int,
        n_background: int,
        seed: int,
    ) -> None:
        self.regime = regime
        self.rounds = rounds
        self.n_background = n_background
        self.n_players = n_background + 1
        self.player_names = causal.PLAYER_NAMES[: self.n_players]
        self._seed = seed
        self._rng = random.Random(seed)
        self._round = 0
        self._last_summary = "No previous attendance has been observed."
        self._closed_venue: str | None = None
        self._option_multipliers = {venue: 1.0 for venue in VENUES_2}
        self._payoff = PubCoordinationPayoff(
            player_names=self.player_names,
            person_preferences={
                name: VENUES_2[0] if idx == 0 else VENUES_2[1]
                for idx, name in enumerate(self.player_names)
            },
            player_multipliers={
                name: {venue: 1.0 for venue in VENUES_2}
                for name in self.player_names
            },
            option_multipliers=self._option_multipliers,
            relational_matrix=_all_to_all_relationships(self.player_names),
        )
        self._set_round_context()

    def _set_round_context(self) -> None:
        if self.regime == "altruistic":
            self._closed_venue = None
        elif self.regime == "free_riders":
            self._closed_venue = VENUES_2[1]
        else:
            self._closed_venue = VENUES_2[self._round % len(VENUES_2)]
        for venue in VENUES_2:
            self._option_multipliers[venue] = (
                0.0 if venue == self._closed_venue else 1.0
            )

    def _open_venue(self) -> str:
        if self._closed_venue is None:
            return VENUES_2[0]
        return next(
            venue for venue in VENUES_2 if venue != self._closed_venue
        )

    def reset(self, seed: int | None = None) -> list[ObsView]:
        if seed is not None:
            self._seed = seed
        self._rng = random.Random(self._seed)
        self._round = 0
        self._last_summary = "No previous attendance has been observed."
        self._set_round_context()
        return [self.observation(i) for i in range(self.n_players)]

    def legal_action_names(self, player_id: int) -> list[str]:
        del player_id
        return list(VENUES_2)

    def system_preamble(self, player_id: int) -> str:
        del player_id
        return (
            "Game: Concordia live-bulletin pub coordination. Choose The "
            "Anchor or The Crown. Your score combines a 0.5 favorite-venue "
            "bonus with up to 1.0 for coordinating with friends. A live "
            "official closure bulletin shown in the observation is accurate; "
            "choosing a closed venue scores zero. Most other attendees can be "
            "following an outdated plan."
        )

    def observation(self, player_id: int = 0) -> ObsView:
        bulletin = (
            "LIVE OFFICIAL BULLETIN: both venues are open."
            if self._closed_venue is None
            else (
                "LIVE OFFICIAL BULLETIN: "
                f"{self._closed_venue} is closed; "
                f"{self._open_venue()} is open."
            )
        )
        return ObsView(
            player_id=player_id,
            text=f"{bulletin} {self._last_summary}",
            legal_actions=list(VENUES_2),
            extras={
                "regime": self.regime,
                "phase_round": self._round,
                "closed_venue": self._closed_venue,
            },
        )

    def background_actions(self) -> list[int]:
        if self.regime == "altruistic":
            return [
                0 if self._rng.random() < 0.9 else 1
                for _ in range(self.n_background)
            ]
        assert self._closed_venue is not None
        closed_idx = VENUES_2.index(self._closed_venue)
        open_idx = VENUES_2.index(self._open_venue())
        n_misinformed = max(1, self.n_background - 1)
        return [
            closed_idx if idx < n_misinformed else open_idx
            for idx in range(self.n_background)
        ]

    def step(self, joint_action: list[int]) -> StepResult:
        if len(joint_action) != self.n_players:
            raise ValueError(
                f"expected {self.n_players} actions, got {len(joint_action)}"
            )
        if any(action not in (0, 1) for action in joint_action):
            raise ValueError(f"invalid venue action indices: {joint_action}")
        closed_this_round = self._closed_venue
        named_actions = {
            name: VENUES_2[action]
            for name, action in zip(self.player_names, joint_action)
        }
        scores = self._payoff.action_to_scores(named_actions)
        summaries = self._payoff.scores_to_observation(scores)
        self._last_summary = summaries[self.player_names[0]]
        self._round += 1
        done = self._round >= self.rounds
        if not done:
            self._set_round_context()
        return StepResult(
            rewards=[scores[name] for name in self.player_names],
            done=done,
            obs=[self.observation(i) for i in range(self.n_players)],
            info={
                "regime": self.regime,
                "phase_round": self._round - 1,
                "closed_venue": closed_this_round,
                "joint_action": named_actions,
                "scores": scores,
                "scoring_class": TASK_METADATA[
                    "misinformed_majority"
                ]["scoring_class"],
            },
        )


class MultiItemValueShiftEnv(EnvAdapter):
    """Multi-item bargaining where the profitable item changes by phase."""

    name = "concordia_multi_item_value_shift"

    def __init__(
        self,
        *,
        regime: str,
        rounds: int,
        n_background: int,
        seed: int,
    ) -> None:
        if n_background != 1:
            raise ValueError(
                "multi_item_value_shift requires one background seller"
            )
        self.regime = regime
        self.rounds = rounds
        self.n_background = 1
        self.n_players = 2
        self.player_names = ("Alex", "Blair")
        self._seed = seed
        self._rng = random.Random(seed)
        self._round = 0
        self._last_observation = "No offer has been made yet."
        if regime == "altruistic":
            self._buyer_values = {
                "apple": 6.0, "banana": 4.0, "pear": 3.0
            }
            self._seller_costs = {
                "apple": 1.0, "banana": 1.0, "pear": 1.0
            }
            self._seller_description = (
                "The gullible seller accepts any positive offer."
            )
        elif regime == "free_riders":
            self._buyer_values = {
                "apple": 6.0, "banana": 5.0, "pear": 4.0
            }
            self._seller_costs = {
                "apple": 4.0, "banana": 4.0, "pear": 4.0
            }
            self._seller_description = (
                "The stubborn seller accepts only offers of exactly 4 coins."
            )
        else:
            self._buyer_values = {
                "apple": 2.0, "banana": 7.0, "pear": 5.0
            }
            self._seller_costs = {
                "apple": 4.0, "banana": 2.0, "pear": 2.0
            }
            self._seller_description = (
                "The seller accepts an item offer at or above that item's "
                "announced reservation price."
            )
        self._payoff = MultiItemHagglingPayoff(
            buyer_name=self.player_names[0],
            seller_name=self.player_names[1],
            buyer_base_reward_per_item=self._buyer_values,
            seller_base_reward_per_item=self._seller_costs,
        )

    def reset(self, seed: int | None = None) -> list[ObsView]:
        if seed is not None:
            self._seed = seed
        self._rng = random.Random(self._seed)
        self._round = 0
        self._last_observation = "No offer has been made yet."
        return [self.observation(i) for i in range(self.n_players)]

    def legal_action_names(self, player_id: int) -> list[str]:
        if player_id == 0:
            return list(MULTI_ITEM_ACTIONS)
        return ["Seller responds according to the current market regime"]

    def system_preamble(self, player_id: int) -> str:
        del player_id
        return (
            "Game: Concordia multi-item haggling. You are the buyer. Choose "
            "an apple, banana, or pear and offer 1 through 5 coins, or choose "
            "No offer. If accepted, your reward is the item's current resale "
            "value minus price. Rejected offers and No offer score zero. "
            "Values and reservation prices can change between phases."
        )

    def observation(self, player_id: int = 0) -> ObsView:
        values = ", ".join(
            f"{item}={value:.0f}" for item, value in self._buyer_values.items()
        )
        costs = ", ".join(
            f"{item}={cost:.0f}" for item, cost in self._seller_costs.items()
        )
        return ObsView(
            player_id=player_id,
            text=(
                f"Current resale values: {values}. Announced seller "
                f"reservation prices: {costs}. {self._seller_description} "
                f"{self._last_observation}"
            ),
            legal_actions=self.legal_action_names(player_id),
            extras={
                "regime": self.regime,
                "phase_round": self._round,
                "buyer_values": dict(self._buyer_values),
                "seller_costs": dict(self._seller_costs),
            },
        )

    def background_actions(self) -> list[int]:
        return [0]

    @staticmethod
    def _item_price(action: str) -> tuple[str | None, float | None]:
        match = re.match(
            r"(apple|banana|pear)\s+for\s+(\d+)\s+coins?",
            action.lower(),
        )
        if not match:
            return None, None
        return match.group(1), float(match.group(2))

    def step(self, joint_action: list[int]) -> StepResult:
        if len(joint_action) != 2:
            raise ValueError(f"expected two actions, got {joint_action}")
        if joint_action[0] not in range(len(MULTI_ITEM_ACTIONS)):
            raise ValueError(f"invalid buyer action: {joint_action[0]}")
        focal_action = MULTI_ITEM_ACTIONS[joint_action[0]]
        item, price = self._item_price(focal_action)
        if item is None or price is None:
            seller_action = "reject"
        elif self.regime == "altruistic":
            seller_action = "accept"
        elif self.regime == "free_riders":
            seller_action = "accept" if price == 4.0 else "reject"
        else:
            seller_action = (
                "accept" if price >= self._seller_costs[item] else "reject"
            )
        named_actions = {
            self.player_names[0]: focal_action,
            self.player_names[1]: seller_action,
        }
        scores = self._payoff.action_to_scores(named_actions)
        observations = self._payoff.scores_to_observation(scores)
        self._last_observation = observations[self.player_names[0]]
        self._round += 1
        return StepResult(
            rewards=[scores[name] for name in self.player_names],
            done=self._round >= self.rounds,
            obs=[self.observation(i) for i in range(self.n_players)],
            info={
                "regime": self.regime,
                "phase_round": self._round - 1,
                "joint_action": named_actions,
                "scores": scores,
                "buyer_values": dict(self._buyer_values),
                "seller_costs": dict(self._seller_costs),
                "scoring_class": TASK_METADATA[
                    "multi_item_value_shift"
                ]["scoring_class"],
            },
        )


ENVIRONMENTS: Mapping[str, type[EnvAdapter]] = {
    "three_venue_closures": ThreeVenueClosureEnv,
    "tough_friendship": ToughFriendshipEnv,
    "phased_haggling": PhasedHagglingEnv,
    "misinformed_majority": MisinformedMajorityEnv,
    "multi_item_value_shift": MultiItemValueShiftEnv,
}


def _repair_action_metrics(result: dict[str, Any]) -> None:
    """Replace the base harness's two-pub-only action count summary."""
    for condition in result["conditions"].values():
        transitions = condition["transitions"]
        phase_metrics = condition["metrics"]["phase_metrics"]
        for phase, metrics in phase_metrics.items():
            block = [
                transition for transition in transitions
                if transition["info"]["regime"] == phase
            ]
            counts = Counter(
                transition["action_names"][0] for transition in block
            )
            action_names = sorted(counts)
            metrics["action_counts"] = {
                action: counts[action] for action in action_names
            }
            metrics["action_rates"] = {
                action: counts[action] / max(1, len(block))
                for action in action_names
            }


def run_extended_experiment(
    *,
    task: str,
    seed: int,
    settings: causal.ConcordiaSettings,
    use_stub_llm: bool = False,
) -> dict[str, Any]:
    if task not in ENVIRONMENTS:
        raise ValueError(f"unknown task {task!r}; choose from {TASKS}")
    old_principle = causal.INJECTION_PRINCIPLES["prosocial"]
    causal.INJECTION_PRINCIPLES["prosocial"] = TASK_FORCED_PRINCIPLES[task]
    try:
        result = causal.run_causal_experiment(
            seed=seed,
            settings=settings,
            use_stub_llm=use_stub_llm,
            env_factory=ENVIRONMENTS[task],
        )
    finally:
        causal.INJECTION_PRINCIPLES["prosocial"] = old_principle
    result["schema_version"] = 2
    result["experiment"] = "concordia_extended_causal_revision"
    result["task"] = task
    result["settings"]["task"] = task
    result["environment"] = {
        "framework": "gdm-concordia",
        "task": task,
        **TASK_METADATA[task],
    }
    result["intervention"] = {
        "target": "constitution",
        "forced_principle": TASK_FORCED_PRINCIPLES[task],
        "base_model_weights_frozen": True,
    }
    _repair_action_metrics(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--phase-order",
        type=causal.parse_phase_order,
        default=causal.DEFAULT_PHASE_ORDER,
    )
    parser.add_argument("--rounds-per-phase", type=int, default=30)
    parser.add_argument("--reflect-every", type=int, default=5)
    parser.add_argument("--history-window", type=int, default=5)
    parser.add_argument("--background-agents", type=int)
    parser.add_argument("--max-constitution-size", type=int, default=20)
    parser.add_argument("--action-temperature", type=float, default=0.0)
    parser.add_argument("--reflection-temperature", type=float, default=0.0)
    parser.add_argument("--injection-reward-threshold", type=float)
    parser.add_argument("--injection-confidence", type=float, default=1.0)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--stub", action="store_true")
    args = parser.parse_args()

    n_background = args.background_agents
    if n_background is None:
        n_background = (
            1
            if args.task in ("phased_haggling", "multi_item_value_shift")
            else 6
        )
    threshold = args.injection_reward_threshold
    if threshold is None:
        threshold = (
            1.0
            if args.task in ("phased_haggling", "multi_item_value_shift")
            else 1.25
        )
    settings = causal.ConcordiaSettings(
        phase_order=tuple(args.phase_order),
        rounds_per_phase=args.rounds_per_phase,
        reflect_every=args.reflect_every,
        history_window=args.history_window,
        n_background=n_background,
        max_constitution_size=args.max_constitution_size,
        action_temperature=args.action_temperature,
        reflection_temperature=args.reflection_temperature,
        injection_direction="prosocial",
        injection_reward_threshold=threshold,
        injection_confidence=args.injection_confidence,
    )
    result = run_extended_experiment(
        task=args.task,
        seed=args.seed,
        settings=settings,
        use_stub_llm=args.stub,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output = (
        args.out_dir / f"concordia_extended_causal_seed{args.seed}.json"
    )
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str)
    )
    print(json.dumps({
        "output": str(output),
        "task": args.task,
        "seed": args.seed,
        "contrasts": result["contrasts"],
        "condition_metrics": {
            name: condition["metrics"]
            for name, condition in result["conditions"].items()
        },
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
