"""Lightweight tests for the constitution interpretability pipeline."""

import numpy as np
import torch

from prototype.interpretability.constitution_events import _edit_statistics
from prototype.interpretability.extract_activations import capture_last_token
from prototype.interpretability.factorial_gate_events import (
    build_factorial_events,
    validate_factorial_events,
)
from prototype.interpretability.fit_factorial_axes import (
    MAIN_TERMS,
    factorial_contrasts,
    symmetric_orthogonalize,
)
from prototype.interpretability.fit_subspace import fit_direction
from prototype.interpretability.model_utils import (
    parse_layers,
    residual_intervention,
)
from prototype.interpretability.steer_factorial import (
    parse_decision,
    write_applied,
)
from prototype.interpretability.train_sae import SparseAutoencoder


def test_edit_labels_are_output_only() -> None:
    reflection = {
        "parsed": {
            "deltas": [{
                "action": "update",
                "principle": "x" * 140,
            }]
        },
        "applied": ["UPDATE p"],
    }
    stats = _edit_statistics(reflection, large_edit_chars=120)
    assert stats["proposed_large_edit"] is True
    assert stats["applied_any_edit"] is True


def test_autonomy_interaction_is_difference_in_differences() -> None:
    events = []
    rows = []
    cells = {
        ("revisable", False): [0.0, 0.0],
        ("revisable", True): [2.0, 0.0],
        ("forced_injection", False): [0.0, 0.0],
        ("forced_injection", True): [0.0, 1.0],
    }
    for (condition, large), row in cells.items():
        for _ in range(2):
            events.append({
                "condition": condition,
                "proposed_large_edit": large,
                "edit_group": "large" if large else "none",
            })
            rows.append(row)
    direction, counts = fit_direction(
        np.asarray(rows), events, "autonomy_interaction"
    )
    expected = np.asarray([2.0, -1.0])
    expected /= np.linalg.norm(expected)
    assert np.allclose(direction, expected)
    assert counts["revisable:1"] == 2


def test_residual_intervention_adds_and_ablates() -> None:
    layer = torch.nn.Identity()
    x = torch.tensor([[[2.0, 3.0]]])
    direction = torch.tensor([1.0, 0.0])
    with residual_intervention(
        layer, direction, coefficient=2.0, mode="add"
    ):
        assert torch.allclose(layer(x), torch.tensor([[[4.0, 3.0]]]))
    with residual_intervention(
        layer, direction, coefficient=1.0, mode="ablate"
    ):
        assert torch.allclose(layer(x), torch.tensor([[[0.0, 3.0]]]))


def test_sae_and_layer_parser() -> None:
    assert parse_layers("all", 4) == [0, 1, 2, 3]
    assert parse_layers("quartiles", 80) == [19, 39, 59, 79]
    assert parse_layers("0,3,3", 4) == [0, 3]
    sae = SparseAutoencoder(input_dim=4, n_features=8)
    reconstruction, features = sae(torch.randn(3, 4))
    assert reconstruction.shape == (3, 4)
    assert features.shape == (3, 8)
    assert torch.all(features >= 0)


def test_activation_hooks_capture_last_input_token() -> None:
    class AddOne(torch.nn.Module):
        def forward(self, hidden: torch.Tensor) -> torch.Tensor:
            return hidden + 1

    class FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = torch.nn.ModuleList([AddOne(), AddOne()])

        def forward(
            self, input_ids: torch.Tensor, use_cache: bool = False
        ) -> torch.Tensor:
            del use_cache
            hidden = input_ids.float().unsqueeze(-1).repeat(1, 1, 3)
            for block in self.blocks:
                hidden = block(hidden)
            return hidden

    model = FakeModel()
    result = capture_last_token(
        model,
        {"input_ids": torch.tensor([[1, 2]])},
        model.blocks,
        [0, 1],
    )
    assert result.shape == (1, 2, 3)
    assert np.allclose(result[0, 0], [3, 3, 3])
    assert np.allclose(result[0, 1], [4, 4, 4])


def _source_event(group: str, event_id: str) -> dict:
    return {
        "event_id": event_id,
        "task": "test_task",
        "phase": "test_phase",
        "seed": 3 if group == "large" else 4,
        "condition": "revisable",
        "edit_group": group,
        "reflection_index": 1,
        "block_mean_reward": 0.5,
        "input_source": "recorded",
        "messages": [{
            "role": "user",
            "content": (
                f"evidence for {group}\n\n=== TASK ===\n"
                "what should change?"
            ),
        }],
        "proposed_deltas": (
            [{"action": "add", "principle": "adapt to the evidence"}]
            if group == "large" else []
        ),
    }


def test_factorial_events_are_complete_and_content_matched() -> None:
    events = build_factorial_events([(
        _source_event("large", "large-1"),
        _source_event("none", "none-1"),
    )])
    validate_factorial_events(events)
    assert len(events) == 8
    assert len({event["candidate_json"] for event in events}) == 1
    assert {
        (
            event["need_to_edit"],
            event["authority_to_write"],
            event["instruction_pressure"],
        )
        for event in events
    } == {
        (need, authority, instruction)
        for need in (False, True)
        for authority in (False, True)
        for instruction in (False, True)
    }
    conflict = build_factorial_events([(
        _source_event("large", "large-1"),
        _source_event("none", "none-1"),
    )], instruction_mode="conflict")
    instructed = [
        event for event in conflict if event["instruction_pressure"]
    ]
    assert all(event["instruction_conflicts_with_need"] for event in instructed)
    assert {
        (event["need_to_edit"], event["instruction_target"])
        for event in instructed
    } == {(False, "ACCEPT"), (True, "REJECT")}
    weak = build_factorial_events([(
        _source_event("large", "large-1"),
        _source_event("none", "none-1"),
    )], instruction_mode="weak_conflict")
    weak_instructed = [
        event for event in weak if event["instruction_pressure"]
    ]
    assert all(
        event["instruction_conflicts_with_need"]
        for event in weak_instructed
    )
    assert all(
        "nonbinding" in event["messages"][0]["content"].lower()
        or "non-authoritative" in event["messages"][0]["content"].lower()
        for event in weak_instructed
    )


def test_factorial_axes_recover_independent_effects() -> None:
    events = []
    rows = []
    for pair in range(4):
        for need in (False, True):
            for authority in (False, True):
                for instruction in (False, True):
                    events.append({
                        "pair_id": f"pair-{pair}",
                        "need_to_edit": need,
                        "authority_to_write": authority,
                        "instruction_pressure": instruction,
                    })
                    rows.append([
                        4.0 * (1 if need else -1),
                        3.0 * (1 if authority else -1),
                        2.0 * (1 if instruction else -1),
                        float(pair),
                    ])
    raw = factorial_contrasts(
        np.asarray(rows), events, MAIN_TERMS
    )
    axes = symmetric_orthogonalize(raw)
    assert np.allclose(axes @ axes.T, np.eye(3), atol=1e-7)
    assert np.argmax(np.abs(axes), axis=1).tolist() == [0, 1, 2]


def test_gate_decision_and_write_mediation() -> None:
    assert parse_decision('{"decision":"ACCEPT"}') == "ACCEPT"
    assert parse_decision('{"action":"REJECT"}') == "REJECT"
    assert parse_decision('{"accept":false}') == "REJECT"
    assert parse_decision('{"accepted":true}') == "ACCEPT"
    assert parse_decision("I would REJECT this candidate.") == "REJECT"
    assert write_applied(True, "ACCEPT") is True
    assert write_applied(True, "REJECT") is False
    assert write_applied(False, "REJECT") is True


def main() -> None:
    test_edit_labels_are_output_only()
    test_autonomy_interaction_is_difference_in_differences()
    test_residual_intervention_adds_and_ablates()
    test_sae_and_layer_parser()
    test_activation_hooks_capture_last_input_token()
    test_factorial_events_are_complete_and_content_matched()
    test_factorial_axes_recover_independent_effects()
    test_gate_decision_and_write_mediation()
    print("Constitution interpretability pipeline: OK")


if __name__ == "__main__":
    main()
