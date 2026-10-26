"""Train a sparse autoencoder on one residual-stream layer.

The report ranks features by edit contrast and by the self-versus-forced
difference-in-differences.  Decoder directions can then be tested with the
steering CLI; association alone is not treated as causal evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch

from prototype.interpretability.fit_subspace import (
    load_bundle,
)


class SparseAutoencoder(torch.nn.Module):
    def __init__(self, input_dim: int, n_features: int) -> None:
        super().__init__()
        self.encoder = torch.nn.Linear(input_dim, n_features)
        self.decoder = torch.nn.Linear(n_features, input_dim, bias=False)
        torch.nn.init.kaiming_uniform_(self.encoder.weight)
        with torch.no_grad():
            self.decoder.weight.copy_(
                self.encoder.weight.T
                / self.encoder.weight.norm(dim=1).clamp_min(1e-8)
            )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = torch.relu(self.encoder(x))
        reconstruction = self.decoder(features)
        return reconstruction, features

    def normalize_decoder(self) -> None:
        with torch.no_grad():
            self.decoder.weight.div_(
                self.decoder.weight.norm(dim=0, keepdim=True).clamp_min(1e-8)
            )


def _feature_contrasts(
    features: np.ndarray,
    events: list[dict[str, Any]],
) -> dict[str, np.ndarray]:
    condition = np.asarray([event["condition"] for event in events])
    large = np.asarray([
        bool(event["proposed_large_edit"]) for event in events
    ])
    no_edit = np.asarray([
        event["edit_group"] == "none" for event in events
    ])

    def mean(mask: np.ndarray) -> np.ndarray:
        return (
            features[mask].mean(axis=0)
            if mask.any() else np.full(features.shape[1], np.nan)
        )

    self_edit = (
        mean((condition == "revisable") & large)
        - mean((condition == "revisable") & no_edit)
    )
    forced_edit = (
        mean((condition == "forced_injection") & large)
        - mean((condition == "forced_injection") & no_edit)
    )
    return {
        "self_edit": self_edit,
        "forced_edit": forced_edit,
        "autonomy_interaction": self_edit - forced_edit,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activations", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--features", type=int, default=16384)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--l1", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    metadata_path = (
        args.metadata
        or args.activations.with_suffix(".metadata.jsonl")
    )
    activations, layers, events = load_bundle(
        args.activations, metadata_path
    )
    matches = np.flatnonzero(layers == args.layer)
    if len(matches) != 1:
        raise ValueError(
            f"layer {args.layer} not present exactly once in {layers.tolist()}"
        )
    x_np = activations[:, int(matches[0]), :].astype(np.float32)
    train_mask = np.asarray([
        int(event["seed"]) % 5 != 0 for event in events
    ])
    test_mask = ~train_mask
    mean = x_np[train_mask].mean(axis=0, keepdims=True)
    scale = float(np.sqrt(np.mean((x_np[train_mask] - mean) ** 2)))
    x_np = (x_np - mean) / max(scale, 1e-8)
    device = torch.device(args.device)
    model = SparseAutoencoder(x_np.shape[1], args.features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    train_indices = np.flatnonzero(train_mask)

    history = []
    for epoch in range(args.epochs):
        np.random.shuffle(train_indices)
        epoch_reconstruction = 0.0
        epoch_sparsity = 0.0
        seen = 0
        model.train()
        for start in range(0, len(train_indices), args.batch_size):
            indices = train_indices[start:start + args.batch_size]
            batch = torch.from_numpy(x_np[indices]).to(device)
            reconstruction, features = model(batch)
            reconstruction_loss = torch.mean(
                (reconstruction - batch) ** 2
            )
            sparsity_loss = features.abs().mean()
            loss = reconstruction_loss + args.l1 * sparsity_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            model.normalize_decoder()
            epoch_reconstruction += float(reconstruction_loss) * len(indices)
            epoch_sparsity += float(sparsity_loss) * len(indices)
            seen += len(indices)
        record = {
            "epoch": epoch + 1,
            "reconstruction_mse": epoch_reconstruction / max(seen, 1),
            "feature_l1": epoch_sparsity / max(seen, 1),
        }
        history.append(record)
        print(json.dumps(record))

    model.eval()
    feature_batches = []
    test_mse = []
    with torch.inference_mode():
        for start in range(0, len(x_np), args.batch_size):
            batch = torch.from_numpy(
                x_np[start:start + args.batch_size]
            ).to(device)
            reconstruction, features = model(batch)
            feature_batches.append(features.cpu().numpy())
            local_indices = np.arange(
                start, min(start + args.batch_size, len(x_np))
            )
            local_test = test_mask[local_indices]
            if local_test.any():
                errors = (
                    (reconstruction - batch) ** 2
                ).mean(dim=1).cpu().numpy()
                test_mse.extend(errors[local_test].tolist())
    features_np = np.concatenate(feature_batches, axis=0)
    contrasts = _feature_contrasts(features_np, events)

    def top(values: np.ndarray, n: int = 25) -> list[dict[str, float | int]]:
        finite = np.nan_to_num(values, nan=0.0)
        indices = np.argsort(np.abs(finite))[-n:][::-1]
        return [
            {"feature": int(index), "contrast": float(finite[index])}
            for index in indices
        ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.cpu().state_dict(),
        "input_dim": x_np.shape[1],
        "n_features": args.features,
        "layer": args.layer,
        "activation_mean": torch.from_numpy(mean[0]),
        "activation_scale": scale,
    }, args.output)
    report = {
        "layer": args.layer,
        "n": len(events),
        "train_n": int(train_mask.sum()),
        "test_n": int(test_mask.sum()),
        "test_reconstruction_mse": (
            float(np.mean(test_mse)) if test_mse else None
        ),
        "history": history,
        "top_features": {
            name: top(values) for name, values in contrasts.items()
        },
        "warning": (
            "Feature association is not a causal autonomy result. Decoder "
            "directions require matched-content steering and specificity "
            "controls."
        ),
    }
    report_path = args.output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2))
    print(f"Wrote {args.output} and {report_path}")


if __name__ == "__main__":
    main()
