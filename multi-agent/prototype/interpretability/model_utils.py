"""Shared Hugging Face model and residual-stream hook utilities."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Sequence

import torch


def parse_layers(spec: str, n_layers: int) -> list[int]:
    if spec == "all":
        return list(range(n_layers))
    if spec == "quartiles":
        return sorted({
            max(0, n_layers // 4 - 1),
            max(0, n_layers // 2 - 1),
            max(0, (3 * n_layers) // 4 - 1),
            n_layers - 1,
        })
    layers = sorted({int(value) for value in spec.split(",")})
    if not layers or layers[0] < 0 or layers[-1] >= n_layers:
        raise ValueError(
            f"layers must be within 0..{n_layers - 1}, got {layers}"
        )
    return layers


def transformer_layers(model: Any) -> Sequence[torch.nn.Module]:
    candidates = (
        ("model", "layers"),
        ("model", "model", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
    )
    for path in candidates:
        current = model
        try:
            for name in path:
                current = getattr(current, name)
        except AttributeError:
            continue
        if isinstance(current, (torch.nn.ModuleList, list, tuple)):
            return current
    raise ValueError(
        "could not locate transformer blocks; expected model.layers, "
        "model.model.layers, transformer.h, or gpt_neox.layers"
    )


def input_device(model: Any) -> torch.device:
    return model.get_input_embeddings().weight.device


def render_messages(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return "\n\n".join(
        f"{message['role'].upper()}:\n{message['content']}"
        for message in messages
    ) + "\n\nASSISTANT:\n"


def load_model_and_tokenizer(
    model_path: str,
    *,
    dtype: str,
    device_map: str,
) -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype not in dtype_map:
        raise ValueError(f"unsupported dtype: {dtype}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype_map[dtype],
        device_map=device_map,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model, tokenizer


@contextmanager
def residual_intervention(
    layer: torch.nn.Module,
    direction: torch.Tensor,
    *,
    coefficient: float,
    mode: str,
) -> Iterator[None]:
    """Add or remove a normalized direction at the last sequence position."""
    unit = direction / direction.norm().clamp_min(1e-8)

    def hook(
        _module: torch.nn.Module,
        _inputs: tuple[Any, ...],
        output: Any,
    ) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        vector = unit.to(device=hidden.device, dtype=hidden.dtype)
        modified = hidden.clone()
        last = modified[:, -1, :]
        if mode == "add":
            last = last + coefficient * vector
        elif mode == "ablate":
            projection = (last * vector).sum(dim=-1, keepdim=True)
            last = last - coefficient * projection * vector
        else:
            raise ValueError(f"unknown intervention mode: {mode}")
        modified[:, -1, :] = last
        if isinstance(output, tuple):
            return (modified, *output[1:])
        return modified

    handle = layer.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()
