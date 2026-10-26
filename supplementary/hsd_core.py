"""Minimal Human Specification Density calculation.

HSD measures externally supplied normative information per parameter:

    HSD = (rubric_bits + constitution_bits + reward_bits) / parameter_count

This file contains only the calculation used by the survey; data collection,
source annotation, plotting, and model metadata are intentionally omitted.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class SpecificationRecord:
    model_id: str
    parameter_count: float
    rubric_tokens: int = 0
    constitution_tokens: int = 0
    reward_code_tokens: int = 0
    natural_language_bits_per_token: float = 2.0
    code_bits_per_token: float = 4.0


def human_specification_bits(record: SpecificationRecord) -> float:
    rubric_bits = record.rubric_tokens * record.natural_language_bits_per_token
    constitution_bits = record.constitution_tokens * record.natural_language_bits_per_token
    reward_bits = record.reward_code_tokens * record.code_bits_per_token
    return rubric_bits + constitution_bits + reward_bits


def hsd(record: SpecificationRecord) -> float:
    if record.parameter_count <= 0:
        raise ValueError("parameter_count must be positive")
    return human_specification_bits(record) / record.parameter_count


def log10_hsd(record: SpecificationRecord) -> float:
    value = hsd(record)
    if value <= 0:
        return float("-inf")
    return math.log10(value)


def summarize(records: list[SpecificationRecord]) -> list[dict]:
    return [
        {
            "model_id": r.model_id,
            "human_spec_bits": human_specification_bits(r),
            "parameter_count": r.parameter_count,
            "hsd": hsd(r),
            "log10_hsd": log10_hsd(r),
        }
        for r in records
    ]
