"""Readable output helpers for demonstrations."""

from __future__ import annotations

from fractions import Fraction
from typing import Any

from .core import Representation, TensorSeed, as_dense, inverse


def describe_seed(seed: TensorSeed) -> str:
    """Return a compact description of the tensor seed in F0."""

    rank = sum(seed.tensor_type)
    return "\n".join(
        [
            "Seed representation",
            "  frame: F0",
            f"  dimension: {seed.dimension}",
            f"  tensor type: {format_tensor_type(seed.tensor_type)}",
            f"  components in F0: {format_value(as_dense(seed.components, seed.dimension, rank))}",
            f"  rule: {transformation_rule(seed.tensor_type)}",
        ]
    )


def describe_representation(
    representation: Representation,
    *,
    dimension: int,
    step: int | None = None,
) -> str:
    """Return the relevant data for one enumerated representation."""

    rank = sum(representation.tensor_type)
    j = representation.frame.from_reference
    lines = []

    title = f"Step {step}" if step is not None else representation.frame.name
    lines.append(title)
    if representation.step is not None and representation.step != step:
        lines.append(f"  original enumeration step: {representation.step}")
    lines.append(f"  generated frame: {representation.frame.name}")
    lines.append("  frame address: F0 --J--> " + representation.frame.name)
    lines.append(f"  J: {format_matrix(j)}")
    lines.append(f"  J^-1: {format_matrix(inverse(j))}")
    lines.append(f"  tensor type: {format_tensor_type(representation.tensor_type)}")
    lines.append(f"  rule used: {transformation_rule(representation.tensor_type)}")
    lines.append(
        "  transformed components: "
        + format_value(as_dense(representation.components, dimension, rank))
    )
    lines.append(
        "  representation: "
        + f"({representation.frame.name}, {format_value(as_dense(representation.components, dimension, rank))})"
    )

    return "\n".join(lines)


def format_tensor_type(tensor_type: tuple[int, int]) -> str:
    r, s = tensor_type
    return f"({r}, {s})  [{r} upper, {s} lower]"


def transformation_rule(tensor_type: tuple[int, int]) -> str:
    r, s = tensor_type
    pieces = []
    if r:
        pieces.append(f"{r} upper index/indices use J")
    if s:
        pieces.append(f"{s} lower index/indices use J^-1")
    return "; ".join(pieces) if pieces else "scalar: no component change"


def format_matrix(matrix) -> str:
    rows = ["[" + ", ".join(format_scalar(value) for value in row) + "]" for row in matrix]
    return "[" + ", ".join(rows) + "]"


def format_value(value: Any) -> str:
    if isinstance(value, list):
        return "[" + ", ".join(format_value(item) for item in value) + "]"
    return format_scalar(value)


def format_scalar(value: Any) -> str:
    if isinstance(value, Fraction):
        if value.denominator == 1:
            return str(value.numerator)
        return f"{value.numerator}/{value.denominator}"
    return str(value)
