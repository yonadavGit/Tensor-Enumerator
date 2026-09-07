from __future__ import annotations

from fractions import Fraction

from .helpers import dense, inverse
from .model import Matrix, Representation, Seed


def pretty_seed(seed: Seed) -> str:
    rank = sum(seed.tensor_type)
    return "\n".join(
        [
            "SEED",
            "frame: F0",
            f"dimension: {seed.dimension}",
            f"type: {seed.tensor_type}  rule: {rule(seed.tensor_type)}",
            "components in F0 =",
            pretty_array(dense(seed.components, seed.dimension, rank)),
        ]
    )


def pretty_step(rep: Representation, dimension: int) -> str:
    rank = sum(rep.tensor_type)
    return "\n".join(
        [
            f"STEP {rep.step}",
            f"frame: {rep.frame}  (F0 --J--> {rep.frame})",
            f"type: {rep.tensor_type}  rule: {rule(rep.tensor_type)}",
            "J =",
            pretty_matrix(rep.j),
            "J^-1 =",
            pretty_matrix(inverse(rep.j)),
            f"components in {rep.frame} =",
            pretty_array(dense(rep.components, dimension, rank)),
        ]
    )


def pretty_matrix(m: Matrix) -> str:
    widths = [max(len(fmt(row[col])) for row in m) for col in range(len(m[0]))]
    rows = []
    for row in m:
        body = "  ".join(fmt(value).rjust(widths[col]) for col, value in enumerate(row))
        rows.append(f"[ {body} ]")
    return "\n".join(rows)


def pretty_array(value) -> str:
    if not isinstance(value, list):
        return fmt(value)
    if value and all(not isinstance(item, list) for item in value):
        width = max(len(fmt(item)) for item in value)
        return "\n".join(f"[ {fmt(item).rjust(width)} ]" for item in value)
    if value and all(isinstance(item, list) for item in value):
        return pretty_matrix(tuple(tuple(Fraction(x) for x in row) for row in value))
    return str(value)


def rule(tensor_type: tuple[int, int]) -> str:
    r, s = tensor_type
    parts = []
    if r:
        parts.append(f"{r} upper -> J")
    if s:
        parts.append(f"{s} lower -> J^-1")
    return ", ".join(parts) if parts else "scalar"


def fmt(value) -> str:
    value = Fraction(value)
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"
