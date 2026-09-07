from __future__ import annotations

from fractions import Fraction

from .helpers import dense
from .model import BasisRepresentation, Matrix, Representation


def pretty_basis_representation(reference: BasisRepresentation) -> str:
    rank = sum(reference.tensor_type)
    return "\n".join(
        [
            "REFERENCE REPRESENTATION",
            f"{tensor_name(reference.symbol, reference.tensor_type)} in B0 =",
            pretty_array(dense(reference.components, reference.dimension, rank)),
            f"dimension: {reference.dimension}",
            f"type: {reference.tensor_type}  rule: {rule(reference.tensor_type)}",
        ]
    )



def pretty_step(rep: Representation, dimension: int) -> str:
    rank = sum(rep.tensor_type)
    return "\n".join(
        [
            f"STEP {rep.step}",
            f"{tensor_name(rep.symbol, rep.tensor_type)} in {rep.basis} =",
            pretty_array(dense(rep.components, dimension, rank)),
            f"basis: {rep.basis}",
            f"type: {rep.tensor_type}  rule: {rule(rep.tensor_type)}",
            "component map J =",
            pretty_matrix(rep.j),
            f"basis vectors of {rep.basis}, written in B0 = J^-1 =",
            pretty_matrix(rep.basis_vectors),
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


def tensor_name(symbol: str, tensor_type: tuple[int, int]) -> str:
    r, s = tensor_type
    upper = ",".join(upper_index_name(i) for i in range(r))
    lower = ",".join(lower_index_name(i) for i in range(s))

    if upper and lower:
        return f"{symbol}^{upper}_{lower}"
    if upper:
        return f"{symbol}^{upper}"
    if lower:
        return f"{symbol}_{lower}"
    return symbol


def upper_index_name(position: int) -> str:
    names = ["i", "j", "k", "l"]
    return names[position] if position < len(names) else f"i{position + 1}"


def lower_index_name(position: int) -> str:
    names = ["m", "n", "p", "q"]
    return names[position] if position < len(names) else f"m{position + 1}"


def fmt(value) -> str:
    value = Fraction(value)
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"
