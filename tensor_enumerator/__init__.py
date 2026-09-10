"""Tensor Enumerator: tensors as generated basis-indexed representations."""

from __future__ import annotations

import argparse
import ast
import os
import sys
from dataclasses import dataclass
from fractions import Fraction
from itertools import count, islice, product
from typing import Iterable, Iterator, Sequence

from sympy import Matrix as SympyMatrix
from .transform import transform

Index = tuple[int, ...]
Matrix = tuple[tuple[Fraction, ...], ...]
Components = dict[Index, Fraction]


@dataclass(frozen=True)
class BasisRepresentation:
    components: Components
    tensor_type: tuple[int, int]
    dimension: int
    symbol: str = "T"


@dataclass(frozen=True)
class Representation:
    step: int
    basis: str
    symbol: str
    j: Matrix
    basis_vectors: Matrix
    components: Components
    tensor_type: tuple[int, int]


def enumerate_tensor(
    reference: BasisRepresentation,
    matrices: Iterable[Matrix],
) -> Iterator[Representation]:
    for step, j in enumerate(matrices):
        yield Representation(
            step=step,
            basis=f"B_{step}",
            symbol=reference.symbol,
            j=j,
            basis_vectors=inverse(j),
            components=transform(
                reference.components,
                reference.tensor_type,
                j,
                reference.dimension,
            ),
            tensor_type=reference.tensor_type,
        )


def enumerate_steps(
    reference: BasisRepresentation,
    matrices: Iterable[Matrix],
    steps: Iterable[int],
) -> Iterator[Representation]:
    wanted = set(steps)
    if not wanted:
        return
    if min(wanted) < 0:
        raise ValueError("steps must be non-negative")

    for rep in enumerate_tensor(reference, matrices):
        if rep.step in wanted:
            yield rep
            wanted.remove(rep.step)
            if not wanted:
                return


def glq(dimension: int) -> Iterator[Matrix]:
    """Enumerate GL(n, Q) by rational height."""

    identity = identity_matrix(dimension)
    seen: set[Matrix] = {identity}
    yield identity

    for height in count(1):
        rationals = rationals_up_to_height(height)
        for flat in product(rationals, repeat=dimension * dimension):
            if all(rational_height(x) < height for x in flat):
                continue
            j = square_matrix(flat, dimension)
            if j not in seen and det(j) != 0:
                seen.add(j)
                yield j


def find_step(matrices: Iterable[Matrix], target: Matrix) -> int:
    for step, j in enumerate(matrices):
        if j == target:
            return step
    raise ValueError("target was not found")


def pretty_basis_representation(reference: BasisRepresentation) -> str:
    return pretty_pair(
        identity_matrix(reference.dimension),
        reference.symbol,
        reference.tensor_type,
        reference.components,
        reference.dimension,
    )


def pretty_basis_representation_verbose(reference: BasisRepresentation) -> str:
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
    return pretty_pair(
        rep.basis_vectors,
        rep.symbol,
        rep.tensor_type,
        rep.components,
        dimension,
    )


def pretty_step_verbose(rep: Representation, dimension: int) -> str:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Print tensor-enumerator steps.")
    parser.add_argument("--limit", type=int, help="number of generated steps to print")
    parser.add_argument("--steps", help="comma-separated global steps to print, e.g. 0,12")
    parser.add_argument("--symbol", default="T", help="tensor symbol to print")
    parser.add_argument("--verbose", action="store_true", help="print J, J^-1, type, and basis details")
    parser.add_argument("--type", default="1,0", help="tensor type as r,s; metric is 0,2")
    parser.add_argument(
        "--components",
        default="[1,0]",
        help="component values as a Python-style nested list, e.g. '[[1,2],[3,4]]'",
    )
    parser.add_argument("--dimension", type=int, help="dimension; inferred from components by default")
    args = parser.parse_args()

    tensor_type = parse_tensor_type(args.type)
    component_values = ast.literal_eval(args.components)
    dimension = args.dimension or infer_dimension(component_values)
    reference = BasisRepresentation(
        components=sparse(component_values, rank=sum(tensor_type)),
        tensor_type=tensor_type,
        dimension=dimension,
        symbol=args.symbol,
    )
    matrices = glq(reference.dimension)

    print(
        pretty_basis_representation_verbose(reference)
        if args.verbose
        else pretty_basis_representation(reference)
    )
    print()

    if args.steps:
        steps = [int(value) for value in args.steps.split(",")]
        representations = enumerate_steps(reference, matrices, steps)
    elif args.limit is None:
        representations = enumerate_tensor(reference, matrices)
    else:
        representations = islice(enumerate_tensor(reference, matrices), args.limit)

    try:
        for rep in representations:
            print(
                pretty_step_verbose(rep, reference.dimension)
                if args.verbose
                else pretty_step(rep, reference.dimension)
            )
            print()
    except BrokenPipeError:
        sys.stdout = open(os.devnull, "w")


def matrix(values: Sequence[Sequence]) -> Matrix:
    return tuple(tuple(Fraction(value) for value in row) for row in values)


def inverse(m: Matrix) -> Matrix:
    return matrix(SympyMatrix(m).inv().tolist())


def det(m: Matrix) -> Fraction:
    return Fraction(SympyMatrix(m).det())


def sparse(values: Sequence, *, rank: int) -> Components:
    out: Components = {}

    def visit(prefix: Index, value):
        if len(prefix) == rank:
            rational = Fraction(value)
            if rational:
                out[prefix] = rational
            return
        for i, child in enumerate(value):
            visit(prefix + (i,), child)

    visit((), values)
    return out


def dense(components: Components, dimension: int, rank: int):
    if rank == 0:
        return components.get((), Fraction(0))

    def build(prefix: Index):
        if len(prefix) == rank:
            return components.get(prefix, Fraction(0))
        return [build(prefix + (i,)) for i in range(dimension)]

    return build(())


def take(iterable: Iterable, n: int) -> list:
    return list(islice(iterable, n))


def square_matrix(values: Sequence, dimension: int) -> Matrix:
    return tuple(
        tuple(Fraction(values[row * dimension + col]) for col in range(dimension))
        for row in range(dimension)
    )


def rationals_up_to_height(height: int) -> tuple[Fraction, ...]:
    values = {Fraction(0)}
    for denominator in range(1, height + 1):
        for numerator in range(-height, height + 1):
            values.add(Fraction(numerator, denominator))
    return tuple(sorted(values, key=lambda x: (rational_height(x), x.denominator, x)))


def rational_height(value: Fraction) -> int:
    return max(abs(value.numerator), value.denominator)


def parse_tensor_type(raw: str) -> tuple[int, int]:
    parts = raw.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("--type must have the form r,s")
    return int(parts[0]), int(parts[1])


def infer_dimension(values) -> int:
    if not isinstance(values, list):
        return 1
    if not values:
        raise argparse.ArgumentTypeError("--components cannot be an empty list")
    return len(values)


def pretty_matrix(m: Matrix) -> str:
    widths = [max(len(fmt(row[col])) for row in m) for col in range(len(m[0]))]
    rows = []
    for row in m:
        body = "  ".join(fmt(value).rjust(widths[col]) for col, value in enumerate(row))
        rows.append(f"[ {body} ]")
    return "\n".join(rows)


def pretty_pair(
    basis: Matrix,
    symbol: str,
    tensor_type: tuple[int, int],
    components: Components,
    dimension: int,
) -> str:
    rank = sum(tensor_type)
    return "\n".join(
        [
            "(",
            "  basis =",
            indent(pretty_matrix(basis), spaces=4),
            f"  {tensor_name(symbol, tensor_type)} =",
            indent(pretty_array(dense(components, dimension, rank)), spaces=4),
            ")",
        ]
    )


def pretty_array(value) -> str:
    if not isinstance(value, list):
        return fmt(value)
    if value and all(not isinstance(item, list) for item in value):
        width = max(len(fmt(item)) for item in value)
        return "\n".join(f"[ {fmt(item).rjust(width)} ]" for item in value)
    if value and all(isinstance(item, list) for item in value):
        return pretty_matrix(tuple(tuple(Fraction(x) for x in row) for row in value))
    return str(value)


def indent(text: str, *, spaces: int = 2) -> str:
    padding = " " * spaces
    return "\n".join(f"{padding}{line}" for line in text.splitlines())


def identity_matrix(dimension: int) -> Matrix:
    return tuple(
        tuple(Fraction(1 if row == col else 0) for col in range(dimension))
        for row in range(dimension)
    )


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
    upper = ",".join(index_name(["i", "j", "k", "l"], "i", n) for n in range(r))
    lower = ",".join(index_name(["m", "n", "p", "q"], "m", n) for n in range(s))
    if upper and lower:
        return f"{symbol}^{upper}_{lower}"
    if upper:
        return f"{symbol}^{upper}"
    if lower:
        return f"{symbol}_{lower}"
    return symbol


def index_name(names: list[str], prefix: str, position: int) -> str:
    return names[position] if position < len(names) else f"{prefix}{position + 1}"


def fmt(value) -> str:
    value = Fraction(value)
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"
