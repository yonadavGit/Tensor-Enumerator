from __future__ import annotations

from fractions import Fraction
from itertools import count, product
from typing import Iterable, Iterator

from .helpers import det, inverse, matrix
from .model import BasisRepresentation, Matrix, Representation
from .transform import transform


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

    seen: set[Matrix] = set()
    for height in count(1):
        rationals = rationals_up_to_height(height)
        for flat in product(rationals, repeat=dimension * dimension):
            if all(rational_height(x) < height for x in flat):
                continue
            j = tuple(
                tuple(flat[row * dimension + col] for col in range(dimension))
                for row in range(dimension)
            )
            if j not in seen and det(j) != 0:
                seen.add(j)
                yield j


def glz(dimension: int) -> Iterator[Matrix]:
    """Enumerate the cleaner integer subset GL(n, Z)."""

    identity = matrix(
        [[1 if row == col else 0 for col in range(dimension)] for row in range(dimension)]
    )
    seen = {identity}
    yield identity

    for bound in count(1):
        for flat in product(range(-bound, bound + 1), repeat=dimension * dimension):
            if all(abs(x) != bound for x in flat):
                continue
            j = tuple(
                tuple(Fraction(flat[row * dimension + col]) for col in range(dimension))
                for row in range(dimension)
            )
            if j not in seen and det(j) != 0:
                seen.add(j)
                yield j


def find_step(matrices: Iterable[Matrix], target: Matrix) -> int:
    for step, j in enumerate(matrices):
        if j == target:
            return step
    raise ValueError("target was not found")


def rationals_up_to_height(height: int) -> tuple[Fraction, ...]:
    values = {Fraction(0)}
    for denominator in range(1, height + 1):
        for numerator in range(-height, height + 1):
            values.add(Fraction(numerator, denominator))
    return tuple(sorted(values, key=lambda x: (rational_height(x), x.denominator, x)))


def rational_height(value: Fraction) -> int:
    return max(abs(value.numerator), value.denominator)
