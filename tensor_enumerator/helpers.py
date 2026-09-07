from __future__ import annotations

from fractions import Fraction
from itertools import islice
from typing import Iterable, Sequence

from sympy import Matrix as SympyMatrix

from .model import Components, Index, Matrix


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
