from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

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
