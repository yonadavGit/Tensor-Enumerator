from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

Index = tuple[int, ...]
Matrix = tuple[tuple[Fraction, ...], ...]
Components = dict[Index, Fraction]


@dataclass(frozen=True)
class Seed:
    components: Components
    tensor_type: tuple[int, int]
    dimension: int


@dataclass(frozen=True)
class Representation:
    step: int
    frame: str
    j: Matrix
    components: Components
    tensor_type: tuple[int, int]
