"""The core tensor transformation loop."""

from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
from itertools import product
from math import prod

from sympy import Matrix as SympyMatrix

Index = tuple[int, ...]
Matrix = tuple[tuple[Fraction, ...], ...]
Components = dict[Index, Fraction]


def transform(
    old_T: Components,
    tensor_type: tuple[int, int],
    J_fwd: Matrix,
    d: int,
) -> Components:
    """Transform tensor components from B0 to B_J.

    Example for type (2, 1):

        new_T[i,j,n] += old_T[k,l,m] * J_fwd[i,k] * J_fwd[j,l] * J_bwd[m,n]
    """

    r, s = tensor_type
    J_bwd = inverse(J_fwd)
    new_T = defaultdict(Fraction)

    upper_index_tuples = multi_indices(d, r)
    lower_index_tuples = multi_indices(d, s)

    for NEW_upper in upper_index_tuples:  # e.g. (i, j)
        for NEW_lower in lower_index_tuples:  # e.g. (n,)
            NEW = NEW_upper + NEW_lower

            for OLD_upper in upper_index_tuples:  # e.g. (k, l), summed
                for OLD_lower in lower_index_tuples:  # e.g. (m,), summed
                    OLD = OLD_upper + OLD_lower

                    # Add old_T[OLD] times all Jacobian terms that connect OLD to NEW.
                    new_T[NEW] += old_T.get(OLD, 0) * prod(
                        J_fwd[i][k] for i, k in zip(NEW_upper, OLD_upper)
                    ) * prod(
                        J_bwd[m][n] for n, m in zip(NEW_lower, OLD_lower)
                    )

    return {index: value for index, value in new_T.items() if value}


def inverse(m: Matrix) -> Matrix:
    return tuple(
        tuple(Fraction(value) for value in row)
        for row in SympyMatrix(m).inv().tolist()
    )


def multi_indices(d: int, rank: int) -> list[tuple[int, ...]]:
    return list(product(range(d), repeat=rank))
