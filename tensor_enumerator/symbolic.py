"""Optional symbolic bridge for classical coordinate-change examples."""

from __future__ import annotations

from fractions import Fraction


def jacobian_from_map(new_coordinates, old_coordinates):
    """Return the symbolic Jacobian d(new_coordinates)/d(old_coordinates).

    This helper intentionally imports SymPy lazily, so the core project stays
    dependency-free. Install `sympy` when you want symbolic demos.
    """

    try:
        from sympy import Matrix
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "jacobian_from_map requires sympy. Install it with `pip install sympy`."
        ) from exc

    return Matrix(new_coordinates).jacobian(old_coordinates)


def exact_matrix_from_sympy(sympy_matrix, substitutions=None):
    """Evaluate a SymPy matrix and convert rational entries to Fractions."""

    substitutions = substitutions or {}
    evaluated = sympy_matrix.subs(substitutions)

    rows = []
    for i in range(evaluated.rows):
        row = []
        for j in range(evaluated.cols):
            value = evaluated[i, j].simplify()
            if not value.is_Rational:
                raise ValueError(f"entry {value} is not rational after substitution")
            row.append(Fraction(int(value.p), int(value.q)))
        rows.append(tuple(row))

    return tuple(rows)
