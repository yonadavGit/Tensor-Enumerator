"""Optional symbolic example: Cartesian coordinates expressed via polar ones.

Run after installing SymPy:

    pip install sympy
    python examples/cartesian_to_polar_jacobian.py
"""

from sympy import cos, sin, symbols

from tensor_enumerator.symbolic import jacobian_from_map

r, theta = symbols("r theta")
x = r * cos(theta)
y = r * sin(theta)

print(jacobian_from_map([x, y], [r, theta]))
