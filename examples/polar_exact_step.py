"""Find the exact rational-enumerator step for a pointwise polar Jacobian."""

from sympy import Matrix, cos, diff, sin, symbols

from tensor_enumerator import (
    TensorSeed,
    as_dense,
    as_sparse,
    describe_seed,
    describe_representation,
    enumerate_glq,
    find_matrix_step,
    representations_at_steps,
    transform_components,
)
from tensor_enumerator.display import format_matrix, format_value
from tensor_enumerator.symbolic import exact_matrix_from_sympy, jacobian_from_map


r, theta = symbols("r theta")

# Cartesian coordinates as functions of polar coordinates.
x = r * cos(theta)
y = r * sin(theta)

symbolic_j = jacobian_from_map([x, y], [r, theta])
manual_symbolic_j = Matrix(
    [
        [diff(x, r), diff(x, theta)],
        [diff(y, r), diff(y, theta)],
    ]
)

# At theta = 0 and r = 2, the Jacobian is rational:
#
#   [[1, 0],
#    [0, 2]]
#
# That makes it eligible to appear exactly in the rational GL(2, Q) enumerator.
point = {r: 2, theta: 0}
evaluated_symbolic_j = symbolic_j.subs(point)
target_j = exact_matrix_from_sympy(symbolic_j, point)
step, found_j = find_matrix_step(enumerate_glq(2, max_height=2), target_j)

seed = TensorSeed(
    components=as_sparse([3, 4], rank=1),
    tensor_type=(1, 0),
    dimension=2,
)
transformed = transform_components(seed.components, seed.tensor_type, found_j, seed.dimension)
selected_representation = next(
    representations_at_steps(seed, enumerate_glq(2), [step])
)

print("Classical coordinate map")
print("  x(r, theta) = r cos(theta)")
print("  y(r, theta) = r sin(theta)")
print()
print("Symbolic differentiation")
print("  J_ij = partial(new_coordinate_i) / partial(old_coordinate_j)")
print(f"  partial x / partial r     = {diff(x, r)}")
print(f"  partial x / partial theta = {diff(x, theta)}")
print(f"  partial y / partial r     = {diff(y, r)}")
print(f"  partial y / partial theta = {diff(y, theta)}")
print()
print("Manual symbolic Jacobian")
print(f"  {manual_symbolic_j}")
print()
print("Symbolic Jacobian d(x,y)/d(r,theta)")
print(f"  {symbolic_j}")
print()
print("Rational point")
print("  r = 2")
print("  theta = 0")
print()
print("Substitution into symbolic Jacobian")
print("  Matrix([[cos(theta), -r*sin(theta)], [sin(theta), r*cos(theta)]])")
print("    .subs({r: 2, theta: 0})")
print(f"  = {evaluated_symbolic_j}")
print()
print("Exact rational matrix used by the enumerator")
print(f"  target J = {format_matrix(target_j)}")
print()
print("Rational GL(2, Q) enumeration")
print(f"  exact match found at step {step}")
print(f"  enumerated J = {format_matrix(found_j)}")
print()
print(describe_seed(seed))
print()
print("Applying the matched Jacobian to the seed vector")
print(f"  components in F0: {format_value(as_dense(seed.components, seed.dimension, rank=1))}")
print(f"  components in F_J: {format_value(as_dense(transformed, seed.dimension, rank=1))}")
print()
print("Running the enumerator only at the relevant step")
print(f"  requested steps: [{step}]")
print()
print(describe_representation(selected_representation, dimension=seed.dimension))
