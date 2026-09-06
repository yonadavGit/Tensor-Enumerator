import unittest

from sympy import cos, sin, symbols

from tensor_enumerator import enumerate_glq, find_matrix_step, matrix
from tensor_enumerator.symbolic import exact_matrix_from_sympy, jacobian_from_map


class SymbolicJacobianTests(unittest.TestCase):
    def test_pointwise_polar_jacobian_appears_in_glq_enumeration(self):
        r, theta = symbols("r theta")
        symbolic_j = jacobian_from_map(
            [r * cos(theta), r * sin(theta)],
            [r, theta],
        )

        target_j = exact_matrix_from_sympy(symbolic_j, {r: 2, theta: 0})
        step, found_j = find_matrix_step(enumerate_glq(2, max_height=2), target_j)

        self.assertEqual(target_j, matrix([[1, 0], [0, 2]]))
        self.assertEqual(found_j, target_j)
        self.assertEqual(step, 613)


if __name__ == "__main__":
    unittest.main()
