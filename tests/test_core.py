from fractions import Fraction
from itertools import islice
import unittest

from tensor_enumerator import (
    Seed,
    dense,
    enumerate_steps,
    enumerate_tensor,
    find_step,
    glq,
    glz,
    inverse,
    matrix,
    pretty_matrix,
    sparse,
    transform,
)


class TensorEnumeratorTests(unittest.TestCase):
    def test_vector_uses_j(self):
        out = transform(sparse([1, 0], rank=1), (1, 0), matrix([[0, 1], [1, 0]]), 2)
        self.assertEqual(dense(out, 2, 1), [Fraction(0), Fraction(1)])

    def test_covector_uses_inverse_j(self):
        out = transform(sparse([2, 3], rank=1), (0, 1), matrix([[2, 0], [0, 3]]), 2)
        self.assertEqual(dense(out, 2, 1), [Fraction(1), Fraction(1)])

    def test_211_tensor_uses_two_upper_slots_and_one_lower_slot(self):
        old = sparse(
            [
                [[1, 0], [0, 0]],
                [[0, 0], [0, 0]],
            ],
            rank=3,
        )

        out = transform(old, (2, 1), matrix([[2, 0], [0, 3]]), 2)

        self.assertEqual(
            dense(out, 2, 3),
            [
                [[2, 0], [0, 0]],
                [[0, 0], [0, 0]],
            ],
        )

    def test_enumerator_keeps_step_and_frame(self):
        seed = Seed(sparse([1, 0], rank=1), (1, 0), 2)
        first, second = list(enumerate_steps(seed, glz(2), [0, 1]))

        self.assertEqual(first.step, 0)
        self.assertEqual(first.frame, "F_0")
        self.assertEqual(second.step, 1)
        self.assertEqual(second.frame, "F_1")

    def test_selected_step_matches_full_enumeration(self):
        seed = Seed(sparse([1, 0], rank=1), (1, 0), 2)
        selected = next(enumerate_steps(seed, glz(2), [3]))
        full = list(islice(enumerate_tensor(seed, glz(2)), 4))

        self.assertEqual(selected.components, full[3].components)

    def test_glq_reaches_rational_matrix(self):
        target = matrix([[1, 0], [0, Fraction(1, 2)]])
        self.assertGreaterEqual(find_step(glq(2), target), 0)

    def test_inverse(self):
        self.assertEqual(
            inverse(matrix([[1, 2], [3, 5]])),
            matrix([[-5, 2], [3, -1]]),
        )

    def test_pretty_matrix_is_multiline(self):
        self.assertEqual(
            pretty_matrix(matrix([[1, 0], [0, Fraction(1, 2)]])),
            "[ 1    0 ]\n[ 0  1/2 ]",
        )


if __name__ == "__main__":
    unittest.main()
