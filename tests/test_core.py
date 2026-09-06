from fractions import Fraction
from itertools import islice
import unittest

from tensor_enumerator import (
    TensorEnumerator,
    TensorSeed,
    as_dense,
    as_sparse,
    enumerate_glq,
    enumerate_glnz,
    find_matrix_step,
    inverse,
    matrix,
    rational_height,
    representations_at_steps,
    transition_between,
    transform_components,
)


class TensorEnumeratorTests(unittest.TestCase):
    def test_vector_uses_j(self):
        j = matrix([[0, 1], [1, 0]])
        vector = as_sparse([1, 0], rank=1)

        transformed = transform_components(vector, (1, 0), j, 2)

        self.assertEqual(as_dense(transformed, 2, 1), [Fraction(0), Fraction(1)])

    def test_covector_uses_inverse_j(self):
        j = matrix([[2, 0], [0, 3]])
        covector = as_sparse([2, 3], rank=1)

        transformed = transform_components(covector, (0, 1), j, 2)

        self.assertEqual(as_dense(transformed, 2, 1), [Fraction(1), Fraction(1)])

    def test_rank_11_tensor_uses_both_rules(self):
        j = matrix([[2, 0], [0, 3]])
        tensor = as_sparse([[1, 0], [0, 1]], rank=2)

        transformed = transform_components(tensor, (1, 1), j, 2)

        self.assertEqual(
            as_dense(transformed, 2, 2),
            [[Fraction(1), Fraction(0)], [Fraction(0), Fraction(1)]],
        )

    def test_enumerator_outputs_framed_components(self):
        seed = TensorSeed(
            components=as_sparse([1, 0], rank=1),
            tensor_type=(1, 0),
            dimension=2,
        )

        outputs = list(islice(TensorEnumerator(seed, enumerate_glnz(2)), 2))

        self.assertEqual(outputs[0].frame.name, "F_0")
        self.assertEqual(as_dense(outputs[0].components, 2, 1), [1, 0])
        self.assertNotEqual(outputs[0].frame.from_reference, outputs[1].frame.from_reference)

    def test_inverse(self):
        j = matrix([[1, 2], [3, 5]])

        self.assertEqual(
            inverse(j),
            (
                (Fraction(-5), Fraction(2)),
                (Fraction(3), Fraction(-1)),
            ),
        )

    def test_transition_between_generated_frames_is_forced(self):
        seed = TensorSeed(
            components=as_sparse([1, 0], rank=1),
            tensor_type=(1, 0),
            dimension=2,
        )
        source, target = list(islice(TensorEnumerator(seed, enumerate_glnz(2)), 2))

        forced_transition = transition_between(source.frame, target.frame)
        mapped = transform_components(source.components, (1, 0), forced_transition, 2)

        self.assertEqual(mapped, target.components)

    def test_rational_height(self):
        self.assertEqual(rational_height(Fraction(3, 5)), 5)
        self.assertEqual(rational_height(Fraction(-4, 1)), 4)

    def test_glq_reaches_rational_matrices(self):
        target = matrix([[1, 0], [0, Fraction(1, 2)]])

        step, found = find_matrix_step(enumerate_glq(2, max_height=2), target)

        self.assertGreaterEqual(step, 0)
        self.assertEqual(found, target)

    def test_representations_at_steps_preserves_global_step_numbers(self):
        seed = TensorSeed(
            components=as_sparse([1, 0], rank=1),
            tensor_type=(1, 0),
            dimension=2,
        )
        selected = list(representations_at_steps(seed, enumerate_glnz(2), [0, 3]))
        full_prefix = list(islice(TensorEnumerator(seed, enumerate_glnz(2)), 4))

        self.assertEqual([item.step for item in selected], [0, 3])
        self.assertEqual(selected[0].components, full_prefix[0].components)
        self.assertEqual(selected[1].components, full_prefix[3].components)
        self.assertEqual(selected[1].frame.name, "F_3")


if __name__ == "__main__":
    unittest.main()
