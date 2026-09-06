"""A small executable model of tensors as frame-indexed representations."""

from .core import (
    Frame,
    Representation,
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
    rationals_with_height_at_most,
    representation_at_matrix,
    representations_at_steps,
    transition_between,
    transform_components,
)
from .display import describe_representation, describe_seed

__all__ = [
    "Frame",
    "Representation",
    "TensorEnumerator",
    "TensorSeed",
    "as_dense",
    "as_sparse",
    "enumerate_glq",
    "enumerate_glnz",
    "find_matrix_step",
    "inverse",
    "matrix",
    "rational_height",
    "rationals_with_height_at_most",
    "representation_at_matrix",
    "representations_at_steps",
    "transition_between",
    "transform_components",
    "describe_representation",
    "describe_seed",
]
