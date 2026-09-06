# Tensor Enumerator

Understanding tensors by making their transformation law executable.

This is a small proof of concept for a computational definition of tensors:

```text
seed components in F0
        +
tensor type (r, s)
        +
generated frames F_J, each named by a matrix J from F0
        ↓
frame-indexed component representations of the same tensor
```

The reference frame `F0` is not mathematically privileged. It is a storage
convention. Every generated frame is addressed by its fixed transformation from
`F0`, so the enumerator yields framed representations, not raw arrays.

```text
Tensor(A0, type) = {
  (F_J, transform(A0, J, type)) : J in admissible transformations from F0
}
```

That distinction matters. For a vector, some matrix can send almost any nonzero
tuple to almost any other nonzero tuple. The project avoids that collapse by
never identifying bare tuples. It identifies framed tuples:

```text
(F0, [1, 0]) and (F_J, [0, 1])
```

may represent the same vector, while

```text
(F0, [1, 0]) and (F0, [0, 1])
```

do not.

## Core Convention

`J` is the component-change matrix from the reference frame `F0` to the
generated frame `F_J`.

- contravariant / upper indices use `J`
- covariant / lower indices use `J^-1`

So a type `(r, s)` tensor transforms by applying `J` to its first `r` indices and
`J^-1` to its final `s` indices.

## Enumerating Jacobians

The mathematically honest enumerator is `enumerate_glq(n)`.

It enumerates rational matrices by height:

```text
height(a / b) = max(abs(a), b)
```

then forms all `n x n` rational matrices from those entries and yields the ones
with nonzero determinant. In short:

```text
enumerate Q^(n*n)
keep J iff det(J) != 0
yield J
```

For cleaner short demos, `enumerate_glnz(n)` enumerates the integer subset
`GL(n, Z)`.

## Quick Example

```python
from itertools import islice

from tensor_enumerator import TensorEnumerator, TensorSeed, as_dense, as_sparse, enumerate_glnz

seed = TensorSeed(
    components=as_sparse([1, 0], rank=1),
    tensor_type=(1, 0),
    dimension=2,
)

for representation in islice(TensorEnumerator(seed, enumerate_glnz(2)), 3):
    print(representation.frame.name)
    print(representation.frame.from_reference)
    print(as_dense(representation.components, 2, rank=1))
```

## Run

```bash
python -m unittest discover -s tests
python examples/basic_enumerator.py
python examples/covector_demo.py
python examples/polar_exact_step.py
```

The basic example prints the seed, every generated frame address `J`, its
inverse, the index rule being applied, and the resulting framed representation.

`examples/polar_exact_step.py` computes the symbolic polar Jacobian
`d(x,y)/d(r,theta)`, evaluates it at a rational point, and searches
`enumerate_glq(2)` until the exact rational matrix appears.

Once a step is known, you can run only that step:

```python
from tensor_enumerator import TensorSeed, as_sparse, enumerate_glq, representations_at_steps

seed = TensorSeed(
    components=as_sparse([3, 4], rank=1),
    tensor_type=(1, 0),
    dimension=2,
)

for representation in representations_at_steps(seed, enumerate_glq(2), [613]):
    print(representation.frame.name)
    print(representation.frame.from_reference)
    print(representation.components)
```

The core has no dependencies. For symbolic coordinate-map demos:

```bash
pip install -e ".[symbolic]"
python examples/cartesian_to_polar_jacobian.py
python examples/polar_exact_step.py
```
