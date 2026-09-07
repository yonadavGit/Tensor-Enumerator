# Tensor Enumerator

A minimal executable model of a tensor as a generated family of basis-indexed
representations.

## Definition

Let `P` be the tensor-enumerator program.

Let `B0 = (e1, ..., ed)` be a chosen ordered basis, and let `A0` be a component
array written with respect to `B0`. The pair `(B0, A0)` is the initial basis
representation.

The tensor determined by this initial basis representation is defined by
running the actual program `P` indefinitely on the concrete input:

```text
(B0, A0, (r,s), d)
```

During that run, `P` emits basis-indexed representations. The tensor is the
complete set of representations that eventually appear in this output stream:

```text
T = {
  (B_J, A_J) : P eventually emits (B_J, A_J)
               when run on (B0, A0, (r,s), d)
}
```

In short:

```text
A tensor is the complete output stream, viewed as a set, produced by running P
indefinitely on an initial component representation in a chosen ordered basis.
```

```text
component representation in an initial ordered basis B0
  + tensor type (r, s)
  + enumerated component map J
  -> (B_J, transformed components)
```

`B0` is a chosen ordered basis. Each matrix `J` is the component map used by the
transformation law. The actual vectors of the generated basis `B_J`, written in
the original basis `B0`, are the columns of `J^-1`. The enumerator never
identifies bare component arrays; it prints basis-indexed representations.

## Run

```bash
python -m tensor_enumerator
python -m tensor_enumerator --limit 20
python -m tensor_enumerator --steps 0,12
python -m tensor_enumerator --rational --steps 613
python -m tensor_enumerator --symbol V --steps 0
python -m tensor_enumerator --symbol g --type 0,2 --components '[[1,2],[3,4]]' --limit 1000
```

The output starts with the core mathematical information, for example
`T^i in B_0 =`, then shows the generated basis, the component map `J`, the basis
vectors `J^-1`, and the tensor type. Matrices and component vectors are printed
as matrices.

## Shape

The project is split so the defining idea is easy to inspect:

```text
tensor_enumerator/transform.py   core tensor transformation loop
tensor_enumerator/enumerate.py   GL(n,Q) / GL(n,Z) enumerators
tensor_enumerator/pretty.py      readable step printing
tensor_enumerator/helpers.py     sparse/dense conversion + SymPy matrix ops
```

Determinants and inverses are delegated to SymPy. The matrix enumeration itself
stays explicit because it is part of the model being demonstrated.

## Core Loop

The transformer in `transform.py` is the rank-generic version of the handwritten
tensor formula:

```python
for new_index in all_indices:
    for old_index in all_indices:
        new_T[new_index] += old_T[old_index] * coefficient
```

The coefficient has one matrix factor per index slot:

```text
upper index: J[new_i][old_i]
lower index: J^-1[old_i][new_i]
```

So type `(2, 1)` behaves like the explicit loop:

```text
new_T[i,j,n] += old_T[k,l,m] * J[i,k] * J[j,l] * J^-1[m,n]
```

## API

```python
from tensor_enumerator import (
    BasisRepresentation,
    enumerate_tensor,
    glz,
    pretty_basis_representation,
    pretty_step,
    sparse,
    take,
)

reference = BasisRepresentation(
    components=sparse([1, 0], rank=1),
    tensor_type=(1, 0),
    dimension=2,
)

print(pretty_basis_representation(reference))

for rep in take(enumerate_tensor(reference, glz(2)), 5):
    print(pretty_step(rep, reference.dimension))
```

Use `glz(n)` for a readable integer-matrix enumeration. Use `glq(n)` for the
full rational enumeration of `GL(n, Q)`.

To run only selected global steps:

```python
from tensor_enumerator import enumerate_steps

for rep in enumerate_steps(reference, glz(2), [12]):
    print(pretty_step(rep, reference.dimension))
```

## Test

```bash
python -m unittest discover -s tests
```
