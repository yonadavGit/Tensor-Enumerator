# Tensor Enumerator

A small executable model of a tensor as the family of all representations
generated from one initial basis representation.

## Definition

Let `P` be this program.

Choose an ordered basis `B0 = (e1, ..., ed)`. Let `A0` be a component array in
that basis, and let `(r, s)` be the tensor type.

The tensor determined by `(B0, A0, (r, s), d)` is the set of all generated
basis representations obtained by running `P` indefinitely on that concrete
input:

```text
T = { (B_J, A_J) : P emits (B_J, A_J) }
```

For each emitted pair:

```text
J     = enumerated invertible rational component map
B_J   = basis whose vectors, written in B0, are the columns of J^-1
A_J   = components transformed from A0 by the tensor transformation law
```

The program never identifies bare component arrays. It emits component arrays
together with the basis in which they are written.

## Run

```bash
python -m tensor_enumerator
python -m tensor_enumerator --limit 20
python -m tensor_enumerator --steps 0,12
python -m tensor_enumerator --verbose --steps 0
python -m tensor_enumerator --symbol g --type 0,2 --components '[[1,2],[3,4]]' --limit 1000
```

The CLI prints the initial representation first. With no `--limit` or
`--steps`, generated enumeration then runs indefinitely. `--limit N` prints the
initial representation plus the first `N` generated representations.

Compact output prints each basis representation as a basis matrix and tensor
components. For generated representations, the basis matrix is `J^-1`:

```text
(
  basis =
    [ 1  0 ]
    [ 0  1 ]
  T^i =
    [ 1 ]
    [ 0 ]
)
```

Use `--verbose` to also print the basis label, component map `J`, tensor type,
and transformation rule.

## Shape

```text
tensor_enumerator/transform.py   core tensor transformation loop
tensor_enumerator/__init__.py    enumerator, formatting, CLI
tensor_enumerator/__main__.py    python -m tensor_enumerator
```

The program enumerates rational invertible matrices, `GL(n, Q)`. Determinants
and inverses are delegated to SymPy.

## Core Loop

The central code is `transform` in `tensor_enumerator/transform.py`.

For each new component, sum over all old components:

```text
upper index slot: multiply by J[new_i][old_i]
lower index slot: multiply by J^-1[old_i][new_i]
```

For type `(2, 1)`, this is:

```text
new_T[i,j,n] += old_T[k,l,m] * J[i,k] * J[j,l] * J^-1[m,n]
```

## API

```python
from tensor_enumerator import (
    BasisRepresentation,
    enumerate_tensor,
    glq,
    pretty_step,
    sparse,
    take,
)

reference = BasisRepresentation(
    components=sparse([1, 0], rank=1),
    tensor_type=(1, 0),
    dimension=2,
)

for rep in take(enumerate_tensor(reference, glq(2)), 5):
    print(pretty_step(rep, reference.dimension))
```

## Test

```bash
python -m unittest discover -s tests
```
