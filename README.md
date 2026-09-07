# Tensor Enumerator

A minimal executable model of a tensor as a generated family of framed
representations.

```text
seed components in F0
  + tensor type (r, s)
  + enumerated frame address J
  -> (F_J, transformed components)
```

`F0` is just the reference frame used by the program. Each matrix `J` names a
new frame `F_J` relative to `F0`, so the enumerator never identifies bare
component arrays. It prints framed representations.

## Run

```bash
python examples/run.py
python examples/run.py --limit 20
python examples/run.py --steps 0,12
python examples/run.py --rational --steps 613
```

The output shows each step, the generated frame, `J`, `J^-1`, the tensor type,
and the transformed components with matrices printed as matrices.

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
from tensor_enumerator import Seed, enumerate_tensor, glz, pretty_seed, pretty_step, sparse, take

seed = Seed(
    components=sparse([1, 0], rank=1),
    tensor_type=(1, 0),
    dimension=2,
)

print(pretty_seed(seed))

for rep in take(enumerate_tensor(seed, glz(2)), 5):
    print(pretty_step(rep, seed.dimension))
```

Use `glz(n)` for a readable integer-matrix enumeration. Use `glq(n)` for the
full rational enumeration of `GL(n, Q)`.

To run only selected global steps:

```python
from tensor_enumerator import enumerate_steps

for rep in enumerate_steps(seed, glz(2), [12]):
    print(pretty_step(rep, seed.dimension))
```

## Test

```bash
python -m unittest discover -s tests
```
