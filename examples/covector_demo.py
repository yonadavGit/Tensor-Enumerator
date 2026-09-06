from itertools import islice

from tensor_enumerator import (
    TensorEnumerator,
    TensorSeed,
    as_sparse,
    describe_representation,
    describe_seed,
    enumerate_glnz,
)


seed = TensorSeed(
    components=as_sparse([2, 3], rank=1),
    tensor_type=(0, 1),
    dimension=2,
)

print(describe_seed(seed))
print()
print("Enumeration")
print("  This is a covector, so lower indices use J^-1.")
print()

for step, representation in enumerate(islice(TensorEnumerator(seed, enumerate_glnz(2)), 4)):
    print(describe_representation(representation, dimension=seed.dimension, step=step))
    print()
