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
    components=as_sparse([1, 0], rank=1),
    tensor_type=(1, 0),
    dimension=2,
)

print(describe_seed(seed))
print()
print("Enumeration")
print("  Each J defines a new frame F_J relative to F0.")
print("  Outputs are framed components, not bare tuples.")
print()

for step, representation in enumerate(islice(TensorEnumerator(seed, enumerate_glnz(2)), 6)):
    print(describe_representation(representation, dimension=seed.dimension, step=step))
    print()
