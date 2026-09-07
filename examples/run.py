import argparse

from tensor_enumerator import Seed, enumerate_steps, enumerate_tensor, glq, glz, pretty_seed, pretty_step, sparse, take


def main() -> None:
    parser = argparse.ArgumentParser(description="Print tensor-enumerator steps.")
    parser.add_argument("--limit", type=int, default=613, help="number of initial steps to print")
    parser.add_argument("--steps", help="comma-separated global steps to print, e.g. 0,12")
    parser.add_argument("--rational", action="store_true", help="use GL(n,Q) instead of GL(n,Z)")
    args = parser.parse_args()

    seed = Seed(
        components=sparse([1, 0], rank=1),
        tensor_type=(1, 0),
        dimension=2,
    )
    matrices = glq(seed.dimension) if args.rational else glz(seed.dimension)

    print(pretty_seed(seed))
    print()

    if args.steps:
        steps = [int(value) for value in args.steps.split(",")]
        for rep in enumerate_steps(seed, matrices, steps):
            print(pretty_step(rep, seed.dimension))
            print()
    else:
        for rep in take(enumerate_tensor(seed, matrices), args.limit):
            print(pretty_step(rep, seed.dimension))
            print()


if __name__ == "__main__":
    main()
