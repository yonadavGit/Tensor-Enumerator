from __future__ import annotations

import argparse
import ast

from .enumerate import enumerate_steps, enumerate_tensor, glq, glz
from .helpers import sparse, take
from .model import BasisRepresentation
from .pretty import pretty_basis_representation, pretty_step


def main() -> None:
    parser = argparse.ArgumentParser(description="Print tensor-enumerator steps.")
    parser.add_argument("--limit", type=int, default=5, help="number of initial steps to print")
    parser.add_argument("--steps", help="comma-separated global steps to print, e.g. 0,12")
    parser.add_argument("--rational", action="store_true", help="use GL(n,Q) instead of GL(n,Z)")
    parser.add_argument("--symbol", default="T", help="tensor symbol to print")
    parser.add_argument("--type", default="1,0", help="tensor type as r,s; metric is 0,2")
    parser.add_argument(
        "--components",
        default="[1,0]",
        help="component values as a Python-style nested list, e.g. '[[1,2],[3,4]]'",
    )
    parser.add_argument("--dimension", type=int, help="dimension; inferred from components by default")
    args = parser.parse_args()

    tensor_type = parse_tensor_type(args.type)
    component_values = ast.literal_eval(args.components)
    dimension = args.dimension or infer_dimension(component_values)

    reference = BasisRepresentation(
        components=sparse(component_values, rank=sum(tensor_type)),
        tensor_type=tensor_type,
        dimension=dimension,
        symbol=args.symbol,
    )
    matrices = glq(reference.dimension) if args.rational else glz(reference.dimension)

    print(pretty_basis_representation(reference))
    print()

    if args.steps:
        steps = [int(value) for value in args.steps.split(",")]
        representations = enumerate_steps(reference, matrices, steps)
    else:
        representations = take(enumerate_tensor(reference, matrices), args.limit)

    for rep in representations:
        print(pretty_step(rep, reference.dimension))
        print()


def parse_tensor_type(raw: str) -> tuple[int, int]:
    parts = raw.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("--type must have the form r,s")
    return int(parts[0]), int(parts[1])


def infer_dimension(values) -> int:
    if not isinstance(values, list):
        return 1
    if not values:
        raise argparse.ArgumentTypeError("--components cannot be an empty list")
    return len(values)
