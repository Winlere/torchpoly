import argparse
import csv
import os
import torch
import vnnlib
import vnnlib.compat as compat

from typing import List, Tuple

import torchpoly
import torchpoly.cert as cert
import torchpoly.nn as nn
import torchpoly.onnx as torchpoly_onnx


def verify(
    model: nn.Module, precond: List[Tuple], postcond: List[Tuple], device=None
) -> torchpoly.VerifyResult:
    lb, ub = [x[0] for x in precond], [x[1] for x in precond]

    factory_kwargs = {"device": device, "dtype": torch.float32}

    A = torch.tensor(postcond[0], **factory_kwargs)
    b = torch.tensor(postcond[1], **factory_kwargs).reshape(-1)

    model.to(device=device)
    bound = cert.Ticket.from_bound(lb, ub, device=device)

    verifier = torchpoly.Verifier(A, b, device=device)
    return verifier(model, bound)


def run_vnncomp(model, prop, device):
    print(f"Verifying {model} {prop} ...")

    try:
        onnx_model = torchpoly_onnx.load(model)
        model = torchpoly_onnx.as_sequential(onnx_model).to(device=device)

        vnn_node = vnnlib.parse_file(prop)
        properties = compat.CompatTransformer("X", "Y").transform(vnn_node)
    except Exception:
        print(f"Failed to load {model} {prop}")
        return 0, 0, 1

    holds, unknowns, fails = 0, 0, 0
    for index, (precond, postcond) in enumerate(properties):
        try:
            result = verify(model, precond, postcond[0], device)
        except Exception:
            fails += 1
            continue

        print(f"Property {index}:", result.value)
        match result:
            case torchpoly.VerifyResult.HOLD:
                holds += 1
            case torchpoly.VerifyResult.UNKNOWN:
                unknowns += 1

    return holds, unknowns, fails


def main(args: argparse.Namespace):
    home = os.path.dirname(args.instance)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    holds, unknowns, fails = 0, 0, 0

    with open(args.instance, "r") as f:
        reader = csv.reader(f)

        for row in reader:
            model, prop = row[0], row[1]

            new_holds, new_unknowns, new_fails = run_vnncomp(
                os.path.join(home, model), os.path.join(home, prop), device
            )
            holds += new_holds
            unknowns += new_unknowns
            fails += new_fails

    print(f"Holds: {holds}, Unknowns: {unknowns}, Fails: {fails}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VNNCOMP")
    parser.add_argument("instance", type=str, help="Instance file")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
