import torch
import torch.nn as nn
import vnnlib.compat
import deeppoly
import deeppoly.base
import deeppoly.linear
import parse_onnx
import vnnlib
import argparse


def sanity_check():
    W = torch.tensor([1, 1, 1, -1], dtype=torch.float32).reshape(2, 2)
    b = torch.tensor([0, 0], dtype=torch.float32)

    pattern = nn.Linear(2, 2)
    pattern.train(False)
    pattern.weight = nn.Parameter(W)
    pattern.bias = nn.Parameter(b)
    test1 = deeppoly.linear.ArgumentedLinear(pattern)
    test2 = deeppoly.linear.ArgumentedLinear(pattern)
    layers = [test1, test2]

    lb = [-1, -1]
    ub = [1, 1]
    input_info = deeppoly.base.ArgumentedInfo(
        in_features=1, out_features=2, device=pattern.weight.device
    )
    input_info.lb = torch.tensor(lb, dtype=torch.float32)
    input_info.ub = torch.tensor(ub, dtype=torch.float32)
    input_info.alb_bias = torch.tensor(lb, dtype=torch.float32)
    input_info.aub_bias = torch.tensor(ub, dtype=torch.float32)
    x = input_info.clone()

    for idx, layer in enumerate(layers):
        x = layer.forward(x)
        print(f"(Forward) Layer {idx}: {x.lb} {x.ub}")
        back_info = x.clone()

        # for ridx, rlayer in enumerate(layers[:idx:-1]):
        # enumerate from idx-1 to 0
        for ridx, rlayer in enumerate(layers[:idx][::-1]):
            back_info = rlayer.backsubstitution(back_info)
            print(
                f"    (Backsubstitute) Layer {idx - ridx - 1}: {back_info.lb} {back_info.ub}"
            )
        if idx != 0:
            back_info.forward(input_info)
            layer.update_bounds(x)
            print(f"    (Update) Layer {idx}: {back_info.lb} {back_info.ub}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load ONNX model and develop")
    parser.add_argument(
        "--model",
        type=str,
        help="Path to ONNX model",
        default="vnncomp2023_benchmarks-main/benchmarks/acasxu/onnx/ACASXU_run2a_1_1_batch_2000.onnx.gz",
    )
    parser.add_argument(
        "--vnnlib",
        type=str,
        help="Path to VNNLIB file (Property)",
        default="vnncomp2023_benchmarks-main/benchmarks/acasxu/vnnlib/prop_1.vnnlib.gz",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose information",
    )
    model_path = parser.parse_args().model
    model = parse_onnx.load_onnx(model_path)

    ast_node = vnnlib.parse_file(parser.parse_args().vnnlib)
    ast_res = vnnlib.compat.CompatTransformer("X", "Y").transform(ast_node)

    if parser.parse_args().verbose:
        torch.set_printoptions(threshold=1000)
        print(model)
        print(*ast_res)
        sanity_check()
