import torch
import torch.nn as nn
import vnnlib.compat
import deeppoly
import deeppoly.base
import deeppoly.relu
import deeppoly.postcondition
import deeppoly.linear
import deeppoly.sequencial
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
    relu1 = deeppoly.relu.ArgumentedRelu()
    layers = [test1, relu1, test2]

    lb = [-1, -1]
    ub = [2, 1]
    input_info = deeppoly.base.create_input_info(lb, ub)
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

    seq = deeppoly.sequencial.ArgumentedSequential(*layers)
    x = seq.forward(input_info)
    print(f"Sequential: {x.lb} {x.ub}")


def run_exp(precond, model, postcond):
    """run an experiment of the model. 

    Args:
        precond (_type_): should be a list of tuples of lower and upper bounds
        model (_type_): 
        postcond (_type_): _description_
    """
    lb = [x[0] for x in precond]
    ub = [x[1] for x in precond]

    A = torch.tensor(postcond[0], dtype=torch.float32)
    b = torch.tensor(postcond[1], dtype=torch.float32).reshape(-1)
    
    print(A,b)
    input_info = deeppoly.base.create_input_info(lb=lb, ub=ub)
    
    pstcond = deeppoly.postcondition.ArgumentedPostcond(A=A, b=b)
    argumented_model = deeppoly.sequencial.create_sequencial_from_dirty(model, pstcond)
    input_info = argumented_model.forward(input_info)
    print(input_info.lb, input_info.ub)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load ONNX model and develop")
    parser.add_argument(
        "--model",
        type=str,
        help="Path to ONNX model",
        default="vnncomp2023_benchmarks-main/benchmarks/acasxu/onnx/ACASXU_run2a_1_1_batch_2000.onnx",
    )
    parser.add_argument(
        "--vnnlib",
        type=str,
        help="Path to VNNLIB file (Property)",
        default="vnncomp2023_benchmarks-main/benchmarks/acasxu/vnnlib/prop_1.vnnlib",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose information",
    )
    model_path = parser.parse_args().model
    model = parse_onnx.load_onnx(model_path)

    if parser.parse_args().verbose:
        torch.set_printoptions(threshold=1000)
        print(model)
        # sanity_check()
    
    ast_vnn_node = vnnlib.parse_file(parser.parse_args().vnnlib)
    property_list = vnnlib.compat.CompatTransformer("X", "Y").transform(ast_vnn_node)
    
    for prop in property_list:
        precond, postcond = prop
        
        # I DONT KNOW WHAT'S GOING ON IN THE PARSER
        postcond = postcond[0]
        
        print(precond, postcond)
        run_exp(precond, model, postcond)
        
    
