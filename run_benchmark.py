import csv
import os
import vnnlib.compat
import parse_onnx
import deeppoly.base
import deeppoly.postcondition
import deeppoly.sequencial
import torch
import argparse

def run_exp(precond, model, postcond, device=torch.device("cuda")):
    """run an experiment of the model. 

    Args:
        precond (_type_): should be a list of tuples of lower and upper bounds
        model (_type_): 
        postcond (_type_): _description_
    """
    lb = [x[0] for x in precond]
    ub = [x[1] for x in precond]

    A = torch.tensor(postcond[0], dtype=torch.float32)
    b = - torch.tensor(postcond[1], dtype=torch.float32).reshape(-1)
    
    # print(A,b)
    input_info = deeppoly.base.create_input_info(lb=lb, ub=ub,device=device)
    
    pstcond = deeppoly.postcondition.ArgumentedPostcond(A=A, b=b, device=device)

    argumented_model = deeppoly.sequencial.create_sequencial_from_dirty(model, pstcond)

    input_info = argumented_model.forward(input_info)
    res = input_info.ub.cpu().detach().numpy()
    print(res)
    if max(res) <= 0:
        return True
    return False
    # print(input_info.lb, input_info.ub, postcond)
    # check if input_info is on GPU

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run benchmark')
    parser.add_argument('--instance_file', type=str, default='vnncomp2023_benchmarks-main/benchmarks/acasxu/instances.csv', help='Instance file')
    arg = parser.parse_args()
    # arg.instance_file remove the last item
    BENCHPATH = os.path.dirname(arg.instance_file)
    with open(arg.instance_file) as F:
        reader = csv.reader(F)
        for row in reader:
            model_path = os.path.join(BENCHPATH, row[0])
            prop_path = os.path.join(BENCHPATH, row[1])
            print("PATHS", model_path, prop_path)
            model = parse_onnx.load_onnx(model_path)
            model = model.eval()
            try:
                ast_vnn_node = vnnlib.parse_file(prop_path)
                property_list = vnnlib.compat.CompatTransformer("X", "Y").transform(ast_vnn_node)
            except:
                print("Error in parsing the property")
                continue
            
            for prop in property_list:
                precond, postcond = prop
                # I DONT KNOW WHAT'S GOING ON IN THE PARSER
                postcond = postcond[0]

                res = run_exp(precond, model, postcond)
                if res:
                    print("Property holds")
                else:
                    print("Unknown (may be unsafe)")
                    

    