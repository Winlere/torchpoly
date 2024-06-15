import time
import csv
import os
import vnnlib.compat
import parse_onnx
import deeppoly.base
import deeppoly.postcondition
import deeppoly.sequencial
import torch
import argparse
import run_benchmark
import timeout_decorator

def run_exp(*args, **kwargs):
    return run_benchmark.run_exp(*args, **kwargs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Reproduce the results in the report')
    parser.add_argument('--instance_file', type=str, default='vnncomp2023_benchmarks-main/benchmarks/acasxu/instances.csv', help='Instance file')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the experiment')
    parser.add_argument("--skip_cpu", default=False, action="store_true", help="Skip the CPU verification")
    parser.add_argument("--latex_output_folder", type=str, default="./", help="Output folder for the latex formatted output")
    arg = parser.parse_args()
    torch.set_num_threads(1)
    # arg.instance_file remove the last item
    BENCHPATH = os.path.dirname(arg.instance_file)
    
    NVerified = 0
    NUnknown = 0
    TGPU = 0
    TCPU = 0
    
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
                BG = time.time()
                res = run_exp(precond, model, postcond, device=torch.device(arg.device))
                ED = time.time()
                TGPU = TGPU + (ED - BG)
                
                print("GPU", ED - BG)
                if res:
                    print("Property holds")
                    NVerified = NVerified + 1
                else:
                    print("Unknown (may be unsafe)")
                    NUnknown = NUnknown + 1

                
                if not arg.skip_cpu:
                    BG = time.time()
                    res = run_exp(precond, model, postcond, device=torch.device("cpu"))
                    ED = time.time()
                    TCPU = TCPU + (ED - BG)
                    print("CPU", ED - BG)
    
    print("GPU", TGPU, "CPU", TCPU)
    print("Verified", NVerified, "Unknown", NUnknown)
    print(f"{round(TGPU,2)} & {round(TCPU,2)} & {NVerified} & {NUnknown}")
    
    #replace all "/" in BENCHPATH with "_"
    RESNAME = BENCHPATH.replace("/", "_")
    with open(f"{arg.latex_output_folder}/result_{RESNAME}.txt", "w") as F:
        F.write(f"{round(TGPU,2)} & {round(TCPU,2)} & {NVerified} & {NUnknown}\n")
    
    
    #print latex formatted 
    # print(f"{round(TGPU,2)} & {round(TCPU,2)} & {NVerified} & {NUnknown} \\\\")
                

    