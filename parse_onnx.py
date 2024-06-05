import gzip
import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import deeppoly as dp

def load_onnx_gz(file_path):
    with gzip.open(file_path, 'rb') as f:
        model_data = f.read()
    model = onnx.load_model_from_string(model_data)
    return model

class ONNXToPyTorch(nn.Module):
    def __init__(self, onnx_model):
        super(ONNXToPyTorch, self).__init__()
        self.layers = nn.ModuleList()
        self.parse_onnx_model(onnx_model)

    def parse_onnx_model(self, onnx_model):
        for node in onnx_model.graph.node:
            layer = self.convert_onnx_node_to_torch(node)
            if layer:
                self.layers.append(layer)

    def convert_onnx_node_to_torch(self, node):
        if node.op_type == 'Linear':
                # attributes = {attr.name: attr for attr in node.attribute}
                weight_name = node.input[1]
                weight = self.find_initializer_by_name(weight_name, onnx_model)
                weight = torch.tensor(np.frombuffer(weight.raw_data, dtype=np.float32).reshape(weight.dims))
        
                if 'bias' in node.input:
                        bias_name = node.input[2]
                        bias = self.find_initializer_by_name(bias_name, onnx_model)
                        bias = torch.tensor(np.frombuffer(bias.raw_data, dtype=np.float32).reshape(bias.dims))
                        return dp.Linear(in_features=weight.shape[1], out_features=weight.shape[0], bias=True)
                else:
                        return dp.Linear(in_features=weight.shape[1], out_features=weight.shape[0], bias=False)
        # TODO add more layers
        return None

    def find_initializer_by_name(self, name, model):
        for initializer in model.graph.initializer:
            if initializer.name == name:
                return initializer
        return None

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            # TODO design 
        return x
