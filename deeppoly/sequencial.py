import torch
import torch.nn as nn
import deeppoly.base as base
import deeppoly.linear
import deeppoly.relu


class ArgumentedSequential(nn.Sequential):
    def __init__(self, *args):
        """ Argumented Sequential.            
        """
        super().__init__(*args)
        self.layers = list(args)
    
    def forward(self, x: base.ArgumentedInfo) -> base.ArgumentedInfo:
        if isinstance(x, torch.Tensor):
            print(
                "Warning. The input is a tensor. Excecuting the forward method as usual."
            )
            return super().forward(x)
        input_info = x.clone()
        # print(f"Input: {input_info.lb} {input_info.ub}")
        # forward part
        for idx, layer in enumerate(self.layers):
            x = layer.forward(x)
            backsub_info = x.clone()
            # backsubstitution part
            for ridx, rlayer in enumerate(self.layers[:idx][::-1]):
                backsub_info = rlayer.backsubstitution(backsub_info)
            # update the bounds of the current layer
            if idx != 0:
                backsub_info.forward(input_info)
                layer.update_bounds(backsub_info)
        return x
    
    def backsubstitution(self, x: base.ArgumentedInfo):
        for layer in reversed(self.layers):
            x = layer.backsubstitution(x)
        return x


def create_sequencial_from_dirty(model: nn.Module, post_cond_layer:nn.Module = None, device=torch.device("cuda")) -> ArgumentedSequential:
    layers = []
    if model.training:
        print("Warning: The model is in training mode. The model should be in evaluation mode.")
        raise ValueError("The model is in training mode.")
    
    model.to(device)

    def visit_modules(model: torch.nn.Module): 
        if len(list(model.children())) == 0:
            yield model
        else:
            for c in model.children():
                yield from visit_modules(c)
    for layer in visit_modules(model):
        if isinstance(layer, nn.Linear):
            layers.append(deeppoly.linear.ArgumentedLinear(layer))
        elif isinstance(layer, nn.ReLU):
            layers.append(deeppoly.relu.ArgumentedRelu(torch.device("cuda")))
        else:
            # print(f"Warning: Unknown layer (ignoring) {layer}")
            pass
    if post_cond_layer is not None:
        layers.append(post_cond_layer)
    # print(layers)
    return ArgumentedSequential(*layers)

ALL = [ArgumentedSequential, create_sequencial_from_dirty]