import torch 
import torch.nn as nn
import torch.nn.functional as F

class Linear(nn.Module):
        def __init__(self, in_features, out_features, bias=True):
            super(Linear, self).__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.weight = torch.randn(out_features, in_features)
            if bias:
                self.bias = torch.randn(out_features)
            else:
                self.bias = None
            # register buffer can support to_device nicely 
            self.register_buffer('lb', 1000 * torch.ones(out_features))
            self.register_buffer('ub', -1000 * torch.ones(out_features))
            self.register_buffer('alb', 1000 * torch.ones(out_features, in_features))
            self.register_buffer('aub', -1000 * torch.ones(out_features, in_features))

        def forward(self, x):
            if self.bias is not None:
                return F.linear(x, self.weight, self.bias)
            else:
                return F.linear(x, self.weight)
            
        # hook
        def interval_propagate(self, lA, uA):
            # TODO implement interval propagation
            pass
        
        # backward
        def bound_backward(self, last_lA, last_uA, A, lA, uA, index):
            # TODO implement backward bound
            pass