import torch
import torch.nn as nn
import deeppoly.base as base


class ArgumentedLinear(nn.Linear):
    def __init__(self, linear: nn.Linear):
        super().__init__(linear.in_features, linear.out_features)
        # assert is not training
        assert not linear.training
        self.weight = linear.weight
        self.bias = linear.bias
        self.immediate_info = base.ArgumentedInfo(
            linear.in_features, linear.out_features, linear.weight.device
        )
        # Copy only the CONTENT of the parameters
        # Regard the vars in the i_info as non-parameters (ie. not trainable)

        self.immediate_info.alb.copy_(linear.weight)
        self.immediate_info.aub.copy_(linear.weight)
        if linear.bias is not None:
            self.immediate_info.alb_bias.copy_(linear.bias)
            self.immediate_info.aub_bias.copy_(linear.bias)

    def forward(
        self,
        prev: base.ArgumentedInfo,
        *args,
    ) -> base.ArgumentedInfo:
        """
        The Forward Stage of DeepPoly. Accept the previous layer, compute the immediate algebraic expression and try to compute a bound.
        For the linear layer, the immediate algebraic expression is the weight matrix itself.
        """
        if isinstance(prev, torch.Tensor):
            print(
                "Warning. The input is a tensor. Excecuting the forward method as usual."
            )
            return super().forward(prev)
        # The alb/aub is the weight matrix, and was processed in the __init__
        # Only need to update the estimated bounds
        return self.immediate_info.forward(prev, compute_alg=False)

    def backsubstitution(self, prev: base.ArgumentedInfo) -> base.ArgumentedInfo:
        # The alb/aub is the weight matrix
        return prev.forward(self.immediate_info, compute_alg=True)

    def update_bounds(self, x: base.ArgumentedInfo) -> None:
        self.immediate_info.lb.copy_(x.lb)
        self.immediate_info.ub.copy_(x.ub)


ALL = [ArgumentedLinear]
