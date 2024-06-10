import torch
import torch.nn as nn
import deeppoly.base as base


class ArgumentedRelu(nn.ReLU):
    def __init__(self):
        super().__init__()
        self.immediate_info = base.ArgumentedInfo(1, 1, torch.device("cpu"))

    def forward(
        self,
        prev: base.ArgumentedInfo,
        *args,
    ) -> base.ArgumentedInfo:
        """
        The Forward Stage of DeepPoly. Accept the previous layer, compute the immediatealgebraic expression and try to compute a bound.
        For the ReLU layer, the immediate algebraic expression is an abstraction of the ReLU function.
        """
        if isinstance(prev, torch.Tensor):
            print(
                "Warning. The input is a tensor. Excecuting the forward method as usual."
            )
            return super().forward(prev)
        # TODO Implement the special CUDA/C++ operator for this. Reduce 114514x overhead.
        relulb = torch.max(prev.lb, torch.zeros_like(prev.lb))
        reluub = torch.max(prev.ub, torch.zeros_like(prev.ub))
        ubk = torch.div(
            reluub - relulb,
            torch.max(prev.ub - prev.lb, torch.ones_like(prev.ub)),
        )
        ubb = reluub - torch.mul(ubk, reluub)
        lbk = torch.lt(prev.lb + prev.ub, torch.zeros_like(prev.lb)).float()
        lbb = torch.zeros_like(prev.lb)
        self.immediate_info.lb = torch.mul(lbk, prev.lb) + lbb
        self.immediate_info.ub = torch.mul(ubk, prev.ub) + ubb
        self.immediate_info.alb = torch.diag(lbk)
        self.immediate_info.aub = torch.diag(ubk)
        self.immediate_info.alb_bias = torch.mul(lbk, prev.lb)
        self.immediate_info.aub_bias = ubb
        self.immediate_info.in_features = prev.out_features
        self.immediate_info.out_features = prev.out_features
        return self.immediate_info

    def backsubstitution(self, prev: base.ArgumentedInfo) -> base.ArgumentedInfo:
        # The alb/aub is the weight matrix
        return prev.forward(self.immediate_info, compute_alg=True)

    def update_bounds(self, x: base.ArgumentedInfo) -> None:
        self.immediate_info.lb.copy_(x.lb)
        self.immediate_info.ub.copy_(x.ub)
