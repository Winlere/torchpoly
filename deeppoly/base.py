import torch
import torch.nn as nn


class ArgumentedInfo(nn.Module):
    # Each line in the alb(aub) is an equation of the computed lower(upper) bound of the each output.
    # Suppose the input tensor is x and the output is y
    # Then algebraicly, min(y[i]) = lb[i] = alb[i][j] * x[j] + alb_bias[i] for some interpretation of x[j] (x[j]=lb_x[j] if alb[i][j] >= 0)
    # and               max(y[i]) = ub[i] = aub[i][j] * x[j] + ulb_bias[i] for some interpretation of x[j] (x[j]=ub_x[j] if aub[i][j] >= 0)
    # (ie. Ax = y style of linear equation)
    # note that the bias should be included in the last element of the weight tensor.
    def __init__(self, in_features, out_features, device: torch.device):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("_lb", torch.zeros(out_features, device=device))
        self.register_buffer("_ub", torch.zeros(out_features, device=device))
        self.register_buffer(
            "_alb", torch.zeros(out_features, in_features, device=device)
        )
        self.register_buffer(
            "_aub", torch.zeros(out_features, in_features, device=device)
        )
        self.register_buffer("_alb_bias", torch.zeros(out_features, device=device))
        self.register_buffer("_aub_bias", torch.zeros(out_features, device=device))
        # To make the Python type checker happy
        self.lb: torch.Tensor = self._lb
        self.ub: torch.Tensor = self._ub
        self.alb: torch.Tensor = self._alb
        self.aub: torch.Tensor = self._aub
        self.alb_bias: torch.Tensor = self._alb_bias
        self.aub_bias: torch.Tensor = self._aub_bias

    def clone(self) -> "ArgumentedInfo":
        ainfo = ArgumentedInfo(self.in_features, self.out_features, self.lb.device)
        ainfo.lb = self.lb.clone()
        ainfo.ub = self.ub.clone()
        ainfo.alb = self.alb.clone()
        ainfo.aub = self.aub.clone()
        ainfo.alb_bias = self.alb_bias.clone()
        ainfo.aub_bias = self.aub_bias.clone()
        return ainfo

    def forward(self, x: "ArgumentedInfo", compute_alg=True) -> "ArgumentedInfo":
        """
        This is a stateful operation. The input x is the input of the layer.
        After the forward operation, the ArgumentedInfo object will be updated to
        the backpropagated state
        """

        def debug():
            print(f"Forward (lb ub): lb={self.lb} ub={self.ub}")
            print(
                f"Forward: alb_lb={alb_lb} alb_ub={alb_ub} aub_ub={aub_ub} aub_lb={aub_lb}"
            )

        assert x.out_features == self.in_features

        alb_lb = (self.alb + self.alb.abs()) / 2
        alb_ub = (self.alb - self.alb.abs()) / 2
        aub_ub = (self.aub + self.aub.abs()) / 2
        aub_lb = (self.aub - self.aub.abs()) / 2
        # Encoding the condition of the sign of the entry into arithmetic operations
        # Underlying choosing function:
        # f = w * x if w >= 0 else w * y
        #  <==>
        # f = (w + |w|)/2 * x + (w - |w|)/2 * y
        # TODO Implement a direct CUDA/C++ operator for this. Reduce 2x overhead.
        self.lb = (alb_lb @ x.lb) + (alb_ub @ x.ub) + self.alb_bias
        self.ub = (aub_ub @ x.ub) + (aub_lb @ x.lb) + self.aub_bias

        if not compute_alg:
            debug()
            return self
        self.alb = (alb_lb @ x.alb) + (alb_ub @ x.aub)
        self.aub = (aub_ub @ x.aub) + (aub_lb @ x.alb)
        self.alb_bias = (alb_lb @ x.alb_bias) + (alb_ub @ x.aub_bias) + self.alb_bias
        self.aub_bias = (aub_ub @ x.aub_bias) + (aub_lb @ x.alb_bias) + self.aub_bias
        self.in_features, self.out_features = x.in_features, self.in_features
        debug()
        return self

    @property
    def shape(self):
        return self.in_features, self.out_features


ALL = [ArgumentedInfo]
