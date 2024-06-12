import logging
import torch
import torch.nn as nn

from typing import List, Tuple


logger = logging.getLogger(__name__)


def forward_impl(
    self_alb: torch.Tensor,
    self_aub: torch.Tensor,
    self_alb_bias: torch.Tensor,
    self_aub_bias: torch.Tensor,
    x_lb: torch.Tensor,
    x_ub: torch.Tensor,
    x_alb: torch.Tensor,
    x_aub: torch.Tensor,
    x_alb_bias: torch.Tensor,
    x_aub_bias: torch.Tensor,
    with_alg: bool,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    alb_lb = (self_alb + self_alb.abs()) / 2
    alb_ub = (self_alb - self_alb.abs()) / 2

    aub_lb = (self_aub - self_aub.abs()) / 2
    aub_ub = (self_aub + self_aub.abs()) / 2

    new_lb = (alb_lb @ x_lb) + (alb_ub @ x_ub) + self_alb_bias
    new_ub = (aub_lb @ x_lb) + (aub_ub @ x_ub) + self_aub_bias

    if with_alg:
        new_alb = alb_lb @ x_alb + alb_ub @ x_aub
        new_aub = aub_lb @ x_alb + aub_ub @ x_aub

        new_alb_bias = alb_lb @ x_alb_bias + alb_ub @ x_aub_bias + self_alb_bias
        new_aub_bias = aub_lb @ x_alb_bias + aub_ub @ x_aub_bias + self_aub_bias
    else:
        new_alb = self_alb.clone()
        new_aub = self_aub.clone()

        new_alb_bias = self_alb_bias.clone()
        new_aub_bias = self_aub_bias.clone()

    return new_lb, new_ub, new_alb, new_aub, new_alb_bias, new_aub_bias


forward_impl_optimized = None


class Ticket(nn.Module):
    """
    Each line in the alb(aub) is an equation of the computed lower(upper) bound of the each output.
    Suppose the input tensor is x and the output is y

    Then algebraicly,

    min(y[i]) = lb[i] = alb[i][j] * x[j] + alb_bias[i] for some x[j] (x[j]=lb_x[j] if alb[i][j] >= 0)

    max(y[i]) = ub[i] = aub[i][j] * x[j] + ulb_bias[i] for some x[j] (x[j]=ub_x[j] if aub[i][j] >= 0)

    (ie. Ax = y style of linear equation)

    Note that the bias should be included in the last element of the weight tensor.
    """

    def __init__(
        self,
        lb: torch.Tensor,
        ub: torch.Tensor,
        alb: torch.Tensor,
        aub: torch.Tensor,
        alb_bias: torch.Tensor,
        aub_bias: torch.Tensor,
    ):
        super(Ticket, self).__init__()

        self.register_buffer("lb", lb)
        self.register_buffer("ub", ub)
        self.register_buffer("alb", alb)
        self.register_buffer("aub", aub)
        self.register_buffer("alb_bias", alb_bias)
        self.register_buffer("aub_bias", aub_bias)

    @classmethod
    def zeros(cls, in_features: int, out_features: int, device=None):
        factory_kwargs = {"device": device, "dtype": torch.float32}
        return cls(
            torch.zeros(out_features, **factory_kwargs),
            torch.zeros(out_features, **factory_kwargs),
            torch.zeros(out_features, in_features, **factory_kwargs),
            torch.zeros(out_features, in_features, **factory_kwargs),
            torch.zeros(out_features, **factory_kwargs),
            torch.zeros(out_features, **factory_kwargs),
        )

    @classmethod
    def from_ticket(cls, ticket: "Ticket"):
        return cls(
            ticket.lb.clone(),
            ticket.ub.clone(),
            ticket.alb.clone(),
            ticket.aub.clone(),
            ticket.alb_bias.clone(),
            ticket.aub_bias.clone(),
        )

    @classmethod
    def from_bound(cls, lb: List[float], ub: List[float], device=None):
        assert len(lb) == len(ub)

        factory_kwargs = {"device": device, "dtype": torch.float32}

        ticket = cls.zeros(1, len(lb), device=device)
        ticket.lb.copy_(torch.tensor(lb, **factory_kwargs))
        ticket.ub.copy_(torch.tensor(ub, **factory_kwargs))
        ticket.alb_bias.copy_(torch.tensor(lb, **factory_kwargs))
        ticket.aub_bias.copy_(torch.tensor(ub, **factory_kwargs))
        return ticket

    def forward(
        self,
        x: "Ticket",
        with_alg: bool = True,
        use_jit: bool = False,
    ) -> "Ticket":
        if use_jit:
            global forward_impl_optimized
            if forward_impl_optimized is None:
                forward_impl_optimized = torch.compile(forward_impl)

            forward_fn = forward_impl_optimized
        else:
            forward_fn = forward_impl

        with torch.no_grad():
            new_lb, new_ub, new_alb, new_aub, new_alb_bias, new_aub_bias = forward_fn(
                self.alb,
                self.aub,
                self.alb_bias,
                self.aub_bias,
                x.lb,
                x.ub,
                x.alb,
                x.aub,
                x.alb_bias,
                x.aub_bias,
                with_alg,
            )

        ticket = Ticket(new_lb, new_ub, new_alb, new_aub, new_alb_bias, new_aub_bias)

        return ticket
