import logging
import torch
import torch.nn as nn

from typing import List


logger = logging.getLogger(__name__)


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

    def forward(self, x: "Ticket", with_alg: bool = True) -> "Ticket":
        with torch.no_grad():
            alb_lb = (self.alb + self.alb.abs()) / 2
            alb_ub = (self.alb - self.alb.abs()) / 2

            aub_lb = (self.aub - self.aub.abs()) / 2
            aub_ub = (self.aub + self.aub.abs()) / 2

            new_lb = (alb_lb @ x.lb) + (alb_ub @ x.ub) + self.alb_bias
            new_ub = (aub_lb @ x.lb) + (aub_ub @ x.ub) + self.aub_bias

            if with_alg:
                new_alb = alb_lb @ x.alb + alb_ub @ x.aub
                new_aub = aub_lb @ x.alb + aub_ub @ x.aub

                new_alb_bias = alb_lb @ x.alb_bias + alb_ub @ x.aub_bias + self.alb_bias
                new_aub_bias = aub_lb @ x.alb_bias + aub_ub @ x.aub_bias + self.aub_bias
            else:
                new_alb = self.alb.clone()
                new_aub = self.aub.clone()

                new_alb_bias = self.alb_bias.clone()
                new_aub_bias = self.aub_bias.clone()

        ticket = Ticket(new_lb, new_ub, new_alb, new_aub, new_alb_bias, new_aub_bias)

        # logger.debug(f"lb={ticket.lb} ub={ticket.ub}")
        # logger.debug(f"alb_lb={alb_lb} alb_ub={alb_ub} aub_lb={aub_lb} aub_ub={aub_ub}")

        return ticket
