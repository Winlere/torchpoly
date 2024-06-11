import torch
import torch.nn as nn

from typing import Tuple

from torchpoly.cert.certificate import Certificate, Trace
from torchpoly.cert.ticket import Ticket
from torchpoly.nn.module import Module


class ReLUCertificate(Certificate):

    def __init__(self, bound: Ticket, immediate: Ticket):
        super(ReLUCertificate, self).__init__(bound)
        self.immediate = immediate

    def forward(self, state: Tuple[Ticket, Trace]) -> Tuple[Ticket, Trace]:
        x, trace = state

        relu_lb = torch.max(x.lb, torch.zeros_like(x.lb))
        relu_ub = torch.max(x.ub, torch.zeros_like(x.ub))

        ub_k = torch.div(
            relu_ub - relu_lb, torch.max(x.ub - x.lb, torch.ones_like(x.lb))
        )
        ub_b = relu_ub - torch.mul(ub_k, relu_ub)

        lb_k = torch.lt(x.lb + x.ub, torch.zeros_like(x.lb)).float()
        lb_b = torch.zeros_like(x.lb)

        self.immediate.lb = torch.mul(lb_k, x.lb) + lb_b
        self.immediate.ub = torch.mul(ub_k, x.ub) + ub_b
        self.immediate.alb = torch.diag(lb_k)
        self.immediate.aub = torch.diag(ub_k)
        self.immediate.alb_bias = torch.mul(lb_k, x.lb)
        self.immediate.aub_bias = ub_b

        back = Ticket.from_ticket(self.immediate)
        for cert in reversed(trace):
            back = cert.backward(back)

        back = back(self.bound)
        self.immediate.lb.copy_(back.lb)
        self.immediate.ub.copy_(back.ub)

        return Ticket.from_ticket(self.immediate), trace + [self]

    def backward(self, x: Ticket) -> Ticket:
        return x(self.immediate)


class ReLU(Module, nn.ReLU):

    def __init__(self, inplace: bool = False):
        super(ReLU, self).__init__(inplace=inplace)

    def certify(self, bound: Ticket, device=None) -> Certificate:
        return ReLUCertificate(bound, Ticket.zeros(1, 1, device=device))
