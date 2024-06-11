from typing import Tuple
import torch.nn as nn

from torchpoly.cert.certificate import Certificate, Trace
from torchpoly.cert.ticket import Ticket
from torchpoly.nn.module import Module


class LinearCertificate(Certificate):

    def __init__(self, bound: Ticket, immediate: Ticket):
        super(LinearCertificate, self).__init__(bound)
        self.immediate = immediate

    def forward(self, state: Tuple[Ticket, Trace]) -> Tuple[Ticket, Trace]:
        x, trace = state

        self.immediate = self.immediate(x, with_alg=False)

        back = Ticket.from_ticket(self.immediate)
        for cert in reversed(trace):
            back = cert.backward(back)

        back = back(self.bound)
        self.immediate.lb.copy_(back.lb)
        self.immediate.ub.copy_(back.ub)

        return Ticket.from_ticket(self.immediate), trace + [self]

    def backward(self, x: Ticket) -> Ticket:
        return x(self.immediate)


class Linear(Module, nn.Linear):

    def __init__(
        self, in_features: int, out_features: int, bias: bool = True, device=None
    ):
        super(Linear, self).__init__(in_features, out_features, bias, device=device)

    def certify(self, bound: Ticket, device=None) -> Certificate:
        if device is None:
            device = self.weight.device

        immediate = Ticket.zeros(self.in_features, self.out_features, device=device)
        immediate.alb.copy_(self.weight)
        immediate.aub.copy_(self.weight)

        if self.bias is not None:
            immediate.alb_bias.copy_(self.bias)
            immediate.aub_bias.copy_(self.bias)

        return LinearCertificate(bound, immediate)
