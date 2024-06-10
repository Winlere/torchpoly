import torch.nn as nn

from torchpoly.cert.certificate import Certificate
from torchpoly.cert.ticket import Ticket
from torchpoly.nn.module import Module


class LinearCertificate(Certificate):

    def forward(self, x: Ticket) -> Ticket:
        self.ticket = self.ticket(x, with_alg=False)
        return Ticket.from_ticket(self.ticket)


class Linear(Module, nn.Linear):

    def __init__(
        self, in_features: int, out_features: int, bias: bool = True, device=None
    ):
        super(Linear, self).__init__(in_features, out_features, bias, device=device)

    def certify(self, device=None) -> Certificate:
        if device is None:
            device = self.weight.device

        ticket = Ticket.zeros(self.in_features, self.out_features, device=device)
        ticket.alb.copy_(self.weight)
        ticket.aub.copy_(self.weight)

        if self.bias is not None:
            ticket.alb_bias.copy_(self.bias)
            ticket.aub_bias.copy_(self.bias)

        return LinearCertificate(ticket)
