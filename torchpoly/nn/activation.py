import torch
import torch.nn as nn

from torchpoly.cert.certificate import Certificate
from torchpoly.cert.ticket import Ticket
from torchpoly.nn.module import Module


class ReLUCertificate(Certificate):

    def forward(self, x: Ticket) -> Ticket:
        relu_lb = torch.max(x.lb, torch.zeros_like(x.lb))
        relu_ub = torch.max(x.ub, torch.zeros_like(x.ub))

        ub_k = torch.div(
            relu_ub - relu_lb, torch.max(x.ub - x.lb, torch.ones_like(x.lb))
        )
        ub_b = relu_ub - torch.mul(ub_k, relu_ub)

        lb_k = torch.lt(x.lb + x.ub, torch.zeros_like(x.lb))
        lb_b = torch.zeros_like(x.lb)

        self.ticket.lb = torch.mul(lb_k, x.lb) + lb_b
        self.ticket.ub = torch.mul(ub_k, x.ub) + ub_b
        self.ticket.alb = torch.diag(lb_k)
        self.ticket.aub = torch.diag(ub_k)
        self.ticket.alb_bias = torch.mul(lb_k, x.lb)
        self.ticket.aub_bias = ub_b

        return Ticket.from_ticket(self.ticket)


class ReLU(Module, nn.ReLU):

    def __init__(self, inplace: bool = False):
        super(ReLU, self).__init__(inplace=inplace)

    def certify(self, device=None) -> Certificate:
        return ReLUCertificate(Ticket.zeros(1, 1, device=device))
