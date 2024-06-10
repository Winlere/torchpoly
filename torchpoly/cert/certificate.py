import torch

from torchpoly.cert.ticket import Ticket


class Certificate:

    def __init__(self, ticket: Ticket):
        self.ticket = ticket

    def forward(self, x: Ticket) -> Ticket:
        raise NotImplementedError

    def backward(self, x: Ticket) -> Ticket:
        return x(self.ticket)

    def update(self, x: Ticket) -> None:
        self.ticket.lb.copy_(x.lb)
        self.ticket.ub.copy_(x.ub)

    def __call__(self, x: Ticket) -> Ticket:
        with torch.no_grad():
            return Ticket.from_ticket(self.forward(x))

    def cuda(self):
        self.ticket.cuda()
        return self

    def cpu(self):
        self.ticket.cpu()
        return self

    def to(self, device=None, non_blocking: bool = False):
        self.ticket.to(device=device, non_blocking=non_blocking)
        return self
