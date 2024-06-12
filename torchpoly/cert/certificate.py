from typing import List, Tuple

from torchpoly.cert.ticket import Ticket


class Trace(List["Certificate"]):
    pass


class Certificate:

    def __init__(self, bound: Ticket):
        self.bound = bound

    def forward(self, state: Tuple[Ticket, Trace]) -> Tuple[Ticket, Trace]:
        raise NotImplementedError

    def backward(self, x: Ticket) -> Ticket:
        raise NotImplementedError
