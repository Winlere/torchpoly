import torch.nn as nn
import warnings

from typing import Tuple

from torchpoly.cert.certificate import Certificate, Trace
from torchpoly.cert.ticket import Ticket
from torchpoly.nn.module import Module


class SequentialCertificate(Certificate):

    def __init__(self, bound: Ticket, module: nn.Sequential, device=None):
        super(SequentialCertificate, self).__init__(bound)
        self.module = module
        self.device = device

    def forward(self, state: Tuple[Ticket, Trace]) -> Tuple[Ticket, Trace]:
        for layer in self.module:
            cert = layer.certify(self.bound, self.device)
            state = cert.forward(state)

        return state

    def backward(self, x: Ticket) -> Ticket:
        warnings.warn(
            "SequentialCertificate.backward() should not be called directly. "
            "Please use the backward() method of the certificates returned by the forward() method."
        )
        return x


class Sequential(Module, nn.Sequential):

    def certify(self, bound: Ticket, device=None) -> Certificate:
        return SequentialCertificate(bound, self, device)
