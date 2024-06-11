import torch.nn as nn

from torchpoly.cert.certificate import Certificate
from torchpoly.cert.ticket import Ticket


class Module(nn.Module):

    def certify(self, bound: Ticket, device=None) -> Certificate:
        raise NotImplementedError
