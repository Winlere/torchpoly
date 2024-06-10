import torch.nn as nn

from torchpoly.cert.certificate import Certificate


class Module(nn.Module):

    def certify(self, device=None) -> Certificate:
        raise NotImplementedError
