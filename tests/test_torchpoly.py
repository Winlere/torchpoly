import torch

from torchpoly.nn.linear import Linear
from torchpoly.cert.ticket import Ticket


def sanity_check():
    w = torch.tensor([1, 1, 1, -1], dtype=torch.float32).reshape(2, 2)
    b = torch.tensor([0, 0], dtype=torch.float32)

    layer1 = Linear(2, 2)
    layer1.weight = torch.nn.Parameter(w)
    layer1.bias = torch.nn.Parameter(b)

    layer2 = Linear(2, 2)
    layer2.weight = torch.nn.Parameter(w)
    layer2.bias = torch.nn.Parameter(b)

    model = [layer1, layer2]
    bound = Ticket.from_bound([-1, -1], [1, 1])

    state = (Ticket.from_ticket(bound), list())
    for layer in model:
        cert = layer.certify(bound)
        state = cert.forward(state)


if __name__ == "__main__":
    sanity_check()
