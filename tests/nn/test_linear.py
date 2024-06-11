import logging
import torch

from torchpoly.nn.linear import Linear
from torchpoly.cert.ticket import Ticket


def make_linear_simple():
    w = torch.tensor([1, 1, 1, -1], dtype=torch.float32).reshape(2, 2)
    b = torch.tensor([0, 0], dtype=torch.float32)

    layer1 = Linear(2, 2)
    layer1.weight = torch.nn.Parameter(w)
    layer1.bias = torch.nn.Parameter(b)

    layer2 = Linear(2, 2)
    layer2.weight = torch.nn.Parameter(w)
    layer2.bias = torch.nn.Parameter(b)

    return [layer1, layer2]


def test_linear_simple():
    model = make_linear_simple()
    bound = Ticket.from_bound([-1, -1], [1, 1]).cuda()

    state = (Ticket.from_ticket(bound), list())
    for layer in model:
        cert = layer.cuda().certify(bound)
        state = cert.forward(state)

    x, _ = state
    assert torch.allclose(x.cpu().lb, torch.tensor([-2, -2], dtype=torch.float32))
    assert torch.allclose(x.cpu().ub, torch.tensor([2, 2], dtype=torch.float32))


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    test_linear_simple()
