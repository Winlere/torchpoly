import logging
import torch

from torchpoly.nn.activation import ReLU
from torchpoly.nn.linear import Linear
from torchpoly.cert.ticket import Ticket


def test_linear():
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

    x, _ = state
    assert torch.allclose(x.lb, torch.tensor([-2, -2], dtype=torch.float32))
    assert torch.allclose(x.ub, torch.tensor([2, 2], dtype=torch.float32))


def test_relu():
    w = torch.tensor([1, 1, 1, -1], dtype=torch.float32).reshape(2, 2)
    b = torch.tensor([0, 0], dtype=torch.float32)

    linear1 = Linear(2, 2)
    linear1.weight = torch.nn.Parameter(w)
    linear1.bias = torch.nn.Parameter(b)

    relu1 = ReLU()

    linear2 = Linear(2, 2)
    linear2.weight = torch.nn.Parameter(w)
    linear2.bias = torch.nn.Parameter(b)

    relu2 = ReLU()

    w = torch.tensor([1, 1, 0, 1], dtype=torch.float32).reshape(2, 2)
    b = torch.tensor([1, 0], dtype=torch.float32)

    linear3 = Linear(2, 2)
    linear3.weight = torch.nn.Parameter(w)
    linear3.bias = torch.nn.Parameter(b)

    model = [linear1, relu1, linear2, relu2, linear3]
    bound = Ticket.from_bound([-1, -1], [1, 1])

    state = (Ticket.from_ticket(bound), list())
    for layer in model:
        cert = layer.certify(bound)
        state = cert.forward(state)

    x, _ = state
    assert torch.allclose(x.lb, torch.tensor([1, 0], dtype=torch.float32))
    assert torch.allclose(x.ub, torch.tensor([5.5, 2], dtype=torch.float32))


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    test_linear()
    test_relu()
