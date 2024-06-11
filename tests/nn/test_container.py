import logging
import torch

from torchpoly.nn.container import Sequential
from torchpoly.cert.ticket import Ticket

from .test_linear import make_linear_simple
from .test_activation import make_relu_simple


def test_sequential_simple():
    linear = Sequential(*make_linear_simple())
    bound = Ticket.from_bound([-1, -1], [1, 1])

    state = (Ticket.from_ticket(bound), list())
    cert = linear.certify(bound)
    state = cert.forward(state)

    x, _ = state
    assert torch.allclose(x.lb, torch.tensor([-2, -2], dtype=torch.float32))
    assert torch.allclose(x.ub, torch.tensor([2, 2], dtype=torch.float32))

    relu = Sequential(*make_relu_simple())
    bound = Ticket.from_bound([-1, -1], [1, 1])

    state = (Ticket.from_ticket(bound), list())
    cert = relu.certify(bound)
    state = cert.forward(state)

    x, _ = state
    assert torch.allclose(x.lb, torch.tensor([1, 0], dtype=torch.float32))
    assert torch.allclose(x.ub, torch.tensor([5.5, 2], dtype=torch.float32))


def test_sequential_compose():
    layers = make_linear_simple()
    linear = Sequential(Sequential(layers[0]), Sequential(layers[1]))
    bound = Ticket.from_bound([-1, -1], [1, 1])

    state = (Ticket.from_ticket(bound), list())
    cert = linear.certify(bound)
    state = cert.forward(state)

    x, _ = state
    assert torch.allclose(x.lb, torch.tensor([-2, -2], dtype=torch.float32))
    assert torch.allclose(x.ub, torch.tensor([2, 2], dtype=torch.float32))

    layers = make_relu_simple()
    relu = Sequential(Sequential(*layers[:2]), Sequential(*layers[2:]))
    bound = Ticket.from_bound([-1, -1], [1, 1])

    state = (Ticket.from_ticket(bound), list())
    cert = relu.certify(bound)
    state = cert.forward(state)

    x, _ = state
    assert torch.allclose(x.lb, torch.tensor([1, 0], dtype=torch.float32))
    assert torch.allclose(x.ub, torch.tensor([5.5, 2], dtype=torch.float32))

    layers = make_relu_simple()
    relu = Sequential(Sequential(*layers[:3]), Sequential(*layers[3:]))
    bound = Ticket.from_bound([-1, -1], [1, 1])

    state = (Ticket.from_ticket(bound), list())
    cert = relu.certify(bound)
    state = cert.forward(state)

    x, _ = state
    assert torch.allclose(x.lb, torch.tensor([1, 0], dtype=torch.float32))
    assert torch.allclose(x.ub, torch.tensor([5.5, 2], dtype=torch.float32))


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    test_sequential_simple()
    test_sequential_compose()
