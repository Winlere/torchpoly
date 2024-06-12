import logging
import torch

from torchpoly.cert.certificate import Trace
from torchpoly.nn.container import Sequential
from torchpoly.cert.ticket import Ticket

from tests.nn.test_linear import make_linear_simple
from tests.nn.test_activation import make_relu_simple


def test_sequential_simple():
    linear = Sequential(*make_linear_simple()).cuda()
    bound = Ticket.from_bound([-1, -1], [1, 1]).cuda()

    state = (Ticket.from_ticket(bound), Trace())
    cert = linear.certify(bound)
    state = cert.forward(state)

    x, _ = state
    assert torch.allclose(x.cpu().lb, torch.tensor([-2, -2], dtype=torch.float32))
    assert torch.allclose(x.cpu().ub, torch.tensor([2, 2], dtype=torch.float32))

    relu = Sequential(*make_relu_simple()).cuda()
    bound = Ticket.from_bound([-1, -1], [1, 1]).cuda()

    state = (Ticket.from_ticket(bound), Trace())
    cert = relu.certify(bound)
    state = cert.forward(state)

    x, _ = state
    assert torch.allclose(x.cpu().lb, torch.tensor([1, 0], dtype=torch.float32))
    assert torch.allclose(x.cpu().ub, torch.tensor([5.5, 2], dtype=torch.float32))


def test_sequential_compose():
    layers = make_linear_simple()
    linear = Sequential(Sequential(layers[0]), Sequential(layers[1])).cuda()
    bound = Ticket.from_bound([-1, -1], [1, 1]).cuda()

    state = (Ticket.from_ticket(bound), Trace())
    cert = linear.certify(bound)
    state = cert.forward(state)

    x, _ = state
    assert torch.allclose(x.cpu().lb, torch.tensor([-2, -2], dtype=torch.float32))
    assert torch.allclose(x.cpu().ub, torch.tensor([2, 2], dtype=torch.float32))

    layers = make_relu_simple()
    relu = Sequential(Sequential(*layers[:2]), Sequential(*layers[2:])).cuda()
    bound = Ticket.from_bound([-1, -1], [1, 1]).cuda()

    state = (Ticket.from_ticket(bound), Trace())
    cert = relu.certify(bound)
    state = cert.forward(state)

    x, _ = state
    assert torch.allclose(x.cpu().lb, torch.tensor([1, 0], dtype=torch.float32))
    assert torch.allclose(x.cpu().ub, torch.tensor([5.5, 2], dtype=torch.float32))

    layers = make_relu_simple()
    relu = Sequential(Sequential(*layers[:3]), Sequential(*layers[3:])).cuda()
    bound = Ticket.from_bound([-1, -1], [1, 1]).cuda()

    state = (Ticket.from_ticket(bound), Trace())
    cert = relu.certify(bound)
    state = cert.forward(state)

    x, _ = state
    assert torch.allclose(x.cpu().lb, torch.tensor([1, 0], dtype=torch.float32))
    assert torch.allclose(x.cpu().ub, torch.tensor([5.5, 2], dtype=torch.float32))


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    test_sequential_simple()
    test_sequential_compose()
