import logging
import torch


from torchpoly.cert.ticket import Ticket
from torchpoly.cert.verifier import Verifier, VerifyResult
from torchpoly.nn.container import Sequential

from tests.nn.test_activation import make_relu_simple


def test_verifier_simple():
    linear = Sequential(*make_relu_simple()).cuda()
    bound = Ticket.from_bound([-1, -1], [1, 1]).cuda()

    verifier = Verifier(
        A=torch.tensor([[-1, 0]], dtype=torch.float32),
        b=torch.tensor([0], dtype=torch.float32),
    ).cuda()

    result = verifier(linear, bound)
    assert result == VerifyResult.HOLD

    verifier = Verifier(
        A=torch.tensor([[1, 0]], dtype=torch.float32),
        b=torch.tensor([0], dtype=torch.float32),
    ).cuda()

    result = verifier(linear, bound)
    assert result == VerifyResult.UNKNOWN


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    test_verifier_simple()
