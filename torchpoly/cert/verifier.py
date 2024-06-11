import enum
import torch

from torchpoly.cert.ticket import Ticket
from torchpoly.nn.module import Module


class VerifyResult(enum.Enum):
    """The result of the verification."""

    HOLD = "Property holds"
    UNKNOWN = "Property unknown"


class Verifier(Ticket):
    def __init__(self, A: torch.Tensor, b: torch.Tensor, device=None):
        """Accept a postcondition and construct a postcondition layer. (The nature is a linear)
        A postcondition are linear constraint(s) Ax + b <= 0.

        Args:
            A (array-like): The matrix A. Each row represents a constraint.
            b (array-like): The vector b.
        """

        factory_kwargs = {"device": device, "dtype": torch.float32}

        super(Verifier, self).__init__(
            lb=torch.zeros(A.shape[0], **factory_kwargs),
            ub=torch.zeros(A.shape[0], **factory_kwargs),
            alb=torch.clone(A).to(**factory_kwargs),
            aub=torch.clone(A).to(**factory_kwargs),
            alb_bias=torch.clone(b).to(**factory_kwargs),
            aub_bias=torch.clone(b).to(**factory_kwargs),
        )

    def forward(self, model: Module, x: Ticket, device=None) -> VerifyResult:
        """Verify the property of the model."""
        cert = model.certify(x, device=device)
        x, _ = cert.forward((Ticket.from_ticket(x), list()))

        x = super(Verifier, self).forward(x, with_alg=False)

        if torch.any(x.ub <= 0):
            return VerifyResult.HOLD

        return VerifyResult.UNKNOWN
