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

    bound = Ticket.zeros(1, 2)
    bound.lb = torch.tensor([-1, -1], dtype=torch.float32)
    bound.ub = torch.tensor([1, 1], dtype=torch.float32)
    bound.alb_bias = torch.tensor([-1, -1], dtype=torch.float32)
    bound.aub_bias = torch.tensor([1, 1], dtype=torch.float32)

    ticket = Ticket.from_ticket(bound)

    certs = []
    for idx, layer in enumerate(model):
        cert = layer.certify()

        ticket = cert.forward(ticket)
        print(f"(Forward) Layer {idx}: {ticket.lb}, {ticket.ub}")
        back = Ticket.from_ticket(ticket)

        for r_idx, r_cert in enumerate(reversed(certs)):
            back = r_cert.backward(back)
            print(f"\t(Backsubstitute) Layer {idx - r_idx - 1}: {back.lb} {back.ub}")

        if idx != 0:
            back = back(bound)
            cert.update(ticket)
            print(f"\t(Update) Layer {idx}: {back.lb}, {back.ub}")

        certs.append(cert)


if __name__ == "__main__":
    sanity_check()
