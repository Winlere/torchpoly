import gzip
import onnx
import torch.nn as nn

from onnx2pytorch import ConvertModel

import torchpoly.nn as torchpoly_nn

from torchpoly.nn.container import Sequential


def load(path: str) -> onnx.ModelProto:
    """Load an ONNX model from a file. Supports gzip compression."""
    if path.endswith(".gz"):
        with gzip.open(path, "rb") as f:
            model = f.read()
        return onnx.load_model_from_string(model)

    return onnx.load(path)


def as_sequential(model: onnx.ModelProto) -> Sequential:
    """Convert an ONNX model to a torchpoly nn.Sequential model."""
    dirty_model = ConvertModel(model)

    def visit(node: nn.Module):
        children = list(node.children())
        if not children:
            yield node

        for child in children:
            yield from visit(child)

    seq = list()
    for node in visit(dirty_model):
        name = node.__class__.__name__
        correspond = getattr(torchpoly_nn, name, None)
        if correspond is None:
            continue
            # raise ValueError(f"Unsupported layer: {name}")

        child = correspond.__new__(correspond)
        child.__dict__.update(node.__dict__)
        seq.append(child)

    return Sequential(*seq)
