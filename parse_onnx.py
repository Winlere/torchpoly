import gzip
import onnx
from onnx2pytorch import ConvertModel


def load_onnx_gz(file_path):
    with gzip.open(file_path, "rb") as f:
        model_data = f.read()
    model = onnx.load_model_from_string(model_data)
    return model


def load_onnx(file_path: str):
    """Load ONNX model from file path. Support for .gz files."""
    if file_path.endswith(".gz"):
        model = load_onnx_gz(file_path)
    elif file_path.endswith(".onnx"):
        model = onnx.load(file_path)
    else:
        raise ValueError(f"Unsupported file extension: {file_path}")
    # TODO Make PyTorch happy, elinminate the "UserWarning: The given NumPy array is not writable, and PyTorch does not support non-writable tensors
    # print(model)
    return ConvertModel(model)
