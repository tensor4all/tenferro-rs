"""Decode repository tensor objects back into torch tensors."""

from __future__ import annotations


def _dtype_map():
    import torch

    return {
        "float32": torch.float32,
        "float64": torch.float64,
        "complex64": torch.complex64,
        "complex128": torch.complex128,
    }


def decode_tensor(encoded: dict):
    """Decode one repository tensor object into a torch tensor."""
    import torch

    dtype = _dtype_map()[encoded["dtype"]]
    shape = tuple(encoded["shape"])
    if encoded["dtype"] in {"complex64", "complex128"}:
        flat = [complex(real, imag) for real, imag in encoded["data"]]
    else:
        flat = [float(value) for value in encoded["data"]]
    return torch.tensor(flat, dtype=dtype).reshape(shape)


def decode_tensor_map(encoded: dict[str, dict]) -> dict[str, object]:
    """Decode a named map of repository tensor objects."""
    return {name: decode_tensor(tensor) for name, tensor in encoded.items()}
