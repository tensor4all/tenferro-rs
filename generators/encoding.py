"""Tensor encoding helpers for JSON case materialization."""

from __future__ import annotations


def _canonical_dtype_name(dtype) -> str:
    name = str(dtype)
    return name.removeprefix("torch.")


def _flatten_real_data(value) -> list[float]:
    if isinstance(value, list):
        flattened: list[float] = []
        for item in value:
            flattened.extend(_flatten_real_data(item))
        return flattened
    return [float(value)]


def _flatten_complex_data(value) -> list[list[float]]:
    if isinstance(value, list):
        flattened: list[list[float]] = []
        for item in value:
            flattened.extend(_flatten_complex_data(item))
        return flattened
    return [[float(value.real), float(value.imag)]]


def encode_tensor(tensor) -> dict:
    """Encode a tensor-like object into the repository wire format."""
    materialized = tensor.detach().clone().cpu()
    dtype = _canonical_dtype_name(materialized.dtype)
    shape = list(materialized.shape)
    raw = materialized.tolist()
    data = (
        _flatten_complex_data(raw)
        if materialized.is_complex()
        else _flatten_real_data(raw)
    )
    return {
        "dtype": dtype,
        "shape": shape,
        "order": "row_major",
        "data": data,
    }


def encode_tensor_map(tensors: dict[str, object]) -> dict[str, dict]:
    """Encode a named tensor map into repository tensor objects."""
    return {name: encode_tensor(tensor) for name, tensor in tensors.items()}
