"""Probe construction and normalization helpers."""

from __future__ import annotations

import math


def _flat_square_sum(data) -> float:
    total = 0.0
    for item in data:
        if isinstance(item, list):
            if len(item) == 2 and all(isinstance(x, (int, float)) for x in item):
                total += float(item[0]) ** 2 + float(item[1]) ** 2
            else:
                total += _flat_square_sum(item)
        else:
            total += float(item) ** 2
    return total


def tensor_norm(tensor: dict) -> float:
    """Compute the Frobenius / Euclidean norm of an encoded tensor."""
    return math.sqrt(_flat_square_sum(tensor["data"]))


def _scale_data(data, scale: float):
    scaled = []
    for item in data:
        if isinstance(item, list):
            if len(item) == 2 and all(isinstance(x, (int, float)) for x in item):
                scaled.append([float(item[0]) * scale, float(item[1]) * scale])
            else:
                scaled.append(_scale_data(item, scale))
        else:
            scaled.append(float(item) * scale)
    return scaled


def normalize_tensor(tensor: dict) -> dict:
    """Return a unit-norm copy of an encoded tensor."""
    norm = tensor_norm(tensor)
    if norm == 0.0:
        return dict(tensor)
    normalized = dict(tensor)
    normalized["data"] = _scale_data(tensor["data"], 1.0 / norm)
    return normalized


def normalize_tensor_map(tensor_map: dict[str, dict]) -> dict[str, dict]:
    """Normalize each tensor in a named tensor map independently."""
    return {name: normalize_tensor(tensor) for name, tensor in tensor_map.items()}


def make_probe_record(
    *,
    probe_id: str,
    direction: dict[str, dict],
    cotangent: dict[str, dict],
    pytorch_jvp: dict[str, dict],
    pytorch_vjp: dict[str, dict],
    fd_step: float,
    fd_jvp: dict[str, dict],
) -> dict:
    """Assemble one paired derivative probe record."""
    return {
        "probe_id": probe_id,
        "direction": direction,
        "cotangent": cotangent,
        "pytorch_ref": {
            "jvp": pytorch_jvp,
            "vjp": pytorch_vjp,
        },
        "fd_ref": {
            "method": "central_difference",
            "stencil_order": 2,
            "step": fd_step,
            "jvp": fd_jvp,
        },
    }
