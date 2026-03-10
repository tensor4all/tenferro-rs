"""Finite-difference policy helpers for tensor-ad-oracles v1."""

from __future__ import annotations


FD_POLICY_VERSION = "v1"
BASE_STEPS = {
    "float64": 1e-6,
    "complex128": 1e-7,
}


def compute_step(dtype: str, *, input_norm: float) -> float:
    """Return the v1 central-difference step for a materialized input."""
    try:
        base_step = BASE_STEPS[dtype]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype for fd policy: {dtype}") from exc
    return base_step * max(1.0, float(input_norm))
