"""Processed observable helpers mirroring the PyTorch-aligned v1 families."""

from __future__ import annotations


def _is_differentiable_output(value) -> bool:
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        return True
    try:
        import torch
    except Exception:
        return True
    if not (
        isinstance(value, torch.Tensor)
        and (value.is_floating_point() or value.is_complex())
    ):
        return False
    requires_grad = getattr(value, "requires_grad", None)
    grad_fn = getattr(value, "grad_fn", None)
    if requires_grad is None:
        return True
    return bool(requires_grad or grad_fn is not None)


def _svd_parts(result):
    try:
        return result
    except ValueError as exc:
        raise ValueError("expected an SVD result tuple of (u, s, vh)") from exc


def _uniform_svd_parts(result):
    u, s, vh = _svd_parts(result)
    if not hasattr(s, "shape"):
        return u, s, vh
    k = s.shape[-1]
    return u[..., :k], s, vh[..., :k, :]


def _eigh_parts(result):
    try:
        return result
    except ValueError as exc:
        raise ValueError("expected an eigh result tuple of (values, vectors)") from exc


def apply_observable(
    kind: str,
    result,
    *,
    preserve_identity_keys: tuple[str, ...] | None = None,
):
    """Project a raw op result into a derivative-comparison observable."""
    if kind == "identity":
        if isinstance(result, tuple):
            if preserve_identity_keys is not None:
                return {
                    key: result[int(key.removeprefix("output_"))]
                    for key in preserve_identity_keys
                }
            return {
                f"output_{index}": value
                for index, value in enumerate(result)
                if _is_differentiable_output(value)
            }
        return {"value": result}

    if kind == "svd_u_abs":
        u, _, _ = _uniform_svd_parts(result)
        return {"u": u.abs()}

    if kind == "svd_s":
        _, s, _ = _uniform_svd_parts(result)
        return {"s": s}

    if kind == "svd_vh_abs":
        _, s, vh = _uniform_svd_parts(result)
        return {"s": s, "vh": vh.abs()}

    if kind == "svd_uvh_product":
        u, s, vh = _uniform_svd_parts(result)
        return {"uvh": u @ vh, "s": s}

    if kind == "eigh_values_vectors_abs":
        values, vectors = _eigh_parts(result)
        return {"values": values, "vectors": vectors.abs()}

    if kind == "eig_values_vectors_abs":
        values, vectors = _eigh_parts(result)
        return {"values": values, "vectors": vectors.abs()}

    raise ValueError(f"unsupported observable kind: {kind}")
