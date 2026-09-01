"""Deterministic differentiable references for incremental Householder QR."""

from __future__ import annotations


FAMILIES: tuple[str, ...] = (
    "factor_qr",
    "append_qr",
    "from_factors_qr",
    "selected_q_columns",
    "r",
)

_SAMPLE_SPECS: dict[str, tuple[dict[str, object], ...]] = {
    "factor_qr": (
        {"shape": (4, 2)},
        {"shape": (3, 3)},
        {"shape": (2, 4)},
    ),
    "append_qr": (
        {"a_shape": (4, 2), "b_cols": 1},
        {"a_shape": (3, 2), "b_cols": 1},
        {"a_shape": (3, 2), "b_cols": 3},
    ),
    "from_factors_qr": (
        {"q_shape": (4, 2), "r_cols": 2},
        {"q_shape": (3, 3), "r_cols": 3},
        {"q_shape": (2, 2), "r_cols": 4},
    ),
    "selected_q_columns": (
        {"shape": (4, 3), "start": 0, "end": 2},
        {"shape": (3, 3), "start": 1, "end": 3},
        {"shape": (2, 4), "start": 1, "end": 2},
    ),
    "r": (
        {"shape": (4, 2)},
        {"shape": (3, 3)},
        {"shape": (2, 4)},
    ),
}


def sample_specs(family: str) -> tuple[dict[str, object], ...]:
    """Return deterministic shape/metadata specifications for one family."""
    return _SAMPLE_SPECS[family]


def _stable_matrix(torch, *, dtype, shape: tuple[int, int], offset: float = 0.0):
    rows, cols = shape
    row = torch.arange(1, rows + 1, dtype=torch.float64)[:, None]
    col = torch.arange(1, cols + 1, dtype=torch.float64)[None, :]
    real = torch.sin(row * (col + offset)) + 0.2 * torch.cos((row + offset) * col)
    real = real + 2.5 * torch.eye(rows, cols, dtype=torch.float64)
    if dtype.is_complex:
        imag = 0.17 * torch.cos((row + 0.3 + offset) * (col + 0.2))
        return (real + 1j * imag).to(dtype=dtype)
    return real.to(dtype=dtype)


def _upper_trapezoidal(torch, *, dtype, rows: int, cols: int):
    raw = _stable_matrix(torch, dtype=dtype, shape=(rows, cols), offset=0.7)
    return torch.triu(raw)


def canonical_qr(torch, a):
    """Return reduced QR with positive real diagonal in ``R``."""
    q, r = torch.linalg.qr(a, mode="reduced")
    diagonal = torch.diagonal(r)
    phase = diagonal / diagonal.abs()
    q = q * phase
    r = phase.conj()[:, None] * r
    return {"q": q, "r": r}


def make_inputs(torch, *, family: str, dtype, sample_spec: dict[str, object]):
    """Build one deterministic full-rank differentiable input map."""
    if family in {"factor_qr", "selected_q_columns", "r"}:
        shape = tuple(sample_spec["shape"])
        return {"a": _stable_matrix(torch, dtype=dtype, shape=shape)}
    if family == "append_qr":
        a_shape = tuple(sample_spec["a_shape"])
        b_shape = (a_shape[0], int(sample_spec["b_cols"]))
        return {
            "a": _stable_matrix(torch, dtype=dtype, shape=a_shape),
            "b": _stable_matrix(torch, dtype=dtype, shape=b_shape, offset=1.1),
        }
    if family == "from_factors_qr":
        q_shape = tuple(sample_spec["q_shape"])
        q = canonical_qr(
            torch,
            _stable_matrix(torch, dtype=dtype, shape=q_shape, offset=0.4),
        )["q"]
        r = _upper_trapezoidal(
            torch,
            dtype=dtype,
            rows=q_shape[1],
            cols=int(sample_spec["r_cols"]),
        )
        return {"q": q, "r": r}
    raise ValueError(f"unsupported incremental QR family: {family}")


def metadata(*, family: str, sample_spec: dict[str, object]) -> dict[str, object]:
    """Return JSON metadata for one family sample."""
    if family == "selected_q_columns":
        return {
            "start": int(sample_spec["start"]),
            "end": int(sample_spec["end"]),
            "rank_status": "full_rank",
        }
    return {"rank_status": "full_rank"}


def observable(torch, *, family: str, inputs: dict[str, object], op_kwargs: dict[str, object]):
    """Evaluate the canonical differentiable observable for one family."""
    if family == "append_qr":
        return canonical_qr(torch, torch.cat((inputs["a"], inputs["b"]), dim=1))
    if family == "from_factors_qr":
        return canonical_qr(torch, inputs["q"] @ torch.triu(inputs["r"]))

    factors = canonical_qr(torch, inputs["a"])
    if family == "factor_qr":
        return factors
    if family == "selected_q_columns":
        start = int(op_kwargs["start"])
        end = int(op_kwargs["end"])
        return {"q": factors["q"][:, start:end]}
    if family == "r":
        return {"r": factors["r"]}
    raise ValueError(f"unsupported incremental QR family: {family}")


def observable_tuple(torch, *, family: str, input_names, input_values, op_kwargs):
    """Evaluate an observable from ordered inputs for ``torch.func``."""
    output = observable(
        torch,
        family=family,
        inputs=dict(zip(input_names, input_values, strict=True)),
        op_kwargs=op_kwargs,
    )
    return tuple(output.values())


def project_direction(torch, *, family: str, direction: dict[str, object]):
    """Keep factor-import directions in the upper-trapezoidal domain."""
    if family != "from_factors_qr":
        return direction
    return {**direction, "r": torch.triu(direction["r"])}
