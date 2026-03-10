"""Tolerance audit helpers for cross-oracle residuals."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from .runtime import tensor_map_inner_product


RELATIVE_FLOOR = 1e-15
ABSOLUTE_FLOOR = 1e-15
SAFETY_FACTOR = 1e3
LOOSENESS_ORDERS = 10
RELATIVE_DENOMINATOR_FLOOR = 1e-300


@dataclass(frozen=True)
class FamilyAudit:
    """Observed and proposed tolerance data for one `(op, family, dtype)` bucket."""

    op: str
    family: str
    dtype: str
    current_rtol: float
    current_atol: float
    max_rel_residual: float
    max_abs_residual: float
    proposed_rtol: float
    proposed_atol: float
    tighten_rtol: bool
    tighten_atol: bool


def max_abs_diff(torch, left: dict[str, object], right: dict[str, object]) -> float:
    max_diff = 0.0
    for name in left:
        diff = (left[name] - right[name]).abs()
        if diff.numel() == 0:
            continue
        max_diff = max(max_diff, float(diff.max().item()))
    return max_diff


def max_rel_diff(torch, left: dict[str, object], right: dict[str, object]) -> float:
    max_diff = 0.0
    for name in left:
        diff = (left[name] - right[name]).abs()
        if diff.numel() == 0:
            continue
        denom = torch.maximum(left[name].abs(), right[name].abs()).clamp_min(
            RELATIVE_DENOMINATOR_FLOOR
        )
        rel = diff / denom
        max_diff = max(max_diff, float(rel.max().item()))
    return max_diff


def scalar_residual(torch, lhs, rhs) -> tuple[float, float]:
    abs_residual = float((lhs - rhs).abs().item())
    denom = max(float(lhs.abs().item()), float(rhs.abs().item()), RELATIVE_DENOMINATOR_FLOOR)
    rel_residual = abs_residual / denom
    return abs_residual, rel_residual


def _family_key(record: dict) -> tuple[str, str, str]:
    return (record["op"], record["family"], record["dtype"])


def propose_tolerance(*, observed_max: float, safety_factor: float, floor: float) -> float:
    """Propose a rounded-up tolerance from the observed maximum residual."""
    if observed_max <= 0.0:
        return floor
    target = max(observed_max * safety_factor, floor)
    exponent = math.ceil(math.log10(target))
    return 10.0 ** exponent


def needs_tightening(*, current: float, observed_max: float, looseness_orders: int) -> bool:
    """Return whether `current` is more than `looseness_orders` looser than observed."""
    if observed_max <= 0.0:
        return False
    return current > observed_max * (10.0 ** looseness_orders)


def comparison_from_observed_residuals(*, max_rel_residual: float, max_abs_residual: float) -> dict[str, float | str]:
    """Build an allclose comparison block from observed family residual maxima."""
    return {
        "kind": "allclose",
        "rtol": propose_tolerance(
            observed_max=max_rel_residual,
            safety_factor=SAFETY_FACTOR,
            floor=RELATIVE_FLOOR,
        ),
        "atol": propose_tolerance(
            observed_max=max_abs_residual,
            safety_factor=SAFETY_FACTOR,
            floor=ABSOLUTE_FLOOR,
        ),
    }


def audit_case_tree(root: Path) -> list[FamilyAudit]:
    """Audit stored cross-oracle tolerances for all published success cases."""
    import torch
    from validators.case_loader import iter_case_files, load_case_file
    from validators.encoding import decode_tensor_map

    aggregates: dict[tuple[str, str, str], dict[str, float]] = {}
    for path in iter_case_files(root):
        for record in load_case_file(path):
            if record["expected_behavior"] != "success":
                continue
            key = _family_key(record)
            probe = record["probes"][0]
            comparison = record["comparison"]
            direction = decode_tensor_map(probe["direction"])
            cotangent = decode_tensor_map(probe["cotangent"])
            pytorch_jvp = decode_tensor_map(probe["pytorch_ref"]["jvp"])
            pytorch_vjp = decode_tensor_map(probe["pytorch_ref"]["vjp"])
            fd_jvp = decode_tensor_map(probe["fd_ref"]["jvp"])

            jvp_abs = max_abs_diff(torch, pytorch_jvp, fd_jvp)
            jvp_rel = max_rel_diff(torch, pytorch_jvp, fd_jvp)
            lhs = tensor_map_inner_product(torch, cotangent, fd_jvp)
            rhs = tensor_map_inner_product(torch, pytorch_vjp, direction)
            adj_abs, adj_rel = scalar_residual(torch, lhs, rhs)

            bucket = aggregates.setdefault(
                key,
                {
                    "current_rtol": float(comparison["rtol"]),
                    "current_atol": float(comparison["atol"]),
                    "max_rel_residual": 0.0,
                    "max_abs_residual": 0.0,
                },
            )
            bucket["max_rel_residual"] = max(
                bucket["max_rel_residual"],
                jvp_rel,
                adj_rel,
            )
            bucket["max_abs_residual"] = max(
                bucket["max_abs_residual"],
                jvp_abs,
                adj_abs,
            )

    audits: list[FamilyAudit] = []
    for (op, family, dtype), bucket in sorted(aggregates.items()):
        max_rel = bucket["max_rel_residual"]
        max_abs = bucket["max_abs_residual"]
        current_rtol = bucket["current_rtol"]
        current_atol = bucket["current_atol"]
        proposed_rtol = propose_tolerance(
            observed_max=max_rel,
            safety_factor=SAFETY_FACTOR,
            floor=RELATIVE_FLOOR,
        )
        proposed_atol = propose_tolerance(
            observed_max=max_abs,
            safety_factor=SAFETY_FACTOR,
            floor=ABSOLUTE_FLOOR,
        )
        audits.append(
            FamilyAudit(
                op=op,
                family=family,
                dtype=dtype,
                current_rtol=current_rtol,
                current_atol=current_atol,
                max_rel_residual=max_rel,
                max_abs_residual=max_abs,
                proposed_rtol=proposed_rtol,
                proposed_atol=proposed_atol,
                tighten_rtol=needs_tightening(
                    current=current_rtol,
                    observed_max=max_rel,
                    looseness_orders=LOOSENESS_ORDERS,
                ),
                tighten_atol=needs_tightening(
                    current=current_atol,
                    observed_max=max_abs,
                    looseness_orders=LOOSENESS_ORDERS,
                ),
            )
        )
    return audits
