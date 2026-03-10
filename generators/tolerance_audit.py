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
    current_second_order_rtol: float | None
    current_second_order_atol: float | None
    max_second_order_rel_residual: float | None
    max_second_order_abs_residual: float | None
    proposed_second_order_rtol: float | None
    proposed_second_order_atol: float | None
    tighten_second_order_rtol: bool
    tighten_second_order_atol: bool


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


def hvp_residuals(torch, pytorch_hvp: dict[str, object], fd_hvp: dict[str, object]) -> tuple[float, float]:
    """Measure max absolute and relative residuals for scalarized HVP tensor maps."""
    return (
        max_abs_diff(torch, pytorch_hvp, fd_hvp),
        max_rel_diff(torch, pytorch_hvp, fd_hvp),
    )


def _family_key(record: dict) -> tuple[str, str, str]:
    return (record["op"], record["family"], record["dtype"])


def _comparison_block(comparison: dict, order: str) -> dict | None:
    if order in comparison:
        return comparison[order]
    if order == "first_order":
        return comparison
    return None


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

    aggregates: dict[tuple[str, str, str], dict[str, object]] = {}
    for path in iter_case_files(root):
        for record in load_case_file(path):
            if record["expected_behavior"] != "success":
                continue
            key = _family_key(record)
            probe = record["probes"][0]
            comparison = record["comparison"]
            first_order = _comparison_block(comparison, "first_order")
            second_order = _comparison_block(comparison, "second_order")
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
            second_order_abs = None
            second_order_rel = None
            if "hvp" in probe["pytorch_ref"] and "hvp" in probe["fd_ref"]:
                pytorch_hvp = decode_tensor_map(probe["pytorch_ref"]["hvp"])
                fd_hvp = decode_tensor_map(probe["fd_ref"]["hvp"])
                second_order_abs, second_order_rel = hvp_residuals(
                    torch,
                    pytorch_hvp,
                    fd_hvp,
                )

            bucket = aggregates.setdefault(
                key,
                {
                    "current_rtol": float(first_order["rtol"]),
                    "current_atol": float(first_order["atol"]),
                    "max_rel_residual": 0.0,
                    "max_abs_residual": 0.0,
                    "current_second_order_rtol": None,
                    "current_second_order_atol": None,
                    "max_second_order_rel_residual": None,
                    "max_second_order_abs_residual": None,
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
            if second_order_abs is not None and second_order_rel is not None:
                if second_order is not None:
                    bucket["current_second_order_rtol"] = float(second_order["rtol"])
                    bucket["current_second_order_atol"] = float(second_order["atol"])
                existing_rel = bucket["max_second_order_rel_residual"]
                existing_abs = bucket["max_second_order_abs_residual"]
                bucket["max_second_order_rel_residual"] = second_order_rel if existing_rel is None else max(existing_rel, second_order_rel)
                bucket["max_second_order_abs_residual"] = second_order_abs if existing_abs is None else max(existing_abs, second_order_abs)

    audits: list[FamilyAudit] = []
    for (op, family, dtype), bucket in sorted(aggregates.items()):
        max_rel = float(bucket["max_rel_residual"])
        max_abs = float(bucket["max_abs_residual"])
        current_rtol = float(bucket["current_rtol"])
        current_atol = float(bucket["current_atol"])
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
        current_second_order_rtol = bucket["current_second_order_rtol"]
        current_second_order_atol = bucket["current_second_order_atol"]
        max_second_order_rel = bucket["max_second_order_rel_residual"]
        max_second_order_abs = bucket["max_second_order_abs_residual"]
        proposed_second_order_rtol = None
        proposed_second_order_atol = None
        tighten_second_order_rtol = False
        tighten_second_order_atol = False
        if (
            current_second_order_rtol is not None
            and current_second_order_atol is not None
            and max_second_order_rel is not None
            and max_second_order_abs is not None
        ):
            proposed_second_order_rtol = propose_tolerance(
                observed_max=float(max_second_order_rel),
                safety_factor=SAFETY_FACTOR,
                floor=RELATIVE_FLOOR,
            )
            proposed_second_order_atol = propose_tolerance(
                observed_max=float(max_second_order_abs),
                safety_factor=SAFETY_FACTOR,
                floor=ABSOLUTE_FLOOR,
            )
            tighten_second_order_rtol = needs_tightening(
                current=float(current_second_order_rtol),
                observed_max=float(max_second_order_rel),
                looseness_orders=LOOSENESS_ORDERS,
            )
            tighten_second_order_atol = needs_tightening(
                current=float(current_second_order_atol),
                observed_max=float(max_second_order_abs),
                looseness_orders=LOOSENESS_ORDERS,
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
                current_second_order_rtol=(
                    None if current_second_order_rtol is None else float(current_second_order_rtol)
                ),
                current_second_order_atol=(
                    None if current_second_order_atol is None else float(current_second_order_atol)
                ),
                max_second_order_rel_residual=(
                    None if max_second_order_rel is None else float(max_second_order_rel)
                ),
                max_second_order_abs_residual=(
                    None if max_second_order_abs is None else float(max_second_order_abs)
                ),
                proposed_second_order_rtol=proposed_second_order_rtol,
                proposed_second_order_atol=proposed_second_order_atol,
                tighten_second_order_rtol=tighten_second_order_rtol,
                tighten_second_order_atol=tighten_second_order_atol,
            )
        )
    return audits
