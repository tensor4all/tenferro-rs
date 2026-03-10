"""Audit published torch-vs-FD residuals against upstream PyTorch AD tolerances."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CASES_ROOT = REPO_ROOT / "cases"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from generators.pytorch_v1 import build_case_spec_index
from generators.tolerance_audit import hvp_residuals, max_abs_diff, max_rel_diff
from generators.upstream_inventory import resolve_upstream_ad_tolerance
from generators.runtime import map_allclose
from validators.case_loader import iter_case_files, load_case_file
from validators.encoding import decode_tensor_map


@dataclass(frozen=True)
class UpstreamAdToleranceAudit:
    case_key: str
    order: str
    observed_rtol: float
    observed_atol: float
    upstream_rtol: float
    upstream_atol: float
    violates_upstream: bool = False


def audit_against_upstream_ad_tolerances(
    cases_root: Path = CASES_ROOT,
) -> list[UpstreamAdToleranceAudit]:
    """Aggregate direct torch-vs-FD residuals and compare them to upstream AD tolerances."""
    audits: list[UpstreamAdToleranceAudit] = []
    spec_index = build_case_spec_index()
    first_order: dict[tuple[str, str, str], dict[str, float]] = {}
    second_order: dict[tuple[str, str, str], dict[str, float]] = {}

    import torch

    for path in iter_case_files(cases_root):
        for record in load_case_file(path):
            if record["expected_behavior"] != "success":
                continue
            key = (record["op"], record["family"], record["dtype"])
            probe = record["probes"][0]
            pytorch_jvp = decode_tensor_map(probe["pytorch_ref"]["jvp"])
            fd_jvp = decode_tensor_map(probe["fd_ref"]["jvp"])
            bucket = first_order.setdefault(key, {"rel": 0.0, "abs": 0.0})
            bucket["rel"] = max(bucket["rel"], max_rel_diff(torch, pytorch_jvp, fd_jvp))
            bucket["abs"] = max(bucket["abs"], max_abs_diff(torch, pytorch_jvp, fd_jvp))

            if "hvp" in probe["pytorch_ref"] and "hvp" in probe["fd_ref"]:
                pytorch_hvp = decode_tensor_map(probe["pytorch_ref"]["hvp"])
                fd_hvp = decode_tensor_map(probe["fd_ref"]["hvp"])
                abs_residual, rel_residual = hvp_residuals(torch, pytorch_hvp, fd_hvp)
                second_bucket = second_order.setdefault(key, {"rel": 0.0, "abs": 0.0})
                second_bucket["rel"] = max(second_bucket["rel"], rel_residual)
                second_bucket["abs"] = max(second_bucket["abs"], abs_residual)

    for key, residuals in sorted(first_order.items()):
        op, family, dtype = key
        spec = spec_index[(op, family)]
        upstream = resolve_upstream_ad_tolerance(
            spec.upstream_name,
            spec.upstream_variant_name,
            order="first_order",
            dtype_name=dtype,
        )
        violates_upstream = False
        path = cases_root / op / f"{family}.jsonl"
        for record in load_case_file(path):
            if record["expected_behavior"] != "success":
                continue
            probe = record["probes"][0]
            pytorch_jvp = decode_tensor_map(probe["pytorch_ref"]["jvp"])
            fd_jvp = decode_tensor_map(probe["fd_ref"]["jvp"])
            if not map_allclose(
                torch,
                pytorch_jvp,
                fd_jvp,
                rtol=upstream["rtol"],
                atol=upstream["atol"],
            ):
                violates_upstream = True
                break
        audits.append(
            UpstreamAdToleranceAudit(
                case_key=f"{op}/{family}/{dtype}",
                order="first_order",
                observed_rtol=residuals["rel"],
                observed_atol=residuals["abs"],
                upstream_rtol=upstream["rtol"],
                upstream_atol=upstream["atol"],
                violates_upstream=violates_upstream,
            )
        )
    for key, residuals in sorted(second_order.items()):
        op, family, dtype = key
        spec = spec_index[(op, family)]
        upstream = resolve_upstream_ad_tolerance(
            spec.upstream_name,
            spec.upstream_variant_name,
            order="second_order",
            dtype_name=dtype,
        )
        violates_upstream = False
        path = cases_root / op / f"{family}.jsonl"
        for record in load_case_file(path):
            if record["expected_behavior"] != "success":
                continue
            probe = record["probes"][0]
            if "hvp" not in probe["pytorch_ref"] or "hvp" not in probe["fd_ref"]:
                continue
            pytorch_hvp = decode_tensor_map(probe["pytorch_ref"]["hvp"])
            fd_hvp = decode_tensor_map(probe["fd_ref"]["hvp"])
            if not map_allclose(
                torch,
                pytorch_hvp,
                fd_hvp,
                rtol=upstream["rtol"],
                atol=upstream["atol"],
            ):
                violates_upstream = True
                break
        audits.append(
            UpstreamAdToleranceAudit(
                case_key=f"{op}/{family}/{dtype}",
                order="second_order",
                observed_rtol=residuals["rel"],
                observed_atol=residuals["abs"],
                upstream_rtol=upstream["rtol"],
                upstream_atol=upstream["atol"],
                violates_upstream=violates_upstream,
            )
        )
    return audits


def main() -> int:
    audits = audit_against_upstream_ad_tolerances(CASES_ROOT)
    flagged = [
        audit
        for audit in audits
        if getattr(
            audit,
            "violates_upstream",
            audit.observed_rtol > audit.upstream_rtol
            and audit.observed_atol > audit.upstream_atol,
        )
    ]
    if flagged:
        lines = [
            f"{audit.case_key} {audit.order}: "
            f"observed (rtol={audit.observed_rtol:g}, atol={audit.observed_atol:g}) > "
            f"upstream (rtol={audit.upstream_rtol:g}, atol={audit.upstream_atol:g})"
            for audit in flagged
        ]
        raise SystemExit("upstream AD tolerance audit failed:\n" + "\n".join(lines))
    print(f"upstream_ad_tolerance_audits={len(audits)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
