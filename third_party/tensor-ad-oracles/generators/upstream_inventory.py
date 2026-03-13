"""Extract AD-relevant PyTorch linalg OpInfo metadata."""

from __future__ import annotations

from dataclasses import dataclass

from .runtime import import_generation_runtime


DEFAULT_FIRST_ORDER_AD_TOLERANCE = {"rtol": 1e-3, "atol": 1e-5}
DEFAULT_SECOND_ORDER_AD_TOLERANCE = {"rtol": 1e-3, "atol": 1e-5}
PREFERRED_PUBLISH_DTYPE_NAMES = ("float64", "complex128", "float32", "complex64")
DTYPE_BY_NAME = {
    "float32": "float32",
    "float64": "float64",
    "complex64": "complex64",
    "complex128": "complex128",
}


@dataclass(frozen=True)
class UpstreamOpInfoRecord:
    """Normalized metadata for one AD-relevant upstream linalg OpInfo entry."""

    name: str
    variant_name: str
    sample_inputs_func_name: str
    gradcheck_wrapper_name: str | None
    sample_output_process_fn_names: tuple[str, ...]
    gradcheck_fast_mode: bool
    supports_forward_ad: bool
    supports_fwgrad_bwgrad: bool
    supported_dtype_names: tuple[str, ...]


def _normalized_name(obj) -> str | None:
    if obj is None:
        return None
    name = getattr(obj, "__name__", None)
    if name == "<lambda>":
        return None
    return name


def _sample_output_process_names(op, *, torch) -> tuple[str, ...]:
    names = {
        getattr(sample.output_process_fn_grad, "__name__", "<unknown>")
        for sample in op.sample_inputs("cpu", torch.float64, requires_grad=True)
    }
    return tuple(sorted(names))


def _supported_dtype_names(torch, op) -> tuple[str, ...]:
    supported = op.supported_dtypes("cpu")
    return tuple(
        dtype_name
        for dtype_name in PREFERRED_PUBLISH_DTYPE_NAMES
        if getattr(torch, DTYPE_BY_NAME[dtype_name]) in supported
    )


def collect_ad_relevant_linalg_opinfos() -> list[UpstreamOpInfoRecord]:
    """Return all upstream linalg OpInfo entries that participate in AD tests."""

    torch, linalg = import_generation_runtime()
    rows: list[UpstreamOpInfoRecord] = []
    for op in linalg.op_db:
        if not op.name.startswith("linalg."):
            continue
        if not (op.supports_forward_ad or op.supports_fwgrad_bwgrad):
            continue
        rows.append(
            UpstreamOpInfoRecord(
                name=op.name,
                variant_name=getattr(op, "variant_test_name", "") or "",
                sample_inputs_func_name=getattr(
                    op.sample_inputs_func, "__name__", type(op.sample_inputs_func).__name__
                ),
                gradcheck_wrapper_name=_normalized_name(
                    getattr(op, "gradcheck_wrapper", None)
                ),
                sample_output_process_fn_names=_sample_output_process_names(
                    op, torch=torch
                ),
                gradcheck_fast_mode=bool(getattr(op, "gradcheck_fast_mode", False)),
                supports_forward_ad=bool(getattr(op, "supports_forward_ad", False)),
                supports_fwgrad_bwgrad=bool(getattr(op, "supports_fwgrad_bwgrad", False)),
                supported_dtype_names=_supported_dtype_names(torch, op),
            )
        )
    return rows


def resolve_upstream_ad_tolerance(
    name: str,
    variant_name: str,
    *,
    order: str,
    dtype_name: str,
) -> dict[str, float]:
    """Resolve the effective upstream AD tolerance for one pinned linalg OpInfo."""
    torch, linalg = import_generation_runtime()
    dtype = {
        "float64": torch.double,
        "complex128": torch.cdouble,
    }[dtype_name]
    cls_name, test_name, default = {
        "first_order": (
            "TestFwdGradients",
            "test_forward_mode_AD",
            DEFAULT_FIRST_ORDER_AD_TOLERANCE,
        ),
        "second_order": (
            "TestFwdGradients",
            "test_fn_fwgrad_bwgrad",
            DEFAULT_SECOND_ORDER_AD_TOLERANCE,
        ),
    }[order]

    op = next(
        candidate
        for candidate in linalg.op_db
        if candidate.name == name
        and ((getattr(candidate, "variant_test_name", "") or "") == variant_name)
    )
    decorators = op.get_decorators(
        cls_name,
        test_name,
        "cpu",
        dtype,
        {"dtype": dtype},
    )
    tolerance = dict(default)
    for decorator in decorators:
        if decorator.__class__.__name__ != "toleranceOverride":
            continue
        override = getattr(decorator, "d", {}).get(dtype)
        if override is None:
            continue
        tolerance = {
            "rtol": float(override.rtol),
            "atol": float(override.atol),
        }
    return tolerance
