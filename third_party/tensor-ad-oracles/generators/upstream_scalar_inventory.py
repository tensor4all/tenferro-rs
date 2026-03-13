"""Extract AD-relevant dense scalar PyTorch OpInfo metadata."""

from __future__ import annotations

from dataclasses import dataclass

from .runtime import ensure_pinned_torch_version
from .upstream_inventory import (
    DEFAULT_FIRST_ORDER_AD_TOLERANCE,
    DEFAULT_SECOND_ORDER_AD_TOLERANCE,
    _supported_dtype_names,
)


PREFERRED_PROBE_DTYPES = ("float64", "complex128", "float32", "complex64")
DTYPE_BY_NAME = {
    "float32": "float32",
    "float64": "float64",
    "complex64": "complex64",
    "complex128": "complex128",
}
FAMILY_CLASS_BY_OPINFO = {
    "UnaryUfuncInfo": "unary",
    "BinaryUfuncInfo": "binary",
    "ReductionOpInfo": "reduction",
}
EXCLUDED_NAME_PREFIXES = ("linalg.", "masked.")


@dataclass(frozen=True)
class UpstreamScalarOpInfoRecord:
    """Normalized metadata for one AD-relevant dense scalar upstream OpInfo."""

    name: str
    variant_name: str
    opinfo_class_name: str
    family_class: str
    sample_inputs_func_name: str
    gradcheck_wrapper_name: str | None
    sample_output_process_fn_names: tuple[str, ...]
    gradcheck_fast_mode: bool
    supports_forward_ad: bool
    supports_fwgrad_bwgrad: bool
    supported_dtype_names: tuple[str, ...]


def _import_scalar_generation_runtime():
    import torch
    from torch.testing._internal import common_methods_invocations as cmi

    ensure_pinned_torch_version(torch)
    return torch, cmi


def _normalized_name(obj) -> str | None:
    if obj is None:
        return None
    name = getattr(obj, "__name__", None)
    if name == "<lambda>":
        return None
    return name


def _probe_dtype(torch, op) -> object | None:
    supported = op.supported_dtypes("cpu")
    for dtype_name in PREFERRED_PROBE_DTYPES:
        dtype = getattr(torch, DTYPE_BY_NAME[dtype_name])
        if dtype in supported:
            return dtype
    return None


def _sample_output_process_names(op, *, torch) -> tuple[str, ...]:
    dtype = _probe_dtype(torch, op)
    if dtype is None:
        return ()
    names = {
        getattr(sample.output_process_fn_grad, "__name__", "<unknown>")
        for sample in op.sample_inputs("cpu", dtype, requires_grad=True)
    }
    return tuple(sorted(names))


def _is_supported_scalar_opinfo(op) -> bool:
    if type(op).__name__ not in FAMILY_CLASS_BY_OPINFO:
        return False
    if op.name.startswith(EXCLUDED_NAME_PREFIXES):
        return False
    if not (op.supports_forward_ad or op.supports_fwgrad_bwgrad):
        return False
    return True


def collect_ad_relevant_scalar_opinfos() -> list[UpstreamScalarOpInfoRecord]:
    """Return all dense scalar/generic upstream OpInfo entries that participate in AD tests."""

    torch, cmi = _import_scalar_generation_runtime()
    rows: list[UpstreamScalarOpInfoRecord] = []
    for op in cmi.op_db:
        if not _is_supported_scalar_opinfo(op):
            continue
        if _probe_dtype(torch, op) is None:
            continue
        rows.append(
            UpstreamScalarOpInfoRecord(
                name=op.name,
                variant_name=getattr(op, "variant_test_name", "") or "",
                opinfo_class_name=type(op).__name__,
                family_class=FAMILY_CLASS_BY_OPINFO[type(op).__name__],
                sample_inputs_func_name=getattr(
                    op.sample_inputs_func, "__name__", type(op.sample_inputs_func).__name__
                ),
                gradcheck_wrapper_name=_normalized_name(
                    getattr(op, "gradcheck_wrapper", None)
                ),
                sample_output_process_fn_names=_sample_output_process_names(op, torch=torch),
                gradcheck_fast_mode=bool(getattr(op, "gradcheck_fast_mode", False)),
                supports_forward_ad=bool(getattr(op, "supports_forward_ad", False)),
                supports_fwgrad_bwgrad=bool(getattr(op, "supports_fwgrad_bwgrad", False)),
                supported_dtype_names=_supported_dtype_names(torch, op),
            )
        )
    return rows


def resolve_upstream_scalar_ad_tolerance(
    name: str,
    variant_name: str,
    *,
    order: str,
    dtype_name: str,
) -> dict[str, float]:
    """Resolve the effective upstream AD tolerance for one pinned dense scalar OpInfo."""

    torch, cmi = _import_scalar_generation_runtime()
    dtype = getattr(torch, DTYPE_BY_NAME[dtype_name])
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
        for candidate in cmi.op_db
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
