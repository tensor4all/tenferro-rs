"""Extract AD-relevant PyTorch linalg OpInfo metadata."""

from __future__ import annotations

from dataclasses import dataclass

from .runtime import import_generation_runtime


@dataclass(frozen=True)
class UpstreamOpInfoRecord:
    """Normalized metadata for one AD-relevant upstream linalg OpInfo entry."""

    name: str
    variant_name: str
    sample_inputs_func_name: str
    gradcheck_wrapper_name: str | None
    sample_output_process_fn_names: tuple[str, ...]
    gradcheck_fast_mode: bool


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
            )
        )
    return rows
