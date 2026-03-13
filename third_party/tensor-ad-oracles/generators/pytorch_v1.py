"""PyTorch-backed v1 case materialization entrypoint."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from . import encoding
from .fd import FD_POLICY_VERSION, compute_step
from .probes import make_probe_record, normalize_tensor_map
from .runtime import (
    PINNED_TORCH_VERSION,
    apply_spec_observable,
    build_call_metadata,
    build_direction_map,
    build_input_map,
    build_observable_function,
    build_scalarized_observable_function,
    combined_input_norm,
    compute_fd_hvp,
    compute_pytorch_hvp,
    dtype_name,
    import_generation_runtime,
    import_scalar_generation_runtime,
    lookup_upstream_opinfo,
    map_allclose,
    normalize_torch_version,
    normalize_raw_tensor_map,
    randn_like,
    sample_inputs_for_spec,
    tensor_map_isfinite,
    tensor_map_inner_product,
    tensor_map_to_tuple,
    tuple_to_tensor_map,
    zeros_like_input_map,
)
from .tolerance_audit import (
    comparison_from_observed_residuals,
    hvp_residuals,
    max_abs_diff,
    max_rel_diff,
    scalar_residual,
)
from .upstream_scalar_inventory import collect_ad_relevant_scalar_opinfos
from .upstream_inventory import collect_ad_relevant_linalg_opinfos


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO_ROOT / "schema" / "case.schema.json"


@dataclass(frozen=True)
class CaseFamilySpec:
    """Static metadata for a PyTorch-aligned v1 case family."""

    op: str
    family: str
    observable_kind: str
    expected_behavior: str
    source_file: str
    source_function: str
    gradcheck_wrapper: str | None = None
    upstream_name: str | None = None
    upstream_variant_name: str = ""
    sample_process_name: str | None = None
    hvp_enabled: bool = False
    inventory_kind: str = "linalg"
    supported_dtype_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class UpstreamMappedFamily:
    """DB family target associated with one upstream OpInfo variant."""

    op: str
    family: str
    hvp_enabled: bool
    supported_dtype_names: tuple[str, ...]


@dataclass
class SuccessProbePayload:
    """Raw probe payloads and residual measurements for one success-case sample."""

    case_seed: int
    dtype_name: str
    op_args: list[object]
    op_kwargs: dict[str, object]
    inputs: dict[str, object]
    direction: dict[str, object]
    cotangent: dict[str, object]
    pytorch_jvp: dict[str, object]
    pytorch_vjp: dict[str, object]
    pytorch_hvp: dict[str, object] | None
    fd_step: float
    fd_jvp: dict[str, object]
    fd_hvp: dict[str, object] | None
    first_order_max_rel_residual: float
    first_order_max_abs_residual: float
    second_order_max_rel_residual: float
    second_order_max_abs_residual: float


ERROR_REASON_CODES = {
    ("svd", "gauge_ill_defined"): "gauge_ill_defined",
    ("eigh", "gauge_ill_defined"): "gauge_ill_defined",
}

SVD_FAMILY_BY_PROCESS_FN = {
    "fn_U": "u_abs",
    "fn_S": "s",
    "fn_Vh": "vh_abs",
    "fn_UVh": "uvh_product",
}

EIGH_PROCESS_FAMILY = "values_vectors_abs"
EIG_PROCESS_FAMILY = "values_vectors_abs"
SCALAR_PROBE_DTYPE_NAMES = ("float64", "complex128", "float32", "complex64")
SCALAR_SECOND_ORDER_UNSTABLE_KEYS = {
    ("conj", ""),
}

UNSUPPORTED_UPSTREAM_KEYS = {
    ("linalg.norm", "subgradients_at_zero"),
}
UNSUPPORTED_SCALAR_UPSTREAM_KEYS = {
    ("div", "floor_rounding"),
    ("div", "trunc_rounding"),
    ("fmod", ""),
    ("remainder", ""),
}


def _normalized_upstream_op_id(name: str, variant_name: str) -> str:
    base = name.removeprefix("linalg.").replace(".", "_")
    if variant_name:
        return f"{base}_{variant_name}"
    return base


def _normalized_scalar_upstream_op_id(name: str, variant_name: str) -> str:
    base = name.replace(".", "_")
    if variant_name:
        return f"{base}_{variant_name}"
    return base


def build_supported_upstream_mapping_index() -> dict[tuple[str, str], tuple[UpstreamMappedFamily, ...]]:
    """Map upstream AD-relevant OpInfo variants to planned DB families."""
    inventory = {
        (row.name, row.variant_name): row
        for row in collect_ad_relevant_linalg_opinfos()
    }
    mapping: dict[tuple[str, str], tuple[UpstreamMappedFamily, ...]] = {}
    for row in inventory.values():
        key = (row.name, row.variant_name)
        if key in UNSUPPORTED_UPSTREAM_KEYS:
            continue
        if row.name == "linalg.svd":
            families = tuple(
                UpstreamMappedFamily(
                    op="svd",
                    family=family,
                    hvp_enabled=row.supports_fwgrad_bwgrad,
                    supported_dtype_names=row.supported_dtype_names,
                )
                for family in ("u_abs", "s", "vh_abs", "uvh_product")
            )
        elif row.name == "linalg.eigh":
            families = (
                UpstreamMappedFamily(
                    op="eigh",
                    family=EIGH_PROCESS_FAMILY,
                    hvp_enabled=row.supports_fwgrad_bwgrad,
                    supported_dtype_names=row.supported_dtype_names,
                ),
            )
        elif row.name == "linalg.eig":
            families = (
                UpstreamMappedFamily(
                    op="eig",
                    family=EIG_PROCESS_FAMILY,
                    hvp_enabled=row.supports_fwgrad_bwgrad,
                    supported_dtype_names=row.supported_dtype_names,
                ),
            )
        else:
            families = (
                UpstreamMappedFamily(
                    op=_normalized_upstream_op_id(row.name, row.variant_name),
                    family="identity",
                    hvp_enabled=row.supports_fwgrad_bwgrad,
                    supported_dtype_names=row.supported_dtype_names,
                ),
            )
        mapping[key] = families
    return mapping


def build_unsupported_upstream_mapping_index() -> dict[tuple[str, str], str]:
    """Return explicitly classified upstream AD variants that are not DB success/error families yet."""
    return {
        key: "unsupported_or_xfail_family"
        for key in UNSUPPORTED_UPSTREAM_KEYS
    }


def build_supported_scalar_mapping_index() -> dict[tuple[str, str], tuple[UpstreamMappedFamily, ...]]:
    """Map dense scalar upstream OpInfo variants to planned DB families."""
    mapping: dict[tuple[str, str], tuple[UpstreamMappedFamily, ...]] = {}
    for row in collect_ad_relevant_scalar_opinfos():
        if (row.name, row.variant_name) in UNSUPPORTED_SCALAR_UPSTREAM_KEYS:
            continue
        mapping[(row.name, row.variant_name)] = (
            UpstreamMappedFamily(
                op=_normalized_scalar_upstream_op_id(row.name, row.variant_name),
                family="identity",
                hvp_enabled=row.supports_fwgrad_bwgrad,
                supported_dtype_names=row.supported_dtype_names,
            ),
        )
    return mapping


def build_unsupported_scalar_mapping_index() -> dict[tuple[str, str], str]:
    """Return explicitly classified scalar OpInfo variants outside the DB contract."""
    return {
        key: "fd_hostile_piecewise_family"
        for key in UNSUPPORTED_SCALAR_UPSTREAM_KEYS
    }


def _observable_kind_for_target(
    key: tuple[str, str], target: UpstreamMappedFamily
) -> str:
    if key[0] == "linalg.svd":
        return {
            "u_abs": "svd_u_abs",
            "s": "svd_s",
            "vh_abs": "svd_vh_abs",
            "uvh_product": "svd_uvh_product",
        }[target.family]
    if key[0] == "linalg.eigh":
        return "eigh_values_vectors_abs"
    if key[0] == "linalg.eig":
        return "eig_values_vectors_abs"
    return "identity"


def _sample_process_name_for_target(
    key: tuple[str, str], target: UpstreamMappedFamily
) -> str | None:
    if key[0] == "linalg.svd":
        return {
            "u_abs": "fn_U",
            "s": "fn_S",
            "vh_abs": "fn_Vh",
            "uvh_product": "fn_UVh",
        }[target.family]
    if key[0] in {"linalg.eigh", "linalg.eig"}:
        return "out_fn"
    return None


def _build_success_case_specs() -> tuple[CaseFamilySpec, ...]:
    inventory_rows = {
        (row.name, row.variant_name): row
        for row in collect_ad_relevant_linalg_opinfos()
    }
    specs: list[CaseFamilySpec] = []
    for key, targets in build_supported_upstream_mapping_index().items():
        row = inventory_rows[key]
        for target in targets:
            specs.append(
                CaseFamilySpec(
                    op=target.op,
                    family=target.family,
                    observable_kind=_observable_kind_for_target(key, target),
                    expected_behavior="success",
                    source_file="torch/testing/_internal/opinfo/definitions/linalg.py",
                    source_function=row.sample_inputs_func_name,
                    gradcheck_wrapper=row.gradcheck_wrapper_name,
                    upstream_name=row.name,
                    upstream_variant_name=row.variant_name,
                    sample_process_name=_sample_process_name_for_target(key, target),
                    hvp_enabled=target.hvp_enabled,
                    supported_dtype_names=target.supported_dtype_names,
                )
            )
    return tuple(specs)


def _build_error_case_specs() -> tuple[CaseFamilySpec, ...]:
    return (
        CaseFamilySpec(
            op="svd",
            family="gauge_ill_defined",
            observable_kind="svd_uvh_product",
            expected_behavior="error",
            source_file="test/test_linalg.py",
            source_function="test_invariance_error_spectral_decompositions",
            upstream_name="linalg.svd",
            supported_dtype_names=("complex128",),
        ),
        CaseFamilySpec(
            op="eigh",
            family="gauge_ill_defined",
            observable_kind="eigh_values_vectors_abs",
            expected_behavior="error",
            source_file="test/test_linalg.py",
            source_function="test_invariance_error_spectral_decompositions",
            gradcheck_wrapper="gradcheck_wrapper_hermitian_input",
            upstream_name="linalg.eigh",
            supported_dtype_names=("complex128",),
        ),
    )


def _build_scalar_case_specs() -> tuple[CaseFamilySpec, ...]:
    inventory_rows = {
        (row.name, row.variant_name): row
        for row in collect_ad_relevant_scalar_opinfos()
    }
    specs: list[CaseFamilySpec] = []
    for key, targets in build_supported_scalar_mapping_index().items():
        row = inventory_rows[key]
        for target in targets:
            specs.append(
                CaseFamilySpec(
                    op=target.op,
                    family=target.family,
                    observable_kind="identity",
                    expected_behavior="success",
                    source_file="torch/testing/_internal/common_methods_invocations.py",
                    source_function=row.sample_inputs_func_name,
                    gradcheck_wrapper=row.gradcheck_wrapper_name,
                    upstream_name=row.name,
                    upstream_variant_name=row.variant_name,
                    hvp_enabled=target.hvp_enabled,
                    inventory_kind="scalar",
                    supported_dtype_names=target.supported_dtype_names,
                )
            )
    return tuple(specs)


def _build_case_specs() -> tuple[CaseFamilySpec, ...]:
    return _build_success_case_specs() + _build_error_case_specs() + _build_scalar_case_specs()

@lru_cache(maxsize=1)
def _case_specs_cached() -> tuple[CaseFamilySpec, ...]:
    return _build_case_specs()


@lru_cache(maxsize=1)
def _case_families_cached() -> dict[str, tuple[str, ...]]:
    specs = _case_specs_cached()
    target_ops = tuple(dict.fromkeys(spec.op for spec in specs))
    return {
        op: tuple(spec.family for spec in specs if spec.op == op)
        for op in target_ops
    }


def build_case_families() -> dict[str, tuple[str, ...]]:
    """Return the fixed PyTorch-aligned v1 op/family registry."""
    return _case_families_cached().copy()


def build_case_spec_index() -> dict[tuple[str, str], CaseFamilySpec]:
    """Index the fixed v1 case specifications by `(op, family)`."""
    return {(spec.op, spec.family): spec for spec in _case_specs_cached()}


def build_scalar_case_spec_index() -> dict[tuple[str, str], CaseFamilySpec]:
    """Index the planned scalar case specifications by `(op, family)`."""
    return {(spec.op, spec.family): spec for spec in _build_scalar_case_specs()}


def case_output_path(spec: CaseFamilySpec, *, cases_root: Path | None = None) -> Path:
    """Return the canonical JSONL path for a case family."""
    root = cases_root if cases_root is not None else REPO_ROOT / "cases"
    return root / spec.op / f"{spec.family}.jsonl"


def write_case_records(
    spec: CaseFamilySpec,
    records: Iterable[dict],
    *,
    cases_root: Path | None = None,
) -> Path:
    """Write JSONL records for one case family and return the written path."""
    out_path = case_output_path(spec, cases_root=cases_root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True))
            handle.write("\n")
    return out_path


def build_provenance(
    spec: CaseFamilySpec,
    *,
    source_commit: str,
    seed: int,
    torch_version: str,
    generator: str = "python-pytorch-v1",
    comment: str | None = None,
) -> dict:
    """Build the common provenance block for a materialized case record."""
    provenance = {
        "source_repo": "pytorch",
        "source_file": spec.source_file,
        "source_function": spec.source_function,
        "source_commit": source_commit,
        "generator": generator,
        "seed": seed,
        "torch_version": torch_version,
        "fd_policy_version": FD_POLICY_VERSION,
    }
    if comment is not None:
        provenance["comment"] = comment
    return provenance


def make_success_case(
    spec: CaseFamilySpec,
    *,
    case_id: str,
    dtype: str,
    inputs: dict,
    op_args: list[object] | None = None,
    op_kwargs: dict | None = None,
    comparison: dict,
    probes: list[dict],
    provenance: dict,
) -> dict:
    """Assemble one success-case record from materialized payloads."""
    case = {
        "schema_version": 1,
        "case_id": case_id,
        "op": spec.op,
        "dtype": dtype,
        "family": spec.family,
        "expected_behavior": "success",
        "inputs": inputs,
        "observable": {"kind": spec.observable_kind},
        "comparison": comparison,
        "probes": probes,
        "provenance": provenance,
    }
    if op_args:
        case["op_args"] = op_args
    if op_kwargs:
        case["op_kwargs"] = op_kwargs
    return case


def make_error_case(
    spec: CaseFamilySpec,
    *,
    case_id: str,
    dtype: str,
    inputs: dict,
    reason_code: str,
    provenance: dict,
) -> dict:
    """Assemble one expected-error case record."""
    return {
        "schema_version": 1,
        "case_id": case_id,
        "op": spec.op,
        "dtype": dtype,
        "family": spec.family,
        "expected_behavior": "error",
        "inputs": inputs,
        "observable": {"kind": spec.observable_kind},
        "comparison": {
            "kind": "expect_error",
            "reason_code": reason_code,
        },
        "probes": [],
        "provenance": provenance,
    }


def materialize_success_case(
    spec: CaseFamilySpec,
    *,
    case_id: str,
    dtype: str,
    raw_inputs: dict[str, object],
    op_args: list[object] | None = None,
    op_kwargs: dict[str, object] | None = None,
    comparison: dict,
    probe_id: str,
    raw_direction: dict[str, object],
    raw_cotangent: dict[str, object],
    raw_pytorch_jvp: dict[str, object],
    raw_pytorch_vjp: dict[str, object],
    raw_pytorch_hvp: dict[str, object] | None = None,
    fd_step: float,
    raw_fd_jvp: dict[str, object],
    raw_fd_hvp: dict[str, object] | None = None,
    provenance: dict,
) -> dict:
    """Encode raw tensor payloads into one success-case record."""
    probe = make_probe_record(
        probe_id=probe_id,
        direction=normalize_tensor_map(encoding.encode_tensor_map(raw_direction)),
        cotangent=normalize_tensor_map(encoding.encode_tensor_map(raw_cotangent)),
        pytorch_jvp=encoding.encode_tensor_map(raw_pytorch_jvp),
        pytorch_vjp=encoding.encode_tensor_map(raw_pytorch_vjp),
        pytorch_hvp=(
            None
            if raw_pytorch_hvp is None
            else encoding.encode_tensor_map(raw_pytorch_hvp)
        ),
        fd_step=fd_step,
        fd_jvp=encoding.encode_tensor_map(raw_fd_jvp),
        fd_hvp=(
            None
            if raw_fd_hvp is None
            else encoding.encode_tensor_map(raw_fd_hvp)
        ),
    )
    return make_success_case(
        spec,
        case_id=case_id,
        dtype=dtype,
        inputs=encoding.encode_tensor_map(raw_inputs),
        op_args=op_args,
        op_kwargs=op_kwargs,
        comparison=comparison,
        probes=[probe],
        provenance=provenance,
    )


def _case_id(spec: CaseFamilySpec, *, dtype: str, index: int) -> str:
    dtype_tag = {
        "float32": "f32",
        "float64": "f64",
        "complex64": "c64",
        "complex128": "c128",
    }[dtype]
    return f"{spec.op}_{dtype_tag}_{spec.family}_{index:03d}"


def _sample_matches_family(spec: CaseFamilySpec, sample) -> bool:
    if spec.sample_process_name is None:
        return True
    return sample.output_process_fn_grad.__name__ == spec.sample_process_name


def _materialize_hvp_for_spec(spec: CaseFamilySpec, *, dtype_name: str) -> bool:
    if not (spec.hvp_enabled and dtype_name in {"float64", "complex128"}):
        return False
    if (
        spec.inventory_kind == "scalar"
        and (spec.upstream_name, spec.upstream_variant_name) in SCALAR_SECOND_ORDER_UNSTABLE_KEYS
    ):
        return False
    return True


def _is_skippable_hvp_runtime_error(exc: RuntimeError) -> bool:
    message = str(exc)
    return (
        "forward differentiable view operations" in message
        or "has_fw_view" in message
    )


def _resolve_runtime_for_spec(spec: CaseFamilySpec):
    if spec.inventory_kind == "scalar":
        return import_scalar_generation_runtime()
    return import_generation_runtime()


def _supported_dtype_names_for_spec(torch, opinfo, spec: CaseFamilySpec) -> tuple[str, ...]:
    if spec.inventory_kind == "scalar":
        supported = opinfo.supported_dtypes("cpu")
        return tuple(
            dtype_name
            for dtype_name in SCALAR_PROBE_DTYPE_NAMES
            if getattr(torch, dtype_name) in supported
        )
    supported = opinfo.supported_dtypes("cpu")
    return tuple(
        dtype_name
        for dtype_name in spec.supported_dtype_names
        if getattr(torch, dtype_name) in supported
    )


def _success_provenance_comment(spec: CaseFamilySpec, *, dtype_name: str) -> str | None:
    if spec.op == "svd" and dtype_name in {"complex64", "complex128"}:
        return "from PyTorch OpInfo complex SVD success coverage"
    return None


def _validate_success_probe(
    torch,
    *,
    comparison: dict,
    direction: dict[str, object],
    cotangent: dict[str, object],
    pytorch_jvp: dict[str, object],
    pytorch_vjp: dict[str, object],
    fd_jvp: dict[str, object],
    pytorch_hvp: dict[str, object] | None = None,
    fd_hvp: dict[str, object] | None = None,
) -> None:
    first_order = comparison["first_order"]
    if not map_allclose(
        torch,
        pytorch_jvp,
        fd_jvp,
        rtol=first_order["rtol"],
        atol=first_order["atol"],
    ):
        raise ValueError("PyTorch JVP and FD-JVP disagree outside tolerance")

    lhs = tensor_map_inner_product(torch, cotangent, fd_jvp)
    rhs = tensor_map_inner_product(torch, pytorch_vjp, direction)
    residual = (lhs - rhs).abs()
    if not torch.allclose(
        residual,
        torch.zeros_like(residual),
        rtol=first_order["rtol"],
        atol=first_order["atol"],
        ):
        raise ValueError("probe failed adjoint consistency")

    if pytorch_hvp is None and fd_hvp is None:
        return
    if pytorch_hvp is None or fd_hvp is None:
        raise ValueError("HVP references must be present together")
    second_order = comparison["second_order"]
    if not map_allclose(
        torch,
        pytorch_hvp,
        fd_hvp,
        rtol=second_order["rtol"],
        atol=second_order["atol"],
    ):
        raise ValueError("PyTorch HVP and FD-HVP disagree outside tolerance")


def _measured_comparison(
    payloads: list[SuccessProbePayload],
) -> dict[str, float | str]:
    comparison = {
        "first_order": comparison_from_observed_residuals(
            max_rel_residual=max(
                payload.first_order_max_rel_residual for payload in payloads
            ),
            max_abs_residual=max(
                payload.first_order_max_abs_residual for payload in payloads
            ),
        )
    }
    if any(payload.pytorch_hvp is not None for payload in payloads):
        comparison["second_order"] = comparison_from_observed_residuals(
            max_rel_residual=max(
                payload.second_order_max_rel_residual for payload in payloads
            ),
            max_abs_residual=max(
                payload.second_order_max_abs_residual for payload in payloads
            ),
        )
    return comparison


def _generate_success_records(
    spec: CaseFamilySpec,
    *,
    limit: int | None = None,
    seed: int = 17,
) -> list[dict]:
    ensure_runtime_dependencies()
    torch, opinfo_source = _resolve_runtime_for_spec(spec)
    opinfo = lookup_upstream_opinfo(opinfo_source, spec)
    source_commit = getattr(torch.version, "git_version", None) or torch.__version__
    records: list[dict] = []
    for current_dtype_name in _supported_dtype_names_for_spec(torch, opinfo, spec):
        payloads: list[SuccessProbePayload] = []
        current_dtype = getattr(torch, current_dtype_name)
        materialize_hvp = _materialize_hvp_for_spec(
            spec,
            dtype_name=current_dtype_name,
        )
        samples = sample_inputs_for_spec(
            torch,
            opinfo_source,
            spec,
            seed=seed,
            dtype=current_dtype,
        )

        for sample in samples:
            if limit is not None and len(payloads) >= limit:
                break
            if not _sample_matches_family(spec, sample):
                continue

            op_args, op_kwargs = build_call_metadata(torch, sample, spec=spec)
            inputs = build_input_map(torch, spec, sample)
            output = apply_spec_observable(
                torch,
                spec,
                sample,
                inputs,
                linalg=opinfo_source,
                opinfo=opinfo,
            )
            if not output:
                continue
            if not tensor_map_isfinite(output):
                continue

            case_seed = seed + len(payloads)
            generator = torch.Generator(device="cpu")
            generator.manual_seed(case_seed)

            direction = build_direction_map(torch, spec, inputs, generator=generator)
            cotangent = normalize_raw_tensor_map(
                torch,
                {
                    name: randn_like(torch, tensor, generator=generator)
                    for name, tensor in output.items()
                },
            )

            input_names = tuple(inputs.keys())
            output_names = tuple(output.keys())
            observable_fn = build_observable_function(
                torch,
                spec,
                sample,
                input_names,
                linalg=opinfo_source,
                opinfo=opinfo,
                output_names=output_names,
            )

            try:
                _, jvp_tuple = torch.func.jvp(
                    observable_fn,
                    tensor_map_to_tuple(inputs),
                    tensor_map_to_tuple(direction),
                )
            except RuntimeError as exc:
                if "non-empty output" in str(exc):
                    continue
                raise
            pytorch_jvp = tuple_to_tensor_map(output_names, jvp_tuple)
            if not tensor_map_isfinite(pytorch_jvp):
                continue

            try:
                grads = torch.autograd.grad(
                    tensor_map_to_tuple(output),
                    tensor_map_to_tuple(inputs),
                    grad_outputs=tensor_map_to_tuple(cotangent),
                    allow_unused=True,
                )
            except RuntimeError as exc:
                if "does not require grad" in str(exc):
                    continue
                raise
            pytorch_vjp = zeros_like_input_map(torch, inputs, grads)
            if not tensor_map_isfinite(pytorch_vjp):
                continue

            first_input = next(iter(inputs.values()))
            fd_step = compute_step(
                dtype_name(torch, first_input.dtype),
                input_norm=combined_input_norm(torch, inputs),
            )
            plus_inputs = {
                name: tensor + fd_step * direction[name]
                for name, tensor in inputs.items()
            }
            minus_inputs = {
                name: tensor - fd_step * direction[name]
                for name, tensor in inputs.items()
            }
            plus_output = apply_spec_observable(
                torch,
                spec,
                sample,
                plus_inputs,
                linalg=opinfo_source,
                opinfo=opinfo,
            )
            minus_output = apply_spec_observable(
                torch,
                spec,
                sample,
                minus_inputs,
                linalg=opinfo_source,
                opinfo=opinfo,
            )
            fd_jvp = {
                name: (plus_output[name] - minus_output[name]) / (2.0 * fd_step)
                for name in output_names
            }
            if not tensor_map_isfinite(fd_jvp):
                continue
            pytorch_hvp = None
            fd_hvp = None
            second_order_abs = 0.0
            second_order_rel = 0.0
            if materialize_hvp:
                scalarized_fn = build_scalarized_observable_function(
                    torch,
                    observable_fn,
                    output_names=output_names,
                    cotangent=cotangent,
                )
                try:
                    pytorch_hvp = compute_pytorch_hvp(
                        torch,
                        scalarized_fn,
                        inputs=inputs,
                        direction=direction,
                    )
                    fd_hvp = compute_fd_hvp(
                        torch,
                        scalarized_fn,
                        inputs=inputs,
                        direction=direction,
                        step=fd_step,
                    )
                except RuntimeError as exc:
                    if _is_skippable_hvp_runtime_error(exc):
                        continue
                    raise
                if not tensor_map_isfinite(pytorch_hvp) or not tensor_map_isfinite(fd_hvp):
                    continue
                second_order_abs, second_order_rel = hvp_residuals(
                    torch,
                    pytorch_hvp,
                    fd_hvp,
                )
            jvp_abs = max_abs_diff(torch, pytorch_jvp, fd_jvp)
            jvp_rel = max_rel_diff(torch, pytorch_jvp, fd_jvp)
            lhs = tensor_map_inner_product(torch, cotangent, fd_jvp)
            rhs = tensor_map_inner_product(torch, pytorch_vjp, direction)
            adj_abs, adj_rel = scalar_residual(torch, lhs, rhs)
            payloads.append(
                SuccessProbePayload(
                    case_seed=case_seed,
                    dtype_name=current_dtype_name,
                    op_args=op_args,
                    op_kwargs=op_kwargs,
                    inputs=inputs,
                    direction=direction,
                    cotangent=cotangent,
                    pytorch_jvp=pytorch_jvp,
                    pytorch_vjp=pytorch_vjp,
                    pytorch_hvp=pytorch_hvp,
                    fd_step=fd_step,
                    fd_jvp=fd_jvp,
                    fd_hvp=fd_hvp,
                    first_order_max_rel_residual=max(jvp_rel, adj_rel),
                    first_order_max_abs_residual=max(jvp_abs, adj_abs),
                    second_order_max_rel_residual=second_order_rel,
                    second_order_max_abs_residual=second_order_abs,
                )
            )

        if not payloads:
            continue

        comparison = _measured_comparison(payloads)
        for index, payload in enumerate(payloads, start=1):
            _validate_success_probe(
                torch,
                comparison=comparison,
                direction=payload.direction,
                cotangent=payload.cotangent,
                pytorch_jvp=payload.pytorch_jvp,
                pytorch_vjp=payload.pytorch_vjp,
                fd_jvp=payload.fd_jvp,
                pytorch_hvp=payload.pytorch_hvp,
                fd_hvp=payload.fd_hvp,
            )

            provenance = build_provenance(
                spec,
                source_commit=source_commit,
                seed=payload.case_seed,
                torch_version=normalize_torch_version(torch.__version__),
                comment=_success_provenance_comment(spec, dtype_name=payload.dtype_name),
            )
            records.append(
                materialize_success_case(
                    spec,
                    case_id=_case_id(spec, dtype=payload.dtype_name, index=index),
                    dtype=payload.dtype_name,
                    raw_inputs=payload.inputs,
                    op_args=payload.op_args,
                    op_kwargs=payload.op_kwargs,
                    comparison=comparison,
                    probe_id="p0",
                    raw_direction=payload.direction,
                    raw_cotangent=payload.cotangent,
                    raw_pytorch_jvp=payload.pytorch_jvp,
                    raw_pytorch_vjp=payload.pytorch_vjp,
                    raw_pytorch_hvp=payload.pytorch_hvp,
                    fd_step=payload.fd_step,
                    raw_fd_jvp=payload.fd_jvp,
                    raw_fd_hvp=payload.fd_hvp,
                    provenance=provenance,
                )
            )

    return records


def _make_spectral_error_input(torch, spec: CaseFamilySpec, *, generator):
    a = torch.randn((3, 3), dtype=torch.complex128, device="cpu", generator=generator)
    if spec.op == "eigh":
        a = a + a.mH
    return a.requires_grad_(True)


def _validate_error_case(torch, spec: CaseFamilySpec, a) -> None:
    if spec.op == "svd":
        u, _, vh = torch.linalg.svd(a, full_matrices=False)
        loss = (u + vh).sum().abs()
    elif spec.op == "eigh":
        loss = torch.linalg.eigh(a).eigenvectors.sum().abs()
    else:
        raise ValueError(f"unsupported error-case op: {spec.op}")

    try:
        loss.backward()
    except RuntimeError as exc:
        if "ill-defined" not in str(exc):
            raise
        return
    raise ValueError("expected spectral decomposition backward to raise an ill-defined error")


def _generate_error_records(
    spec: CaseFamilySpec,
    *,
    limit: int | None = None,
    seed: int = 17,
) -> list[dict]:
    ensure_runtime_dependencies()
    torch, _ = import_generation_runtime()
    source_commit = getattr(torch.version, "git_version", None) or torch.__version__
    count = 1 if limit is None else limit
    records: list[dict] = []

    for index in range(count):
        case_seed = seed + index
        generator = torch.Generator(device="cpu")
        generator.manual_seed(case_seed)
        a = _make_spectral_error_input(torch, spec, generator=generator)
        _validate_error_case(torch, spec, a)
        provenance = build_provenance(
            spec,
            source_commit=source_commit,
            seed=case_seed,
            torch_version=normalize_torch_version(torch.__version__),
        )
        records.append(
            make_error_case(
                spec,
                case_id=_case_id(spec, dtype="complex128", index=index + 1),
                dtype="complex128",
                inputs=encoding.encode_tensor_map({"a": a}),
                reason_code=ERROR_REASON_CODES[(spec.op, spec.family)],
                provenance=provenance,
            )
        )

    return records


def generate_solve_identity_records(*, limit: int = 1, seed: int = 17) -> list[dict]:
    """Materialize `solve/identity` success cases from PyTorch samples."""
    spec = build_case_spec_index()[("solve", "identity")]
    return _generate_success_records(spec, limit=limit, seed=seed)


def materialize_case_family(
    op: str,
    family: str,
    *,
    limit: int = 1,
    cases_root: Path | None = None,
) -> Path:
    """Generate and write one supported case family."""
    spec = build_case_spec_index()[(op, family)]
    if spec.expected_behavior == "success":
        records = _generate_success_records(spec, limit=limit)
    else:
        records = _generate_error_records(spec, limit=limit)
    return write_case_records(spec, records, cases_root=cases_root)


def materialize_all_case_families(
    *,
    limit: int = 1,
    cases_root: Path | None = None,
) -> list[Path]:
    """Generate and write every fixed v1 case family."""
    paths: list[Path] = []
    for spec in _case_specs_cached():
        paths.append(
            materialize_case_family(
                spec.op,
                spec.family,
                limit=limit,
                cases_root=cases_root,
            )
        )
    return paths


def ensure_runtime_dependencies() -> None:
    """Raise a clear error when optional generation dependencies are missing."""
    missing: list[str] = []
    for module_name in ("torch", "numpy", "jsonschema", "expecttest"):
        try:
            __import__(module_name)
        except ModuleNotFoundError:
            missing.append(module_name)
        except ImportError as exc:
            raise RuntimeError(
                "PyTorch v1 case generation failed to import optional dependency: "
                f"{module_name}"
            ) from exc
    if missing:
        deps = ", ".join(missing)
        raise RuntimeError(
            "PyTorch v1 case generation requires optional dependencies: " f"{deps}"
        )

    import torch

    actual = normalize_torch_version(torch.__version__)
    if actual != PINNED_TORCH_VERSION:
        raise RuntimeError(
            f"tensor-ad-oracles requires torch=={PINNED_TORCH_VERSION}, got {torch.__version__}"
        )


def _iter_registry_lines() -> Iterable[str]:
    case_families = build_case_families()
    for op in case_families:
        families = ", ".join(case_families[op])
        yield f"{op}: {families}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print the fixed v1 op/family registry and exit.",
    )
    parser.add_argument(
        "--materialize",
        choices=tuple(build_case_families()),
        help="Materialize one supported op family into JSONL.",
    )
    parser.add_argument(
        "--materialize-all",
        action="store_true",
        help="Materialize the full supported case registry into JSONL files.",
    )
    parser.add_argument(
        "--family",
        help="Case family to materialize for the selected op.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1,
        help="Maximum number of records to materialize.",
    )
    parser.add_argument(
        "--cases-root",
        type=Path,
        default=None,
        help="Optional output root for generated cases.",
    )
    args = parser.parse_args(argv)

    if args.list:
        for line in _iter_registry_lines():
            print(line)
        return 0

    if args.materialize:
        if not args.family:
            raise SystemExit("--family is required with --materialize")
        out_path = materialize_case_family(
            args.materialize,
            args.family,
            limit=args.limit,
            cases_root=args.cases_root,
        )
        print(out_path)
        return 0

    if args.materialize_all:
        paths = materialize_all_case_families(
            limit=args.limit,
            cases_root=args.cases_root,
        )
        for path in paths:
            print(path)
        return 0

    ensure_runtime_dependencies()
    raise SystemExit("PyTorch v1 case generation is not implemented yet.")


if __name__ == "__main__":
    raise SystemExit(main())
