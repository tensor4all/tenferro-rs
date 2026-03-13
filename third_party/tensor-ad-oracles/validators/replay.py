"""Replay published JSON cases and verify stored references are reproducible."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from generators.pytorch_v1 import build_case_spec_index
from generators.runtime import (
    apply_spec_observable,
    build_call_metadata,
    build_input_map,
    build_observable_function,
    build_scalarized_observable_function,
    compute_fd_hvp,
    compute_pytorch_hvp,
    import_generation_runtime,
    import_scalar_generation_runtime,
    lookup_upstream_opinfo,
    map_allclose,
    sample_inputs_for_spec,
    tensor_map_to_tuple,
    tensor_map_inner_product,
    tuple_to_tensor_map,
    zeros_like_input_map,
)

from .case_loader import iter_case_files, load_case_file
from .encoding import decode_tensor_map


@dataclass
class ReplayResult:
    checked: int = 0
    failures: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PreparedSample:
    sample: object
    inputs: dict[str, object]
    op_args: list[object]
    op_kwargs: dict[str, object]


SAMPLE_INPUT_SEED = 17


def _map_equal(torch, expected: dict[str, object], actual: dict[str, object]) -> bool:
    if expected.keys() != actual.keys():
        return False
    return all(torch.equal(expected[name], actual[name]) for name in expected)


def _map_close(
    torch,
    expected: dict[str, object],
    actual: dict[str, object],
    *,
    comparison: dict,
) -> bool:
    if expected.keys() != actual.keys():
        return False
    first_order = _first_order_comparison(comparison)
    rtol = max(float(first_order["rtol"]), 1e-8)
    atol = max(float(first_order["atol"]), 1e-9)
    return all(
        expected[name].shape == actual[name].shape
        and expected[name].dtype == actual[name].dtype
        and torch.allclose(
            expected[name],
            actual[name],
            rtol=rtol,
            atol=atol,
            equal_nan=True,
        )
        for name in expected
    )


def _scalar_allclose(torch, lhs, rhs, *, rtol: float, atol: float) -> bool:
    common_dtype = torch.promote_types(lhs.dtype, rhs.dtype)
    lhs_cast = lhs.to(dtype=common_dtype)
    rhs_cast = rhs.to(dtype=common_dtype)
    return torch.allclose(lhs_cast, rhs_cast, rtol=rtol, atol=atol, equal_nan=True)


def _decode_record_inputs(record: dict) -> dict[str, object]:
    return {
        name: tensor.detach().clone().requires_grad_(True)
        for name, tensor in decode_tensor_map(record["inputs"]).items()
    }


def _decode_success_probe(record: dict) -> tuple[dict[str, object], dict[str, object], dict[str, object], dict[str, object], float]:
    probe = record["probes"][0]
    return (
        decode_tensor_map(probe["direction"]),
        decode_tensor_map(probe["cotangent"]),
        decode_tensor_map(probe["pytorch_ref"]["jvp"]),
        decode_tensor_map(probe["pytorch_ref"]["vjp"]),
        (
            None
            if "hvp" not in probe["pytorch_ref"]
            else decode_tensor_map(probe["pytorch_ref"]["hvp"])
        ),
        float(probe["fd_ref"]["step"]),
        (
            None
            if "hvp" not in probe["fd_ref"]
            else decode_tensor_map(probe["fd_ref"]["hvp"])
        ),
    )


def _first_order_comparison(comparison: dict) -> dict:
    return comparison.get("first_order", comparison)


def _second_order_comparison(comparison: dict) -> dict | None:
    return comparison.get("second_order")


def validate_live_success_probe(
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
    """Validate live cross-oracle agreement for one success-case probe."""
    first_order = _first_order_comparison(comparison)
    if not map_allclose(
        torch,
        pytorch_jvp,
        fd_jvp,
        rtol=first_order["rtol"],
        atol=first_order["atol"],
    ):
        raise ValueError("live PyTorch JVP and live FD-JVP disagree")

    lhs = tensor_map_inner_product(torch, cotangent, fd_jvp)
    rhs = tensor_map_inner_product(torch, pytorch_vjp, direction)
    if not _scalar_allclose(
        torch,
        lhs,
        rhs,
        rtol=first_order["rtol"],
        atol=first_order["atol"],
    ):
        raise ValueError("live probe failed adjoint consistency")
    if pytorch_hvp is None and fd_hvp is None:
        return
    if pytorch_hvp is None or fd_hvp is None:
        raise ValueError("live probe has incomplete HVP references")
    second_order = _second_order_comparison(comparison)
    if second_order is None:
        raise ValueError("live probe is missing second-order comparison tolerance")
    if not map_allclose(
        torch,
        pytorch_hvp,
        fd_hvp,
        rtol=second_order["rtol"],
        atol=second_order["atol"],
    ):
        raise ValueError("live PyTorch HVP and live FD-HVP disagree")


def _find_candidate_samples(
    torch,
    spec,
    record_inputs: dict[str, object],
    *,
    comparison: dict,
    dtype_name: str = "float64",
    op_args: list[object] | None = None,
    op_kwargs: dict[str, object] | None = None,
    prepared_samples: list[PreparedSample] | None = None,
) -> list[object]:
    metadata_args = [] if op_args is None else op_args
    metadata_kwargs = {} if op_kwargs is None else op_kwargs
    if prepared_samples is None:
        prepared_samples = _prepare_samples_for_spec_dtype(torch, spec, dtype_name=dtype_name)
    candidates: list[object] = []
    for prepared in prepared_samples:
        if _map_close(
            torch,
            record_inputs,
            prepared.inputs,
            comparison=comparison,
        ):
            if metadata_args or metadata_kwargs or getattr(spec, "inventory_kind", "linalg") == "scalar":
                if prepared.op_args != metadata_args or prepared.op_kwargs != metadata_kwargs:
                    continue
            candidates.append(prepared.sample)
    return candidates


def _prepare_samples_for_spec_dtype(torch, spec, *, dtype_name: str) -> list[PreparedSample]:
    if getattr(spec, "inventory_kind", "linalg") == "scalar":
        _, runtime_source = import_scalar_generation_runtime()
    else:
        _, runtime_source = import_generation_runtime()
    samples = sample_inputs_for_spec(
        torch,
        runtime_source,
        spec,
        seed=SAMPLE_INPUT_SEED,
        dtype=getattr(torch, dtype_name),
    )
    prepared: list[PreparedSample] = []
    for sample in samples:
        sample_args, sample_kwargs = build_call_metadata(torch, sample, spec=spec)
        prepared.append(
            PreparedSample(
                sample=sample,
                inputs=build_input_map(torch, spec, sample),
                op_args=sample_args,
                op_kwargs=sample_kwargs,
            )
        )
    return prepared


def _replay_success_case_for_sample(
    torch,
    *,
    record: dict,
    spec,
    sample,
    inputs: dict[str, object],
    direction: dict[str, object],
    cotangent: dict[str, object],
    stored_pytorch_jvp: dict[str, object],
    stored_pytorch_vjp: dict[str, object],
    stored_pytorch_hvp: dict[str, object] | None,
    stored_fd_jvp: dict[str, object],
    fd_step: float,
    stored_fd_hvp: dict[str, object] | None,
    op_args: list[object],
    op_kwargs: dict[str, object],
) -> str | None:
    comparison = record["comparison"]
    input_names = tuple(inputs.keys())
    first_order = _first_order_comparison(comparison)
    second_order = _second_order_comparison(comparison)
    if getattr(spec, "inventory_kind", "linalg") == "scalar":
        _, runtime_source = import_scalar_generation_runtime()
    else:
        _, runtime_source = import_generation_runtime()
    opinfo = lookup_upstream_opinfo(runtime_source, spec)

    try:
        observable = apply_spec_observable(
            torch,
            spec,
            sample,
            inputs,
            linalg=runtime_source,
            opinfo=opinfo,
            op_args=op_args,
            op_kwargs=op_kwargs,
        )
        output_names = tuple(observable.keys())
        observable_fn = build_observable_function(
            torch,
            spec,
            sample,
            input_names,
            linalg=runtime_source,
            opinfo=opinfo,
            op_args=op_args,
            op_kwargs=op_kwargs,
            output_names=output_names,
        )
        _, jvp_tuple = torch.func.jvp(
            observable_fn,
            tensor_map_to_tuple(inputs),
            tensor_map_to_tuple(direction),
        )
        pytorch_jvp = tuple_to_tensor_map(output_names, jvp_tuple)
        grads = torch.autograd.grad(
            tensor_map_to_tuple(observable),
            tensor_map_to_tuple(inputs),
            grad_outputs=tuple(cotangent[name] for name in output_names),
            allow_unused=True,
        )
        pytorch_vjp = zeros_like_input_map(torch, inputs, grads)
        plus_inputs = {name: tensor + fd_step * direction[name] for name, tensor in inputs.items()}
        minus_inputs = {name: tensor - fd_step * direction[name] for name, tensor in inputs.items()}
        plus_output = apply_spec_observable(
            torch,
            spec,
            sample,
            plus_inputs,
            linalg=runtime_source,
            opinfo=opinfo,
            op_args=op_args,
            op_kwargs=op_kwargs,
        )
        minus_output = apply_spec_observable(
            torch,
            spec,
            sample,
            minus_inputs,
            linalg=runtime_source,
            opinfo=opinfo,
            op_args=op_args,
            op_kwargs=op_kwargs,
        )
        fd_jvp = {
            name: (plus_output[name] - minus_output[name]) / (2.0 * fd_step)
            for name in output_names
        }
        pytorch_hvp = None
        fd_hvp = None
        if stored_pytorch_hvp is not None or stored_fd_hvp is not None:
            scalarized_fn = build_scalarized_observable_function(
                torch,
                observable_fn,
                output_names=output_names,
                cotangent=cotangent,
            )
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
    except Exception as exc:
        return str(exc)

    if not map_allclose(
        torch,
        stored_pytorch_jvp,
        pytorch_jvp,
        rtol=first_order["rtol"],
        atol=first_order["atol"],
    ):
        return "stored and replayed PyTorch JVP disagree"
    if not map_allclose(
        torch,
        stored_pytorch_vjp,
        pytorch_vjp,
        rtol=first_order["rtol"],
        atol=first_order["atol"],
    ):
        return "stored and replayed PyTorch VJP disagree"
    if not map_allclose(
        torch,
        stored_fd_jvp,
        fd_jvp,
        rtol=first_order["rtol"],
        atol=first_order["atol"],
    ):
        return "stored and replayed FD-JVP disagree"
    if stored_pytorch_hvp is not None and stored_fd_hvp is not None:
        if second_order is None:
            return "missing second-order comparison block"
        if not map_allclose(
            torch,
            stored_pytorch_hvp,
            pytorch_hvp,
            rtol=second_order["rtol"],
            atol=second_order["atol"],
        ):
            return "stored and replayed PyTorch HVP disagree"
        if not map_allclose(
            torch,
            stored_fd_hvp,
            fd_hvp,
            rtol=second_order["rtol"],
            atol=second_order["atol"],
        ):
            return "stored and replayed FD-HVP disagree"
    try:
        validate_live_success_probe(
            torch,
            comparison=comparison,
            direction=direction,
            cotangent=cotangent,
            pytorch_jvp=pytorch_jvp,
            pytorch_vjp=pytorch_vjp,
            fd_jvp=fd_jvp,
            pytorch_hvp=pytorch_hvp,
            fd_hvp=fd_hvp,
        )
    except ValueError as exc:
        return str(exc)
    return None


def _replay_success_case(
    record: dict,
    *,
    prepared_sample_cache: dict[tuple[str, str, str], list[PreparedSample]],
) -> None:
    import torch

    spec = build_case_spec_index()[(record["op"], record["family"])]
    inputs = _decode_record_inputs(record)
    op_args = list(record.get("op_args", []))
    op_kwargs = dict(record.get("op_kwargs", {}))
    (
        direction,
        cotangent,
        stored_pytorch_jvp,
        stored_pytorch_vjp,
        stored_pytorch_hvp,
        fd_step,
        stored_fd_hvp,
    ) = (
        _decode_success_probe(record)
    )
    stored_fd_jvp = decode_tensor_map(record["probes"][0]["fd_ref"]["jvp"])
    cache_key = (record["op"], record["family"], record["dtype"])
    prepared_samples = prepared_sample_cache.get(cache_key)
    if prepared_samples is None:
        prepared_samples = _prepare_samples_for_spec_dtype(
            torch,
            spec,
            dtype_name=record["dtype"],
        )
        prepared_sample_cache[cache_key] = prepared_samples

    candidates = _find_candidate_samples(
        torch,
        spec,
        inputs,
        comparison=record["comparison"],
        dtype_name=record["dtype"],
        op_args=op_args,
        op_kwargs=op_kwargs,
        prepared_samples=prepared_samples,
    )
    if not candidates:
        raise ValueError("no matching PyTorch SampleInput found for record inputs")

    mismatch_reasons: list[str] = []
    for sample in candidates:
        mismatch = _replay_success_case_for_sample(
            torch,
            record=record,
            spec=spec,
            sample=sample,
            inputs=inputs,
            direction=direction,
            cotangent=cotangent,
            stored_pytorch_jvp=stored_pytorch_jvp,
            stored_pytorch_vjp=stored_pytorch_vjp,
            stored_pytorch_hvp=stored_pytorch_hvp,
            stored_fd_jvp=stored_fd_jvp,
            fd_step=fd_step,
            stored_fd_hvp=stored_fd_hvp,
            op_args=op_args,
            op_kwargs=op_kwargs,
        )
        if mismatch is None:
            return
        mismatch_reasons.append(mismatch)

    unique_reasons = ", ".join(sorted(set(mismatch_reasons)))
    raise ValueError(f"no matching SampleInput replay matched stored references: {unique_reasons}")


def _replay_error_case(record: dict) -> None:
    import torch

    reason_code = record["comparison"]["reason_code"]
    if reason_code != "gauge_ill_defined":
        raise ValueError(f"unsupported error-case reason_code: {reason_code}")

    inputs = _decode_record_inputs(record)
    a = inputs["a"]

    if record["op"] == "svd":
        u, _, vh = torch.linalg.svd(a, full_matrices=False)
        loss = (u + vh).sum().abs()
    elif record["op"] == "eigh":
        loss = torch.linalg.eigh(a).eigenvectors.sum().abs()
    else:
        raise ValueError(f"unsupported error-case op: {record['op']}")

    try:
        loss.backward()
    except RuntimeError as exc:
        if "ill-defined" not in str(exc):
            raise ValueError(f"unexpected error while replaying expected failure: {exc}") from exc
        return
    raise ValueError("expected spectral decomposition backward to raise an ill-defined error")


def replay_case_file(path: Path, *, limit: int | None = None) -> ReplayResult:
    """Replay one JSONL case file and report verification failures."""
    result = ReplayResult()
    records = load_case_file(path)
    prepared_sample_cache: dict[tuple[str, str, str], list[PreparedSample]] = {}
    for record in records[:limit]:
        try:
            if record["expected_behavior"] == "success":
                _replay_success_case(
                    record,
                    prepared_sample_cache=prepared_sample_cache,
                )
            elif record["expected_behavior"] == "error":
                _replay_error_case(record)
            else:
                raise ValueError(
                    f"unsupported expected_behavior: {record['expected_behavior']}"
                )
        except Exception as exc:
            result.failures.append(f"{record['case_id']}: {exc}")
        else:
            result.checked += 1
    return result


def replay_case_tree(root: Path) -> ReplayResult:
    """Replay all published JSONL files under one case root."""
    combined = ReplayResult()
    for path in iter_case_files(root):
        result = replay_case_file(path)
        combined.checked += result.checked
        combined.failures.extend(result.failures)
    return combined
