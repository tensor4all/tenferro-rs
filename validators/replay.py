"""Replay published JSON cases and verify stored references are reproducible."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from generators.pytorch_v1 import build_case_spec_index
from generators.runtime import (
    apply_spec_observable,
    build_input_map,
    build_observable_function,
    import_generation_runtime,
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


SAMPLE_INPUT_SEED = 17


def _map_equal(torch, expected: dict[str, object], actual: dict[str, object]) -> bool:
    if expected.keys() != actual.keys():
        return False
    return all(torch.equal(expected[name], actual[name]) for name in expected)


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
        float(probe["fd_ref"]["step"]),
    )


def validate_live_success_probe(
    torch,
    *,
    comparison: dict,
    direction: dict[str, object],
    cotangent: dict[str, object],
    pytorch_jvp: dict[str, object],
    pytorch_vjp: dict[str, object],
    fd_jvp: dict[str, object],
) -> None:
    """Validate live cross-oracle agreement for one success-case probe."""
    if not map_allclose(
        torch,
        pytorch_jvp,
        fd_jvp,
        rtol=comparison["rtol"],
        atol=comparison["atol"],
    ):
        raise ValueError("live PyTorch JVP and live FD-JVP disagree")

    lhs = tensor_map_inner_product(torch, cotangent, fd_jvp)
    rhs = tensor_map_inner_product(torch, pytorch_vjp, direction)
    if not torch.allclose(
        lhs,
        rhs,
        rtol=comparison["rtol"],
        atol=comparison["atol"],
    ):
        raise ValueError("live probe failed adjoint consistency")


def _find_candidate_samples(torch, spec, record_inputs: dict[str, object]) -> list[object]:
    _, linalg = import_generation_runtime()
    samples = sample_inputs_for_spec(torch, linalg, spec, seed=SAMPLE_INPUT_SEED)
    candidates: list[object] = []
    for sample in samples:
        sample_inputs = build_input_map(torch, spec, sample)
        if _map_equal(torch, record_inputs, sample_inputs):
            candidates.append(sample)
    return candidates


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
    stored_fd_jvp: dict[str, object],
    fd_step: float,
) -> str | None:
    comparison = record["comparison"]
    input_names = tuple(inputs.keys())

    try:
        observable = apply_spec_observable(torch, spec, sample, inputs)
        output_names = tuple(observable.keys())
        observable_fn = build_observable_function(
            torch,
            spec,
            sample,
            input_names,
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
        plus_output = apply_spec_observable(torch, spec, sample, plus_inputs)
        minus_output = apply_spec_observable(torch, spec, sample, minus_inputs)
        fd_jvp = {
            name: (plus_output[name] - minus_output[name]) / (2.0 * fd_step)
            for name in output_names
        }
    except Exception as exc:
        return str(exc)

    if not map_allclose(
        torch,
        stored_pytorch_jvp,
        pytorch_jvp,
        rtol=comparison["rtol"],
        atol=comparison["atol"],
    ):
        return "stored and replayed PyTorch JVP disagree"
    if not map_allclose(
        torch,
        stored_pytorch_vjp,
        pytorch_vjp,
        rtol=comparison["rtol"],
        atol=comparison["atol"],
    ):
        return "stored and replayed PyTorch VJP disagree"
    if not map_allclose(
        torch,
        stored_fd_jvp,
        fd_jvp,
        rtol=comparison["rtol"],
        atol=comparison["atol"],
    ):
        return "stored and replayed FD-JVP disagree"
    try:
        validate_live_success_probe(
            torch,
            comparison=comparison,
            direction=direction,
            cotangent=cotangent,
            pytorch_jvp=pytorch_jvp,
            pytorch_vjp=pytorch_vjp,
            fd_jvp=fd_jvp,
        )
    except ValueError as exc:
        return str(exc)
    return None


def _replay_success_case(record: dict) -> None:
    import torch

    spec = build_case_spec_index()[(record["op"], record["family"])]
    inputs = _decode_record_inputs(record)
    direction, cotangent, stored_pytorch_jvp, stored_pytorch_vjp, fd_step = (
        _decode_success_probe(record)
    )
    stored_fd_jvp = decode_tensor_map(record["probes"][0]["fd_ref"]["jvp"])

    candidates = _find_candidate_samples(torch, spec, inputs)
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
            stored_fd_jvp=stored_fd_jvp,
            fd_step=fd_step,
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
    for record in records[:limit]:
        try:
            if record["expected_behavior"] == "success":
                _replay_success_case(record)
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
