"""Shared PyTorch runtime helpers for v1 case generation."""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Iterable

from . import observables

PINNED_TORCH_VERSION = "2.10.0"


def normalize_torch_version(version: str) -> str:
    """Strip local build metadata from a torch version string."""
    return version.split("+", 1)[0]


def ensure_pinned_torch_version(torch) -> None:
    """Raise when the active torch runtime does not match the repository pin."""
    actual = normalize_torch_version(torch.__version__)
    if actual != PINNED_TORCH_VERSION:
        raise RuntimeError(
            f"tensor-ad-oracles requires torch=={PINNED_TORCH_VERSION}, got {torch.__version__}"
        )


def import_generation_runtime():
    import torch
    from torch.testing._internal.opinfo.definitions import linalg

    ensure_pinned_torch_version(torch)
    return torch, linalg


def dtype_name(torch, dtype) -> str:
    if dtype == torch.float64:
        return "float64"
    if dtype == torch.complex128:
        return "complex128"
    raise ValueError(f"unsupported torch dtype for v1 generation: {dtype}")


def raw_tensor_norm(torch, tensor) -> float:
    if tensor.numel() == 0:
        return 0.0
    return float(torch.linalg.vector_norm(tensor).item())


def normalize_raw_tensor(torch, tensor):
    norm = raw_tensor_norm(torch, tensor)
    if norm == 0.0:
        return tensor.clone()
    return tensor / norm


def normalize_raw_tensor_map(torch, tensor_map: dict[str, object]) -> dict[str, object]:
    return {name: normalize_raw_tensor(torch, tensor) for name, tensor in tensor_map.items()}


def combined_input_norm(torch, tensor_map: dict[str, object]) -> float:
    square_sum = 0.0
    for tensor in tensor_map.values():
        norm = raw_tensor_norm(torch, tensor)
        square_sum += norm * norm
    return math.sqrt(square_sum)


def randn_like(torch, tensor, *, generator):
    return torch.randn(
        tensor.shape,
        dtype=tensor.dtype,
        device=tensor.device,
        generator=generator,
    )


def tensor_map_inner_product(torch, left: dict[str, object], right: dict[str, object]):
    left_items = list(left.items())
    if not left_items:
        return torch.tensor(0.0, dtype=torch.float64)
    total = None
    for name, left_tensor in left_items:
        right_tensor = right[name]
        if left_tensor.numel() == 0:
            continue
        term = torch.vdot(left_tensor.reshape(-1), right_tensor.reshape(-1))
        if torch.is_complex(term):
            term = term.real
        total = term if total is None else total + term
    if total is not None:
        return total
    first = left_items[0][1]
    return torch.zeros((), dtype=first.real.dtype, device=first.device)


def lookup_upstream_opinfo(linalg, spec):
    """Resolve the pinned PyTorch OpInfo entry corresponding to one DB spec."""
    for opinfo in linalg.op_db:
        if opinfo.name != spec.upstream_name:
            continue
        if (getattr(opinfo, "variant_test_name", "") or "") != spec.upstream_variant_name:
            continue
        return opinfo
    raise ValueError(
        f"no upstream OpInfo found for {spec.upstream_name!r} variant {spec.upstream_variant_name!r}"
    )


def sample_inputs_for_spec(torch, linalg, spec, *, seed: int):
    rng_state = torch.random.get_rng_state()
    py_state = random.getstate()
    try:
        torch.manual_seed(seed)
        random.seed(seed)
        opinfo = lookup_upstream_opinfo(linalg, spec)
        return list(opinfo.sample_inputs("cpu", torch.float64, requires_grad=True))
    finally:
        torch.random.set_rng_state(rng_state)
        random.setstate(py_state)


def tensor_map_to_tuple(tensor_map: dict[str, object]) -> tuple[object, ...]:
    return tuple(tensor_map.values())


def tuple_to_tensor_map(keys: Iterable[str], values) -> dict[str, object]:
    if isinstance(values, tuple):
        values_tuple = values
    else:
        values_tuple = (values,)
    return dict(zip(keys, values_tuple, strict=True))


def _is_differentiable_input_tensor(torch, value) -> bool:
    return isinstance(value, torch.Tensor) and (
        value.is_floating_point() or value.is_complex()
    )


def _bind_sample_tensors(torch, sample, inputs: dict[str, object]):
    index_ref = [0]

    def replace(value):
        if _is_differentiable_input_tensor(torch, value):
            name = _tensor_input_name(index_ref[0])
            index_ref[0] += 1
            return inputs[name]
        if isinstance(value, tuple):
            return tuple(replace(item) for item in value)
        if isinstance(value, list):
            return [replace(item) for item in value]
        return value

    return replace(sample.input), replace(sample.args)


def _call_upstream_op(torch, opinfo, sample, inputs: dict[str, object]):
    input_value, args = _bind_sample_tensors(torch, sample, inputs)
    wrapper = getattr(opinfo, "gradcheck_wrapper", None)
    wrapper_name = getattr(wrapper, "__name__", None)
    if wrapper is not None and wrapper_name != "<lambda>":
        return wrapper(opinfo.op, input_value, *args, **sample.kwargs)
    return opinfo.op(input_value, *args, **sample.kwargs)


def apply_spec_observable(
    torch,
    spec,
    sample,
    inputs: dict[str, object],
    *,
    linalg=None,
    opinfo=None,
    preserve_identity_keys: tuple[str, ...] | None = None,
) -> dict[str, object]:
    if opinfo is None:
        if linalg is None:
            _, linalg = import_generation_runtime()
        opinfo = lookup_upstream_opinfo(linalg, spec)
    result = _call_upstream_op(torch, opinfo, sample, inputs)
    return observables.apply_observable(
        spec.observable_kind,
        result,
        preserve_identity_keys=preserve_identity_keys,
    )


def zeros_like_input_map(torch, inputs: dict[str, object], grads) -> dict[str, object]:
    out: dict[str, object] = {}
    for (name, tensor), grad in zip(inputs.items(), grads, strict=True):
        out[name] = torch.zeros_like(tensor) if grad is None else grad
    return out


def map_allclose(torch, expected: dict[str, object], actual: dict[str, object], *, rtol: float, atol: float) -> bool:
    if expected.keys() != actual.keys():
        return False
    return all(
        torch.allclose(expected[name], actual[name], rtol=rtol, atol=atol)
        for name in expected
    )


def _uses_hermitian_wrapper(spec) -> bool:
    return getattr(spec, "gradcheck_wrapper", None) in {
        "hermitian_input",
        "gradcheck_wrapper_hermitian_input",
    }


def structured_input_tensor(torch, spec, tensor, *, is_primary_input: bool):
    cloned = tensor.detach().clone()
    if is_primary_input and _uses_hermitian_wrapper(spec):
        cloned = cloned + cloned.mH
    return cloned.requires_grad_(True)


def structured_direction_tensor(torch, spec, tensor, *, generator, is_primary_input: bool):
    direction = randn_like(torch, tensor, generator=generator)
    if is_primary_input and _uses_hermitian_wrapper(spec) and tensor.ndim >= 2:
        direction = direction + direction.mH
    return direction


def _tensor_input_name(index: int) -> str:
    names = "abcdefghijklmnopqrstuvwxyz"
    if index < len(names):
        return names[index]
    return f"t{index}"


def _collect_tensor_inputs(torch, spec, value, out: dict[str, object], index_ref: list[int]) -> None:
    if _is_differentiable_input_tensor(torch, value):
        index = index_ref[0]
        index_ref[0] += 1
        out[_tensor_input_name(index)] = structured_input_tensor(
            torch,
            spec,
            value,
            is_primary_input=(index == 0),
        )
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            _collect_tensor_inputs(torch, spec, item, out, index_ref)


def build_input_map(torch, spec, sample) -> dict[str, object]:
    inputs: dict[str, object] = {}
    index_ref = [0]
    _collect_tensor_inputs(torch, spec, sample.input, inputs, index_ref)
    _collect_tensor_inputs(torch, spec, sample.args, inputs, index_ref)
    if not inputs:
        raise ValueError(f"unsupported sample has no tensor inputs: {spec.op}")
    return inputs


def build_direction_map(torch, spec, inputs: dict[str, object], *, generator) -> dict[str, object]:
    return normalize_raw_tensor_map(
        torch,
        {
            name: structured_direction_tensor(
                torch,
                spec,
                tensor,
                generator=generator,
                is_primary_input=(index == 0),
            )
            for index, (name, tensor) in enumerate(inputs.items())
        },
    )


def build_observable_function(
    torch,
    spec,
    sample,
    input_names: tuple[str, ...],
    *,
    linalg=None,
    opinfo=None,
    output_names: tuple[str, ...] | None = None,
) -> Callable[..., tuple[object, ...]]:
    def observable_fn(*args):
        inputs = dict(zip(input_names, args, strict=True))
        output = apply_spec_observable(
            torch,
            spec,
            sample,
            inputs,
            linalg=linalg,
            opinfo=opinfo,
            preserve_identity_keys=output_names,
        )
        return tensor_map_to_tuple(output)

    return observable_fn
