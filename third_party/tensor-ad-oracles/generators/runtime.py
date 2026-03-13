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


def import_scalar_generation_runtime():
    import torch
    from torch.testing._internal import common_methods_invocations as cmi

    ensure_pinned_torch_version(torch)
    return torch, cmi


def dtype_name(torch, dtype) -> str:
    if dtype == torch.float32:
        return "float32"
    if dtype == torch.float64:
        return "float64"
    if dtype == torch.complex64:
        return "complex64"
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


def sample_inputs_for_spec(torch, linalg, spec, *, seed: int, dtype=None):
    rng_state = torch.random.get_rng_state()
    py_state = random.getstate()
    try:
        torch.manual_seed(seed)
        random.seed(seed)
        opinfo = lookup_upstream_opinfo(linalg, spec)
        sample_dtype = torch.float64 if dtype is None else dtype
        return list(opinfo.sample_inputs("cpu", sample_dtype, requires_grad=True))
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


def _contains_input_tensor(torch, value) -> bool:
    if _is_differentiable_input_tensor(torch, value):
        return True
    if isinstance(value, tuple):
        return any(_contains_input_tensor(torch, item) for item in value)
    if isinstance(value, list):
        return any(_contains_input_tensor(torch, item) for item in value)
    if isinstance(value, dict):
        return any(_contains_input_tensor(torch, item) for item in value.values())
    return False


def _contains_any_tensor(value) -> bool:
    try:
        import torch
    except Exception:
        torch = None
    if torch is not None and isinstance(value, torch.Tensor):
        return True
    if isinstance(value, tuple):
        return any(_contains_any_tensor(item) for item in value)
    if isinstance(value, list):
        return any(_contains_any_tensor(item) for item in value)
    if isinstance(value, dict):
        return any(_contains_any_tensor(item) for item in value.values())
    return False


def _is_metadata_tensor_kwarg(spec, key: str, value) -> bool:
    try:
        import torch
    except Exception:
        return False
    return (
        getattr(spec, "upstream_name", None) == "linalg.pinv"
        and key == "rtol"
        and isinstance(value, torch.Tensor)
    )


def _canonical_scalar_tensor_metadata(value):
    if value.numel() != 1:
        raise ValueError("only scalar tensor metadata is supported")
    item = value.detach().cpu().reshape(-1)[0].item()
    if isinstance(item, complex):
        return [float(item.real), float(item.imag)]
    if isinstance(item, bool):
        return item
    if isinstance(item, int):
        return int(item)
    return float(item)


def _canonical_metadata_value(torch, sample_value, value):
    if isinstance(value, tuple):
        return [
            _canonical_metadata_value(torch, sample_item, item)
            for sample_item, item in zip(sample_value, value, strict=True)
        ]
    if isinstance(value, list):
        return [
            _canonical_metadata_value(torch, sample_item, item)
            for sample_item, item in zip(sample_value, value, strict=True)
        ]
    if isinstance(value, dict):
        return {
            key: _canonical_metadata_value(torch, sample_value[key], item)
            for key, item in value.items()
        }
    if isinstance(value, torch.dtype):
        return dtype_name(torch, value)
    if isinstance(value, torch.memory_format):
        memory_format_names = {
            torch.contiguous_format: "contiguous_format",
            torch.channels_last: "channels_last",
            torch.channels_last_3d: "channels_last_3d",
            torch.preserve_format: "preserve_format",
        }
        if value not in memory_format_names:
            raise ValueError(f"unsupported torch memory_format for JSON metadata: {value!r}")
        return memory_format_names[value]
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, torch.layout):
        if value == torch.strided:
            return "strided"
        raise ValueError(f"unsupported torch layout for JSON metadata: {value!r}")
    return value


def _restore_metadata_like(torch, sample_value, metadata_value):
    if isinstance(sample_value, tuple):
        return tuple(
            _restore_metadata_like(torch, sample_item, metadata_item)
            for sample_item, metadata_item in zip(sample_value, metadata_value, strict=True)
        )
    if isinstance(sample_value, list):
        return [
            _restore_metadata_like(torch, sample_item, metadata_item)
            for sample_item, metadata_item in zip(sample_value, metadata_value, strict=True)
        ]
    if isinstance(sample_value, dict):
        return {
            key: _restore_metadata_like(torch, sample_value[key], metadata_value[key])
            for key in sample_value
        }
    if isinstance(sample_value, torch.Tensor):
        if isinstance(metadata_value, torch.Tensor):
            return metadata_value
        if isinstance(metadata_value, list):
            if len(metadata_value) != 2:
                raise ValueError(f"unsupported complex tensor metadata payload: {metadata_value!r}")
            scalar = complex(float(metadata_value[0]), float(metadata_value[1]))
        else:
            scalar = metadata_value
        return torch.tensor(
            scalar,
            dtype=sample_value.dtype,
            device=sample_value.device,
        ).reshape(sample_value.shape)
    if isinstance(sample_value, torch.dtype):
        if isinstance(metadata_value, torch.dtype):
            return metadata_value
        return getattr(torch, metadata_value)
    if isinstance(sample_value, torch.memory_format):
        if isinstance(metadata_value, torch.memory_format):
            return metadata_value
        return getattr(torch, metadata_value)
    if isinstance(sample_value, torch.device):
        if isinstance(metadata_value, torch.device):
            return metadata_value
        return torch.device(metadata_value)
    if isinstance(sample_value, torch.layout):
        if isinstance(metadata_value, torch.layout):
            return metadata_value
        if metadata_value == "strided":
            return torch.strided
        raise ValueError(f"unsupported JSON layout metadata: {metadata_value!r}")
    return metadata_value


def build_call_metadata(torch, sample, *, spec=None) -> tuple[list[object], dict[str, object]]:
    op_args = [
        _canonical_metadata_value(torch, arg, arg) for arg in sample.args if not _contains_any_tensor(arg)
    ]
    op_kwargs = {}
    for key, value in sample.kwargs.items():
        if _contains_any_tensor(value):
            if _is_metadata_tensor_kwarg(spec, key, value):
                op_kwargs[key] = _canonical_scalar_tensor_metadata(value)
            continue
        op_kwargs[key] = _canonical_metadata_value(torch, value, value)
    return op_args, op_kwargs


def _bind_sample_tensors(
    torch,
    spec,
    sample,
    inputs: dict[str, object],
    *,
    op_args: list[object] | tuple[object, ...] | None = None,
    op_kwargs: dict[str, object] | None = None,
):
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
        if isinstance(value, dict):
            return {key: replace(item) for key, item in value.items()}
        return value

    input_value = replace(sample.input)
    arg_iter = iter(op_args or ())
    args = []
    for value in sample.args:
        if _contains_input_tensor(torch, value):
            args.append(replace(value))
        else:
            metadata_value = next(arg_iter, value)
            args.append(_restore_metadata_like(torch, value, metadata_value))
    kwargs = {}
    metadata_kwargs = op_kwargs or {}
    for key, value in sample.kwargs.items():
        if _is_metadata_tensor_kwarg(spec, key, value):
            metadata_value = metadata_kwargs.get(key, value)
            kwargs[key] = _restore_metadata_like(torch, value, metadata_value)
            continue
        if _contains_input_tensor(torch, value):
            kwargs[key] = replace(value)
        else:
            metadata_value = metadata_kwargs.get(key, value)
            kwargs[key] = _restore_metadata_like(torch, value, metadata_value)
    for key, value in metadata_kwargs.items():
        if key not in kwargs:
            kwargs[key] = value
    return input_value, tuple(args), kwargs


def call_upstream_op(
    torch,
    spec,
    opinfo,
    sample,
    inputs: dict[str, object],
    *,
    op_args: list[object] | tuple[object, ...] | None = None,
    op_kwargs: dict[str, object] | None = None,
):
    input_value, args, kwargs = _bind_sample_tensors(
        torch,
        spec,
        sample,
        inputs,
        op_args=op_args,
        op_kwargs=op_kwargs,
    )
    wrapper = getattr(opinfo, "gradcheck_wrapper", None)
    wrapper_name = getattr(wrapper, "__name__", None)
    if wrapper is not None and wrapper_name != "<lambda>":
        return wrapper(opinfo.op, input_value, *args, **kwargs)
    return opinfo.op(input_value, *args, **kwargs)


def apply_spec_observable(
    torch,
    spec,
    sample,
    inputs: dict[str, object],
    *,
    linalg=None,
    opinfo=None,
    op_args: list[object] | tuple[object, ...] | None = None,
    op_kwargs: dict[str, object] | None = None,
    preserve_identity_keys: tuple[str, ...] | None = None,
) -> dict[str, object]:
    if opinfo is None:
        if linalg is None:
            _, linalg = import_generation_runtime()
        opinfo = lookup_upstream_opinfo(linalg, spec)
    result = call_upstream_op(
        torch,
        spec,
        opinfo,
        sample,
        inputs,
        op_args=op_args,
        op_kwargs=op_kwargs,
    )
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
        torch.allclose(expected[name], actual[name], rtol=rtol, atol=atol, equal_nan=True)
        for name in expected
    )


def tensor_map_isfinite(tensor_map: dict[str, object]) -> bool:
    for tensor in tensor_map.values():
        if tensor.numel() == 0:
            continue
        if not tensor.isfinite().all().item():
            return False
    return True


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
    if isinstance(value, dict):
        for key, item in value.items():
            if (
                key == "rtol"
                and getattr(spec, "upstream_name", None) == "linalg.pinv"
                and _is_differentiable_input_tensor(torch, item)
            ):
                continue
            _collect_tensor_inputs(torch, spec, item, out, index_ref)
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            _collect_tensor_inputs(torch, spec, item, out, index_ref)


def build_input_map(torch, spec, sample) -> dict[str, object]:
    inputs: dict[str, object] = {}
    index_ref = [0]
    _collect_tensor_inputs(torch, spec, sample.input, inputs, index_ref)
    _collect_tensor_inputs(torch, spec, sample.args, inputs, index_ref)
    _collect_tensor_inputs(torch, spec, sample.kwargs, inputs, index_ref)
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
    op_args: list[object] | tuple[object, ...] | None = None,
    op_kwargs: dict[str, object] | None = None,
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
            op_args=op_args,
            op_kwargs=op_kwargs,
            preserve_identity_keys=output_names,
        )
        return tensor_map_to_tuple(output)

    return observable_fn


def build_scalarized_observable_function(
    torch,
    observable_fn: Callable[..., tuple[object, ...]],
    *,
    output_names: tuple[str, ...],
    cotangent: dict[str, object],
) -> Callable[..., object]:
    """Build `phi(x) = <cotangent, observable(x)>` using the repository scalarization."""

    def scalarized_fn(*args):
        observable = tuple_to_tensor_map(output_names, observable_fn(*args))
        return tensor_map_inner_product(torch, cotangent, observable)

    return scalarized_fn


def compute_pytorch_hvp(
    torch,
    scalarized_fn: Callable[..., object],
    *,
    inputs: dict[str, object],
    direction: dict[str, object],
) -> dict[str, object]:
    """Compute a scalarized HVP with PyTorch `grad` + `jvp`."""
    input_names = tuple(inputs.keys())
    argnums = tuple(range(len(input_names)))
    grad_fn = torch.func.grad(scalarized_fn, argnums=argnums)
    _, hvp = torch.func.jvp(
        grad_fn,
        tensor_map_to_tuple(inputs),
        tensor_map_to_tuple(direction),
    )
    return tuple_to_tensor_map(input_names, hvp)


def compute_fd_hvp(
    torch,
    scalarized_fn: Callable[..., object],
    *,
    inputs: dict[str, object],
    direction: dict[str, object],
    step: float,
) -> dict[str, object]:
    """Compute a scalarized HVP by central differences on the scalarized gradient."""
    input_names = tuple(inputs.keys())
    argnums = tuple(range(len(input_names)))
    grad_fn = torch.func.grad(scalarized_fn, argnums=argnums)
    plus_inputs = {
        name: tensor + step * direction[name] for name, tensor in inputs.items()
    }
    minus_inputs = {
        name: tensor - step * direction[name] for name, tensor in inputs.items()
    }
    plus_grad = tuple_to_tensor_map(input_names, grad_fn(*tensor_map_to_tuple(plus_inputs)))
    minus_grad = tuple_to_tensor_map(
        input_names,
        grad_fn(*tensor_map_to_tuple(minus_inputs)),
    )
    return {
        name: (plus_grad[name] - minus_grad[name]) / (2.0 * step)
        for name in input_names
    }
