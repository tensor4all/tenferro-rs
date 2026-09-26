//! CUDA operation bodies, as free functions over a `CudaBackend`.
//!
//! The owner-level operation traits (`TensorElementwise`, `TensorAnalytic`,
//! `TensorStructural`, `TensorReduction`, `TensorDot`, `TensorIndexing`,
//! `TensorFusion`) and the owner `BackendSession` no longer exist: the session
//! is the only route, so these bodies live here and `CudaExecSession`'s
//! implementations forward to them. Each function takes the same arguments as
//! the trait method it came from, with the receiver replaced by an explicit
//! `&mut CudaBackend`.

use super::*;

pub(super) fn add_read(
    backend: &mut CudaBackend,
    lhs: TensorRead<'_>,
    rhs: TensorRead<'_>,
) -> crate::Result<Tensor> {
    if let Some(result) =
        backend.binary_read_native(ElementwiseReadOp::Add, lhs.clone(), rhs.clone())
    {
        return result;
    }
    let lhs = backend.read_input(lhs)?;
    let rhs = backend.read_input(rhs)?;
    let lhs = lhs.as_tensor();
    let rhs = rhs.as_tensor();
    if let Some(result) =
        promoted_real_complex_scalar_binary(backend, lhs, rhs, "add", elementwise::MIXED_ADD)
    {
        return result;
    }
    dispatch::dispatch_binary_float_complex_int!(
        backend,
        lhs,
        rhs,
        PrimitiveOpKind::Add,
        add_float,
        add_int,
        add_complex
    )
}

pub(super) fn sub_read(
    backend: &mut CudaBackend,
    lhs: TensorRead<'_>,
    rhs: TensorRead<'_>,
) -> crate::Result<Tensor> {
    if let Some(result) =
        backend.binary_read_native(ElementwiseReadOp::Subtract, lhs.clone(), rhs.clone())
    {
        return result;
    }
    let lhs = backend.read_input(lhs)?;
    let rhs = backend.read_input(rhs)?;
    let lhs = lhs.as_tensor();
    let rhs = rhs.as_tensor();
    if let Some(result) =
        promoted_real_complex_scalar_binary(backend, lhs, rhs, "sub", elementwise::MIXED_SUB)
    {
        return result;
    }
    dispatch::dispatch_binary_float_complex_int!(
        backend,
        lhs,
        rhs,
        PrimitiveOpKind::Sub,
        sub_float,
        sub_int,
        sub_complex
    )
}

pub(super) fn mul_read(
    backend: &mut CudaBackend,
    lhs: TensorRead<'_>,
    rhs: TensorRead<'_>,
) -> crate::Result<Tensor> {
    if let Some(result) =
        backend.binary_read_native(ElementwiseReadOp::Multiply, lhs.clone(), rhs.clone())
    {
        return result;
    }
    let lhs = backend.read_input(lhs)?;
    let rhs = backend.read_input(rhs)?;
    let lhs = lhs.as_tensor();
    let rhs = rhs.as_tensor();
    if let Some(result) =
        promoted_real_complex_scalar_binary(backend, lhs, rhs, "mul", elementwise::MIXED_MUL)
    {
        return result;
    }
    dispatch::dispatch_binary_float_complex_int!(
        backend,
        lhs,
        rhs,
        PrimitiveOpKind::Mul,
        mul_float,
        mul_int,
        mul_complex
    )
}

pub(super) fn neg_read(backend: &mut CudaBackend, input: TensorRead<'_>) -> crate::Result<Tensor> {
    if let Some(result) = backend.unary_read_native(UnaryReadOp::Neg, input.clone()) {
        return result;
    }
    let input = backend.read_input(input)?;
    let input = input.as_tensor();
    dispatch::dispatch_unary_float_complex_int!(
        backend,
        input,
        PrimitiveOpKind::Neg,
        neg_float,
        neg_int,
        neg_complex
    )
}

pub(super) fn conj_read(backend: &mut CudaBackend, input: TensorRead<'_>) -> crate::Result<Tensor> {
    let input = backend.read_input(input)?;
    let input = input.as_tensor();
    let op = op_name(
        PrimitiveOpKind::Conj,
        op_descriptor::GpuLaunchKind::UnaryFloatComplex,
    )?;
    // Dispatch on the tag and recover the typed tensor, which is what `as_typed` exists for.
    match input.dtype() {
        DType::F32 => {
            let tensor = typed_or_unsupported::<f32>(input, op)?;
            ensure_resident_on_runtime(backend.runtime(), tensor, op)?;
            backend
                .to_contiguous_view_typed(&tensor.as_view(), op)
                .map(Tensor::from_typed::<f32>)
        }
        DType::F64 => {
            let tensor = typed_or_unsupported::<f64>(input, op)?;
            ensure_resident_on_runtime(backend.runtime(), tensor, op)?;
            backend
                .to_contiguous_view_typed(&tensor.as_view(), op)
                .map(Tensor::from_typed::<f64>)
        }
        DType::I32 | DType::I64 | DType::Bool => Err(unsupported_dtype(op, input.dtype())),
        DType::C32 => {
            let tensor = typed_or_unsupported::<Complex32>(input, op)?;
            launch_unary(
                backend.runtime(),
                tensor,
                tensor.shape(),
                op,
                |client, count, dim, out, input_arg| unsafe {
                    elementwise::conj_complex::launch_unchecked::<Complex32, CubeclCudaRuntime>(
                        client, count, dim, out, input_arg,
                    );
                },
            )
            .map(Tensor::from_typed::<num_complex::Complex32>)
        }
        DType::C64 => {
            let tensor = typed_or_unsupported::<Complex64>(input, op)?;
            launch_unary(
                backend.runtime(),
                tensor,
                tensor.shape(),
                op,
                |client, count, dim, out, input_arg| unsafe {
                    elementwise::conj_complex::launch_unchecked::<Complex64, CubeclCudaRuntime>(
                        client, count, dim, out, input_arg,
                    );
                },
            )
            .map(Tensor::from_typed::<num_complex::Complex64>)
        }
        // A caller-owned payload has no GPU implementation for this operation.
        DType::External(_) => Err(crate::Error::unsupported(
            "conj",
            "an externally defined payload is not supported by this GPU operation",
        )),
    }
}

pub(super) fn div_read(
    backend: &mut CudaBackend,
    lhs: TensorRead<'_>,
    rhs: TensorRead<'_>,
) -> crate::Result<Tensor> {
    if let Some(result) =
        backend.binary_read_native(ElementwiseReadOp::Divide, lhs.clone(), rhs.clone())
    {
        return result;
    }
    let lhs = backend.read_input(lhs)?;
    let rhs = backend.read_input(rhs)?;
    let lhs = lhs.as_tensor();
    let rhs = rhs.as_tensor();
    let op = op_name(
        PrimitiveOpKind::Div,
        op_descriptor::GpuLaunchKind::BinaryFloatComplexInt,
    )?;
    if let Some(result) =
        promoted_real_complex_scalar_binary(backend, lhs, rhs, op, elementwise::MIXED_DIV)
    {
        return result;
    }
    match (lhs.dtype(), rhs.dtype()) {
        (DType::F32, DType::F32) if lhs.shape() != rhs.shape() => launch_scalar_binary(
            backend,
            typed_or_unsupported::<f32>(lhs, op)?,
            typed_or_unsupported::<f32>(rhs, op)?,
            op,
            |client, count, dim, out, lhs_arg, rhs_arg, lhs_scalar| unsafe {
                elementwise::scalar_div_float::launch_unchecked::<f32, CubeclCudaRuntime>(
                    client, count, dim, out, lhs_arg, rhs_arg, lhs_scalar,
                );
            },
        )
        .map(Tensor::from_typed::<f32>),
        (DType::F32, DType::F32) => launch_binary(
            backend.runtime(),
            typed_or_unsupported::<f32>(lhs, op)?,
            typed_or_unsupported::<f32>(rhs, op)?,
            lhs.shape(),
            op,
            |client, count, dim, out, lhs_arg, rhs_arg| unsafe {
                elementwise::div_float::launch_unchecked::<f32, CubeclCudaRuntime>(
                    client, count, dim, out, lhs_arg, rhs_arg,
                );
            },
        )
        .map(Tensor::from_typed::<f32>),
        (DType::F64, DType::F64) if lhs.shape() != rhs.shape() => launch_scalar_binary(
            backend,
            typed_or_unsupported::<f64>(lhs, op)?,
            typed_or_unsupported::<f64>(rhs, op)?,
            op,
            |client, count, dim, out, lhs_arg, rhs_arg, lhs_scalar| unsafe {
                elementwise::scalar_div_float::launch_unchecked::<f64, CubeclCudaRuntime>(
                    client, count, dim, out, lhs_arg, rhs_arg, lhs_scalar,
                );
            },
        )
        .map(Tensor::from_typed::<f64>),
        (DType::F64, DType::F64) => launch_binary(
            backend.runtime(),
            typed_or_unsupported::<f64>(lhs, op)?,
            typed_or_unsupported::<f64>(rhs, op)?,
            lhs.shape(),
            op,
            |client, count, dim, out, lhs_arg, rhs_arg| unsafe {
                elementwise::div_float::launch_unchecked::<f64, CubeclCudaRuntime>(
                    client, count, dim, out, lhs_arg, rhs_arg,
                );
            },
        )
        .map(Tensor::from_typed::<f64>),
        (DType::I32, DType::I32) if lhs.shape() != rhs.shape() => {
            launch_checked_integer_scalar_binary(
                backend,
                typed_or_unsupported::<i32>(lhs, op)?,
                typed_or_unsupported::<i32>(rhs, op)?,
                op,
                crate::DType::I32,
                CheckedIntegerDomain::DivisionByZero,
                |client, count, dim, out, lhs_arg, rhs_arg, err_arg, lhs_scalar| unsafe {
                    elementwise::scalar_div_int_checked::launch_unchecked::<i32, CubeclCudaRuntime>(
                        client, count, dim, out, lhs_arg, rhs_arg, err_arg, lhs_scalar,
                    );
                },
            )
            .map(Tensor::from_typed::<i32>)
        }
        (DType::I32, DType::I32) => launch_checked_integer_binary(
            backend,
            typed_or_unsupported::<i32>(lhs, op)?,
            typed_or_unsupported::<i32>(rhs, op)?,
            op,
            crate::DType::I32,
            CheckedIntegerDomain::DivisionByZero,
            |client, count, dim, out, lhs_arg, rhs_arg, err_arg| unsafe {
                elementwise::div_int_checked::launch_unchecked::<i32, CubeclCudaRuntime>(
                    client, count, dim, out, lhs_arg, rhs_arg, err_arg,
                );
            },
        )
        .map(Tensor::from_typed::<i32>),
        (DType::I64, DType::I64) if lhs.shape() != rhs.shape() => {
            launch_checked_integer_scalar_binary(
                backend,
                typed_or_unsupported::<i64>(lhs, op)?,
                typed_or_unsupported::<i64>(rhs, op)?,
                op,
                crate::DType::I64,
                CheckedIntegerDomain::DivisionByZero,
                |client, count, dim, out, lhs_arg, rhs_arg, err_arg, lhs_scalar| unsafe {
                    elementwise::scalar_div_int_checked::launch_unchecked::<i64, CubeclCudaRuntime>(
                        client, count, dim, out, lhs_arg, rhs_arg, err_arg, lhs_scalar,
                    );
                },
            )
            .map(Tensor::from_typed::<i64>)
        }
        (DType::I64, DType::I64) => launch_checked_integer_binary(
            backend,
            typed_or_unsupported::<i64>(lhs, op)?,
            typed_or_unsupported::<i64>(rhs, op)?,
            op,
            crate::DType::I64,
            CheckedIntegerDomain::DivisionByZero,
            |client, count, dim, out, lhs_arg, rhs_arg, err_arg| unsafe {
                elementwise::div_int_checked::launch_unchecked::<i64, CubeclCudaRuntime>(
                    client, count, dim, out, lhs_arg, rhs_arg, err_arg,
                );
            },
        )
        .map(Tensor::from_typed::<i64>),
        (DType::C32, DType::C32) => launch_binary(
            backend.runtime(),
            typed_or_unsupported::<Complex32>(lhs, op)?,
            typed_or_unsupported::<Complex32>(rhs, op)?,
            lhs.shape(),
            op,
            |client, count, dim, out, lhs_arg, rhs_arg| unsafe {
                elementwise::div_complex::launch_unchecked::<Complex32, CubeclCudaRuntime>(
                    client, count, dim, out, lhs_arg, rhs_arg,
                );
            },
        )
        .map(Tensor::from_typed::<num_complex::Complex32>),
        (DType::C64, DType::C64) => launch_binary(
            backend.runtime(),
            typed_or_unsupported::<Complex64>(lhs, op)?,
            typed_or_unsupported::<Complex64>(rhs, op)?,
            lhs.shape(),
            op,
            |client, count, dim, out, lhs_arg, rhs_arg| unsafe {
                elementwise::div_complex::launch_unchecked::<Complex64, CubeclCudaRuntime>(
                    client, count, dim, out, lhs_arg, rhs_arg,
                );
            },
        )
        .map(Tensor::from_typed::<num_complex::Complex64>),
        _ => Err(dtype_mismatch(op, lhs, rhs)),
    }
}

pub(super) fn rem_read(
    backend: &mut CudaBackend,
    lhs: TensorRead<'_>,
    rhs: TensorRead<'_>,
) -> crate::Result<Tensor> {
    let lhs = backend.read_input(lhs)?;
    let rhs = backend.read_input(rhs)?;
    ops::rem(backend, lhs.as_tensor(), rhs.as_tensor())
}

pub(super) fn abs_read(backend: &mut CudaBackend, input: TensorRead<'_>) -> crate::Result<Tensor> {
    let input = backend.read_input(input)?;
    let input = input.as_tensor();
    let descriptor = op_descriptor::require_gpu_descriptor(
        PrimitiveOpKind::Abs,
        op_descriptor::GpuLaunchKind::UnaryFloatInt,
    )?;
    let op = descriptor.name;
    dispatch::require_owned_capability(backend, PrimitiveOpKind::Abs, input.dtype())?;
    // Dispatch on the tag and recover the typed tensor, which is what `as_typed` exists for.
    match input.dtype() {
        DType::F32 => {
            let tensor = typed_or_unsupported::<f32>(input, op)?;
            dispatch::launch_unary_elementwise_kernel!(backend, tensor, op, abs_float, f32, F32)
        }
        DType::F64 => {
            let tensor = typed_or_unsupported::<f64>(input, op)?;
            dispatch::launch_unary_elementwise_kernel!(backend, tensor, op, abs_float, f64, F64)
        }
        DType::I32 => {
            let tensor = typed_or_unsupported::<i32>(input, op)?;
            dispatch::launch_unary_elementwise_kernel!(backend, tensor, op, abs_int, i32, I32)
        }
        DType::I64 => {
            let tensor = typed_or_unsupported::<i64>(input, op)?;
            dispatch::launch_unary_elementwise_kernel!(backend, tensor, op, abs_int, i64, I64)
        }
        DType::C32 => {
            let tensor = typed_or_unsupported::<Complex32>(input, op)?;
            dispatch::launch_unary(
                backend.runtime(),
                tensor,
                tensor.shape(),
                op,
                |client, count, dim, out, input_arg| unsafe {
                    elementwise::abs_complex32::launch_unchecked::<CubeclCudaRuntime>(
                        client, count, dim, out, input_arg,
                    );
                },
            )
            .map(Tensor::from_typed::<f32>)
        }
        DType::C64 => {
            let tensor = typed_or_unsupported::<Complex64>(input, op)?;
            dispatch::launch_unary(
                backend.runtime(),
                tensor,
                tensor.shape(),
                op,
                |client, count, dim, out, input_arg| unsafe {
                    elementwise::abs_complex64::launch_unchecked::<CubeclCudaRuntime>(
                        client, count, dim, out, input_arg,
                    );
                },
            )
            .map(Tensor::from_typed::<f64>)
        }
        DType::Bool => Err(unsupported_dtype(op, input.dtype())),
        // A caller-owned payload has no GPU implementation for this operation.
        DType::External(_) => Err(crate::Error::unsupported(
            "abs",
            "an externally defined payload is not supported by this GPU operation",
        )),
    }
}

pub(super) fn sign_read(backend: &mut CudaBackend, input: TensorRead<'_>) -> crate::Result<Tensor> {
    let input = backend.read_input(input)?;
    let input = input.as_tensor();
    dispatch::dispatch_unary_float_complex_int!(
        backend,
        input,
        PrimitiveOpKind::Sign,
        sign_float,
        sign_int,
        sign_complex
    )
}

pub(super) fn maximum_read(
    backend: &mut CudaBackend,
    lhs: TensorRead<'_>,
    rhs: TensorRead<'_>,
) -> crate::Result<Tensor> {
    let lhs = backend.read_input(lhs)?;
    let rhs = backend.read_input(rhs)?;
    let lhs = lhs.as_tensor();
    let rhs = rhs.as_tensor();
    dispatch::dispatch_binary_float_int!(
        backend,
        lhs,
        rhs,
        PrimitiveOpKind::Maximum,
        maximum_float,
        maximum_int
    )
}

pub(super) fn minimum_read(
    backend: &mut CudaBackend,
    lhs: TensorRead<'_>,
    rhs: TensorRead<'_>,
) -> crate::Result<Tensor> {
    let lhs = backend.read_input(lhs)?;
    let rhs = backend.read_input(rhs)?;
    let lhs = lhs.as_tensor();
    let rhs = rhs.as_tensor();
    dispatch::dispatch_binary_float_int!(
        backend,
        lhs,
        rhs,
        PrimitiveOpKind::Minimum,
        minimum_float,
        minimum_int
    )
}

pub(super) fn compare_read(
    backend: &mut CudaBackend,
    lhs: TensorRead<'_>,
    rhs: TensorRead<'_>,
    dir: &CompareDir,
) -> crate::Result<Tensor> {
    let lhs = backend.read_input(lhs)?;
    let lhs = lhs.as_tensor();
    let rhs = backend.read_input(rhs)?;
    let rhs = rhs.as_tensor();
    let op = op_name(
        PrimitiveOpKind::Compare,
        op_descriptor::GpuLaunchKind::CompareFloatIntToBool,
    )?;
    match (lhs.dtype(), rhs.dtype()) {
        (DType::F32, DType::F32) => launch_compare_bool(
            backend.runtime(),
            typed_or_unsupported::<f32>(lhs, op)?,
            typed_or_unsupported::<f32>(rhs, op)?,
            lhs.shape(),
            op,
            |client, count, dim, out, lhs_arg, rhs_arg| unsafe {
                elementwise::compare_float_bool::launch_unchecked::<f32, CubeclCudaRuntime>(
                    client,
                    count,
                    dim,
                    out,
                    lhs_arg,
                    rhs_arg,
                    dispatch::compare_mode(dir),
                );
            },
        )
        .map(Tensor::from_typed::<bool>),
        (DType::F64, DType::F64) => launch_compare_bool(
            backend.runtime(),
            typed_or_unsupported::<f64>(lhs, op)?,
            typed_or_unsupported::<f64>(rhs, op)?,
            lhs.shape(),
            op,
            |client, count, dim, out, lhs_arg, rhs_arg| unsafe {
                elementwise::compare_float_bool::launch_unchecked::<f64, CubeclCudaRuntime>(
                    client,
                    count,
                    dim,
                    out,
                    lhs_arg,
                    rhs_arg,
                    dispatch::compare_mode(dir),
                );
            },
        )
        .map(Tensor::from_typed::<bool>),
        (DType::I32, DType::I32) => launch_compare_bool(
            backend.runtime(),
            typed_or_unsupported::<i32>(lhs, op)?,
            typed_or_unsupported::<i32>(rhs, op)?,
            lhs.shape(),
            op,
            |client, count, dim, out, lhs_arg, rhs_arg| unsafe {
                elementwise::compare_int_bool::launch_unchecked::<i32, CubeclCudaRuntime>(
                    client,
                    count,
                    dim,
                    out,
                    lhs_arg,
                    rhs_arg,
                    dispatch::compare_mode(dir),
                );
            },
        )
        .map(Tensor::from_typed::<bool>),
        (DType::I64, DType::I64) => launch_compare_bool(
            backend.runtime(),
            typed_or_unsupported::<i64>(lhs, op)?,
            typed_or_unsupported::<i64>(rhs, op)?,
            lhs.shape(),
            op,
            |client, count, dim, out, lhs_arg, rhs_arg| unsafe {
                elementwise::compare_int_bool::launch_unchecked::<i64, CubeclCudaRuntime>(
                    client,
                    count,
                    dim,
                    out,
                    lhs_arg,
                    rhs_arg,
                    dispatch::compare_mode(dir),
                );
            },
        )
        .map(Tensor::from_typed::<bool>),
        (DType::C32, DType::C32) | (DType::C64, DType::C64) => {
            Err(unsupported_dtype(op, lhs.dtype()))
        }
        _ => Err(dtype_mismatch(op, lhs, rhs)),
    }
}

pub(super) fn select_read(
    backend: &mut CudaBackend,
    pred: TensorRead<'_>,
    on_true: TensorRead<'_>,
    on_false: TensorRead<'_>,
) -> crate::Result<Tensor> {
    let pred_owned = backend.read_input(pred)?;
    let on_true_owned = backend.read_input(on_true)?;
    let on_false_owned = backend.read_input(on_false)?;
    let pred = pred_owned.as_tensor();
    let on_true = on_true_owned.as_tensor();
    let on_false = on_false_owned.as_tensor();
    let op = op_name(
        PrimitiveOpKind::Select,
        op_descriptor::GpuLaunchKind::SelectBoolFloatInt,
    )?;
    match (pred.dtype(), on_true.dtype(), on_false.dtype()) {
        (DType::Bool, DType::F32, DType::F32) => {
            let pred = typed_or_unsupported::<bool>(pred, op)?;
            let on_true = typed_or_unsupported::<f32>(on_true, op)?;
            let on_false = typed_or_unsupported::<f32>(on_false, op)?;
            launch_select_bool(
                backend.runtime(),
                pred,
                on_true,
                on_false,
                pred.shape(),
                op,
                |client, count, dim, out, pred_arg, true_arg, false_arg| unsafe {
                    elementwise::select_bool_float::launch_unchecked::<f32, CubeclCudaRuntime>(
                        client, count, dim, out, pred_arg, true_arg, false_arg,
                    );
                },
            )
            .map(Tensor::from_typed::<f32>)
        }
        (DType::Bool, DType::F64, DType::F64) => {
            let pred = typed_or_unsupported::<bool>(pred, op)?;
            let on_true = typed_or_unsupported::<f64>(on_true, op)?;
            let on_false = typed_or_unsupported::<f64>(on_false, op)?;
            launch_select_bool(
                backend.runtime(),
                pred,
                on_true,
                on_false,
                pred.shape(),
                op,
                |client, count, dim, out, pred_arg, true_arg, false_arg| unsafe {
                    elementwise::select_bool_float::launch_unchecked::<f64, CubeclCudaRuntime>(
                        client, count, dim, out, pred_arg, true_arg, false_arg,
                    );
                },
            )
            .map(Tensor::from_typed::<f64>)
        }
        (DType::Bool, DType::I32, DType::I32) => {
            let pred = typed_or_unsupported::<bool>(pred, op)?;
            let on_true = typed_or_unsupported::<i32>(on_true, op)?;
            let on_false = typed_or_unsupported::<i32>(on_false, op)?;
            launch_select_bool(
                backend.runtime(),
                pred,
                on_true,
                on_false,
                pred.shape(),
                op,
                |client, count, dim, out, pred_arg, true_arg, false_arg| unsafe {
                    elementwise::select_bool_int::launch_unchecked::<i32, CubeclCudaRuntime>(
                        client, count, dim, out, pred_arg, true_arg, false_arg,
                    );
                },
            )
            .map(Tensor::from_typed::<i32>)
        }
        (DType::Bool, DType::I64, DType::I64) => {
            let pred = typed_or_unsupported::<bool>(pred, op)?;
            let on_true = typed_or_unsupported::<i64>(on_true, op)?;
            let on_false = typed_or_unsupported::<i64>(on_false, op)?;
            launch_select_bool(
                backend.runtime(),
                pred,
                on_true,
                on_false,
                pred.shape(),
                op,
                |client, count, dim, out, pred_arg, true_arg, false_arg| unsafe {
                    elementwise::select_bool_int::launch_unchecked::<i64, CubeclCudaRuntime>(
                        client, count, dim, out, pred_arg, true_arg, false_arg,
                    );
                },
            )
            .map(Tensor::from_typed::<i64>)
        }
        (DType::C32, DType::C32, DType::C32) | (DType::C64, DType::C64, DType::C64) => {
            Err(unsupported_dtype(op, pred.dtype()))
        }
        _ => Err(ternary_dtype_mismatch(op, pred, on_true, on_false)),
    }
}

pub(super) fn clamp_read(
    backend: &mut CudaBackend,
    input: TensorRead<'_>,
    lower: TensorRead<'_>,
    upper: TensorRead<'_>,
) -> crate::Result<Tensor> {
    let input = backend.read_input(input)?;
    let input = input.as_tensor();
    let lower = backend.read_input(lower)?;
    let lower = lower.as_tensor();
    let upper = backend.read_input(upper)?;
    let upper = upper.as_tensor();
    let op = op_name(
        PrimitiveOpKind::Clamp,
        op_descriptor::GpuLaunchKind::ClampFloat,
    )?;
    // Dispatch on the tags and recover the typed tensors, which is what `as_typed` exists for.
    match (input.dtype(), lower.dtype(), upper.dtype()) {
        (DType::F32, DType::F32, DType::F32) => {
            let input = typed_or_unsupported::<f32>(input, op)?;
            let lower = typed_or_unsupported::<f32>(lower, op)?;
            let upper = typed_or_unsupported::<f32>(upper, op)?;
            launch_ternary(
                backend.runtime(),
                input,
                lower,
                upper,
                input.shape(),
                op,
                |client, count, dim, out, input_arg, lower_arg, upper_arg| unsafe {
                    elementwise::clamp_float::launch_unchecked::<f32, CubeclCudaRuntime>(
                        client, count, dim, out, input_arg, lower_arg, upper_arg,
                    );
                },
            )
            .map(Tensor::from_typed::<f32>)
        }
        (DType::F64, DType::F64, DType::F64) => {
            let input = typed_or_unsupported::<f64>(input, op)?;
            let lower = typed_or_unsupported::<f64>(lower, op)?;
            let upper = typed_or_unsupported::<f64>(upper, op)?;
            launch_ternary(
                backend.runtime(),
                input,
                lower,
                upper,
                input.shape(),
                op,
                |client, count, dim, out, input_arg, lower_arg, upper_arg| unsafe {
                    elementwise::clamp_float::launch_unchecked::<f64, CubeclCudaRuntime>(
                        client, count, dim, out, input_arg, lower_arg, upper_arg,
                    );
                },
            )
            .map(Tensor::from_typed::<f64>)
        }
        (DType::C32, DType::C32, DType::C32) | (DType::C64, DType::C64, DType::C64) => {
            Err(unsupported_dtype(op, input.dtype()))
        }
        _ => Err(ternary_dtype_mismatch(op, input, lower, upper)),
    }
}

pub(super) fn rem(backend: &mut CudaBackend, lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor> {
    let op = op_name(
        PrimitiveOpKind::Rem,
        op_descriptor::GpuLaunchKind::BinaryFloatInt,
    )?;
    match (lhs.dtype(), rhs.dtype()) {
        (DType::F32, DType::F32) if lhs.shape() != rhs.shape() => launch_scalar_binary(
            backend,
            typed_or_unsupported::<f32>(lhs, op)?,
            typed_or_unsupported::<f32>(rhs, op)?,
            op,
            |client, count, dim, out, lhs_arg, rhs_arg, lhs_scalar| unsafe {
                elementwise::scalar_rem_float::launch_unchecked::<f32, CubeclCudaRuntime>(
                    client, count, dim, out, lhs_arg, rhs_arg, lhs_scalar,
                );
            },
        )
        .map(Tensor::from_typed::<f32>),
        (DType::F32, DType::F32) => launch_binary(
            backend.runtime(),
            typed_or_unsupported::<f32>(lhs, op)?,
            typed_or_unsupported::<f32>(rhs, op)?,
            lhs.shape(),
            op,
            |client, count, dim, out, lhs_arg, rhs_arg| unsafe {
                elementwise::rem_float::launch_unchecked::<f32, CubeclCudaRuntime>(
                    client, count, dim, out, lhs_arg, rhs_arg,
                );
            },
        )
        .map(Tensor::from_typed::<f32>),
        (DType::F64, DType::F64) if lhs.shape() != rhs.shape() => launch_scalar_binary(
            backend,
            typed_or_unsupported::<f64>(lhs, op)?,
            typed_or_unsupported::<f64>(rhs, op)?,
            op,
            |client, count, dim, out, lhs_arg, rhs_arg, lhs_scalar| unsafe {
                elementwise::scalar_rem_float::launch_unchecked::<f64, CubeclCudaRuntime>(
                    client, count, dim, out, lhs_arg, rhs_arg, lhs_scalar,
                );
            },
        )
        .map(Tensor::from_typed::<f64>),
        (DType::F64, DType::F64) => launch_binary(
            backend.runtime(),
            typed_or_unsupported::<f64>(lhs, op)?,
            typed_or_unsupported::<f64>(rhs, op)?,
            lhs.shape(),
            op,
            |client, count, dim, out, lhs_arg, rhs_arg| unsafe {
                elementwise::rem_float::launch_unchecked::<f64, CubeclCudaRuntime>(
                    client, count, dim, out, lhs_arg, rhs_arg,
                );
            },
        )
        .map(Tensor::from_typed::<f64>),
        (DType::I32, DType::I32) if lhs.shape() != rhs.shape() => {
            launch_checked_integer_scalar_binary(
                backend,
                typed_or_unsupported::<i32>(lhs, op)?,
                typed_or_unsupported::<i32>(rhs, op)?,
                op,
                crate::DType::I32,
                CheckedIntegerDomain::DivisionByZero,
                |client, count, dim, out, lhs_arg, rhs_arg, err_arg, lhs_scalar| unsafe {
                    elementwise::scalar_rem_int_checked::launch_unchecked::<i32, CubeclCudaRuntime>(
                        client, count, dim, out, lhs_arg, rhs_arg, err_arg, lhs_scalar,
                    );
                },
            )
            .map(Tensor::from_typed::<i32>)
        }
        (DType::I32, DType::I32) => launch_checked_integer_binary(
            backend,
            typed_or_unsupported::<i32>(lhs, op)?,
            typed_or_unsupported::<i32>(rhs, op)?,
            op,
            crate::DType::I32,
            CheckedIntegerDomain::DivisionByZero,
            |client, count, dim, out, lhs_arg, rhs_arg, err_arg| unsafe {
                elementwise::rem_int_checked::launch_unchecked::<i32, CubeclCudaRuntime>(
                    client, count, dim, out, lhs_arg, rhs_arg, err_arg,
                );
            },
        )
        .map(Tensor::from_typed::<i32>),
        (DType::I64, DType::I64) if lhs.shape() != rhs.shape() => {
            launch_checked_integer_scalar_binary(
                backend,
                typed_or_unsupported::<i64>(lhs, op)?,
                typed_or_unsupported::<i64>(rhs, op)?,
                op,
                crate::DType::I64,
                CheckedIntegerDomain::DivisionByZero,
                |client, count, dim, out, lhs_arg, rhs_arg, err_arg, lhs_scalar| unsafe {
                    elementwise::scalar_rem_int_checked::launch_unchecked::<i64, CubeclCudaRuntime>(
                        client, count, dim, out, lhs_arg, rhs_arg, err_arg, lhs_scalar,
                    );
                },
            )
            .map(Tensor::from_typed::<i64>)
        }
        (DType::I64, DType::I64) => launch_checked_integer_binary(
            backend,
            typed_or_unsupported::<i64>(lhs, op)?,
            typed_or_unsupported::<i64>(rhs, op)?,
            op,
            crate::DType::I64,
            CheckedIntegerDomain::DivisionByZero,
            |client, count, dim, out, lhs_arg, rhs_arg, err_arg| unsafe {
                elementwise::rem_int_checked::launch_unchecked::<i64, CubeclCudaRuntime>(
                    client, count, dim, out, lhs_arg, rhs_arg, err_arg,
                );
            },
        )
        .map(Tensor::from_typed::<i64>),
        (DType::C32, DType::C32) | (DType::C64, DType::C64) => {
            Err(unsupported_dtype(op, lhs.dtype()))
        }
        _ => Err(dtype_mismatch(op, lhs, rhs)),
    }
}

pub(super) fn exp_read(backend: &mut CudaBackend, input: TensorRead<'_>) -> crate::Result<Tensor> {
    if let Some(result) = backend.unary_read_native(UnaryReadOp::Exp, input.clone()) {
        return result;
    }
    let input = backend.read_input(input)?;
    dispatch::dispatch_unary_float_only!(
        backend,
        input.as_tensor(),
        PrimitiveOpKind::Exp,
        exp_float
    )
}

pub(super) fn log_read(backend: &mut CudaBackend, input: TensorRead<'_>) -> crate::Result<Tensor> {
    if let Some(result) = backend.unary_read_native(UnaryReadOp::Log, input.clone()) {
        return result;
    }
    let input = backend.read_input(input)?;
    dispatch::dispatch_unary_float_only!(
        backend,
        input.as_tensor(),
        PrimitiveOpKind::Log,
        log_float
    )
}

pub(super) fn sin_read(backend: &mut CudaBackend, input: TensorRead<'_>) -> crate::Result<Tensor> {
    if let Some(result) = backend.unary_read_native(UnaryReadOp::Sin, input.clone()) {
        return result;
    }
    let input = backend.read_input(input)?;
    dispatch::dispatch_unary_float_only!(
        backend,
        input.as_tensor(),
        PrimitiveOpKind::Sin,
        sin_float
    )
}

pub(super) fn cos_read(backend: &mut CudaBackend, input: TensorRead<'_>) -> crate::Result<Tensor> {
    if let Some(result) = backend.unary_read_native(UnaryReadOp::Cos, input.clone()) {
        return result;
    }
    let input = backend.read_input(input)?;
    dispatch::dispatch_unary_float_only!(
        backend,
        input.as_tensor(),
        PrimitiveOpKind::Cos,
        cos_float
    )
}

pub(super) fn tanh_read(backend: &mut CudaBackend, input: TensorRead<'_>) -> crate::Result<Tensor> {
    if let Some(result) = backend.unary_read_native(UnaryReadOp::Tanh, input.clone()) {
        return result;
    }
    let input = backend.read_input(input)?;
    dispatch::dispatch_unary_float_only!(
        backend,
        input.as_tensor(),
        PrimitiveOpKind::Tanh,
        tanh_float
    )
}

pub(super) fn sqrt_read(backend: &mut CudaBackend, input: TensorRead<'_>) -> crate::Result<Tensor> {
    if let Some(result) = backend.unary_read_native(UnaryReadOp::Sqrt, input.clone()) {
        return result;
    }
    let input = backend.read_input(input)?;
    dispatch::dispatch_unary_float_only!(
        backend,
        input.as_tensor(),
        PrimitiveOpKind::Sqrt,
        sqrt_float
    )
}

pub(super) fn rsqrt_read(
    backend: &mut CudaBackend,
    input: TensorRead<'_>,
) -> crate::Result<Tensor> {
    if let Some(result) = backend.unary_read_native(UnaryReadOp::Rsqrt, input.clone()) {
        return result;
    }
    let input = backend.read_input(input)?;
    dispatch::dispatch_unary_float_only!(
        backend,
        input.as_tensor(),
        PrimitiveOpKind::Rsqrt,
        rsqrt_float
    )
}

pub(super) fn pow_read(
    backend: &mut CudaBackend,
    lhs: TensorRead<'_>,
    rhs: TensorRead<'_>,
) -> crate::Result<Tensor> {
    let lhs = backend.read_input(lhs)?;
    let rhs = backend.read_input(rhs)?;
    let lhs = lhs.as_tensor();
    let rhs = rhs.as_tensor();
    let op = op_name(
        PrimitiveOpKind::Pow,
        op_descriptor::GpuLaunchKind::BinaryFloatInt,
    )?;
    if lhs.dtype() != rhs.dtype() {
        return Err(dtype_mismatch(op, lhs, rhs));
    }
    match (lhs.dtype(), rhs.dtype()) {
        (DType::F32, DType::F32) if lhs.shape() != rhs.shape() => launch_scalar_binary(
            backend,
            typed_or_unsupported::<f32>(lhs, op)?,
            typed_or_unsupported::<f32>(rhs, op)?,
            op,
            |client, count, dim, out, lhs_arg, rhs_arg, lhs_scalar| unsafe {
                elementwise::scalar_pow_float::launch_unchecked::<f32, CubeclCudaRuntime>(
                    client, count, dim, out, lhs_arg, rhs_arg, lhs_scalar,
                );
            },
        )
        .map(Tensor::from_typed::<f32>),
        (DType::F32, DType::F32) => launch_binary(
            backend.runtime(),
            typed_or_unsupported::<f32>(lhs, op)?,
            typed_or_unsupported::<f32>(rhs, op)?,
            lhs.shape(),
            op,
            |client, count, dim, out, lhs_arg, rhs_arg| unsafe {
                elementwise::pow_float::launch_unchecked::<f32, CubeclCudaRuntime>(
                    client, count, dim, out, lhs_arg, rhs_arg,
                );
            },
        )
        .map(Tensor::from_typed::<f32>),
        (DType::F64, DType::F64) if lhs.shape() != rhs.shape() => launch_scalar_binary(
            backend,
            typed_or_unsupported::<f64>(lhs, op)?,
            typed_or_unsupported::<f64>(rhs, op)?,
            op,
            |client, count, dim, out, lhs_arg, rhs_arg, lhs_scalar| unsafe {
                elementwise::scalar_pow_float::launch_unchecked::<f64, CubeclCudaRuntime>(
                    client, count, dim, out, lhs_arg, rhs_arg, lhs_scalar,
                );
            },
        )
        .map(Tensor::from_typed::<f64>),
        (DType::F64, DType::F64) => launch_binary(
            backend.runtime(),
            typed_or_unsupported::<f64>(lhs, op)?,
            typed_or_unsupported::<f64>(rhs, op)?,
            lhs.shape(),
            op,
            |client, count, dim, out, lhs_arg, rhs_arg| unsafe {
                elementwise::pow_float::launch_unchecked::<f64, CubeclCudaRuntime>(
                    client, count, dim, out, lhs_arg, rhs_arg,
                );
            },
        )
        .map(Tensor::from_typed::<f64>),
        (DType::I32, DType::I32) if lhs.shape() != rhs.shape() => {
            launch_checked_integer_scalar_binary(
                backend,
                typed_or_unsupported::<i32>(lhs, op)?,
                typed_or_unsupported::<i32>(rhs, op)?,
                op,
                crate::DType::I32,
                CheckedIntegerDomain::NegativeExponent,
                |client, count, dim, out, lhs_arg, rhs_arg, err_arg, lhs_scalar| unsafe {
                    elementwise::scalar_pow_int_checked::launch_unchecked::<i32, CubeclCudaRuntime>(
                        client, count, dim, out, lhs_arg, rhs_arg, err_arg, lhs_scalar,
                    );
                },
            )
            .map(Tensor::from_typed::<i32>)
        }
        (DType::I32, DType::I32) => launch_checked_integer_binary(
            backend,
            typed_or_unsupported::<i32>(lhs, op)?,
            typed_or_unsupported::<i32>(rhs, op)?,
            op,
            crate::DType::I32,
            CheckedIntegerDomain::NegativeExponent,
            |client, count, dim, out, lhs_arg, rhs_arg, err_arg| unsafe {
                elementwise::pow_int_checked::launch_unchecked::<i32, CubeclCudaRuntime>(
                    client, count, dim, out, lhs_arg, rhs_arg, err_arg,
                );
            },
        )
        .map(Tensor::from_typed::<i32>),
        (DType::I64, DType::I64) if lhs.shape() != rhs.shape() => {
            launch_checked_integer_scalar_binary(
                backend,
                typed_or_unsupported::<i64>(lhs, op)?,
                typed_or_unsupported::<i64>(rhs, op)?,
                op,
                crate::DType::I64,
                CheckedIntegerDomain::NegativeExponent,
                |client, count, dim, out, lhs_arg, rhs_arg, err_arg, lhs_scalar| unsafe {
                    elementwise::scalar_pow_int_checked::launch_unchecked::<i64, CubeclCudaRuntime>(
                        client, count, dim, out, lhs_arg, rhs_arg, err_arg, lhs_scalar,
                    );
                },
            )
            .map(Tensor::from_typed::<i64>)
        }
        (DType::I64, DType::I64) => launch_checked_integer_binary(
            backend,
            typed_or_unsupported::<i64>(lhs, op)?,
            typed_or_unsupported::<i64>(rhs, op)?,
            op,
            crate::DType::I64,
            CheckedIntegerDomain::NegativeExponent,
            |client, count, dim, out, lhs_arg, rhs_arg, err_arg| unsafe {
                elementwise::pow_int_checked::launch_unchecked::<i64, CubeclCudaRuntime>(
                    client, count, dim, out, lhs_arg, rhs_arg, err_arg,
                );
            },
        )
        .map(Tensor::from_typed::<i64>),
        (DType::C32, DType::C32) => {
            dispatch::ensure_same_shape(op, lhs.shape(), rhs.shape())?;
            Err(unsupported_dtype(op, crate::DType::C32))
        }
        (DType::C64, DType::C64) => {
            dispatch::ensure_same_shape(op, lhs.shape(), rhs.shape())?;
            Err(unsupported_dtype(op, crate::DType::C64))
        }
        _ => {
            dispatch::ensure_same_shape(op, lhs.shape(), rhs.shape())?;
            Err(dtype_mismatch(op, lhs, rhs))
        }
    }
}

pub(super) fn expm1_read(
    backend: &mut CudaBackend,
    input: TensorRead<'_>,
) -> crate::Result<Tensor> {
    if let Some(result) = backend.unary_read_native(UnaryReadOp::Expm1, input.clone()) {
        return result;
    }
    let input = backend.read_input(input)?;
    dispatch::dispatch_unary_float_only!(
        backend,
        input.as_tensor(),
        PrimitiveOpKind::Expm1,
        expm1_float
    )
}

pub(super) fn log1p_read(
    backend: &mut CudaBackend,
    input: TensorRead<'_>,
) -> crate::Result<Tensor> {
    if let Some(result) = backend.unary_read_native(UnaryReadOp::Log1p, input.clone()) {
        return result;
    }
    let input = backend.read_input(input)?;
    dispatch::dispatch_unary_float_only!(
        backend,
        input.as_tensor(),
        PrimitiveOpKind::Log1p,
        log1p_float
    )
}

pub(super) fn transpose_read(
    backend: &mut CudaBackend,
    input: TensorRead<'_>,
    perm: &[usize],
) -> crate::Result<Tensor> {
    let input = backend.read_input(input)?;
    let input = input.as_tensor();
    match input.dtype() {
        DType::F32 => {
            let t = typed_or_unsupported::<f32>(input, "transpose")?;
            permutation::transpose(backend, t, perm).map(Tensor::from_typed::<f32>)
        }
        DType::F64 => {
            let t = typed_or_unsupported::<f64>(input, "transpose")?;
            permutation::transpose(backend, t, perm).map(Tensor::from_typed::<f64>)
        }
        DType::I32 => {
            let t = typed_or_unsupported::<i32>(input, "transpose")?;
            backend
                .transpose_typed(t, perm)
                .map(Tensor::from_typed::<i32>)
        }
        DType::I64 => {
            let t = typed_or_unsupported::<i64>(input, "transpose")?;
            backend
                .transpose_typed(t, perm)
                .map(Tensor::from_typed::<i64>)
        }
        DType::Bool => {
            let t = typed_or_unsupported::<bool>(input, "transpose")?;
            backend
                .transpose_bool(t, perm)
                .map(Tensor::from_typed::<bool>)
        }
        DType::C32 => {
            let t = typed_or_unsupported::<Complex32>(input, "transpose")?;
            permutation::transpose(backend, t, perm)
                .map(Tensor::from_typed::<num_complex::Complex32>)
        }
        DType::C64 => {
            let t = typed_or_unsupported::<Complex64>(input, "transpose")?;
            permutation::transpose(backend, t, perm)
                .map(Tensor::from_typed::<num_complex::Complex64>)
        }
        // A caller-owned payload has no GPU implementation for this operation.
        DType::External(_) => Err(crate::Error::unsupported(
            "transpose",
            "an externally defined payload is not supported by this GPU operation",
        )),
    }
}

pub(super) fn reshape_read(
    backend: &mut CudaBackend,
    input: TensorRead<'_>,
    shape: &[usize],
) -> crate::Result<Tensor> {
    let input = backend.read_input(input)?;
    let input = input.as_tensor();
    let old_n = checked_dim_product("reshape", "input shape", input.shape())?;
    let new_n = checked_dim_product("reshape", "output shape", shape)?;
    if old_n != new_n {
        return Err(crate::Error::validation(
            "reshape",
            tenferro_tensor::ShapeMismatch::ReshapeElementCount {
                from: old_n,
                to: new_n,
            }
            .into(),
        ));
    }
    // An owned tensor cannot be returned by shallowly reusing a backend
    // buffer. Materialize one explicit same-placement copy first, then
    // change only its compact metadata.
    let contiguous = match input.dtype() {
        DType::Bool => backend
            .duplicate_bool(
                input.as_typed::<bool>().ok_or_else(|| {
                    crate::Error::unsupported(
                        "reshape",
                        "an externally defined payload is not supported by this GPU operation",
                    )
                })?,
                "reshape",
            )
            .map(Tensor::from_typed::<bool>)?,
        // Every other tag is materialized through the read path, which refuses the
        // externally defined payload itself.
        _ => ops::to_contiguous_read(backend, TensorRead::from_tensor(input))?,
    };
    match contiguous.dtype() {
        DType::F32 => {
            cubecl_reshape_metadata(contiguous.into_typed::<f32>()?, shape.to_vec(), "reshape")
                .map(Tensor::from_typed::<f32>)
        }
        DType::F64 => {
            cubecl_reshape_metadata(contiguous.into_typed::<f64>()?, shape.to_vec(), "reshape")
                .map(Tensor::from_typed::<f64>)
        }
        DType::I32 => {
            cubecl_reshape_metadata(contiguous.into_typed::<i32>()?, shape.to_vec(), "reshape")
                .map(Tensor::from_typed::<i32>)
        }
        DType::I64 => {
            cubecl_reshape_metadata(contiguous.into_typed::<i64>()?, shape.to_vec(), "reshape")
                .map(Tensor::from_typed::<i64>)
        }
        DType::Bool => {
            cubecl_reshape_metadata(contiguous.into_typed::<bool>()?, shape.to_vec(), "reshape")
                .map(Tensor::from_typed::<bool>)
        }
        DType::C32 => cubecl_reshape_metadata(
            contiguous.into_typed::<Complex32>()?,
            shape.to_vec(),
            "reshape",
        )
        .map(Tensor::from_typed::<num_complex::Complex32>),
        DType::C64 => cubecl_reshape_metadata(
            contiguous.into_typed::<Complex64>()?,
            shape.to_vec(),
            "reshape",
        )
        .map(Tensor::from_typed::<num_complex::Complex64>),
        // A caller-owned payload has no GPU implementation for this operation.
        DType::External(_) => Err(crate::Error::unsupported(
            "reshape",
            "an externally defined payload is not supported by this GPU operation",
        )),
    }
}

pub(super) fn broadcast_in_dim_read(
    backend: &mut CudaBackend,
    input: TensorRead<'_>,
    shape: &[usize],
    dims: &[usize],
) -> crate::Result<Tensor> {
    let input = backend.read_input(input)?;
    let input = input.as_tensor();
    match input.dtype() {
        DType::F32 => {
            let t = typed_or_unsupported::<f32>(input, "broadcast_in_dim")?;
            backend
                .broadcast_typed(t, shape, dims)
                .map(Tensor::from_typed::<f32>)
        }
        DType::F64 => {
            let t = typed_or_unsupported::<f64>(input, "broadcast_in_dim")?;
            backend
                .broadcast_typed(t, shape, dims)
                .map(Tensor::from_typed::<f64>)
        }
        DType::I32 => {
            let t = typed_or_unsupported::<i32>(input, "broadcast_in_dim")?;
            backend
                .broadcast_typed(t, shape, dims)
                .map(Tensor::from_typed::<i32>)
        }
        DType::I64 => {
            let t = typed_or_unsupported::<i64>(input, "broadcast_in_dim")?;
            backend
                .broadcast_typed(t, shape, dims)
                .map(Tensor::from_typed::<i64>)
        }
        DType::Bool => {
            let t = typed_or_unsupported::<bool>(input, "broadcast_in_dim")?;
            backend
                .broadcast_bool(t, shape, dims)
                .map(Tensor::from_typed::<bool>)
        }
        DType::C32 => {
            let t = typed_or_unsupported::<Complex32>(input, "broadcast_in_dim")?;
            backend
                .broadcast_typed(t, shape, dims)
                .map(Tensor::from_typed::<num_complex::Complex32>)
        }
        DType::C64 => {
            let t = typed_or_unsupported::<Complex64>(input, "broadcast_in_dim")?;
            backend
                .broadcast_typed(t, shape, dims)
                .map(Tensor::from_typed::<num_complex::Complex64>)
        }
        // A caller-owned payload has no GPU implementation for this operation.
        DType::External(_) => Err(crate::Error::unsupported(
            "broadcast_in_dim",
            "an externally defined payload is not supported by this GPU operation",
        )),
    }
}

pub(super) fn to_contiguous_read(
    backend: &mut CudaBackend,
    input: TensorRead<'_>,
) -> crate::Result<Tensor> {
    macro_rules! materialize_cutensor {
        ($variant:ident, $view:expr) => {{
            let view = $view;
            backend
                .to_contiguous_view_cutensor_or_cubecl(&view, "CudaBackend::to_contiguous_read")
                .map(Tensor::from_typed::<preset_scalar!($variant)>)
        }};
    }
    macro_rules! materialize_cubecl {
        ($variant:ident, $view:expr) => {{
            let view = $view;
            backend
                .to_contiguous_view_typed(&view, "CudaBackend::to_contiguous_read")
                .map(Tensor::from_typed::<preset_scalar!($variant)>)
        }};
    }

    match input {
        TensorRead::Tensor(tensor) => match tensor.dtype() {
            DType::F32 => {
                materialize_cutensor!(F32, contiguous_read_typed::<f32>(tensor)?.as_view())
            }
            DType::F64 => {
                materialize_cutensor!(F64, contiguous_read_typed::<f64>(tensor)?.as_view())
            }
            DType::I32 => {
                materialize_cubecl!(I32, contiguous_read_typed::<i32>(tensor)?.as_view())
            }
            DType::I64 => {
                materialize_cubecl!(I64, contiguous_read_typed::<i64>(tensor)?.as_view())
            }
            DType::Bool => Err(unsupported_dtype(
                "CudaBackend::to_contiguous_read",
                crate::DType::Bool,
            )),
            DType::C32 => {
                materialize_cutensor!(C32, contiguous_read_typed::<Complex32>(tensor)?.as_view())
            }
            DType::C64 => {
                materialize_cutensor!(C64, contiguous_read_typed::<Complex64>(tensor)?.as_view())
            }
            // A caller-owned payload has no GPU implementation for this operation.
            DType::External(_) => Err(crate::Error::unsupported(
                "CudaBackend::to_contiguous_read",
                "an externally defined payload is not supported by this GPU operation",
            )),
        },
        TensorRead::View(TensorView::F32(input)) => materialize_cutensor!(F32, input),
        TensorRead::View(TensorView::F64(input)) => materialize_cutensor!(F64, input),
        TensorRead::View(TensorView::I32(input)) => materialize_cubecl!(I32, input),
        TensorRead::View(TensorView::I64(input)) => materialize_cubecl!(I64, input),
        TensorRead::View(TensorView::Bool(_)) => Err(unsupported_dtype(
            "CudaBackend::to_contiguous_read",
            crate::DType::Bool,
        )),
        TensorRead::View(TensorView::C32(input)) => materialize_cutensor!(C32, input),
        TensorRead::View(TensorView::C64(input)) => materialize_cutensor!(C64, input),
    }
}

pub(super) fn copy_read_into(
    backend: &mut CudaBackend,
    src: TensorRead<'_>,
    dst: TensorWrite<'_>,
) -> crate::Result<()> {
    let src_dtype = src.dtype();
    let dst_dtype = dst.dtype();
    macro_rules! copy_source_typed {
        ($variant:ident, $src:expr) => {{
            let src = $src;
            match dst {
                TensorWrite::Tensor(dst)
                    if dst.dtype()
                        == <preset_scalar!($variant) as tenferro_tensor::TensorScalar>::dtype() =>
                {
                    let dst = dst
                        .as_typed_mut::<preset_scalar!($variant)>()
                        .expect("the dtype guard selects this arm");
                    let mut dst = dst.as_view_mut();
                    backend.copy_view_to_view_typed(&src, &mut dst, "CudaBackend::copy_read_into")
                }
                TensorWrite::View(TensorViewMut::$variant(mut dst)) => {
                    backend.copy_view_to_view_typed(&src, &mut dst, "CudaBackend::copy_read_into")
                }
                _ => Err(crate::Error::dtype_mismatch(
                    "CudaBackend::copy_read_into",
                    src_dtype,
                    dst_dtype,
                )),
            }
        }};
    }
    // Every numeric dtype uses the cuTENSOR permutation path, which is the
    // bandwidth-bound optimum for a multi-axis permutation destination.
    // Complex dtypes are planned through their real view with a real
    // `alpha = 1`, so the scaling multiply is exact and cannot turn
    // `(inf, finite)` into `(inf, NaN)` (issue #1891).
    macro_rules! copy_source_cutensor {
        ($variant:ident, $src:expr) => {{
            let src = $src;
            match dst {
                TensorWrite::Tensor(dst)
                    if dst.dtype()
                        == <preset_scalar!($variant) as tenferro_tensor::TensorScalar>::dtype() =>
                {
                    let dst = dst
                        .as_typed_mut::<preset_scalar!($variant)>()
                        .expect("the dtype guard selects this arm");
                    let mut dst = dst.as_view_mut();
                    backend.copy_view_to_view_cutensor_or_cubecl(
                        &src,
                        &mut dst,
                        "CudaBackend::copy_read_into",
                    )
                }
                TensorWrite::View(TensorViewMut::$variant(mut dst)) => backend
                    .copy_view_to_view_cutensor_or_cubecl(
                        &src,
                        &mut dst,
                        "CudaBackend::copy_read_into",
                    ),
                _ => Err(crate::Error::dtype_mismatch(
                    "CudaBackend::copy_read_into",
                    src_dtype,
                    dst_dtype,
                )),
            }
        }};
    }
    macro_rules! reject_bool_source {
        () => {{
            match dst {
                TensorWrite::Tensor(tensor) if tensor.dtype() == crate::DType::Bool => Err(
                    unsupported_dtype("CudaBackend::copy_read_into", crate::DType::Bool),
                ),
                TensorWrite::View(TensorViewMut::Bool(_)) => Err(unsupported_dtype(
                    "CudaBackend::copy_read_into",
                    crate::DType::Bool,
                )),
                _ => Err(crate::Error::dtype_mismatch(
                    "CudaBackend::copy_read_into",
                    src_dtype,
                    dst_dtype,
                )),
            }
        }};
    }

    match src {
        TensorRead::Tensor(tensor) => match tensor.dtype() {
            DType::F32 => copy_source_cutensor!(F32, copy_read_typed::<f32>(tensor)?.as_view()),
            DType::F64 => copy_source_cutensor!(F64, copy_read_typed::<f64>(tensor)?.as_view()),
            DType::I32 => copy_source_typed!(I32, copy_read_typed::<i32>(tensor)?.as_view()),
            DType::I64 => copy_source_typed!(I64, copy_read_typed::<i64>(tensor)?.as_view()),
            DType::Bool => reject_bool_source!(),
            DType::C32 => {
                copy_source_cutensor!(C32, copy_read_typed::<Complex32>(tensor)?.as_view())
            }
            DType::C64 => {
                copy_source_cutensor!(C64, copy_read_typed::<Complex64>(tensor)?.as_view())
            }
            // A caller-owned payload has no GPU implementation for this operation.
            DType::External(_) => Err(crate::Error::unsupported(
                "copy_read_into",
                "an externally defined payload is not supported by this GPU operation",
            )),
        },
        TensorRead::View(TensorView::F32(src)) => copy_source_cutensor!(F32, src),
        TensorRead::View(TensorView::F64(src)) => copy_source_cutensor!(F64, src),
        TensorRead::View(TensorView::I32(src)) => copy_source_typed!(I32, src),
        TensorRead::View(TensorView::I64(src)) => copy_source_typed!(I64, src),
        TensorRead::View(TensorView::Bool(_)) => reject_bool_source!(),
        TensorRead::View(TensorView::C32(src)) => copy_source_cutensor!(C32, src),
        TensorRead::View(TensorView::C64(src)) => copy_source_cutensor!(C64, src),
    }
}

pub(super) fn cast(
    backend: &mut CudaBackend,
    input: &Tensor,
    to: crate::DType,
) -> crate::Result<Tensor> {
    match (input.dtype(), to) {
        // An externally defined destination has no CUDA conversion, so the
        // backend rejects it instead of guessing a representation.
        (_, crate::DType::External(_)) => Err(crate::Error::unsupported(
            "cast",
            "an externally defined scalar has no CUDA conversion",
        )),
        (DType::F32, crate::DType::F32) => backend
            .duplicate_typed(typed_or_unsupported::<f32>(input, "cast")?)
            .map(Tensor::from_typed::<f32>),
        (DType::F64, crate::DType::F64) => backend
            .duplicate_typed(typed_or_unsupported::<f64>(input, "cast")?)
            .map(Tensor::from_typed::<f64>),
        (DType::I32, crate::DType::I32) => backend
            .duplicate_typed(typed_or_unsupported::<i32>(input, "cast")?)
            .map(Tensor::from_typed::<i32>),
        (DType::I64, crate::DType::I64) => backend
            .duplicate_typed(typed_or_unsupported::<i64>(input, "cast")?)
            .map(Tensor::from_typed::<i64>),
        (DType::Bool, crate::DType::Bool) => backend
            .duplicate_bool(typed_or_unsupported::<bool>(input, "cast")?, "cast")
            .map(Tensor::from_typed::<bool>),
        (DType::C32, crate::DType::C32) => backend
            .duplicate_typed(typed_or_unsupported::<Complex32>(input, "cast")?)
            .map(Tensor::from_typed::<num_complex::Complex32>),
        (DType::C64, crate::DType::C64) => backend
            .duplicate_typed(typed_or_unsupported::<Complex64>(input, "cast")?)
            .map(Tensor::from_typed::<num_complex::Complex64>),
        (DType::F32, crate::DType::F64) => backend
            .convert_float_to_float::<f32, f64>(typed_or_unsupported::<f32>(input, "cast")?)
            .map(Tensor::from_typed::<f64>),
        (DType::F32, crate::DType::I32) => {
            validate_cuda_real_cast::<f32, f32>(
                backend,
                typed_or_unsupported::<f32>(input, "cast")?,
                1,
                CastIntegerTarget::I32,
            )?;
            backend
                .convert_numeric::<f32, i32>(typed_or_unsupported::<f32>(input, "cast")?)
                .map(Tensor::from_typed::<i32>)
        }
        (DType::F32, crate::DType::I64) => {
            validate_cuda_real_cast::<f32, f32>(
                backend,
                typed_or_unsupported::<f32>(input, "cast")?,
                1,
                CastIntegerTarget::I64,
            )?;
            backend
                .convert_numeric::<f32, i64>(typed_or_unsupported::<f32>(input, "cast")?)
                .map(Tensor::from_typed::<i64>)
        }
        (DType::F32, crate::DType::Bool) => backend
            .convert_numeric_to_bool(typed_or_unsupported::<f32>(input, "cast")?)
            .map(Tensor::from_typed::<bool>),
        (DType::F32, crate::DType::C32) => backend
            .convert_f32_to_c32(typed_or_unsupported::<f32>(input, "cast")?)
            .map(Tensor::from_typed::<num_complex::Complex32>),
        (DType::F32, crate::DType::C64) => backend
            .convert_f32_to_c64(typed_or_unsupported::<f32>(input, "cast")?)
            .map(Tensor::from_typed::<num_complex::Complex64>),
        (DType::F64, crate::DType::F32) => backend
            .convert_float_to_float::<f64, f32>(typed_or_unsupported::<f64>(input, "cast")?)
            .map(Tensor::from_typed::<f32>),
        (DType::F64, crate::DType::I32) => {
            validate_cuda_real_cast::<f64, f64>(
                backend,
                typed_or_unsupported::<f64>(input, "cast")?,
                1,
                CastIntegerTarget::I32,
            )?;
            backend
                .convert_numeric::<f64, i32>(typed_or_unsupported::<f64>(input, "cast")?)
                .map(Tensor::from_typed::<i32>)
        }
        (DType::F64, crate::DType::I64) => {
            validate_cuda_real_cast::<f64, f64>(
                backend,
                typed_or_unsupported::<f64>(input, "cast")?,
                1,
                CastIntegerTarget::I64,
            )?;
            backend
                .convert_numeric::<f64, i64>(typed_or_unsupported::<f64>(input, "cast")?)
                .map(Tensor::from_typed::<i64>)
        }
        (DType::F64, crate::DType::Bool) => backend
            .convert_numeric_to_bool(typed_or_unsupported::<f64>(input, "cast")?)
            .map(Tensor::from_typed::<bool>),
        (DType::F64, crate::DType::C32) => backend
            .convert_f64_to_c32(typed_or_unsupported::<f64>(input, "cast")?)
            .map(Tensor::from_typed::<num_complex::Complex32>),
        (DType::F64, crate::DType::C64) => backend
            .convert_f64_to_c64(typed_or_unsupported::<f64>(input, "cast")?)
            .map(Tensor::from_typed::<num_complex::Complex64>),
        (DType::I32, crate::DType::F32) => backend
            .convert_numeric::<i32, f32>(typed_or_unsupported::<i32>(input, "cast")?)
            .map(Tensor::from_typed::<f32>),
        (DType::I32, crate::DType::F64) => backend
            .convert_numeric::<i32, f64>(typed_or_unsupported::<i32>(input, "cast")?)
            .map(Tensor::from_typed::<f64>),
        (DType::I32, crate::DType::I64) => backend
            .convert_numeric::<i32, i64>(typed_or_unsupported::<i32>(input, "cast")?)
            .map(Tensor::from_typed::<i64>),
        (DType::I32, crate::DType::Bool) => backend
            .convert_numeric_to_bool(typed_or_unsupported::<i32>(input, "cast")?)
            .map(Tensor::from_typed::<bool>),
        (DType::I32, crate::DType::C32) => backend
            .convert_numeric_to_complex::<i32, Complex32, f32>(typed_or_unsupported::<i32>(
                input, "cast",
            )?)
            .map(Tensor::from_typed::<num_complex::Complex32>),
        (DType::I32, crate::DType::C64) => backend
            .convert_numeric_to_complex::<i32, Complex64, f64>(typed_or_unsupported::<i32>(
                input, "cast",
            )?)
            .map(Tensor::from_typed::<num_complex::Complex64>),
        (DType::I64, crate::DType::F32) => backend
            .convert_numeric::<i64, f32>(typed_or_unsupported::<i64>(input, "cast")?)
            .map(Tensor::from_typed::<f32>),
        (DType::I64, crate::DType::F64) => backend
            .convert_numeric::<i64, f64>(typed_or_unsupported::<i64>(input, "cast")?)
            .map(Tensor::from_typed::<f64>),
        (DType::I64, crate::DType::I32) => backend
            .convert_numeric::<i64, i32>(typed_or_unsupported::<i64>(input, "cast")?)
            .map(Tensor::from_typed::<i32>),
        (DType::I64, crate::DType::Bool) => backend
            .convert_numeric_to_bool(typed_or_unsupported::<i64>(input, "cast")?)
            .map(Tensor::from_typed::<bool>),
        (DType::I64, crate::DType::C32) => backend
            .convert_numeric_to_complex::<i64, Complex32, f32>(typed_or_unsupported::<i64>(
                input, "cast",
            )?)
            .map(Tensor::from_typed::<num_complex::Complex32>),
        (DType::I64, crate::DType::C64) => backend
            .convert_numeric_to_complex::<i64, Complex64, f64>(typed_or_unsupported::<i64>(
                input, "cast",
            )?)
            .map(Tensor::from_typed::<num_complex::Complex64>),
        (DType::Bool, crate::DType::F32) => backend
            .convert_bool_to_numeric::<f32>(typed_or_unsupported::<bool>(input, "cast")?)
            .map(Tensor::from_typed::<f32>),
        (DType::Bool, crate::DType::F64) => backend
            .convert_bool_to_numeric::<f64>(typed_or_unsupported::<bool>(input, "cast")?)
            .map(Tensor::from_typed::<f64>),
        (DType::Bool, crate::DType::I32) => backend
            .convert_bool_to_numeric::<i32>(typed_or_unsupported::<bool>(input, "cast")?)
            .map(Tensor::from_typed::<i32>),
        (DType::Bool, crate::DType::I64) => backend
            .convert_bool_to_numeric::<i64>(typed_or_unsupported::<bool>(input, "cast")?)
            .map(Tensor::from_typed::<i64>),
        (DType::Bool, crate::DType::C32) => backend
            .convert_bool_to_complex::<Complex32, f32>(typed_or_unsupported::<bool>(input, "cast")?)
            .map(Tensor::from_typed::<num_complex::Complex32>),
        (DType::Bool, crate::DType::C64) => backend
            .convert_bool_to_complex::<Complex64, f64>(typed_or_unsupported::<bool>(input, "cast")?)
            .map(Tensor::from_typed::<num_complex::Complex64>),
        (DType::C32, crate::DType::F32) => backend
            .convert_c32_to_f32(typed_or_unsupported::<Complex32>(input, "cast")?)
            .map(Tensor::from_typed::<f32>),
        (DType::C32, crate::DType::F64) => backend
            .convert_c32_to_f64(typed_or_unsupported::<Complex32>(input, "cast")?)
            .map(Tensor::from_typed::<f64>),
        (DType::C32, crate::DType::I32) => {
            validate_cuda_real_cast::<Complex32, f32>(
                backend,
                typed_or_unsupported::<Complex32>(input, "cast")?,
                2,
                CastIntegerTarget::I32,
            )?;
            backend
                .convert_complex_to_numeric::<Complex32, i32>(typed_or_unsupported::<Complex32>(
                    input, "cast",
                )?)
                .map(Tensor::from_typed::<i32>)
        }
        (DType::C32, crate::DType::I64) => {
            validate_cuda_real_cast::<Complex32, f32>(
                backend,
                typed_or_unsupported::<Complex32>(input, "cast")?,
                2,
                CastIntegerTarget::I64,
            )?;
            backend
                .convert_complex_to_numeric::<Complex32, i64>(typed_or_unsupported::<Complex32>(
                    input, "cast",
                )?)
                .map(Tensor::from_typed::<i64>)
        }
        (DType::C32, crate::DType::Bool) => backend
            .convert_complex_to_bool::<Complex32, f32>(typed_or_unsupported::<Complex32>(
                input, "cast",
            )?)
            .map(Tensor::from_typed::<bool>),
        (DType::C32, crate::DType::C64) => {
            backend
                .convert_complex_to_complex::<Complex32, Complex64, f32, f64>(
                    typed_or_unsupported::<Complex32>(input, "cast")?,
                )
                .map(Tensor::from_typed::<num_complex::Complex64>)
        }
        (DType::C64, crate::DType::F32) => backend
            .convert_c64_to_f32(typed_or_unsupported::<Complex64>(input, "cast")?)
            .map(Tensor::from_typed::<f32>),
        (DType::C64, crate::DType::F64) => backend
            .convert_c64_to_f64(typed_or_unsupported::<Complex64>(input, "cast")?)
            .map(Tensor::from_typed::<f64>),
        (DType::C64, crate::DType::I32) => {
            validate_cuda_real_cast::<Complex64, f64>(
                backend,
                typed_or_unsupported::<Complex64>(input, "cast")?,
                2,
                CastIntegerTarget::I32,
            )?;
            backend
                .convert_complex_to_numeric::<Complex64, i32>(typed_or_unsupported::<Complex64>(
                    input, "cast",
                )?)
                .map(Tensor::from_typed::<i32>)
        }
        (DType::C64, crate::DType::I64) => {
            validate_cuda_real_cast::<Complex64, f64>(
                backend,
                typed_or_unsupported::<Complex64>(input, "cast")?,
                2,
                CastIntegerTarget::I64,
            )?;
            backend
                .convert_complex_to_numeric::<Complex64, i64>(typed_or_unsupported::<Complex64>(
                    input, "cast",
                )?)
                .map(Tensor::from_typed::<i64>)
        }
        (DType::C64, crate::DType::Bool) => backend
            .convert_complex_to_bool::<Complex64, f64>(typed_or_unsupported::<Complex64>(
                input, "cast",
            )?)
            .map(Tensor::from_typed::<bool>),
        (DType::C64, crate::DType::C32) => {
            backend
                .convert_complex_to_complex::<Complex64, Complex32, f64, f32>(
                    typed_or_unsupported::<Complex64>(input, "cast")?,
                )
                .map(Tensor::from_typed::<num_complex::Complex32>)
        }
        // A caller-owned payload has no GPU implementation for this operation.
        (DType::External(_), _) => Err(crate::Error::unsupported(
            "cast",
            "an externally defined payload is not supported by this GPU operation",
        )),
    }
}

pub(super) fn extract_diagonal(
    backend: &mut CudaBackend,
    input: &Tensor,
    axis_a: usize,
    axis_b: usize,
) -> crate::Result<Tensor> {
    // Dispatch on the tag and recover the typed tensor, which is what `as_typed` exists for.
    match input.dtype() {
        DType::F32 => {
            let t = typed_or_unsupported::<f32>(input, "extract_diagonal")?;
            backend
                .extract_diagonal_typed(t, axis_a, axis_b)
                .map(Tensor::from_typed::<f32>)
        }
        DType::F64 => {
            let t = typed_or_unsupported::<f64>(input, "extract_diagonal")?;
            backend
                .extract_diagonal_typed(t, axis_a, axis_b)
                .map(Tensor::from_typed::<f64>)
        }
        DType::I32 => {
            let t = typed_or_unsupported::<i32>(input, "extract_diagonal")?;
            backend
                .extract_diagonal_typed(t, axis_a, axis_b)
                .map(Tensor::from_typed::<i32>)
        }
        DType::I64 => {
            let t = typed_or_unsupported::<i64>(input, "extract_diagonal")?;
            backend
                .extract_diagonal_typed(t, axis_a, axis_b)
                .map(Tensor::from_typed::<i64>)
        }
        DType::Bool => {
            let t = typed_or_unsupported::<bool>(input, "extract_diagonal")?;
            backend
                .extract_diagonal_bool(t, axis_a, axis_b)
                .map(Tensor::from_typed::<bool>)
        }
        DType::C32 => {
            let t = typed_or_unsupported::<Complex32>(input, "extract_diagonal")?;
            backend
                .extract_diagonal_typed(t, axis_a, axis_b)
                .map(Tensor::from_typed::<num_complex::Complex32>)
        }
        DType::C64 => {
            let t = typed_or_unsupported::<Complex64>(input, "extract_diagonal")?;
            backend
                .extract_diagonal_typed(t, axis_a, axis_b)
                .map(Tensor::from_typed::<num_complex::Complex64>)
        }
        // A caller-owned payload has no GPU implementation for this operation.
        DType::External(_) => Err(crate::Error::unsupported(
            "extract_diagonal",
            "an externally defined payload is not supported by this GPU operation",
        )),
    }
}

pub(super) fn embed_diagonal(
    backend: &mut CudaBackend,
    input: &Tensor,
    axis_a: usize,
    axis_b: usize,
) -> crate::Result<Tensor> {
    // Dispatch on the tag and recover the typed tensor, which is what `as_typed` exists for.
    match input.dtype() {
        DType::F32 => {
            let t = typed_or_unsupported::<f32>(input, "embed_diagonal")?;
            backend
                .embed_diagonal_typed(t, axis_a, axis_b)
                .map(Tensor::from_typed::<f32>)
        }
        DType::F64 => {
            let t = typed_or_unsupported::<f64>(input, "embed_diagonal")?;
            backend
                .embed_diagonal_typed(t, axis_a, axis_b)
                .map(Tensor::from_typed::<f64>)
        }
        DType::I32 => {
            let t = typed_or_unsupported::<i32>(input, "embed_diagonal")?;
            backend
                .embed_diagonal_typed(t, axis_a, axis_b)
                .map(Tensor::from_typed::<i32>)
        }
        DType::I64 => {
            let t = typed_or_unsupported::<i64>(input, "embed_diagonal")?;
            backend
                .embed_diagonal_typed(t, axis_a, axis_b)
                .map(Tensor::from_typed::<i64>)
        }
        DType::Bool => {
            let t = typed_or_unsupported::<bool>(input, "embed_diagonal")?;
            backend
                .embed_diagonal_bool(t, axis_a, axis_b)
                .map(Tensor::from_typed::<bool>)
        }
        DType::C32 => {
            let t = typed_or_unsupported::<Complex32>(input, "embed_diagonal")?;
            backend
                .embed_diagonal_typed(t, axis_a, axis_b)
                .map(Tensor::from_typed::<num_complex::Complex32>)
        }
        DType::C64 => {
            let t = typed_or_unsupported::<Complex64>(input, "embed_diagonal")?;
            backend
                .embed_diagonal_typed(t, axis_a, axis_b)
                .map(Tensor::from_typed::<num_complex::Complex64>)
        }
        // A caller-owned payload has no GPU implementation for this operation.
        DType::External(_) => Err(crate::Error::unsupported(
            "embed_diagonal",
            "an externally defined payload is not supported by this GPU operation",
        )),
    }
}

pub(super) fn tril(backend: &mut CudaBackend, input: &Tensor, k: i64) -> crate::Result<Tensor> {
    match input.dtype() {
        DType::F32 => {
            let t = typed_or_unsupported::<f32>(input, "tril")?;
            backend.tril_typed(t, k).map(Tensor::from_typed::<f32>)
        }
        DType::F64 => {
            let t = typed_or_unsupported::<f64>(input, "tril")?;
            backend.tril_typed(t, k).map(Tensor::from_typed::<f64>)
        }
        DType::I32 => {
            let t = typed_or_unsupported::<i32>(input, "tril")?;
            backend.tril_typed(t, k).map(Tensor::from_typed::<i32>)
        }
        DType::I64 => {
            let t = typed_or_unsupported::<i64>(input, "tril")?;
            backend.tril_typed(t, k).map(Tensor::from_typed::<i64>)
        }
        DType::Bool => {
            let t = typed_or_unsupported::<bool>(input, "tril")?;
            backend.tril_bool(t, k).map(Tensor::from_typed::<bool>)
        }
        DType::C32 => {
            let t = typed_or_unsupported::<Complex32>(input, "tril")?;
            backend
                .tril_typed(t, k)
                .map(Tensor::from_typed::<num_complex::Complex32>)
        }
        DType::C64 => {
            let t = typed_or_unsupported::<Complex64>(input, "tril")?;
            backend
                .tril_typed(t, k)
                .map(Tensor::from_typed::<num_complex::Complex64>)
        }
        // A caller-owned payload has no GPU implementation for this operation.
        DType::External(_) => Err(crate::Error::unsupported(
            "tril",
            "an externally defined payload is not supported by this GPU operation",
        )),
    }
}

pub(super) fn triu(backend: &mut CudaBackend, input: &Tensor, k: i64) -> crate::Result<Tensor> {
    match input.dtype() {
        DType::F32 => {
            let t = typed_or_unsupported::<f32>(input, "triu")?;
            backend.triu_typed(t, k).map(Tensor::from_typed::<f32>)
        }
        DType::F64 => {
            let t = typed_or_unsupported::<f64>(input, "triu")?;
            backend.triu_typed(t, k).map(Tensor::from_typed::<f64>)
        }
        DType::I32 => {
            let t = typed_or_unsupported::<i32>(input, "triu")?;
            backend.triu_typed(t, k).map(Tensor::from_typed::<i32>)
        }
        DType::I64 => {
            let t = typed_or_unsupported::<i64>(input, "triu")?;
            backend.triu_typed(t, k).map(Tensor::from_typed::<i64>)
        }
        DType::Bool => {
            let t = typed_or_unsupported::<bool>(input, "triu")?;
            backend.triu_bool(t, k).map(Tensor::from_typed::<bool>)
        }
        DType::C32 => {
            let t = typed_or_unsupported::<Complex32>(input, "triu")?;
            backend
                .triu_typed(t, k)
                .map(Tensor::from_typed::<num_complex::Complex32>)
        }
        DType::C64 => {
            let t = typed_or_unsupported::<Complex64>(input, "triu")?;
            backend
                .triu_typed(t, k)
                .map(Tensor::from_typed::<num_complex::Complex64>)
        }
        // A caller-owned payload has no GPU implementation for this operation.
        DType::External(_) => Err(crate::Error::unsupported(
            "triu",
            "an externally defined payload is not supported by this GPU operation",
        )),
    }
}

pub(super) fn reduce_sum_read(
    backend: &mut CudaBackend,
    input: TensorRead<'_>,
    axes: &[usize],
) -> crate::Result<Tensor> {
    let input = backend.read_input(input)?;
    let input = input.as_tensor();
    let op = op_name(
        PrimitiveOpKind::ReduceSum,
        op_descriptor::GpuLaunchKind::Reduction,
    )?;
    match input.dtype() {
        DType::F32 => {
            let t = typed_or_unsupported::<f32>(input, op)?;
            backend
                .reduce_sum_float_typed(t, axes)
                .map(Tensor::from_typed::<f32>)
        }
        DType::F64 => {
            let t = typed_or_unsupported::<f64>(input, op)?;
            backend
                .reduce_sum_float_typed(t, axes)
                .map(Tensor::from_typed::<f64>)
        }
        DType::I32 => {
            let t = typed_or_unsupported::<i32>(input, op)?;
            backend
                .reduce_sum_int_typed(t, axes)
                .map(Tensor::from_typed::<i32>)
        }
        DType::I64 => {
            let t = typed_or_unsupported::<i64>(input, op)?;
            backend
                .reduce_sum_int_typed(t, axes)
                .map(Tensor::from_typed::<i64>)
        }
        DType::Bool => Err(unsupported_dtype(op, input.dtype())),
        DType::C32 => {
            let t = typed_or_unsupported::<Complex32>(input, op)?;
            backend
                .reduce_sum_complex_typed(t, axes)
                .map(Tensor::from_typed::<num_complex::Complex32>)
        }
        DType::C64 => {
            let t = typed_or_unsupported::<Complex64>(input, op)?;
            backend
                .reduce_sum_complex_typed(t, axes)
                .map(Tensor::from_typed::<num_complex::Complex64>)
        }
        // A caller-owned payload has no GPU implementation for this operation.
        DType::External(_) => Err(crate::Error::unsupported(
            "reduce_sum",
            "an externally defined payload is not supported by this GPU operation",
        )),
    }
}

pub(super) fn reduce_prod_read(
    backend: &mut CudaBackend,
    input: TensorRead<'_>,
    axes: &[usize],
) -> crate::Result<Tensor> {
    let input = backend.read_input(input)?;
    let input = input.as_tensor();
    let op = op_name(
        PrimitiveOpKind::ReduceProd,
        op_descriptor::GpuLaunchKind::Reduction,
    )?;
    match input.dtype() {
        DType::F32 => {
            let t = typed_or_unsupported::<f32>(input, op)?;
            backend
                .reduce_prod_float_typed(t, axes)
                .map(Tensor::from_typed::<f32>)
        }
        DType::F64 => {
            let t = typed_or_unsupported::<f64>(input, op)?;
            backend
                .reduce_prod_float_typed(t, axes)
                .map(Tensor::from_typed::<f64>)
        }
        DType::I32 => {
            let t = typed_or_unsupported::<i32>(input, op)?;
            backend
                .reduce_prod_int_typed(t, axes)
                .map(Tensor::from_typed::<i32>)
        }
        DType::I64 => {
            let t = typed_or_unsupported::<i64>(input, op)?;
            backend
                .reduce_prod_int_typed(t, axes)
                .map(Tensor::from_typed::<i64>)
        }
        DType::Bool => Err(unsupported_dtype(op, input.dtype())),
        DType::C32 => {
            let t = typed_or_unsupported::<Complex32>(input, op)?;
            backend
                .reduce_prod_complex_typed(t, axes)
                .map(Tensor::from_typed::<num_complex::Complex32>)
        }
        DType::C64 => {
            let t = typed_or_unsupported::<Complex64>(input, op)?;
            backend
                .reduce_prod_complex_typed(t, axes)
                .map(Tensor::from_typed::<num_complex::Complex64>)
        }
        // A caller-owned payload has no GPU implementation for this operation.
        DType::External(_) => Err(crate::Error::unsupported(
            "reduce_prod",
            "an externally defined payload is not supported by this GPU operation",
        )),
    }
}

pub(super) fn reduce_max_read(
    backend: &mut CudaBackend,
    input: TensorRead<'_>,
    axes: &[usize],
) -> crate::Result<Tensor> {
    let input = backend.read_input(input)?;
    let input = input.as_tensor();
    let op = op_name(
        PrimitiveOpKind::ReduceMax,
        op_descriptor::GpuLaunchKind::Reduction,
    )?;
    match input.dtype() {
        DType::F32 => {
            let t = typed_or_unsupported::<f32>(input, op)?;
            backend
                .reduce_max_float_typed(t, axes)
                .map(Tensor::from_typed::<f32>)
        }
        DType::F64 => {
            let t = typed_or_unsupported::<f64>(input, op)?;
            backend
                .reduce_max_float_typed(t, axes)
                .map(Tensor::from_typed::<f64>)
        }
        DType::I32 => {
            let t = typed_or_unsupported::<i32>(input, op)?;
            backend
                .reduce_max_int_typed(t, axes)
                .map(Tensor::from_typed::<i32>)
        }
        DType::I64 => {
            let t = typed_or_unsupported::<i64>(input, op)?;
            backend
                .reduce_max_int_typed(t, axes)
                .map(Tensor::from_typed::<i64>)
        }
        DType::Bool | DType::C32 | DType::C64 => Err(unsupported_dtype(op, input.dtype())),
        // A caller-owned payload has no GPU implementation for this operation.
        DType::External(_) => Err(crate::Error::unsupported(
            "reduce_max",
            "an externally defined payload is not supported by this GPU operation",
        )),
    }
}

pub(super) fn reduce_min_read(
    backend: &mut CudaBackend,
    input: TensorRead<'_>,
    axes: &[usize],
) -> crate::Result<Tensor> {
    let input = backend.read_input(input)?;
    let input = input.as_tensor();
    let op = op_name(
        PrimitiveOpKind::ReduceMin,
        op_descriptor::GpuLaunchKind::Reduction,
    )?;
    match input.dtype() {
        DType::F32 => {
            let t = typed_or_unsupported::<f32>(input, op)?;
            backend
                .reduce_min_float_typed(t, axes)
                .map(Tensor::from_typed::<f32>)
        }
        DType::F64 => {
            let t = typed_or_unsupported::<f64>(input, op)?;
            backend
                .reduce_min_float_typed(t, axes)
                .map(Tensor::from_typed::<f64>)
        }
        DType::I32 => {
            let t = typed_or_unsupported::<i32>(input, op)?;
            backend
                .reduce_min_int_typed(t, axes)
                .map(Tensor::from_typed::<i32>)
        }
        DType::I64 => {
            let t = typed_or_unsupported::<i64>(input, op)?;
            backend
                .reduce_min_int_typed(t, axes)
                .map(Tensor::from_typed::<i64>)
        }
        DType::Bool | DType::C32 | DType::C64 => Err(unsupported_dtype(op, input.dtype())),
        // A caller-owned payload has no GPU implementation for this operation.
        DType::External(_) => Err(crate::Error::unsupported(
            "reduce_min",
            "an externally defined payload is not supported by this GPU operation",
        )),
    }
}

pub(super) fn reduce_sum_squares_read(
    backend: &mut CudaBackend,
    input: TensorRead<'_>,
    axes: &[usize],
) -> crate::Result<Tensor> {
    let op = op_name(
        PrimitiveOpKind::ReduceSumSquares,
        op_descriptor::GpuLaunchKind::Reduction,
    )?;
    let Some(input) = input.as_tensor() else {
        return Err(crate::Error::unsupported(
            op,
            "CUDA sum-of-squares requires a resident tensor",
        ));
    };
    // Dispatch on the tag and recover the typed tensor, which is what `as_typed` exists for.
    if axes.is_empty() {
        return match input.dtype() {
            DType::F32 | DType::F64 => ops::mul_read(
                backend,
                TensorRead::from_tensor(input),
                TensorRead::from_tensor(input),
            ),
            DType::I32 | DType::I64 | DType::Bool | DType::C32 | DType::C64 => {
                Err(unsupported_dtype(op, input.dtype()))
            }
            DType::External(_) => Err(crate::Error::unsupported(
                op,
                "an externally defined payload is not supported by this GPU operation",
            )),
        };
    }
    match input.dtype() {
        DType::F32 => {
            let t = typed_or_unsupported::<f32>(input, op)?;
            backend
                .reduce_sum_squares_float_typed(t, axes)
                .map(Tensor::from_typed::<f32>)
        }
        DType::F64 => {
            let t = typed_or_unsupported::<f64>(input, op)?;
            backend
                .reduce_sum_squares_float_typed(t, axes)
                .map(Tensor::from_typed::<f64>)
        }
        DType::I32 | DType::I64 | DType::Bool | DType::C32 | DType::C64 => {
            Err(unsupported_dtype(op, input.dtype()))
        }
        DType::External(_) => Err(crate::Error::unsupported(
            op,
            "an externally defined payload is not supported by this GPU operation",
        )),
    }
}

pub(super) fn dot_general_with_conj(
    backend: &mut CudaBackend,
    lhs: &Tensor,
    rhs: &Tensor,
    config: &DotGeneralConfig,
    lhs_conj: bool,
    rhs_conj: bool,
) -> crate::Result<Tensor> {
    gemm::dot_general_with_conj(backend, lhs, rhs, config, lhs_conj, rhs_conj)
}

pub(super) fn dot_general_read(
    backend: &mut CudaBackend,
    lhs: TensorRead<'_>,
    rhs: TensorRead<'_>,
    config: &DotGeneralConfig,
) -> crate::Result<Tensor> {
    gemm::dot_general_read_allocating(backend, lhs, rhs, config, false, false)
}

pub(super) fn dot_general_read_into_accum(
    backend: &mut CudaBackend,
    lhs: TensorRead<'_>,
    rhs: TensorRead<'_>,
    config: &DotGeneralConfig,
    accumulation: DotGeneralAccumulation,
    mut out: TensorWrite<'_>,
) -> crate::Result<()> {
    tenferro_tensor::backend::validate_dot_general_accumulation(
        &lhs,
        &rhs,
        config,
        accumulation,
        &out,
        "dot_general",
    )?;
    gemm::dot_general_read_into_accum(backend, &lhs, &rhs, config, accumulation, &mut out)
}

pub(super) fn gather(
    backend: &mut CudaBackend,
    operand: &Tensor,
    start_indices: &Tensor,
    config: &GatherConfig,
) -> crate::Result<Tensor> {
    match (operand.dtype(), start_indices.dtype()) {
        (DType::F32, DType::F32) => backend
            .gather_typed(
                typed_or_unsupported::<f32>(operand, "gather")?,
                typed_or_unsupported::<f32>(start_indices, "gather")?,
                config,
            )
            .map(Tensor::from_typed::<f32>),
        (DType::F64, DType::F32) => backend
            .gather_typed(
                typed_or_unsupported::<f64>(operand, "gather")?,
                typed_or_unsupported::<f32>(start_indices, "gather")?,
                config,
            )
            .map(Tensor::from_typed::<f64>),
        (DType::C32, DType::F32) => backend
            .gather_typed(
                typed_or_unsupported::<Complex32>(operand, "gather")?,
                typed_or_unsupported::<f32>(start_indices, "gather")?,
                config,
            )
            .map(Tensor::from_typed::<num_complex::Complex32>),
        (DType::C64, DType::F32) => backend
            .gather_typed(
                typed_or_unsupported::<Complex64>(operand, "gather")?,
                typed_or_unsupported::<f32>(start_indices, "gather")?,
                config,
            )
            .map(Tensor::from_typed::<num_complex::Complex64>),
        (DType::I32, DType::F32) => backend
            .gather_typed(
                typed_or_unsupported::<i32>(operand, "gather")?,
                typed_or_unsupported::<f32>(start_indices, "gather")?,
                config,
            )
            .map(Tensor::from_typed::<i32>),
        (DType::F32, DType::F64) => backend
            .gather_typed(
                typed_or_unsupported::<f32>(operand, "gather")?,
                typed_or_unsupported::<f64>(start_indices, "gather")?,
                config,
            )
            .map(Tensor::from_typed::<f32>),
        (DType::F64, DType::F64) => backend
            .gather_typed(
                typed_or_unsupported::<f64>(operand, "gather")?,
                typed_or_unsupported::<f64>(start_indices, "gather")?,
                config,
            )
            .map(Tensor::from_typed::<f64>),
        (DType::C32, DType::F64) => backend
            .gather_typed(
                typed_or_unsupported::<Complex32>(operand, "gather")?,
                typed_or_unsupported::<f64>(start_indices, "gather")?,
                config,
            )
            .map(Tensor::from_typed::<num_complex::Complex32>),
        (DType::C64, DType::F64) => backend
            .gather_typed(
                typed_or_unsupported::<Complex64>(operand, "gather")?,
                typed_or_unsupported::<f64>(start_indices, "gather")?,
                config,
            )
            .map(Tensor::from_typed::<num_complex::Complex64>),
        (DType::I32, DType::F64) => backend
            .gather_typed(
                typed_or_unsupported::<i32>(operand, "gather")?,
                typed_or_unsupported::<f64>(start_indices, "gather")?,
                config,
            )
            .map(Tensor::from_typed::<i32>),
        (DType::F32, DType::I32) => backend
            .gather_typed(
                typed_or_unsupported::<f32>(operand, "gather")?,
                typed_or_unsupported::<i32>(start_indices, "gather")?,
                config,
            )
            .map(Tensor::from_typed::<f32>),
        (DType::F64, DType::I32) => backend
            .gather_typed(
                typed_or_unsupported::<f64>(operand, "gather")?,
                typed_or_unsupported::<i32>(start_indices, "gather")?,
                config,
            )
            .map(Tensor::from_typed::<f64>),
        (DType::C32, DType::I32) => backend
            .gather_typed(
                typed_or_unsupported::<Complex32>(operand, "gather")?,
                typed_or_unsupported::<i32>(start_indices, "gather")?,
                config,
            )
            .map(Tensor::from_typed::<num_complex::Complex32>),
        (DType::C64, DType::I32) => backend
            .gather_typed(
                typed_or_unsupported::<Complex64>(operand, "gather")?,
                typed_or_unsupported::<i32>(start_indices, "gather")?,
                config,
            )
            .map(Tensor::from_typed::<num_complex::Complex64>),
        (DType::I32, DType::I32) => backend
            .gather_typed(
                typed_or_unsupported::<i32>(operand, "gather")?,
                typed_or_unsupported::<i32>(start_indices, "gather")?,
                config,
            )
            .map(Tensor::from_typed::<i32>),
        (DType::F32, DType::I64) => backend
            .gather_typed(
                typed_or_unsupported::<f32>(operand, "gather")?,
                typed_or_unsupported::<i64>(start_indices, "gather")?,
                config,
            )
            .map(Tensor::from_typed::<f32>),
        (DType::F64, DType::I64) => backend
            .gather_typed(
                typed_or_unsupported::<f64>(operand, "gather")?,
                typed_or_unsupported::<i64>(start_indices, "gather")?,
                config,
            )
            .map(Tensor::from_typed::<f64>),
        (DType::C32, DType::I64) => backend
            .gather_typed(
                typed_or_unsupported::<Complex32>(operand, "gather")?,
                typed_or_unsupported::<i64>(start_indices, "gather")?,
                config,
            )
            .map(Tensor::from_typed::<num_complex::Complex32>),
        (DType::C64, DType::I64) => backend
            .gather_typed(
                typed_or_unsupported::<Complex64>(operand, "gather")?,
                typed_or_unsupported::<i64>(start_indices, "gather")?,
                config,
            )
            .map(Tensor::from_typed::<num_complex::Complex64>),
        (DType::I32, DType::I64) => backend
            .gather_typed(
                typed_or_unsupported::<i32>(operand, "gather")?,
                typed_or_unsupported::<i64>(start_indices, "gather")?,
                config,
            )
            .map(Tensor::from_typed::<i32>),
        (DType::Bool, DType::F32) => backend
            .gather_bool(
                typed_or_unsupported::<bool>(operand, "gather")?,
                typed_or_unsupported::<f32>(start_indices, "gather")?,
                config,
            )
            .map(Tensor::from_typed::<bool>),
        (DType::Bool, DType::F64) => backend
            .gather_bool(
                typed_or_unsupported::<bool>(operand, "gather")?,
                typed_or_unsupported::<f64>(start_indices, "gather")?,
                config,
            )
            .map(Tensor::from_typed::<bool>),
        (DType::Bool, DType::I32) => backend
            .gather_bool(
                typed_or_unsupported::<bool>(operand, "gather")?,
                typed_or_unsupported::<i32>(start_indices, "gather")?,
                config,
            )
            .map(Tensor::from_typed::<bool>),
        (DType::Bool, DType::I64) => backend
            .gather_bool(
                typed_or_unsupported::<bool>(operand, "gather")?,
                typed_or_unsupported::<i64>(start_indices, "gather")?,
                config,
            )
            .map(Tensor::from_typed::<bool>),
        (_, DType::Bool) => Err(unsupported_dtype("gather", start_indices.dtype())),
        (_, DType::C32 | DType::C64) => Err(unsupported_dtype("gather", start_indices.dtype())),
        (DType::I64, _) => Err(unsupported_dtype("gather", operand.dtype())),
        // A caller-owned payload has no GPU implementation for this operation.
        (DType::External(_), _) | (_, DType::External(_)) => Err(crate::Error::unsupported(
            "gather",
            "an externally defined payload is not supported by this GPU operation",
        )),
    }
}

pub(super) fn scatter(
    backend: &mut CudaBackend,
    operand: &Tensor,
    scatter_indices: &Tensor,
    updates: &Tensor,
    config: &ScatterConfig,
) -> crate::Result<Tensor> {
    match (operand.dtype(), scatter_indices.dtype(), updates.dtype()) {
        (DType::F32, DType::F32, DType::F32) => backend
            .scatter_float_typed(
                typed_or_unsupported::<f32>(operand, "scatter")?,
                typed_or_unsupported::<f32>(scatter_indices, "scatter")?,
                typed_or_unsupported::<f32>(updates, "scatter")?,
                config,
            )
            .map(Tensor::from_typed::<f32>),
        (DType::F64, DType::F32, DType::F64) => backend
            .scatter_float_typed(
                typed_or_unsupported::<f64>(operand, "scatter")?,
                typed_or_unsupported::<f32>(scatter_indices, "scatter")?,
                typed_or_unsupported::<f64>(updates, "scatter")?,
                config,
            )
            .map(Tensor::from_typed::<f64>),
        (DType::C32, DType::F32, DType::C32) => backend
            .scatter_complex_typed::<_, f32, _>(
                typed_or_unsupported::<Complex32>(operand, "scatter")?,
                typed_or_unsupported::<f32>(scatter_indices, "scatter")?,
                typed_or_unsupported::<Complex32>(updates, "scatter")?,
                config,
            )
            .map(Tensor::from_typed::<num_complex::Complex32>),
        (DType::C64, DType::F32, DType::C64) => backend
            .scatter_complex_typed::<_, f64, _>(
                typed_or_unsupported::<Complex64>(operand, "scatter")?,
                typed_or_unsupported::<f32>(scatter_indices, "scatter")?,
                typed_or_unsupported::<Complex64>(updates, "scatter")?,
                config,
            )
            .map(Tensor::from_typed::<num_complex::Complex64>),
        (DType::F32, DType::F64, DType::F32) => backend
            .scatter_float_typed(
                typed_or_unsupported::<f32>(operand, "scatter")?,
                typed_or_unsupported::<f64>(scatter_indices, "scatter")?,
                typed_or_unsupported::<f32>(updates, "scatter")?,
                config,
            )
            .map(Tensor::from_typed::<f32>),
        (DType::F64, DType::F64, DType::F64) => backend
            .scatter_float_typed(
                typed_or_unsupported::<f64>(operand, "scatter")?,
                typed_or_unsupported::<f64>(scatter_indices, "scatter")?,
                typed_or_unsupported::<f64>(updates, "scatter")?,
                config,
            )
            .map(Tensor::from_typed::<f64>),
        (DType::C32, DType::F64, DType::C32) => backend
            .scatter_complex_typed::<_, f32, _>(
                typed_or_unsupported::<Complex32>(operand, "scatter")?,
                typed_or_unsupported::<f64>(scatter_indices, "scatter")?,
                typed_or_unsupported::<Complex32>(updates, "scatter")?,
                config,
            )
            .map(Tensor::from_typed::<num_complex::Complex32>),
        (DType::C64, DType::F64, DType::C64) => backend
            .scatter_complex_typed::<_, f64, _>(
                typed_or_unsupported::<Complex64>(operand, "scatter")?,
                typed_or_unsupported::<f64>(scatter_indices, "scatter")?,
                typed_or_unsupported::<Complex64>(updates, "scatter")?,
                config,
            )
            .map(Tensor::from_typed::<num_complex::Complex64>),
        (DType::F32, DType::I32, DType::F32) => backend
            .scatter_float_typed(
                typed_or_unsupported::<f32>(operand, "scatter")?,
                typed_or_unsupported::<i32>(scatter_indices, "scatter")?,
                typed_or_unsupported::<f32>(updates, "scatter")?,
                config,
            )
            .map(Tensor::from_typed::<f32>),
        (DType::F64, DType::I32, DType::F64) => backend
            .scatter_float_typed(
                typed_or_unsupported::<f64>(operand, "scatter")?,
                typed_or_unsupported::<i32>(scatter_indices, "scatter")?,
                typed_or_unsupported::<f64>(updates, "scatter")?,
                config,
            )
            .map(Tensor::from_typed::<f64>),
        (DType::C32, DType::I32, DType::C32) => backend
            .scatter_complex_typed::<_, f32, _>(
                typed_or_unsupported::<Complex32>(operand, "scatter")?,
                typed_or_unsupported::<i32>(scatter_indices, "scatter")?,
                typed_or_unsupported::<Complex32>(updates, "scatter")?,
                config,
            )
            .map(Tensor::from_typed::<num_complex::Complex32>),
        (DType::C64, DType::I32, DType::C64) => backend
            .scatter_complex_typed::<_, f64, _>(
                typed_or_unsupported::<Complex64>(operand, "scatter")?,
                typed_or_unsupported::<i32>(scatter_indices, "scatter")?,
                typed_or_unsupported::<Complex64>(updates, "scatter")?,
                config,
            )
            .map(Tensor::from_typed::<num_complex::Complex64>),
        (DType::F32, DType::I64, DType::F32) => backend
            .scatter_float_typed(
                typed_or_unsupported::<f32>(operand, "scatter")?,
                typed_or_unsupported::<i64>(scatter_indices, "scatter")?,
                typed_or_unsupported::<f32>(updates, "scatter")?,
                config,
            )
            .map(Tensor::from_typed::<f32>),
        (DType::F64, DType::I64, DType::F64) => backend
            .scatter_float_typed(
                typed_or_unsupported::<f64>(operand, "scatter")?,
                typed_or_unsupported::<i64>(scatter_indices, "scatter")?,
                typed_or_unsupported::<f64>(updates, "scatter")?,
                config,
            )
            .map(Tensor::from_typed::<f64>),
        (DType::C32, DType::I64, DType::C32) => backend
            .scatter_complex_typed::<_, f32, _>(
                typed_or_unsupported::<Complex32>(operand, "scatter")?,
                typed_or_unsupported::<i64>(scatter_indices, "scatter")?,
                typed_or_unsupported::<Complex32>(updates, "scatter")?,
                config,
            )
            .map(Tensor::from_typed::<num_complex::Complex32>),
        (DType::C64, DType::I64, DType::C64) => backend
            .scatter_complex_typed::<_, f64, _>(
                typed_or_unsupported::<Complex64>(operand, "scatter")?,
                typed_or_unsupported::<i64>(scatter_indices, "scatter")?,
                typed_or_unsupported::<Complex64>(updates, "scatter")?,
                config,
            )
            .map(Tensor::from_typed::<num_complex::Complex64>),
        (_, DType::Bool, _) => Err(unsupported_dtype("scatter", scatter_indices.dtype())),
        (_, DType::C32 | DType::C64, _) => {
            Err(unsupported_dtype("scatter", scatter_indices.dtype()))
        }
        (DType::Bool, _, _) => Err(unsupported_operation(
            "scatter",
            "Bool data tensors are not supported by additive scatter",
        )),
        (DType::I32, _, _) | (DType::I64, _, _) => {
            Err(unsupported_dtype("scatter", operand.dtype()))
        }
        (_, _, _) => Err(ternary_dtype_mismatch(
            "scatter",
            operand,
            scatter_indices,
            updates,
        )),
    }
}

pub(super) fn slice(
    backend: &mut CudaBackend,
    input: &Tensor,
    config: &SliceConfig,
) -> crate::Result<Tensor> {
    match input.dtype() {
        DType::F32 => {
            let t = typed_or_unsupported::<f32>(input, "slice")?;
            backend
                .slice_typed(t, config)
                .map(Tensor::from_typed::<f32>)
        }
        DType::F64 => {
            let t = typed_or_unsupported::<f64>(input, "slice")?;
            backend
                .slice_typed(t, config)
                .map(Tensor::from_typed::<f64>)
        }
        DType::I32 => {
            let t = typed_or_unsupported::<i32>(input, "slice")?;
            backend
                .slice_typed(t, config)
                .map(Tensor::from_typed::<i32>)
        }
        DType::I64 => {
            let t = typed_or_unsupported::<i64>(input, "slice")?;
            backend
                .slice_typed(t, config)
                .map(Tensor::from_typed::<i64>)
        }
        DType::Bool => {
            let t = typed_or_unsupported::<bool>(input, "slice")?;
            backend
                .slice_bool(t, config)
                .map(Tensor::from_typed::<bool>)
        }
        DType::C32 => {
            let t = typed_or_unsupported::<Complex32>(input, "slice")?;
            backend
                .slice_typed(t, config)
                .map(Tensor::from_typed::<num_complex::Complex32>)
        }
        DType::C64 => {
            let t = typed_or_unsupported::<Complex64>(input, "slice")?;
            backend
                .slice_typed(t, config)
                .map(Tensor::from_typed::<num_complex::Complex64>)
        }
        // A caller-owned payload has no GPU implementation for this operation.
        DType::External(_) => Err(crate::Error::unsupported(
            "slice",
            "an externally defined payload is not supported by this GPU operation",
        )),
    }
}

pub(super) fn dynamic_slice(
    backend: &mut CudaBackend,
    input: &Tensor,
    starts: &Tensor,
    slice_sizes: &[usize],
) -> crate::Result<Tensor> {
    match (input.dtype(), starts.dtype()) {
        (DType::F32, DType::F32) => backend
            .dynamic_slice_typed(
                typed_or_unsupported::<f32>(input, "dynamic_slice")?,
                typed_or_unsupported::<f32>(starts, "dynamic_slice")?,
                slice_sizes,
            )
            .map(Tensor::from_typed::<f32>),
        (DType::F64, DType::F32) => backend
            .dynamic_slice_typed(
                typed_or_unsupported::<f64>(input, "dynamic_slice")?,
                typed_or_unsupported::<f32>(starts, "dynamic_slice")?,
                slice_sizes,
            )
            .map(Tensor::from_typed::<f64>),
        (DType::C32, DType::F32) => backend
            .dynamic_slice_typed(
                typed_or_unsupported::<Complex32>(input, "dynamic_slice")?,
                typed_or_unsupported::<f32>(starts, "dynamic_slice")?,
                slice_sizes,
            )
            .map(Tensor::from_typed::<num_complex::Complex32>),
        (DType::C64, DType::F32) => backend
            .dynamic_slice_typed(
                typed_or_unsupported::<Complex64>(input, "dynamic_slice")?,
                typed_or_unsupported::<f32>(starts, "dynamic_slice")?,
                slice_sizes,
            )
            .map(Tensor::from_typed::<num_complex::Complex64>),
        (DType::I32, DType::F32) => backend
            .dynamic_slice_typed(
                typed_or_unsupported::<i32>(input, "dynamic_slice")?,
                typed_or_unsupported::<f32>(starts, "dynamic_slice")?,
                slice_sizes,
            )
            .map(Tensor::from_typed::<i32>),
        (DType::F32, DType::F64) => backend
            .dynamic_slice_typed(
                typed_or_unsupported::<f32>(input, "dynamic_slice")?,
                typed_or_unsupported::<f64>(starts, "dynamic_slice")?,
                slice_sizes,
            )
            .map(Tensor::from_typed::<f32>),
        (DType::F64, DType::F64) => backend
            .dynamic_slice_typed(
                typed_or_unsupported::<f64>(input, "dynamic_slice")?,
                typed_or_unsupported::<f64>(starts, "dynamic_slice")?,
                slice_sizes,
            )
            .map(Tensor::from_typed::<f64>),
        (DType::C32, DType::F64) => backend
            .dynamic_slice_typed(
                typed_or_unsupported::<Complex32>(input, "dynamic_slice")?,
                typed_or_unsupported::<f64>(starts, "dynamic_slice")?,
                slice_sizes,
            )
            .map(Tensor::from_typed::<num_complex::Complex32>),
        (DType::C64, DType::F64) => backend
            .dynamic_slice_typed(
                typed_or_unsupported::<Complex64>(input, "dynamic_slice")?,
                typed_or_unsupported::<f64>(starts, "dynamic_slice")?,
                slice_sizes,
            )
            .map(Tensor::from_typed::<num_complex::Complex64>),
        (DType::I32, DType::F64) => backend
            .dynamic_slice_typed(
                typed_or_unsupported::<i32>(input, "dynamic_slice")?,
                typed_or_unsupported::<f64>(starts, "dynamic_slice")?,
                slice_sizes,
            )
            .map(Tensor::from_typed::<i32>),
        (DType::F32, DType::I32) => backend
            .dynamic_slice_typed(
                typed_or_unsupported::<f32>(input, "dynamic_slice")?,
                typed_or_unsupported::<i32>(starts, "dynamic_slice")?,
                slice_sizes,
            )
            .map(Tensor::from_typed::<f32>),
        (DType::F64, DType::I32) => backend
            .dynamic_slice_typed(
                typed_or_unsupported::<f64>(input, "dynamic_slice")?,
                typed_or_unsupported::<i32>(starts, "dynamic_slice")?,
                slice_sizes,
            )
            .map(Tensor::from_typed::<f64>),
        (DType::C32, DType::I32) => backend
            .dynamic_slice_typed(
                typed_or_unsupported::<Complex32>(input, "dynamic_slice")?,
                typed_or_unsupported::<i32>(starts, "dynamic_slice")?,
                slice_sizes,
            )
            .map(Tensor::from_typed::<num_complex::Complex32>),
        (DType::C64, DType::I32) => backend
            .dynamic_slice_typed(
                typed_or_unsupported::<Complex64>(input, "dynamic_slice")?,
                typed_or_unsupported::<i32>(starts, "dynamic_slice")?,
                slice_sizes,
            )
            .map(Tensor::from_typed::<num_complex::Complex64>),
        (DType::I32, DType::I32) => backend
            .dynamic_slice_typed(
                typed_or_unsupported::<i32>(input, "dynamic_slice")?,
                typed_or_unsupported::<i32>(starts, "dynamic_slice")?,
                slice_sizes,
            )
            .map(Tensor::from_typed::<i32>),
        (DType::F32, DType::I64) => backend
            .dynamic_slice_typed(
                typed_or_unsupported::<f32>(input, "dynamic_slice")?,
                typed_or_unsupported::<i64>(starts, "dynamic_slice")?,
                slice_sizes,
            )
            .map(Tensor::from_typed::<f32>),
        (DType::F64, DType::I64) => backend
            .dynamic_slice_typed(
                typed_or_unsupported::<f64>(input, "dynamic_slice")?,
                typed_or_unsupported::<i64>(starts, "dynamic_slice")?,
                slice_sizes,
            )
            .map(Tensor::from_typed::<f64>),
        (DType::C32, DType::I64) => backend
            .dynamic_slice_typed(
                typed_or_unsupported::<Complex32>(input, "dynamic_slice")?,
                typed_or_unsupported::<i64>(starts, "dynamic_slice")?,
                slice_sizes,
            )
            .map(Tensor::from_typed::<num_complex::Complex32>),
        (DType::C64, DType::I64) => backend
            .dynamic_slice_typed(
                typed_or_unsupported::<Complex64>(input, "dynamic_slice")?,
                typed_or_unsupported::<i64>(starts, "dynamic_slice")?,
                slice_sizes,
            )
            .map(Tensor::from_typed::<num_complex::Complex64>),
        (DType::I32, DType::I64) => backend
            .dynamic_slice_typed(
                typed_or_unsupported::<i32>(input, "dynamic_slice")?,
                typed_or_unsupported::<i64>(starts, "dynamic_slice")?,
                slice_sizes,
            )
            .map(Tensor::from_typed::<i32>),
        (DType::Bool, DType::I32) => backend
            .dynamic_slice_bool(
                typed_or_unsupported::<bool>(input, "dynamic_slice")?,
                typed_or_unsupported::<i32>(starts, "dynamic_slice")?,
                slice_sizes,
            )
            .map(Tensor::from_typed::<bool>),
        (DType::Bool, DType::I64) => backend
            .dynamic_slice_bool(
                typed_or_unsupported::<bool>(input, "dynamic_slice")?,
                typed_or_unsupported::<i64>(starts, "dynamic_slice")?,
                slice_sizes,
            )
            .map(Tensor::from_typed::<bool>),
        (DType::Bool, DType::F32) => backend
            .dynamic_slice_bool(
                typed_or_unsupported::<bool>(input, "dynamic_slice")?,
                typed_or_unsupported::<f32>(starts, "dynamic_slice")?,
                slice_sizes,
            )
            .map(Tensor::from_typed::<bool>),
        (DType::Bool, DType::F64) => backend
            .dynamic_slice_bool(
                typed_or_unsupported::<bool>(input, "dynamic_slice")?,
                typed_or_unsupported::<f64>(starts, "dynamic_slice")?,
                slice_sizes,
            )
            .map(Tensor::from_typed::<bool>),
        (_, DType::Bool) => Err(unsupported_dtype("dynamic_slice", starts.dtype())),
        (_, DType::C32 | DType::C64) => Err(unsupported_dtype("dynamic_slice", starts.dtype())),
        (DType::I64, _) => Err(unsupported_dtype("dynamic_slice", input.dtype())),
        // A caller-owned payload has no GPU implementation for this operation.
        (DType::External(_), _) | (_, DType::External(_)) => Err(crate::Error::unsupported(
            "dynamic_slice",
            "an externally defined payload is not supported by this GPU operation",
        )),
    }
}

pub(super) fn dynamic_update_slice(
    _backend: &mut CudaBackend,
    _operand: &Tensor,
    _update: &Tensor,
    _starts: &Tensor,
) -> crate::Result<Tensor> {
    Err(unsupported_operation(
        "dynamic_update_slice",
        "not implemented for the CubeCL backend",
    ))
}

pub(super) fn pad(
    backend: &mut CudaBackend,
    input: &Tensor,
    config: &PadConfig,
) -> crate::Result<Tensor> {
    match input.dtype() {
        DType::F32 => {
            let t = typed_or_unsupported::<f32>(input, "pad")?;
            backend.pad_typed(t, config).map(Tensor::from_typed::<f32>)
        }
        DType::F64 => {
            let t = typed_or_unsupported::<f64>(input, "pad")?;
            backend.pad_typed(t, config).map(Tensor::from_typed::<f64>)
        }
        DType::I32 => {
            let t = typed_or_unsupported::<i32>(input, "pad")?;
            backend.pad_typed(t, config).map(Tensor::from_typed::<i32>)
        }
        DType::I64 => {
            let t = typed_or_unsupported::<i64>(input, "pad")?;
            backend.pad_typed(t, config).map(Tensor::from_typed::<i64>)
        }
        DType::Bool => {
            let t = typed_or_unsupported::<bool>(input, "pad")?;
            backend.pad_bool(t, config).map(Tensor::from_typed::<bool>)
        }
        DType::C32 => {
            let t = typed_or_unsupported::<Complex32>(input, "pad")?;
            backend
                .pad_typed(t, config)
                .map(Tensor::from_typed::<num_complex::Complex32>)
        }
        DType::C64 => {
            let t = typed_or_unsupported::<Complex64>(input, "pad")?;
            backend
                .pad_typed(t, config)
                .map(Tensor::from_typed::<num_complex::Complex64>)
        }
        // A caller-owned payload has no GPU implementation for this operation.
        DType::External(_) => Err(crate::Error::unsupported(
            "pad",
            "an externally defined payload is not supported by this GPU operation",
        )),
    }
}

pub(super) fn concatenate(
    backend: &mut CudaBackend,
    inputs: &[&Tensor],
    axis: usize,
) -> crate::Result<Tensor> {
    let first = inputs.first().copied().ok_or_else(|| {
        crate::Error::invalid_argument(
            "concatenate",
            "inputs",
            "concatenate requires at least one input",
        )
    })?;
    match first.dtype() {
        DType::F32 => {
            let typed: crate::Result<Vec<&TypedTensor<f32>>> = inputs
                .iter()
                .map(|tensor| {
                    tensor
                        .as_typed::<f32>()
                        .ok_or_else(|| dtype_mismatch("concatenate", first, tensor))
                })
                .collect();
            backend
                .concatenate_typed(&typed?, axis)
                .map(Tensor::from_typed::<f32>)
        }
        DType::F64 => {
            let typed: crate::Result<Vec<&TypedTensor<f64>>> = inputs
                .iter()
                .map(|tensor| {
                    tensor
                        .as_typed::<f64>()
                        .ok_or_else(|| dtype_mismatch("concatenate", first, tensor))
                })
                .collect();
            backend
                .concatenate_typed(&typed?, axis)
                .map(Tensor::from_typed::<f64>)
        }
        DType::I32 => {
            let typed: crate::Result<Vec<&TypedTensor<i32>>> = inputs
                .iter()
                .map(|tensor| {
                    tensor
                        .as_typed::<i32>()
                        .ok_or_else(|| dtype_mismatch("concatenate", first, tensor))
                })
                .collect();
            backend
                .concatenate_typed(&typed?, axis)
                .map(Tensor::from_typed::<i32>)
        }
        DType::I64 => {
            let typed: crate::Result<Vec<&TypedTensor<i64>>> = inputs
                .iter()
                .map(|tensor| {
                    tensor
                        .as_typed::<i64>()
                        .ok_or_else(|| dtype_mismatch("concatenate", first, tensor))
                })
                .collect();
            backend
                .concatenate_typed(&typed?, axis)
                .map(Tensor::from_typed::<i64>)
        }
        DType::Bool => {
            let typed: crate::Result<Vec<&TypedTensor<bool>>> = inputs
                .iter()
                .map(|tensor| {
                    tensor
                        .as_typed::<bool>()
                        .ok_or_else(|| dtype_mismatch("concatenate", first, tensor))
                })
                .collect();
            backend
                .concatenate_bool(&typed?, axis)
                .map(Tensor::from_typed::<bool>)
        }
        DType::C32 => {
            let typed: crate::Result<Vec<&TypedTensor<Complex32>>> = inputs
                .iter()
                .map(|tensor| {
                    tensor
                        .as_typed::<Complex32>()
                        .ok_or_else(|| dtype_mismatch("concatenate", first, tensor))
                })
                .collect();
            backend
                .concatenate_typed(&typed?, axis)
                .map(Tensor::from_typed::<num_complex::Complex32>)
        }
        DType::C64 => {
            let typed: crate::Result<Vec<&TypedTensor<Complex64>>> = inputs
                .iter()
                .map(|tensor| {
                    tensor
                        .as_typed::<Complex64>()
                        .ok_or_else(|| dtype_mismatch("concatenate", first, tensor))
                })
                .collect();
            backend
                .concatenate_typed(&typed?, axis)
                .map(Tensor::from_typed::<num_complex::Complex64>)
        }
        // A caller-owned payload has no GPU implementation for this operation.
        DType::External(_) => Err(crate::Error::unsupported(
            "concatenate",
            "an externally defined payload is not supported by this GPU operation",
        )),
    }
}

pub(super) fn reverse(
    backend: &mut CudaBackend,
    input: &Tensor,
    axes: &[usize],
) -> crate::Result<Tensor> {
    match input.dtype() {
        DType::F32 => {
            let t = typed_or_unsupported::<f32>(input, "reverse")?;
            backend
                .reverse_typed(t, axes)
                .map(Tensor::from_typed::<f32>)
        }
        DType::F64 => {
            let t = typed_or_unsupported::<f64>(input, "reverse")?;
            backend
                .reverse_typed(t, axes)
                .map(Tensor::from_typed::<f64>)
        }
        DType::I32 => {
            let t = typed_or_unsupported::<i32>(input, "reverse")?;
            backend
                .reverse_typed(t, axes)
                .map(Tensor::from_typed::<i32>)
        }
        DType::I64 => {
            let t = typed_or_unsupported::<i64>(input, "reverse")?;
            backend
                .reverse_typed(t, axes)
                .map(Tensor::from_typed::<i64>)
        }
        DType::Bool => {
            let t = typed_or_unsupported::<bool>(input, "reverse")?;
            backend
                .reverse_bool(t, axes)
                .map(Tensor::from_typed::<bool>)
        }
        DType::C32 => {
            let t = typed_or_unsupported::<Complex32>(input, "reverse")?;
            backend
                .reverse_typed(t, axes)
                .map(Tensor::from_typed::<num_complex::Complex32>)
        }
        DType::C64 => {
            let t = typed_or_unsupported::<Complex64>(input, "reverse")?;
            backend
                .reverse_typed(t, axes)
                .map(Tensor::from_typed::<num_complex::Complex64>)
        }
        // A caller-owned payload has no GPU implementation for this operation.
        DType::External(_) => Err(crate::Error::unsupported(
            "reverse",
            "an externally defined payload is not supported by this GPU operation",
        )),
    }
}

pub(super) fn execute_elementwise_fusion(
    backend: &mut CudaBackend,
    inputs: &[&Tensor],
    plan: &crate::backend::ElementwiseFusionPlan,
) -> crate::Result<Option<Vec<Tensor>>> {
    fusion::execute_elementwise_fusion(backend, inputs, plan)
}

pub(super) fn execute_broadcast_multiply(
    backend: &mut CudaBackend,
    lhs: TensorRead<'_>,
    lhs_shape: &[usize],
    lhs_dims: &[usize],
    rhs: TensorRead<'_>,
    rhs_shape: &[usize],
    rhs_dims: &[usize],
) -> crate::Result<Option<Tensor>> {
    match (lhs, rhs) {
        // Owned operands dispatch on the runtime tags, which is what `as_typed` exists for.
        (TensorRead::Tensor(lhs), TensorRead::Tensor(rhs)) => match (lhs.dtype(), rhs.dtype()) {
            (DType::F32, DType::F32) => {
                let lhs = typed_or_unsupported::<f32>(lhs, "broadcast_multiply")?;
                let rhs = typed_or_unsupported::<f32>(rhs, "broadcast_multiply")?;
                launch_broadcast_multiply_typed(
                    backend,
                    &CompactOperand::Tensor(lhs),
                    lhs_shape,
                    lhs_dims,
                    &CompactOperand::Tensor(rhs),
                    rhs_shape,
                    rhs_dims,
                )
                .map(Tensor::from_typed::<f32>)
                .map(Some)
            }
            (DType::F64, DType::F64) => {
                let lhs = typed_or_unsupported::<f64>(lhs, "broadcast_multiply")?;
                let rhs = typed_or_unsupported::<f64>(rhs, "broadcast_multiply")?;
                launch_broadcast_multiply_typed(
                    backend,
                    &CompactOperand::Tensor(lhs),
                    lhs_shape,
                    lhs_dims,
                    &CompactOperand::Tensor(rhs),
                    rhs_shape,
                    rhs_dims,
                )
                .map(Tensor::from_typed::<f64>)
                .map(Some)
            }
            (DType::I32, DType::I32) => {
                let lhs = typed_or_unsupported::<i32>(lhs, "broadcast_multiply")?;
                let rhs = typed_or_unsupported::<i32>(rhs, "broadcast_multiply")?;
                launch_broadcast_multiply_int_typed(
                    backend,
                    &CompactOperand::Tensor(lhs),
                    lhs_shape,
                    lhs_dims,
                    &CompactOperand::Tensor(rhs),
                    rhs_shape,
                    rhs_dims,
                )
                .map(Tensor::from_typed::<i32>)
                .map(Some)
            }
            (DType::I64, DType::I64) => {
                let lhs = typed_or_unsupported::<i64>(lhs, "broadcast_multiply")?;
                let rhs = typed_or_unsupported::<i64>(rhs, "broadcast_multiply")?;
                launch_broadcast_multiply_int_typed(
                    backend,
                    &CompactOperand::Tensor(lhs),
                    lhs_shape,
                    lhs_dims,
                    &CompactOperand::Tensor(rhs),
                    rhs_shape,
                    rhs_dims,
                )
                .map(Tensor::from_typed::<i64>)
                .map(Some)
            }
            (DType::C32, DType::C32) => {
                let lhs = typed_or_unsupported::<Complex32>(lhs, "broadcast_multiply")?;
                let rhs = typed_or_unsupported::<Complex32>(rhs, "broadcast_multiply")?;
                launch_broadcast_multiply_complex_typed(
                    backend,
                    &CompactOperand::Tensor(lhs),
                    lhs_shape,
                    lhs_dims,
                    &CompactOperand::Tensor(rhs),
                    rhs_shape,
                    rhs_dims,
                )
                .map(Tensor::from_typed::<Complex32>)
                .map(Some)
            }
            (DType::C64, DType::C64) => {
                let lhs = typed_or_unsupported::<Complex64>(lhs, "broadcast_multiply")?;
                let rhs = typed_or_unsupported::<Complex64>(rhs, "broadcast_multiply")?;
                launch_broadcast_multiply_complex_typed(
                    backend,
                    &CompactOperand::Tensor(lhs),
                    lhs_shape,
                    lhs_dims,
                    &CompactOperand::Tensor(rhs),
                    rhs_shape,
                    rhs_dims,
                )
                .map(Tensor::from_typed::<Complex64>)
                .map(Some)
            }
            (DType::Bool, DType::Bool) => Ok(None),
            _ => Err(dtype_mismatch("broadcast_multiply", lhs, rhs)),
        },
        // The eager einsum path prepares operands as borrowed views over
        // already allocated device storage. A compact view is consumed
        // directly; other read forms keep the caller's fallback.
        (TensorRead::View(lhs), TensorRead::View(rhs)) => {
            let (Some(lhs), Some(rhs)) = (compact_view(lhs)?, compact_view(rhs)?) else {
                return Ok(None);
            };
            match (lhs, rhs) {
                (BroadcastMultiplyView::F32(lhs), BroadcastMultiplyView::F32(rhs)) => {
                    launch_broadcast_multiply_typed(
                        backend,
                        &CompactOperand::View(lhs),
                        lhs_shape,
                        lhs_dims,
                        &CompactOperand::View(rhs),
                        rhs_shape,
                        rhs_dims,
                    )
                    .map(Tensor::from_typed::<f32>)
                    .map(Some)
                }
                (BroadcastMultiplyView::F64(lhs), BroadcastMultiplyView::F64(rhs)) => {
                    launch_broadcast_multiply_typed(
                        backend,
                        &CompactOperand::View(lhs),
                        lhs_shape,
                        lhs_dims,
                        &CompactOperand::View(rhs),
                        rhs_shape,
                        rhs_dims,
                    )
                    .map(Tensor::from_typed::<f64>)
                    .map(Some)
                }
                (BroadcastMultiplyView::I32(lhs), BroadcastMultiplyView::I32(rhs)) => {
                    launch_broadcast_multiply_int_typed(
                        backend,
                        &CompactOperand::View(lhs),
                        lhs_shape,
                        lhs_dims,
                        &CompactOperand::View(rhs),
                        rhs_shape,
                        rhs_dims,
                    )
                    .map(Tensor::from_typed::<i32>)
                    .map(Some)
                }
                (BroadcastMultiplyView::I64(lhs), BroadcastMultiplyView::I64(rhs)) => {
                    launch_broadcast_multiply_int_typed(
                        backend,
                        &CompactOperand::View(lhs),
                        lhs_shape,
                        lhs_dims,
                        &CompactOperand::View(rhs),
                        rhs_shape,
                        rhs_dims,
                    )
                    .map(Tensor::from_typed::<i64>)
                    .map(Some)
                }
                (BroadcastMultiplyView::C32(lhs), BroadcastMultiplyView::C32(rhs)) => {
                    launch_broadcast_multiply_complex_typed(
                        backend,
                        &CompactOperand::View(lhs),
                        lhs_shape,
                        lhs_dims,
                        &CompactOperand::View(rhs),
                        rhs_shape,
                        rhs_dims,
                    )
                    .map(Tensor::from_typed::<Complex32>)
                    .map(Some)
                }
                (BroadcastMultiplyView::C64(lhs), BroadcastMultiplyView::C64(rhs)) => {
                    launch_broadcast_multiply_complex_typed(
                        backend,
                        &CompactOperand::View(lhs),
                        lhs_shape,
                        lhs_dims,
                        &CompactOperand::View(rhs),
                        rhs_shape,
                        rhs_dims,
                    )
                    .map(Tensor::from_typed::<Complex64>)
                    .map(Some)
                }
                // Mismatched dtypes keep the caller's fallback, which
                // reports the mismatch on the owned path.
                _ => Ok(None),
            }
        }
        // Mixed owned and borrowed operands keep the caller's fallback:
        // the traced runtime hands both operands in the same form.
        _ => Ok(None),
    }
}

pub(super) fn vdot_read(
    backend: &mut CudaBackend,
    lhs: TensorRead<'_>,
    rhs: TensorRead<'_>,
) -> crate::Result<Tensor> {
    blas1::vdot_read(backend, lhs, rhs)
}

pub(super) fn norm_squared_read(
    backend: &mut CudaBackend,
    input: TensorRead<'_>,
) -> crate::Result<Tensor> {
    blas1::norm_squared_read(backend, input)
}

pub(super) fn axpby_read_into_accum(
    backend: &mut CudaBackend,
    alpha: ContractionScalar,
    x: TensorRead<'_>,
    beta: ContractionScalar,
    y: TensorWrite<'_>,
) -> crate::Result<()> {
    blas1::axpby_read_into_accum(backend, alpha, x, beta, y)
}
