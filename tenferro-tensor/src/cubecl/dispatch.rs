use cubecl::client::ComputeClient;
use cubecl::prelude::*;
use cubecl_cuda::CudaRuntime;

use crate::config::CompareDir;
use crate::cubecl::CubeclRuntime;
use crate::types::{
    Buffer, ComputeDevice, CubeclBuffer, MemoryKind, Placement, Tensor, TypedTensor,
};

pub(crate) const DEFAULT_CUBE_DIM_X: u32 = 256;

pub(crate) fn cube_count_for_len(len: usize) -> CubeCount {
    let cubes = len.div_ceil(DEFAULT_CUBE_DIM_X as usize) as u32;
    CubeCount::Static(cubes.max(1), 1, 1)
}

pub(crate) fn cube_dim_1d() -> CubeDim {
    CubeDim::new_1d(DEFAULT_CUBE_DIM_X)
}

pub(crate) fn single_thread_launch_config() -> (CubeCount, CubeDim) {
    (CubeCount::new_single(), CubeDim::new_1d(1))
}

pub(crate) fn comptime_sequence<T: CubeType>(values: &[T]) -> Sequence<T>
where
    T: Clone,
{
    let mut out = Sequence::new();
    for value in values {
        out.push(value.clone());
    }
    out
}

pub(crate) fn cubecl_buffer<'a, T>(
    tensor: &'a TypedTensor<T>,
    op: &'static str,
) -> crate::Result<&'a CubeclBuffer<T>> {
    match &tensor.buffer {
        Buffer::Host(_) | Buffer::Backend(_) => Err(crate::Error::BackendFailure {
            op,
            message: "expected GPU tensor (Buffer::Cubecl), got CPU tensor. \
                      Use cubecl::upload_tensor() to transfer to GPU before calling GPU ops."
                .into(),
        }),
        Buffer::Cubecl(buffer) => Ok(buffer),
    }
}

pub(crate) fn ensure_resident_on_runtime<T>(
    rt: &CubeclRuntime,
    tensor: &TypedTensor<T>,
    op: &'static str,
) -> crate::Result<()> {
    cubecl_buffer(tensor, op)?;
    if !matches!(&tensor.placement.memory_kind, MemoryKind::Device) {
        return Err(crate::Error::BackendFailure {
            op,
            message: format!(
                "expected GPU tensor placement, got {:?}",
                tensor.placement.memory_kind
            ),
        });
    }
    match &tensor.placement.resident_device {
        Some(device) if device.kind == "cuda" && device.ordinal == rt.device_ordinal() => Ok(()),
        Some(device) => Err(crate::Error::BackendFailure {
            op,
            message: format!(
                "expected GPU tensor resident on cuda:{}, got {}:{}",
                rt.device_ordinal(),
                device.kind,
                device.ordinal
            ),
        }),
        None => Err(crate::Error::BackendFailure {
            op,
            message: format!(
                "expected GPU tensor resident on cuda:{}, got missing resident_device metadata",
                rt.device_ordinal()
            ),
        }),
    }
}

pub(crate) fn typed_from_cubecl<T>(
    shape: Vec<usize>,
    buffer: CubeclBuffer<T>,
    device_ordinal: usize,
) -> TypedTensor<T> {
    TypedTensor {
        buffer: Buffer::Cubecl(buffer),
        shape,
        placement: Placement {
            memory_kind: MemoryKind::Device,
            resident_device: Some(ComputeDevice {
                kind: "cuda".into(),
                ordinal: device_ordinal,
            }),
        },
    }
}

pub(crate) fn alloc_output<T: CubeElement + Clone>(
    rt: &CubeclRuntime,
    shape: &[usize],
) -> TypedTensor<T> {
    let len: usize = shape.iter().product();
    let handle = rt.client().empty(len * core::mem::size_of::<T>());
    typed_from_cubecl(
        shape.to_vec(),
        CubeclBuffer::new(handle, len),
        rt.device_ordinal(),
    )
}

pub(crate) fn typed_tensor_array_arg<T: CubeElement + Clone>(
    tensor: &TypedTensor<T>,
    op: &'static str,
) -> crate::Result<ArrayArg<CudaRuntime>> {
    let buffer = cubecl_buffer(tensor, op)?;
    // SAFETY: `CubeclBuffer::len` tracks the allocation length in elements.
    Ok(unsafe { ArrayArg::from_raw_parts(buffer.handle.clone(), buffer.len) })
}

pub(crate) fn launch_unary<TIn, TOut>(
    rt: &CubeclRuntime,
    input: &TypedTensor<TIn>,
    out_shape: &[usize],
    op: &'static str,
    launch: impl FnOnce(
        &ComputeClient<CudaRuntime>,
        CubeCount,
        CubeDim,
        ArrayArg<CudaRuntime>,
        ArrayArg<CudaRuntime>,
    ),
) -> crate::Result<TypedTensor<TOut>>
where
    TIn: CubeElement + Clone,
    TOut: CubeElement + Clone,
{
    let output = alloc_output::<TOut>(rt, out_shape);
    let len = output.n_elements();
    if len == 0 {
        return Ok(output);
    }
    let client = rt.client();
    let output_arg = typed_tensor_array_arg(&output, op)?;
    let input_arg = typed_tensor_array_arg(input, op)?;
    launch(
        client,
        cube_count_for_len(len),
        cube_dim_1d(),
        output_arg,
        input_arg,
    );
    Ok(output)
}

pub(crate) fn launch_nullary_into<TOut>(
    rt: &CubeclRuntime,
    output: &TypedTensor<TOut>,
    op: &'static str,
    count: CubeCount,
    dim: CubeDim,
    launch: impl FnOnce(&ComputeClient<CudaRuntime>, CubeCount, CubeDim, ArrayArg<CudaRuntime>),
) -> crate::Result<()>
where
    TOut: CubeElement + Clone,
{
    if output.n_elements() == 0 {
        return Ok(());
    }
    let output_arg = typed_tensor_array_arg(output, op)?;
    launch(rt.client(), count, dim, output_arg);
    Ok(())
}

pub(crate) fn launch_unary_with_config<TIn, TOut>(
    rt: &CubeclRuntime,
    input: &TypedTensor<TIn>,
    out_shape: &[usize],
    op: &'static str,
    count: CubeCount,
    dim: CubeDim,
    launch: impl FnOnce(
        &ComputeClient<CudaRuntime>,
        CubeCount,
        CubeDim,
        ArrayArg<CudaRuntime>,
        ArrayArg<CudaRuntime>,
    ),
) -> crate::Result<TypedTensor<TOut>>
where
    TIn: CubeElement + Clone,
    TOut: CubeElement + Clone,
{
    let output = alloc_output::<TOut>(rt, out_shape);
    if output.n_elements() == 0 {
        return Ok(output);
    }
    let client = rt.client();
    let output_arg = typed_tensor_array_arg(&output, op)?;
    let input_arg = typed_tensor_array_arg(input, op)?;
    launch(client, count, dim, output_arg, input_arg);
    Ok(output)
}

pub(crate) fn launch_unary_into<TIn, TOut>(
    rt: &CubeclRuntime,
    output: &TypedTensor<TOut>,
    input: &TypedTensor<TIn>,
    op: &'static str,
    count: CubeCount,
    dim: CubeDim,
    launch: impl FnOnce(
        &ComputeClient<CudaRuntime>,
        CubeCount,
        CubeDim,
        ArrayArg<CudaRuntime>,
        ArrayArg<CudaRuntime>,
    ),
) -> crate::Result<()>
where
    TIn: CubeElement + Clone,
    TOut: CubeElement + Clone,
{
    if output.n_elements() == 0 {
        return Ok(());
    }
    let output_arg = typed_tensor_array_arg(output, op)?;
    let input_arg = typed_tensor_array_arg(input, op)?;
    launch(rt.client(), count, dim, output_arg, input_arg);
    Ok(())
}

pub(crate) fn launch_binary<TLhs, TRhs, TOut>(
    rt: &CubeclRuntime,
    lhs: &TypedTensor<TLhs>,
    rhs: &TypedTensor<TRhs>,
    out_shape: &[usize],
    op: &'static str,
    launch: impl FnOnce(
        &ComputeClient<CudaRuntime>,
        CubeCount,
        CubeDim,
        ArrayArg<CudaRuntime>,
        ArrayArg<CudaRuntime>,
        ArrayArg<CudaRuntime>,
    ),
) -> crate::Result<TypedTensor<TOut>>
where
    TLhs: CubeElement + Clone,
    TRhs: CubeElement + Clone,
    TOut: CubeElement + Clone,
{
    let output = alloc_output::<TOut>(rt, out_shape);
    let len = output.n_elements();
    if len == 0 {
        return Ok(output);
    }
    let client = rt.client();
    let output_arg = typed_tensor_array_arg(&output, op)?;
    let lhs_arg = typed_tensor_array_arg(lhs, op)?;
    let rhs_arg = typed_tensor_array_arg(rhs, op)?;
    launch(
        client,
        cube_count_for_len(len),
        cube_dim_1d(),
        output_arg,
        lhs_arg,
        rhs_arg,
    );
    Ok(output)
}

pub(crate) fn launch_binary_with_config<TLhs, TRhs, TOut>(
    rt: &CubeclRuntime,
    lhs: &TypedTensor<TLhs>,
    rhs: &TypedTensor<TRhs>,
    out_shape: &[usize],
    op: &'static str,
    count: CubeCount,
    dim: CubeDim,
    launch: impl FnOnce(
        &ComputeClient<CudaRuntime>,
        CubeCount,
        CubeDim,
        ArrayArg<CudaRuntime>,
        ArrayArg<CudaRuntime>,
        ArrayArg<CudaRuntime>,
    ),
) -> crate::Result<TypedTensor<TOut>>
where
    TLhs: CubeElement + Clone,
    TRhs: CubeElement + Clone,
    TOut: CubeElement + Clone,
{
    let output = alloc_output::<TOut>(rt, out_shape);
    if output.n_elements() == 0 {
        return Ok(output);
    }
    let client = rt.client();
    let output_arg = typed_tensor_array_arg(&output, op)?;
    let lhs_arg = typed_tensor_array_arg(lhs, op)?;
    let rhs_arg = typed_tensor_array_arg(rhs, op)?;
    launch(client, count, dim, output_arg, lhs_arg, rhs_arg);
    Ok(output)
}

pub(crate) fn launch_binary_into<TLhs, TRhs, TOut>(
    rt: &CubeclRuntime,
    output: &TypedTensor<TOut>,
    lhs: &TypedTensor<TLhs>,
    rhs: &TypedTensor<TRhs>,
    op: &'static str,
    count: CubeCount,
    dim: CubeDim,
    launch: impl FnOnce(
        &ComputeClient<CudaRuntime>,
        CubeCount,
        CubeDim,
        ArrayArg<CudaRuntime>,
        ArrayArg<CudaRuntime>,
        ArrayArg<CudaRuntime>,
    ),
) -> crate::Result<()>
where
    TLhs: CubeElement + Clone,
    TRhs: CubeElement + Clone,
    TOut: CubeElement + Clone,
{
    if output.n_elements() == 0 {
        return Ok(());
    }
    let output_arg = typed_tensor_array_arg(output, op)?;
    let lhs_arg = typed_tensor_array_arg(lhs, op)?;
    let rhs_arg = typed_tensor_array_arg(rhs, op)?;
    launch(rt.client(), count, dim, output_arg, lhs_arg, rhs_arg);
    Ok(())
}

pub(crate) fn launch_ternary<TA, TB, TC, TOut>(
    rt: &CubeclRuntime,
    a: &TypedTensor<TA>,
    b: &TypedTensor<TB>,
    c: &TypedTensor<TC>,
    out_shape: &[usize],
    op: &'static str,
    launch: impl FnOnce(
        &ComputeClient<CudaRuntime>,
        CubeCount,
        CubeDim,
        ArrayArg<CudaRuntime>,
        ArrayArg<CudaRuntime>,
        ArrayArg<CudaRuntime>,
        ArrayArg<CudaRuntime>,
    ),
) -> crate::Result<TypedTensor<TOut>>
where
    TA: CubeElement + Clone,
    TB: CubeElement + Clone,
    TC: CubeElement + Clone,
    TOut: CubeElement + Clone,
{
    let output = alloc_output::<TOut>(rt, out_shape);
    let len = output.n_elements();
    if len == 0 {
        return Ok(output);
    }
    let client = rt.client();
    let output_arg = typed_tensor_array_arg(&output, op)?;
    let a_arg = typed_tensor_array_arg(a, op)?;
    let b_arg = typed_tensor_array_arg(b, op)?;
    let c_arg = typed_tensor_array_arg(c, op)?;
    launch(
        client,
        cube_count_for_len(len),
        cube_dim_1d(),
        output_arg,
        a_arg,
        b_arg,
        c_arg,
    );
    Ok(output)
}

pub(crate) fn launch_ternary_with_config<TA, TB, TC, TOut>(
    rt: &CubeclRuntime,
    a: &TypedTensor<TA>,
    b: &TypedTensor<TB>,
    c: &TypedTensor<TC>,
    out_shape: &[usize],
    op: &'static str,
    count: CubeCount,
    dim: CubeDim,
    launch: impl FnOnce(
        &ComputeClient<CudaRuntime>,
        CubeCount,
        CubeDim,
        ArrayArg<CudaRuntime>,
        ArrayArg<CudaRuntime>,
        ArrayArg<CudaRuntime>,
        ArrayArg<CudaRuntime>,
    ),
) -> crate::Result<TypedTensor<TOut>>
where
    TA: CubeElement + Clone,
    TB: CubeElement + Clone,
    TC: CubeElement + Clone,
    TOut: CubeElement + Clone,
{
    let output = alloc_output::<TOut>(rt, out_shape);
    if output.n_elements() == 0 {
        return Ok(output);
    }
    let client = rt.client();
    let output_arg = typed_tensor_array_arg(&output, op)?;
    let a_arg = typed_tensor_array_arg(a, op)?;
    let b_arg = typed_tensor_array_arg(b, op)?;
    let c_arg = typed_tensor_array_arg(c, op)?;
    launch(client, count, dim, output_arg, a_arg, b_arg, c_arg);
    Ok(output)
}

pub(crate) fn dtype_mismatch(op: &'static str, lhs: &Tensor, rhs: &Tensor) -> crate::Error {
    crate::Error::DTypeMismatch {
        op,
        lhs: lhs.dtype(),
        rhs: rhs.dtype(),
    }
}

pub(crate) fn ternary_dtype_mismatch(
    op: &'static str,
    first: &Tensor,
    second: &Tensor,
    third: &Tensor,
) -> crate::Error {
    crate::Error::BackendFailure {
        op,
        message: format!(
            "dtype mismatch first={:?} second={:?} third={:?}",
            first.dtype(),
            second.dtype(),
            third.dtype()
        ),
    }
}

pub(crate) fn ensure_same_shape(
    op: &'static str,
    lhs: &[usize],
    rhs: &[usize],
) -> crate::Result<()> {
    if lhs != rhs {
        return Err(crate::Error::ShapeMismatch {
            op,
            lhs: lhs.to_vec(),
            rhs: rhs.to_vec(),
        });
    }
    Ok(())
}

pub(crate) fn ensure_rank(op: &'static str, expected: usize, actual: usize) -> crate::Result<()> {
    if expected != actual {
        return Err(crate::Error::RankMismatch {
            op,
            expected,
            actual,
        });
    }
    Ok(())
}

pub(crate) fn ensure_axis(op: &'static str, axis: usize, rank: usize) -> crate::Result<()> {
    if axis >= rank {
        return Err(crate::Error::AxisOutOfBounds { op, axis, rank });
    }
    Ok(())
}

pub(crate) fn ensure_axes_unique(
    op: &'static str,
    role: &'static str,
    axes: &[usize],
    rank: usize,
) -> crate::Result<()> {
    let mut seen = vec![false; rank];
    for &axis in axes {
        ensure_axis(op, axis, rank)?;
        if seen[axis] {
            return Err(crate::Error::DuplicateAxis { op, axis, role });
        }
        seen[axis] = true;
    }
    Ok(())
}

pub(crate) fn compare_mode(dir: &CompareDir) -> usize {
    match dir {
        CompareDir::Eq => 0,
        CompareDir::Lt => 1,
        CompareDir::Le => 2,
        CompareDir::Gt => 3,
        CompareDir::Ge => 4,
    }
}

macro_rules! dispatch_unary_both {
    ($backend:expr, $input:expr, $op:literal, $float_launch:path, $complex_launch:path) => {{
        match $input {
            $crate::Tensor::F32(tensor) => $crate::cubecl::dispatch::launch_unary(
                $backend.runtime(),
                tensor,
                &tensor.shape,
                $op,
                |client, count, dim, out, input| unsafe {
                    $float_launch::<f32, cubecl_cuda::CudaRuntime>(client, count, dim, out, input);
                },
            )
            .map($crate::Tensor::F32),
            $crate::Tensor::F64(tensor) => $crate::cubecl::dispatch::launch_unary(
                $backend.runtime(),
                tensor,
                &tensor.shape,
                $op,
                |client, count, dim, out, input| unsafe {
                    $float_launch::<f64, cubecl_cuda::CudaRuntime>(client, count, dim, out, input);
                },
            )
            .map($crate::Tensor::F64),
            $crate::Tensor::C32(tensor) => $crate::cubecl::dispatch::launch_unary(
                $backend.runtime(),
                tensor,
                &tensor.shape,
                $op,
                |client, count, dim, out, input| unsafe {
                    $complex_launch::<num_complex::Complex32, cubecl_cuda::CudaRuntime>(
                        client, count, dim, out, input,
                    );
                },
            )
            .map($crate::Tensor::C32),
            $crate::Tensor::C64(tensor) => $crate::cubecl::dispatch::launch_unary(
                $backend.runtime(),
                tensor,
                &tensor.shape,
                $op,
                |client, count, dim, out, input| unsafe {
                    $complex_launch::<num_complex::Complex64, cubecl_cuda::CudaRuntime>(
                        client, count, dim, out, input,
                    );
                },
            )
            .map($crate::Tensor::C64),
        }
    }};
}

macro_rules! dispatch_unary_float_only {
    ($backend:expr, $input:expr, $op:literal, $float_launch:path) => {{
        match $input {
            $crate::Tensor::F32(tensor) => $crate::cubecl::dispatch::launch_unary(
                $backend.runtime(),
                tensor,
                &tensor.shape,
                $op,
                |client, count, dim, out, input| unsafe {
                    $float_launch::<f32, cubecl_cuda::CudaRuntime>(client, count, dim, out, input);
                },
            )
            .map($crate::Tensor::F32),
            $crate::Tensor::F64(tensor) => $crate::cubecl::dispatch::launch_unary(
                $backend.runtime(),
                tensor,
                &tensor.shape,
                $op,
                |client, count, dim, out, input| unsafe {
                    $float_launch::<f64, cubecl_cuda::CudaRuntime>(client, count, dim, out, input);
                },
            )
            .map($crate::Tensor::F64),
            $crate::Tensor::C32(_) | $crate::Tensor::C64(_) => Err($crate::Error::BackendFailure {
                op: $op,
                message: format!("unsupported dtype {:?}", $input.dtype()),
            }),
        }
    }};
}

macro_rules! dispatch_unary_complex_or_clone {
    ($backend:expr, $input:expr, $op:literal, $complex_launch:path) => {{
        match $input {
            $crate::Tensor::F32(tensor) => {
                $crate::cubecl::dispatch::ensure_resident_on_runtime(
                    $backend.runtime(),
                    tensor,
                    $op,
                )?;
                Ok($crate::Tensor::F32(tensor.clone()))
            }
            $crate::Tensor::F64(tensor) => {
                $crate::cubecl::dispatch::ensure_resident_on_runtime(
                    $backend.runtime(),
                    tensor,
                    $op,
                )?;
                Ok($crate::Tensor::F64(tensor.clone()))
            }
            $crate::Tensor::C32(tensor) => $crate::cubecl::dispatch::launch_unary(
                $backend.runtime(),
                tensor,
                &tensor.shape,
                $op,
                |client, count, dim, out, input| unsafe {
                    $complex_launch::<num_complex::Complex32, cubecl_cuda::CudaRuntime>(
                        client, count, dim, out, input,
                    );
                },
            )
            .map($crate::Tensor::C32),
            $crate::Tensor::C64(tensor) => $crate::cubecl::dispatch::launch_unary(
                $backend.runtime(),
                tensor,
                &tensor.shape,
                $op,
                |client, count, dim, out, input| unsafe {
                    $complex_launch::<num_complex::Complex64, cubecl_cuda::CudaRuntime>(
                        client, count, dim, out, input,
                    );
                },
            )
            .map($crate::Tensor::C64),
        }
    }};
}

macro_rules! dispatch_binary_both {
    ($backend:expr, $lhs:expr, $rhs:expr, $op:literal, $float_launch:path, $complex_launch:path) => {{
        match ($lhs, $rhs) {
            ($crate::Tensor::F32(lhs), $crate::Tensor::F32(rhs)) => {
                $crate::cubecl::dispatch::ensure_same_shape($op, &lhs.shape, &rhs.shape)?;
                $crate::cubecl::dispatch::launch_binary(
                    $backend.runtime(),
                    lhs,
                    rhs,
                    &lhs.shape,
                    $op,
                    |client, count, dim, out, lhs, rhs| unsafe {
                        $float_launch::<f32, cubecl_cuda::CudaRuntime>(
                            client, count, dim, out, lhs, rhs,
                        );
                    },
                )
                .map($crate::Tensor::F32)
            }
            ($crate::Tensor::F64(lhs), $crate::Tensor::F64(rhs)) => {
                $crate::cubecl::dispatch::ensure_same_shape($op, &lhs.shape, &rhs.shape)?;
                $crate::cubecl::dispatch::launch_binary(
                    $backend.runtime(),
                    lhs,
                    rhs,
                    &lhs.shape,
                    $op,
                    |client, count, dim, out, lhs, rhs| unsafe {
                        $float_launch::<f64, cubecl_cuda::CudaRuntime>(
                            client, count, dim, out, lhs, rhs,
                        );
                    },
                )
                .map($crate::Tensor::F64)
            }
            ($crate::Tensor::C32(lhs), $crate::Tensor::C32(rhs)) => {
                $crate::cubecl::dispatch::ensure_same_shape($op, &lhs.shape, &rhs.shape)?;
                $crate::cubecl::dispatch::launch_binary(
                    $backend.runtime(),
                    lhs,
                    rhs,
                    &lhs.shape,
                    $op,
                    |client, count, dim, out, lhs, rhs| unsafe {
                        $complex_launch::<num_complex::Complex32, cubecl_cuda::CudaRuntime>(
                            client, count, dim, out, lhs, rhs,
                        );
                    },
                )
                .map($crate::Tensor::C32)
            }
            ($crate::Tensor::C64(lhs), $crate::Tensor::C64(rhs)) => {
                $crate::cubecl::dispatch::ensure_same_shape($op, &lhs.shape, &rhs.shape)?;
                $crate::cubecl::dispatch::launch_binary(
                    $backend.runtime(),
                    lhs,
                    rhs,
                    &lhs.shape,
                    $op,
                    |client, count, dim, out, lhs, rhs| unsafe {
                        $complex_launch::<num_complex::Complex64, cubecl_cuda::CudaRuntime>(
                            client, count, dim, out, lhs, rhs,
                        );
                    },
                )
                .map($crate::Tensor::C64)
            }
            _ => Err($crate::cubecl::dispatch::dtype_mismatch($op, $lhs, $rhs)),
        }
    }};
}

macro_rules! dispatch_binary_float_only {
    ($backend:expr, $lhs:expr, $rhs:expr, $op:literal, $float_launch:path) => {{
        match ($lhs, $rhs) {
            ($crate::Tensor::F32(lhs), $crate::Tensor::F32(rhs)) => {
                $crate::cubecl::dispatch::ensure_same_shape($op, &lhs.shape, &rhs.shape)?;
                $crate::cubecl::dispatch::launch_binary(
                    $backend.runtime(),
                    lhs,
                    rhs,
                    &lhs.shape,
                    $op,
                    |client, count, dim, out, lhs, rhs| unsafe {
                        $float_launch::<f32, cubecl_cuda::CudaRuntime>(
                            client, count, dim, out, lhs, rhs,
                        );
                    },
                )
                .map($crate::Tensor::F32)
            }
            ($crate::Tensor::F64(lhs), $crate::Tensor::F64(rhs)) => {
                $crate::cubecl::dispatch::ensure_same_shape($op, &lhs.shape, &rhs.shape)?;
                $crate::cubecl::dispatch::launch_binary(
                    $backend.runtime(),
                    lhs,
                    rhs,
                    &lhs.shape,
                    $op,
                    |client, count, dim, out, lhs, rhs| unsafe {
                        $float_launch::<f64, cubecl_cuda::CudaRuntime>(
                            client, count, dim, out, lhs, rhs,
                        );
                    },
                )
                .map($crate::Tensor::F64)
            }
            ($crate::Tensor::C32(_), $crate::Tensor::C32(_))
            | ($crate::Tensor::C64(_), $crate::Tensor::C64(_)) => {
                Err($crate::Error::BackendFailure {
                    op: $op,
                    message: format!("unsupported dtype {:?}", $lhs.dtype()),
                })
            }
            _ => Err($crate::cubecl::dispatch::dtype_mismatch($op, $lhs, $rhs)),
        }
    }};
}

macro_rules! dispatch_ternary_float_only {
    ($backend:expr, $a:expr, $b:expr, $c:expr, $op:literal, $float_launch:path $(, $extra:expr )* $(,)?) => {{
        match ($a, $b, $c) {
            ($crate::Tensor::F32(a), $crate::Tensor::F32(b), $crate::Tensor::F32(c)) => {
                $crate::cubecl::dispatch::ensure_same_shape($op, &a.shape, &b.shape)?;
                $crate::cubecl::dispatch::ensure_same_shape($op, &a.shape, &c.shape)?;
                $crate::cubecl::dispatch::launch_ternary(
                    $backend.runtime(),
                    a,
                    b,
                    c,
                    &a.shape,
                    $op,
                    |client, count, dim, out, a, b, c| unsafe {
                        $float_launch::<f32, cubecl_cuda::CudaRuntime>(
                            client, count, dim, out, a, b, c $(, $extra )*
                        );
                    },
                )
                .map($crate::Tensor::F32)
            }
            ($crate::Tensor::F64(a), $crate::Tensor::F64(b), $crate::Tensor::F64(c)) => {
                $crate::cubecl::dispatch::ensure_same_shape($op, &a.shape, &b.shape)?;
                $crate::cubecl::dispatch::ensure_same_shape($op, &a.shape, &c.shape)?;
                $crate::cubecl::dispatch::launch_ternary(
                    $backend.runtime(),
                    a,
                    b,
                    c,
                    &a.shape,
                    $op,
                    |client, count, dim, out, a, b, c| unsafe {
                        $float_launch::<f64, cubecl_cuda::CudaRuntime>(
                            client, count, dim, out, a, b, c $(, $extra )*
                        );
                    },
                )
                .map($crate::Tensor::F64)
            }
            ($crate::Tensor::C32(_), $crate::Tensor::C32(_), $crate::Tensor::C32(_))
            | ($crate::Tensor::C64(_), $crate::Tensor::C64(_), $crate::Tensor::C64(_)) => {
                Err($crate::Error::BackendFailure {
                    op: $op,
                    message: format!("unsupported dtype {:?}", $a.dtype()),
                })
            }
            _ => Err($crate::cubecl::dispatch::ternary_dtype_mismatch($op, $a, $b, $c)),
        }
    }};
}

pub(crate) use dispatch_binary_both;
pub(crate) use dispatch_binary_float_only;
pub(crate) use dispatch_ternary_float_only;
pub(crate) use dispatch_unary_both;
pub(crate) use dispatch_unary_complex_or_clone;
pub(crate) use dispatch_unary_float_only;
