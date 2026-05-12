use cubecl::client::ComputeClient;
use cubecl::prelude::*;
use cubecl_cuda::CudaRuntime;

use crate::config::CompareDir;
use crate::cubecl::CubeclRuntime;
use crate::types::{
    Buffer, ComputeDevice, CubeclBuffer, MemoryKind, MemoryOrder, Placement, Tensor, TypedTensor,
};

pub(crate) const DEFAULT_CUBE_DIM_X: u32 = 256;

pub(crate) fn cube_count_for_len(len: usize) -> CubeCount {
    let cubes = len.div_ceil(DEFAULT_CUBE_DIM_X as usize) as u32;
    CubeCount::Static(cubes.max(1), 1, 1)
}

pub(crate) fn cube_dim_1d() -> CubeDim {
    CubeDim::new_1d(DEFAULT_CUBE_DIM_X)
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

pub(crate) fn cubecl_shape_and_strides(shape: &[usize]) -> (Vec<usize>, Vec<usize>) {
    let strides = crate::types::col_major_strides(shape)
        .into_iter()
        .map(|stride| stride as usize)
        .collect();
    (shape.to_vec(), strides)
}

pub(crate) fn typed_tensor_binding<T: CubeElement + Clone>(
    tensor: &TypedTensor<T>,
    op: &'static str,
) -> crate::Result<TensorBinding<CudaRuntime>> {
    let buffer = cubecl_buffer(tensor, op)?;
    let expected_len = tensor
        .shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| crate::Error::BackendFailure {
            op,
            message: format!(
                "shape product overflow for CubeCL tensor shape {:?}",
                tensor.shape
            ),
        })?;
    if expected_len != buffer.len {
        return Err(crate::Error::BackendFailure {
            op,
            message: format!(
                "expected shape product {expected_len} elements, actual CubeclBuffer::len {}",
                buffer.len
            ),
        });
    }
    if tensor.order != MemoryOrder::ColMajor {
        return Err(crate::Error::BackendFailure {
            op,
            message: "expected column-major GPU tensor; row-major host tensors must be canonicalized during upload".into(),
        });
    }
    let (shape, strides) = cubecl_shape_and_strides(&tensor.shape);

    // SAFETY: `buffer.handle` references the CubeCL allocation for `tensor`.
    // The checked invariant above proves `buffer.len` equals the dense
    // element count of `tensor.shape`; `strides` is the matching dense
    // column-major layout metadata, so kernel indexing stays within that
    // allocation.
    Ok(unsafe {
        TensorBinding::from_raw_parts(buffer.handle.clone(), strides.into(), shape.into())
    })
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
        order: crate::MemoryOrder::ColMajor,
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

pub(crate) fn typed_tensor_array_arg_as<T, U>(
    tensor: &TypedTensor<T>,
    len: usize,
    op: &'static str,
) -> crate::Result<ArrayArg<CudaRuntime>>
where
    T: CubeElement + Clone,
    U: CubeElement + Clone,
{
    let buffer = cubecl_buffer(tensor, op)?;
    let requested_bytes =
        len.checked_mul(core::mem::size_of::<U>())
            .ok_or_else(|| crate::Error::BackendFailure {
                op,
                message: format!("reinterpreted CubeCL array length overflow for len {len}"),
            })?;
    let available_bytes = buffer
        .len
        .checked_mul(core::mem::size_of::<T>())
        .ok_or_else(|| crate::Error::BackendFailure {
            op,
            message: format!(
                "CubeCL buffer byte length overflow for {} elements",
                buffer.len
            ),
        })?;
    if requested_bytes > available_bytes {
        return Err(crate::Error::BackendFailure {
            op,
            message: format!(
                "reinterpreted CubeCL array needs {requested_bytes} bytes, buffer has {available_bytes}"
            ),
        });
    }

    // SAFETY: The checked byte-size invariant proves the requested view stays
    // within the same CubeCL allocation. Kernels using this helper are
    // responsible for using a representation-compatible scalar view.
    Ok(unsafe { ArrayArg::from_raw_parts(buffer.handle.clone(), len) })
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

pub(crate) fn launch_unary_tensor<TIn, TOut>(
    rt: &CubeclRuntime,
    input: &TypedTensor<TIn>,
    out_shape: &[usize],
    op: &'static str,
    launch: impl FnOnce(
        &ComputeClient<CudaRuntime>,
        CubeCount,
        CubeDim,
        TensorBinding<CudaRuntime>,
        TensorBinding<CudaRuntime>,
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
    let output_arg = typed_tensor_binding(&output, op)?;
    let input_arg = typed_tensor_binding(input, op)?;
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

pub(crate) fn launch_unary_tensor_into<TIn, TOut>(
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
        TensorBinding<CudaRuntime>,
        TensorBinding<CudaRuntime>,
    ),
) -> crate::Result<()>
where
    TIn: CubeElement + Clone,
    TOut: CubeElement + Clone,
{
    if output.n_elements() == 0 {
        return Ok(());
    }
    let output_arg = typed_tensor_binding(output, op)?;
    let input_arg = typed_tensor_binding(input, op)?;
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

pub(crate) fn launch_binary_tensor<TLhs, TRhs, TOut>(
    rt: &CubeclRuntime,
    lhs: &TypedTensor<TLhs>,
    rhs: &TypedTensor<TRhs>,
    out_shape: &[usize],
    op: &'static str,
    launch: impl FnOnce(
        &ComputeClient<CudaRuntime>,
        CubeCount,
        CubeDim,
        TensorBinding<CudaRuntime>,
        TensorBinding<CudaRuntime>,
        TensorBinding<CudaRuntime>,
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
    let output_arg = typed_tensor_binding(&output, op)?;
    let lhs_arg = typed_tensor_binding(lhs, op)?;
    let rhs_arg = typed_tensor_binding(rhs, op)?;
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
