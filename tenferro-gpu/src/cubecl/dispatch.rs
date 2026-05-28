use cubecl::client::ComputeClient;
use cubecl::prelude::*;
use cubecl_cuda::CudaRuntime;
use std::sync::Arc;

use crate::config::CompareDir;
use crate::cubecl::CubeclRuntime;
use crate::types::{
    Buffer, ComputeDevice, CubeclBuffer, DeviceKind, GpuBackendKind, MemoryKind, Placement, Tensor,
    TypedTensor,
};

pub(crate) const DEFAULT_CUBE_DIM_X: u32 = 256;

#[doc(hidden)]
pub fn cube_count_for_len(len: usize) -> CubeCount {
    let cubes = len.div_ceil(DEFAULT_CUBE_DIM_X as usize) as u32;
    CubeCount::Static(cubes.max(1), 1, 1)
}

#[doc(hidden)]
pub fn cube_dim_1d() -> CubeDim {
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

#[doc(hidden)]
pub fn cubecl_buffer<'a, T: 'static>(
    tensor: &'a TypedTensor<T>,
    op: &'static str,
) -> crate::Result<&'a CubeclBuffer<T>> {
    match &tensor.buffer {
        Buffer::Host(_) => Err(crate::Error::backend_failure(
            op,
            "expected CubeCL GPU tensor, got host tensor. \
                      Use upload_tensor() to transfer to GPU before calling GPU ops.",
        )),
        Buffer::Backend(buffer) => buffer
            .as_any()
            .downcast_ref::<CubeclBuffer<T>>()
            .ok_or_else(|| {
                crate::Error::backend_failure(
                    op,
                    format!(
                        "expected CubeCL GPU tensor, got backend buffer family `{}`",
                        buffer.backend_family()
                    ),
                )
            }),
    }
}

pub(crate) fn cubecl_shape_and_strides(shape: &[usize]) -> (Vec<usize>, Vec<usize>) {
    let strides = crate::types::col_major_strides(shape)
        .into_iter()
        .map(|stride| stride as usize)
        .collect();
    (shape.to_vec(), strides)
}

fn checked_shape_product(op: &'static str, shape: &[usize]) -> crate::Result<usize> {
    shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| {
            crate::Error::backend_failure(
                op,
                format!("shape product overflow for CubeCL tensor shape {shape:?}"),
            )
        })
}

fn validate_cubecl_buffer_len<T>(
    tensor: &TypedTensor<T>,
    buffer: &CubeclBuffer<T>,
    op: &'static str,
) -> crate::Result<()> {
    let expected_len = checked_shape_product(op, tensor.shape())?;
    if expected_len != buffer.len {
        return Err(crate::Error::backend_failure(
            op,
            format!(
                "expected shape product {expected_len} elements, actual CubeclBuffer::len {}",
                buffer.len
            ),
        ));
    }
    Ok(())
}

fn validate_raw_unary_shapes<TIn>(
    input: &TypedTensor<TIn>,
    out_shape: &[usize],
    op: &'static str,
) -> crate::Result<()> {
    ensure_same_shape(op, input.shape(), out_shape)
}

fn validate_raw_binary_shapes<TLhs, TRhs>(
    lhs: &TypedTensor<TLhs>,
    rhs: &TypedTensor<TRhs>,
    out_shape: &[usize],
    op: &'static str,
) -> crate::Result<()> {
    ensure_same_shape(op, lhs.shape(), out_shape)?;
    ensure_same_shape(op, rhs.shape(), out_shape)
}

fn validate_raw_ternary_shapes<TA, TB, TC>(
    a: &TypedTensor<TA>,
    b: &TypedTensor<TB>,
    c: &TypedTensor<TC>,
    out_shape: &[usize],
    op: &'static str,
) -> crate::Result<()> {
    ensure_same_shape(op, a.shape(), out_shape)?;
    ensure_same_shape(op, b.shape(), out_shape)?;
    ensure_same_shape(op, c.shape(), out_shape)
}

#[doc(hidden)]
pub fn typed_tensor_binding<T: CubeElement + Clone>(
    tensor: &TypedTensor<T>,
    op: &'static str,
) -> crate::Result<TensorBinding<CudaRuntime>> {
    let buffer = cubecl_buffer(tensor, op)?;
    validate_cubecl_buffer_len(tensor, buffer, op)?;
    let (shape, strides) = cubecl_shape_and_strides(tensor.shape());

    // SAFETY: `buffer.handle` references the CubeCL allocation for `tensor`.
    // The checked invariant above proves `buffer.len` equals the dense
    // element count of `tensor.shape`; `strides` is the matching dense
    // column-major layout metadata, so kernel indexing stays within that
    // allocation.
    Ok(unsafe {
        TensorBinding::from_raw_parts(buffer.handle.clone(), strides.into(), shape.into())
    })
}

pub(crate) fn ensure_resident_on_runtime<T: 'static>(
    rt: &CubeclRuntime,
    tensor: &TypedTensor<T>,
    op: &'static str,
) -> crate::Result<()> {
    cubecl_buffer(tensor, op)?;
    if !matches!(&tensor.placement.memory_kind, MemoryKind::Device) {
        return Err(crate::Error::backend_failure(
            op,
            format!(
                "expected GPU tensor placement, got {:?}",
                tensor.placement.memory_kind
            ),
        ));
    }
    match &tensor.placement.device {
        Some(device)
            if device.kind == DeviceKind::Gpu(GpuBackendKind::Cuda)
                && device.ordinal == rt.device_ordinal() =>
        {
            Ok(())
        }
        Some(device) => Err(crate::Error::backend_failure(
            op,
            format!(
                "expected GPU tensor resident on cuda:{}, got {:?}:{}",
                rt.device_ordinal(),
                device.kind,
                device.ordinal
            ),
        )),
        None => Err(crate::Error::backend_failure(
            op,
            format!(
                "expected GPU tensor resident on cuda:{}, got missing device metadata",
                rt.device_ordinal()
            ),
        )),
    }
}

#[doc(hidden)]
pub fn typed_from_cubecl<T: Send + Sync + 'static>(
    shape: Vec<usize>,
    buffer: CubeclBuffer<T>,
    device_ordinal: usize,
) -> TypedTensor<T> {
    TypedTensor::from_buffer_col_major(
        shape,
        Buffer::Backend(Arc::new(buffer)),
        Placement {
            memory_kind: MemoryKind::Device,
            device: Some(ComputeDevice {
                kind: DeviceKind::Gpu(GpuBackendKind::Cuda),
                ordinal: device_ordinal,
            }),
        },
    )
}

#[doc(hidden)]
pub fn alloc_output<T: CubeElement + Clone + Send + Sync + 'static>(
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

pub(crate) fn alloc_bool_output(rt: &CubeclRuntime, shape: &[usize]) -> TypedTensor<bool> {
    let len: usize = shape.iter().product();
    let handle = rt.client().empty(len);
    typed_from_cubecl(
        shape.to_vec(),
        CubeclBuffer::new(handle, len),
        rt.device_ordinal(),
    )
}

#[doc(hidden)]
pub fn typed_tensor_array_arg<T: CubeElement + Clone>(
    tensor: &TypedTensor<T>,
    op: &'static str,
) -> crate::Result<ArrayArg<CudaRuntime>> {
    let buffer = cubecl_buffer(tensor, op)?;
    validate_cubecl_buffer_len(tensor, buffer, op)?;

    // SAFETY: `buffer.handle` references the CubeCL allocation for `tensor`.
    // `validate_cubecl_buffer_len` proves `buffer.len` equals the dense shape
    // product, so raw linear kernels that index `0..out.len()` cannot observe
    // an array longer than the logical tensor allocation.
    Ok(unsafe { ArrayArg::from_raw_parts(buffer.handle.clone(), buffer.len) })
}

pub(crate) fn bool_tensor_array_arg(
    tensor: &TypedTensor<bool>,
    op: &'static str,
) -> crate::Result<ArrayArg<CudaRuntime>> {
    let buffer = cubecl_buffer(tensor, op)?;
    validate_cubecl_buffer_len(tensor, buffer, op)?;

    // SAFETY: CubeCL bool tensors are stored as one-byte predicate buffers by
    // `memory::upload_bool` and `alloc_bool_output`. The validated buffer
    // length is the logical element count consumed by raw Array<bool> kernels.
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
    validate_cubecl_buffer_len(tensor, buffer, op)?;
    let requested_bytes = len.checked_mul(core::mem::size_of::<U>()).ok_or_else(|| {
        crate::Error::backend_failure(
            op,
            format!("reinterpreted CubeCL array length overflow for len {len}"),
        )
    })?;
    let available_bytes = buffer
        .len
        .checked_mul(core::mem::size_of::<T>())
        .ok_or_else(|| {
            crate::Error::backend_failure(
                op,
                format!(
                    "CubeCL buffer byte length overflow for {} elements",
                    buffer.len
                ),
            )
        })?;
    if requested_bytes > available_bytes {
        return Err(crate::Error::backend_failure(op, format!(
                "reinterpreted CubeCL array needs {requested_bytes} bytes, buffer has {available_bytes}"
            )));
    }

    // SAFETY: `validate_cubecl_buffer_len` first proves the typed tensor shape
    // matches the backing allocation. The checked byte-size invariant then
    // proves the requested representation view stays within the same
    // allocation. Kernels using this helper are responsible for using a
    // representation-compatible scalar view, for example complex values as
    // their real and imaginary scalar parts.
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
    validate_raw_unary_shapes(input, out_shape, op)?;
    let output = alloc_output::<TOut>(rt, out_shape);
    let len = output.n_elements();
    if len == 0 {
        return Ok(output);
    }
    let client = rt.client();
    let output_arg = typed_tensor_array_arg(&output, op)?;
    let input_arg = typed_tensor_array_arg(input, op)?;
    // SAFETY: This helper is the host-side unchecked launch boundary for raw
    // shape-preserving unary kernels. The shape validation above proves input
    // and output have the same dense element count; `typed_tensor_array_arg`
    // proves each raw array length matches its tensor shape. The launch domain
    // covers `len == output.n_elements()`, and these kernels guard writes with
    // `ABSOLUTE_POS < out.len()`.
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
    // SAFETY: Logical tensor kernels use `TensorBinding`, whose construction
    // validates shape product against the CubeCL allocation length. The caller
    // supplies output shape and launch metadata already validated for the
    // specific structural/indexing operation, and the kernels guard their
    // launched index domain before mapping logical indices.
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
    // SAFETY: Nullary raw kernels write only to the validated output array.
    // The caller-supplied `count`/`dim` must describe the initialized domain,
    // and kernels using this path guard with `ABSOLUTE_POS < out.len()`.
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
    // SAFETY: `TensorBinding` construction validates shape and backing buffer
    // length for both tensors. The caller supplies a launch domain derived
    // from validated operation metadata, and the target kernel guards the
    // output or update domain before logical tensor indexing.
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
    validate_raw_binary_shapes(lhs, rhs, out_shape, op)?;
    let output = alloc_output::<TOut>(rt, out_shape);
    let len = output.n_elements();
    if len == 0 {
        return Ok(output);
    }
    let client = rt.client();
    let output_arg = typed_tensor_array_arg(&output, op)?;
    let lhs_arg = typed_tensor_array_arg(lhs, op)?;
    let rhs_arg = typed_tensor_array_arg(rhs, op)?;
    // SAFETY: This helper is the host-side unchecked launch boundary for raw
    // shape-preserving binary kernels. The shared shape validation above
    // proves all arrays have the same dense element count; the raw array
    // helpers prove every CubeCL buffer length matches its tensor shape. The
    // launch covers `len == output.n_elements()`, and elementwise kernels guard
    // with `ABSOLUTE_POS < out.len()`.
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

pub(crate) fn launch_compare_bool<T>(
    rt: &CubeclRuntime,
    lhs: &TypedTensor<T>,
    rhs: &TypedTensor<T>,
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
) -> crate::Result<TypedTensor<bool>>
where
    T: CubeElement + Clone,
{
    validate_raw_binary_shapes(lhs, rhs, out_shape, op)?;
    let output = alloc_bool_output(rt, out_shape);
    let len = output.n_elements();
    if len == 0 {
        return Ok(output);
    }
    let client = rt.client();
    let output_arg = bool_tensor_array_arg(&output, op)?;
    let lhs_arg = typed_tensor_array_arg(lhs, op)?;
    let rhs_arg = typed_tensor_array_arg(rhs, op)?;
    // SAFETY: Shape validation proves all raw arrays share the dense element
    // count. Bool output storage uses one byte per element and the kernel
    // guards with `ABSOLUTE_POS < out.len()`.
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
    // SAFETY: Logical tensor kernels receive only `TensorBinding` arguments,
    // each validated against its backing buffer length. Shape/config
    // compatibility is checked by the operation-specific metadata builder
    // before this launch helper is called, and the kernel guards its launched
    // index domain.
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

pub(crate) fn launch_select_bool<T>(
    rt: &CubeclRuntime,
    pred: &TypedTensor<bool>,
    on_true: &TypedTensor<T>,
    on_false: &TypedTensor<T>,
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
) -> crate::Result<TypedTensor<T>>
where
    T: CubeElement + Clone,
{
    validate_raw_ternary_shapes(pred, on_true, on_false, out_shape, op)?;
    let output = alloc_output::<T>(rt, out_shape);
    let len = output.n_elements();
    if len == 0 {
        return Ok(output);
    }
    let client = rt.client();
    let output_arg = typed_tensor_array_arg(&output, op)?;
    let pred_arg = bool_tensor_array_arg(pred, op)?;
    let true_arg = typed_tensor_array_arg(on_true, op)?;
    let false_arg = typed_tensor_array_arg(on_false, op)?;
    // SAFETY: Shape validation proves all raw arrays share the dense element
    // count. The predicate buffer is a validated one-byte Bool tensor buffer,
    // matching the Array<bool> kernel view.
    launch(
        client,
        cube_count_for_len(len),
        cube_dim_1d(),
        output_arg,
        pred_arg,
        true_arg,
        false_arg,
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
    validate_raw_ternary_shapes(a, b, c, out_shape, op)?;
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
    // SAFETY: This helper is the host-side unchecked launch boundary for raw
    // shape-preserving ternary kernels. The shared shape validation above
    // proves all inputs and output have the same dense element count; raw array
    // construction validates each backing buffer length. Kernels launched by
    // this helper guard with `ABSOLUTE_POS < out.len()`.
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
    crate::Error::backend_failure(
        op,
        format!(
            "dtype mismatch first={:?} second={:?} third={:?}",
            first.dtype(),
            second.dtype(),
            third.dtype()
        ),
    )
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

macro_rules! launch_binary_elementwise_kernel {
    ($backend:expr, $lhs:ident, $rhs:ident, $op:expr, $kernel:ident, $scalar:ty, $variant:ident) => {
        launch_binary(
            $backend.runtime(),
            $lhs,
            $rhs,
            $lhs.shape(),
            $op,
            |client, count, dim, out, lhs_arg, rhs_arg| unsafe {
                crate::kernels::elementwise::$kernel::launch_unchecked::<$scalar, CudaRuntime>(
                    client, count, dim, out, lhs_arg, rhs_arg,
                );
            },
        )
        .map(Tensor::$variant)
    };
}

macro_rules! launch_unary_elementwise_kernel {
    ($backend:expr, $input:ident, $op:expr, $kernel:ident, $scalar:ty, $variant:ident) => {
        launch_unary(
            $backend.runtime(),
            $input,
            $input.shape(),
            $op,
            |client, count, dim, out, input_arg| unsafe {
                crate::kernels::elementwise::$kernel::launch_unchecked::<$scalar, CudaRuntime>(
                    client, count, dim, out, input_arg,
                );
            },
        )
        .map(Tensor::$variant)
    };
}

macro_rules! dispatch_binary_float_complex {
    ($backend:expr, $lhs:expr, $rhs:expr, $kind:expr, $float_kernel:ident, $complex_kernel:ident) => {{
        let descriptor = $crate::cubecl::op_descriptor::require_gpu_descriptor(
            $kind,
            $crate::cubecl::op_descriptor::GpuLaunchKind::BinaryFloatComplex,
        )?;
        let op = descriptor.name;
        match ($lhs, $rhs) {
            (Tensor::F32(lhs), Tensor::F32(rhs)) => {
                $crate::cubecl::dispatch::launch_binary_elementwise_kernel!(
                    $backend,
                    lhs,
                    rhs,
                    op,
                    $float_kernel,
                    f32,
                    F32
                )
            }
            (Tensor::F64(lhs), Tensor::F64(rhs)) => {
                $crate::cubecl::dispatch::launch_binary_elementwise_kernel!(
                    $backend,
                    lhs,
                    rhs,
                    op,
                    $float_kernel,
                    f64,
                    F64
                )
            }
            (Tensor::C32(lhs), Tensor::C32(rhs)) => {
                $crate::cubecl::dispatch::launch_binary_elementwise_kernel!(
                    $backend,
                    lhs,
                    rhs,
                    op,
                    $complex_kernel,
                    num_complex::Complex32,
                    C32
                )
            }
            (Tensor::C64(lhs), Tensor::C64(rhs)) => {
                $crate::cubecl::dispatch::launch_binary_elementwise_kernel!(
                    $backend,
                    lhs,
                    rhs,
                    op,
                    $complex_kernel,
                    num_complex::Complex64,
                    C64
                )
            }
            _ => Err(dtype_mismatch(op, $lhs, $rhs)),
        }
    }};
}

macro_rules! dispatch_binary_float_only {
    ($backend:expr, $lhs:expr, $rhs:expr, $kind:expr, $float_kernel:ident) => {{
        let descriptor = $crate::cubecl::op_descriptor::require_gpu_descriptor(
            $kind,
            $crate::cubecl::op_descriptor::GpuLaunchKind::BinaryFloatOnly,
        )?;
        let op = descriptor.name;
        match ($lhs, $rhs) {
            (Tensor::F32(lhs), Tensor::F32(rhs)) => {
                $crate::cubecl::dispatch::launch_binary_elementwise_kernel!(
                    $backend,
                    lhs,
                    rhs,
                    op,
                    $float_kernel,
                    f32,
                    F32
                )
            }
            (Tensor::F64(lhs), Tensor::F64(rhs)) => {
                $crate::cubecl::dispatch::launch_binary_elementwise_kernel!(
                    $backend,
                    lhs,
                    rhs,
                    op,
                    $float_kernel,
                    f64,
                    F64
                )
            }
            (Tensor::C32(_), Tensor::C32(_)) | (Tensor::C64(_), Tensor::C64(_)) => Err(
                crate::Error::backend_failure(op, format!("unsupported dtype {:?}", $lhs.dtype())),
            ),
            _ => Err(dtype_mismatch(op, $lhs, $rhs)),
        }
    }};
}

macro_rules! dispatch_unary_float_complex {
    ($backend:expr, $input:expr, $kind:expr, $float_kernel:ident, $complex_kernel:ident) => {{
        let descriptor = $crate::cubecl::op_descriptor::require_gpu_descriptor(
            $kind,
            $crate::cubecl::op_descriptor::GpuLaunchKind::UnaryFloatComplex,
        )?;
        let op = descriptor.name;
        match $input {
            Tensor::F32(tensor) => {
                $crate::cubecl::dispatch::launch_unary_elementwise_kernel!(
                    $backend,
                    tensor,
                    op,
                    $float_kernel,
                    f32,
                    F32
                )
            }
            Tensor::F64(tensor) => {
                $crate::cubecl::dispatch::launch_unary_elementwise_kernel!(
                    $backend,
                    tensor,
                    op,
                    $float_kernel,
                    f64,
                    F64
                )
            }
            Tensor::C32(tensor) => $crate::cubecl::dispatch::launch_unary_elementwise_kernel!(
                $backend,
                tensor,
                op,
                $complex_kernel,
                num_complex::Complex32,
                C32
            ),
            Tensor::C64(tensor) => $crate::cubecl::dispatch::launch_unary_elementwise_kernel!(
                $backend,
                tensor,
                op,
                $complex_kernel,
                num_complex::Complex64,
                C64
            ),
            Tensor::I32(_) | Tensor::I64(_) | Tensor::Bool(_) => {
                Err(crate::Error::backend_failure(
                    op,
                    format!("unsupported dtype {:?}", $input.dtype()),
                ))
            }
        }
    }};
}

macro_rules! dispatch_unary_float_only {
    ($backend:expr, $input:expr, $kind:expr, $float_kernel:ident) => {{
        let descriptor = $crate::cubecl::op_descriptor::require_gpu_descriptor(
            $kind,
            $crate::cubecl::op_descriptor::GpuLaunchKind::UnaryFloatOnly,
        )?;
        let op = descriptor.name;
        match $input {
            Tensor::F32(tensor) => {
                $crate::cubecl::dispatch::launch_unary_elementwise_kernel!(
                    $backend,
                    tensor,
                    op,
                    $float_kernel,
                    f32,
                    F32
                )
            }
            Tensor::F64(tensor) => {
                $crate::cubecl::dispatch::launch_unary_elementwise_kernel!(
                    $backend,
                    tensor,
                    op,
                    $float_kernel,
                    f64,
                    F64
                )
            }
            _ => Err(crate::Error::backend_failure(
                op,
                format!("unsupported dtype {:?}", $input.dtype()),
            )),
        }
    }};
}

pub(crate) use dispatch_binary_float_complex;
pub(crate) use dispatch_binary_float_only;
pub(crate) use dispatch_unary_float_complex;
pub(crate) use dispatch_unary_float_only;
pub(crate) use launch_binary_elementwise_kernel;
pub(crate) use launch_unary_elementwise_kernel;
