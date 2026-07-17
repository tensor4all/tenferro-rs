use cubecl::client::ComputeClient;
use cubecl::prelude::*;
use cubecl_cuda::CudaRuntime as CubeclCudaRuntime;
use std::sync::Arc;

use crate::config::CompareDir;
use crate::cubecl::CudaRuntime;
use crate::types::{
    Buffer, CubeclBuffer, DeviceId, DeviceKind, GpuBackendKind, MemoryKind, Placement, Tensor,
    TensorRank, TypedTensor, TypedTensorView, TypedTensorViewMut,
};
use tenferro_tensor::{
    CapabilityAxis, CapabilityQuery, DType, TensorBackendCapability as TensorBackendCapabilityTrait,
};

pub(crate) const DEFAULT_CUBE_DIM_X: u32 = 256;

pub(crate) fn cube_count_for_len(len: usize) -> crate::Result<CubeCount> {
    let cubes = len.div_ceil(DEFAULT_CUBE_DIM_X as usize);
    let cubes = u32::try_from(cubes).map_err(|_| {
        crate::Error::invalid_argument(
            "cube_count_for_len",
            "length",
            format!(
                "1D CubeCL launch for {len} elements requires {cubes} cubes, \
                 which exceeds u32::MAX"
            ),
        )
    })?;
    Ok(CubeCount::Static(cubes.max(1), 1, 1))
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

pub(crate) fn cubecl_buffer<'a, T: 'static>(
    tensor: &'a TypedTensor<T, impl TensorRank>,
    op: &'static str,
) -> crate::Result<&'a CubeclBuffer<T>> {
    match tensor.buffer() {
        Buffer::Host(_) => Err(crate::Error::runtime_state(
            op,
            "expected CubeCL GPU tensor, got host tensor. \
                      Use upload_tensor() to transfer to GPU before calling GPU ops.",
        )),
        Buffer::Backend(buffer) => buffer
            .as_any()
            .downcast_ref::<CubeclBuffer<T>>()
            .ok_or_else(|| {
                crate::Error::runtime_state(
                    op,
                    format!(
                        "expected CubeCL GPU tensor, got backend buffer family `{}`",
                        buffer.backend_family()
                    ),
                )
            }),
    }
}

pub(crate) fn cubecl_view_buffer<'a, T: 'static>(
    view: &'a TypedTensorView<'_, T, impl TensorRank>,
    op: &'static str,
) -> crate::Result<&'a CubeclBuffer<T>> {
    let buffer = view.backend_buffer().ok_or_else(|| {
        crate::Error::runtime_state(
            op,
            "expected CubeCL GPU tensor view, got host tensor. \
                      Use upload_tensor() to transfer to GPU before calling GPU ops.",
        )
    })?;
    buffer
        .as_any()
        .downcast_ref::<CubeclBuffer<T>>()
        .ok_or_else(|| {
            crate::Error::runtime_state(
                op,
                format!(
                    "expected CubeCL GPU tensor view, got backend buffer family `{}`",
                    buffer.backend_family()
                ),
            )
        })
}

pub(crate) fn cubecl_view_mut_buffer<'a, T: 'static>(
    view: &'a TypedTensorViewMut<'_, T, impl TensorRank>,
    op: &'static str,
) -> crate::Result<&'a CubeclBuffer<T>> {
    let buffer = view.backend_buffer().ok_or_else(|| {
        crate::Error::runtime_state(
            op,
            "expected CubeCL GPU tensor view, got host tensor. \
                      Use upload_tensor() to transfer to GPU before calling GPU ops.",
        )
    })?;
    buffer
        .as_any()
        .downcast_ref::<CubeclBuffer<T>>()
        .ok_or_else(|| {
            crate::Error::runtime_state(
                op,
                format!(
                    "expected CubeCL GPU tensor view, got backend buffer family `{}`",
                    buffer.backend_family()
                ),
            )
        })
}

pub(crate) fn cubecl_shape_and_strides(shape: &[usize]) -> crate::Result<(Vec<usize>, Vec<usize>)> {
    // CubeCL CUDA kernels still receive a dynamic metadata pointer for tensor
    // args. Rank-0 tensors need one dense metadata element so that launch
    // argument layout stays consistent, while tenferro keeps the public shape
    // as `[]` and passes the logical rank separately where needed.
    if shape.is_empty() {
        return Ok((vec![1], vec![1]));
    }
    let strides = crate::types::col_major_strides(shape)?
        .into_iter()
        // INVARIANT: `col_major_strides` starts from `1isize` and uses
        // checked multiplication over non-negative extents, so strides cannot
        // be negative before this CubeCL metadata conversion.
        .map(|stride| stride as usize)
        .collect();
    Ok((shape.to_vec(), strides))
}

fn checked_shape_product(op: &'static str, shape: &[usize]) -> crate::Result<usize> {
    shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| {
            crate::Error::invalid_argument(
                op,
                "shape",
                format!("shape product overflow for CubeCL tensor shape {shape:?}"),
            )
        })
}

fn validate_cubecl_buffer_len<T>(
    tensor: &TypedTensor<T, impl TensorRank>,
    buffer: &CubeclBuffer<T>,
    op: &'static str,
) -> crate::Result<()> {
    let expected_len = checked_shape_product(op, tensor.shape())?;
    let actual_len = buffer.element_len();
    if expected_len != actual_len {
        return Err(crate::Error::runtime_state(
            op,
            format!(
                "expected shape product {expected_len} elements, actual CubeclBuffer::len {}",
                actual_len
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

pub(crate) fn typed_tensor_binding<T: Clone + 'static>(
    tensor: &TypedTensor<T, impl TensorRank>,
    op: &'static str,
) -> crate::Result<TensorBinding<CubeclCudaRuntime>> {
    let buffer = cubecl_buffer(tensor, op)?;
    validate_cubecl_buffer_len(tensor, buffer, op)?;
    let (shape, strides) = cubecl_shape_and_strides(tensor.shape())?;

    // SAFETY: `buffer.handle()` references the CubeCL allocation for `tensor`.
    // The checked invariant above proves `buffer.element_len()` equals the dense
    // element count of `tensor.shape`; `strides` is the matching dense
    // column-major layout metadata, so kernel indexing stays within that
    // allocation.
    Ok(unsafe {
        TensorBinding::from_raw_parts(buffer.handle().clone(), strides.into(), shape.into())
    })
}

pub(crate) fn launch_unary_bool_tensor(
    rt: &CudaRuntime,
    input: &TypedTensor<bool>,
    out_shape: &[usize],
    op: &'static str,
    launch: impl FnOnce(
        &ComputeClient<CubeclCudaRuntime>,
        CubeCount,
        CubeDim,
        TensorBinding<CubeclCudaRuntime>,
        TensorBinding<CubeclCudaRuntime>,
    ),
) -> crate::Result<TypedTensor<bool>> {
    ensure_resident_on_runtime(rt, input, op)?;
    let input_arg = typed_tensor_binding(input, op)?;
    let output_len = checked_shape_product(op, out_shape)?;
    let launch_count = if output_len == 0 {
        None
    } else {
        Some(cube_count_for_len(output_len)?)
    };
    let output = alloc_bool_output(rt, out_shape)?;
    let Some(launch_count) = launch_count else {
        return Ok(output);
    };
    let output_arg = typed_tensor_binding(&output, op)?;
    launch(
        rt.client(),
        launch_count,
        cube_dim_1d(),
        output_arg,
        input_arg,
    );
    Ok(output)
}

pub(crate) fn launch_binary_bool_tensor<I: CubeElement + Clone>(
    rt: &CudaRuntime,
    input: &TypedTensor<bool>,
    indices: &TypedTensor<I>,
    out_shape: &[usize],
    op: &'static str,
    launch: impl FnOnce(
        &ComputeClient<CubeclCudaRuntime>,
        CubeCount,
        CubeDim,
        TensorBinding<CubeclCudaRuntime>,
        TensorBinding<CubeclCudaRuntime>,
        TensorBinding<CubeclCudaRuntime>,
    ),
) -> crate::Result<TypedTensor<bool>> {
    ensure_resident_on_runtime(rt, input, op)?;
    ensure_resident_on_runtime(rt, indices, op)?;
    let input_arg = typed_tensor_binding(input, op)?;
    let indices_arg = typed_tensor_binding(indices, op)?;
    let output_len = checked_shape_product(op, out_shape)?;
    let launch_count = if output_len == 0 {
        None
    } else {
        Some(cube_count_for_len(output_len)?)
    };
    let output = alloc_bool_output(rt, out_shape)?;
    let Some(launch_count) = launch_count else {
        return Ok(output);
    };
    let output_arg = typed_tensor_binding(&output, op)?;
    launch(
        rt.client(),
        launch_count,
        cube_dim_1d(),
        output_arg,
        input_arg,
        indices_arg,
    );
    Ok(output)
}

pub(crate) fn launch_bool_tensor_into(
    rt: &CudaRuntime,
    output: &TypedTensor<bool>,
    input: &TypedTensor<bool>,
    op: &'static str,
    count: CubeCount,
    dim: CubeDim,
    launch: impl FnOnce(
        &ComputeClient<CubeclCudaRuntime>,
        CubeCount,
        CubeDim,
        TensorBinding<CubeclCudaRuntime>,
        TensorBinding<CubeclCudaRuntime>,
    ),
) -> crate::Result<()> {
    ensure_resident_on_runtime(rt, output, op)?;
    ensure_resident_on_runtime(rt, input, op)?;
    let output_arg = typed_tensor_binding(output, op)?;
    let input_arg = typed_tensor_binding(input, op)?;
    if output.n_elements() != 0 {
        launch(rt.client(), count, dim, output_arg, input_arg);
    }
    Ok(())
}

pub(crate) fn launch_nullary_bool_into(
    rt: &CudaRuntime,
    output: &TypedTensor<bool>,
    op: &'static str,
    count: CubeCount,
    dim: CubeDim,
    launch: impl FnOnce(
        &ComputeClient<CubeclCudaRuntime>,
        CubeCount,
        CubeDim,
        ArrayArg<CubeclCudaRuntime>,
    ),
) -> crate::Result<()> {
    ensure_resident_on_runtime(rt, output, op)?;
    let output_arg = bool_tensor_array_arg(output, op)?;
    if output.n_elements() != 0 {
        launch(rt.client(), count, dim, output_arg);
    }
    Ok(())
}

pub(crate) fn ensure_resident_on_runtime<T: 'static>(
    rt: &CudaRuntime,
    tensor: &TypedTensor<T, impl TensorRank>,
    op: &'static str,
) -> crate::Result<()> {
    cubecl_buffer(tensor, op)?;
    ensure_placement_resident_on_runtime(rt, tensor.placement(), op)
}

pub(crate) fn ensure_view_resident_on_runtime<T: 'static>(
    rt: &CudaRuntime,
    view: &TypedTensorView<'_, T, impl TensorRank>,
    op: &'static str,
) -> crate::Result<()> {
    cubecl_view_buffer(view, op)?;
    ensure_placement_resident_on_runtime(rt, view.placement(), op)
}

pub(crate) fn ensure_view_mut_resident_on_runtime<T: 'static>(
    rt: &CudaRuntime,
    view: &TypedTensorViewMut<'_, T, impl TensorRank>,
    op: &'static str,
) -> crate::Result<()> {
    cubecl_view_mut_buffer(view, op)?;
    ensure_placement_resident_on_runtime(rt, view.placement(), op)
}

fn ensure_placement_resident_on_runtime(
    rt: &CudaRuntime,
    placement: &Placement,
    op: &'static str,
) -> crate::Result<()> {
    if !matches!(&placement.memory_kind, MemoryKind::Device) {
        return Err(crate::Error::runtime_state(
            op,
            format!(
                "expected GPU tensor placement, got {:?}",
                placement.memory_kind
            ),
        ));
    }
    match &placement.device {
        Some(device)
            if device.kind == DeviceKind::Gpu(GpuBackendKind::Cuda)
                && device.ordinal == rt.device_ordinal() =>
        {
            Ok(())
        }
        Some(device) => Err(crate::Error::runtime_state(
            op,
            format!(
                "expected GPU tensor resident on cuda:{}, got {:?}:{}",
                rt.device_ordinal(),
                device.kind,
                device.ordinal
            ),
        )),
        None => Err(crate::Error::runtime_state(
            op,
            format!(
                "expected GPU tensor resident on cuda:{}, got missing device metadata",
                rt.device_ordinal()
            ),
        )),
    }
}

pub(crate) fn typed_from_cubecl<T: Send + Sync + 'static>(
    shape: Vec<usize>,
    buffer: CubeclBuffer<T>,
    device_ordinal: usize,
) -> crate::Result<TypedTensor<T>> {
    TypedTensor::from_buffer_col_major(
        shape,
        Buffer::Backend(Arc::new(buffer)),
        Placement {
            memory_kind: MemoryKind::Device,
            device: Some(DeviceId {
                kind: DeviceKind::Gpu(GpuBackendKind::Cuda),
                ordinal: device_ordinal,
            }),
        },
    )
}

pub(crate) fn alloc_output<T: CubeElement + Clone + Send + Sync + 'static>(
    rt: &CudaRuntime,
    shape: &[usize],
) -> crate::Result<TypedTensor<T>> {
    let len = checked_shape_product("cubecl_alloc_output", shape)?;
    let byte_len = len.checked_mul(core::mem::size_of::<T>()).ok_or_else(|| {
        crate::Error::invalid_argument(
            "cubecl_alloc_output",
            "shape",
            format!("output byte length overflow for shape {shape:?}"),
        )
    })?;
    let handle = rt.client().empty(byte_len);
    typed_from_cubecl(
        shape.to_vec(),
        CubeclBuffer::new(handle, len),
        rt.device_ordinal(),
    )
}

pub(crate) fn alloc_bool_output(
    rt: &CudaRuntime,
    shape: &[usize],
) -> crate::Result<TypedTensor<bool>> {
    let len = checked_shape_product("cubecl_alloc_bool_output", shape)?;
    let handle = rt.client().empty(len);
    typed_from_cubecl(
        shape.to_vec(),
        CubeclBuffer::new(handle, len),
        rt.device_ordinal(),
    )
}

pub(crate) fn typed_tensor_array_arg<T: CubeElement + Clone>(
    tensor: &TypedTensor<T, impl TensorRank>,
    op: &'static str,
) -> crate::Result<ArrayArg<CubeclCudaRuntime>> {
    let buffer = cubecl_buffer(tensor, op)?;
    validate_cubecl_buffer_len(tensor, buffer, op)?;

    // SAFETY: `buffer.handle()` references the CubeCL allocation for `tensor`.
    // `validate_cubecl_buffer_len` proves `buffer.element_len()` equals the dense shape
    // product, so raw linear kernels that index `0..out.len()` cannot observe
    // an array longer than the logical tensor allocation.
    Ok(unsafe { ArrayArg::from_raw_parts(buffer.handle().clone(), buffer.element_len()) })
}

pub(crate) fn typed_view_array_arg<T: CubeElement + Clone>(
    view: &TypedTensorView<'_, T, impl TensorRank>,
    op: &'static str,
) -> crate::Result<ArrayArg<CubeclCudaRuntime>> {
    let buffer = cubecl_view_buffer(view, op)?;

    // SAFETY: `TensorLayout` validation at view construction proves the
    // reachable logical offsets are within this backing allocation. Kernels
    // using this raw array also receive the validated signed layout metadata
    // because CubeCL TensorBinding cannot express signed view strides.
    Ok(unsafe { ArrayArg::from_raw_parts(buffer.handle().clone(), buffer.element_len()) })
}

pub(crate) fn typed_view_mut_array_arg<T: CubeElement + Clone>(
    view: &TypedTensorViewMut<'_, T, impl TensorRank>,
    op: &'static str,
) -> crate::Result<ArrayArg<CubeclCudaRuntime>> {
    let buffer = cubecl_view_mut_buffer(view, op)?;

    // SAFETY: `TypedTensorViewMut` construction validates both reachable
    // offsets and no-overlap. The raw array length covers the backing
    // allocation, while the kernel launch domain covers only the logical view.
    Ok(unsafe { ArrayArg::from_raw_parts(buffer.handle().clone(), buffer.element_len()) })
}

pub(crate) fn bool_tensor_array_arg(
    tensor: &TypedTensor<bool, impl TensorRank>,
    op: &'static str,
) -> crate::Result<ArrayArg<CubeclCudaRuntime>> {
    let buffer = cubecl_buffer(tensor, op)?;
    validate_cubecl_buffer_len(tensor, buffer, op)?;

    // SAFETY: CubeCL bool tensors are stored as one-byte predicate buffers by
    // `memory::upload_bool` and `alloc_bool_output`. The validated buffer
    // length is the logical element count consumed by raw Array<bool> kernels.
    Ok(unsafe { ArrayArg::from_raw_parts(buffer.handle().clone(), buffer.element_len()) })
}

pub(crate) fn typed_tensor_array_arg_as<T, U>(
    tensor: &TypedTensor<T, impl TensorRank>,
    len: usize,
    op: &'static str,
) -> crate::Result<ArrayArg<CubeclCudaRuntime>>
where
    T: CubeElement + Clone,
    U: CubeElement + Clone,
{
    let buffer = cubecl_buffer(tensor, op)?;
    validate_cubecl_buffer_len(tensor, buffer, op)?;
    let requested_bytes = len.checked_mul(core::mem::size_of::<U>()).ok_or_else(|| {
        crate::Error::invalid_argument(
            op,
            "length",
            format!("reinterpreted CubeCL array length overflow for len {len}"),
        )
    })?;
    let available_bytes = buffer
        .element_len()
        .checked_mul(core::mem::size_of::<T>())
        .ok_or_else(|| {
            crate::Error::runtime_state(
                op,
                format!(
                    "CubeCL buffer byte length overflow for {} elements",
                    buffer.element_len()
                ),
            )
        })?;
    if requested_bytes > available_bytes {
        return Err(crate::Error::runtime_state(op, format!(
                "reinterpreted CubeCL array needs {requested_bytes} bytes, buffer has {available_bytes}"
            )));
    }

    // SAFETY: `validate_cubecl_buffer_len` first proves the typed tensor shape
    // matches the backing allocation. The checked byte-size invariant then
    // proves the requested representation view stays within the same
    // allocation. Kernels using this helper are responsible for using a
    // representation-compatible scalar view, for example complex values as
    // their real and imaginary scalar parts.
    Ok(unsafe { ArrayArg::from_raw_parts(buffer.handle().clone(), len) })
}

pub(crate) fn launch_unary<TIn, TOut>(
    rt: &CudaRuntime,
    input: &TypedTensor<TIn>,
    out_shape: &[usize],
    op: &'static str,
    launch: impl FnOnce(
        &ComputeClient<CubeclCudaRuntime>,
        CubeCount,
        CubeDim,
        ArrayArg<CubeclCudaRuntime>,
        ArrayArg<CubeclCudaRuntime>,
    ),
) -> crate::Result<TypedTensor<TOut>>
where
    TIn: CubeElement + Clone,
    TOut: CubeElement + Clone,
{
    validate_raw_unary_shapes(input, out_shape, op)?;
    let output = alloc_output::<TOut>(rt, out_shape)?;
    let len = output.n_elements();
    ensure_resident_on_runtime(rt, input, op)?;
    let input_arg = typed_tensor_array_arg(input, op)?;
    if len == 0 {
        return Ok(output);
    }
    let client = rt.client();
    let output_arg = typed_tensor_array_arg(&output, op)?;
    // SAFETY: This helper is the host-side unchecked launch boundary for raw
    // shape-preserving unary kernels. The shape validation above proves input
    // and output have the same dense element count; `typed_tensor_array_arg`
    // proves each raw array length matches its tensor shape. The launch domain
    // covers `len == output.n_elements()`, and these kernels guard writes with
    // `ABSOLUTE_POS < out.len()`.
    launch(
        client,
        cube_count_for_len(len)?,
        cube_dim_1d(),
        output_arg,
        input_arg,
    );
    Ok(output)
}

pub(crate) fn launch_unary_tensor<TIn, TOut>(
    rt: &CudaRuntime,
    input: &TypedTensor<TIn>,
    out_shape: &[usize],
    op: &'static str,
    launch: impl FnOnce(
        &ComputeClient<CubeclCudaRuntime>,
        CubeCount,
        CubeDim,
        TensorBinding<CubeclCudaRuntime>,
        TensorBinding<CubeclCudaRuntime>,
    ),
) -> crate::Result<TypedTensor<TOut>>
where
    TIn: CubeElement + Clone,
    TOut: CubeElement + Clone,
{
    let output = alloc_output::<TOut>(rt, out_shape)?;
    let len = output.n_elements();
    ensure_resident_on_runtime(rt, input, op)?;
    let input_arg = typed_tensor_binding(input, op)?;
    if len == 0 {
        return Ok(output);
    }
    let client = rt.client();
    let output_arg = typed_tensor_binding(&output, op)?;
    // SAFETY: Logical tensor kernels use `TensorBinding`, whose construction
    // validates shape product against the CubeCL allocation length. The caller
    // supplies output shape and launch metadata already validated for the
    // specific structural/indexing operation, and the kernels guard their
    // launched index domain before mapping logical indices.
    launch(
        client,
        cube_count_for_len(len)?,
        cube_dim_1d(),
        output_arg,
        input_arg,
    );
    Ok(output)
}

pub(crate) fn launch_nullary_into<TOut>(
    rt: &CudaRuntime,
    output: &TypedTensor<TOut>,
    op: &'static str,
    count: CubeCount,
    dim: CubeDim,
    launch: impl FnOnce(
        &ComputeClient<CubeclCudaRuntime>,
        CubeCount,
        CubeDim,
        ArrayArg<CubeclCudaRuntime>,
    ),
) -> crate::Result<()>
where
    TOut: CubeElement + Clone,
{
    ensure_resident_on_runtime(rt, output, op)?;
    let output_arg = typed_tensor_array_arg(output, op)?;
    if output.n_elements() == 0 {
        return Ok(());
    }
    // SAFETY: Nullary raw kernels write only to the validated output array.
    // The caller-supplied `count`/`dim` must describe the initialized domain,
    // and kernels using this path guard with `ABSOLUTE_POS < out.len()`.
    launch(rt.client(), count, dim, output_arg);
    Ok(())
}

pub(crate) fn launch_unary_tensor_into<TIn, TOut>(
    rt: &CudaRuntime,
    output: &TypedTensor<TOut>,
    input: &TypedTensor<TIn>,
    op: &'static str,
    count: CubeCount,
    dim: CubeDim,
    launch: impl FnOnce(
        &ComputeClient<CubeclCudaRuntime>,
        CubeCount,
        CubeDim,
        TensorBinding<CubeclCudaRuntime>,
        TensorBinding<CubeclCudaRuntime>,
    ),
) -> crate::Result<()>
where
    TIn: CubeElement + Clone,
    TOut: CubeElement + Clone,
{
    ensure_resident_on_runtime(rt, output, op)?;
    ensure_resident_on_runtime(rt, input, op)?;
    let output_arg = typed_tensor_binding(output, op)?;
    let input_arg = typed_tensor_binding(input, op)?;
    if output.n_elements() == 0 {
        return Ok(());
    }
    // SAFETY: `TensorBinding` construction validates shape and backing buffer
    // length for both tensors. The caller supplies a launch domain derived
    // from validated operation metadata, and the target kernel guards the
    // output or update domain before logical tensor indexing.
    launch(rt.client(), count, dim, output_arg, input_arg);
    Ok(())
}

pub(crate) fn launch_binary<TLhs, TRhs, TOut>(
    rt: &CudaRuntime,
    lhs: &TypedTensor<TLhs>,
    rhs: &TypedTensor<TRhs>,
    out_shape: &[usize],
    op: &'static str,
    launch: impl FnOnce(
        &ComputeClient<CubeclCudaRuntime>,
        CubeCount,
        CubeDim,
        ArrayArg<CubeclCudaRuntime>,
        ArrayArg<CubeclCudaRuntime>,
        ArrayArg<CubeclCudaRuntime>,
    ),
) -> crate::Result<TypedTensor<TOut>>
where
    TLhs: CubeElement + Clone,
    TRhs: CubeElement + Clone,
    TOut: CubeElement + Clone,
{
    validate_raw_binary_shapes(lhs, rhs, out_shape, op)?;
    let output = alloc_output::<TOut>(rt, out_shape)?;
    let len = output.n_elements();
    ensure_resident_on_runtime(rt, lhs, op)?;
    ensure_resident_on_runtime(rt, rhs, op)?;
    let lhs_arg = typed_tensor_array_arg(lhs, op)?;
    let rhs_arg = typed_tensor_array_arg(rhs, op)?;
    if len == 0 {
        return Ok(output);
    }
    let client = rt.client();
    let output_arg = typed_tensor_array_arg(&output, op)?;
    // SAFETY: This helper is the host-side unchecked launch boundary for raw
    // shape-preserving binary kernels. The shared shape validation above
    // proves all arrays have the same dense element count; the raw array
    // helpers prove every CubeCL buffer length matches its tensor shape. The
    // launch covers `len == output.n_elements()`, and elementwise kernels guard
    // with `ABSOLUTE_POS < out.len()`.
    launch(
        client,
        cube_count_for_len(len)?,
        cube_dim_1d(),
        output_arg,
        lhs_arg,
        rhs_arg,
    );
    Ok(output)
}

pub(crate) fn launch_compare_bool<T>(
    rt: &CudaRuntime,
    lhs: &TypedTensor<T>,
    rhs: &TypedTensor<T>,
    out_shape: &[usize],
    op: &'static str,
    launch: impl FnOnce(
        &ComputeClient<CubeclCudaRuntime>,
        CubeCount,
        CubeDim,
        ArrayArg<CubeclCudaRuntime>,
        ArrayArg<CubeclCudaRuntime>,
        ArrayArg<CubeclCudaRuntime>,
    ),
) -> crate::Result<TypedTensor<bool>>
where
    T: CubeElement + Clone,
{
    validate_raw_binary_shapes(lhs, rhs, out_shape, op)?;
    let output = alloc_bool_output(rt, out_shape)?;
    let len = output.n_elements();
    ensure_resident_on_runtime(rt, lhs, op)?;
    ensure_resident_on_runtime(rt, rhs, op)?;
    let lhs_arg = typed_tensor_array_arg(lhs, op)?;
    let rhs_arg = typed_tensor_array_arg(rhs, op)?;
    if len == 0 {
        return Ok(output);
    }
    let client = rt.client();
    let output_arg = bool_tensor_array_arg(&output, op)?;
    // SAFETY: Shape validation proves all raw arrays share the dense element
    // count. Bool output storage uses one byte per element and the kernel
    // guards with `ABSOLUTE_POS < out.len()`.
    launch(
        client,
        cube_count_for_len(len)?,
        cube_dim_1d(),
        output_arg,
        lhs_arg,
        rhs_arg,
    );
    Ok(output)
}

pub(crate) fn launch_binary_tensor<TLhs, TRhs, TOut>(
    rt: &CudaRuntime,
    lhs: &TypedTensor<TLhs>,
    rhs: &TypedTensor<TRhs>,
    out_shape: &[usize],
    op: &'static str,
    launch: impl FnOnce(
        &ComputeClient<CubeclCudaRuntime>,
        CubeCount,
        CubeDim,
        TensorBinding<CubeclCudaRuntime>,
        TensorBinding<CubeclCudaRuntime>,
        TensorBinding<CubeclCudaRuntime>,
    ),
) -> crate::Result<TypedTensor<TOut>>
where
    TLhs: CubeElement + Clone,
    TRhs: CubeElement + Clone,
    TOut: CubeElement + Clone,
{
    let output = alloc_output::<TOut>(rt, out_shape)?;
    let len = output.n_elements();
    ensure_resident_on_runtime(rt, lhs, op)?;
    ensure_resident_on_runtime(rt, rhs, op)?;
    let lhs_arg = typed_tensor_binding(lhs, op)?;
    let rhs_arg = typed_tensor_binding(rhs, op)?;
    if len == 0 {
        return Ok(output);
    }
    let client = rt.client();
    let output_arg = typed_tensor_binding(&output, op)?;
    // SAFETY: Logical tensor kernels receive only `TensorBinding` arguments,
    // each validated against its backing buffer length. Shape/config
    // compatibility is checked by the operation-specific metadata builder
    // before this launch helper is called, and the kernel guards its launched
    // index domain.
    launch(
        client,
        cube_count_for_len(len)?,
        cube_dim_1d(),
        output_arg,
        lhs_arg,
        rhs_arg,
    );
    Ok(output)
}

pub(crate) fn launch_select_bool<T>(
    rt: &CudaRuntime,
    pred: &TypedTensor<bool>,
    on_true: &TypedTensor<T>,
    on_false: &TypedTensor<T>,
    out_shape: &[usize],
    op: &'static str,
    launch: impl FnOnce(
        &ComputeClient<CubeclCudaRuntime>,
        CubeCount,
        CubeDim,
        ArrayArg<CubeclCudaRuntime>,
        ArrayArg<CubeclCudaRuntime>,
        ArrayArg<CubeclCudaRuntime>,
        ArrayArg<CubeclCudaRuntime>,
    ),
) -> crate::Result<TypedTensor<T>>
where
    T: CubeElement + Clone,
{
    validate_raw_ternary_shapes(pred, on_true, on_false, out_shape, op)?;
    let output = alloc_output::<T>(rt, out_shape)?;
    let len = output.n_elements();
    ensure_resident_on_runtime(rt, pred, op)?;
    ensure_resident_on_runtime(rt, on_true, op)?;
    ensure_resident_on_runtime(rt, on_false, op)?;
    let pred_arg = bool_tensor_array_arg(pred, op)?;
    let true_arg = typed_tensor_array_arg(on_true, op)?;
    let false_arg = typed_tensor_array_arg(on_false, op)?;
    if len == 0 {
        return Ok(output);
    }
    let client = rt.client();
    let output_arg = typed_tensor_array_arg(&output, op)?;
    // SAFETY: Shape validation proves all raw arrays share the dense element
    // count. The predicate buffer is a validated one-byte Bool tensor buffer,
    // matching the Array<bool> kernel view.
    launch(
        client,
        cube_count_for_len(len)?,
        cube_dim_1d(),
        output_arg,
        pred_arg,
        true_arg,
        false_arg,
    );
    Ok(output)
}

pub(crate) fn launch_ternary<TA, TB, TC, TOut>(
    rt: &CudaRuntime,
    a: &TypedTensor<TA>,
    b: &TypedTensor<TB>,
    c: &TypedTensor<TC>,
    out_shape: &[usize],
    op: &'static str,
    launch: impl FnOnce(
        &ComputeClient<CubeclCudaRuntime>,
        CubeCount,
        CubeDim,
        ArrayArg<CubeclCudaRuntime>,
        ArrayArg<CubeclCudaRuntime>,
        ArrayArg<CubeclCudaRuntime>,
        ArrayArg<CubeclCudaRuntime>,
    ),
) -> crate::Result<TypedTensor<TOut>>
where
    TA: CubeElement + Clone,
    TB: CubeElement + Clone,
    TC: CubeElement + Clone,
    TOut: CubeElement + Clone,
{
    validate_raw_ternary_shapes(a, b, c, out_shape, op)?;
    let output = alloc_output::<TOut>(rt, out_shape)?;
    let len = output.n_elements();
    ensure_resident_on_runtime(rt, a, op)?;
    ensure_resident_on_runtime(rt, b, op)?;
    ensure_resident_on_runtime(rt, c, op)?;
    let a_arg = typed_tensor_array_arg(a, op)?;
    let b_arg = typed_tensor_array_arg(b, op)?;
    let c_arg = typed_tensor_array_arg(c, op)?;
    if len == 0 {
        return Ok(output);
    }
    let client = rt.client();
    let output_arg = typed_tensor_array_arg(&output, op)?;
    // SAFETY: This helper is the host-side unchecked launch boundary for raw
    // shape-preserving ternary kernels. The shared shape validation above
    // proves all inputs and output have the same dense element count; raw array
    // construction validates each backing buffer length. Kernels launched by
    // this helper guard with `ABSOLUTE_POS < out.len()`.
    launch(
        client,
        cube_count_for_len(len)?,
        cube_dim_1d(),
        output_arg,
        a_arg,
        b_arg,
        c_arg,
    );
    Ok(output)
}

pub(crate) fn dtype_mismatch(op: &'static str, lhs: &Tensor, rhs: &Tensor) -> crate::Error {
    crate::Error::dtype_mismatch(op, lhs.dtype(), rhs.dtype())
}

pub(crate) fn ternary_dtype_mismatch(
    op: &'static str,
    first: &Tensor,
    second: &Tensor,
    third: &Tensor,
) -> crate::Error {
    let (expected, actual) = if first.dtype() != second.dtype() {
        (first.dtype(), second.dtype())
    } else {
        (first.dtype(), third.dtype())
    };
    crate::Error::dtype_mismatch(op, expected, actual)
}

pub(crate) fn ensure_same_shape(
    op: &'static str,
    lhs: &[usize],
    rhs: &[usize],
) -> crate::Result<()> {
    if lhs != rhs {
        return Err(crate::Error::shape_mismatch(op, lhs.to_vec(), rhs.to_vec()));
    }
    Ok(())
}

pub(crate) fn ensure_rank(op: &'static str, expected: usize, actual: usize) -> crate::Result<()> {
    if expected != actual {
        return Err(crate::Error::rank_mismatch(op, expected, actual));
    }
    Ok(())
}

pub(crate) fn ensure_axis(op: &'static str, axis: usize, rank: usize) -> crate::Result<()> {
    if axis >= rank {
        return Err(crate::Error::axis_out_of_bounds(op, axis, rank));
    }
    Ok(())
}

pub(crate) fn require_owned_capability<B>(
    backend: &B,
    kind: tenferro_core_ops::PrimitiveOpKind,
    dtype: DType,
) -> crate::Result<()>
where
    B: TensorBackendCapabilityTrait + ?Sized,
{
    backend
        .require_capability(
            CapabilityQuery::new(kind, dtype),
            CapabilityAxis::OwnedResult,
        )
        .map(|_| ())
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
            return Err(crate::Error::duplicate_axis(op, axis, role));
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
                crate::kernels::elementwise::$kernel::launch_unchecked::<$scalar, CubeclCudaRuntime>(
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
                crate::kernels::elementwise::$kernel::launch_unchecked::<$scalar, CubeclCudaRuntime>(
                    client, count, dim, out, input_arg,
                );
            },
        )
        .map(Tensor::$variant)
    };
}

macro_rules! dispatch_binary_float_complex_int {
    ($backend:expr, $lhs:expr, $rhs:expr, $kind:expr, $float_kernel:ident, $int_kernel:ident, $complex_kernel:ident) => {{
        let descriptor = $crate::cubecl::op_descriptor::require_gpu_descriptor(
            $kind,
            $crate::cubecl::op_descriptor::GpuLaunchKind::BinaryFloatComplexInt,
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
            (Tensor::I32(lhs), Tensor::I32(rhs)) => {
                $crate::cubecl::dispatch::launch_binary_elementwise_kernel!(
                    $backend,
                    lhs,
                    rhs,
                    op,
                    $int_kernel,
                    i32,
                    I32
                )
            }
            (Tensor::I64(lhs), Tensor::I64(rhs)) => {
                $crate::cubecl::dispatch::launch_binary_elementwise_kernel!(
                    $backend,
                    lhs,
                    rhs,
                    op,
                    $int_kernel,
                    i64,
                    I64
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

macro_rules! dispatch_binary_float_int {
    ($backend:expr, $lhs:expr, $rhs:expr, $kind:expr, $float_kernel:ident, $int_kernel:ident) => {{
        let descriptor = $crate::cubecl::op_descriptor::require_gpu_descriptor(
            $kind,
            $crate::cubecl::op_descriptor::GpuLaunchKind::BinaryFloatInt,
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
            (Tensor::I32(lhs), Tensor::I32(rhs)) => {
                $crate::cubecl::dispatch::launch_binary_elementwise_kernel!(
                    $backend,
                    lhs,
                    rhs,
                    op,
                    $int_kernel,
                    i32,
                    I32
                )
            }
            (Tensor::I64(lhs), Tensor::I64(rhs)) => {
                $crate::cubecl::dispatch::launch_binary_elementwise_kernel!(
                    $backend,
                    lhs,
                    rhs,
                    op,
                    $int_kernel,
                    i64,
                    I64
                )
            }
            (Tensor::C32(_), Tensor::C32(_)) | (Tensor::C64(_), Tensor::C64(_)) => {
                Err($crate::cubecl::unsupported_dtype(op, $lhs.dtype()))
            }
            _ => Err(dtype_mismatch(op, $lhs, $rhs)),
        }
    }};
}

macro_rules! dispatch_unary_float_complex_int {
    ($backend:expr, $input:expr, $kind:expr, $float_kernel:ident, $int_kernel:ident, $complex_kernel:ident) => {{
        let descriptor = $crate::cubecl::op_descriptor::require_gpu_descriptor(
            $kind,
            $crate::cubecl::op_descriptor::GpuLaunchKind::UnaryFloatComplexInt,
        )?;
        let op = descriptor.name;
        let input = $input;
        $crate::cubecl::dispatch::require_owned_capability($backend, $kind, input.dtype())?;
        match input {
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
            Tensor::I32(tensor) => {
                $crate::cubecl::dispatch::launch_unary_elementwise_kernel!(
                    $backend,
                    tensor,
                    op,
                    $int_kernel,
                    i32,
                    I32
                )
            }
            Tensor::I64(tensor) => {
                $crate::cubecl::dispatch::launch_unary_elementwise_kernel!(
                    $backend,
                    tensor,
                    op,
                    $int_kernel,
                    i64,
                    I64
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
            Tensor::Bool(_) => Err($crate::cubecl::unsupported_dtype(op, input.dtype())),
        }
    }};
}

macro_rules! dispatch_unary_float_int {
    ($backend:expr, $input:expr, $kind:expr, $float_kernel:ident, $int_kernel:ident) => {{
        let descriptor = $crate::cubecl::op_descriptor::require_gpu_descriptor(
            $kind,
            $crate::cubecl::op_descriptor::GpuLaunchKind::UnaryFloatInt,
        )?;
        let op = descriptor.name;
        let input = $input;
        $crate::cubecl::dispatch::require_owned_capability($backend, $kind, input.dtype())?;
        match input {
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
            Tensor::I32(tensor) => {
                $crate::cubecl::dispatch::launch_unary_elementwise_kernel!(
                    $backend,
                    tensor,
                    op,
                    $int_kernel,
                    i32,
                    I32
                )
            }
            Tensor::I64(tensor) => {
                $crate::cubecl::dispatch::launch_unary_elementwise_kernel!(
                    $backend,
                    tensor,
                    op,
                    $int_kernel,
                    i64,
                    I64
                )
            }
            Tensor::Bool(_) | Tensor::C32(_) | Tensor::C64(_) => {
                Err($crate::cubecl::unsupported_dtype(op, input.dtype()))
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
        let input = $input;
        $crate::cubecl::dispatch::require_owned_capability($backend, $kind, input.dtype())?;
        match input {
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
            _ => Err($crate::cubecl::unsupported_dtype(op, input.dtype())),
        }
    }};
}

pub(crate) use dispatch_binary_float_complex_int;
pub(crate) use dispatch_binary_float_int;
pub(crate) use dispatch_unary_float_complex_int;
pub(crate) use dispatch_unary_float_int;
pub(crate) use dispatch_unary_float_only;
pub(crate) use launch_binary_elementwise_kernel;
pub(crate) use launch_unary_elementwise_kernel;
