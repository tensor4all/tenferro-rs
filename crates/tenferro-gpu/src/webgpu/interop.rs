use cubecl::client::ComputeClient;
use cubecl::frontend::CubePrimitive;
use cubecl::std::tensor::TensorHandle;
use cubecl_runtime::server::Handle;
use cubecl_wgpu::WgpuRuntime;
use num_complex::Complex32;
use tenferro_tensor::{Error, TypedTensor};

use super::{
    checked_shape_product, ensure_resident_on_runtime, typed_from_webgpu, webgpu_buffer,
    WebGpuBuffer, WebGpuExecSession,
};

/// Return the exact client owned by `backend` for an extension launch.
pub fn client<'a>(session: &'a WebGpuExecSession<'a>) -> &'a ComputeClient<WgpuRuntime> {
    session.runtime().client()
}

/// Return the active client's hardware-reported shared-memory budget.
pub fn max_shared_memory_size(session: &WebGpuExecSession<'_>) -> usize {
    client(session).properties().hardware.max_shared_memory_size
}

/// Return the active client's maximum number of units per cube.
pub fn max_units_per_cube(session: &WebGpuExecSession<'_>) -> u32 {
    client(session).properties().hardware.max_units_per_cube
}

/// Validate and clone an F32 tensor into CubeCL launch metadata.
///
/// # Errors
///
/// Returns [`tenferro_tensor::Error::Unsupported`] for noncompact or negative
/// layouts, [`tenferro_tensor::Error::HostAccess`] for a foreign managed domain,
/// and validation or runtime-state errors for incompatible placement, shape,
/// buffer, or stride metadata.
pub fn f32_input(
    session: &WebGpuExecSession<'_>,
    tensor: &TypedTensor<f32>,
    op: &'static str,
) -> tenferro_tensor::Result<TensorHandle<WgpuRuntime>> {
    let (handle, shape, strides) = input_parts(session, tensor, op)?;
    Ok(TensorHandle::new(
        handle,
        shape,
        strides,
        f32::as_type_native_unchecked().storage_type(),
    ))
}

/// Validate and clone the raw allocation plus logical metadata of a C32 tensor.
///
/// # Errors
///
/// Returns [`tenferro_tensor::Error::Unsupported`] for noncompact or negative
/// layouts, [`tenferro_tensor::Error::HostAccess`] for a foreign managed domain,
/// and validation or runtime-state errors for incompatible placement, shape,
/// buffer, or stride metadata.
pub fn c32_input_parts(
    session: &WebGpuExecSession<'_>,
    tensor: &TypedTensor<Complex32>,
    op: &'static str,
) -> tenferro_tensor::Result<(Handle, Vec<usize>, Vec<usize>)> {
    input_parts(session, tensor, op)
}

/// Allocate one unaliased output range on the backend's exact client.
///
/// CubeCL's pool may represent the returned range with start or end offsets;
/// completion validates the used range rather than assuming a whole page.
pub fn allocate_raw(session: &WebGpuExecSession<'_>, bytes: usize) -> Handle {
    client(session).empty(bytes)
}

/// Consume an initialized, exactly sized F32 handle range into a tenferro tensor.
///
/// # Errors
///
/// Returns a validation or runtime-state error when shape-byte arithmetic
/// overflows, the handle range is invalid, misaligned, incorrectly sized, or
/// still has another live raw owner. Backend resource-resolution errors retain
/// their typed source.
pub fn finish_f32(
    session: &WebGpuExecSession<'_>,
    shape: Vec<usize>,
    handle: Handle,
    op: &'static str,
) -> tenferro_tensor::Result<TypedTensor<f32>> {
    finish(session, shape, handle, op)
}

/// Consume an initialized, exactly sized C32 handle range into a tenferro tensor.
///
/// # Errors
///
/// Returns a validation or runtime-state error when shape-byte arithmetic
/// overflows, the handle range is invalid, misaligned, incorrectly sized, or
/// still has another live raw owner. Backend resource-resolution errors retain
/// their typed source.
pub fn finish_c32(
    session: &WebGpuExecSession<'_>,
    shape: Vec<usize>,
    handle: Handle,
    op: &'static str,
) -> tenferro_tensor::Result<TypedTensor<Complex32>> {
    finish(session, shape, handle, op)
}

fn input_parts<T: Send + Sync + 'static>(
    session: &WebGpuExecSession<'_>,
    tensor: &TypedTensor<T>,
    op: &'static str,
) -> tenferro_tensor::Result<(Handle, Vec<usize>, Vec<usize>)> {
    ensure_resident_on_runtime(session.runtime(), tensor, op)?;
    validate_compact_column_major(tensor, op)?;
    let buffer = webgpu_buffer(tensor, op)?;
    let expected = checked_shape_product(op, tensor.shape())?;
    if buffer.element_len::<T>() != expected {
        return Err(Error::runtime_state(
            op,
            format!(
                "WebGPU allocation has {} elements but shape requires {expected}",
                buffer.element_len::<T>()
            ),
        ));
    }
    Ok((
        buffer.handle().clone(),
        tensor.shape().to_vec(),
        checked_logical_strides(tensor, op)?,
    ))
}

fn validate_compact_column_major<T>(
    tensor: &TypedTensor<T>,
    op: &'static str,
) -> tenferro_tensor::Result<()> {
    if tensor.layout().offset() != 0 {
        return Err(Error::unsupported(
            op,
            "WebGPU FFT requires a zero-offset compact column-major input",
        ));
    }
    let actual = checked_logical_strides(tensor, op)?;
    let expected = column_major_strides(tensor.shape(), op)?;
    if actual != expected {
        return Err(Error::unsupported(
            op,
            "WebGPU FFT requires a zero-offset compact column-major input",
        ));
    }
    Ok(())
}

fn checked_logical_strides<T>(
    tensor: &TypedTensor<T>,
    op: &'static str,
) -> tenferro_tensor::Result<Vec<usize>> {
    tensor
        .layout()
        .strides()
        .iter()
        .map(|&stride| {
            usize::try_from(stride).map_err(|_| {
                Error::unsupported(op, "WebGPU FFT does not support negative input strides")
            })
        })
        .collect()
}

fn column_major_strides(shape: &[usize], op: &'static str) -> tenferro_tensor::Result<Vec<usize>> {
    let mut stride = 1usize;
    let mut strides = Vec::with_capacity(shape.len());
    for &extent in shape {
        strides.push(stride);
        stride = stride
            .checked_mul(extent)
            .ok_or_else(|| Error::invalid_argument(op, "shape", "column-major stride overflow"))?;
    }
    Ok(strides)
}

fn finish<T: Send + Sync + 'static>(
    session: &WebGpuExecSession<'_>,
    shape: Vec<usize>,
    handle: Handle,
    op: &'static str,
) -> tenferro_tensor::Result<TypedTensor<T>> {
    let len = checked_shape_product(op, &shape)?;
    let expected_bytes = len.checked_mul(core::mem::size_of::<T>()).ok_or_else(|| {
        Error::invalid_argument(op, "shape", "WebGPU output byte length overflow")
    })?;
    let expected_bytes = u64::try_from(expected_bytes).map_err(|_| {
        Error::invalid_argument(
            op,
            "shape",
            "WebGPU output byte length exceeds the handle size range",
        )
    })?;
    let offset_start = handle.offset_start.unwrap_or(0);
    let offset_end = handle.offset_end.unwrap_or(0);
    let checked_range_bytes = handle
        .size()
        .checked_sub(offset_start)
        .and_then(|remaining| remaining.checked_sub(offset_end))
        .ok_or_else(|| {
            Error::runtime_state(
                op,
                format!(
                    "WebGPU output handle range is invalid: size {}, start offset \
                     {offset_start}, end offset {offset_end}",
                    handle.size()
                ),
            )
        })?;
    if !offset_start.is_multiple_of(core::mem::align_of::<T>() as u64) {
        return Err(Error::runtime_state(
            op,
            format!(
                "WebGPU output handle start offset {offset_start} is not aligned for {}",
                std::any::type_name::<T>()
            ),
        ));
    }
    let actual_bytes = handle.size_in_used();
    if actual_bytes != checked_range_bytes {
        return Err(Error::runtime_state(
            op,
            "WebGPU output handle reported inconsistent used-range size",
        ));
    }
    if actual_bytes != expected_bytes {
        return Err(Error::runtime_state(
            op,
            format!(
                "WebGPU output handle has {actual_bytes} usable bytes but shape requires \
                 {expected_bytes}"
            ),
        ));
    }
    if !handle.can_mut() {
        return Err(Error::runtime_state(
            op,
            "WebGPU output completion requires unique raw-handle ownership",
        ));
    }
    let buffer = WebGpuBuffer::new_for_runtime(
        session.runtime(),
        handle,
        usize::try_from(expected_bytes).map_err(|_| {
            Error::invalid_argument(op, "shape", "WebGPU output byte length exceeds usize")
        })?,
        op,
    )?;
    typed_from_webgpu(shape, buffer, session.runtime())
}
