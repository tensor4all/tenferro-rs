use cubecl::client::ComputeClient;
use cubecl::frontend::CubePrimitive;
use cubecl::std::tensor::TensorHandle;
use cubecl_runtime::server::Handle;
use cubecl_wgpu::WgpuRuntime;
use cubek_fft::{
    cfft_interleaved_launch, irfft_interleaved_launch_padded, rfft_interleaved_launch_padded,
    ComplexTensorHandle, FftMode, FftNormalization,
};
use num_complex::Complex32;
use tenferro_tensor::{Error, TensorScalar, TypedTensor};

use super::{
    checked_shape_product, ensure_resident_on_runtime, prepared_webgpu_tensor, typed_from_webgpu,
    WebGpuBuffer, WebGpuExecSession,
};

/// Direction of a WebGPU complex FFT launch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WebGpuFftMode {
    /// Compute the forward transform.
    Forward,
    /// Compute the inverse transform.
    Inverse,
}

/// Normalization applied by a WebGPU FFT launch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WebGpuFftNormalization {
    /// Do not scale the transform.
    None,
    /// Scale by the transform length.
    ByN,
    /// Apply orthonormal scaling.
    Ortho,
}

/// Hardware limits used by the WebGPU FFT extension.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WebGpuFftLimits {
    /// Maximum shared-memory allocation reported by the adapter.
    pub max_shared_memory_size: usize,
    /// Maximum number of compute units reported by the adapter.
    pub max_units_per_cube: u32,
}

/// Output layout supplied to a session-scoped WebGPU FFT launch.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WebGpuFftOutput {
    /// Logical output shape.
    pub shape: Vec<usize>,
    /// Logical column-major output strides.
    pub strides: Vec<usize>,
}

impl WebGpuFftOutput {
    /// Create an output layout for a provider FFT operation.
    pub fn new(shape: Vec<usize>, strides: Vec<usize>) -> Self {
        Self { shape, strides }
    }
}

/// Read the adapter limits needed to validate a WebGPU FFT plan.
pub fn fft_limits(session: &WebGpuExecSession<'_>) -> WebGpuFftLimits {
    let client = client(session);
    WebGpuFftLimits {
        max_shared_memory_size: client.properties().hardware.max_shared_memory_size,
        max_units_per_cube: client.properties().hardware.max_units_per_cube,
    }
}

/// Execute a compact C32 forward or inverse FFT and return its owned output.
///
/// The CubeCL client, bindings, allocation, and completion checks remain
/// inside this session-scoped provider boundary. No backend handle escapes.
///
/// # Errors
///
/// Returns [`Error::Unsupported`] for an unsupported placement or FFT
/// configuration, [`Error::Validation`] for invalid shape/layout metadata,
/// [`Error::BackendSource`] for a provider launch failure, or
/// [`Error::RuntimeState`] when output completion or allocation state is
/// invalid.
pub fn execute_c32_fft(
    session: &WebGpuExecSession<'_>,
    input: &TypedTensor<Complex32>,
    output: WebGpuFftOutput,
    axis: usize,
    mode: WebGpuFftMode,
    normalization: WebGpuFftNormalization,
    op: &'static str,
) -> tenferro_tensor::Result<TypedTensor<Complex32>> {
    let WebGpuFftOutput {
        shape: output_shape,
        strides: output_strides,
    } = output;
    let (handle, shape, strides) = input_parts(session, input, op)?;
    let input =
        ComplexTensorHandle::<WgpuRuntime>::new_strided(shape, strides, handle, f32_storage())
            .map_err(|error| Error::backend_source(op, error))?;
    let output = ComplexTensorHandle::<WgpuRuntime>::new_strided(
        output_shape.clone(),
        output_strides,
        allocate_raw(session, output_bytes::<Complex32>(op, &output_shape)?),
        f32_storage(),
    )
    .map_err(|error| Error::backend_source(op, error))?;
    cfft_interleaved_launch(
        client(session),
        input.binding(),
        output.binding(),
        axis,
        match mode {
            WebGpuFftMode::Forward => FftMode::Forward,
            WebGpuFftMode::Inverse => FftMode::Inverse,
        },
        normalization.into(),
    )
    .map_err(|error| Error::backend_source(op, error))?;
    finish_c32(session, output_shape, output.into_raw_parts().handle, op)
}

/// Execute a padded one-sided F32-to-C32 real FFT and return its owned output.
///
/// The input must be a compact column-major tensor already resident on the
/// session's adapter. Padding is performed by the provider kernel.
///
/// # Errors
///
/// Returns [`Error::Unsupported`] for an unsupported placement or FFT
/// configuration, [`Error::Validation`] for invalid shape/layout metadata,
/// [`Error::BackendSource`] for a provider launch failure, or
/// [`Error::RuntimeState`] when output completion or allocation state is
/// invalid.
pub fn execute_f32_rfft(
    session: &WebGpuExecSession<'_>,
    input: &TypedTensor<f32>,
    output: WebGpuFftOutput,
    axis: usize,
    signal_len: usize,
    normalization: WebGpuFftNormalization,
    op: &'static str,
) -> tenferro_tensor::Result<TypedTensor<Complex32>> {
    let WebGpuFftOutput {
        shape: output_shape,
        strides: output_strides,
    } = output;
    let input = f32_input(session, input, op)?;
    let output = ComplexTensorHandle::<WgpuRuntime>::new_strided(
        output_shape.clone(),
        output_strides,
        allocate_raw(session, output_bytes::<Complex32>(op, &output_shape)?),
        f32_storage(),
    )
    .map_err(|error| Error::backend_source(op, error))?;
    rfft_interleaved_launch_padded(
        client(session),
        &input,
        output.binding(),
        axis,
        signal_len,
        normalization.into(),
    )
    .map_err(|error| Error::backend_source(op, error))?;
    finish_c32(session, output_shape, output.into_raw_parts().handle, op)
}

/// Execute a padded C32-to-F32 inverse real FFT and return its owned output.
///
/// The input must be a compact column-major tensor already resident on the
/// session's adapter. `spectrum_len` is the number of input bins on `axis`.
///
/// # Errors
///
/// Returns [`Error::Unsupported`] for an unsupported placement or FFT
/// configuration, [`Error::Validation`] for invalid shape/layout metadata,
/// [`Error::BackendSource`] for a provider launch failure, or
/// [`Error::RuntimeState`] when output completion or allocation state is
/// invalid.
pub fn execute_c32_irfft(
    session: &WebGpuExecSession<'_>,
    input: &TypedTensor<Complex32>,
    output: WebGpuFftOutput,
    axis: usize,
    spectrum_len: usize,
    normalization: WebGpuFftNormalization,
    op: &'static str,
) -> tenferro_tensor::Result<TypedTensor<f32>> {
    let WebGpuFftOutput {
        shape: output_shape,
        strides: output_strides,
    } = output;
    let (handle, shape, strides) = input_parts(session, input, op)?;
    let input =
        ComplexTensorHandle::<WgpuRuntime>::new_strided(shape, strides, handle, f32_storage())
            .map_err(|error| Error::backend_source(op, error))?;
    let output = TensorHandle::<WgpuRuntime>::new(
        allocate_raw(session, output_bytes::<f32>(op, &output_shape)?),
        output_shape.clone(),
        output_strides,
        f32_storage(),
    );
    irfft_interleaved_launch_padded(
        client(session),
        input.binding(),
        &output,
        axis,
        spectrum_len,
        normalization.into(),
    )
    .map_err(|error| Error::backend_source(op, error))?;
    finish_f32(session, output_shape, output.handle, op)
}

fn client<'a>(session: &'a WebGpuExecSession<'a>) -> &'a ComputeClient<WgpuRuntime> {
    session.runtime().client()
}

fn allocate_raw(session: &WebGpuExecSession<'_>, bytes: usize) -> Handle {
    client(session).empty(bytes)
}

fn f32_input(
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

fn input_parts<T: TensorScalar + Send + Sync + 'static>(
    session: &WebGpuExecSession<'_>,
    tensor: &TypedTensor<T>,
    op: &'static str,
) -> tenferro_tensor::Result<(Handle, Vec<usize>, Vec<usize>)> {
    ensure_resident_on_runtime(session.runtime(), tensor, op)?;
    validate_compact_column_major(tensor, op)?;
    let prepared = prepared_webgpu_tensor(tensor, op)?;
    let expected = checked_shape_product(op, tensor.shape())?;
    let actual = prepared.byte_len / core::mem::size_of::<T>();
    if actual != expected {
        return Err(Error::runtime_state(
            op,
            format!("WebGPU allocation has {actual} elements but shape requires {expected}"),
        ));
    }
    Ok((
        prepared.handle,
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

fn output_bytes<T>(op: &'static str, shape: &[usize]) -> tenferro_tensor::Result<usize> {
    checked_shape_product(op, shape)?
        .checked_mul(core::mem::size_of::<T>())
        .ok_or_else(|| {
            Error::invalid_argument(op, "shape", "WebGPU FFT output byte length overflow")
        })
}

fn finish_f32(
    session: &WebGpuExecSession<'_>,
    shape: Vec<usize>,
    handle: Handle,
    op: &'static str,
) -> tenferro_tensor::Result<TypedTensor<f32>> {
    finish(session, shape, handle, op)
}

fn finish_c32(
    session: &WebGpuExecSession<'_>,
    shape: Vec<usize>,
    handle: Handle,
    op: &'static str,
) -> tenferro_tensor::Result<TypedTensor<Complex32>> {
    finish(session, shape, handle, op)
}

fn finish<T: TensorScalar + Send + Sync + 'static>(
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

impl From<WebGpuFftMode> for FftMode {
    fn from(mode: WebGpuFftMode) -> Self {
        match mode {
            WebGpuFftMode::Forward => Self::Forward,
            WebGpuFftMode::Inverse => Self::Inverse,
        }
    }
}

impl From<WebGpuFftNormalization> for FftNormalization {
    fn from(normalization: WebGpuFftNormalization) -> Self {
        match normalization {
            WebGpuFftNormalization::None => Self::None,
            WebGpuFftNormalization::ByN => Self::ByN,
            WebGpuFftNormalization::Ortho => Self::Ortho,
        }
    }
}

fn f32_storage() -> cubecl::prelude::StorageType {
    f32::as_type_native_unchecked().storage_type()
}
