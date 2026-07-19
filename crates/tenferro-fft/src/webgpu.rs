use cubecl::frontend::CubePrimitive;
use cubecl::std::tensor::TensorHandle;
use cubecl_wgpu::WgpuRuntime;
use cubek_fft::{
    cfft_interleaved_launch, irfft_interleaved_launch_padded, rfft_interleaved_launch_padded,
    ComplexTensorHandle, FftMode, FftNormalization,
};
use tenferro_gpu::{webgpu_interop, WebGpuBackend};
use tenferro_tensor::{Error, Tensor};

use crate::{FftBackend, FftExecutionCache, FftNorm, FftOperation, FftPlanSpec};

impl FftBackend for WebGpuBackend {
    fn execute_fft(
        &mut self,
        input: &Tensor,
        spec: &FftPlanSpec,
        _cache: FftExecutionCache<'_>,
    ) -> tenferro_tensor::Result<Tensor> {
        // CubeCL's compute client owns compiled-kernel caching. There is no host-side
        // CubeK plan to duplicate in tenferro's extension cache.
        validate_spec_input(input, spec)?;
        let plan = MetalFftPlan::new(self, spec)?;
        match (spec.operation(), input) {
            (FftOperation::C2cForward | FftOperation::C2cInverse, Tensor::C32(input)) => {
                execute_cfft(self, input, spec, &plan)
            }
            (FftOperation::R2cOnesided, Tensor::F32(input)) => {
                execute_rfft(self, input, spec, &plan)
            }
            (FftOperation::C2r, Tensor::C32(input)) => execute_irfft(self, input, spec, &plan),
            _ => Err(unsupported(
                spec,
                "Metal FFT supports C32 CFFT, F32 one-sided RFFT, and C32-to-F32 IRFFT",
            )),
        }
    }
}

#[derive(Debug)]
struct MetalFftPlan {
    axis: usize,
    n_fft: usize,
    output_shape: Vec<usize>,
    logical_strides: Vec<usize>,
    normalization: FftNormalization,
}

impl MetalFftPlan {
    fn new(backend: &WebGpuBackend, spec: &FftPlanSpec) -> tenferro_tensor::Result<Self> {
        let op = op_name(spec.operation());
        let axis = spec.normalized_axis();
        let input_len = spec.input_shape()[axis];
        let n_fft = spec.requested_len().unwrap_or(input_len);
        if n_fft < 2 || !n_fft.is_power_of_two() {
            return Err(unsupported(
                spec,
                "Metal FFT requires a power-of-two transform length of at least 2",
            ));
        }
        if matches!(spec.operation(), FftOperation::R2cFull) {
            return Err(unsupported(
                spec,
                "Metal FFT does not support full-spectrum real FFT",
            ));
        }
        if spec.operation().is_c2c() && n_fft != input_len {
            return Err(unsupported(
                spec,
                "Metal CFFT does not support padding or truncation",
            ));
        }

        let max_shared_bytes = webgpu_interop::max_shared_memory_size(backend);
        let max_units = webgpu_interop::max_units_per_cube(backend);
        let max_shared_elements = max_shared_bytes / (2 * core::mem::size_of::<f32>());
        if max_shared_elements == 0 || max_units == 0 {
            return Err(Error::runtime_state(
                op,
                "WebGPU device reports no usable shared memory or compute units for FFT",
            ));
        }
        let max_shared = floor_power_of_two(max_shared_elements)
            .ok_or_else(|| Error::runtime_state(op, "WebGPU shared-memory FFT limit overflow"))?;
        let max_n_fft = match spec.operation() {
            FftOperation::C2cForward | FftOperation::C2cInverse => {
                max_shared.saturating_mul(max_shared)
            }
            FftOperation::R2cOnesided | FftOperation::C2r => {
                max_shared.saturating_mul(max_shared).saturating_mul(2)
            }
            FftOperation::R2cFull => 0,
        };
        if n_fft > max_n_fft {
            return Err(unsupported(
                spec,
                format!("transform length {n_fft} exceeds this WebGPU device limit {max_n_fft}"),
            ));
        }
        u32::try_from(n_fft).map_err(|_| {
            Error::unsupported(op, "Metal FFT length exceeds the CubeK launch range")
        })?;

        let lane_count = spec
            .input_shape()
            .iter()
            .enumerate()
            .filter(|(candidate, _)| *candidate != axis)
            .try_fold(1usize, |count, (_, &extent)| count.checked_mul(extent))
            .ok_or_else(|| Error::invalid_argument(op, "shape", "FFT lane count overflow"))?;
        u32::try_from(lane_count).map_err(|_| {
            Error::unsupported(op, "Metal FFT lane count exceeds the CubeK launch range")
        })?;

        let mut output_shape = spec.input_shape().to_vec();
        output_shape[axis] = match spec.operation() {
            FftOperation::C2cForward | FftOperation::C2cInverse => n_fft,
            FftOperation::R2cOnesided => n_fft
                .checked_div(2)
                .and_then(|half| half.checked_add(1))
                .ok_or_else(|| {
                    Error::invalid_argument(op, "length", "RFFT output length overflow")
                })?,
            FftOperation::C2r => n_fft,
            FftOperation::R2cFull => {
                return Err(unsupported(
                    spec,
                    "Metal FFT does not support full-spectrum real FFT",
                ));
            }
        };
        checked_product(&output_shape, op)?;
        if matches!(spec.operation(), FftOperation::R2cOnesided) {
            u32::try_from(input_len.min(n_fft)).map_err(|_| {
                Error::unsupported(
                    op,
                    "Metal RFFT signal length exceeds the CubeK launch range",
                )
            })?;
        }
        if matches!(spec.operation(), FftOperation::C2r) {
            u32::try_from(spec.input_shape()[axis]).map_err(|_| {
                Error::unsupported(
                    op,
                    "Metal IRFFT spectrum length exceeds the CubeK launch range",
                )
            })?;
        }
        let logical_strides = column_major_strides(&output_shape, op)?;
        let inverse = matches!(
            spec.operation(),
            FftOperation::C2cInverse | FftOperation::C2r
        );
        Ok(Self {
            axis,
            n_fft,
            output_shape,
            logical_strides,
            normalization: normalization(spec.norm(), inverse),
        })
    }
}

fn execute_cfft(
    backend: &WebGpuBackend,
    input: &tenferro_tensor::TypedTensor<num_complex::Complex32>,
    spec: &FftPlanSpec,
    plan: &MetalFftPlan,
) -> tenferro_tensor::Result<Tensor> {
    let op = op_name(spec.operation());
    let (handle, shape, strides) = webgpu_interop::c32_input_parts(backend, input, op)?;
    let input =
        ComplexTensorHandle::<WgpuRuntime>::new_strided(shape, strides, handle, f32_storage())
            .map_err(|error| Error::backend_source(op, error))?;
    let output_len = checked_product(&plan.output_shape, op)?;
    let bytes = output_len
        .checked_mul(core::mem::size_of::<num_complex::Complex32>())
        .ok_or_else(|| Error::invalid_argument(op, "shape", "FFT output byte length overflow"))?;
    let output = ComplexTensorHandle::<WgpuRuntime>::new_strided(
        plan.output_shape.clone(),
        plan.logical_strides.clone(),
        webgpu_interop::allocate_raw(backend, bytes),
        f32_storage(),
    )
    .map_err(|error| Error::backend_source(op, error))?;
    let mode = if spec.operation() == FftOperation::C2cForward {
        FftMode::Forward
    } else {
        FftMode::Inverse
    };
    cfft_interleaved_launch(
        webgpu_interop::client(backend),
        input.binding(),
        output.binding(),
        plan.axis,
        mode,
        plan.normalization,
    )
    .map_err(|error| Error::backend_source(op, error))?;
    let handle = output.into_raw_parts().handle;
    webgpu_interop::finish_c32(backend, plan.output_shape.clone(), handle, op).map(Tensor::C32)
}

fn execute_rfft(
    backend: &WebGpuBackend,
    input: &tenferro_tensor::TypedTensor<f32>,
    spec: &FftPlanSpec,
    plan: &MetalFftPlan,
) -> tenferro_tensor::Result<Tensor> {
    let op = op_name(spec.operation());
    let input = webgpu_interop::f32_input(backend, input, op)?;
    let output_len = checked_product(&plan.output_shape, op)?;
    let bytes = output_len
        .checked_mul(core::mem::size_of::<num_complex::Complex32>())
        .ok_or_else(|| Error::invalid_argument(op, "shape", "FFT output byte length overflow"))?;
    let output = ComplexTensorHandle::<WgpuRuntime>::new_strided(
        plan.output_shape.clone(),
        plan.logical_strides.clone(),
        webgpu_interop::allocate_raw(backend, bytes),
        f32_storage(),
    )
    .map_err(|error| Error::backend_source(op, error))?;
    let signal_len = spec.input_shape()[plan.axis].min(plan.n_fft);
    rfft_interleaved_launch_padded(
        webgpu_interop::client(backend),
        &input,
        output.binding(),
        plan.axis,
        signal_len,
        plan.normalization,
    )
    .map_err(|error| Error::backend_source(op, error))?;
    let handle = output.into_raw_parts().handle;
    webgpu_interop::finish_c32(backend, plan.output_shape.clone(), handle, op).map(Tensor::C32)
}

fn execute_irfft(
    backend: &WebGpuBackend,
    input: &tenferro_tensor::TypedTensor<num_complex::Complex32>,
    spec: &FftPlanSpec,
    plan: &MetalFftPlan,
) -> tenferro_tensor::Result<Tensor> {
    let op = op_name(spec.operation());
    let (handle, shape, strides) = webgpu_interop::c32_input_parts(backend, input, op)?;
    let input =
        ComplexTensorHandle::<WgpuRuntime>::new_strided(shape, strides, handle, f32_storage())
            .map_err(|error| Error::backend_source(op, error))?;
    let output_len = checked_product(&plan.output_shape, op)?;
    let bytes = output_len
        .checked_mul(core::mem::size_of::<f32>())
        .ok_or_else(|| Error::invalid_argument(op, "shape", "FFT output byte length overflow"))?;
    let output = TensorHandle::<WgpuRuntime>::new(
        webgpu_interop::allocate_raw(backend, bytes),
        plan.output_shape.clone(),
        plan.logical_strides.clone(),
        f32_storage(),
    );
    let spec_bins = spec.input_shape()[plan.axis];
    irfft_interleaved_launch_padded(
        webgpu_interop::client(backend),
        input.binding(),
        &output,
        plan.axis,
        spec_bins,
        plan.normalization,
    )
    .map_err(|error| Error::backend_source(op, error))?;
    webgpu_interop::finish_f32(backend, plan.output_shape.clone(), output.handle, op)
        .map(Tensor::F32)
}

fn validate_spec_input(input: &Tensor, spec: &FftPlanSpec) -> tenferro_tensor::Result<()> {
    let op = op_name(spec.operation());
    if input.dtype() != spec.input_dtype() {
        return Err(Error::dtype_mismatch(op, spec.input_dtype(), input.dtype()));
    }
    if input.shape() != spec.input_shape() {
        return Err(Error::shape_mismatch(op, spec.input_shape(), input.shape()));
    }
    Ok(())
}

fn checked_product(shape: &[usize], op: &'static str) -> tenferro_tensor::Result<usize> {
    shape
        .iter()
        .try_fold(1usize, |total, &extent| total.checked_mul(extent))
        .ok_or_else(|| Error::invalid_argument(op, "shape", "FFT tensor element count overflow"))
}

fn column_major_strides(shape: &[usize], op: &'static str) -> tenferro_tensor::Result<Vec<usize>> {
    let mut stride = 1usize;
    let mut strides = Vec::with_capacity(shape.len());
    for &extent in shape {
        strides.push(stride);
        stride = stride
            .checked_mul(extent)
            .ok_or_else(|| Error::invalid_argument(op, "shape", "FFT output stride overflow"))?;
    }
    Ok(strides)
}

fn normalization(norm: FftNorm, inverse: bool) -> FftNormalization {
    match (norm, inverse) {
        (FftNorm::Backward, false) | (FftNorm::Forward, true) => FftNormalization::None,
        (FftNorm::Backward, true) | (FftNorm::Forward, false) => FftNormalization::ByN,
        (FftNorm::Ortho, _) => FftNormalization::Ortho,
    }
}

fn floor_power_of_two(value: usize) -> Option<usize> {
    if value.is_power_of_two() {
        Some(value)
    } else {
        value.checked_next_power_of_two().map(|next| next >> 1)
    }
}

fn f32_storage() -> cubecl::prelude::StorageType {
    f32::as_type_native_unchecked().storage_type()
}

fn unsupported(spec: &FftPlanSpec, message: impl Into<String>) -> Error {
    Error::unsupported(op_name(spec.operation()), message)
}

fn op_name(operation: FftOperation) -> &'static str {
    match operation {
        FftOperation::C2cForward => "fft",
        FftOperation::C2cInverse => "ifft",
        FftOperation::R2cFull => "fft",
        FftOperation::R2cOnesided => "rfft",
        FftOperation::C2r => "irfft",
    }
}
