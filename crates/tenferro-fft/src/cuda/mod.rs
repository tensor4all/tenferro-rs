mod hermitian;

pub(crate) mod descriptor;
pub(crate) mod error;
pub(crate) mod ffi;
pub(crate) mod plan;

#[cfg(test)]
mod tests;

use tenferro_gpu::cuda::interop::{alloc_output, scale_tensor_write, with_typed_device_ptr};
use tenferro_gpu::cuda::{CudaExecSession, CudaRuntime};
use tenferro_tensor::{
    DType, SliceConfig, Tensor, TensorElementwise, TensorIndexing, TensorReduction,
    TensorStructural, TensorWrite,
};

use crate::backend::FftExecutionCache;
use crate::{
    expected_dtype_description, fft_op_name, output_shape_c2c, output_shape_c2r, output_shape_r2c,
    FftBackend, FftNorm, FftOperation, FftPlanSpec,
};
use descriptor::{CufftDirection, CufftPlanDescriptor, CufftPlanKey, CufftTransformKind};
use error::{into_tensor_error, CudaFftError};
use plan::{extension_plan_key_for_runtime, with_cufft_plan_for_batch, CufftPlanEntry};

const OP: &str = "cuda_fft";

#[derive(Debug, thiserror::Error)]
#[error("CUDA FFT requires device/runtime residency: {source}")]
struct CudaFftPlacementError {
    #[source]
    source: tenferro_tensor::Error,
}

/// Metadata for the compact final-axis representation consumed by cuFFT.
struct CanonicalCudaFft {
    permutation: Vec<usize>,
    inverse_permutation: Vec<usize>,
    canonical_input_shape: Vec<usize>,
    output_shape: Vec<usize>,
    n: usize,
    batch: usize,
    transform: CufftTransformKind,
    direction: CufftDirection,
}

impl FftBackend for CudaExecSession<'_> {
    fn execute_fft(
        &mut self,
        input: &Tensor,
        spec: &FftPlanSpec,
        mut cache: FftExecutionCache<'_>,
    ) -> tenferro_tensor::Result<Tensor> {
        validate_cuda_input(self.runtime(), input, spec)?;

        let (transform, direction, output_dtype) =
            transform_mapping(spec.operation(), input.dtype())?;
        let output_shape = output_shape_for(input, spec)?;
        checked_shape_product(OP, "output shape", &output_shape)?;
        let canonical = canonical_metadata(input, spec, output_shape, transform, direction)?;

        // The gate is intentionally the single owner of the complete
        // non-empty library/cache/plan closure. For batch == 0 it returns
        // `None` without invoking this closure, so the empty output is
        // allocated below without loading cuFFT or constructing a cache key.
        let executed = with_cufft_plan_for_batch(canonical.batch, || {
            let canonical_input_owner = canonicalize_input(self, input, spec, &canonical)
                .map_err(|source| CudaFftError::interop("cuda_fft_canonicalize", source))?;
            let canonical_input = canonical_input_owner.as_ref().unwrap_or(input);
            let canonical_output_shape = canonical_output_shape(&canonical, spec.operation())
                .map_err(|source| CudaFftError::interop("cuda_fft_output_shape", source))?;
            let mut canonical_output =
                allocate_cuda_output(self.runtime(), output_dtype, &canonical_output_shape)
                    .map_err(|source| CudaFftError::interop("cuda_fft_output_allocate", source))?;

            let descriptor = CufftPlanDescriptor::new(
                canonical.transform,
                canonical.direction,
                canonical.n,
                canonical.batch,
            )?;
            let device_ordinal =
                usize::try_from(self.runtime().device_id().ordinal()).map_err(|_| {
                    CudaFftError::InvalidConfiguration {
                        field: "device_ordinal",
                    }
                })?;
            let key = CufftPlanKey {
                runtime_identity: self.runtime_identity(),
                device_ordinal,
                kind: canonical.transform,
                direction: canonical.direction,
                n: canonical.n,
                batch: canonical.batch,
                istride: descriptor.istride,
                idist: descriptor.idist,
                ostride: descriptor.ostride,
                odist: descriptor.odist,
            };
            let cache_key = extension_plan_key_for_runtime(&key, self.runtime());
            let mut cached = false;
            {
                let store = cache.store_mut();
                if let Some(entry) = store.get_mut::<CufftPlanEntry>(&cache_key) {
                    if entry.matches_key(&key) {
                        entry.execute(canonical_input, &mut canonical_output)?;
                        cached = true;
                    }
                }
                if !cached {
                    let mut entry = CufftPlanEntry::create(self.runtime(), key, descriptor)?;
                    entry.execute(canonical_input, &mut canonical_output)?;
                    let retained_bytes = entry.retained_bytes();
                    store.put(cache_key, entry, retained_bytes);
                }
            }

            let inverse = matches!(
                spec.operation(),
                FftOperation::C2cInverse | FftOperation::C2r
            );
            let factor = fft_scale(spec.norm(), inverse, canonical.n);
            if factor != 1.0 {
                scale_tensor_write(
                    self.runtime(),
                    TensorWrite::from_tensor(&mut canonical_output),
                    factor,
                )
                .map_err(|source| CudaFftError::interop("cuda_fft_scale", source))?;
            }

            if spec.operation() == FftOperation::R2cFull {
                canonical_output = hermitian::complete(self, canonical_output, canonical.n)
                    .map_err(|source| CudaFftError::interop("cuda_fft_hermitian", source))?;
            }
            if !is_identity_permutation(&canonical.inverse_permutation) {
                canonical_output = self
                    .transpose(&canonical_output, &canonical.inverse_permutation)
                    .map_err(|source| {
                        CudaFftError::interop("cuda_fft_inverse_transpose", source)
                    })?;
            }

            if canonical_output.shape() != canonical.output_shape
                || canonical_output.dtype() != output_dtype
            {
                return Err(CudaFftError::internal(
                    "CUDA FFT returned unexpected final shape or dtype",
                ));
            }
            Ok(canonical_output)
        })
        .map_err(|source| into_tensor_error(fft_op_name(spec.operation()), source))?;

        match executed {
            Some(output) => Ok(output),
            None => allocate_cuda_output(self.runtime(), output_dtype, &canonical.output_shape),
        }
    }
}

fn validate_cuda_input(
    runtime: &CudaRuntime,
    input: &Tensor,
    spec: &FftPlanSpec,
) -> tenferro_tensor::Result<()> {
    let op = fft_op_name(spec.operation());
    if input.dtype() != spec.input_dtype() {
        return Err(tenferro_tensor::Error::dtype_mismatch(
            op,
            spec.input_dtype(),
            input.dtype(),
        ));
    }
    if input.shape() != spec.input_shape() {
        return Err(tenferro_tensor::Error::invalid_argument(
            op,
            "input shape",
            format!(
                "validated FFT spec shape {:?} does not match execution input shape {:?}",
                spec.input_shape(),
                input.shape()
            ),
        ));
    }
    if !spec.requires_compact_column_major() {
        return Err(tenferro_tensor::Error::unsupported(
            op,
            "CUDA FFT requires compact column-major input",
        ));
    }

    // Placement is checked before layout, dtype dispatch, zero-batch returns,
    // allocation, or any vendor-library/cache work. This rejects host and
    // foreign-runtime buffers without an implicit transfer.
    ensure_cuda_tensor_resident(runtime, input)?;
    if !input.is_col_major_contiguous()? {
        return Err(tenferro_tensor::Error::unsupported(
            op,
            "CUDA FFT requires compact column-major input",
        ));
    }
    Ok(())
}

fn ensure_cuda_tensor_resident(
    runtime: &CudaRuntime,
    input: &Tensor,
) -> tenferro_tensor::Result<()> {
    let result = match input {
        Tensor::F32(input) => with_typed_device_ptr(runtime, input, OP, |_| {}),
        Tensor::F64(input) => with_typed_device_ptr(runtime, input, OP, |_| {}),
        Tensor::I32(input) => with_typed_device_ptr(runtime, input, OP, |_| {}),
        Tensor::I64(input) => with_typed_device_ptr(runtime, input, OP, |_| {}),
        Tensor::Bool(input) => with_typed_device_ptr(runtime, input, OP, |_| {}),
        Tensor::C32(input) => with_typed_device_ptr(runtime, input, OP, |_| {}),
        Tensor::C64(input) => with_typed_device_ptr(runtime, input, OP, |_| {}),
    };
    result.map_err(|source| {
        tenferro_tensor::Error::runtime_state_source(OP, CudaFftPlacementError { source })
    })
}

fn transform_mapping(
    operation: FftOperation,
    dtype: DType,
) -> tenferro_tensor::Result<(CufftTransformKind, CufftDirection, DType)> {
    let mapping = match (operation, dtype) {
        (FftOperation::C2cForward, DType::C32) => (
            CufftTransformKind::C2c32,
            CufftDirection::Forward,
            DType::C32,
        ),
        (FftOperation::C2cInverse, DType::C32) => (
            CufftTransformKind::C2c32,
            CufftDirection::Inverse,
            DType::C32,
        ),
        (FftOperation::C2cForward, DType::C64) => (
            CufftTransformKind::C2c64,
            CufftDirection::Forward,
            DType::C64,
        ),
        (FftOperation::C2cInverse, DType::C64) => (
            CufftTransformKind::C2c64,
            CufftDirection::Inverse,
            DType::C64,
        ),
        (FftOperation::R2cFull | FftOperation::R2cOnesided, DType::F32) => (
            CufftTransformKind::R2c32,
            CufftDirection::Forward,
            DType::C32,
        ),
        (FftOperation::R2cFull | FftOperation::R2cOnesided, DType::F64) => (
            CufftTransformKind::R2c64,
            CufftDirection::Forward,
            DType::C64,
        ),
        (FftOperation::C2r, DType::C32) => (
            CufftTransformKind::C2r32,
            CufftDirection::Inverse,
            DType::F32,
        ),
        (FftOperation::C2r, DType::C64) => (
            CufftTransformKind::C2r64,
            CufftDirection::Inverse,
            DType::F64,
        ),
        _ => {
            return Err(crate::tensor_unsupported_dtype(
                fft_op_name(operation),
                dtype,
                expected_dtype_description(operation),
            ));
        }
    };
    Ok(mapping)
}

fn output_shape_for(input: &Tensor, spec: &FftPlanSpec) -> tenferro_tensor::Result<Vec<usize>> {
    let axis = spec.normalized_axis();
    if axis >= input.shape().len() {
        return Err(tenferro_tensor::Error::axis_out_of_bounds(
            fft_op_name(spec.operation()),
            axis,
            input.shape().len(),
        ));
    }
    match spec.operation() {
        FftOperation::C2cForward | FftOperation::C2cInverse => {
            output_shape_c2c(input.shape(), axis, spec.requested_len())
        }
        FftOperation::R2cFull => output_shape_r2c(input.shape(), axis, spec.requested_len(), false),
        FftOperation::R2cOnesided => {
            output_shape_r2c(input.shape(), axis, spec.requested_len(), true)
        }
        FftOperation::C2r => output_shape_c2r(input.shape(), axis, spec.requested_len()),
    }
}

fn canonical_metadata(
    input: &Tensor,
    spec: &FftPlanSpec,
    output_shape: Vec<usize>,
    transform: CufftTransformKind,
    direction: CufftDirection,
) -> tenferro_tensor::Result<CanonicalCudaFft> {
    let rank = input.shape().len();
    let axis = spec.normalized_axis();
    if rank == 0 {
        return Err(tenferro_tensor::Error::invalid_argument(
            fft_op_name(spec.operation()),
            "rank",
            "FFT requires rank >= 1",
        ));
    }
    if axis >= rank {
        return Err(tenferro_tensor::Error::axis_out_of_bounds(
            fft_op_name(spec.operation()),
            axis,
            rank,
        ));
    }
    let last = rank.checked_sub(1).ok_or_else(|| {
        tenferro_tensor::Error::invalid_argument(OP, "rank", "FFT requires rank >= 1")
    })?;

    let mut permutation = Vec::with_capacity(rank);
    for original_axis in 0..rank {
        if original_axis != axis {
            permutation.push(original_axis);
        }
    }
    permutation.push(axis);

    let mut inverse_permutation = vec![usize::MAX; rank];
    for (canonical_axis, &original_axis) in permutation.iter().enumerate() {
        let slot = inverse_permutation.get_mut(original_axis).ok_or_else(|| {
            tenferro_tensor::Error::Internal("FFT permutation axis is out of bounds".into())
        })?;
        if *slot != usize::MAX {
            return Err(tenferro_tensor::Error::Internal(
                "FFT permutation contains a duplicate axis".into(),
            ));
        }
        *slot = canonical_axis;
    }
    if inverse_permutation.contains(&usize::MAX) {
        return Err(tenferro_tensor::Error::Internal(
            "FFT inverse permutation is incomplete".into(),
        ));
    }

    let mut canonical_input_shape = Vec::with_capacity(rank);
    for &original_axis in &permutation {
        let extent = input.shape().get(original_axis).copied().ok_or_else(|| {
            tenferro_tensor::Error::Internal("FFT input shape metadata is inconsistent".into())
        })?;
        canonical_input_shape.push(extent);
    }
    let n = match spec.operation() {
        FftOperation::C2r => output_shape.get(axis).copied().ok_or_else(|| {
            tenferro_tensor::Error::Internal("FFT output shape metadata is inconsistent".into())
        })?,
        FftOperation::C2cForward
        | FftOperation::C2cInverse
        | FftOperation::R2cFull
        | FftOperation::R2cOnesided => {
            crate::transform_len(input.shape(), axis, spec.requested_len())?
        }
    };
    if n == 0 {
        return Err(tenferro_tensor::Error::invalid_argument(
            fft_op_name(spec.operation()),
            "transform length",
            "must be positive",
        ));
    }
    if spec.operation() != FftOperation::C2r {
        canonical_input_shape[last] = n;
    }
    let batch = checked_shape_product(OP, "batch", &canonical_input_shape[..last])?;
    checked_shape_product(OP, "canonical input shape", &canonical_input_shape)?;

    Ok(CanonicalCudaFft {
        permutation,
        inverse_permutation,
        canonical_input_shape,
        output_shape,
        n,
        batch,
        transform,
        direction,
    })
}

fn canonicalize_input(
    session: &mut CudaExecSession<'_>,
    input: &Tensor,
    spec: &FftPlanSpec,
    canonical: &CanonicalCudaFft,
) -> tenferro_tensor::Result<Option<Tensor>> {
    let mut owner = if is_identity_permutation(&canonical.permutation) {
        None
    } else {
        Some(session.transpose(input, &canonical.permutation)?)
    };

    if spec.operation() != FftOperation::C2r {
        let last = canonical
            .canonical_input_shape
            .len()
            .checked_sub(1)
            .ok_or_else(|| {
                tenferro_tensor::Error::invalid_argument(OP, "rank", "FFT requires rank >= 1")
            })?;
        let current_shape = owner
            .as_ref()
            .map_or_else(|| input.shape().to_vec(), |tensor| tensor.shape().to_vec());
        let current_len = current_shape.get(last).copied().ok_or_else(|| {
            tenferro_tensor::Error::Internal("FFT canonical input shape is inconsistent".into())
        })?;
        if current_len != canonical.n {
            let transformed = if current_len > canonical.n {
                let mut limits = current_shape.clone();
                limits[last] = canonical.n;
                let current = owner.as_ref().map_or(input, |tensor| tensor);
                session.slice(
                    current,
                    &SliceConfig {
                        starts: vec![0; current_shape.len()],
                        limits,
                        strides: vec![1; current_shape.len()],
                    },
                )?
            } else {
                // CubeCL's generic complex pad path is not accepted by all
                // CUDA toolkits. Build the same-placement zero tail from a
                // reduced `x - x` scalar, then broadcast and concatenate it;
                // no host transfer or custom kernel is introduced.
                let current = owner.as_ref().map_or(input, |tensor| tensor);
                let zero = session.sub(current, current)?;
                let axes: Vec<usize> = (0..current_shape.len()).collect();
                let zero_scalar = session.reduce_sum(&zero, &axes)?;
                let mut tail_shape = current_shape.clone();
                tail_shape[last] = canonical.n - current_len;
                let zero_tail = session.broadcast_in_dim(&zero_scalar, &tail_shape, &[])?;
                session.concatenate(&[current, &zero_tail], last)?
            };
            owner = Some(transformed);
        }
    }

    let actual = owner.as_ref().map_or(input, |tensor| tensor);
    if actual.shape() != canonical.canonical_input_shape {
        return Err(tenferro_tensor::Error::Internal(
            "CUDA FFT canonicalization returned an unexpected shape".into(),
        ));
    }
    Ok(owner)
}

fn canonical_output_shape(
    canonical: &CanonicalCudaFft,
    operation: FftOperation,
) -> tenferro_tensor::Result<Vec<usize>> {
    let rank = canonical.canonical_input_shape.len();
    let last = rank.checked_sub(1).ok_or_else(|| {
        tenferro_tensor::Error::invalid_argument(OP, "rank", "FFT requires rank >= 1")
    })?;
    let mut shape = canonical.canonical_input_shape.clone();
    shape[last] = if matches!(operation, FftOperation::R2cFull | FftOperation::R2cOnesided) {
        canonical
            .n
            .checked_div(2)
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| {
                tenferro_tensor::Error::invalid_argument(
                    OP,
                    "half spectrum length",
                    "overflows usize",
                )
            })?
    } else {
        canonical.n
    };
    checked_shape_product(OP, "canonical output shape", &shape)?;
    Ok(shape)
}

fn allocate_cuda_output(
    runtime: &CudaRuntime,
    dtype: DType,
    shape: &[usize],
) -> tenferro_tensor::Result<Tensor> {
    match dtype {
        DType::F32 => alloc_output::<f32>(runtime, shape).map(Tensor::F32),
        DType::F64 => alloc_output::<f64>(runtime, shape).map(Tensor::F64),
        DType::C32 => alloc_output::<num_complex::Complex32>(runtime, shape).map(Tensor::C32),
        DType::C64 => alloc_output::<num_complex::Complex64>(runtime, shape).map(Tensor::C64),
        _ => Err(crate::tensor_unsupported_dtype(
            OP,
            dtype,
            "F32, F64, C32, or C64",
        )),
    }
}

fn checked_shape_product(
    op: &'static str,
    role: &'static str,
    shape: &[usize],
) -> tenferro_tensor::Result<usize> {
    shape.iter().try_fold(1usize, |product, &extent| {
        product.checked_mul(extent).ok_or_else(|| {
            tenferro_tensor::Error::invalid_argument(
                op,
                role,
                format!("{role} product overflows usize for shape {shape:?}"),
            )
        })
    })
}

fn is_identity_permutation(permutation: &[usize]) -> bool {
    permutation
        .iter()
        .enumerate()
        .all(|(axis, &mapped)| axis == mapped)
}

fn fft_scale(norm: FftNorm, inverse: bool, n: usize) -> f64 {
    match (norm, inverse) {
        (FftNorm::Backward, false) | (FftNorm::Forward, true) => 1.0,
        (FftNorm::Backward, true) | (FftNorm::Forward, false) => 1.0 / n as f64,
        (FftNorm::Ortho, _) => 1.0 / (n as f64).sqrt(),
    }
}
