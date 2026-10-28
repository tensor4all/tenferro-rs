use tenferro_tensor::DType;

use super::*;
use crate::{rank_revealing_qr::validate_rank_revealing_qr_options, RankRevealingQrOptions};

/// The Rust scalar type behind a preset variant name a macro received.
macro_rules! preset_scalar {
    (F32) => {
        f32
    };
    (F64) => {
        f64
    };
    (I32) => {
        i32
    };
    (I64) => {
        i64
    };
    (Bool) => {
        bool
    };
    (C32) => {
        num_complex::Complex32
    };
    (C64) => {
        num_complex::Complex64
    };
}
const OP: &str = "rank_revealing_qr";

pub(super) fn rank_revealing_qr(
    backend: &mut CudaEngine,
    input: &Tensor,
    options: RankRevealingQrOptions,
) -> Result<Vec<Tensor>> {
    validate_rank_revealing_qr_options(OP, options)?;
    match input.dtype() {
        DType::F32 => {
            let input = gpu_linalg_typed::<f32>(OP, input)?;
            rank_revealing_qr_typed(backend, input, options).map(|(q, r, p, rank)| {
                vec![
                    Tensor::from_typed::<f32>(q),
                    Tensor::from_typed::<f32>(r),
                    Tensor::from_typed::<i64>(p),
                    Tensor::from_typed::<i64>(rank),
                ]
            })
        }
        DType::F64 => {
            let input = gpu_linalg_typed::<f64>(OP, input)?;
            rank_revealing_qr_typed(backend, input, options).map(|(q, r, p, rank)| {
                vec![
                    Tensor::from_typed::<f64>(q),
                    Tensor::from_typed::<f64>(r),
                    Tensor::from_typed::<i64>(p),
                    Tensor::from_typed::<i64>(rank),
                ]
            })
        }
        DType::C32 => {
            let input = gpu_linalg_typed::<Complex32>(OP, input)?;
            rank_revealing_qr_typed(backend, input, options).map(|(q, r, p, rank)| {
                vec![
                    Tensor::from_typed::<tenferro_tensor::Complex32>(q),
                    Tensor::from_typed::<tenferro_tensor::Complex32>(r),
                    Tensor::from_typed::<i64>(p),
                    Tensor::from_typed::<i64>(rank),
                ]
            })
        }
        DType::C64 => {
            let input = gpu_linalg_typed::<Complex64>(OP, input)?;
            rank_revealing_qr_typed(backend, input, options).map(|(q, r, p, rank)| {
                vec![
                    Tensor::from_typed::<tenferro_tensor::Complex64>(q),
                    Tensor::from_typed::<tenferro_tensor::Complex64>(r),
                    Tensor::from_typed::<i64>(p),
                    Tensor::from_typed::<i64>(rank),
                ]
            })
        }
        DType::I32 | DType::I64 | DType::Bool | DType::External(_) => {
            Err(unsupported_linalg_dtype(OP, input))
        }
    }
}

/// The typed tensor behind `input`, or this module's standard refusal.
///
/// Callers reach this from a match on `input.dtype()`, so `None` means the tag table
/// and the runtime dtype disagree rather than a caller mistake.
fn gpu_linalg_typed<'a, T: tenferro_tensor::TensorScalar>(
    op: &'static str,
    input: &'a Tensor,
) -> Result<&'a tenferro_tensor::TypedTensor<T>> {
    input
        .as_typed::<T>()
        .ok_or_else(|| unsupported_linalg_dtype(op, input))
}

#[allow(clippy::type_complexity)]
fn rank_revealing_qr_typed<T>(
    backend: &mut CudaEngine,
    input: &TypedTensor<T>,
    options: RankRevealingQrOptions,
) -> Result<(
    TypedTensor<T>,
    TypedTensor<T>,
    TypedTensor<i64>,
    TypedTensor<i64>,
)>
where
    T: LinalgScalar + RrqrTensorVariant + TensorScalar,
{
    ensure_device_resident_typed(OP, input)?;
    let (m, n) = matrix_dims(OP, input.shape())?;
    let k = m.min(n);
    let batch_shape = &input.shape()[2..];
    let actual_batch_count = checked_shape_product(OP, "batch shape", batch_shape)?;
    let _batch_total = actual_batch_count.max(1);
    u32::try_from(n).map_err(|_| {
        Error::invalid_argument(OP, "column count", "column index exceeds u32 range")
    })?;
    u32::try_from(k).map_err(|_| Error::invalid_argument(OP, "rank", "rank exceeds u32 range"))?;

    let q_shape = matrix_shape(m, k, batch_shape);
    let r_shape = matrix_shape(k, n, batch_shape);
    let permutation_shape = vector_shape(n, batch_shape);
    let rank_shape = batch_shape.to_vec();

    let dtype = T::dtype();
    if m == 0 || n == 0 || actual_batch_count == 0 {
        let q = backend.alloc_output(dtype, &q_shape)?;
        let r = backend.alloc_output(dtype, &r_shape)?;
        let permutation = backend.alloc_output(DType::I64, &permutation_shape)?;
        let rank = backend.alloc_output(DType::I64, &rank_shape)?;
        backend.linalg_initialize_permutation(&permutation, n as i64, OP)?;
        return Ok((
            typed::<T>(q, OP)?,
            typed::<T>(r, OP)?,
            typed::<i64>(permutation, OP)?,
            typed::<i64>(rank, OP)?,
        ));
    }

    let work = clone_device_tensor(backend, input, OP)?;
    let coeff = backend.alloc_output(dtype, &vector_shape(k, batch_shape))?;
    let norms = backend.alloc_output(
        <T as LinalgScalar>::Real::dtype(),
        &vector_shape(n, batch_shape),
    )?;
    let pivots = backend.alloc_output(DType::I64, &rank_shape)?;
    let status = backend.alloc_output(DType::I64, &rank_shape)?;
    let permutation = backend.alloc_output(DType::I64, &permutation_shape)?;
    backend.linalg_initialize_permutation(&permutation, n as i64, OP)?;

    for step in 0..k {
        backend.linalg_column_norms(&work, &norms, step as i64, OP)?;
        backend.linalg_select_pivot(&norms, &permutation, &pivots, &status, step as i64, OP)?;
        backend.linalg_swap_columns(&work, &permutation, &pivots, step as i64, OP)?;
        backend.linalg_reflector(&work, &coeff, step as i64, OP)?;
        let trailing = n - step - 1;
        if trailing > 0 {
            // The reflector lives in column `step`, so writing columns `step + 1..` of the same
            // matrix leaves its write set disjoint.
            backend.linalg_apply_reflector(
                &work,
                &work,
                &coeff,
                step as i64,
                (step + 1) as i64,
                trailing as i64,
                OP,
            )?;
        }
    }

    let r = backend.alloc_output(dtype, &r_shape)?;
    backend.linalg_extract_r(&work, &r, OP)?;
    let q = backend.alloc_output(dtype, &q_shape)?;
    backend.linalg_identity_shifted(&q, 0, OP)?;
    for step in (0..k).rev() {
        backend.linalg_apply_reflector(&work, &q, &coeff, step as i64, 0, k as i64, OP)?;
    }

    let rank = backend.alloc_output(DType::I64, &rank_shape)?;
    backend.linalg_prefix_rank(&r, &rank, &status, options.rtol, options.atol, OP)?;
    let status_tensor = status;

    // The only CUDA-to-host read is this bounded provider-status vector. Matrix
    // payloads, norms, pivots, permutation, and rank remain device-resident.
    backend
        .runtime()
        .synchronize()
        .map_err(|error| Error::Internal(format!("{OP}: stream barrier failed: {error}")))?;
    let host_status = download_tensor(backend.runtime(), &status_tensor)?
        .into_typed::<i64>()
        .map_err(|_| {
            Error::Internal("rank_revealing_qr: unexpected provider-status dtype".into())
        })?;
    if host_status.host_data()?.iter().any(|&value| value != 0) {
        return Err(crate::error::into_tensor_error(
            OP,
            crate::Error::NonFinite {
                op: OP,
                role: "input or computed R diagonal",
            },
        ));
    }

    let mut factors = vec![
        wrap_tensor(typed::<T>(q, OP)?),
        wrap_tensor(typed::<T>(r, OP)?),
    ];
    if options.gauge == QrGauge::PositiveDiagonal {
        apply_qr_gauge_device(backend, &mut factors, 0, OP)?;
    }
    let q = unwrap_tensor(factors.remove(0))?;
    let r = unwrap_tensor(factors.remove(0))?;
    Ok((
        q,
        r,
        typed::<i64>(permutation, OP)?,
        typed::<i64>(rank, OP)?,
    ))
}

fn matrix_shape(rows: usize, columns: usize, batch_shape: &[usize]) -> Vec<usize> {
    let mut shape = vec![rows, columns];
    shape.extend_from_slice(batch_shape);
    shape
}

fn vector_shape(length: usize, batch_shape: &[usize]) -> Vec<usize> {
    let mut shape = vec![length];
    shape.extend_from_slice(batch_shape);
    shape
}

trait RrqrTensorVariant: Sized {
    fn wrap(tensor: TypedTensor<Self>) -> Tensor;
    fn unwrap(tensor: Tensor) -> Result<TypedTensor<Self>>;
}

macro_rules! impl_rrqr_tensor_variant {
    ($scalar:ty, $variant:ident) => {
        impl RrqrTensorVariant for $scalar {
            fn wrap(tensor: TypedTensor<Self>) -> Tensor {
                Tensor::from_typed::<preset_scalar!($variant)>(tensor)
            }

            fn unwrap(tensor: Tensor) -> Result<TypedTensor<Self>> {
                let actual = tensor.dtype();
                tensor
                    .into_typed::<preset_scalar!($variant)>()
                    .map_err(|_| {
                        Error::dtype_mismatch(
                            OP,
                            <$scalar as tenferro_tensor::TensorScalar>::dtype(),
                            actual,
                        )
                    })
            }
        }
    };
}

impl_rrqr_tensor_variant!(f32, F32);
impl_rrqr_tensor_variant!(f64, F64);
impl_rrqr_tensor_variant!(Complex32, C32);
impl_rrqr_tensor_variant!(Complex64, C64);

fn wrap_tensor<T: RrqrTensorVariant>(tensor: TypedTensor<T>) -> Tensor {
    T::wrap(tensor)
}

fn unwrap_tensor<T: RrqrTensorVariant>(tensor: Tensor) -> Result<TypedTensor<T>> {
    T::unwrap(tensor)
}

/// A freshly allocated tensor as its typed form.
fn typed<T: TensorScalar>(tensor: Tensor, op: &'static str) -> Result<TypedTensor<T>> {
    tensor
        .into_typed::<T>()
        .map_err(|_| Error::Internal(format!("{op}: an allocation changed its dtype")))
}
