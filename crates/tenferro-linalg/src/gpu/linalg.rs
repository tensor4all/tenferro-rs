use std::ffi::c_void;
use std::ops::Neg;
use std::os::raw::c_char;

use cubecl::prelude::{ComplexCore, CubeElement, CubePrimitive};
use cubecl_cuda::CudaRuntime as CubeclCudaRuntime;
use num_complex::{Complex32, Complex64};
use num_traits::{One, Zero};

use super::ffi::cusolver::{
    CublasDiagType, CublasFillMode, CublasOperation, CublasPointerMode, CublasSideMode,
    CudaDataType, CudaLinalgHandles, CudaStream, CusolverEigMode,
};
use super::kernels as cubecl_linalg;
use crate::backend::CompactQrResult;
use crate::extension::{QrGauge, QrOptions};
// validate_nonsingular_gpu uses backend ops (extract_diagonal, magnitude,
// reduce_min/reduce_max) then downloads scalar summaries — no bulk host
// roundtrip.
use tenferro_gpu::{
    cuda::cubecl::Session as CubeclSession, cuda::download_tensor, cuda::CudaExecSession,
};
use tenferro_tensor::config::SliceConfig;
use tenferro_tensor::{
    DType, Error, StorageBuffer, Tensor, TensorElementwise, TensorRead, TensorReduction,
    TensorScalar, TensorStructural, TypedTensor, ValidationError,
};

type Result<T> = tenferro_tensor::Result<T>;

trait LinalgScalar:
    CubeElement + CubePrimitive + Copy + Clone + One + Zero + Neg<Output = Self>
{
    type Real: TensorScalar + CubeElement + CubePrimitive + Copy + Clone + Zero;

    const DATA_TYPE: CudaDataType;
    const NEEDS_RWORK: bool;

    fn copy_matrix_adjoint(
        backend: &mut CudaExecSession<'_>,
        v: &TypedTensor<Self>,
        vt_shape: &[usize],
        op: &'static str,
    ) -> Result<TypedTensor<Self>>;

    fn apply_positive_qr_gauge(
        backend: &mut CudaExecSession<'_>,
        q: &mut TypedTensor<Self>,
        r: &mut TypedTensor<Self>,
        q_start: usize,
        op: &'static str,
    ) -> Result<()>;
}

impl LinalgScalar for f32 {
    type Real = f32;

    const DATA_TYPE: CudaDataType = CudaDataType::F32;
    const NEEDS_RWORK: bool = false;

    fn copy_matrix_adjoint(
        backend: &mut CudaExecSession<'_>,
        v: &TypedTensor<Self>,
        vt_shape: &[usize],
        op: &'static str,
    ) -> Result<TypedTensor<Self>> {
        copy_matrix_adjoint_real(backend, v, vt_shape, op)
    }

    fn apply_positive_qr_gauge(
        backend: &mut CudaExecSession<'_>,
        q: &mut TypedTensor<Self>,
        r: &mut TypedTensor<Self>,
        q_start: usize,
        op: &'static str,
    ) -> Result<()> {
        apply_positive_qr_gauge_real(backend, q, r, q_start, op)
    }
}

impl LinalgScalar for f64 {
    type Real = f64;

    const DATA_TYPE: CudaDataType = CudaDataType::F64;
    const NEEDS_RWORK: bool = false;

    fn copy_matrix_adjoint(
        backend: &mut CudaExecSession<'_>,
        v: &TypedTensor<Self>,
        vt_shape: &[usize],
        op: &'static str,
    ) -> Result<TypedTensor<Self>> {
        copy_matrix_adjoint_real(backend, v, vt_shape, op)
    }

    fn apply_positive_qr_gauge(
        backend: &mut CudaExecSession<'_>,
        q: &mut TypedTensor<Self>,
        r: &mut TypedTensor<Self>,
        q_start: usize,
        op: &'static str,
    ) -> Result<()> {
        apply_positive_qr_gauge_real(backend, q, r, q_start, op)
    }
}

impl LinalgScalar for Complex32 {
    type Real = f32;

    const DATA_TYPE: CudaDataType = CudaDataType::Complex32;
    const NEEDS_RWORK: bool = true;

    fn copy_matrix_adjoint(
        backend: &mut CudaExecSession<'_>,
        v: &TypedTensor<Self>,
        vt_shape: &[usize],
        op: &'static str,
    ) -> Result<TypedTensor<Self>> {
        copy_matrix_adjoint_complex(backend, v, vt_shape, op)
    }

    fn apply_positive_qr_gauge(
        backend: &mut CudaExecSession<'_>,
        q: &mut TypedTensor<Self>,
        r: &mut TypedTensor<Self>,
        q_start: usize,
        op: &'static str,
    ) -> Result<()> {
        apply_positive_qr_gauge_c32(backend, q, r, q_start, op)
    }
}

impl LinalgScalar for Complex64 {
    type Real = f64;

    const DATA_TYPE: CudaDataType = CudaDataType::Complex64;
    const NEEDS_RWORK: bool = true;

    fn copy_matrix_adjoint(
        backend: &mut CudaExecSession<'_>,
        v: &TypedTensor<Self>,
        vt_shape: &[usize],
        op: &'static str,
    ) -> Result<TypedTensor<Self>> {
        copy_matrix_adjoint_complex(backend, v, vt_shape, op)
    }

    fn apply_positive_qr_gauge(
        backend: &mut CudaExecSession<'_>,
        q: &mut TypedTensor<Self>,
        r: &mut TypedTensor<Self>,
        q_start: usize,
        op: &'static str,
    ) -> Result<()> {
        apply_positive_qr_gauge_c64(backend, q, r, q_start, op)
    }
}

const JAX_COMPATIBLE_GESVDJ_MAX_DIM: usize = 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SvdDriver {
    Gesvdj,
    Gesvd,
}

fn select_svd_driver(m: usize, n: usize) -> SvdDriver {
    if m <= JAX_COMPATIBLE_GESVDJ_MAX_DIM && n <= JAX_COMPATIBLE_GESVDJ_MAX_DIM {
        SvdDriver::Gesvdj
    } else {
        SvdDriver::Gesvd
    }
}

fn unsupported_linalg_dtype(op: &'static str, input: &Tensor) -> Error {
    crate::error::unsupported_dtype(op, input.dtype())
}

fn ensure_supported_linalg_pair(op: &'static str, lhs: &Tensor, rhs: &Tensor) -> Result<()> {
    if lhs.dtype() != rhs.dtype() {
        return Err(Error::dtype_mismatch(op, lhs.dtype(), rhs.dtype()));
    }
    match lhs {
        Tensor::F32(_) | Tensor::F64(_) | Tensor::C32(_) | Tensor::C64(_) => Ok(()),
        Tensor::I32(_) | Tensor::I64(_) | Tensor::Bool(_) => Err(unsupported_linalg_dtype(op, lhs)),
    }
}

fn ensure_cubecl_resident_tensor(op: &'static str, input: &Tensor) -> Result<()> {
    match input {
        Tensor::F32(t) => ensure_cubecl_resident_typed(op, t),
        Tensor::F64(t) => ensure_cubecl_resident_typed(op, t),
        Tensor::I32(t) => ensure_cubecl_resident_typed(op, t),
        Tensor::I64(t) => ensure_cubecl_resident_typed(op, t),
        Tensor::Bool(t) => ensure_cubecl_resident_typed(op, t),
        Tensor::C32(t) => ensure_cubecl_resident_typed(op, t),
        Tensor::C64(t) => ensure_cubecl_resident_typed(op, t),
    }
}

/// Validate that a tensor is CubeCL-resident using only public tensor
/// metadata (no runtime or launch), so the guard lives entirely on the
/// credentialed public seam.
fn ensure_cubecl_resident_typed<T: 'static>(
    op: &'static str,
    input: &TypedTensor<T>,
) -> Result<()> {
    match input.buffer() {
        StorageBuffer::Host(_) => Err(Error::runtime_state(
            op,
            "expected CubeCL GPU tensor, got host tensor. \
                      Use upload_tensor() to transfer to GPU before calling GPU ops.",
        )),
        StorageBuffer::Backend(buffer) if buffer.backend_family() == "cubecl" => Ok(()),
        StorageBuffer::Backend(buffer) => Err(Error::runtime_state(
            op,
            format!(
                "expected CubeCL GPU tensor, got backend buffer family `{}`",
                buffer.backend_family()
            ),
        )),
    }
}

pub(super) fn cholesky(backend: &mut CudaExecSession<'_>, input: &Tensor) -> Result<Tensor> {
    match input {
        Tensor::F32(t) => cholesky_typed(backend, t).map(Tensor::F32),
        Tensor::F64(t) => cholesky_typed(backend, t).map(Tensor::F64),
        Tensor::C32(t) => cholesky_typed(backend, t).map(Tensor::C32),
        Tensor::C64(t) => cholesky_typed(backend, t).map(Tensor::C64),
        Tensor::I32(_) | Tensor::I64(_) | Tensor::Bool(_) => {
            Err(unsupported_linalg_dtype("cholesky", input))
        }
    }
}

pub(super) fn triangular_solve(
    backend: &mut CudaExecSession<'_>,
    a: &Tensor,
    b: &Tensor,
    left_side: bool,
    lower: bool,
    transpose_a: bool,
    unit_diagonal: bool,
) -> Result<Tensor> {
    match (a, b) {
        (Tensor::F32(a), Tensor::F32(b)) => {
            triangular_solve_typed(backend, a, b, left_side, lower, transpose_a, unit_diagonal)
                .map(Tensor::F32)
        }
        (Tensor::F64(a), Tensor::F64(b)) => {
            triangular_solve_typed(backend, a, b, left_side, lower, transpose_a, unit_diagonal)
                .map(Tensor::F64)
        }
        (Tensor::C32(a), Tensor::C32(b)) => {
            triangular_solve_typed(backend, a, b, left_side, lower, transpose_a, unit_diagonal)
                .map(Tensor::C32)
        }
        (Tensor::C64(a), Tensor::C64(b)) => {
            triangular_solve_typed(backend, a, b, left_side, lower, transpose_a, unit_diagonal)
                .map(Tensor::C64)
        }
        _ if a.dtype() != b.dtype() => Err(Error::dtype_mismatch(
            "triangular_solve",
            a.dtype(),
            b.dtype(),
        )),
        _ => Err(crate::error::unsupported_dtype(
            "triangular_solve",
            a.dtype(),
        )),
    }
}

pub(super) fn lu(backend: &mut CudaExecSession<'_>, input: &Tensor) -> Result<Vec<Tensor>> {
    match input {
        Tensor::F32(t) => lu_typed(backend, t).map(|(p, l, u, parity)| {
            vec![
                Tensor::F32(p),
                Tensor::F32(l),
                Tensor::F32(u),
                Tensor::F32(parity),
            ]
        }),
        Tensor::F64(t) => lu_typed(backend, t).map(|(p, l, u, parity)| {
            vec![
                Tensor::F64(p),
                Tensor::F64(l),
                Tensor::F64(u),
                Tensor::F64(parity),
            ]
        }),
        Tensor::C32(t) => lu_typed(backend, t).map(|(p, l, u, parity)| {
            vec![
                Tensor::C32(p),
                Tensor::C32(l),
                Tensor::C32(u),
                Tensor::C32(parity),
            ]
        }),
        Tensor::C64(t) => lu_typed(backend, t).map(|(p, l, u, parity)| {
            vec![
                Tensor::C64(p),
                Tensor::C64(l),
                Tensor::C64(u),
                Tensor::C64(parity),
            ]
        }),
        Tensor::I32(_) | Tensor::I64(_) | Tensor::Bool(_) => {
            Err(unsupported_linalg_dtype("lu", input))
        }
    }
}

pub(super) fn lu_factor(backend: &mut CudaExecSession<'_>, input: &Tensor) -> Result<Vec<Tensor>> {
    match input {
        Tensor::F32(t) => lu_factor_typed(backend, t).map(|(packed_lu, pivots, parity)| {
            vec![
                Tensor::F32(packed_lu),
                Tensor::I32(pivots),
                Tensor::F32(parity),
            ]
        }),
        Tensor::F64(t) => lu_factor_typed(backend, t).map(|(packed_lu, pivots, parity)| {
            vec![
                Tensor::F64(packed_lu),
                Tensor::I32(pivots),
                Tensor::F64(parity),
            ]
        }),
        Tensor::C32(t) => lu_factor_typed(backend, t).map(|(packed_lu, pivots, parity)| {
            vec![
                Tensor::C32(packed_lu),
                Tensor::I32(pivots),
                Tensor::C32(parity),
            ]
        }),
        Tensor::C64(t) => lu_factor_typed(backend, t).map(|(packed_lu, pivots, parity)| {
            vec![
                Tensor::C64(packed_lu),
                Tensor::I32(pivots),
                Tensor::C64(parity),
            ]
        }),
        Tensor::I32(_) | Tensor::I64(_) | Tensor::Bool(_) => {
            Err(unsupported_linalg_dtype("lu_factor", input))
        }
    }
}

pub(super) fn full_piv_lu(
    _backend: &mut CudaExecSession<'_>,
    _input: &Tensor,
) -> Result<Vec<Tensor>> {
    Err(Error::unsupported(
        "full_piv_lu",
        "complete-pivoting LU is not implemented for the CubeCL backend",
    ))
}

pub(super) fn full_piv_lu_solve(
    _backend: &mut CudaExecSession<'_>,
    _a: &Tensor,
    _b: &Tensor,
    _transpose_a: bool,
) -> Result<Tensor> {
    Err(Error::unsupported(
        "full_piv_lu_solve",
        "complete-pivoting LU solve is not implemented for the CubeCL backend",
    ))
}

pub(super) fn svd(backend: &mut CudaExecSession<'_>, input: &Tensor) -> Result<Vec<Tensor>> {
    match input {
        Tensor::F32(t) => svd_typed(backend, t)
            .map(|(u, s, vt)| vec![Tensor::F32(u), Tensor::F32(s), Tensor::F32(vt)]),
        Tensor::F64(t) => svd_typed(backend, t)
            .map(|(u, s, vt)| vec![Tensor::F64(u), Tensor::F64(s), Tensor::F64(vt)]),
        Tensor::C32(t) => svd_typed(backend, t)
            .map(|(u, s, vt)| vec![Tensor::C32(u), Tensor::F32(s), Tensor::C32(vt)]),
        Tensor::C64(t) => svd_typed(backend, t)
            .map(|(u, s, vt)| vec![Tensor::C64(u), Tensor::F64(s), Tensor::C64(vt)]),
        Tensor::I32(_) | Tensor::I64(_) | Tensor::Bool(_) => {
            Err(unsupported_linalg_dtype("svd", input))
        }
    }
}

pub(super) fn svd_values(backend: &mut CudaExecSession<'_>, input: &Tensor) -> Result<Tensor> {
    match input {
        Tensor::F32(t) => svd_values_typed(backend, t).map(Tensor::F32),
        Tensor::F64(t) => svd_values_typed(backend, t).map(Tensor::F64),
        Tensor::C32(t) => svd_values_typed(backend, t).map(Tensor::F32),
        Tensor::C64(t) => svd_values_typed(backend, t).map(Tensor::F64),
        Tensor::I32(_) | Tensor::I64(_) | Tensor::Bool(_) => {
            Err(unsupported_linalg_dtype("svd_values", input))
        }
    }
}

pub(super) fn qr(backend: &mut CudaExecSession<'_>, input: &Tensor) -> Result<Vec<Tensor>> {
    match input {
        Tensor::F32(t) => qr_typed(backend, t).map(|(q, r)| vec![Tensor::F32(q), Tensor::F32(r)]),
        Tensor::F64(t) => qr_typed(backend, t).map(|(q, r)| vec![Tensor::F64(q), Tensor::F64(r)]),
        Tensor::C32(t) => qr_typed(backend, t).map(|(q, r)| vec![Tensor::C32(q), Tensor::C32(r)]),
        Tensor::C64(t) => qr_typed(backend, t).map(|(q, r)| vec![Tensor::C64(q), Tensor::C64(r)]),
        Tensor::I32(_) | Tensor::I64(_) | Tensor::Bool(_) => {
            Err(unsupported_linalg_dtype("qr", input))
        }
    }
}

pub(super) fn rank_revealing_qr(
    backend: &mut CudaExecSession<'_>,
    input: &Tensor,
    options: crate::RankRevealingQrOptions,
) -> Result<Vec<Tensor>> {
    rank_revealing_qr::rank_revealing_qr(backend, input, options)
}

pub(super) fn householder_qr(
    backend: &mut CudaExecSession<'_>,
    input: &Tensor,
) -> Result<CompactQrResult> {
    let (packed, coeff) = match input {
        Tensor::F32(input) => {
            let (packed, coeff) = compact_qr_typed(backend, input, "householder_qr")?;
            (Tensor::F32(packed), Tensor::F32(coeff))
        }
        Tensor::F64(input) => {
            let (packed, coeff) = compact_qr_typed(backend, input, "householder_qr")?;
            (Tensor::F64(packed), Tensor::F64(coeff))
        }
        Tensor::C32(input) => {
            let (packed, coeff) = compact_qr_typed(backend, input, "householder_qr")?;
            (Tensor::C32(packed), Tensor::C32(coeff))
        }
        Tensor::C64(input) => {
            let (packed, coeff) = compact_qr_typed(backend, input, "householder_qr")?;
            (Tensor::C64(packed), Tensor::C64(coeff))
        }
        Tensor::I32(_) | Tensor::I64(_) | Tensor::Bool(_) => {
            return Err(unsupported_linalg_dtype("householder_qr", input));
        }
    };
    Ok(CompactQrResult { packed, coeff })
}

fn upper_trapezoidal_violation_typed<T>(
    backend: &mut CudaExecSession<'_>,
    input: &TypedTensor<T>,
    op: &'static str,
) -> Result<TypedTensor<i32>>
where
    T: LinalgScalar + TensorScalar + PartialEq,
{
    backend.with_cubecl(op, |cubecl| {
        let output = cubecl.alloc_output::<i32>(input.shape())?;
        if output.n_elements() == 0 {
            return Ok(output);
        }
        let output_arg = cubecl.tensor_binding(&output, op)?;
        let input_arg = cubecl.tensor_binding(input, op)?;
        let launch_count = cubecl.cube_count_1d(output.n_elements())?;
        // SAFETY: bindings are live device tensors and each thread writes one
        // independent validation flag.
        unsafe {
            cubecl_linalg::upper_trapezoidal_violation::launch_unchecked::<T, CubeclCudaRuntime>(
                cubecl.client(),
                launch_count,
                cubecl.cube_dim_1d(),
                output_arg.into_tensor_arg(),
                input_arg.into_tensor_arg(),
            );
        }
        Ok(output)
    })
}

fn validate_upper_trapezoidal_gpu(
    backend: &mut CudaExecSession<'_>,
    r: &Tensor,
    op: &'static str,
) -> Result<()> {
    if r.shape().len() != 2 {
        return Err(Error::rank_mismatch(op, 2, r.shape().len()));
    }
    if r.shape().contains(&0) {
        return Ok(());
    }
    let flags = match r {
        Tensor::F32(r) => upper_trapezoidal_violation_typed(backend, r, op)?,
        Tensor::F64(r) => upper_trapezoidal_violation_typed(backend, r, op)?,
        Tensor::C32(r) => upper_trapezoidal_violation_typed(backend, r, op)?,
        Tensor::C64(r) => upper_trapezoidal_violation_typed(backend, r, op)?,
        Tensor::I32(_) | Tensor::I64(_) | Tensor::Bool(_) => {
            return Err(unsupported_linalg_dtype(op, r));
        }
    };
    let maximum = backend.reduce_max(&Tensor::I32(flags), &[0, 1])?;
    backend.runtime().synchronize()?;
    let host = download_tensor(backend.runtime(), &maximum)?;
    let Tensor::I32(value) = host else {
        return Err(Error::Internal(format!(
            "{op}: unexpected upper-trapezoidal validation dtype"
        )));
    };
    if value.host_data()?[0] == 0 {
        Ok(())
    } else {
        Err(Error::invalid_argument(
            op,
            "r",
            "R must be exactly upper trapezoidal",
        ))
    }
}

pub(super) fn householder_qr_from_factors(
    backend: &mut CudaExecSession<'_>,
    q: &Tensor,
    r: &Tensor,
) -> Result<CompactQrResult> {
    const OP: &str = "householder_qr_from_factors";
    ensure_supported_linalg_pair(OP, q, r)?;
    validate_upper_trapezoidal_gpu(backend, r, OP)?;
    let (packed, coeff) = match (q, r) {
        (Tensor::F32(q), Tensor::F32(r)) => {
            let (packed, coeff) = compact_qr_from_factors_typed(backend, q, r, OP)?;
            (Tensor::F32(packed), Tensor::F32(coeff))
        }
        (Tensor::F64(q), Tensor::F64(r)) => {
            let (packed, coeff) = compact_qr_from_factors_typed(backend, q, r, OP)?;
            (Tensor::F64(packed), Tensor::F64(coeff))
        }
        (Tensor::C32(q), Tensor::C32(r)) => {
            let (packed, coeff) = compact_qr_from_factors_typed(backend, q, r, OP)?;
            (Tensor::C32(packed), Tensor::C32(coeff))
        }
        (Tensor::C64(q), Tensor::C64(r)) => {
            let (packed, coeff) = compact_qr_from_factors_typed(backend, q, r, OP)?;
            (Tensor::C64(packed), Tensor::C64(coeff))
        }
        _ => return Err(Error::dtype_mismatch(OP, q.dtype(), r.dtype())),
    };
    Ok(CompactQrResult { packed, coeff })
}

pub(super) fn householder_qr_append(
    backend: &mut CudaExecSession<'_>,
    packed: &Tensor,
    coeff: &Tensor,
    block: &Tensor,
) -> Result<CompactQrResult> {
    ensure_supported_linalg_pair("householder_qr_append", packed, coeff)?;
    ensure_supported_linalg_pair("householder_qr_append", packed, block)?;
    let (packed, coeff) = match (packed, coeff, block) {
        (Tensor::F32(packed), Tensor::F32(coeff), Tensor::F32(block)) => {
            let (packed, coeff) =
                compact_qr_append_typed(backend, packed, coeff, block, "householder_qr_append")?;
            (Tensor::F32(packed), Tensor::F32(coeff))
        }
        (Tensor::F64(packed), Tensor::F64(coeff), Tensor::F64(block)) => {
            let (packed, coeff) =
                compact_qr_append_typed(backend, packed, coeff, block, "householder_qr_append")?;
            (Tensor::F64(packed), Tensor::F64(coeff))
        }
        (Tensor::C32(packed), Tensor::C32(coeff), Tensor::C32(block)) => {
            let (packed, coeff) =
                compact_qr_append_typed(backend, packed, coeff, block, "householder_qr_append")?;
            (Tensor::C32(packed), Tensor::C32(coeff))
        }
        (Tensor::C64(packed), Tensor::C64(coeff), Tensor::C64(block)) => {
            let (packed, coeff) =
                compact_qr_append_typed(backend, packed, coeff, block, "householder_qr_append")?;
            (Tensor::C64(packed), Tensor::C64(coeff))
        }
        _ => {
            return Err(Error::dtype_mismatch(
                "householder_qr_append",
                packed.dtype(),
                block.dtype(),
            ));
        }
    };
    Ok(CompactQrResult { packed, coeff })
}

pub(super) fn householder_qr_q_columns(
    backend: &mut CudaExecSession<'_>,
    packed: &Tensor,
    coeff: &Tensor,
    start: usize,
    end: usize,
    options: QrOptions,
) -> Result<Tensor> {
    ensure_supported_linalg_pair("householder_qr_q_columns", packed, coeff)?;
    match (packed, coeff) {
        (Tensor::F32(packed), Tensor::F32(coeff)) => compact_qr_q_columns_typed(
            backend,
            packed,
            coeff,
            start,
            end,
            options,
            "householder_qr_q_columns",
        )
        .map(Tensor::F32),
        (Tensor::F64(packed), Tensor::F64(coeff)) => compact_qr_q_columns_typed(
            backend,
            packed,
            coeff,
            start,
            end,
            options,
            "householder_qr_q_columns",
        )
        .map(Tensor::F64),
        (Tensor::C32(packed), Tensor::C32(coeff)) => compact_qr_q_columns_typed(
            backend,
            packed,
            coeff,
            start,
            end,
            options,
            "householder_qr_q_columns",
        )
        .map(Tensor::C32),
        (Tensor::C64(packed), Tensor::C64(coeff)) => compact_qr_q_columns_typed(
            backend,
            packed,
            coeff,
            start,
            end,
            options,
            "householder_qr_q_columns",
        )
        .map(Tensor::C64),
        _ => Err(Error::dtype_mismatch(
            "householder_qr_q_columns",
            packed.dtype(),
            coeff.dtype(),
        )),
    }
}

pub(super) fn householder_qr_r(
    backend: &mut CudaExecSession<'_>,
    packed: &Tensor,
    coeff: &Tensor,
    options: QrOptions,
) -> Result<Tensor> {
    ensure_supported_linalg_pair("householder_qr_r", packed, coeff)?;
    match (packed, coeff) {
        (Tensor::F32(packed), Tensor::F32(coeff)) => {
            compact_qr_r_typed(backend, packed, coeff, options, "householder_qr_r").map(Tensor::F32)
        }
        (Tensor::F64(packed), Tensor::F64(coeff)) => {
            compact_qr_r_typed(backend, packed, coeff, options, "householder_qr_r").map(Tensor::F64)
        }
        (Tensor::C32(packed), Tensor::C32(coeff)) => {
            compact_qr_r_typed(backend, packed, coeff, options, "householder_qr_r").map(Tensor::C32)
        }
        (Tensor::C64(packed), Tensor::C64(coeff)) => {
            compact_qr_r_typed(backend, packed, coeff, options, "householder_qr_r").map(Tensor::C64)
        }
        _ => Err(Error::dtype_mismatch(
            "householder_qr_r",
            packed.dtype(),
            coeff.dtype(),
        )),
    }
}

pub(super) fn qr_with_options(
    backend: &mut CudaExecSession<'_>,
    input: &Tensor,
    options: QrOptions,
) -> Result<Vec<Tensor>> {
    let mut outputs = qr(backend, input)?;
    if options.gauge == QrGauge::PositiveDiagonal {
        apply_qr_gauge_device(backend, &mut outputs, 0, "qr")?;
    }
    Ok(outputs)
}

fn apply_qr_gauge_device(
    backend: &mut CudaExecSession<'_>,
    outputs: &mut [Tensor],
    q_start: usize,
    op: &'static str,
) -> Result<()> {
    if outputs.len() != 2 {
        return Err(Error::Internal(format!(
            "{op}: CUDA QR returned {} outputs",
            outputs.len()
        )));
    }
    let (q, r) = outputs.split_at_mut(1);
    match (&mut q[0], &mut r[0]) {
        (Tensor::F32(q), Tensor::F32(r)) => {
            f32::apply_positive_qr_gauge(backend, q, r, q_start, op)
        }
        (Tensor::F64(q), Tensor::F64(r)) => {
            f64::apply_positive_qr_gauge(backend, q, r, q_start, op)
        }
        (Tensor::C32(q), Tensor::C32(r)) => {
            Complex32::apply_positive_qr_gauge(backend, q, r, q_start, op)
        }
        (Tensor::C64(q), Tensor::C64(r)) => {
            Complex64::apply_positive_qr_gauge(backend, q, r, q_start, op)
        }
        (q, r) => Err(Error::dtype_mismatch(op, q.dtype(), r.dtype())),
    }
}

pub(super) fn eigh(backend: &mut CudaExecSession<'_>, input: &Tensor) -> Result<Vec<Tensor>> {
    match input {
        Tensor::F32(t) => eigh_typed(backend, t).map(|(w, v)| vec![Tensor::F32(w), Tensor::F32(v)]),
        Tensor::F64(t) => eigh_typed(backend, t).map(|(w, v)| vec![Tensor::F64(w), Tensor::F64(v)]),
        Tensor::C32(t) => eigh_typed(backend, t).map(|(w, v)| vec![Tensor::F32(w), Tensor::C32(v)]),
        Tensor::C64(t) => eigh_typed(backend, t).map(|(w, v)| vec![Tensor::F64(w), Tensor::C64(v)]),
        Tensor::I32(_) | Tensor::I64(_) | Tensor::Bool(_) => {
            Err(unsupported_linalg_dtype("eigh", input))
        }
    }
}

pub(super) fn eigh_values(backend: &mut CudaExecSession<'_>, input: &Tensor) -> Result<Tensor> {
    match input {
        Tensor::F32(t) => eigh_values_typed(backend, t).map(Tensor::F32),
        Tensor::F64(t) => eigh_values_typed(backend, t).map(Tensor::F64),
        Tensor::C32(t) => eigh_values_typed(backend, t).map(Tensor::F32),
        Tensor::C64(t) => eigh_values_typed(backend, t).map(Tensor::F64),
        Tensor::I32(_) | Tensor::I64(_) | Tensor::Bool(_) => {
            Err(unsupported_linalg_dtype("eigh_values", input))
        }
    }
}

pub(super) fn eig(_backend: &mut CudaExecSession<'_>, _input: &Tensor) -> Result<Vec<Tensor>> {
    Err(Error::unsupported(
        "eig",
        "non-symmetric eigendecomposition is not supported on the CubeCL GPU backend \
                  because cuSOLVER does not provide it. Download to CPU explicitly via \
                  `backend.download_to_host(&gpu_tensor)?` and then call `CpuBackend::eig`."
            .to_string(),
    ))
}

pub(super) fn solve(backend: &mut CudaExecSession<'_>, a: &Tensor, b: &Tensor) -> Result<Tensor> {
    const OP: &str = "solve";

    ensure_cubecl_resident_tensor(OP, a)?;
    ensure_cubecl_resident_tensor(OP, b)?;
    ensure_supported_linalg_pair(OP, a, b)?;
    if has_zero_dim(a.shape()) || has_zero_dim(b.shape()) {
        return zero_like_linalg_device_tensor(backend, b, OP);
    }

    let (rhs, restore_shape) = if let Some(matrix_rhs_shape) = batched_vector_rhs_shape(a, b) {
        (
            backend.reshape(b, &matrix_rhs_shape)?,
            Some(b.shape().to_vec()),
        )
    } else {
        (
            backend.to_contiguous_read(TensorRead::from_tensor(b))?,
            None,
        )
    };

    let factors = lu_factor(backend, a)?;
    let [packed_lu, pivots, _parity] = factors.as_slice() else {
        return Err(Error::Internal(
            "solve: lu_factor returned an unexpected number of outputs".into(),
        ));
    };
    let x = lu_solve_prepared(backend, a, packed_lu, pivots, &rhs, false, false)?;
    if let Some(shape) = restore_shape {
        backend.reshape(&x, &shape)
    } else {
        Ok(x)
    }
}

pub(super) fn lu_solve_prepared(
    backend: &mut CudaExecSession<'_>,
    a: &Tensor,
    packed_lu: &Tensor,
    pivots: &Tensor,
    b: &Tensor,
    transpose_a: bool,
    conjugate_a: bool,
) -> Result<Tensor> {
    const OP: &str = "lu_solve_prepared";

    ensure_cubecl_resident_tensor(OP, a)?;
    ensure_cubecl_resident_tensor(OP, packed_lu)?;
    ensure_cubecl_resident_tensor(OP, pivots)?;
    ensure_cubecl_resident_tensor(OP, b)?;
    ensure_supported_linalg_pair(OP, a, b)?;
    ensure_supported_linalg_pair(OP, a, packed_lu)?;
    if !matches!(pivots, Tensor::I32(_)) {
        return Err(Error::dtype_mismatch(OP, DType::I32, pivots.dtype()));
    }
    if has_zero_dim(a.shape()) || has_zero_dim(b.shape()) {
        return zero_like_linalg_device_tensor(backend, b, OP);
    }

    let (rhs, restore_shape) = if let Some(matrix_rhs_shape) = batched_vector_rhs_shape(a, b) {
        (
            backend.reshape(b, &matrix_rhs_shape)?,
            Some(b.shape().to_vec()),
        )
    } else {
        (
            backend.to_contiguous_read(TensorRead::from_tensor(b))?,
            None,
        )
    };

    validate_nonsingular_gpu(backend, packed_lu)?;
    let result = match (packed_lu, pivots, &rhs) {
        (Tensor::F32(lu), Tensor::I32(pivots), Tensor::F32(rhs)) => {
            lu_solve_prepared_typed(backend, lu, pivots, rhs, transpose_a, conjugate_a)
                .map(Tensor::F32)
        }
        (Tensor::F64(lu), Tensor::I32(pivots), Tensor::F64(rhs)) => {
            lu_solve_prepared_typed(backend, lu, pivots, rhs, transpose_a, conjugate_a)
                .map(Tensor::F64)
        }
        (Tensor::C32(lu), Tensor::I32(pivots), Tensor::C32(rhs)) => {
            lu_solve_prepared_typed(backend, lu, pivots, rhs, transpose_a, conjugate_a)
                .map(Tensor::C32)
        }
        (Tensor::C64(lu), Tensor::I32(pivots), Tensor::C64(rhs)) => {
            lu_solve_prepared_typed(backend, lu, pivots, rhs, transpose_a, conjugate_a)
                .map(Tensor::C64)
        }
        _ => Err(Error::Internal(
            "lu_solve_prepared: packed LU, pivots, and rhs dtypes are inconsistent".into(),
        )),
    }?;

    if let Some(shape) = restore_shape {
        backend.reshape(&result, &shape)
    } else {
        Ok(result)
    }
}

fn cholesky_typed<T>(
    backend: &mut CudaExecSession<'_>,
    input: &TypedTensor<T>,
) -> Result<TypedTensor<T>>
where
    T: LinalgScalar + TensorScalar,
{
    const OP: &str = "cholesky";

    let n = square_matrix_dim(OP, input.shape())?;
    if has_zero_dim(input.shape()) {
        return Ok(backend.with_raw(OP, |raw| {
            // The fast path still validates residency before allocating the
            // empty output: `raw.tensor` checks the tensor is resident on this
            // runtime/device.
            raw.tensor(input)?;
            raw.alloc_output(input.shape())
        })?);
    }

    // All CUDA work for this operation happens inside a single raw session:
    // the tenferro primary context is current, library handles are acquired
    // once per runtime, and the captured stream orders every enqueued call.
    let work = backend.with_raw(OP, |raw| {
        let handles = raw.resource(CudaLinalgHandles::load)?;
        // SAFETY: the stream handle is valid only for this raw-session scope;
        // it is used immediately to bind the cuSOLVER handle and not retained.
        let stream = unsafe { raw.stream().raw_handle() } as usize as CudaStream;
        handles.cusolver().set_stream(stream, OP)?;

        // Clone `input` into a fresh work matrix on the session stream.
        let mut work = raw.alloc_output::<T>(input.shape())?;
        {
            let src = raw.tensor(input)?;
            let dst = raw.tensor_mut(&mut work)?;
            // SAFETY: both spans were validated resident on this runtime and
            // the copy is stream-ordered; `dst` is exclusively borrowed.
            unsafe {
                raw.copy_bytes(dst.raw_ptr(), src.raw_ptr() as *const _, src.byte_len(), OP)?
            };
        }

        let batch_total = batch_count(OP, &input.shape()[2..])?;
        let lda = as_i32(n, OP, "lda")?;
        let n_i32 = as_i32(n, OP, "n")?;
        let lwork = {
            let first_ref = raw.tensor(&work)?;
            handles.cusolver().potrf_buffer_size(
                T::DATA_TYPE,
                CublasFillMode::Lower,
                n_i32,
                // SAFETY: `first_ref` is a validated device span on this
                // runtime; cuSOLVER only queries the leading dimension here.
                unsafe { first_ref.raw_ptr() },
                lda,
                OP,
            )?
        };
        let workspace_nbytes = {
            let lwork = usize::try_from(lwork).map_err(|_| {
                Error::invalid_argument(
                    OP,
                    "workspace_length",
                    format!("must be non-negative, got {lwork}"),
                )
            })?;
            lwork.checked_mul(std::mem::size_of::<T>()).ok_or_else(|| {
                Error::invalid_argument(OP, "workspace_length", "byte size overflowed")
            })?
        };
        let workspace = raw.alloc_bytes(workspace_nbytes, OP)?;
        let mut workspace_ptr = std::ptr::null_mut::<c_void>();
        workspace.with_ptr(|ptr| workspace_ptr = ptr);

        let mut info = raw.alloc_output::<i32>(&[batch_total])?;
        let info_ref = raw.tensor_mut(&mut info)?;
        // SAFETY: `info_ref` is a validated device span on this runtime; the pointer is used only within the raw-session scope.
        let info_ptr = unsafe { info_ref.raw_ptr() };
        let matrix_stride = checked_mul_usize(OP, "cholesky matrix stride", n, n)?;

        let first_ref = raw.tensor_mut(&mut work)?;
        // SAFETY: `first_ref` is a validated device span on this runtime; the pointer is used only within the raw-session scope.
        let first_ptr = unsafe { first_ref.raw_ptr() };
        for batch in 0..batch_total {
            let a_offset =
                checked_batch_offset(OP, "cholesky matrix batch offset", batch, matrix_stride)?;
            // SAFETY: `a_offset` and `batch` were checked against the matrix
            // and info strides, and both base pointers come from live device
            // tensors validated by this raw session.
            let (batch_a, batch_info) = unsafe {
                (
                    batch_ptr::<T>(first_ptr, a_offset),
                    batch_ptr::<i32>(info_ptr, batch).cast::<i32>(),
                )
            };
            // SAFETY: the batch pointers, workspace, dimensions, and
            // stream-bound handle satisfy cuSOLVER potrf's in-place matrix
            // and info contracts.
            unsafe {
                handles.cusolver().potrf(
                    T::DATA_TYPE,
                    CublasFillMode::Lower,
                    n_i32,
                    batch_a,
                    lda,
                    workspace_ptr,
                    lwork as i32,
                    batch_info,
                    OP,
                )?;
            }
        }

        // Host barrier (only for reading the solver diagnostics).
        let host_info = raw.download_tensor::<i32>(&info, OP)?;
        for &value in host_info.host_data()? {
            check_solver_info(OP, "cusolverDn*potrf", value)?;
        }
        Ok(work)
    })?;

    backend.tril_typed(&work, 0)
}

fn triangular_solve_typed<T>(
    backend: &mut CudaExecSession<'_>,
    a: &TypedTensor<T>,
    b: &TypedTensor<T>,
    left_side: bool,
    lower: bool,
    transpose_a: bool,
    unit_diagonal: bool,
) -> Result<TypedTensor<T>>
where
    T: LinalgScalar + TensorScalar,
{
    let trans = if transpose_a {
        CublasOperation::T
    } else {
        CublasOperation::N
    };
    triangular_solve_typed_with_op(
        backend,
        a,
        b,
        left_side,
        lower,
        trans,
        unit_diagonal,
        "triangular_solve",
    )
}

fn triangular_solve_typed_with_op<T>(
    backend: &mut CudaExecSession<'_>,
    a: &TypedTensor<T>,
    b: &TypedTensor<T>,
    left_side: bool,
    lower: bool,
    trans: CublasOperation,
    unit_diagonal: bool,
    op: &'static str,
) -> Result<TypedTensor<T>>
where
    T: LinalgScalar + TensorScalar,
{
    ensure_cubecl_resident_typed(op, a)?;
    ensure_cubecl_resident_typed(op, b)?;
    let n = square_matrix_dim(op, a.shape())?;
    validate_triangular_rhs(op, a.shape(), b.shape(), left_side)?;
    let side = if left_side {
        CublasSideMode::Left
    } else {
        CublasSideMode::Right
    };
    let uplo = if lower {
        CublasFillMode::Lower
    } else {
        CublasFillMode::Upper
    };
    let diag = if unit_diagonal {
        CublasDiagType::Unit
    } else {
        CublasDiagType::NonUnit
    };
    let rows = b.shape()[0];
    let cols = b.shape()[1];
    let a_stride = checked_mul_usize(op, "triangular matrix stride", n, n)?;
    let out_stride = checked_mul_usize(op, "triangular rhs stride", rows, cols)?;
    let lda = as_i32(n, op, "lda")?;
    let ldb = as_i32(rows, op, "ldb")?;
    let m = as_i32(rows, op, "m")?;
    let n_rhs = as_i32(cols, op, "n")?;
    let alpha = T::one();

    let batch_total = batch_count(op, &b.shape()[2..])?;
    let zero_dim = has_zero_dim(a.shape()) || has_zero_dim(b.shape());

    backend.with_raw(op, |raw| {
        let handles = raw.resource(CudaLinalgHandles::load)?;
        // SAFETY: the stream handle is valid only for this raw-session scope;
        // it is used immediately to bind the cuBLAS handle and not retained.
        let stream = unsafe { raw.stream().raw_handle() } as usize as CudaStream;
        handles.cublas().set_stream(stream, op)?;

        // The fast path still validates residency before allocating the empty
        // output: `raw.tensor` checks the tensors are resident on this
        // runtime/device.
        raw.tensor(a)?;
        raw.tensor(b)?;
        if zero_dim {
            return Ok(raw.alloc_output::<T>(b.shape())?);
        }

        // Clone `b` into a fresh output tensor on the session stream.
        let mut out = raw.alloc_output::<T>(b.shape())?;
        {
            let src = raw.tensor(b)?;
            let dst = raw.tensor_mut(&mut out)?;
            // SAFETY: both spans were validated resident on this runtime and
            // the copy is stream-ordered; `dst` is exclusively borrowed.
            unsafe {
                raw.copy_bytes(dst.raw_ptr(), src.raw_ptr() as *const _, src.byte_len(), op)?
            };
        }

        let a_ref = raw.tensor(a)?;
        let out_ref = raw.tensor_mut(&mut out)?;
        // SAFETY: `a_ref` is a validated device span on this runtime; the pointer is used only within the raw-session scope.
        let a_ptr = unsafe { a_ref.raw_ptr() };
        // SAFETY: `out_ref` is a validated device span on this runtime; the pointer is used only within the raw-session scope.
        let out_ptr = unsafe { out_ref.raw_ptr() };

        if batch_total > 1 {
            let mut a_pointers = Vec::with_capacity(batch_total);
            let mut b_pointers = Vec::with_capacity(batch_total);
            for batch in 0..batch_total {
                let a_offset =
                    checked_batch_offset(op, "triangular matrix batch offset", batch, a_stride)?;
                let b_offset =
                    checked_batch_offset(op, "triangular rhs batch offset", batch, out_stride)?;
                // SAFETY: checked offsets keep both pointers inside the live
                // triangular matrix and RHS device allocations for this batch.
                let (batch_a, batch_b) = unsafe {
                    (
                        batch_const_ptr::<T>(a_ptr.cast_const(), a_offset),
                        batch_ptr::<T>(out_ptr, b_offset),
                    )
                };
                a_pointers.push(batch_a as usize);
                b_pointers.push(batch_b as usize);
            }
            // SAFETY: pointer arrays are built from validated device spans;
            // reinterpreting them as bytes preserves the device-address layout.
            let (a_bytes, b_bytes) = unsafe {
                (
                    std::slice::from_raw_parts(
                        a_pointers.as_ptr().cast::<u8>(),
                        std::mem::size_of_val(&a_pointers[..]),
                    ),
                    std::slice::from_raw_parts(
                        b_pointers.as_ptr().cast::<u8>(),
                        std::mem::size_of_val(&b_pointers[..]),
                    ),
                )
            };
            let a_array = raw.upload_bytes(a_bytes, op)?;
            let b_array = raw.upload_bytes(b_bytes, op)?;
            let mut a_array_ptr = std::ptr::null_mut::<c_void>();
            a_array.with_ptr(|ptr| a_array_ptr = ptr);
            let mut b_array_ptr = std::ptr::null_mut::<c_void>();
            b_array.with_ptr(|ptr| b_array_ptr = ptr);
            // SAFETY: uploaded pointer arrays contain one valid matrix/RHS
            // device pointer per batch, and scalar dimensions/leading
            // dimensions are validated.
            unsafe {
                handles.cublas().trsm_batched(
                    T::DATA_TYPE,
                    side,
                    uplo,
                    trans,
                    diag,
                    m,
                    n_rhs,
                    (&alpha as *const T).cast(),
                    a_array_ptr.cast_const(),
                    lda,
                    b_array_ptr,
                    ldb,
                    as_i32(batch_total, op, "batch_count")?,
                    op,
                )?;
            }
        } else {
            // SAFETY: zero offset points at the first element of the live
            // matrix and RHS device allocations already validated for this
            // single batch.
            let (batch_a, batch_b) = unsafe {
                (
                    batch_const_ptr::<T>(a_ptr.cast_const(), 0),
                    batch_ptr::<T>(out_ptr, 0),
                )
            };
            // SAFETY: pointers, scalar dimensions, and leading dimensions
            // satisfy cuBLAS trsm for the validated single-batch triangular
            // solve.
            unsafe {
                handles.cublas().trsm(
                    T::DATA_TYPE,
                    side,
                    uplo,
                    trans,
                    diag,
                    m,
                    n_rhs,
                    (&alpha as *const T).cast(),
                    batch_a.cast(),
                    lda,
                    batch_b,
                    ldb,
                    op,
                )?;
            }
        }
        Ok(out)
    })
}

fn lu_typed<T>(
    backend: &mut CudaExecSession<'_>,
    input: &TypedTensor<T>,
) -> Result<(
    TypedTensor<T>,
    TypedTensor<T>,
    TypedTensor<T>,
    TypedTensor<T>,
)>
where
    T: LinalgScalar + TensorScalar,
{
    const OP: &str = "lu";

    let (packed_lu, pivots, parity) = lu_factor_typed(backend, input)?;
    let (m, n) = matrix_dims(OP, input.shape())?;
    let (p, l, u, _extracted_parity) =
        build_lu_outputs_device(backend, &packed_lu, &pivots, m, n, &input.shape()[2..])?;
    Ok((p, l, u, parity))
}

fn lu_factor_typed<T>(
    backend: &mut CudaExecSession<'_>,
    input: &TypedTensor<T>,
) -> Result<(TypedTensor<T>, TypedTensor<i32>, TypedTensor<T>)>
where
    T: LinalgScalar + TensorScalar,
{
    const OP: &str = "lu_factor";

    ensure_cubecl_resident_typed(OP, input)?;
    let (m, n) = matrix_dims(OP, input.shape())?;
    let k = m.min(n);
    if has_zero_dim(input.shape()) {
        return zero_sized_lu_factor_outputs(backend, input.shape());
    }

    let mut pivot_shape = vec![k];
    pivot_shape.extend_from_slice(&input.shape()[2..]);
    let batch_total = batch_count(OP, &input.shape()[2..])?;
    let lda = as_i32(m, OP, "lda")?;
    let m_i32 = as_i32(m, OP, "m")?;
    let n_i32 = as_i32(n, OP, "n")?;
    let matrix_stride = checked_mul_usize(OP, "lu matrix stride", m, n)?;

    let (work, pivots) = backend.with_raw(OP, |raw| {
        let handles = raw.resource(CudaLinalgHandles::load)?;
        // SAFETY: the stream handle is valid only for this raw-session scope;
        // it is used immediately to bind the cuSOLVER handle and not retained.
        let stream = unsafe { raw.stream().raw_handle() } as usize as CudaStream;
        handles.cusolver().set_stream(stream, OP)?;

        // Clone `input` into a fresh work matrix on the session stream.
        let mut work = raw.alloc_output::<T>(input.shape())?;
        {
            let src = raw.tensor(input)?;
            let dst = raw.tensor_mut(&mut work)?;
            // SAFETY: both spans were validated resident on this runtime and
            // the copy is stream-ordered; `dst` is exclusively borrowed.
            unsafe {
                raw.copy_bytes(dst.raw_ptr(), src.raw_ptr() as *const _, src.byte_len(), OP)?
            };
        }

        let lwork = {
            let a_ref = raw.tensor(&work)?;
            handles.cusolver().getrf_buffer_size(
                T::DATA_TYPE,
                m_i32,
                n_i32,
                // SAFETY: `a_ref` is a validated device span on this runtime;
                // cuSOLVER only queries the leading dimension here.
                unsafe { a_ref.raw_ptr() },
                lda,
                OP,
            )?
        };
        let workspace_nbytes = {
            let lwork = usize::try_from(lwork).map_err(|_| {
                Error::invalid_argument(
                    OP,
                    "workspace_length",
                    format!("must be non-negative, got {lwork}"),
                )
            })?;
            lwork.checked_mul(std::mem::size_of::<T>()).ok_or_else(|| {
                Error::invalid_argument(OP, "workspace_length", "byte size overflowed")
            })?
        };
        let workspace = raw.alloc_bytes(workspace_nbytes, OP)?;
        let mut workspace_ptr = std::ptr::null_mut::<c_void>();
        workspace.with_ptr(|ptr| workspace_ptr = ptr);

        let mut pivots = raw.alloc_output::<i32>(&pivot_shape)?;
        let mut info = raw.alloc_output::<i32>(&[batch_total])?;
        let a_ref = raw.tensor_mut(&mut work)?;
        // SAFETY: `a_ref` is a validated device span on this runtime; the pointer is used only within the raw-session scope.
        let a_ptr = unsafe { a_ref.raw_ptr() };
        let pivots_ref = raw.tensor_mut(&mut pivots)?;
        // SAFETY: `pivots_ref` is a validated device span on this runtime; the pointer is used only within the raw-session scope.
        let pivots_ptr = unsafe { pivots_ref.raw_ptr() };
        let info_ref = raw.tensor_mut(&mut info)?;
        // SAFETY: `info_ref` is a validated device span on this runtime; the pointer is used only within the raw-session scope.
        let info_ptr = unsafe { info_ref.raw_ptr() };

        for batch in 0..batch_total {
            let a_offset =
                checked_batch_offset(OP, "lu matrix batch offset", batch, matrix_stride)?;
            let pivot_offset = checked_batch_offset(OP, "lu pivot batch offset", batch, k)?;
            // SAFETY: checked matrix, pivot, and info offsets stay inside their
            // live device allocations for this batch.
            let (batch_a, batch_pivots, batch_info) = unsafe {
                (
                    batch_ptr::<T>(a_ptr, a_offset),
                    batch_ptr::<i32>(pivots_ptr, pivot_offset),
                    batch_ptr::<i32>(info_ptr, batch).cast::<i32>(),
                )
            };
            // SAFETY: batch pointers, dimensions, workspace, and stream-bound
            // handle satisfy cuSOLVER getrf's in-place LU and pivot contracts.
            unsafe {
                handles.cusolver().getrf(
                    T::DATA_TYPE,
                    m_i32,
                    n_i32,
                    batch_a,
                    lda,
                    workspace_ptr,
                    batch_pivots.cast::<i32>(),
                    batch_info.cast::<i32>(),
                    OP,
                )?;
            }
        }

        // Host barrier (only for reading the solver diagnostics).
        let host_info = raw.download_tensor::<i32>(&info, OP)?;
        for &info_value in host_info.host_data()? {
            if info_value < 0 {
                return Err(Error::invalid_argument(
                    OP,
                    "cusolver_argument",
                    format!(
                        "cusolverDn*getrf reported invalid parameter {}",
                        -info_value
                    ),
                ));
            }
        }
        Ok((work, pivots))
    })?;

    let parity = build_lu_parity_device(backend, &pivots, k, &input.shape()[2..])?;
    Ok((work, pivots, parity))
}

fn lu_solve_prepared_typed<T>(
    backend: &mut CudaExecSession<'_>,
    packed_lu: &TypedTensor<T>,
    pivots: &TypedTensor<i32>,
    b: &TypedTensor<T>,
    transpose_a: bool,
    conjugate_a: bool,
) -> Result<TypedTensor<T>>
where
    T: LinalgScalar + TensorScalar,
{
    const OP: &str = "lu_solve_prepared";

    ensure_cubecl_resident_typed(OP, packed_lu)?;
    ensure_cubecl_resident_typed(OP, pivots)?;
    ensure_cubecl_resident_typed(OP, b)?;
    validate_lu_solve_prepared_shapes(packed_lu.shape(), pivots.shape(), b.shape())?;
    if has_zero_dim(packed_lu.shape()) || has_zero_dim(b.shape()) {
        return Ok(backend.with_raw(OP, |raw| {
            // The fast path still validates residency before allocating the
            // empty output: `raw.tensor` checks both tensors are resident.
            raw.tensor(packed_lu)?;
            raw.tensor(b)?;
            raw.alloc_output::<T>(b.shape())
        })?);
    }

    match (transpose_a, conjugate_a) {
        (false, false) => {
            let pb = apply_lu_pivots_typed(backend, b, pivots, false)?;
            let z = triangular_solve_typed_with_op(
                backend,
                packed_lu,
                &pb,
                true,
                true,
                CublasOperation::N,
                true,
                OP,
            )?;
            triangular_solve_typed_with_op(
                backend,
                packed_lu,
                &z,
                true,
                false,
                CublasOperation::N,
                false,
                OP,
            )
        }
        (true, conjugate_a) => {
            let trans = if conjugate_a {
                CublasOperation::C
            } else {
                CublasOperation::T
            };
            let z = triangular_solve_typed_with_op(
                backend, packed_lu, b, true, false, trans, false, OP,
            )?;
            let y = triangular_solve_typed_with_op(
                backend, packed_lu, &z, true, true, trans, true, OP,
            )?;
            apply_lu_pivots_typed(backend, &y, pivots, true)
        }
        (false, true) => Err(Error::unsupported(
            OP,
            "conjugate-only prepared LU solve is unsupported on CUDA; use transpose+conjugate or solve the conjugated matrix explicitly",
        )),
    }
}

fn copy_matrix_adjoint_real<T>(
    backend: &mut CudaExecSession<'_>,
    v: &TypedTensor<T>,
    vt_shape: &[usize],
    op: &'static str,
) -> Result<TypedTensor<T>>
where
    T: LinalgScalar + TensorScalar,
{
    backend.with_cubecl(op, |cubecl| {
        let vt = cubecl.alloc_output::<T>(vt_shape)?;
        launch_matrix_adjoint_real(cubecl, v, &vt, op)?;
        Ok(vt)
    })
}

fn copy_matrix_adjoint_complex<T>(
    backend: &mut CudaExecSession<'_>,
    v: &TypedTensor<T>,
    vt_shape: &[usize],
    op: &'static str,
) -> Result<TypedTensor<T>>
where
    T: LinalgScalar + TensorScalar + ComplexCore,
{
    backend.with_cubecl(op, |cubecl| {
        let vt = cubecl.alloc_output::<T>(vt_shape)?;
        launch_matrix_adjoint_complex(cubecl, v, &vt, op)?;
        Ok(vt)
    })
}

fn clone_device_tensor<T>(
    backend: &mut CudaExecSession<'_>,
    input: &TypedTensor<T>,
    op: &'static str,
) -> Result<TypedTensor<T>>
where
    T: LinalgScalar + TensorScalar,
{
    backend.with_raw(op, |raw| {
        let mut output = raw.alloc_output::<T>(input.shape())?;
        let input_ref = raw.tensor(input)?;
        let output_ref = raw.tensor_mut(&mut output)?;
        // SAFETY: both spans are validated on this runtime, output is freshly
        // allocated and exclusively borrowed, and the copy is stream ordered.
        unsafe {
            raw.copy_bytes(
                output_ref.raw_ptr(),
                input_ref.raw_ptr() as *const _,
                input_ref.byte_len(),
                op,
            )?;
        }
        Ok(output)
    })
}

fn apply_positive_qr_gauge_real<T>(
    backend: &mut CudaExecSession<'_>,
    q: &mut TypedTensor<T>,
    r: &mut TypedTensor<T>,
    q_start: usize,
    op: &'static str,
) -> Result<()>
where
    T: LinalgScalar + TensorScalar + PartialOrd + std::ops::Mul<Output = T>,
{
    backend.with_cubecl(op, |cubecl| {
        let mut phase_shape = vec![r.shape()[0]];
        phase_shape.extend_from_slice(&r.shape()[2..]);
        let phase = cubecl.alloc_output::<T>(&phase_shape)?;
        let launch_len = q.n_elements().max(r.n_elements());
        if launch_len == 0 {
            return Ok(());
        }
        let phase_output_arg = cubecl.tensor_binding(&phase, op)?;
        let r_input_arg = cubecl.tensor_binding(r, op)?;
        let phase_count = cubecl.cube_count_1d(phase.n_elements())?;
        // SAFETY: phase/R bindings are live and phase_count covers each batch diagonal.
        unsafe {
            cubecl_linalg::qr_phase_real::launch_unchecked::<T, CubeclCudaRuntime>(
                cubecl.client(),
                phase_count,
                cubecl.cube_dim_1d(),
                phase_output_arg.into_tensor_arg(),
                r_input_arg.into_tensor_arg(),
                r.shape().len(),
            );
        }
        let phase_arg = cubecl.tensor_binding(&phase, op)?;
        let q_arg = cubecl.tensor_binding(q, op)?;
        let r_arg = cubecl.tensor_binding(r, op)?;
        let launch_count = cubecl.cube_count_1d(launch_len)?;
        // SAFETY: phase computation is stream-ordered before scaling, Q/R are
        // live mutable outputs, and the launch covers both domains.
        unsafe {
            cubecl_linalg::qr_apply_phase_real::launch_unchecked::<T, CubeclCudaRuntime>(
                cubecl.client(),
                launch_count,
                cubecl.cube_dim_1d(),
                q_arg.into_tensor_arg(),
                r_arg.into_tensor_arg(),
                phase_arg.into_tensor_arg(),
                q_start,
                r.shape().len(),
            );
        }
        Ok(())
    })
}

fn apply_positive_qr_gauge_c32(
    backend: &mut CudaExecSession<'_>,
    q: &mut TypedTensor<Complex32>,
    r: &mut TypedTensor<Complex32>,
    q_start: usize,
    op: &'static str,
) -> Result<()> {
    backend.with_cubecl(op, |cubecl| {
        let mut phase_shape = vec![r.shape()[0]];
        phase_shape.extend_from_slice(&r.shape()[2..]);
        let phase = cubecl.alloc_output::<Complex32>(&phase_shape)?;
        let launch_len = q.n_elements().max(r.n_elements());
        if launch_len == 0 {
            return Ok(());
        }
        let phase_output_arg = cubecl.tensor_binding(&phase, op)?;
        let r_input_arg = cubecl.tensor_binding(r, op)?;
        let phase_count = cubecl.cube_count_1d(phase.n_elements())?;
        // SAFETY: phase/R bindings are live and phase_count covers each batch diagonal.
        unsafe {
            cubecl_linalg::qr_phase_c32::launch_unchecked::<CubeclCudaRuntime>(
                cubecl.client(),
                phase_count,
                cubecl.cube_dim_1d(),
                phase_output_arg.into_tensor_arg(),
                r_input_arg.into_tensor_arg(),
                r.shape().len(),
            );
        }
        let phase_arg = cubecl.tensor_binding(&phase, op)?;
        let q_arg = cubecl.tensor_binding(q, op)?;
        let r_arg = cubecl.tensor_binding(r, op)?;
        let launch_count = cubecl.cube_count_1d(launch_len)?;
        // SAFETY: phase computation is stream-ordered before scaling and the
        // second launch covers Q/R exactly.
        unsafe {
            cubecl_linalg::qr_apply_phase_complex::launch_unchecked::<Complex32, CubeclCudaRuntime>(
                cubecl.client(),
                launch_count,
                cubecl.cube_dim_1d(),
                q_arg.into_tensor_arg(),
                r_arg.into_tensor_arg(),
                phase_arg.into_tensor_arg(),
                q_start,
                r.shape().len(),
            );
        }
        Ok(())
    })
}

fn apply_positive_qr_gauge_c64(
    backend: &mut CudaExecSession<'_>,
    q: &mut TypedTensor<Complex64>,
    r: &mut TypedTensor<Complex64>,
    q_start: usize,
    op: &'static str,
) -> Result<()> {
    backend.with_cubecl(op, |cubecl| {
        let mut phase_shape = vec![r.shape()[0]];
        phase_shape.extend_from_slice(&r.shape()[2..]);
        let phase = cubecl.alloc_output::<Complex64>(&phase_shape)?;
        let launch_len = q.n_elements().max(r.n_elements());
        if launch_len == 0 {
            return Ok(());
        }
        let phase_output_arg = cubecl.tensor_binding(&phase, op)?;
        let r_input_arg = cubecl.tensor_binding(r, op)?;
        let phase_count = cubecl.cube_count_1d(phase.n_elements())?;
        // SAFETY: phase/R bindings are live and phase_count covers each batch diagonal.
        unsafe {
            cubecl_linalg::qr_phase_c64::launch_unchecked::<CubeclCudaRuntime>(
                cubecl.client(),
                phase_count,
                cubecl.cube_dim_1d(),
                phase_output_arg.into_tensor_arg(),
                r_input_arg.into_tensor_arg(),
                r.shape().len(),
            );
        }
        let phase_arg = cubecl.tensor_binding(&phase, op)?;
        let q_arg = cubecl.tensor_binding(q, op)?;
        let r_arg = cubecl.tensor_binding(r, op)?;
        let launch_count = cubecl.cube_count_1d(launch_len)?;
        // SAFETY: phase computation is stream-ordered before scaling and the
        // second launch covers Q/R exactly.
        unsafe {
            cubecl_linalg::qr_apply_phase_complex::launch_unchecked::<Complex64, CubeclCudaRuntime>(
                cubecl.client(),
                launch_count,
                cubecl.cube_dim_1d(),
                q_arg.into_tensor_arg(),
                r_arg.into_tensor_arg(),
                phase_arg.into_tensor_arg(),
                q_start,
                r.shape().len(),
            );
        }
        Ok(())
    })
}

fn launch_matrix_adjoint_real<T>(
    cubecl: &CubeclSession<'_>,
    v: &TypedTensor<T>,
    vt: &TypedTensor<T>,
    op: &'static str,
) -> Result<()>
where
    T: LinalgScalar + TensorScalar,
{
    if vt.n_elements() == 0 {
        return Ok(());
    }
    let vt_arg = cubecl.tensor_binding(vt, op)?;
    let v_arg = cubecl.tensor_binding(v, op)?;
    let launch_count = cubecl.cube_count_1d(vt.n_elements())?;
    // SAFETY: tensor bindings describe live CUDA tensors, and `launch_count`
    // covers exactly the output domain for the matrix-adjoint copy kernel.
    unsafe {
        cubecl_linalg::matrix_adjoint_real::launch_unchecked::<T, CubeclCudaRuntime>(
            cubecl.client(),
            launch_count,
            cubecl.cube_dim_1d(),
            vt_arg.into_tensor_arg(),
            v_arg.into_tensor_arg(),
            vt.shape().len(),
        );
    }
    Ok(())
}

fn launch_matrix_adjoint_complex<T>(
    cubecl: &CubeclSession<'_>,
    v: &TypedTensor<T>,
    vt: &TypedTensor<T>,
    op: &'static str,
) -> Result<()>
where
    T: LinalgScalar + TensorScalar + ComplexCore,
{
    if vt.n_elements() == 0 {
        return Ok(());
    }
    let vt_arg = cubecl.tensor_binding(vt, op)?;
    let v_arg = cubecl.tensor_binding(v, op)?;
    let launch_count = cubecl.cube_count_1d(vt.n_elements())?;
    // SAFETY: tensor bindings describe live CUDA tensors, and `launch_count`
    // covers exactly the output domain for the conjugating matrix-adjoint kernel.
    unsafe {
        cubecl_linalg::matrix_adjoint_complex::launch_unchecked::<T, CubeclCudaRuntime>(
            cubecl.client(),
            launch_count,
            cubecl.cube_dim_1d(),
            vt_arg.into_tensor_arg(),
            v_arg.into_tensor_arg(),
            vt.shape().len(),
        );
    }
    Ok(())
}

fn svd_typed<T>(
    backend: &mut CudaExecSession<'_>,
    input: &TypedTensor<T>,
) -> Result<(
    TypedTensor<T>,
    TypedTensor<<T as LinalgScalar>::Real>,
    TypedTensor<T>,
)>
where
    T: LinalgScalar + TensorScalar,
{
    const OP: &str = "svd";

    ensure_cubecl_resident_typed(OP, input)?;
    let (m, n) = matrix_dims(OP, input.shape())?;
    let k = m.min(n);
    let batch_shape = &input.shape()[2..];
    let mut u_shape = vec![m, k];
    u_shape.extend_from_slice(batch_shape);
    let mut s_shape = vec![k];
    s_shape.extend_from_slice(batch_shape);
    let mut vt_shape = vec![k, n];
    vt_shape.extend_from_slice(batch_shape);
    if has_zero_dim(input.shape()) {
        return Ok(backend.with_raw(OP, |raw| {
            // The fast path still validates residency before allocating the
            // empty outputs: `raw.tensor` checks the tensor is resident.
            raw.tensor(input)?;
            Ok((
                raw.alloc_output::<T>(&u_shape)?,
                raw.alloc_output::<<T as LinalgScalar>::Real>(&s_shape)?,
                raw.alloc_output::<T>(&vt_shape)?,
            ))
        })?);
    }
    let batch_total = batch_count(OP, batch_shape)?;
    let a_stride = checked_mul_usize(OP, "svd input stride", m, n)?;
    let s_stride = k;

    match select_svd_driver(m, n) {
        SvdDriver::Gesvdj => {
            let mut v_shape = vec![n, k];
            v_shape.extend_from_slice(batch_shape);
            let (u, s, v) = backend.with_raw(OP, |raw| {
                let handles = raw.resource(CudaLinalgHandles::load)?;
                // SAFETY: the stream handle is valid only for this raw-session
                // scope; it is used immediately to bind cuSOLVER and not retained.
                let stream = unsafe { raw.stream().raw_handle() } as usize as CudaStream;
                handles.cusolver().set_stream(stream, OP)?;

                // Clone `input` into a fresh work matrix on the session stream.
                let mut work = raw.alloc_output::<T>(input.shape())?;
                {
                    let src = raw.tensor(input)?;
                    let dst = raw.tensor_mut(&mut work)?;
                    // SAFETY: both spans were validated resident on this runtime
                    // and the copy is stream-ordered; `dst` is exclusively borrowed.
                    unsafe {
                        raw.copy_bytes(
                            dst.raw_ptr(),
                            src.raw_ptr() as *const _,
                            src.byte_len(),
                            OP,
                        )?
                    };
                }

                let mut u = raw.alloc_output::<T>(&u_shape)?;
                let mut v = raw.alloc_output::<T>(&v_shape)?;
                let mut s = raw.alloc_output::<<T as LinalgScalar>::Real>(&s_shape)?;

                let m_i32 = as_i32(m, OP, "m")?;
                let n_i32 = as_i32(n, OP, "n")?;
                let lda = as_i32(m, OP, "lda")?;
                let ldu = as_i32(m, OP, "ldu")?;
                let ldv = as_i32(n, OP, "ldv")?;
                let params = handles.cusolver().create_gesvdj_info(OP)?;
                let lwork = {
                    let a_ref = raw.tensor(&work)?;
                    let u_ref = raw.tensor(&u)?;
                    let s_ref = raw.tensor(&s)?;
                    let v_ref = raw.tensor(&v)?;
                    handles.cusolver().gesvdj_buffer_size(
                        T::DATA_TYPE,
                        CusolverEigMode::Vector,
                        1,
                        m_i32,
                        n_i32,
                        // SAFETY: all spans are validated device allocations on
                        // this runtime; only leading dimensions are queried here.
                        unsafe { a_ref.raw_ptr().cast_const() },
                        lda,
                        // SAFETY: `s_ref` is a validated device span on this runtime; the pointer is used only within the raw-session scope.
                        unsafe { s_ref.raw_ptr().cast_const() },
                        // SAFETY: `u_ref` is a validated device span on this runtime; the pointer is used only within the raw-session scope.
                        unsafe { u_ref.raw_ptr().cast_const() },
                        ldu,
                        // SAFETY: `v_ref` is a validated device span on this runtime; the pointer is used only within the raw-session scope.
                        unsafe { v_ref.raw_ptr().cast_const() },
                        ldv,
                        &params,
                        OP,
                    )?
                };
                let workspace_nbytes = {
                    let lwork = usize::try_from(lwork).map_err(|_| {
                        Error::invalid_argument(
                            OP,
                            "workspace_length",
                            format!("must be non-negative, got {lwork}"),
                        )
                    })?;
                    lwork.checked_mul(std::mem::size_of::<T>()).ok_or_else(|| {
                        Error::invalid_argument(OP, "workspace_length", "byte size overflowed")
                    })?
                };
                let workspace = raw.alloc_bytes(workspace_nbytes, OP)?;
                let mut workspace_ptr = std::ptr::null_mut::<c_void>();
                workspace.with_ptr(|ptr| workspace_ptr = ptr);

                let mut info = raw.alloc_output::<i32>(&[batch_total])?;
                let a_ref = raw.tensor_mut(&mut work)?;
                // SAFETY: `a_ref` is a validated device span on this runtime; the pointer is used only within the raw-session scope.
                let a_ptr = unsafe { a_ref.raw_ptr() };
                let u_ref = raw.tensor_mut(&mut u)?;
                // SAFETY: `u_ref` is a validated device span on this runtime; the pointer is used only within the raw-session scope.
                let u_ptr = unsafe { u_ref.raw_ptr() };
                let s_ref = raw.tensor_mut(&mut s)?;
                // SAFETY: `s_ref` is a validated device span on this runtime; the pointer is used only within the raw-session scope.
                let s_ptr = unsafe { s_ref.raw_ptr() };
                let v_ref = raw.tensor_mut(&mut v)?;
                // SAFETY: `v_ref` is a validated device span on this runtime; the pointer is used only within the raw-session scope.
                let v_ptr = unsafe { v_ref.raw_ptr() };
                let info_ref = raw.tensor_mut(&mut info)?;
                // SAFETY: `info_ref` is a validated device span on this runtime; the pointer is used only within the raw-session scope.
                let info_ptr = unsafe { info_ref.raw_ptr() };
                let u_stride = checked_mul_usize(OP, "svd u stride", m, k)?;
                let v_stride = checked_mul_usize(OP, "svd v stride", n, k)?;

                for batch in 0..batch_total {
                    let a_offset =
                        checked_batch_offset(OP, "svd input batch offset", batch, a_stride)?;
                    let s_offset = checked_batch_offset(
                        OP,
                        "svd singular value batch offset",
                        batch,
                        s_stride,
                    )?;
                    let u_offset = checked_batch_offset(OP, "svd u batch offset", batch, u_stride)?;
                    let v_offset = checked_batch_offset(OP, "svd v batch offset", batch, v_stride)?;
                    // SAFETY: all offsets are checked against their per-batch
                    // strides and each base pointer belongs to a live device tensor.
                    let (batch_a, batch_s, batch_u, batch_v, batch_info) = unsafe {
                        (
                            batch_ptr::<T>(a_ptr, a_offset),
                            batch_ptr::<<T as LinalgScalar>::Real>(s_ptr, s_offset),
                            batch_ptr::<T>(u_ptr, u_offset),
                            batch_ptr::<T>(v_ptr, v_offset),
                            batch_ptr::<i32>(info_ptr, batch).cast::<i32>(),
                        )
                    };
                    // SAFETY: batch pointers, workspaces, dimensions, and
                    // gesvdj params satisfy cuSOLVER's vector SVD contract.
                    unsafe {
                        handles.cusolver().gesvdj(
                            T::DATA_TYPE,
                            CusolverEigMode::Vector,
                            1,
                            m_i32,
                            n_i32,
                            batch_a,
                            lda,
                            batch_s,
                            batch_u,
                            ldu,
                            batch_v,
                            ldv,
                            workspace_ptr,
                            lwork,
                            batch_info,
                            &params,
                            OP,
                        )?;
                    }
                }

                // Host barrier (only for reading the solver diagnostics).
                let host_info = raw.download_tensor::<i32>(&info, OP)?;
                for &value in host_info.host_data()? {
                    check_solver_info(OP, "cusolverDn*gesvdj", value)?;
                }
                Ok((u, s, v))
            })?;
            let vt = T::copy_matrix_adjoint(backend, &v, &vt_shape, OP)?;
            Ok((u, s, vt))
        }
        SvdDriver::Gesvd => {
            let transpose_for_gesvd = m < n;
            let (gesvd_m, gesvd_n) = if transpose_for_gesvd { (n, m) } else { (m, n) };
            let mut gesvd_u_shape = vec![gesvd_m, k];
            gesvd_u_shape.extend_from_slice(batch_shape);
            let mut gesvd_vt_shape = vec![k, gesvd_n];
            gesvd_vt_shape.extend_from_slice(batch_shape);

            // When `m < n` gesvd factors `adjoint(input)` (n×m); the CubeCL
            // adjoint kernel is flushed before the raw session, so the SVD
            // below observes it. The SVD results are adjointed again below.
            let transposed_work = if transpose_for_gesvd {
                let mut work_shape = vec![n, m];
                work_shape.extend_from_slice(batch_shape);
                Some(T::copy_matrix_adjoint(backend, input, &work_shape, OP)?)
            } else {
                None
            };

            let (gesvd_u, s, gesvd_vt) = backend.with_raw(OP, |raw| {
                let handles = raw.resource(CudaLinalgHandles::load)?;
                // SAFETY: the stream handle is valid only for this raw-session
                // scope; it is used immediately to bind cuSOLVER and not retained.
                let stream = unsafe { raw.stream().raw_handle() } as usize as CudaStream;
                handles.cusolver().set_stream(stream, OP)?;

                // `work` is either the CubeCL-adjointed input (m < n) or a
                // clone of `input` made on the session stream.
                let mut work = match transposed_work {
                    Some(work) => work,
                    None => {
                        let mut work = raw.alloc_output::<T>(input.shape())?;
                        {
                            let src = raw.tensor(input)?;
                            let dst = raw.tensor_mut(&mut work)?;
                            // SAFETY: both spans were validated resident on this
                            // runtime and the copy is stream-ordered; `dst` is
                            // exclusively borrowed.
                            unsafe {
                                raw.copy_bytes(
                                    dst.raw_ptr(),
                                    src.raw_ptr() as *const _,
                                    src.byte_len(),
                                    OP,
                                )?
                            };
                        }
                        work
                    }
                };
                let mut s = raw.alloc_output::<<T as LinalgScalar>::Real>(&s_shape)?;
                let mut gesvd_u = raw.alloc_output::<T>(&gesvd_u_shape)?;
                let mut gesvd_vt = raw.alloc_output::<T>(&gesvd_vt_shape)?;

                let a_ref = raw.tensor_mut(&mut work)?;
                // SAFETY: `a_ref` is a validated device span on this runtime; the pointer is used only within the raw-session scope.
                let a_ptr = unsafe { a_ref.raw_ptr() };
                let u_ref = raw.tensor_mut(&mut gesvd_u)?;
                // SAFETY: `u_ref` is a validated device span on this runtime; the pointer is used only within the raw-session scope.
                let u_ptr = unsafe { u_ref.raw_ptr() };
                let s_ref = raw.tensor_mut(&mut s)?;
                // SAFETY: `s_ref` is a validated device span on this runtime; the pointer is used only within the raw-session scope.
                let s_ptr = unsafe { s_ref.raw_ptr() };
                let vt_ref = raw.tensor_mut(&mut gesvd_vt)?;
                // SAFETY: `vt_ref` is a validated device span on this runtime; the pointer is used only within the raw-session scope.
                let vt_ptr = unsafe { vt_ref.raw_ptr() };
                let gesvd_m_i32 = as_i32(gesvd_m, OP, "gesvd m")?;
                let gesvd_n_i32 = as_i32(gesvd_n, OP, "gesvd n")?;
                let lda = gesvd_m_i32;
                let ldu = gesvd_m_i32;
                let ldvt = as_i32(k, OP, "ldvt")?;
                let lwork = handles.cusolver().gesvd_buffer_size(
                    T::DATA_TYPE,
                    gesvd_m_i32,
                    gesvd_n_i32,
                    OP,
                )?;
                let workspace_nbytes = {
                    let lwork = usize::try_from(lwork).map_err(|_| {
                        Error::invalid_argument(
                            OP,
                            "workspace_length",
                            format!("must be non-negative, got {lwork}"),
                        )
                    })?;
                    lwork.checked_mul(std::mem::size_of::<T>()).ok_or_else(|| {
                        Error::invalid_argument(OP, "workspace_length", "byte size overflowed")
                    })?
                };
                let workspace = raw.alloc_bytes(workspace_nbytes, OP)?;
                let mut workspace_ptr = std::ptr::null_mut::<c_void>();
                workspace.with_ptr(|ptr| workspace_ptr = ptr);
                let mut rwork_ptr = std::ptr::null_mut::<c_void>();
                if T::NEEDS_RWORK {
                    let rwork_len = as_i32(
                        checked_mul_usize(OP, "svd rwork length", 5, k)?,
                        OP,
                        "rwork",
                    )?;
                    let rwork_nbytes = usize::try_from(rwork_len)
                        .map_err(|_| {
                            Error::invalid_argument(OP, "rwork_length", "must be non-negative")
                        })?
                        .checked_mul(std::mem::size_of::<<T as LinalgScalar>::Real>())
                        .ok_or_else(|| {
                            Error::invalid_argument(OP, "rwork_length", "byte size overflowed")
                        })?;
                    let rwork = raw.alloc_bytes(rwork_nbytes, OP)?;
                    rwork.with_ptr(|ptr| rwork_ptr = ptr);
                }
                let mut info = raw.alloc_output::<i32>(&[batch_total])?;
                let info_ref = raw.tensor_mut(&mut info)?;
                // SAFETY: `info_ref` is a validated device span on this runtime; the pointer is used only within the raw-session scope.
                let info_ptr = unsafe { info_ref.raw_ptr() };
                let u_stride = checked_mul_usize(OP, "svd gesvd u stride", gesvd_m, k)?;
                let vt_stride = checked_mul_usize(OP, "svd gesvd vt stride", k, gesvd_n)?;
                let job = b'S' as c_char;

                for batch in 0..batch_total {
                    let a_offset =
                        checked_batch_offset(OP, "svd input batch offset", batch, a_stride)?;
                    let s_offset = checked_batch_offset(
                        OP,
                        "svd singular value batch offset",
                        batch,
                        s_stride,
                    )?;
                    let u_offset = checked_batch_offset(OP, "svd u batch offset", batch, u_stride)?;
                    let vt_offset =
                        checked_batch_offset(OP, "svd vt batch offset", batch, vt_stride)?;
                    // SAFETY: all offsets are checked against their per-batch
                    // strides and each base pointer belongs to a live device tensor.
                    let (batch_a, batch_s, batch_u, batch_vt, batch_info) = unsafe {
                        (
                            batch_ptr::<T>(a_ptr, a_offset),
                            batch_ptr::<<T as LinalgScalar>::Real>(s_ptr, s_offset),
                            batch_ptr::<T>(u_ptr, u_offset),
                            batch_ptr::<T>(vt_ptr, vt_offset),
                            batch_ptr::<i32>(info_ptr, batch).cast::<i32>(),
                        )
                    };
                    // SAFETY: batch pointers, workspace/rwork, dimensions, and
                    // stream-bound handle satisfy cuSOLVER gesvd's thin SVD contract.
                    unsafe {
                        handles.cusolver().gesvd(
                            T::DATA_TYPE,
                            job,
                            job,
                            gesvd_m_i32,
                            gesvd_n_i32,
                            batch_a,
                            lda,
                            batch_s,
                            batch_u,
                            ldu,
                            batch_vt,
                            ldvt,
                            workspace_ptr,
                            lwork,
                            rwork_ptr,
                            batch_info,
                            OP,
                        )?;
                    }
                }

                // Host barrier (only for reading the solver diagnostics).
                let host_info = raw.download_tensor::<i32>(&info, OP)?;
                for &value in host_info.host_data()? {
                    check_solver_info(OP, "cusolverDn*gesvd", value)?;
                }
                Ok((gesvd_u, s, gesvd_vt))
            })?;

            if transpose_for_gesvd {
                let u = T::copy_matrix_adjoint(backend, &gesvd_vt, &u_shape, OP)?;
                let vt = T::copy_matrix_adjoint(backend, &gesvd_u, &vt_shape, OP)?;
                Ok((u, s, vt))
            } else {
                Ok((gesvd_u, s, gesvd_vt))
            }
        }
    }
}

fn svd_values_typed<T>(
    backend: &mut CudaExecSession<'_>,
    input: &TypedTensor<T>,
) -> Result<TypedTensor<<T as LinalgScalar>::Real>>
where
    T: LinalgScalar + TensorScalar,
{
    const OP: &str = "svd_values";

    ensure_cubecl_resident_typed(OP, input)?;
    let (m, n) = matrix_dims(OP, input.shape())?;
    let k = m.min(n);
    let batch_shape = &input.shape()[2..];
    let mut s_shape = vec![k];
    s_shape.extend_from_slice(batch_shape);
    if has_zero_dim(input.shape()) {
        return Ok(backend.with_raw(OP, |raw| {
            // The fast path still validates residency before allocating the
            // empty output: `raw.tensor` checks the tensor is resident.
            raw.tensor(input)?;
            raw.alloc_output::<<T as LinalgScalar>::Real>(&s_shape)
        })?);
    }
    let batch_total = batch_count(OP, batch_shape)?;
    let a_stride = checked_mul_usize(OP, "svd_values input stride", m, n)?;
    let s_stride = k;

    match select_svd_driver(m, n) {
        SvdDriver::Gesvdj => {
            let mut u_shape = vec![m, k];
            u_shape.extend_from_slice(batch_shape);
            let mut v_shape = vec![n, k];
            v_shape.extend_from_slice(batch_shape);
            let s = backend.with_raw(OP, |raw| {
                let handles = raw.resource(CudaLinalgHandles::load)?;
                // SAFETY: the stream handle is valid only for this raw-session
                // scope; it is used immediately to bind cuSOLVER and not retained.
                let stream = unsafe { raw.stream().raw_handle() } as usize as CudaStream;
                handles.cusolver().set_stream(stream, OP)?;

                // Clone `input` into a fresh work matrix on the session stream.
                let mut work = raw.alloc_output::<T>(input.shape())?;
                {
                    let src = raw.tensor(input)?;
                    let dst = raw.tensor_mut(&mut work)?;
                    // SAFETY: both spans were validated resident on this runtime
                    // and the copy is stream-ordered; `dst` is exclusively borrowed.
                    unsafe {
                        raw.copy_bytes(
                            dst.raw_ptr(),
                            src.raw_ptr() as *const _,
                            src.byte_len(),
                            OP,
                        )?
                    };
                }

                let mut u = raw.alloc_output::<T>(&u_shape)?;
                let mut v = raw.alloc_output::<T>(&v_shape)?;
                let mut s = raw.alloc_output::<<T as LinalgScalar>::Real>(&s_shape)?;

                let m_i32 = as_i32(m, OP, "m")?;
                let n_i32 = as_i32(n, OP, "n")?;
                let lda = as_i32(m, OP, "lda")?;
                let params = handles.cusolver().create_gesvdj_info(OP)?;
                let lwork = {
                    let a_ref = raw.tensor(&work)?;
                    let s_ref = raw.tensor(&s)?;
                    let u_ref = raw.tensor(&u)?;
                    let v_ref = raw.tensor(&v)?;
                    handles.cusolver().gesvdj_buffer_size(
                        T::DATA_TYPE,
                        CusolverEigMode::NoVector,
                        1,
                        m_i32,
                        n_i32,
                        // SAFETY: all spans are validated device allocations on
                        // this runtime; only leading dimensions are queried here.
                        unsafe { a_ref.raw_ptr().cast_const() },
                        lda,
                        // SAFETY: `s_ref` is a validated device span on this runtime; the pointer is used only within the raw-session scope.
                        unsafe { s_ref.raw_ptr().cast_const() },
                        // SAFETY: `u_ref` is a validated device span on this runtime; the pointer is used only within the raw-session scope.
                        unsafe { u_ref.raw_ptr().cast_const() },
                        lda,
                        // SAFETY: `v_ref` is a validated device span on this runtime; the pointer is used only within the raw-session scope.
                        unsafe { v_ref.raw_ptr().cast_const() },
                        n_i32,
                        &params,
                        OP,
                    )?
                };
                let workspace_nbytes = {
                    let lwork = usize::try_from(lwork).map_err(|_| {
                        Error::invalid_argument(
                            OP,
                            "workspace_length",
                            format!("must be non-negative, got {lwork}"),
                        )
                    })?;
                    lwork.checked_mul(std::mem::size_of::<T>()).ok_or_else(|| {
                        Error::invalid_argument(OP, "workspace_length", "byte size overflowed")
                    })?
                };
                let workspace = raw.alloc_bytes(workspace_nbytes, OP)?;
                let mut workspace_ptr = std::ptr::null_mut::<c_void>();
                workspace.with_ptr(|ptr| workspace_ptr = ptr);

                let mut info = raw.alloc_output::<i32>(&[batch_total])?;
                let a_ref = raw.tensor_mut(&mut work)?;
                // SAFETY: `a_ref` is a validated device span on this runtime; the pointer is used only within the raw-session scope.
                let a_ptr = unsafe { a_ref.raw_ptr() };
                let s_ref = raw.tensor_mut(&mut s)?;
                // SAFETY: `s_ref` is a validated device span on this runtime; the pointer is used only within the raw-session scope.
                let s_ptr = unsafe { s_ref.raw_ptr() };
                let u_ref = raw.tensor_mut(&mut u)?;
                // SAFETY: `u_ref` is a validated device span on this runtime; the pointer is used only within the raw-session scope.
                let u_ptr = unsafe { u_ref.raw_ptr() };
                let v_ref = raw.tensor_mut(&mut v)?;
                // SAFETY: `v_ref` is a validated device span on this runtime; the pointer is used only within the raw-session scope.
                let v_ptr = unsafe { v_ref.raw_ptr() };
                let info_ref = raw.tensor_mut(&mut info)?;
                // SAFETY: `info_ref` is a validated device span on this runtime; the pointer is used only within the raw-session scope.
                let info_ptr = unsafe { info_ref.raw_ptr() };
                let u_stride = checked_mul_usize(OP, "svd_values u stride", m, k)?;
                let v_stride = checked_mul_usize(OP, "svd_values v stride", n, k)?;

                for batch in 0..batch_total {
                    let a_offset =
                        checked_batch_offset(OP, "svd_values input batch offset", batch, a_stride)?;
                    let s_offset = checked_batch_offset(
                        OP,
                        "svd_values singular value batch offset",
                        batch,
                        s_stride,
                    )?;
                    let u_offset =
                        checked_batch_offset(OP, "svd_values u batch offset", batch, u_stride)?;
                    let v_offset =
                        checked_batch_offset(OP, "svd_values v batch offset", batch, v_stride)?;
                    // SAFETY: all offsets are checked against their per-batch
                    // strides and each base pointer belongs to a live device tensor.
                    let (batch_a, batch_s, batch_u, batch_v, batch_info) = unsafe {
                        (
                            batch_ptr::<T>(a_ptr, a_offset),
                            batch_ptr::<<T as LinalgScalar>::Real>(s_ptr, s_offset),
                            batch_ptr::<T>(u_ptr, u_offset),
                            batch_ptr::<T>(v_ptr, v_offset),
                            batch_ptr::<i32>(info_ptr, batch).cast::<i32>(),
                        )
                    };
                    // SAFETY: batch pointers, scratch U/V buffers, workspace, and
                    // params satisfy cuSOLVER gesvdj's no-vector SVD contract.
                    unsafe {
                        handles.cusolver().gesvdj(
                            T::DATA_TYPE,
                            CusolverEigMode::NoVector,
                            1,
                            m_i32,
                            n_i32,
                            batch_a,
                            lda,
                            batch_s,
                            batch_u,
                            lda,
                            batch_v,
                            n_i32,
                            workspace_ptr,
                            lwork,
                            batch_info,
                            &params,
                            OP,
                        )?;
                    }
                }

                // Host barrier (only for reading the solver diagnostics).
                let host_info = raw.download_tensor::<i32>(&info, OP)?;
                for &value in host_info.host_data()? {
                    check_solver_info(OP, "cusolverDn*gesvdj", value)?;
                }
                Ok(s)
            })?;
            Ok(s)
        }
        SvdDriver::Gesvd => {
            let (gesvd_m, gesvd_n) = if m < n { (n, m) } else { (m, n) };
            // When `m < n` gesvd factors `adjoint(input)` (n×m); the CubeCL
            // adjoint kernel is flushed before the raw session, so the SVD
            // below observes it.
            let transposed_work = if m < n {
                let mut work_shape = vec![n, m];
                work_shape.extend_from_slice(batch_shape);
                Some(T::copy_matrix_adjoint(backend, input, &work_shape, OP)?)
            } else {
                None
            };

            let s = backend.with_raw(OP, |raw| {
                let handles = raw.resource(CudaLinalgHandles::load)?;
                // SAFETY: the stream handle is valid only for this raw-session
                // scope; it is used immediately to bind cuSOLVER and not retained.
                let stream = unsafe { raw.stream().raw_handle() } as usize as CudaStream;
                handles.cusolver().set_stream(stream, OP)?;

                // `work` is either the CubeCL-adjointed input (m < n) or a
                // clone of `input` made on the session stream.
                let mut work = match transposed_work {
                    Some(work) => work,
                    None => {
                        let mut work = raw.alloc_output::<T>(input.shape())?;
                        {
                            let src = raw.tensor(input)?;
                            let dst = raw.tensor_mut(&mut work)?;
                            // SAFETY: both spans were validated resident on this
                            // runtime and the copy is stream-ordered; `dst` is
                            // exclusively borrowed.
                            unsafe {
                                raw.copy_bytes(
                                    dst.raw_ptr(),
                                    src.raw_ptr() as *const _,
                                    src.byte_len(),
                                    OP,
                                )?
                            };
                        }
                        work
                    }
                };
                let mut s = raw.alloc_output::<<T as LinalgScalar>::Real>(&s_shape)?;

                let a_ref = raw.tensor_mut(&mut work)?;
                // SAFETY: `a_ref` is a validated device span on this runtime; the pointer is used only within the raw-session scope.
                let a_ptr = unsafe { a_ref.raw_ptr() };
                let s_ref = raw.tensor_mut(&mut s)?;
                // SAFETY: `s_ref` is a validated device span on this runtime; the pointer is used only within the raw-session scope.
                let s_ptr = unsafe { s_ref.raw_ptr() };
                let gesvd_m_i32 = as_i32(gesvd_m, OP, "gesvd m")?;
                let gesvd_n_i32 = as_i32(gesvd_n, OP, "gesvd n")?;
                let lda = gesvd_m_i32;
                let lwork = handles.cusolver().gesvd_buffer_size(
                    T::DATA_TYPE,
                    gesvd_m_i32,
                    gesvd_n_i32,
                    OP,
                )?;
                let workspace_nbytes = {
                    let lwork = usize::try_from(lwork).map_err(|_| {
                        Error::invalid_argument(
                            OP,
                            "workspace_length",
                            format!("must be non-negative, got {lwork}"),
                        )
                    })?;
                    lwork.checked_mul(std::mem::size_of::<T>()).ok_or_else(|| {
                        Error::invalid_argument(OP, "workspace_length", "byte size overflowed")
                    })?
                };
                let workspace = raw.alloc_bytes(workspace_nbytes, OP)?;
                let mut workspace_ptr = std::ptr::null_mut::<c_void>();
                workspace.with_ptr(|ptr| workspace_ptr = ptr);
                let mut rwork_ptr = std::ptr::null_mut::<c_void>();
                if T::NEEDS_RWORK {
                    let rwork_len = as_i32(
                        checked_mul_usize(OP, "svd_values rwork length", 5, k)?,
                        OP,
                        "rwork",
                    )?;
                    let rwork_nbytes = usize::try_from(rwork_len)
                        .map_err(|_| {
                            Error::invalid_argument(OP, "rwork_length", "must be non-negative")
                        })?
                        .checked_mul(std::mem::size_of::<<T as LinalgScalar>::Real>())
                        .ok_or_else(|| {
                            Error::invalid_argument(OP, "rwork_length", "byte size overflowed")
                        })?;
                    let rwork = raw.alloc_bytes(rwork_nbytes, OP)?;
                    rwork.with_ptr(|ptr| rwork_ptr = ptr);
                }
                let mut info = raw.alloc_output::<i32>(&[batch_total])?;
                let info_ref = raw.tensor_mut(&mut info)?;
                // SAFETY: `info_ref` is a validated device span on this runtime; the pointer is used only within the raw-session scope.
                let info_ptr = unsafe { info_ref.raw_ptr() };
                let job = b'N' as c_char;

                for batch in 0..batch_total {
                    let a_offset =
                        checked_batch_offset(OP, "svd_values input batch offset", batch, a_stride)?;
                    let s_offset = checked_batch_offset(
                        OP,
                        "svd_values singular value batch offset",
                        batch,
                        s_stride,
                    )?;
                    // SAFETY: checked offsets keep the input, singular-value,
                    // and info pointers inside live device allocations for this batch.
                    let (batch_a, batch_s, batch_info) = unsafe {
                        (
                            batch_ptr::<T>(a_ptr, a_offset),
                            batch_ptr::<<T as LinalgScalar>::Real>(s_ptr, s_offset),
                            batch_ptr::<i32>(info_ptr, batch).cast::<i32>(),
                        )
                    };
                    // SAFETY: no-vector gesvd permits null U/VT pointers with
                    // unit leading dimensions; other pointers and workspace are validated.
                    unsafe {
                        handles.cusolver().gesvd(
                            T::DATA_TYPE,
                            job,
                            job,
                            gesvd_m_i32,
                            gesvd_n_i32,
                            batch_a,
                            lda,
                            batch_s,
                            std::ptr::null_mut(),
                            1,
                            std::ptr::null_mut(),
                            1,
                            workspace_ptr,
                            lwork,
                            rwork_ptr,
                            batch_info,
                            OP,
                        )?;
                    }
                }

                // Host barrier (only for reading the solver diagnostics).
                let host_info = raw.download_tensor::<i32>(&info, OP)?;
                for &value in host_info.host_data()? {
                    check_solver_info(OP, "cusolverDn*gesvd", value)?;
                }
                Ok(s)
            })?;
            Ok(s)
        }
    }
}

mod householder_qr;
mod rank_revealing_qr;
use householder_qr::*;

fn qr_typed<T>(
    backend: &mut CudaExecSession<'_>,
    input: &TypedTensor<T>,
) -> Result<(TypedTensor<T>, TypedTensor<T>)>
where
    T: LinalgScalar + TensorScalar,
{
    const OP: &str = "qr";

    ensure_cubecl_resident_typed(OP, input)?;
    let (m, n) = matrix_dims(OP, input.shape())?;
    let k = m.min(n);
    let batch_shape = &input.shape()[2..];
    let mut q_shape = vec![m, k];
    q_shape.extend_from_slice(batch_shape);
    let mut r_shape = vec![k, n];
    r_shape.extend_from_slice(batch_shape);
    if has_zero_dim(input.shape()) {
        return Ok(backend.with_raw(OP, |raw| {
            // The fast path still validates residency before allocating the
            // empty outputs: `raw.tensor` checks the tensor is resident.
            raw.tensor(input)?;
            Ok((
                raw.alloc_output::<T>(&q_shape)?,
                raw.alloc_output::<T>(&r_shape)?,
            ))
        })?);
    }

    let m_i32 = as_i32(m, OP, "m")?;
    let n_i32 = as_i32(n, OP, "n")?;
    let k_i32 = as_i32(k, OP, "k")?;
    let lda = as_i32(m, OP, "lda")?;
    let batch_total = batch_count(OP, batch_shape)?;
    let work_stride = checked_mul_usize(OP, "qr work stride", m, n)?;
    let q_stride = checked_mul_usize(OP, "qr q stride", m, k)?;
    let tau_bytes = checked_mul_usize(OP, "qr tau workspace bytes", k, std::mem::size_of::<T>())?;

    let (q, work) = backend.with_raw(OP, |raw| {
        let handles = raw.resource(CudaLinalgHandles::load)?;
        // SAFETY: the stream handle is valid only for this raw-session scope;
        // it is used immediately to bind the cuSOLVER handle and not retained.
        let stream = unsafe { raw.stream().raw_handle() } as usize as CudaStream;
        handles.cusolver().set_stream(stream, OP)?;

        // Clone `input` into a fresh work matrix on the session stream.
        let mut work = raw.alloc_output::<T>(input.shape())?;
        {
            let src = raw.tensor(input)?;
            let dst = raw.tensor_mut(&mut work)?;
            // SAFETY: both spans were validated resident on this runtime and
            // the copy is stream-ordered; `dst` is exclusively borrowed.
            unsafe {
                raw.copy_bytes(dst.raw_ptr(), src.raw_ptr() as *const _, src.byte_len(), OP)?
            };
        }
        let mut q = raw.alloc_output::<T>(&q_shape)?;

        let geqrf_lwork = {
            let work_ref = raw.tensor(&work)?;
            handles.cusolver().geqrf_buffer_size(
                T::DATA_TYPE,
                m_i32,
                n_i32,
                // SAFETY: `work_ref` is a validated device span on this
                // runtime; cuSOLVER only queries the leading dimension here.
                unsafe { work_ref.raw_ptr() },
                lda,
                OP,
            )?
        };
        let geqrf_workspace_nbytes = {
            let lwork = usize::try_from(geqrf_lwork).map_err(|_| {
                Error::invalid_argument(
                    OP,
                    "workspace_length",
                    format!("must be non-negative, got {geqrf_lwork}"),
                )
            })?;
            lwork.checked_mul(std::mem::size_of::<T>()).ok_or_else(|| {
                Error::invalid_argument(OP, "workspace_length", "byte size overflowed")
            })?
        };
        let geqrf_workspace = raw.alloc_bytes(geqrf_workspace_nbytes, OP)?;
        let mut geqrf_workspace_ptr = std::ptr::null_mut::<c_void>();
        geqrf_workspace.with_ptr(|ptr| geqrf_workspace_ptr = ptr);

        let tau = raw.alloc_bytes(tau_bytes, OP)?;
        let mut tau_ptr = std::ptr::null_mut::<c_void>();
        tau.with_ptr(|ptr| tau_ptr = ptr);

        let orgqr_lwork = {
            let q_ref = raw.tensor(&q)?;
            handles.cusolver().orgqr_buffer_size(
                T::DATA_TYPE,
                m_i32,
                k_i32,
                k_i32,
                // SAFETY: `q_ref` and `tau_ptr` are validated device spans on
                // this runtime; only the leading dimensions are queried here.
                unsafe { q_ref.raw_ptr().cast_const() },
                lda,
                tau_ptr.cast_const(),
                OP,
            )?
        };
        let orgqr_workspace_nbytes = {
            let lwork = usize::try_from(orgqr_lwork).map_err(|_| {
                Error::invalid_argument(
                    OP,
                    "workspace_length",
                    format!("must be non-negative, got {orgqr_lwork}"),
                )
            })?;
            lwork.checked_mul(std::mem::size_of::<T>()).ok_or_else(|| {
                Error::invalid_argument(OP, "workspace_length", "byte size overflowed")
            })?
        };
        let orgqr_workspace = raw.alloc_bytes(orgqr_workspace_nbytes, OP)?;
        let mut orgqr_workspace_ptr = std::ptr::null_mut::<c_void>();
        orgqr_workspace.with_ptr(|ptr| orgqr_workspace_ptr = ptr);

        let mut geqrf_info = raw.alloc_output::<i32>(&[batch_total])?;
        let mut orgqr_info = raw.alloc_output::<i32>(&[batch_total])?;
        let work_ref = raw.tensor_mut(&mut work)?;
        // SAFETY: `work_ref` is a validated device span on this runtime; the pointer is used only within the raw-session scope.
        let work_ptr = unsafe { work_ref.raw_ptr() };
        let q_ref = raw.tensor_mut(&mut q)?;
        // SAFETY: `q_ref` is a validated device span on this runtime; the pointer is used only within the raw-session scope.
        let q_ptr = unsafe { q_ref.raw_ptr() };
        let geqrf_info_ref = raw.tensor_mut(&mut geqrf_info)?;
        // SAFETY: `geqrf_info_ref` is a validated device span on this runtime; the pointer is used only within the raw-session scope.
        let geqrf_info_ptr = unsafe { geqrf_info_ref.raw_ptr() };
        let orgqr_info_ref = raw.tensor_mut(&mut orgqr_info)?;
        // SAFETY: `orgqr_info_ref` is a validated device span on this runtime; the pointer is used only within the raw-session scope.
        let orgqr_info_ptr = unsafe { orgqr_info_ref.raw_ptr() };

        for batch in 0..batch_total {
            let work_offset = checked_batch_offset(OP, "qr work batch offset", batch, work_stride)?;
            let q_offset = checked_batch_offset(OP, "qr q batch offset", batch, q_stride)?;
            // SAFETY: checked offsets keep work, Q, and info pointers inside
            // their live device allocations for this batch.
            let (batch_work, batch_q, batch_geqrf_info, batch_orgqr_info) = unsafe {
                (
                    batch_ptr::<T>(work_ptr, work_offset),
                    batch_ptr::<T>(q_ptr, q_offset),
                    batch_ptr::<i32>(geqrf_info_ptr, batch).cast::<i32>(),
                    batch_ptr::<i32>(orgqr_info_ptr, batch).cast::<i32>(),
                )
            };
            // SAFETY: batch pointers, tau/workspace buffers, dimensions, and
            // stream-bound handle satisfy cuSOLVER geqrf's QR factorization contract.
            unsafe {
                handles.cusolver().geqrf(
                    T::DATA_TYPE,
                    m_i32,
                    n_i32,
                    batch_work,
                    lda,
                    tau_ptr,
                    geqrf_workspace_ptr,
                    geqrf_lwork,
                    batch_geqrf_info,
                    OP,
                )?;
            }

            // Copy the reflectors from `work` into `q` on the session stream.
            // SAFETY: `batch_q` and `batch_work` are validated device spans on
            // this runtime, the copy is stream-ordered, and `batch_q` is written
            // exclusively before orgqr reads it back.
            unsafe {
                raw.copy_bytes(
                    batch_q,
                    batch_work.cast_const(),
                    q_stride * std::mem::size_of::<T>(),
                    OP,
                )?;
            }
            // SAFETY: `batch_q` contains the copied reflectors, `tau` and
            // workspace are live, and dimensions match the validated reduced-Q shape.
            unsafe {
                handles.cusolver().orgqr(
                    T::DATA_TYPE,
                    m_i32,
                    k_i32,
                    k_i32,
                    batch_q,
                    lda,
                    tau_ptr.cast_const(),
                    orgqr_workspace_ptr,
                    orgqr_lwork,
                    batch_orgqr_info,
                    OP,
                )?;
            }
        }

        // Host barrier (only for reading the solver diagnostics).
        let host_geqrf = raw.download_tensor::<i32>(&geqrf_info, OP)?;
        for &value in host_geqrf.host_data()? {
            check_solver_info(OP, "cusolverDn*geqrf", value)?;
        }
        let host_orgqr = raw.download_tensor::<i32>(&orgqr_info, OP)?;
        for &value in host_orgqr.host_data()? {
            check_solver_info(OP, "cusolverDn*orgqr", value)?;
        }
        Ok((q, work))
    })?;

    let r_input = backend.slice_typed(&work, &matrix_slice_config(input.shape(), k, n))?;
    let r = backend.triu_typed(&r_input, 0)?;
    Ok((q, r))
}

fn eigh_typed<T>(
    backend: &mut CudaExecSession<'_>,
    input: &TypedTensor<T>,
) -> Result<(TypedTensor<<T as LinalgScalar>::Real>, TypedTensor<T>)>
where
    T: LinalgScalar + TensorScalar,
{
    const OP: &str = "eigh";

    ensure_cubecl_resident_typed(OP, input)?;
    let n = square_matrix_dim(OP, input.shape())?;
    let batch_shape = &input.shape()[2..];
    let mut values_shape = vec![n];
    values_shape.extend_from_slice(batch_shape);
    let n_i32 = as_i32(n, OP, "n")?;
    let lda = as_i32(n, OP, "lda")?;
    let batch_total = batch_count(OP, batch_shape)?;
    let matrix_stride = checked_mul_usize(OP, "eigh matrix stride", n, n)?;
    let values_stride = n;
    if has_zero_dim(input.shape()) {
        return Ok(backend.with_raw(OP, |raw| {
            // The fast path still validates residency before allocating the
            // empty outputs: `raw.tensor` checks the tensor is resident.
            raw.tensor(input)?;
            Ok((
                raw.alloc_output::<<T as LinalgScalar>::Real>(&values_shape)?,
                raw.alloc_output::<T>(input.shape())?,
            ))
        })?);
    }

    let (values, work) = backend.with_raw(OP, |raw| {
        let handles = raw.resource(CudaLinalgHandles::load)?;
        // SAFETY: the stream handle is valid only for this raw-session scope;
        // it is used immediately to bind the cuSOLVER handle and not retained.
        let stream = unsafe { raw.stream().raw_handle() } as usize as CudaStream;
        handles.cusolver().set_stream(stream, OP)?;

        // Clone `input` into a fresh work matrix on the session stream.
        let mut work = raw.alloc_output::<T>(input.shape())?;
        {
            let src = raw.tensor(input)?;
            let dst = raw.tensor_mut(&mut work)?;
            // SAFETY: both spans were validated resident on this runtime and
            // the copy is stream-ordered; `dst` is exclusively borrowed.
            unsafe {
                raw.copy_bytes(dst.raw_ptr(), src.raw_ptr() as *const _, src.byte_len(), OP)?
            };
        }
        let mut values = raw.alloc_output::<<T as LinalgScalar>::Real>(&values_shape)?;

        let lwork = {
            let a_ref = raw.tensor(&work)?;
            let values_ref = raw.tensor(&values)?;
            handles.cusolver().syevd_buffer_size(
                T::DATA_TYPE,
                CusolverEigMode::Vector,
                CublasFillMode::Lower,
                n_i32,
                // SAFETY: both spans are validated device allocations on this
                // runtime; only the leading dimensions are queried here.
                unsafe { a_ref.raw_ptr().cast_const() },
                lda,
                // SAFETY: `values_ref` is a validated device span on this runtime; the pointer is used only within the raw-session scope.
                unsafe { values_ref.raw_ptr().cast_const() },
                OP,
            )?
        };
        let workspace_nbytes = {
            let lwork = usize::try_from(lwork).map_err(|_| {
                Error::invalid_argument(
                    OP,
                    "workspace_length",
                    format!("must be non-negative, got {lwork}"),
                )
            })?;
            lwork.checked_mul(std::mem::size_of::<T>()).ok_or_else(|| {
                Error::invalid_argument(OP, "workspace_length", "byte size overflowed")
            })?
        };
        let workspace = raw.alloc_bytes(workspace_nbytes, OP)?;
        let mut workspace_ptr = std::ptr::null_mut::<c_void>();
        workspace.with_ptr(|ptr| workspace_ptr = ptr);

        let mut info = raw.alloc_output::<i32>(&[batch_total])?;
        let a_ref = raw.tensor_mut(&mut work)?;
        // SAFETY: `a_ref` is a validated device span on this runtime; the pointer is used only within the raw-session scope.
        let a_ptr = unsafe { a_ref.raw_ptr() };
        let values_ref = raw.tensor_mut(&mut values)?;
        // SAFETY: `values_ref` is a validated device span on this runtime; the pointer is used only within the raw-session scope.
        let values_ptr = unsafe { values_ref.raw_ptr() };
        let info_ref = raw.tensor_mut(&mut info)?;
        // SAFETY: `info_ref` is a validated device span on this runtime; the pointer is used only within the raw-session scope.
        let info_ptr = unsafe { info_ref.raw_ptr() };

        for batch in 0..batch_total {
            let a_offset =
                checked_batch_offset(OP, "eigh matrix batch offset", batch, matrix_stride)?;
            let values_offset =
                checked_batch_offset(OP, "eigh values batch offset", batch, values_stride)?;
            // SAFETY: checked offsets keep matrix, eigenvalue, and info pointers
            // inside live device allocations for this batch.
            let (batch_a, batch_w, batch_info) = unsafe {
                (
                    batch_ptr::<T>(a_ptr, a_offset),
                    batch_ptr::<<T as LinalgScalar>::Real>(values_ptr, values_offset),
                    batch_ptr::<i32>(info_ptr, batch).cast::<i32>(),
                )
            };
            // SAFETY: batch pointers, workspace, dimensions, and stream-bound
            // handle satisfy cuSOLVER syevd's vector eigensolver contract.
            unsafe {
                handles.cusolver().syevd(
                    T::DATA_TYPE,
                    CusolverEigMode::Vector,
                    CublasFillMode::Lower,
                    n_i32,
                    batch_a,
                    lda,
                    batch_w,
                    workspace_ptr,
                    lwork,
                    batch_info,
                    OP,
                )?;
            }
        }

        // Host barrier (only for reading the solver diagnostics).
        let host_info = raw.download_tensor::<i32>(&info, OP)?;
        for &value in host_info.host_data()? {
            check_solver_info(OP, "cusolverDn*syevd", value)?;
        }
        Ok((values, work))
    })?;

    Ok((values, work))
}

fn eigh_values_typed<T>(
    backend: &mut CudaExecSession<'_>,
    input: &TypedTensor<T>,
) -> Result<TypedTensor<<T as LinalgScalar>::Real>>
where
    T: LinalgScalar + TensorScalar,
{
    const OP: &str = "eigh_values";

    ensure_cubecl_resident_typed(OP, input)?;
    let n = square_matrix_dim(OP, input.shape())?;
    let batch_shape = &input.shape()[2..];
    let mut values_shape = vec![n];
    values_shape.extend_from_slice(batch_shape);
    let n_i32 = as_i32(n, OP, "n")?;
    let lda = as_i32(n, OP, "lda")?;
    let batch_total = batch_count(OP, batch_shape)?;
    let matrix_stride = checked_mul_usize(OP, "eigh_values matrix stride", n, n)?;
    let values_stride = n;
    if has_zero_dim(input.shape()) {
        return Ok(backend.with_raw(OP, |raw| {
            // The fast path still validates residency before allocating the
            // empty output: `raw.tensor` checks the tensor is resident.
            raw.tensor(input)?;
            raw.alloc_output::<<T as LinalgScalar>::Real>(&values_shape)
        })?);
    }

    let values = backend.with_raw(OP, |raw| {
        let handles = raw.resource(CudaLinalgHandles::load)?;
        // SAFETY: the stream handle is valid only for this raw-session scope;
        // it is used immediately to bind the cuSOLVER handle and not retained.
        let stream = unsafe { raw.stream().raw_handle() } as usize as CudaStream;
        handles.cusolver().set_stream(stream, OP)?;

        // Clone `input` into a fresh work matrix on the session stream.
        let mut work = raw.alloc_output::<T>(input.shape())?;
        {
            let src = raw.tensor(input)?;
            let dst = raw.tensor_mut(&mut work)?;
            // SAFETY: both spans were validated resident on this runtime and
            // the copy is stream-ordered; `dst` is exclusively borrowed.
            unsafe {
                raw.copy_bytes(dst.raw_ptr(), src.raw_ptr() as *const _, src.byte_len(), OP)?
            };
        }
        let mut values = raw.alloc_output::<<T as LinalgScalar>::Real>(&values_shape)?;

        let lwork = {
            let a_ref = raw.tensor(&work)?;
            let values_ref = raw.tensor(&values)?;
            handles.cusolver().syevd_buffer_size(
                T::DATA_TYPE,
                CusolverEigMode::NoVector,
                CublasFillMode::Lower,
                n_i32,
                // SAFETY: both spans are validated device allocations on this
                // runtime; only the leading dimensions are queried here.
                unsafe { a_ref.raw_ptr().cast_const() },
                lda,
                // SAFETY: `values_ref` is a validated device span on this runtime; the pointer is used only within the raw-session scope.
                unsafe { values_ref.raw_ptr().cast_const() },
                OP,
            )?
        };
        let workspace_nbytes = {
            let lwork = usize::try_from(lwork).map_err(|_| {
                Error::invalid_argument(
                    OP,
                    "workspace_length",
                    format!("must be non-negative, got {lwork}"),
                )
            })?;
            lwork.checked_mul(std::mem::size_of::<T>()).ok_or_else(|| {
                Error::invalid_argument(OP, "workspace_length", "byte size overflowed")
            })?
        };
        let workspace = raw.alloc_bytes(workspace_nbytes, OP)?;
        let mut workspace_ptr = std::ptr::null_mut::<c_void>();
        workspace.with_ptr(|ptr| workspace_ptr = ptr);

        let mut info = raw.alloc_output::<i32>(&[batch_total])?;
        let a_ref = raw.tensor_mut(&mut work)?;
        // SAFETY: `a_ref` is a validated device span on this runtime; the pointer is used only within the raw-session scope.
        let a_ptr = unsafe { a_ref.raw_ptr() };
        let values_ref = raw.tensor_mut(&mut values)?;
        // SAFETY: `values_ref` is a validated device span on this runtime; the pointer is used only within the raw-session scope.
        let values_ptr = unsafe { values_ref.raw_ptr() };
        let info_ref = raw.tensor_mut(&mut info)?;
        // SAFETY: `info_ref` is a validated device span on this runtime; the pointer is used only within the raw-session scope.
        let info_ptr = unsafe { info_ref.raw_ptr() };

        for batch in 0..batch_total {
            let a_offset =
                checked_batch_offset(OP, "eigh_values matrix batch offset", batch, matrix_stride)?;
            let values_offset =
                checked_batch_offset(OP, "eigh_values values batch offset", batch, values_stride)?;
            // SAFETY: checked offsets keep matrix, eigenvalue, and info pointers
            // inside live device allocations for this batch.
            let (batch_a, batch_w, batch_info) = unsafe {
                (
                    batch_ptr::<T>(a_ptr, a_offset),
                    batch_ptr::<<T as LinalgScalar>::Real>(values_ptr, values_offset),
                    batch_ptr::<i32>(info_ptr, batch).cast::<i32>(),
                )
            };
            // SAFETY: batch pointers, workspace, dimensions, and stream-bound
            // handle satisfy cuSOLVER syevd's no-vector eigensolver contract.
            unsafe {
                handles.cusolver().syevd(
                    T::DATA_TYPE,
                    CusolverEigMode::NoVector,
                    CublasFillMode::Lower,
                    n_i32,
                    batch_a,
                    lda,
                    batch_w,
                    workspace_ptr,
                    lwork,
                    batch_info,
                    OP,
                )?;
            }
        }

        // Host barrier (only for reading the solver diagnostics).
        let host_info = raw.download_tensor::<i32>(&info, OP)?;
        for &value in host_info.host_data()? {
            check_solver_info(OP, "cusolverDn*syevd", value)?;
        }
        Ok(values)
    })?;

    Ok(values)
}

fn build_lu_outputs_device<T>(
    backend: &mut CudaExecSession<'_>,
    lu: &TypedTensor<T>,
    pivots: &TypedTensor<i32>,
    m: usize,
    n: usize,
    batch_shape: &[usize],
) -> Result<(
    TypedTensor<T>,
    TypedTensor<T>,
    TypedTensor<T>,
    TypedTensor<T>,
)>
where
    T: LinalgScalar + TensorScalar,
{
    let k = m.min(n);
    let mut p_shape = vec![m, m];
    p_shape.extend_from_slice(batch_shape);
    let mut l_shape = vec![m, k];
    l_shape.extend_from_slice(batch_shape);
    let mut u_shape = vec![k, n];
    u_shape.extend_from_slice(batch_shape);
    let parity_shape = batch_shape.to_vec();

    backend.with_cubecl("lu", |cubecl| {
        let p = cubecl.alloc_output::<T>(&p_shape)?;
        let l = cubecl.alloc_output::<T>(&l_shape)?;
        let u = cubecl.alloc_output::<T>(&u_shape)?;
        let parity = cubecl.alloc_output::<T>(&parity_shape)?;
        let launch_len = p
            .n_elements()
            .max(l.n_elements())
            .max(u.n_elements())
            .max(parity.n_elements());
        let p_arg = cubecl.tensor_binding(&p, "lu")?;
        let l_arg = cubecl.tensor_binding(&l, "lu")?;
        let u_arg = cubecl.tensor_binding(&u, "lu")?;
        let parity_arg = cubecl.tensor_binding(&parity, "lu")?;
        let work_arg = cubecl.tensor_binding(lu, "lu")?;
        let pivots_arg = cubecl.array_arg(pivots, "lu")?;
        let launch_count = cubecl.cube_count_1d(launch_len)?;
        // SAFETY: tensor bindings describe live CUDA tensors, and `launch_count`
        // covers the maximum P/L/U/parity output domain consumed by the kernel.
        unsafe {
            cubecl_linalg::lu_extract_outputs::launch_unchecked::<T, CubeclCudaRuntime>(
                cubecl.client(),
                launch_count,
                cubecl.cube_dim_1d(),
                p_arg.into_tensor_arg(),
                l_arg.into_tensor_arg(),
                u_arg.into_tensor_arg(),
                parity_arg.into_tensor_arg(),
                work_arg.into_tensor_arg(),
                pivots_arg,
                k,
                lu.shape().len(),
            );
        }
        Ok((p, l, u, parity))
    })
}

fn build_lu_parity_device<T>(
    backend: &mut CudaExecSession<'_>,
    pivots: &TypedTensor<i32>,
    k: usize,
    batch_shape: &[usize],
) -> Result<TypedTensor<T>>
where
    T: LinalgScalar + TensorScalar,
{
    backend.with_cubecl("lu_factor", |cubecl| {
        let parity = cubecl.alloc_output::<T>(batch_shape)?;
        let parity_arg = cubecl.tensor_binding(&parity, "lu_factor")?;
        let pivots_arg = cubecl.array_arg(pivots, "lu_factor")?;
        let launch_count = cubecl.cube_count_1d(parity.n_elements())?;
        // SAFETY: tensor bindings describe live CUDA tensors, and `launch_count`
        // covers exactly the parity output domain.
        unsafe {
            cubecl_linalg::lu_parity::launch_unchecked::<T, CubeclCudaRuntime>(
                cubecl.client(),
                launch_count,
                cubecl.cube_dim_1d(),
                parity_arg.into_tensor_arg(),
                pivots_arg,
                k,
            );
        }
        Ok(parity)
    })
}

fn fill_one_device_tensor<T>(
    backend: &mut CudaExecSession<'_>,
    shape: &[usize],
    op: &'static str,
) -> Result<TypedTensor<T>>
where
    T: LinalgScalar + TensorScalar,
{
    backend.with_cubecl(op, |cubecl| {
        let out = cubecl.alloc_output::<T>(shape)?;
        let out_arg = cubecl.tensor_binding(&out, op)?;
        let launch_count = cubecl.cube_count_1d(out.n_elements())?;
        // SAFETY: `out_arg` describes a live CUDA output tensor, and
        // `launch_count` covers every output element exactly once.
        unsafe {
            cubecl_linalg::fill_one_kernel::launch_unchecked::<T, CubeclCudaRuntime>(
                cubecl.client(),
                launch_count,
                cubecl.cube_dim_1d(),
                out_arg.into_tensor_arg(),
            );
        }
        Ok(out)
    })
}

fn apply_lu_pivots_typed<T>(
    backend: &mut CudaExecSession<'_>,
    input: &TypedTensor<T>,
    pivots: &TypedTensor<i32>,
    inverse: bool,
) -> Result<TypedTensor<T>>
where
    T: LinalgScalar + TensorScalar,
{
    backend.with_cubecl("lu_solve_prepared", |cubecl| {
        let out = cubecl.alloc_output::<T>(input.shape())?;
        if out.n_elements() == 0 {
            return Ok(out);
        }
        let k = pivots.shape()[0];
        let out_arg = cubecl.tensor_binding(&out, "lu_solve_prepared")?;
        let input_arg = cubecl.tensor_binding(input, "lu_solve_prepared")?;
        let pivots_arg = cubecl.array_arg(pivots, "lu_solve_prepared")?;
        let launch_count = cubecl.cube_count_1d(out.n_elements())?;
        // SAFETY: tensor bindings describe live CUDA tensors, pivot metadata
        // matches the validated LU shape, and `launch_count` covers the output domain.
        unsafe {
            cubecl_linalg::lu_apply_pivots::launch_unchecked::<T, CubeclCudaRuntime>(
                cubecl.client(),
                launch_count,
                cubecl.cube_dim_1d(),
                out_arg.into_tensor_arg(),
                input_arg.into_tensor_arg(),
                pivots_arg,
                k,
                input.shape().len(),
                inverse,
            );
        }
        Ok(out)
    })
}

fn zero_sized_lu_factor_outputs<T>(
    backend: &mut CudaExecSession<'_>,
    shape: &[usize],
) -> Result<(TypedTensor<T>, TypedTensor<i32>, TypedTensor<T>)>
where
    T: LinalgScalar + TensorScalar,
{
    let m = shape[0];
    let n = shape[1];
    let k = m.min(n);
    let batch_shape = &shape[2..];
    let mut pivot_shape = vec![k];
    pivot_shape.extend_from_slice(batch_shape);
    let parity_shape = batch_shape.to_vec();
    let parity = fill_one_device_tensor(backend, &parity_shape, "lu_factor")?;
    let lu = backend.with_cubecl("lu_factor", |cubecl| cubecl.alloc_output::<T>(shape))?;
    let pivots = backend.with_cubecl("lu_factor", |cubecl| {
        cubecl.alloc_output::<i32>(&pivot_shape)
    })?;
    Ok((lu, pivots, parity))
}

fn matrix_slice_config(shape: &[usize], row_limit: usize, col_limit: usize) -> SliceConfig {
    let mut limits = shape.to_vec();
    limits[0] = row_limit;
    limits[1] = col_limit;
    SliceConfig {
        starts: vec![0; shape.len()],
        limits,
        strides: vec![1; shape.len()],
    }
}

fn matrix_dims(op: &'static str, shape: &[usize]) -> Result<(usize, usize)> {
    if shape.len() < 2 {
        return Err(Error::rank_mismatch(op, 2, shape.len()));
    }
    Ok((shape[0], shape[1]))
}

fn square_matrix_dim(op: &'static str, shape: &[usize]) -> Result<usize> {
    let (rows, cols) = matrix_dims(op, shape)?;
    if rows != cols {
        return Err(Error::shape_mismatch(op, [rows], [cols]));
    }
    Ok(rows)
}

fn validate_triangular_rhs(
    op: &'static str,
    a_shape: &[usize],
    b_shape: &[usize],
    left_side: bool,
) -> Result<()> {
    let n = square_matrix_dim(op, a_shape)?;
    let (b_rows, b_cols) = matrix_dims(op, b_shape)?;
    if a_shape[2..] != b_shape[2..] {
        return Err(Error::shape_mismatch(
            op,
            a_shape.to_vec(),
            b_shape.to_vec(),
        ));
    }
    if left_side && b_rows != n {
        return Err(Error::shape_mismatch(op, [n], [b_rows]));
    }
    if !left_side && b_cols != n {
        return Err(Error::shape_mismatch(op, [n], [b_cols]));
    }
    Ok(())
}

fn validate_lu_solve_prepared_shapes(
    lu_shape: &[usize],
    pivots_shape: &[usize],
    b_shape: &[usize],
) -> Result<()> {
    let n = square_matrix_dim("lu_solve_prepared", lu_shape)?;
    let (b_rows, _) = matrix_dims("lu_solve_prepared", b_shape)?;
    if b_rows != n {
        return Err(Error::shape_mismatch("lu_solve_prepared", [n], [b_rows]));
    }
    if lu_shape[2..] != b_shape[2..] {
        return Err(Error::shape_mismatch(
            "lu_solve_prepared",
            lu_shape.to_vec(),
            b_shape.to_vec(),
        ));
    }
    let mut expected_pivots = vec![n];
    expected_pivots.extend_from_slice(&lu_shape[2..]);
    if pivots_shape != expected_pivots {
        return Err(Error::shape_mismatch(
            "lu_solve_prepared",
            expected_pivots,
            pivots_shape.to_vec(),
        ));
    }
    Ok(())
}

fn check_solver_info(op: &'static str, call: &'static str, info: i32) -> Result<()> {
    if info == 0 {
        return Ok(());
    }
    if info < 0 {
        return Err(Error::invalid_argument(
            op,
            "cusolver_parameter",
            format!("{call} reported invalid parameter {}", -info),
        ));
    }
    Err(crate::error::into_tensor_error(
        op,
        crate::Error::NonConvergence { op },
    ))
}

fn has_zero_dim(shape: &[usize]) -> bool {
    shape.contains(&0)
}

fn checked_shape_product(op: &'static str, label: &'static str, shape: &[usize]) -> Result<usize> {
    shape.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim).ok_or_else(|| {
            let _ = label;
            Error::validation(op, ValidationError::IntegerOverflow)
        })
    })
}

fn batch_count(op: &'static str, batch_shape: &[usize]) -> Result<usize> {
    let count = checked_shape_product(op, "batch shape", batch_shape)?;
    Ok(count.max(1))
}

fn checked_mul_usize(
    op: &'static str,
    label: &'static str,
    lhs: usize,
    rhs: usize,
) -> Result<usize> {
    lhs.checked_mul(rhs).ok_or_else(|| {
        let _ = label;
        Error::validation(op, ValidationError::IntegerOverflow)
    })
}

fn checked_batch_offset(
    op: &'static str,
    label: &'static str,
    batch: usize,
    stride: usize,
) -> Result<usize> {
    checked_mul_usize(op, label, batch, stride)
}

fn as_i32(value: usize, op: &'static str, label: &'static str) -> Result<i32> {
    i32::try_from(value)
        .map_err(|_| Error::invalid_argument(op, label, format!("{value} does not fit in i32")))
}

/// Offset a mutable device pointer by `offset` typed elements.
///
/// # Safety
///
/// `base` must point to a device allocation containing at least `offset`
/// additional `T` elements, and the resulting pointer must remain within the
/// same allocation. The caller is responsible for ensuring any later FFI call
/// observes CUDA aliasing and mutability requirements.
unsafe fn batch_ptr<T>(base: *mut c_void, offset: usize) -> *mut c_void {
    base.cast::<T>().add(offset).cast()
}

/// Offset an immutable device pointer by `offset` typed elements.
///
/// # Safety
///
/// `base` must point to a device allocation containing at least `offset`
/// additional `T` elements, and the resulting pointer must remain within the
/// same allocation for the lifetime required by the receiving CUDA library.
unsafe fn batch_const_ptr<T>(base: *const c_void, offset: usize) -> *const c_void {
    base.cast::<T>().add(offset).cast()
}

fn zero_like_linalg_device_tensor(
    backend: &mut CudaExecSession<'_>,
    input: &Tensor,
    op: &'static str,
) -> Result<Tensor> {
    match input {
        Tensor::F32(t) => {
            Ok(Tensor::F32(backend.with_cubecl(op, |cubecl| {
                cubecl.alloc_output::<f32>(t.shape())
            })?))
        }
        Tensor::F64(t) => {
            Ok(Tensor::F64(backend.with_cubecl(op, |cubecl| {
                cubecl.alloc_output::<f64>(t.shape())
            })?))
        }
        Tensor::C32(t) => Ok(Tensor::C32(backend.with_cubecl(op, |cubecl| {
            cubecl.alloc_output::<num_complex::Complex32>(t.shape())
        })?)),
        Tensor::C64(t) => Ok(Tensor::C64(backend.with_cubecl(op, |cubecl| {
            cubecl.alloc_output::<num_complex::Complex64>(t.shape())
        })?)),
        Tensor::I32(_) | Tensor::I64(_) | Tensor::Bool(_) => {
            Err(unsupported_linalg_dtype(op, input))
        }
    }
}

fn batched_vector_rhs_shape(a: &Tensor, b: &Tensor) -> Option<Vec<usize>> {
    if b.shape().len() == 1 {
        return Some(vec![b.shape()[0], 1]);
    }

    let is_batched_vector_rhs = a.shape().len() == b.shape().len() + 1
        && !b.shape().is_empty()
        && b.shape()[0] == a.shape()[0]
        && b.shape()[1..] == a.shape()[2..];
    if !is_batched_vector_rhs {
        return None;
    }

    let mut rhs_shape = vec![b.shape()[0], 1];
    rhs_shape.extend_from_slice(&b.shape()[1..]);
    Some(rhs_shape)
}

/// Validate that U's diagonal has no singular/zero entries — GPU-accelerated.
///
/// Strategy: extract_diagonal → magnitude → reshape to 1D →
/// reduce_min/reduce_max(axis=0) → download scalar summaries → tolerance check.
///
/// Only the final scalar summaries are transferred to host.
fn validate_nonsingular_gpu(backend: &mut CudaExecSession<'_>, u: &Tensor) -> Result<()> {
    let diag = backend.extract_diagonal(u, 0, 1)?;
    let abs_diag = diagonal_magnitude(backend, &diag)?;

    // Flatten to 1D then reduce_min on axis 0 to get a single scalar.
    let total = checked_shape_product("validate_nonsingular_gpu", "diagonal", abs_diag.shape())?;
    let flat = backend.reshape(&abs_diag, &[total])?;
    let min_val = backend.reduce_min(&flat, &[0])?;
    let max_val = backend.reduce_max(&flat, &[0])?;

    // Host reads must observe the queued GPU reduction result.
    backend.runtime().synchronize()?;
    let host_min = download_tensor(backend.runtime(), &min_val)?;
    let host_max = download_tensor(backend.runtime(), &max_val)?;
    let (value, max_magnitude) = host_min_max_magnitudes(&host_min, &host_max)?;
    let tolerance = singularity_tolerance(abs_diag.dtype(), max_magnitude);
    let is_singular = !value.is_finite() || !max_magnitude.is_finite() || value <= tolerance;

    if is_singular {
        Err(crate::error::into_tensor_error(
            "solve",
            crate::Error::Singular { op: "solve" },
        ))
    } else {
        Ok(())
    }
}

fn diagonal_magnitude(backend: &mut CudaExecSession<'_>, diag: &Tensor) -> Result<Tensor> {
    match diag {
        Tensor::F32(_) | Tensor::F64(_) => backend.abs(diag),
        Tensor::C32(t) => complex32_magnitude(backend, t).map(Tensor::F32),
        Tensor::C64(t) => complex64_magnitude(backend, t).map(Tensor::F64),
        Tensor::I32(_) | Tensor::I64(_) | Tensor::Bool(_) => {
            Err(unsupported_linalg_dtype("validate_nonsingular_gpu", diag))
        }
    }
}

fn complex32_magnitude(
    backend: &mut CudaExecSession<'_>,
    input: &TypedTensor<Complex32>,
) -> Result<TypedTensor<f32>> {
    backend.with_cubecl("validate_nonsingular_gpu", |cubecl| {
        let output = cubecl.alloc_output::<f32>(input.shape())?;
        let input_arg = cubecl.array_arg(input, "validate_nonsingular_gpu")?;
        let output_arg = cubecl.array_arg(&output, "validate_nonsingular_gpu")?;
        if output.n_elements() == 0 {
            return Ok(output);
        }
        let launch_count = cubecl.cube_count_1d(output.n_elements())?;
        // SAFETY: bindings describe live CUDA tensors, and `launch_count`
        // covers exactly the complex32 magnitude output domain.
        unsafe {
            cubecl_linalg::complex32_magnitude::launch_unchecked::<CubeclCudaRuntime>(
                cubecl.client(),
                launch_count,
                cubecl.cube_dim_1d(),
                output_arg,
                input_arg,
            );
        }
        Ok(output)
    })
}

fn complex64_magnitude(
    backend: &mut CudaExecSession<'_>,
    input: &TypedTensor<Complex64>,
) -> Result<TypedTensor<f64>> {
    backend.with_cubecl("validate_nonsingular_gpu", |cubecl| {
        let output = cubecl.alloc_output::<f64>(input.shape())?;
        let input_arg = cubecl.array_arg(input, "validate_nonsingular_gpu")?;
        let output_arg = cubecl.array_arg(&output, "validate_nonsingular_gpu")?;
        if output.n_elements() == 0 {
            return Ok(output);
        }
        let launch_count = cubecl.cube_count_1d(output.n_elements())?;
        // SAFETY: bindings describe live CUDA tensors, and `launch_count`
        // covers exactly the complex64 magnitude output domain.
        unsafe {
            cubecl_linalg::complex64_magnitude::launch_unchecked::<CubeclCudaRuntime>(
                cubecl.client(),
                launch_count,
                cubecl.cube_dim_1d(),
                output_arg,
                input_arg,
            );
        }
        Ok(output)
    })
}

fn host_min_max_magnitudes(host_min: &Tensor, host_max: &Tensor) -> Result<(f64, f64)> {
    match (host_min, host_max) {
        (Tensor::F64(min), Tensor::F64(max)) => Ok((min.host_data()?[0], max.host_data()?[0])),
        (Tensor::F32(min), Tensor::F32(max)) => Ok((
            f64::from(min.host_data()?[0]),
            f64::from(max.host_data()?[0]),
        )),
        _ => Err(Error::Internal(
            "solve: unexpected dtype after magnitude reduction".into(),
        )),
    }
}

fn singularity_tolerance(dtype: DType, max_magnitude: f64) -> f64 {
    let eps = match dtype {
        DType::F32 | DType::C32 => f32::EPSILON as f64,
        DType::F64 | DType::C64 => f64::EPSILON,
        DType::I32 | DType::I64 | DType::Bool => f64::EPSILON,
    };
    eps * max_magnitude.max(1.0)
}
