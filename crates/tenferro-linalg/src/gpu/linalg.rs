use std::ffi::c_void;
use std::ops::Neg;
use std::os::raw::c_char;

use num_complex::{Complex32, Complex64};
use num_traits::{One, Zero};

use super::ffi::cusolver::{
    CublasDiagType, CublasFillMode, CublasOperation, CublasPointerMode, CublasSideMode,
    CudaDataType, CudaLinalgHandles, CudaStream, CusolverEigMode,
};
use crate::backend::CompactQrResult;
use crate::extension::{EighDriver, QrGauge, QrOptions, SvdDriver};
use tenferro_gpu::cuda::{download_tensor, CudaEngine};
use tenferro_tensor::config::SliceConfig;
use tenferro_tensor::{
    DType, Error, StorageBuffer, Tensor, TensorDeviceTransfer, TensorRead, TensorReduction,
    TensorScalar, TensorStructural, TypedTensor, ValidationError,
};

type Result<T> = tenferro_tensor::Result<T>;

trait LinalgScalar: TensorScalar + Copy + Clone + One + Zero + Neg<Output = Self> {
    type Real: TensorScalar + Copy + Clone + Zero + Into<f64>;

    const DATA_TYPE: CudaDataType;
    const NEEDS_RWORK: bool;

    fn copy_matrix_adjoint(
        backend: &mut CudaEngine,
        v: &TypedTensor<Self>,
        vt_shape: &[usize],
        op: &'static str,
    ) -> Result<TypedTensor<Self>>;

    fn apply_positive_qr_gauge(
        backend: &mut CudaEngine,
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
        backend: &mut CudaEngine,
        v: &TypedTensor<Self>,
        vt_shape: &[usize],
        op: &'static str,
    ) -> Result<TypedTensor<Self>> {
        copy_matrix_adjoint_real(backend, v, vt_shape, op)
    }

    fn apply_positive_qr_gauge(
        backend: &mut CudaEngine,
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
        backend: &mut CudaEngine,
        v: &TypedTensor<Self>,
        vt_shape: &[usize],
        op: &'static str,
    ) -> Result<TypedTensor<Self>> {
        copy_matrix_adjoint_real(backend, v, vt_shape, op)
    }

    fn apply_positive_qr_gauge(
        backend: &mut CudaEngine,
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
        backend: &mut CudaEngine,
        v: &TypedTensor<Self>,
        vt_shape: &[usize],
        op: &'static str,
    ) -> Result<TypedTensor<Self>> {
        copy_matrix_adjoint_complex(backend, v, vt_shape, op)
    }

    fn apply_positive_qr_gauge(
        backend: &mut CudaEngine,
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
        backend: &mut CudaEngine,
        v: &TypedTensor<Self>,
        vt_shape: &[usize],
        op: &'static str,
    ) -> Result<TypedTensor<Self>> {
        copy_matrix_adjoint_complex(backend, v, vt_shape, op)
    }

    fn apply_positive_qr_gauge(
        backend: &mut CudaEngine,
        q: &mut TypedTensor<Self>,
        r: &mut TypedTensor<Self>,
        q_start: usize,
        op: &'static str,
    ) -> Result<()> {
        apply_positive_qr_gauge_c64(backend, q, r, q_start, op)
    }
}

const JAX_COMPATIBLE_GESVDJ_MAX_DIM: usize = 1024;

/// Relative `h_err_sigma` above which an explicit Xgesvdp factorization is
/// rejected.
///
/// cuSOLVER reports `h_err_sigma` as the absolute perturbation of the computed
/// singular values. On single-precision rank-deficient input it can be as large
/// as the spectrum itself, so an explicit `Xgesvdp` call returns an error
/// instead of wrong factors (#2182). `Auto` never selects that driver, so this
/// guard is only reachable through an explicit request.
const XGESVDP_MAX_RELATIVE_ERROR: f64 = 1.0e-6;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CusolverSvdRoutine {
    Gesvdj,
    Gesvd,
    Xgesvdp,
}

/// `Auto` follows the JAX-compatible accurate size policy and is the default;
/// an explicit driver always wins.
///
/// `Auto` never selects the polar `Xgesvdp`: its small singular values are only
/// accurate to about `1e-6` of the largest (#2182), which an `eps`-scale
/// truncation cannot tolerate, so it stays opt-in.
fn select_svd_driver(driver: SvdDriver, m: usize, n: usize) -> CusolverSvdRoutine {
    match driver {
        SvdDriver::Gesvdj => CusolverSvdRoutine::Gesvdj,
        SvdDriver::Gesvd => CusolverSvdRoutine::Gesvd,
        SvdDriver::Xgesvdp => CusolverSvdRoutine::Xgesvdp,
        SvdDriver::Auto => {
            if m <= JAX_COMPATIBLE_GESVDJ_MAX_DIM && n <= JAX_COMPATIBLE_GESVDJ_MAX_DIM {
                CusolverSvdRoutine::Gesvdj
            } else {
                CusolverSvdRoutine::Gesvd
            }
        }
    }
}

/// The typed host tensor behind `input`, or this module's standard refusal.
///
/// Callers reach this from a match on `input.dtype()`, so `None` here means the tag
/// table and the runtime dtype disagree rather than a caller mistake.
fn typed_host<'a, T: TensorScalar>(
    op: &'static str,
    input: &'a Tensor,
) -> Result<&'a TypedTensor<T>> {
    input
        .as_typed::<T>()
        .ok_or_else(|| unsupported_linalg_dtype(op, input))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CusolverEighRoutine {
    Syevd,
    Syevj,
    SyevjBatched,
    XsyevBatched,
}

/// A real batch always takes a batched cuSOLVER routine, which replaces one
/// launch per matrix with a single launch. Which batched routine depends only
/// on the driver: `Auto`/`Syevd` take divide-and-conquer `Xsyev`, `Syevj`
/// takes Jacobi. Single matrices have no launch overhead to amortize and keep
/// the per-matrix entry points.
///
/// There is no size threshold. Measured on an A100 (issue #1852), batched
/// Jacobi beats a per-matrix Jacobi loop at every order tested from 8 to 512,
/// and `XsyevBatched` beats a per-matrix `syevd` loop at every one of them, so
/// no order exists at which falling back to a loop is the faster choice.
fn select_eigh_driver(driver: EighDriver, batch_total: usize) -> CusolverEighRoutine {
    let batched = batch_total > 1;
    match driver {
        EighDriver::Auto | EighDriver::Syevd => {
            if batched {
                CusolverEighRoutine::XsyevBatched
            } else {
                CusolverEighRoutine::Syevd
            }
        }
        EighDriver::Syevj => {
            if batched {
                CusolverEighRoutine::SyevjBatched
            } else {
                CusolverEighRoutine::Syevj
            }
        }
    }
}

fn unsupported_linalg_dtype(op: &'static str, input: &Tensor) -> Error {
    crate::error::unsupported_dtype(op, input.dtype())
}

fn ensure_supported_linalg_pair(op: &'static str, lhs: &Tensor, rhs: &Tensor) -> Result<()> {
    if lhs.dtype() != rhs.dtype() {
        return Err(Error::dtype_mismatch(op, lhs.dtype(), rhs.dtype()));
    }
    match lhs.dtype() {
        DType::F32 | DType::F64 | DType::C32 | DType::C64 => Ok(()),
        DType::I32 | DType::I64 | DType::Bool => Err(unsupported_linalg_dtype(op, lhs)),
        DType::External(_) => Err(unsupported_linalg_dtype(op, lhs)),
    }
}

fn ensure_device_resident_tensor(op: &'static str, input: &Tensor) -> Result<()> {
    match input.dtype() {
        DType::F32 => ensure_device_resident_typed(op, typed_host::<f32>(op, input)?),
        DType::F64 => ensure_device_resident_typed(op, typed_host::<f64>(op, input)?),
        DType::I32 => ensure_device_resident_typed(op, typed_host::<i32>(op, input)?),
        DType::I64 => ensure_device_resident_typed(op, typed_host::<i64>(op, input)?),
        DType::Bool => ensure_device_resident_typed(op, typed_host::<bool>(op, input)?),
        DType::C32 => ensure_device_resident_typed(op, typed_host::<Complex32>(op, input)?),
        DType::C64 => ensure_device_resident_typed(op, typed_host::<Complex64>(op, input)?),
        DType::External(_) => Err(unsupported_linalg_dtype(op, input)),
    }
}

/// Validate that a tensor is device-resident using only public tensor
/// metadata (no runtime or launch), so the guard lives entirely on the
/// credentialed public seam.
fn ensure_device_resident_typed<T: 'static>(
    op: &'static str,
    input: &TypedTensor<T>,
) -> Result<()> {
    match input.buffer() {
        StorageBuffer::Host(_) => Err(Error::runtime_state(
            op,
            "expected a CUDA tensor, got a host tensor. \
                      Use upload_tensor() to transfer to GPU before calling GPU ops.",
        )),
        StorageBuffer::Backend(buffer) if buffer.backend_family() == "cuda" => Ok(()),
        StorageBuffer::Backend(buffer) => Err(Error::runtime_state(
            op,
            format!(
                "expected a CUDA tensor, got backend buffer family `{}`",
                buffer.backend_family()
            ),
        )),
    }
}

pub(super) fn cholesky(backend: &mut CudaEngine, input: &Tensor) -> Result<Tensor> {
    match input.dtype() {
        DType::F32 => cholesky_typed(backend, typed_host::<f32>("cholesky", input)?)
            .map(Tensor::from_typed::<f32>),
        DType::F64 => cholesky_typed(backend, typed_host::<f64>("cholesky", input)?)
            .map(Tensor::from_typed::<f64>),
        DType::C32 => cholesky_typed(backend, typed_host::<Complex32>("cholesky", input)?)
            .map(Tensor::from_typed::<num_complex::Complex32>),
        DType::C64 => cholesky_typed(backend, typed_host::<Complex64>("cholesky", input)?)
            .map(Tensor::from_typed::<num_complex::Complex64>),
        DType::I32 | DType::I64 | DType::Bool => Err(unsupported_linalg_dtype("cholesky", input)),
        DType::External(_) => Err(unsupported_linalg_dtype("cholesky", input)),
    }
}

pub(super) fn triangular_solve(
    backend: &mut CudaEngine,
    a: &Tensor,
    b: &Tensor,
    left_side: bool,
    lower: bool,
    transpose_a: bool,
    unit_diagonal: bool,
) -> Result<Tensor> {
    match (a.dtype(), b.dtype()) {
        (DType::F32, DType::F32) => {
            let (a, b) = triangular_solve_operands::<f32>(a, b)?;
            triangular_solve_typed(backend, a, b, left_side, lower, transpose_a, unit_diagonal)
                .map(Tensor::from_typed::<f32>)
        }
        (DType::F64, DType::F64) => {
            let (a, b) = triangular_solve_operands::<f64>(a, b)?;
            triangular_solve_typed(backend, a, b, left_side, lower, transpose_a, unit_diagonal)
                .map(Tensor::from_typed::<f64>)
        }
        (DType::C32, DType::C32) => {
            let (a, b) = triangular_solve_operands::<num_complex::Complex32>(a, b)?;
            triangular_solve_typed(backend, a, b, left_side, lower, transpose_a, unit_diagonal)
                .map(Tensor::from_typed::<num_complex::Complex32>)
        }
        (DType::C64, DType::C64) => {
            let (a, b) = triangular_solve_operands::<num_complex::Complex64>(a, b)?;
            triangular_solve_typed(backend, a, b, left_side, lower, transpose_a, unit_diagonal)
                .map(Tensor::from_typed::<num_complex::Complex64>)
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
/// The typed operands behind a same-dtype pair, or this module's refusal for one.
///
/// Callers reach this from a match on the pair of dtypes, so `None` means the tags and the runtime
/// dtypes disagree rather than a caller mistake.
fn triangular_solve_operands<'a, T: TensorScalar>(
    a: &'a Tensor,
    b: &'a Tensor,
) -> Result<(&'a TypedTensor<T>, &'a TypedTensor<T>)> {
    let a_t = a
        .as_typed::<T>()
        .ok_or_else(|| crate::error::unsupported_dtype("triangular_solve", a.dtype()))?;
    let b_t = b
        .as_typed::<T>()
        .ok_or_else(|| crate::error::unsupported_dtype("triangular_solve", b.dtype()))?;
    Ok((a_t, b_t))
}

pub(super) fn lu(backend: &mut CudaEngine, input: &Tensor) -> Result<Vec<Tensor>> {
    match input.dtype() {
        DType::F32 => {
            lu_typed(backend, typed_host::<f32>("lu", input)?).map(|(p, l, u, parity)| {
                vec![
                    Tensor::from_typed::<f32>(p),
                    Tensor::from_typed::<f32>(l),
                    Tensor::from_typed::<f32>(u),
                    Tensor::from_typed::<f32>(parity),
                ]
            })
        }
        DType::F64 => {
            lu_typed(backend, typed_host::<f64>("lu", input)?).map(|(p, l, u, parity)| {
                vec![
                    Tensor::from_typed::<f64>(p),
                    Tensor::from_typed::<f64>(l),
                    Tensor::from_typed::<f64>(u),
                    Tensor::from_typed::<f64>(parity),
                ]
            })
        }
        DType::C32 => {
            lu_typed(backend, typed_host::<Complex32>("lu", input)?).map(|(p, l, u, parity)| {
                vec![
                    Tensor::from_typed::<Complex32>(p),
                    Tensor::from_typed::<Complex32>(l),
                    Tensor::from_typed::<Complex32>(u),
                    Tensor::from_typed::<Complex32>(parity),
                ]
            })
        }
        DType::C64 => {
            lu_typed(backend, typed_host::<Complex64>("lu", input)?).map(|(p, l, u, parity)| {
                vec![
                    Tensor::from_typed::<Complex64>(p),
                    Tensor::from_typed::<Complex64>(l),
                    Tensor::from_typed::<Complex64>(u),
                    Tensor::from_typed::<Complex64>(parity),
                ]
            })
        }
        DType::I32 | DType::I64 | DType::Bool => Err(unsupported_linalg_dtype("lu", input)),
        DType::External(_) => Err(unsupported_linalg_dtype("lu", input)),
    }
}

pub(super) fn lu_factor(backend: &mut CudaEngine, input: &Tensor) -> Result<Vec<Tensor>> {
    lu_factor_with_op(backend, input, "lu_factor")
}

/// LU factorization returning the public `(packed_lu, pivots, parity)` outputs.
///
/// The standalone `lu_factor`/`lu` operations keep the cuSOLVER status check
/// and the permutation parity. The forward solve uses `lu_factor_for_solve`
/// instead and skips both.
fn lu_factor_with_op(
    backend: &mut CudaEngine,
    input: &Tensor,
    op: &'static str,
) -> Result<Vec<Tensor>> {
    match input.dtype() {
        DType::F32 => lu_factor_typed(backend, typed_host::<f32>(op, input)?, op).map(
            |(packed_lu, pivots, parity)| {
                vec![
                    Tensor::from_typed::<f32>(packed_lu),
                    Tensor::from_typed::<i32>(pivots),
                    Tensor::from_typed::<f32>(parity),
                ]
            },
        ),
        DType::F64 => lu_factor_typed(backend, typed_host::<f64>(op, input)?, op).map(
            |(packed_lu, pivots, parity)| {
                vec![
                    Tensor::from_typed::<f64>(packed_lu),
                    Tensor::from_typed::<i32>(pivots),
                    Tensor::from_typed::<f64>(parity),
                ]
            },
        ),
        DType::C32 => lu_factor_typed(backend, typed_host::<Complex32>(op, input)?, op).map(
            |(packed_lu, pivots, parity)| {
                vec![
                    Tensor::from_typed::<Complex32>(packed_lu),
                    Tensor::from_typed::<i32>(pivots),
                    Tensor::from_typed::<Complex32>(parity),
                ]
            },
        ),
        DType::C64 => lu_factor_typed(backend, typed_host::<Complex64>(op, input)?, op).map(
            |(packed_lu, pivots, parity)| {
                vec![
                    Tensor::from_typed::<Complex64>(packed_lu),
                    Tensor::from_typed::<i32>(pivots),
                    Tensor::from_typed::<Complex64>(parity),
                ]
            },
        ),
        DType::I32 | DType::I64 | DType::Bool => Err(unsupported_linalg_dtype(op, input)),
        DType::External(_) => Err(unsupported_linalg_dtype(op, input)),
    }
}

/// Factor `input` for the forward solve path.
///
/// The forward solve does not inspect cuSOLVER's `getrf` status and does not
/// materialize the permutation parity that the public `lu_factor` returns. A
/// singular input therefore reaches the triangular solve instead of failing at
/// factorization; the caller's residual check decides whether the result is
/// acceptable. The public `lu_factor`/`lu` operations keep the status check and
/// parity output.
fn lu_factor_for_solve(backend: &mut CudaEngine, input: &Tensor) -> Result<(Tensor, Tensor)> {
    match input.dtype() {
        DType::F32 => {
            lu_factor_for_solve_typed(backend, typed_host::<f32>("solve", input)?, "solve").map(
                |(packed_lu, pivots)| {
                    (
                        Tensor::from_typed::<f32>(packed_lu),
                        Tensor::from_typed::<i32>(pivots),
                    )
                },
            )
        }
        DType::F64 => {
            lu_factor_for_solve_typed(backend, typed_host::<f64>("solve", input)?, "solve").map(
                |(packed_lu, pivots)| {
                    (
                        Tensor::from_typed::<f64>(packed_lu),
                        Tensor::from_typed::<i32>(pivots),
                    )
                },
            )
        }
        DType::C32 => {
            lu_factor_for_solve_typed(backend, typed_host::<Complex32>("solve", input)?, "solve")
                .map(|(packed_lu, pivots)| {
                    (
                        Tensor::from_typed::<Complex32>(packed_lu),
                        Tensor::from_typed::<i32>(pivots),
                    )
                })
        }
        DType::C64 => {
            lu_factor_for_solve_typed(backend, typed_host::<Complex64>("solve", input)?, "solve")
                .map(|(packed_lu, pivots)| {
                    (
                        Tensor::from_typed::<Complex64>(packed_lu),
                        Tensor::from_typed::<i32>(pivots),
                    )
                })
        }
        DType::I32 | DType::I64 | DType::Bool => Err(unsupported_linalg_dtype("solve", input)),
        DType::External(_) => Err(unsupported_linalg_dtype("solve", input)),
    }
}

pub(super) fn full_piv_lu(_backend: &mut CudaEngine, _input: &Tensor) -> Result<Vec<Tensor>> {
    Err(Error::unsupported(
        "full_piv_lu",
        "complete-pivoting LU is not implemented for the CubeCL backend",
    ))
}

pub(super) fn full_piv_lu_solve(
    _backend: &mut CudaEngine,
    _a: &Tensor,
    _b: &Tensor,
    _transpose_a: bool,
) -> Result<Tensor> {
    Err(Error::unsupported(
        "full_piv_lu_solve",
        "complete-pivoting LU solve is not implemented for the CubeCL backend",
    ))
}

pub(super) fn svd(
    backend: &mut CudaEngine,
    input: &Tensor,
    driver: SvdDriver,
) -> Result<Vec<Tensor>> {
    svd_dispatch(backend, input, false, driver)
}

/// Full-matrices SVD: `U` is `m x m` and `Vt` is `n x n`.
pub(super) fn svd_full(backend: &mut CudaEngine, input: &Tensor) -> Result<Vec<Tensor>> {
    svd_dispatch(backend, input, true, SvdDriver::Auto)
}

fn svd_dispatch(
    backend: &mut CudaEngine,
    input: &Tensor,
    full: bool,
    driver: SvdDriver,
) -> Result<Vec<Tensor>> {
    let op: &'static str = if full { "svd_full" } else { "svd" };
    match input.dtype() {
        DType::F32 => {
            svd_typed(backend, typed_host::<f32>(op, input)?, full, driver).map(|(u, s, vt)| {
                vec![
                    Tensor::from_typed::<f32>(u),
                    Tensor::from_typed::<f32>(s),
                    Tensor::from_typed::<f32>(vt),
                ]
            })
        }
        DType::F64 => {
            svd_typed(backend, typed_host::<f64>(op, input)?, full, driver).map(|(u, s, vt)| {
                vec![
                    Tensor::from_typed::<f64>(u),
                    Tensor::from_typed::<f64>(s),
                    Tensor::from_typed::<f64>(vt),
                ]
            })
        }
        DType::C32 => svd_typed(backend, typed_host::<Complex32>(op, input)?, full, driver).map(
            |(u, s, vt)| {
                vec![
                    Tensor::from_typed::<Complex32>(u),
                    Tensor::from_typed::<f32>(s),
                    Tensor::from_typed::<Complex32>(vt),
                ]
            },
        ),
        DType::C64 => svd_typed(backend, typed_host::<Complex64>(op, input)?, full, driver).map(
            |(u, s, vt)| {
                vec![
                    Tensor::from_typed::<Complex64>(u),
                    Tensor::from_typed::<f64>(s),
                    Tensor::from_typed::<Complex64>(vt),
                ]
            },
        ),
        DType::I32 | DType::I64 | DType::Bool => Err(unsupported_linalg_dtype(op, input)),
        DType::External(_) => Err(unsupported_linalg_dtype(op, input)),
    }
}

pub(super) fn svd_values(
    backend: &mut CudaEngine,
    input: &Tensor,
    driver: SvdDriver,
) -> Result<Tensor> {
    match input.dtype() {
        DType::F32 => svd_values_typed(backend, typed_host::<f32>("svd_values", input)?, driver)
            .map(Tensor::from_typed::<f32>),
        DType::F64 => svd_values_typed(backend, typed_host::<f64>("svd_values", input)?, driver)
            .map(Tensor::from_typed::<f64>),
        DType::C32 => svd_values_typed(
            backend,
            typed_host::<Complex32>("svd_values", input)?,
            driver,
        )
        .map(Tensor::from_typed::<f32>),
        DType::C64 => svd_values_typed(
            backend,
            typed_host::<Complex64>("svd_values", input)?,
            driver,
        )
        .map(Tensor::from_typed::<f64>),
        DType::I32 | DType::I64 | DType::Bool => Err(unsupported_linalg_dtype("svd_values", input)),
        DType::External(_) => Err(unsupported_linalg_dtype("svd_values", input)),
    }
}

pub(super) fn qr(backend: &mut CudaEngine, input: &Tensor) -> Result<Vec<Tensor>> {
    match input.dtype() {
        DType::F32 => qr_typed(backend, typed_host::<f32>("qr", input)?)
            .map(|(q, r)| vec![Tensor::from_typed::<f32>(q), Tensor::from_typed::<f32>(r)]),
        DType::F64 => qr_typed(backend, typed_host::<f64>("qr", input)?)
            .map(|(q, r)| vec![Tensor::from_typed::<f64>(q), Tensor::from_typed::<f64>(r)]),
        DType::C32 => qr_typed(backend, typed_host::<Complex32>("qr", input)?).map(|(q, r)| {
            vec![
                Tensor::from_typed::<Complex32>(q),
                Tensor::from_typed::<Complex32>(r),
            ]
        }),
        DType::C64 => qr_typed(backend, typed_host::<Complex64>("qr", input)?).map(|(q, r)| {
            vec![
                Tensor::from_typed::<Complex64>(q),
                Tensor::from_typed::<Complex64>(r),
            ]
        }),
        DType::I32 | DType::I64 | DType::Bool => Err(unsupported_linalg_dtype("qr", input)),
        DType::External(_) => Err(unsupported_linalg_dtype("qr", input)),
    }
}

pub(super) fn rank_revealing_qr(
    backend: &mut CudaEngine,
    input: &Tensor,
    options: crate::RankRevealingQrOptions,
) -> Result<Vec<Tensor>> {
    rank_revealing_qr::rank_revealing_qr(backend, input, options)
}

pub(super) fn householder_qr(backend: &mut CudaEngine, input: &Tensor) -> Result<CompactQrResult> {
    let (packed, coeff) = match input.dtype() {
        DType::F32 => {
            let (packed, coeff) = compact_qr_typed(
                backend,
                typed_host::<f32>("householder_qr", input)?,
                "householder_qr",
            )?;
            (
                Tensor::from_typed::<f32>(packed),
                Tensor::from_typed::<f32>(coeff),
            )
        }
        DType::F64 => {
            let (packed, coeff) = compact_qr_typed(
                backend,
                typed_host::<f64>("householder_qr", input)?,
                "householder_qr",
            )?;
            (
                Tensor::from_typed::<f64>(packed),
                Tensor::from_typed::<f64>(coeff),
            )
        }
        DType::C32 => {
            let (packed, coeff) = compact_qr_typed(
                backend,
                typed_host::<Complex32>("householder_qr", input)?,
                "householder_qr",
            )?;
            (
                Tensor::from_typed::<Complex32>(packed),
                Tensor::from_typed::<Complex32>(coeff),
            )
        }
        DType::C64 => {
            let (packed, coeff) = compact_qr_typed(
                backend,
                typed_host::<Complex64>("householder_qr", input)?,
                "householder_qr",
            )?;
            (
                Tensor::from_typed::<Complex64>(packed),
                Tensor::from_typed::<Complex64>(coeff),
            )
        }
        DType::I32 | DType::I64 | DType::Bool => {
            return Err(unsupported_linalg_dtype("householder_qr", input));
        }
        DType::External(_) => return Err(unsupported_linalg_dtype("householder_qr", input)),
    };
    Ok(CompactQrResult { packed, coeff })
}

fn upper_trapezoidal_violation_typed<T>(
    backend: &mut CudaEngine,
    input: &TypedTensor<T>,
    op: &'static str,
) -> Result<TypedTensor<i32>>
where
    T: LinalgScalar + TensorScalar + PartialEq,
{
    let output = backend.alloc_output(DType::I32, input.shape())?;
    if input.shape().iter().all(|&extent| extent != 0) {
        backend.linalg_upper_trapezoidal_violation(input, &output, op)?;
    }
    output
        .into_typed::<i32>()
        .map_err(|_| Error::Internal("upper trapezoid output changed dtype".into()))
}

fn validate_upper_trapezoidal_gpu(
    backend: &mut CudaEngine,
    r: &Tensor,
    op: &'static str,
) -> Result<()> {
    if r.shape().len() != 2 {
        return Err(Error::rank_mismatch(op, 2, r.shape().len()));
    }
    if r.shape().contains(&0) {
        return Ok(());
    }
    let flags = match r.dtype() {
        DType::F32 => upper_trapezoidal_violation_typed(backend, typed_host::<f32>(op, r)?, op)?,
        DType::F64 => upper_trapezoidal_violation_typed(backend, typed_host::<f64>(op, r)?, op)?,
        DType::C32 => {
            upper_trapezoidal_violation_typed(backend, typed_host::<Complex32>(op, r)?, op)?
        }
        DType::C64 => {
            upper_trapezoidal_violation_typed(backend, typed_host::<Complex64>(op, r)?, op)?
        }
        DType::I32 | DType::I64 | DType::Bool => {
            return Err(unsupported_linalg_dtype(op, r));
        }
        DType::External(_) => return Err(unsupported_linalg_dtype(op, r)),
    };
    let flags = Tensor::from_typed::<i32>(flags);
    let maximum = backend.reduce_max_read(TensorRead::from_tensor(&flags), &[0, 1])?;
    backend
        .runtime()
        .synchronize()
        .map_err(|error| Error::Internal(format!("{op}: stream barrier failed: {error}")))?;
    let host = download_tensor(backend.runtime(), &maximum)?;
    let value = host.into_typed::<i32>().map_err(|_| {
        Error::Internal(format!(
            "{op}: unexpected upper-trapezoidal validation dtype"
        ))
    })?;
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

/// The typed operands behind a same-dtype pair, or the refusal this entry point reports.
fn compact_qr_operands<'a, T: TensorScalar>(
    q: &'a Tensor,
    r: &'a Tensor,
    op: &'static str,
) -> Result<(&'a TypedTensor<T>, &'a TypedTensor<T>)> {
    let q_t = q
        .as_typed::<T>()
        .ok_or_else(|| Error::dtype_mismatch(op, q.dtype(), r.dtype()))?;
    let r_t = r
        .as_typed::<T>()
        .ok_or_else(|| Error::dtype_mismatch(op, q.dtype(), r.dtype()))?;
    Ok((q_t, r_t))
}

pub(super) fn householder_qr_from_factors(
    backend: &mut CudaEngine,
    q: &Tensor,
    r: &Tensor,
) -> Result<CompactQrResult> {
    const OP: &str = "householder_qr_from_factors";
    ensure_supported_linalg_pair(OP, q, r)?;
    validate_upper_trapezoidal_gpu(backend, r, OP)?;
    let (packed, coeff) = match (q.dtype(), r.dtype()) {
        (DType::F32, DType::F32) => {
            let (q, r) = compact_qr_operands::<f32>(q, r, OP)?;
            let (packed, coeff) = compact_qr_from_factors_typed(backend, q, r, OP)?;
            (
                Tensor::from_typed::<f32>(packed),
                Tensor::from_typed::<f32>(coeff),
            )
        }
        (DType::F64, DType::F64) => {
            let (q, r) = compact_qr_operands::<f64>(q, r, OP)?;
            let (packed, coeff) = compact_qr_from_factors_typed(backend, q, r, OP)?;
            (
                Tensor::from_typed::<f64>(packed),
                Tensor::from_typed::<f64>(coeff),
            )
        }
        (DType::C32, DType::C32) => {
            let (q, r) = compact_qr_operands::<Complex32>(q, r, OP)?;
            let (packed, coeff) = compact_qr_from_factors_typed(backend, q, r, OP)?;
            (
                Tensor::from_typed::<Complex32>(packed),
                Tensor::from_typed::<Complex32>(coeff),
            )
        }
        (DType::C64, DType::C64) => {
            let (q, r) = compact_qr_operands::<Complex64>(q, r, OP)?;
            let (packed, coeff) = compact_qr_from_factors_typed(backend, q, r, OP)?;
            (
                Tensor::from_typed::<Complex64>(packed),
                Tensor::from_typed::<Complex64>(coeff),
            )
        }
        _ => return Err(Error::dtype_mismatch(OP, q.dtype(), r.dtype())),
    };
    Ok(CompactQrResult { packed, coeff })
}
/// The typed operands behind a same-dtype triple, or this entry point's refusal for one.
fn qr_append_operands<'a, T: TensorScalar>(
    packed: &'a Tensor,
    coeff: &'a Tensor,
    block: &'a Tensor,
) -> Result<(&'a TypedTensor<T>, &'a TypedTensor<T>, &'a TypedTensor<T>)> {
    let mismatch = || Error::dtype_mismatch("householder_qr_append", packed.dtype(), block.dtype());
    let packed_t = packed.as_typed::<T>().ok_or_else(mismatch)?;
    let coeff_t = coeff.as_typed::<T>().ok_or_else(mismatch)?;
    let block_t = block.as_typed::<T>().ok_or_else(mismatch)?;
    Ok((packed_t, coeff_t, block_t))
}

pub(super) fn householder_qr_append(
    backend: &mut CudaEngine,
    packed: &Tensor,
    coeff: &Tensor,
    block: &Tensor,
) -> Result<CompactQrResult> {
    ensure_supported_linalg_pair("householder_qr_append", packed, coeff)?;
    ensure_supported_linalg_pair("householder_qr_append", packed, block)?;
    let (packed, coeff) = match (packed.dtype(), coeff.dtype(), block.dtype()) {
        (DType::F32, DType::F32, DType::F32) => {
            let (packed, coeff, block) = qr_append_operands::<f32>(packed, coeff, block)?;
            let (packed, coeff) =
                compact_qr_append_typed(backend, packed, coeff, block, "householder_qr_append")?;
            (
                Tensor::from_typed::<f32>(packed),
                Tensor::from_typed::<f32>(coeff),
            )
        }
        (DType::F64, DType::F64, DType::F64) => {
            let (packed, coeff, block) = qr_append_operands::<f64>(packed, coeff, block)?;
            let (packed, coeff) =
                compact_qr_append_typed(backend, packed, coeff, block, "householder_qr_append")?;
            (
                Tensor::from_typed::<f64>(packed),
                Tensor::from_typed::<f64>(coeff),
            )
        }
        (DType::C32, DType::C32, DType::C32) => {
            let (packed, coeff, block) = qr_append_operands::<Complex32>(packed, coeff, block)?;
            let (packed, coeff) =
                compact_qr_append_typed(backend, packed, coeff, block, "householder_qr_append")?;
            (
                Tensor::from_typed::<Complex32>(packed),
                Tensor::from_typed::<Complex32>(coeff),
            )
        }
        (DType::C64, DType::C64, DType::C64) => {
            let (packed, coeff, block) = qr_append_operands::<Complex64>(packed, coeff, block)?;
            let (packed, coeff) =
                compact_qr_append_typed(backend, packed, coeff, block, "householder_qr_append")?;
            (
                Tensor::from_typed::<Complex64>(packed),
                Tensor::from_typed::<Complex64>(coeff),
            )
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
/// The typed operands behind a same-dtype pair, or this entry point's refusal for one.
fn qr_q_columns_operands<'a, T: TensorScalar>(
    packed: &'a Tensor,
    coeff: &'a Tensor,
) -> Result<(&'a TypedTensor<T>, &'a TypedTensor<T>)> {
    let packed_t = packed.as_typed::<T>().ok_or_else(|| {
        Error::dtype_mismatch("householder_qr_q_columns", packed.dtype(), coeff.dtype())
    })?;
    let coeff_t = coeff.as_typed::<T>().ok_or_else(|| {
        Error::dtype_mismatch("householder_qr_q_columns", packed.dtype(), coeff.dtype())
    })?;
    Ok((packed_t, coeff_t))
}

pub(super) fn householder_qr_q_columns(
    backend: &mut CudaEngine,
    packed: &Tensor,
    coeff: &Tensor,
    start: usize,
    end: usize,
    options: QrOptions,
) -> Result<Tensor> {
    ensure_supported_linalg_pair("householder_qr_q_columns", packed, coeff)?;
    match (packed.dtype(), coeff.dtype()) {
        (DType::F32, DType::F32) => {
            let (packed, coeff) = qr_q_columns_operands::<f32>(packed, coeff)?;
            compact_qr_q_columns_typed(
                backend,
                packed,
                coeff,
                start,
                end,
                options,
                "householder_qr_q_columns",
            )
            .map(Tensor::from_typed::<f32>)
        }
        (DType::F64, DType::F64) => {
            let (packed, coeff) = qr_q_columns_operands::<f64>(packed, coeff)?;
            compact_qr_q_columns_typed(
                backend,
                packed,
                coeff,
                start,
                end,
                options,
                "householder_qr_q_columns",
            )
            .map(Tensor::from_typed::<f64>)
        }
        (DType::C32, DType::C32) => {
            let (packed, coeff) = qr_q_columns_operands::<num_complex::Complex32>(packed, coeff)?;
            compact_qr_q_columns_typed(
                backend,
                packed,
                coeff,
                start,
                end,
                options,
                "householder_qr_q_columns",
            )
            .map(Tensor::from_typed::<num_complex::Complex32>)
        }
        (DType::C64, DType::C64) => {
            let (packed, coeff) = qr_q_columns_operands::<num_complex::Complex64>(packed, coeff)?;
            compact_qr_q_columns_typed(
                backend,
                packed,
                coeff,
                start,
                end,
                options,
                "householder_qr_q_columns",
            )
            .map(Tensor::from_typed::<num_complex::Complex64>)
        }
        _ => Err(Error::dtype_mismatch(
            "householder_qr_q_columns",
            packed.dtype(),
            coeff.dtype(),
        )),
    }
}
/// The typed operands behind a same-dtype pair, or this entry point's refusal for one.
fn qr_r_operands<'a, T: TensorScalar>(
    packed: &'a Tensor,
    coeff: &'a Tensor,
) -> Result<(&'a TypedTensor<T>, &'a TypedTensor<T>)> {
    let packed_t = packed
        .as_typed::<T>()
        .ok_or_else(|| Error::dtype_mismatch("householder_qr_r", packed.dtype(), coeff.dtype()))?;
    let coeff_t = coeff
        .as_typed::<T>()
        .ok_or_else(|| Error::dtype_mismatch("householder_qr_r", packed.dtype(), coeff.dtype()))?;
    Ok((packed_t, coeff_t))
}

pub(super) fn householder_qr_r(
    backend: &mut CudaEngine,
    packed: &Tensor,
    coeff: &Tensor,
    options: QrOptions,
) -> Result<Tensor> {
    ensure_supported_linalg_pair("householder_qr_r", packed, coeff)?;
    match (packed.dtype(), coeff.dtype()) {
        (DType::F32, DType::F32) => {
            let (packed, coeff) = qr_r_operands::<f32>(packed, coeff)?;
            compact_qr_r_typed(backend, packed, coeff, options, "householder_qr_r")
                .map(Tensor::from_typed::<f32>)
        }
        (DType::F64, DType::F64) => {
            let (packed, coeff) = qr_r_operands::<f64>(packed, coeff)?;
            compact_qr_r_typed(backend, packed, coeff, options, "householder_qr_r")
                .map(Tensor::from_typed::<f64>)
        }
        (DType::C32, DType::C32) => {
            let (packed, coeff) = qr_r_operands::<num_complex::Complex32>(packed, coeff)?;
            compact_qr_r_typed(backend, packed, coeff, options, "householder_qr_r")
                .map(Tensor::from_typed::<num_complex::Complex32>)
        }
        (DType::C64, DType::C64) => {
            let (packed, coeff) = qr_r_operands::<num_complex::Complex64>(packed, coeff)?;
            compact_qr_r_typed(backend, packed, coeff, options, "householder_qr_r")
                .map(Tensor::from_typed::<num_complex::Complex64>)
        }
        _ => Err(Error::dtype_mismatch(
            "householder_qr_r",
            packed.dtype(),
            coeff.dtype(),
        )),
    }
}

pub(super) fn qr_with_options(
    backend: &mut CudaEngine,
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
    backend: &mut CudaEngine,
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
    match (q[0].dtype(), r[0].dtype()) {
        (DType::F32, DType::F32) => {
            let (q, r) = gauge_pair_mut::<f32>(&mut q[0], &mut r[0], op)?;
            f32::apply_positive_qr_gauge(backend, q, r, q_start, op)
        }
        (DType::F64, DType::F64) => {
            let (q, r) = gauge_pair_mut::<f64>(&mut q[0], &mut r[0], op)?;
            f64::apply_positive_qr_gauge(backend, q, r, q_start, op)
        }
        (DType::C32, DType::C32) => {
            let (q, r) = gauge_pair_mut::<Complex32>(&mut q[0], &mut r[0], op)?;
            Complex32::apply_positive_qr_gauge(backend, q, r, q_start, op)
        }
        (DType::C64, DType::C64) => {
            let (q, r) = gauge_pair_mut::<Complex64>(&mut q[0], &mut r[0], op)?;
            Complex64::apply_positive_qr_gauge(backend, q, r, q_start, op)
        }
        _ => Err(Error::dtype_mismatch(op, q[0].dtype(), r[0].dtype())),
    }
}

/// The typed pair behind a same-dtype pair of mutable tensors, or this module's refusal.
fn gauge_pair_mut<'a, T: TensorScalar>(
    q: &'a mut Tensor,
    r: &'a mut Tensor,
    op: &'static str,
) -> Result<(&'a mut TypedTensor<T>, &'a mut TypedTensor<T>)> {
    let (q_dtype, r_dtype) = (q.dtype(), r.dtype());
    let q_t = q
        .as_typed_mut::<T>()
        .ok_or_else(|| Error::dtype_mismatch(op, q_dtype, r_dtype))?;
    let r_t = r
        .as_typed_mut::<T>()
        .ok_or_else(|| Error::dtype_mismatch(op, q_dtype, r_dtype))?;
    Ok((q_t, r_t))
}

pub(super) fn eigh(
    backend: &mut CudaEngine,
    input: &Tensor,
    driver: EighDriver,
) -> Result<Vec<Tensor>> {
    match input.dtype() {
        DType::F32 => eigh_typed(backend, typed_host::<f32>("eigh", input)?, driver)
            .map(|(w, v)| vec![Tensor::from_typed::<f32>(w), Tensor::from_typed::<f32>(v)]),
        DType::F64 => eigh_typed(backend, typed_host::<f64>("eigh", input)?, driver)
            .map(|(w, v)| vec![Tensor::from_typed::<f64>(w), Tensor::from_typed::<f64>(v)]),
        DType::C32 => {
            eigh_typed(backend, typed_host::<Complex32>("eigh", input)?, driver).map(|(w, v)| {
                vec![
                    Tensor::from_typed::<f32>(w),
                    Tensor::from_typed::<Complex32>(v),
                ]
            })
        }
        DType::C64 => {
            eigh_typed(backend, typed_host::<Complex64>("eigh", input)?, driver).map(|(w, v)| {
                vec![
                    Tensor::from_typed::<f64>(w),
                    Tensor::from_typed::<Complex64>(v),
                ]
            })
        }
        DType::I32 | DType::I64 | DType::Bool => Err(unsupported_linalg_dtype("eigh", input)),
        DType::External(_) => Err(unsupported_linalg_dtype("eigh", input)),
    }
}

pub(super) fn eigh_values(
    backend: &mut CudaEngine,
    input: &Tensor,
    driver: EighDriver,
) -> Result<Tensor> {
    match input.dtype() {
        DType::F32 => eigh_values_typed(backend, typed_host::<f32>("eigh_values", input)?, driver)
            .map(Tensor::from_typed::<f32>),
        DType::F64 => eigh_values_typed(backend, typed_host::<f64>("eigh_values", input)?, driver)
            .map(Tensor::from_typed::<f64>),
        DType::C32 => eigh_values_typed(
            backend,
            typed_host::<Complex32>("eigh_values", input)?,
            driver,
        )
        .map(Tensor::from_typed::<f32>),
        DType::C64 => eigh_values_typed(
            backend,
            typed_host::<Complex64>("eigh_values", input)?,
            driver,
        )
        .map(Tensor::from_typed::<f64>),
        DType::I32 | DType::I64 | DType::Bool => {
            Err(unsupported_linalg_dtype("eigh_values", input))
        }
        DType::External(_) => Err(unsupported_linalg_dtype("eigh_values", input)),
    }
}

pub(super) fn eig(_backend: &mut CudaEngine, _input: &Tensor) -> Result<Vec<Tensor>> {
    Err(Error::unsupported(
        "eig",
        "non-symmetric eigendecomposition is not supported on the CubeCL GPU backend \
                  because cuSOLVER does not provide it. Download to CPU explicitly via \
                  `backend.download_to_host(&gpu_tensor)?` and then call `CpuEngine::eig`."
            .to_string(),
    ))
}

pub(super) fn solve(backend: &mut CudaEngine, a: &Tensor, b: &Tensor) -> Result<Tensor> {
    const OP: &str = "solve";

    ensure_device_resident_tensor(OP, a)?;
    ensure_device_resident_tensor(OP, b)?;
    ensure_supported_linalg_pair(OP, a, b)?;
    if has_zero_dim(a.shape()) || has_zero_dim(b.shape()) {
        return zero_like_linalg_device_tensor(backend, b, OP);
    }

    let (rhs, restore_shape) = if let Some(matrix_rhs_shape) = batched_vector_rhs_shape(a, b) {
        (
            backend.reshape_read(TensorRead::from_tensor(b), &matrix_rhs_shape)?,
            Some(b.shape().to_vec()),
        )
    } else {
        (
            backend.to_contiguous_read(TensorRead::from_tensor(b))?,
            None,
        )
    };

    let (packed_lu, pivots) = lu_factor_for_solve(backend, a)?;
    let x = lu_solve_prepared(backend, a, &packed_lu, &pivots, &rhs, false, false)?;
    if let Some(shape) = restore_shape {
        backend.reshape_read(TensorRead::from_tensor(&x), &shape)
    } else {
        Ok(x)
    }
}
/// The typed operands behind a prepared LU solve, or this entry point's refusal for one.
fn lu_solve_operands<'a, T: TensorScalar>(
    lu: &'a Tensor,
    pivots: &'a Tensor,
    rhs: &'a Tensor,
) -> Result<(&'a TypedTensor<T>, &'a TypedTensor<i32>, &'a TypedTensor<T>)> {
    let inconsistent = || {
        Error::Internal(
            "lu_solve_prepared: packed LU, pivots, and rhs dtypes are inconsistent".into(),
        )
    };
    let lu_t = lu.as_typed::<T>().ok_or_else(inconsistent)?;
    let pivots_t = pivots.as_typed::<i32>().ok_or_else(inconsistent)?;
    let rhs_t = rhs.as_typed::<T>().ok_or_else(inconsistent)?;
    Ok((lu_t, pivots_t, rhs_t))
}

pub(super) fn lu_solve_prepared(
    backend: &mut CudaEngine,
    a: &Tensor,
    packed_lu: &Tensor,
    pivots: &Tensor,
    b: &Tensor,
    transpose_a: bool,
    conjugate_a: bool,
) -> Result<Tensor> {
    const OP: &str = "lu_solve_prepared";

    ensure_device_resident_tensor(OP, a)?;
    ensure_device_resident_tensor(OP, packed_lu)?;
    ensure_device_resident_tensor(OP, pivots)?;
    ensure_device_resident_tensor(OP, b)?;
    ensure_supported_linalg_pair(OP, a, b)?;
    ensure_supported_linalg_pair(OP, a, packed_lu)?;
    if !matches!(pivots.dtype(), DType::I32) {
        return Err(Error::dtype_mismatch(OP, DType::I32, pivots.dtype()));
    }
    if has_zero_dim(a.shape()) || has_zero_dim(b.shape()) {
        return zero_like_linalg_device_tensor(backend, b, OP);
    }

    let (rhs, restore_shape) = if let Some(matrix_rhs_shape) = batched_vector_rhs_shape(a, b) {
        (
            backend.reshape_read(TensorRead::from_tensor(b), &matrix_rhs_shape)?,
            Some(b.shape().to_vec()),
        )
    } else {
        (
            backend.to_contiguous_read(TensorRead::from_tensor(b))?,
            None,
        )
    };

    let result = match (packed_lu.dtype(), pivots.dtype(), rhs.dtype()) {
        (DType::F32, DType::I32, DType::F32) => {
            let (lu, pivots, rhs) = lu_solve_operands::<f32>(packed_lu, pivots, &rhs)?;
            lu_solve_prepared_typed(backend, lu, pivots, rhs, transpose_a, conjugate_a)
                .map(Tensor::from_typed::<f32>)
        }
        (DType::F64, DType::I32, DType::F64) => {
            let (lu, pivots, rhs) = lu_solve_operands::<f64>(packed_lu, pivots, &rhs)?;
            lu_solve_prepared_typed(backend, lu, pivots, rhs, transpose_a, conjugate_a)
                .map(Tensor::from_typed::<f64>)
        }
        (DType::C32, DType::I32, DType::C32) => {
            let (lu, pivots, rhs) =
                lu_solve_operands::<num_complex::Complex32>(packed_lu, pivots, &rhs)?;
            lu_solve_prepared_typed(backend, lu, pivots, rhs, transpose_a, conjugate_a)
                .map(Tensor::from_typed::<num_complex::Complex32>)
        }
        (DType::C64, DType::I32, DType::C64) => {
            let (lu, pivots, rhs) =
                lu_solve_operands::<num_complex::Complex64>(packed_lu, pivots, &rhs)?;
            lu_solve_prepared_typed(backend, lu, pivots, rhs, transpose_a, conjugate_a)
                .map(Tensor::from_typed::<num_complex::Complex64>)
        }
        _ => Err(Error::Internal(
            "lu_solve_prepared: packed LU, pivots, and rhs dtypes are inconsistent".into(),
        )),
    }?;

    if let Some(shape) = restore_shape {
        backend.reshape_read(TensorRead::from_tensor(&result), &shape)
    } else {
        Ok(result)
    }
}

fn cholesky_typed<T>(backend: &mut CudaEngine, input: &TypedTensor<T>) -> Result<TypedTensor<T>>
where
    T: LinalgScalar + TensorScalar,
{
    const OP: &str = "cholesky";

    let n = square_matrix_dim(OP, input.shape())?;
    if has_zero_dim(input.shape()) {
        return backend.with_raw(OP, |raw| {
            // The fast path still validates residency before allocating the
            // empty output: `raw.tensor` checks the tensor is resident on this
            // runtime/device.
            raw.tensor(input)?;
            raw.alloc_output(input.shape())
        });
    }

    // All CUDA work for this operation happens inside a single raw session:
    // the tenferro primary context is current, library handles are acquired
    // once per runtime, and the captured stream orders every enqueued call.
    let work = backend.with_raw(OP, |raw| {
        let handles = raw.resource(CudaLinalgHandles::load)?;
        // SAFETY: the stream handle is valid only for this raw-session scope;
        // it is used immediately to bind the cuSOLVER handle and not retained.
        let stream = raw.stream().raw_handle() as usize as CudaStream;
        handles.cusolver().set_stream(stream, OP)?;

        // Clone `input` into a fresh work matrix on the session stream.
        let work = raw.alloc_output::<T>(input.shape())?;
        {
            let src = raw.tensor(input)?;
            let dst = raw.tensor_mut(&work)?;
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
                first_ref.raw_ptr(),
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

        let info = raw.alloc_output::<i32>(&[batch_total])?;
        let info_ref = raw.tensor_mut(&info)?;
        // SAFETY: `info_ref` is a validated device span on this runtime; the pointer is used only within the raw-session scope.
        let info_ptr = info_ref.raw_ptr();
        let matrix_stride = checked_mul_usize(OP, "cholesky matrix stride", n, n)?;

        let first_ref = raw.tensor_mut(&work)?;
        // SAFETY: `first_ref` is a validated device span on this runtime; the pointer is used only within the raw-session scope.
        let first_ptr = first_ref.raw_ptr();
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
                )
            }?;
        }

        // Host barrier (only for reading the solver diagnostics).
        let host_info = raw.download_tensor::<i32>(&info, OP)?;
        for &value in host_info.host_data()? {
            check_solver_info(OP, "cusolverDn*potrf", value)?;
        }
        Ok(work)
    })?;

    backend
        .linalg_triangular(&work, 0, true, "potrf")?
        .into_typed::<T>()
        .map_err(|_| Error::Internal("cholesky triangle changed dtype".into()))
}

fn triangular_solve_typed<T>(
    backend: &mut CudaEngine,
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

#[allow(clippy::too_many_arguments)]
fn triangular_solve_typed_with_op<T>(
    backend: &mut CudaEngine,
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
    ensure_device_resident_typed(op, a)?;
    ensure_device_resident_typed(op, b)?;
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
        let stream = raw.stream().raw_handle() as usize as CudaStream;
        handles.cublas().set_stream(stream, op)?;

        // The fast path still validates residency before allocating the empty
        // output: `raw.tensor` checks the tensors are resident on this
        // runtime/device.
        raw.tensor(a)?;
        raw.tensor(b)?;
        if zero_dim {
            return raw.alloc_output::<T>(b.shape());
        }

        // Clone `b` into a fresh output tensor on the session stream.
        let out = raw.alloc_output::<T>(b.shape())?;
        {
            let src = raw.tensor(b)?;
            let dst = raw.tensor_mut(&out)?;
            // SAFETY: both spans were validated resident on this runtime and
            // the copy is stream-ordered; `dst` is exclusively borrowed.
            unsafe {
                raw.copy_bytes(dst.raw_ptr(), src.raw_ptr() as *const _, src.byte_len(), op)?
            };
        }

        let a_ref = raw.tensor(a)?;
        let out_ref = raw.tensor_mut(&out)?;
        // SAFETY: `a_ref` is a validated device span on this runtime; the pointer is used only within the raw-session scope.
        let a_ptr = a_ref.raw_ptr();
        // SAFETY: `out_ref` is a validated device span on this runtime; the pointer is used only within the raw-session scope.
        let out_ptr = out_ref.raw_ptr();

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

#[allow(clippy::type_complexity)]
fn lu_typed<T>(
    backend: &mut CudaEngine,
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

    let (packed_lu, pivots, parity) = lu_factor_typed(backend, input, OP)?;
    let (m, n) = matrix_dims(OP, input.shape())?;
    let (p, l, u, _extracted_parity) =
        build_lu_outputs_device(backend, &packed_lu, &pivots, m, n, &input.shape()[2..])?;
    Ok((p, l, u, parity))
}

fn lu_factor_typed<T>(
    backend: &mut CudaEngine,
    input: &TypedTensor<T>,
    op: &'static str,
) -> Result<(TypedTensor<T>, TypedTensor<i32>, TypedTensor<T>)>
where
    T: LinalgScalar + TensorScalar,
{
    let (packed_lu, pivots) = lu_factor_core(backend, input, op, true)?;
    let (m, n) = matrix_dims(op, input.shape())?;
    let parity = if has_zero_dim(input.shape()) {
        fill_one_device_tensor(backend, &input.shape()[2..], "lu_factor")?
    } else {
        build_lu_parity_device(backend, &pivots, m.min(n), &input.shape()[2..])?
    };
    Ok((packed_lu, pivots, parity))
}

fn lu_factor_for_solve_typed<T>(
    backend: &mut CudaEngine,
    input: &TypedTensor<T>,
    op: &'static str,
) -> Result<(TypedTensor<T>, TypedTensor<i32>)>
where
    T: LinalgScalar + TensorScalar,
{
    lu_factor_core(backend, input, op, false)
}

/// The shared `getrf` factorization.
///
/// `check_status` downloads cuSOLVER's per-batch `info` and reports a zero
/// pivot as `Singular`; the forward solve passes `false` and issues no host
/// round trip.
fn lu_factor_core<T>(
    backend: &mut CudaEngine,
    input: &TypedTensor<T>,
    op: &'static str,
    check_status: bool,
) -> Result<(TypedTensor<T>, TypedTensor<i32>)>
where
    T: LinalgScalar + TensorScalar,
{
    ensure_device_resident_typed(op, input)?;
    let (m, n) = matrix_dims(op, input.shape())?;
    let k = m.min(n);
    if has_zero_dim(input.shape()) {
        return zero_sized_lu_factor_pair(backend, input.shape());
    }

    let mut pivot_shape = vec![k];
    pivot_shape.extend_from_slice(&input.shape()[2..]);
    let batch_total = batch_count(op, &input.shape()[2..])?;
    let lda = as_i32(m, op, "lda")?;
    let m_i32 = as_i32(m, op, "m")?;
    let n_i32 = as_i32(n, op, "n")?;
    let matrix_stride = checked_mul_usize(op, "lu matrix stride", m, n)?;

    backend.with_raw(op, |raw| {
        let handles = raw.resource(CudaLinalgHandles::load)?;
        // SAFETY: the stream handle is valid only for this raw-session scope;
        // it is used immediately to bind the cuSOLVER handle and not retained.
        let stream = raw.stream().raw_handle() as usize as CudaStream;
        handles.cusolver().set_stream(stream, op)?;

        // Clone `input` into a fresh work matrix on the session stream.
        let work = raw.alloc_output::<T>(input.shape())?;
        {
            let src = raw.tensor(input)?;
            let dst = raw.tensor_mut(&work)?;
            // SAFETY: both spans were validated resident on this runtime and
            // the copy is stream-ordered; `dst` is exclusively borrowed.
            unsafe {
                raw.copy_bytes(dst.raw_ptr(), src.raw_ptr() as *const _, src.byte_len(), op)?
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
                a_ref.raw_ptr(),
                lda,
                op,
            )?
        };
        let workspace_nbytes = {
            let lwork = usize::try_from(lwork).map_err(|_| {
                Error::invalid_argument(
                    op,
                    "workspace_length",
                    format!("must be non-negative, got {lwork}"),
                )
            })?;
            lwork.checked_mul(std::mem::size_of::<T>()).ok_or_else(|| {
                Error::invalid_argument(op, "workspace_length", "byte size overflowed")
            })?
        };
        let workspace = raw.alloc_bytes(workspace_nbytes, op)?;
        let mut workspace_ptr = std::ptr::null_mut::<c_void>();
        workspace.with_ptr(|ptr| workspace_ptr = ptr);

        let pivots = raw.alloc_output::<i32>(&pivot_shape)?;
        let info = raw.alloc_output::<i32>(&[batch_total])?;
        let a_ref = raw.tensor_mut(&work)?;
        // SAFETY: `a_ref` is a validated device span on this runtime; the pointer is used only within the raw-session scope.
        let a_ptr = a_ref.raw_ptr();
        let pivots_ref = raw.tensor_mut(&pivots)?;
        // SAFETY: `pivots_ref` is a validated device span on this runtime; the pointer is used only within the raw-session scope.
        let pivots_ptr = pivots_ref.raw_ptr();
        let info_ref = raw.tensor_mut(&info)?;
        // SAFETY: `info_ref` is a validated device span on this runtime; the pointer is used only within the raw-session scope.
        let info_ptr = info_ref.raw_ptr();

        for batch in 0..batch_total {
            let a_offset =
                checked_batch_offset(op, "lu matrix batch offset", batch, matrix_stride)?;
            let pivot_offset = checked_batch_offset(op, "lu pivot batch offset", batch, k)?;
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
                    op,
                )
            }?;
        }

        if check_status {
            // Host barrier (only when the caller needs the solver diagnostics).
            let host_info = raw.download_tensor::<i32>(&info, op)?;
            for &info_value in host_info.host_data()? {
                if info_value < 0 {
                    return Err(Error::invalid_argument(
                        op,
                        "cusolver_argument",
                        format!(
                            "cusolverDn*getrf reported invalid parameter {}",
                            -info_value
                        ),
                    ));
                }
                if info_value > 0 {
                    // cuSOLVER reports the first zero pivot here: singular matrix.
                    return Err(crate::error::into_tensor_error(
                        op,
                        crate::Error::Singular { op },
                    ));
                }
            }
        }
        Ok((work, pivots))
    })
}

fn lu_solve_prepared_typed<T>(
    backend: &mut CudaEngine,
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

    ensure_device_resident_typed(OP, packed_lu)?;
    ensure_device_resident_typed(OP, pivots)?;
    ensure_device_resident_typed(OP, b)?;
    validate_lu_solve_prepared_shapes(packed_lu.shape(), pivots.shape(), b.shape())?;
    if has_zero_dim(packed_lu.shape()) || has_zero_dim(b.shape()) {
        return backend.with_raw(OP, |raw| {
            // The fast path still validates residency before allocating the
            // empty output: `raw.tensor` checks both tensors are resident.
            raw.tensor(packed_lu)?;
            raw.tensor(b)?;
            raw.alloc_output::<T>(b.shape())
        });
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
    backend: &mut CudaEngine,
    v: &TypedTensor<T>,
    vt_shape: &[usize],
    op: &'static str,
) -> Result<TypedTensor<T>>
where
    T: LinalgScalar + TensorScalar,
{
    let vt = backend.alloc_output(T::dtype(), vt_shape)?;
    backend.linalg_matrix_adjoint(v, &vt, op)?;
    vt.into_typed::<T>()
        .map_err(|_| Error::Internal("matrix adjoint changed dtype".into()))
}

fn copy_matrix_adjoint_complex<T>(
    backend: &mut CudaEngine,
    v: &TypedTensor<T>,
    vt_shape: &[usize],
    op: &'static str,
) -> Result<TypedTensor<T>>
where
    T: LinalgScalar + TensorScalar,
{
    let vt = backend.alloc_output(T::dtype(), vt_shape)?;
    backend.linalg_matrix_adjoint(v, &vt, op)?;
    vt.into_typed::<T>()
        .map_err(|_| Error::Internal("matrix adjoint changed dtype".into()))
}

fn clone_device_tensor<T>(
    backend: &mut CudaEngine,
    input: &TypedTensor<T>,
    op: &'static str,
) -> Result<TypedTensor<T>>
where
    T: LinalgScalar + TensorScalar,
{
    backend.with_raw(op, |raw| {
        let output = raw.alloc_output::<T>(input.shape())?;
        let input_ref = raw.tensor(input)?;
        let output_ref = raw.tensor_mut(&output)?;
        // SAFETY: both spans are validated on this runtime, output is freshly
        // allocated and exclusively borrowed, and the copy is stream ordered.
        unsafe {
            raw.copy_bytes(
                output_ref.raw_ptr(),
                input_ref.raw_ptr() as *const _,
                input_ref.byte_len(),
                op,
            )?
        };

        Ok(output)
    })
}

fn apply_positive_qr_gauge_real<T>(
    backend: &mut CudaEngine,
    q: &mut TypedTensor<T>,
    r: &mut TypedTensor<T>,
    q_start: usize,
    op: &'static str,
) -> Result<()>
where
    T: LinalgScalar + TensorScalar + PartialOrd + std::ops::Mul<Output = T>,
{
    let mut phase_shape = vec![r.shape()[0]];
    phase_shape.extend_from_slice(&r.shape()[2..]);
    // An empty Q has no column to phase: the factor the caller keeps here is scratch.
    if q.n_elements() != 0 && r.n_elements() != 0 {
        let phase = backend.alloc_output(T::dtype(), &phase_shape)?;
        backend.linalg_qr_phase(r, &phase, op)?;
        backend.linalg_qr_apply_phase(q, r, &phase, q_start as i64, op)?;
    }
    Ok(())
}

fn apply_positive_qr_gauge_c32(
    backend: &mut CudaEngine,
    q: &mut TypedTensor<Complex32>,
    r: &mut TypedTensor<Complex32>,
    q_start: usize,
    op: &'static str,
) -> Result<()> {
    let mut phase_shape = vec![r.shape()[0]];
    phase_shape.extend_from_slice(&r.shape()[2..]);
    // An empty Q has no column to phase: the factor the caller keeps here is scratch.
    if q.n_elements() != 0 && r.n_elements() != 0 {
        let phase = backend.alloc_output(DType::C32, &phase_shape)?;
        backend.linalg_qr_phase(r, &phase, op)?;
        backend.linalg_qr_apply_phase(q, r, &phase, q_start as i64, op)?;
    }
    Ok(())
}

fn apply_positive_qr_gauge_c64(
    backend: &mut CudaEngine,
    q: &mut TypedTensor<Complex64>,
    r: &mut TypedTensor<Complex64>,
    q_start: usize,
    op: &'static str,
) -> Result<()> {
    let mut phase_shape = vec![r.shape()[0]];
    phase_shape.extend_from_slice(&r.shape()[2..]);
    // An empty Q has no column to phase: the factor the caller keeps here is scratch.
    if q.n_elements() != 0 && r.n_elements() != 0 {
        let phase = backend.alloc_output(DType::C64, &phase_shape)?;
        backend.linalg_qr_phase(r, &phase, op)?;
        backend.linalg_qr_apply_phase(q, r, &phase, q_start as i64, op)?;
    }
    Ok(())
}

#[allow(clippy::type_complexity)]
fn svd_typed<T>(
    backend: &mut CudaEngine,
    input: &TypedTensor<T>,
    full: bool,
    driver: SvdDriver,
) -> Result<(
    TypedTensor<T>,
    TypedTensor<<T as LinalgScalar>::Real>,
    TypedTensor<T>,
)>
where
    T: LinalgScalar + TensorScalar,
{
    let op: &'static str = if full { "svd_full" } else { "svd" };

    ensure_device_resident_typed(op, input)?;
    let (m, n) = matrix_dims(op, input.shape())?;
    let k = m.min(n);
    // The full variant keeps square factors: `U` is `m x m` and `Vt` is
    // `n x n`, so the trailing columns and rows span the left and right
    // nullspaces the thin decomposition cannot represent.
    let u_cols = if full { m } else { k };
    let vt_rows = if full { n } else { k };
    let batch_shape = &input.shape()[2..];
    let mut u_shape = vec![m, u_cols];
    u_shape.extend_from_slice(batch_shape);
    let mut s_shape = vec![k];
    s_shape.extend_from_slice(batch_shape);
    let mut vt_shape = vec![vt_rows, n];
    vt_shape.extend_from_slice(batch_shape);
    if has_zero_dim(input.shape()) {
        return empty_svd_outputs(backend, input, full, &u_shape, &s_shape, &vt_shape, op);
    }
    let batch_total = batch_count(op, batch_shape)?;
    let a_stride = checked_mul_usize(op, "svd input stride", m, n)?;
    let s_stride = k;

    match select_svd_driver(driver, m, n) {
        CusolverSvdRoutine::Xgesvdp => {
            let (u, s, v) = svd_polar::svd_polar_typed(backend, input, full, true, op)?;
            let vt = T::copy_matrix_adjoint(backend, &v, &vt_shape, op)?;
            Ok((u, s, vt))
        }
        CusolverSvdRoutine::Gesvdj => {
            let mut v_shape = vec![n, vt_rows];
            v_shape.extend_from_slice(batch_shape);
            // `econ = 0` asks gesvdj for the full square factors; `econ = 1`
            // is the economy (thin) form. cuSOLVER places no `m`/`n` ordering
            // constraint on gesvdj, so no transpose trick is needed here.
            let econ = if full { 0 } else { 1 };
            let (u, s, v) = backend.with_raw(op, |raw| {
                let handles = raw.resource(CudaLinalgHandles::load)?;
                // SAFETY: the stream handle is valid only for this raw-session
                // scope; it is used immediately to bind cuSOLVER and not retained.
                let stream = raw.stream().raw_handle() as usize as CudaStream;
                handles.cusolver().set_stream(stream, op)?;

                // Clone `input` into a fresh work matrix on the session stream.
                let work = raw.alloc_output::<T>(input.shape())?;
                {
                    let src = raw.tensor(input)?;
                    let dst = raw.tensor_mut(&work)?;
                    // SAFETY: both spans were validated resident on this runtime
                    // and the copy is stream-ordered; `dst` is exclusively borrowed.
                    unsafe {
                        raw.copy_bytes(
                            dst.raw_ptr(),
                            src.raw_ptr() as *const _,
                            src.byte_len(),
                            op,
                        )?
                    };
                }

                let u = raw.alloc_output::<T>(&u_shape)?;
                let v = raw.alloc_output::<T>(&v_shape)?;
                let s = raw.alloc_output::<<T as LinalgScalar>::Real>(&s_shape)?;

                let m_i32 = as_i32(m, op, "m")?;
                let n_i32 = as_i32(n, op, "n")?;
                let lda = as_i32(m, op, "lda")?;
                let ldu = as_i32(m, op, "ldu")?;
                let ldv = as_i32(n, op, "ldv")?;
                let params = handles.cusolver().create_gesvdj_info(op)?;
                let lwork = {
                    let a_ref = raw.tensor(&work)?;
                    let u_ref = raw.tensor(&u)?;
                    let s_ref = raw.tensor(&s)?;
                    let v_ref = raw.tensor(&v)?;
                    handles.cusolver().gesvdj_buffer_size(
                        T::DATA_TYPE,
                        CusolverEigMode::Vector,
                        econ,
                        m_i32,
                        n_i32,
                        // SAFETY: all spans are validated device allocations on
                        // this runtime; only leading dimensions are queried here.
                        a_ref.raw_ptr().cast_const(),
                        lda,
                        // SAFETY: `s_ref` is a validated device span on this runtime; the pointer is used only within the raw-session scope.
                        s_ref.raw_ptr().cast_const(),
                        // SAFETY: `u_ref` is a validated device span on this runtime; the pointer is used only within the raw-session scope.
                        u_ref.raw_ptr().cast_const(),
                        ldu,
                        // SAFETY: `v_ref` is a validated device span on this runtime; the pointer is used only within the raw-session scope.
                        v_ref.raw_ptr().cast_const(),
                        ldv,
                        &params,
                        op,
                    )?
                };
                let workspace_nbytes = {
                    let lwork = usize::try_from(lwork).map_err(|_| {
                        Error::invalid_argument(
                            op,
                            "workspace_length",
                            format!("must be non-negative, got {lwork}"),
                        )
                    })?;
                    lwork.checked_mul(std::mem::size_of::<T>()).ok_or_else(|| {
                        Error::invalid_argument(op, "workspace_length", "byte size overflowed")
                    })?
                };
                let workspace = raw.alloc_bytes(workspace_nbytes, op)?;
                let mut workspace_ptr = std::ptr::null_mut::<c_void>();
                workspace.with_ptr(|ptr| workspace_ptr = ptr);

                let info = raw.alloc_output::<i32>(&[batch_total])?;
                let a_ref = raw.tensor_mut(&work)?;
                // SAFETY: `a_ref` is a validated device span on this runtime; the pointer is used only within the raw-session scope.
                let a_ptr = a_ref.raw_ptr();
                let u_ref = raw.tensor_mut(&u)?;
                // SAFETY: `u_ref` is a validated device span on this runtime; the pointer is used only within the raw-session scope.
                let u_ptr = u_ref.raw_ptr();
                let s_ref = raw.tensor_mut(&s)?;
                // SAFETY: `s_ref` is a validated device span on this runtime; the pointer is used only within the raw-session scope.
                let s_ptr = s_ref.raw_ptr();
                let v_ref = raw.tensor_mut(&v)?;
                // SAFETY: `v_ref` is a validated device span on this runtime; the pointer is used only within the raw-session scope.
                let v_ptr = v_ref.raw_ptr();
                let info_ref = raw.tensor_mut(&info)?;
                // SAFETY: `info_ref` is a validated device span on this runtime; the pointer is used only within the raw-session scope.
                let info_ptr = info_ref.raw_ptr();
                let u_stride = checked_mul_usize(op, "svd u stride", m, u_cols)?;
                let v_stride = checked_mul_usize(op, "svd v stride", n, vt_rows)?;

                for batch in 0..batch_total {
                    let a_offset =
                        checked_batch_offset(op, "svd input batch offset", batch, a_stride)?;
                    let s_offset = checked_batch_offset(
                        op,
                        "svd singular value batch offset",
                        batch,
                        s_stride,
                    )?;
                    let u_offset = checked_batch_offset(op, "svd u batch offset", batch, u_stride)?;
                    let v_offset = checked_batch_offset(op, "svd v batch offset", batch, v_stride)?;
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
                            econ,
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
                            op,
                        )
                    }?;
                }

                // Host barrier (only for reading the solver diagnostics).
                let host_info = raw.download_tensor::<i32>(&info, op)?;
                for &value in host_info.host_data()? {
                    check_solver_info(op, "cusolverDn*gesvdj", value)?;
                }
                Ok((u, s, v))
            })?;
            let vt = T::copy_matrix_adjoint(backend, &v, &vt_shape, op)?;
            Ok((u, s, vt))
        }
        CusolverSvdRoutine::Gesvd => {
            let transpose_for_gesvd = m < n;
            let (gesvd_m, gesvd_n) = if transpose_for_gesvd { (n, m) } else { (m, n) };
            // `jobu = jobvt = 'A'` returns `gesvd_m x gesvd_m` and
            // `gesvd_n x gesvd_n` factors. With the wide-matrix adjoint trick
            // those become `Vt` and `U` respectively after the final adjoint,
            // which is exactly the full contract for the original orientation.
            let gesvd_u_cols = if full { gesvd_m } else { k };
            let gesvd_vt_rows = if full { gesvd_n } else { k };
            let mut gesvd_u_shape = vec![gesvd_m, gesvd_u_cols];
            gesvd_u_shape.extend_from_slice(batch_shape);
            let mut gesvd_vt_shape = vec![gesvd_vt_rows, gesvd_n];
            gesvd_vt_shape.extend_from_slice(batch_shape);

            // When `m < n` gesvd factors `adjoint(input)` (n×m); the CubeCL
            // adjoint kernel is flushed before the raw session, so the SVD
            // below observes it. The SVD results are adjointed again below.
            let transposed_work = if transpose_for_gesvd {
                let mut work_shape = vec![n, m];
                work_shape.extend_from_slice(batch_shape);
                Some(T::copy_matrix_adjoint(backend, input, &work_shape, op)?)
            } else {
                None
            };

            let (gesvd_u, s, gesvd_vt) = backend.with_raw(op, |raw| {
                let handles = raw.resource(CudaLinalgHandles::load)?;
                // SAFETY: the stream handle is valid only for this raw-session
                // scope; it is used immediately to bind cuSOLVER and not retained.
                let stream = raw.stream().raw_handle() as usize as CudaStream;
                handles.cusolver().set_stream(stream, op)?;

                // `work` is either the CubeCL-adjointed input (m < n) or a
                // clone of `input` made on the session stream.
                let work = match transposed_work {
                    Some(work) => work,
                    None => {
                        let work = raw.alloc_output::<T>(input.shape())?;
                        {
                            let src = raw.tensor(input)?;
                            let dst = raw.tensor_mut(&work)?;
                            // SAFETY: both spans were validated resident on this
                            // runtime and the copy is stream-ordered; `dst` is
                            // exclusively borrowed.
                            unsafe {
                                raw.copy_bytes(
                                    dst.raw_ptr(),
                                    src.raw_ptr() as *const _,
                                    src.byte_len(),
                                    op,
                                )?
                            };
                        }
                        work
                    }
                };
                let s = raw.alloc_output::<<T as LinalgScalar>::Real>(&s_shape)?;
                let gesvd_u = raw.alloc_output::<T>(&gesvd_u_shape)?;
                let gesvd_vt = raw.alloc_output::<T>(&gesvd_vt_shape)?;

                let a_ref = raw.tensor_mut(&work)?;
                // SAFETY: `a_ref` is a validated device span on this runtime; the pointer is used only within the raw-session scope.
                let a_ptr = a_ref.raw_ptr();
                let u_ref = raw.tensor_mut(&gesvd_u)?;
                // SAFETY: `u_ref` is a validated device span on this runtime; the pointer is used only within the raw-session scope.
                let u_ptr = u_ref.raw_ptr();
                let s_ref = raw.tensor_mut(&s)?;
                // SAFETY: `s_ref` is a validated device span on this runtime; the pointer is used only within the raw-session scope.
                let s_ptr = s_ref.raw_ptr();
                let vt_ref = raw.tensor_mut(&gesvd_vt)?;
                // SAFETY: `vt_ref` is a validated device span on this runtime; the pointer is used only within the raw-session scope.
                let vt_ptr = vt_ref.raw_ptr();
                let gesvd_m_i32 = as_i32(gesvd_m, op, "gesvd m")?;
                let gesvd_n_i32 = as_i32(gesvd_n, op, "gesvd n")?;
                let lda = gesvd_m_i32;
                let ldu = gesvd_m_i32;
                let ldvt = as_i32(gesvd_vt_rows, op, "ldvt")?;
                let lwork = handles.cusolver().gesvd_buffer_size(
                    T::DATA_TYPE,
                    gesvd_m_i32,
                    gesvd_n_i32,
                    op,
                )?;
                let workspace_nbytes = {
                    let lwork = usize::try_from(lwork).map_err(|_| {
                        Error::invalid_argument(
                            op,
                            "workspace_length",
                            format!("must be non-negative, got {lwork}"),
                        )
                    })?;
                    lwork.checked_mul(std::mem::size_of::<T>()).ok_or_else(|| {
                        Error::invalid_argument(op, "workspace_length", "byte size overflowed")
                    })?
                };
                let workspace = raw.alloc_bytes(workspace_nbytes, op)?;
                let mut workspace_ptr = std::ptr::null_mut::<c_void>();
                workspace.with_ptr(|ptr| workspace_ptr = ptr);
                let mut rwork_ptr = std::ptr::null_mut::<c_void>();
                if T::NEEDS_RWORK {
                    let rwork_len = as_i32(
                        checked_mul_usize(op, "svd rwork length", 5, k)?,
                        op,
                        "rwork",
                    )?;
                    let rwork_nbytes = usize::try_from(rwork_len)
                        .map_err(|_| {
                            Error::invalid_argument(op, "rwork_length", "must be non-negative")
                        })?
                        .checked_mul(std::mem::size_of::<<T as LinalgScalar>::Real>())
                        .ok_or_else(|| {
                            Error::invalid_argument(op, "rwork_length", "byte size overflowed")
                        })?;
                    let rwork = raw.alloc_bytes(rwork_nbytes, op)?;
                    rwork.with_ptr(|ptr| rwork_ptr = ptr);
                }
                let info = raw.alloc_output::<i32>(&[batch_total])?;
                let info_ref = raw.tensor_mut(&info)?;
                // SAFETY: `info_ref` is a validated device span on this runtime; the pointer is used only within the raw-session scope.
                let info_ptr = info_ref.raw_ptr();
                let u_stride = checked_mul_usize(op, "svd gesvd u stride", gesvd_m, gesvd_u_cols)?;
                let vt_stride =
                    checked_mul_usize(op, "svd gesvd vt stride", gesvd_vt_rows, gesvd_n)?;
                let job = if full { b'A' } else { b'S' } as c_char;

                for batch in 0..batch_total {
                    let a_offset =
                        checked_batch_offset(op, "svd input batch offset", batch, a_stride)?;
                    let s_offset = checked_batch_offset(
                        op,
                        "svd singular value batch offset",
                        batch,
                        s_stride,
                    )?;
                    let u_offset = checked_batch_offset(op, "svd u batch offset", batch, u_stride)?;
                    let vt_offset =
                        checked_batch_offset(op, "svd vt batch offset", batch, vt_stride)?;
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
                    // stream-bound handle satisfy cuSOLVER gesvd's SVD contract
                    // for the selected `job` mode.
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
                            op,
                        )
                    }?;
                }

                // Host barrier (only for reading the solver diagnostics).
                let host_info = raw.download_tensor::<i32>(&info, op)?;
                for &value in host_info.host_data()? {
                    check_solver_info(op, "cusolverDn*gesvd", value)?;
                }
                Ok((gesvd_u, s, gesvd_vt))
            })?;

            if transpose_for_gesvd {
                let u = T::copy_matrix_adjoint(backend, &gesvd_vt, &u_shape, op)?;
                let vt = T::copy_matrix_adjoint(backend, &gesvd_u, &vt_shape, op)?;
                Ok((u, s, vt))
            } else {
                Ok((gesvd_u, s, gesvd_vt))
            }
        }
    }
}

/// SVD factors for a device input with an empty core dimension.
///
/// The thin variant's factors are all empty, so uninitialized allocations are
/// correct there. The full variant is not: `U` stays `m x m` and `Vt` stays
/// `n x n`, so when only the other core dimension is zero the surviving factor
/// still owes a unitary matrix. The identity is the canonical choice — it keeps
/// `U Uᴴ = I` and `Vᴴ V = I` true and reconstructs the (empty) input exactly —
/// and matches the CPU providers, so one public call has one shape and one
/// unitarity contract on every backend.
///
/// The identity is materialized on the host and uploaded. Nothing is read back
/// from the device: the input carries no elements, so this is a constant
/// upload, not a device-to-host-to-device roundtrip of computed values.
#[allow(clippy::type_complexity)]
fn empty_svd_outputs<T>(
    backend: &mut CudaEngine,
    input: &TypedTensor<T>,
    full: bool,
    u_shape: &[usize],
    s_shape: &[usize],
    vt_shape: &[usize],
    op: &'static str,
) -> Result<(
    TypedTensor<T>,
    TypedTensor<<T as LinalgScalar>::Real>,
    TypedTensor<T>,
)>
where
    T: LinalgScalar + TensorScalar,
{
    let (s, allocated) = backend.with_raw(op, |raw| {
        // The fast path still validates residency before allocating the
        // outputs: `raw.tensor` checks the tensor is resident.
        raw.tensor(input)?;
        let s = raw.alloc_output::<<T as LinalgScalar>::Real>(s_shape)?;
        if full {
            return Ok((s, None));
        }
        Ok((
            s,
            Some((
                raw.alloc_output::<T>(u_shape)?,
                raw.alloc_output::<T>(vt_shape)?,
            )),
        ))
    })?;
    if let Some((u, vt)) = allocated {
        return Ok((u, s, vt));
    }
    let u = upload_identity_stack::<T>(backend, u_shape, op)?;
    let vt = upload_identity_stack::<T>(backend, vt_shape, op)?;
    Ok((u, s, vt))
}

/// Upload a stack of column-major identity matrices shaped `[dim, dim, ..batch]`.
fn upload_identity_stack<T>(
    backend: &mut CudaEngine,
    shape: &[usize],
    op: &'static str,
) -> Result<TypedTensor<T>>
where
    T: LinalgScalar + TensorScalar,
{
    let dim = shape[0];
    let blocks = checked_shape_product(op, "identity batch", &shape[2..])?;
    let per_block = checked_mul_usize(op, "identity block", dim, dim)?;
    let len = checked_mul_usize(op, "identity stack", per_block, blocks)?;
    let mut data = vec![T::zero(); len];
    for block in 0..blocks {
        let base = block * per_block;
        for index in 0..dim {
            data[base + index + index * dim] = T::one();
        }
    }
    let host = TypedTensor::<T>::from_vec_col_major(shape.to_vec(), data)?;
    let uploaded =
        backend.upload_host_tensor(TensorRead::from_tensor(&Tensor::from_typed::<T>(host)))?;
    // The upload preserves the host dtype `T`; a refusal keeps its typed cause.
    uploaded
        .into_typed::<T>()
        .map_err(|failure| failure.into_parts().1)
}

fn svd_values_typed<T>(
    backend: &mut CudaEngine,
    input: &TypedTensor<T>,
    driver: SvdDriver,
) -> Result<TypedTensor<<T as LinalgScalar>::Real>>
where
    T: LinalgScalar + TensorScalar,
{
    const OP: &str = "svd_values";

    ensure_device_resident_typed(OP, input)?;
    let (m, n) = matrix_dims(OP, input.shape())?;
    let k = m.min(n);
    let batch_shape = &input.shape()[2..];
    let mut s_shape = vec![k];
    s_shape.extend_from_slice(batch_shape);
    if has_zero_dim(input.shape()) {
        return backend.with_raw(OP, |raw| {
            // The fast path still validates residency before allocating the
            // empty output: `raw.tensor` checks the tensor is resident.
            raw.tensor(input)?;
            raw.alloc_output::<<T as LinalgScalar>::Real>(&s_shape)
        });
    }
    let batch_total = batch_count(OP, batch_shape)?;
    let a_stride = checked_mul_usize(OP, "svd_values input stride", m, n)?;
    let s_stride = k;

    match select_svd_driver(driver, m, n) {
        CusolverSvdRoutine::Xgesvdp => {
            let (_, s, _) = svd_polar::svd_polar_typed(backend, input, false, false, OP)?;
            Ok(s)
        }
        CusolverSvdRoutine::Gesvdj => {
            let mut u_shape = vec![m, k];
            u_shape.extend_from_slice(batch_shape);
            let mut v_shape = vec![n, k];
            v_shape.extend_from_slice(batch_shape);
            backend.with_raw(OP, |raw| {
                let handles = raw.resource(CudaLinalgHandles::load)?;
                // SAFETY: the stream handle is valid only for this raw-session
                // scope; it is used immediately to bind cuSOLVER and not retained.
                let stream = raw.stream().raw_handle() as usize as CudaStream;
                handles.cusolver().set_stream(stream, OP)?;

                // Clone `input` into a fresh work matrix on the session stream.
                let work = raw.alloc_output::<T>(input.shape())?;
                {
                    let src = raw.tensor(input)?;
                    let dst = raw.tensor_mut(&work)?;
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

                let u = raw.alloc_output::<T>(&u_shape)?;
                let v = raw.alloc_output::<T>(&v_shape)?;
                let s = raw.alloc_output::<<T as LinalgScalar>::Real>(&s_shape)?;

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
                        a_ref.raw_ptr().cast_const(),
                        lda,
                        // SAFETY: `s_ref` is a validated device span on this runtime; the pointer is used only within the raw-session scope.
                        s_ref.raw_ptr().cast_const(),
                        // SAFETY: `u_ref` is a validated device span on this runtime; the pointer is used only within the raw-session scope.
                        u_ref.raw_ptr().cast_const(),
                        lda,
                        // SAFETY: `v_ref` is a validated device span on this runtime; the pointer is used only within the raw-session scope.
                        v_ref.raw_ptr().cast_const(),
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

                let info = raw.alloc_output::<i32>(&[batch_total])?;
                let a_ref = raw.tensor_mut(&work)?;
                // SAFETY: `a_ref` is a validated device span on this runtime; the pointer is used only within the raw-session scope.
                let a_ptr = a_ref.raw_ptr();
                let s_ref = raw.tensor_mut(&s)?;
                // SAFETY: `s_ref` is a validated device span on this runtime; the pointer is used only within the raw-session scope.
                let s_ptr = s_ref.raw_ptr();
                let u_ref = raw.tensor_mut(&u)?;
                // SAFETY: `u_ref` is a validated device span on this runtime; the pointer is used only within the raw-session scope.
                let u_ptr = u_ref.raw_ptr();
                let v_ref = raw.tensor_mut(&v)?;
                // SAFETY: `v_ref` is a validated device span on this runtime; the pointer is used only within the raw-session scope.
                let v_ptr = v_ref.raw_ptr();
                let info_ref = raw.tensor_mut(&info)?;
                // SAFETY: `info_ref` is a validated device span on this runtime; the pointer is used only within the raw-session scope.
                let info_ptr = info_ref.raw_ptr();
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
                        )
                    }?;
                }

                // Host barrier (only for reading the solver diagnostics).
                let host_info = raw.download_tensor::<i32>(&info, OP)?;
                for &value in host_info.host_data()? {
                    check_solver_info(OP, "cusolverDn*gesvdj", value)?;
                }
                Ok(s)
            })
        }
        CusolverSvdRoutine::Gesvd => {
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
                let stream = raw.stream().raw_handle() as usize as CudaStream;
                handles.cusolver().set_stream(stream, OP)?;

                // `work` is either the CubeCL-adjointed input (m < n) or a
                // clone of `input` made on the session stream.
                let work = match transposed_work {
                    Some(work) => work,
                    None => {
                        let work = raw.alloc_output::<T>(input.shape())?;
                        {
                            let src = raw.tensor(input)?;
                            let dst = raw.tensor_mut(&work)?;
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
                let s = raw.alloc_output::<<T as LinalgScalar>::Real>(&s_shape)?;

                let a_ref = raw.tensor_mut(&work)?;
                // SAFETY: `a_ref` is a validated device span on this runtime; the pointer is used only within the raw-session scope.
                let a_ptr = a_ref.raw_ptr();
                let s_ref = raw.tensor_mut(&s)?;
                // SAFETY: `s_ref` is a validated device span on this runtime; the pointer is used only within the raw-session scope.
                let s_ptr = s_ref.raw_ptr();
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
                let info = raw.alloc_output::<i32>(&[batch_total])?;
                let info_ref = raw.tensor_mut(&info)?;
                // SAFETY: `info_ref` is a validated device span on this runtime; the pointer is used only within the raw-session scope.
                let info_ptr = info_ref.raw_ptr();
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
mod svd_polar;
#[cfg(test)]
mod tests;
use householder_qr::*;

fn qr_typed<T>(
    backend: &mut CudaEngine,
    input: &TypedTensor<T>,
) -> Result<(TypedTensor<T>, TypedTensor<T>)>
where
    T: LinalgScalar + TensorScalar,
{
    const OP: &str = "qr";

    ensure_device_resident_typed(OP, input)?;
    let (m, n) = matrix_dims(OP, input.shape())?;
    let k = m.min(n);
    let batch_shape = &input.shape()[2..];
    let mut q_shape = vec![m, k];
    q_shape.extend_from_slice(batch_shape);
    let mut r_shape = vec![k, n];
    r_shape.extend_from_slice(batch_shape);
    if has_zero_dim(input.shape()) {
        return backend.with_raw(OP, |raw| {
            // The fast path still validates residency before allocating the
            // empty outputs: `raw.tensor` checks the tensor is resident.
            raw.tensor(input)?;
            Ok((
                raw.alloc_output::<T>(&q_shape)?,
                raw.alloc_output::<T>(&r_shape)?,
            ))
        });
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
        let stream = raw.stream().raw_handle() as usize as CudaStream;
        handles.cusolver().set_stream(stream, OP)?;

        // Clone `input` into a fresh work matrix on the session stream.
        let work = raw.alloc_output::<T>(input.shape())?;
        {
            let src = raw.tensor(input)?;
            let dst = raw.tensor_mut(&work)?;
            // SAFETY: both spans were validated resident on this runtime and
            // the copy is stream-ordered; `dst` is exclusively borrowed.
            unsafe {
                raw.copy_bytes(dst.raw_ptr(), src.raw_ptr() as *const _, src.byte_len(), OP)?
            };
        }
        let q = raw.alloc_output::<T>(&q_shape)?;

        let geqrf_lwork = {
            let work_ref = raw.tensor(&work)?;
            handles.cusolver().geqrf_buffer_size(
                T::DATA_TYPE,
                m_i32,
                n_i32,
                // SAFETY: `work_ref` is a validated device span on this
                // runtime; cuSOLVER only queries the leading dimension here.
                work_ref.raw_ptr(),
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
                q_ref.raw_ptr().cast_const(),
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

        let geqrf_info = raw.alloc_output::<i32>(&[batch_total])?;
        let orgqr_info = raw.alloc_output::<i32>(&[batch_total])?;
        let work_ref = raw.tensor_mut(&work)?;
        // SAFETY: `work_ref` is a validated device span on this runtime; the pointer is used only within the raw-session scope.
        let work_ptr = work_ref.raw_ptr();
        let q_ref = raw.tensor_mut(&q)?;
        // SAFETY: `q_ref` is a validated device span on this runtime; the pointer is used only within the raw-session scope.
        let q_ptr = q_ref.raw_ptr();
        let geqrf_info_ref = raw.tensor_mut(&geqrf_info)?;
        // SAFETY: `geqrf_info_ref` is a validated device span on this runtime; the pointer is used only within the raw-session scope.
        let geqrf_info_ptr = geqrf_info_ref.raw_ptr();
        let orgqr_info_ref = raw.tensor_mut(&orgqr_info)?;
        // SAFETY: `orgqr_info_ref` is a validated device span on this runtime; the pointer is used only within the raw-session scope.
        let orgqr_info_ptr = orgqr_info_ref.raw_ptr();

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
                )
            }?;

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
                )?
            };

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
                )
            }?;
        }

        // The vendor status flags stay on the device: reading them is a host
        // barrier that waits for the factorization kernels, and cuSOLVER's QR
        // flags report neither the illegal arguments (this path validates
        // shape, dtype, and dimensions up front) nor a rank-deficient factor
        // (a rank-deficient input is still a valid QR; the caller's own
        // verification decides whether to accept the factors). This matches
        // the CPU providers and `torch.linalg.qr`.
        Ok((q, work))
    })?;

    let r_input = backend.linalg_slice(&work, &matrix_slice_config(input.shape(), k, n), "qr")?;
    let r = backend
        .linalg_triangular(&r_input, 0, false, "qr")?
        .into_typed::<T>()
        .map_err(|_| Error::Internal("qr: the triangular factor changed dtype".into()))?;
    Ok((q, r))
}

fn eigh_typed<T>(
    backend: &mut CudaEngine,
    input: &TypedTensor<T>,
    driver: EighDriver,
) -> Result<(TypedTensor<<T as LinalgScalar>::Real>, TypedTensor<T>)>
where
    T: LinalgScalar + TensorScalar,
{
    const OP: &str = "eigh";

    ensure_device_resident_typed(OP, input)?;
    let n = square_matrix_dim(OP, input.shape())?;
    let batch_shape = &input.shape()[2..];
    let mut values_shape = vec![n];
    values_shape.extend_from_slice(batch_shape);
    let n_i32 = as_i32(n, OP, "n")?;
    let lda = as_i32(n, OP, "lda")?;
    let batch_total = batch_count(OP, batch_shape)?;
    let matrix_stride = checked_mul_usize(OP, "eigh matrix stride", n, n)?;
    let values_stride = n;
    let routine = select_eigh_driver(driver, batch_total);
    let batch_total_i32 = as_i32(batch_total, OP, "batch_total")?;
    // The 64-bit batched entry point takes i64 dimensions.
    let n_i64 = i64::from(n_i32);
    let lda_i64 = i64::from(lda);
    let batch_total_i64 = i64::from(batch_total_i32);
    if has_zero_dim(input.shape()) {
        return backend.with_raw(OP, |raw| {
            // The fast path still validates residency before allocating the
            // empty outputs: `raw.tensor` checks the tensor is resident.
            raw.tensor(input)?;
            Ok((
                raw.alloc_output::<<T as LinalgScalar>::Real>(&values_shape)?,
                raw.alloc_output::<T>(input.shape())?,
            ))
        });
    }

    let (values, work) = backend.with_raw(OP, |raw| {
        let handles = raw.resource(CudaLinalgHandles::load)?;
        // SAFETY: the stream handle is valid only for this raw-session scope;
        // it is used immediately to bind the cuSOLVER handle and not retained.
        let stream = raw.stream().raw_handle() as usize as CudaStream;
        handles.cusolver().set_stream(stream, OP)?;
        // The Jacobi routine needs a parameter object: it is queried for the
        // workspace size and reused by every call below. It outlives the
        // launches because the solver-status download later in this scope is
        // a host barrier, so the device work has retired before the object
        // drops at the end of the closure. This matches the `gesvdj` path.
        let syevj_params = match routine {
            CusolverEighRoutine::Syevd | CusolverEighRoutine::XsyevBatched => None,
            CusolverEighRoutine::Syevj | CusolverEighRoutine::SyevjBatched => {
                Some(handles.cusolver().create_syevj_info(OP)?)
            }
        };

        // Clone `input` into a fresh work matrix on the session stream.
        let work = raw.alloc_output::<T>(input.shape())?;
        {
            let src = raw.tensor(input)?;
            let dst = raw.tensor_mut(&work)?;
            // SAFETY: both spans were validated resident on this runtime and
            // the copy is stream-ordered; `dst` is exclusively borrowed.
            unsafe {
                raw.copy_bytes(dst.raw_ptr(), src.raw_ptr() as *const _, src.byte_len(), OP)?
            };
        }
        let values = raw.alloc_output::<<T as LinalgScalar>::Real>(&values_shape)?;

        // The batched divide-and-conquer entry point reports byte sizes, so
        // its query is separate from the element-count query below.
        let xsyev_bytes = if routine == CusolverEighRoutine::XsyevBatched {
            let a_ref = raw.tensor(&work)?;
            let values_ref = raw.tensor(&values)?;
            // SAFETY: both spans are validated device allocations on this
            // runtime; only the leading dimensions are queried here.
            let a_ptr = a_ref.raw_ptr().cast_const();
            // SAFETY: `values_ref` is a validated device span on this runtime.
            let values_ptr = values_ref.raw_ptr().cast_const();
            Some(handles.cusolver().xsyev_batched_buffer_size(
                T::DATA_TYPE,
                CusolverEigMode::Vector,
                CublasFillMode::Lower,
                n_i64,
                a_ptr,
                lda_i64,
                values_ptr,
                batch_total_i64,
                OP,
            )?)
        } else {
            None
        };
        let lwork = if routine == CusolverEighRoutine::XsyevBatched {
            0
        } else {
            let a_ref = raw.tensor(&work)?;
            let values_ref = raw.tensor(&values)?;
            // SAFETY: both spans are validated device allocations on this
            // runtime; only the leading dimensions are queried here.
            let a_ptr = a_ref.raw_ptr().cast_const();
            // SAFETY: `values_ref` is a validated device span on this runtime; the pointer is used only within the raw-session scope.
            let values_ptr = values_ref.raw_ptr().cast_const();
            match &syevj_params {
                None => handles.cusolver().syevd_buffer_size(
                    T::DATA_TYPE,
                    CusolverEigMode::Vector,
                    CublasFillMode::Lower,
                    n_i32,
                    a_ptr,
                    lda,
                    values_ptr,
                    OP,
                )?,
                Some(params) if routine == CusolverEighRoutine::SyevjBatched => {
                    handles.cusolver().syevj_batched_buffer_size(
                        T::DATA_TYPE,
                        CusolverEigMode::Vector,
                        CublasFillMode::Lower,
                        n_i32,
                        a_ptr,
                        lda,
                        values_ptr,
                        params,
                        batch_total_i32,
                        OP,
                    )?
                }
                Some(params) => handles.cusolver().syevj_buffer_size(
                    T::DATA_TYPE,
                    CusolverEigMode::Vector,
                    CublasFillMode::Lower,
                    n_i32,
                    a_ptr,
                    lda,
                    values_ptr,
                    params,
                    OP,
                )?,
            }
        };
        // `Xsyev` reports byte sizes for a device and a host workspace; the
        // legacy entry points report an element count and need device scratch
        // only. The host buffer stays uninitialized because cuSOLVER owns it
        // and Rust never reads it, and it outlives the launch because the
        // solver-status download below is a host barrier.
        let (workspace_nbytes, host_nbytes) = match xsyev_bytes {
            Some(bytes) => bytes,
            None => {
                let lwork = usize::try_from(lwork).map_err(|_| {
                    Error::invalid_argument(
                        OP,
                        "workspace_length",
                        format!("must be non-negative, got {lwork}"),
                    )
                })?;
                (
                    lwork.checked_mul(std::mem::size_of::<T>()).ok_or_else(|| {
                        Error::invalid_argument(OP, "workspace_length", "byte size overflowed")
                    })?,
                    0,
                )
            }
        };
        let workspace = raw.alloc_bytes(workspace_nbytes.max(1), OP)?;
        let mut workspace_ptr = std::ptr::null_mut::<c_void>();
        workspace.with_ptr(|ptr| workspace_ptr = ptr);
        let mut host_workspace = Vec::<std::mem::MaybeUninit<u8>>::new();
        host_workspace
            .try_reserve_exact(host_nbytes)
            .map_err(|error| Error::backend_source(OP, error))?;
        let host_ptr = if host_nbytes == 0 {
            std::ptr::null_mut::<c_void>()
        } else {
            host_workspace.as_mut_ptr().cast::<c_void>()
        };

        let info = raw.alloc_output::<i32>(&[batch_total])?;
        let a_ref = raw.tensor_mut(&work)?;
        // SAFETY: `a_ref` is a validated device span on this runtime; the pointer is used only within the raw-session scope.
        let a_ptr = a_ref.raw_ptr();
        let values_ref = raw.tensor_mut(&values)?;
        // SAFETY: `values_ref` is a validated device span on this runtime; the pointer is used only within the raw-session scope.
        let values_ptr = values_ref.raw_ptr();
        let info_ref = raw.tensor_mut(&info)?;
        // SAFETY: `info_ref` is a validated device span on this runtime; the pointer is used only within the raw-session scope.
        let info_ptr = info_ref.raw_ptr();

        if routine == CusolverEighRoutine::SyevjBatched {
            let params = syevj_params.as_ref().ok_or_else(|| {
                Error::runtime_state(
                    OP,
                    "batched Jacobi eigensolver selected without its cuSOLVER parameters",
                )
            })?;
            // SAFETY: the batch-wide matrix, eigenvalue, info, and workspace
            // allocations match the strides and lengths cuSOLVER queried, and
            // `params` is the object that query used.
            unsafe {
                handles.cusolver().syevj_batched(
                    T::DATA_TYPE,
                    CusolverEigMode::Vector,
                    CublasFillMode::Lower,
                    n_i32,
                    a_ptr,
                    lda,
                    values_ptr,
                    workspace_ptr,
                    lwork,
                    info_ptr.cast::<i32>(),
                    params,
                    batch_total_i32,
                    OP,
                )
            }?;
        }
        if routine == CusolverEighRoutine::XsyevBatched {
            // SAFETY: the batch-wide matrix, eigenvalue, and info allocations
            // match the strides cuSOLVER expects, and both workspaces have the
            // queried sizes and stay live until the barrier below.
            unsafe {
                handles.cusolver().xsyev_batched(
                    T::DATA_TYPE,
                    CusolverEigMode::Vector,
                    CublasFillMode::Lower,
                    n_i64,
                    a_ptr,
                    lda_i64,
                    values_ptr,
                    workspace_ptr,
                    workspace_nbytes,
                    host_ptr,
                    host_nbytes,
                    info_ptr.cast::<i32>(),
                    batch_total_i64,
                    OP,
                )
            }?;
        }
        // The batched launch above already covered every matrix.
        let per_matrix_batches = match routine {
            CusolverEighRoutine::SyevjBatched | CusolverEighRoutine::XsyevBatched => 0,
            CusolverEighRoutine::Syevd | CusolverEighRoutine::Syevj => batch_total,
        };
        for batch in 0..per_matrix_batches {
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
            match &syevj_params {
                // SAFETY: batch pointers, workspace, dimensions, and
                // stream-bound handle satisfy cuSOLVER syevd's vector
                // eigensolver contract.
                None => {
                    // SAFETY: the batch pointers, dimensions, workspace, and the stream-bound handle
                    // satisfy cuSOLVER's syevd contract for this arm.
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
                        )
                    }?;
                }
                // SAFETY: as above, plus `params` is the same object the
                // workspace size was queried with.
                Some(params) => {
                    unsafe {
                        handles.cusolver().syevj(
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
                            params,
                            OP,
                        )
                    }?;
                }
            }
        }

        // Host barrier (only for reading the solver diagnostics).
        let call = match routine {
            CusolverEighRoutine::Syevd => "cusolverDn*syevd",
            CusolverEighRoutine::Syevj => "cusolverDn*syevj",
            CusolverEighRoutine::SyevjBatched => "cusolverDn*syevjBatched",
            CusolverEighRoutine::XsyevBatched => "cusolverDnXsyevBatched",
        };
        let host_info = raw.download_tensor::<i32>(&info, OP)?;
        for &value in host_info.host_data()? {
            check_solver_info(OP, call, value)?;
        }
        Ok((values, work))
    })?;

    Ok((values, work))
}

fn eigh_values_typed<T>(
    backend: &mut CudaEngine,
    input: &TypedTensor<T>,
    driver: EighDriver,
) -> Result<TypedTensor<<T as LinalgScalar>::Real>>
where
    T: LinalgScalar + TensorScalar,
{
    const OP: &str = "eigh_values";

    ensure_device_resident_typed(OP, input)?;
    let n = square_matrix_dim(OP, input.shape())?;
    let batch_shape = &input.shape()[2..];
    let mut values_shape = vec![n];
    values_shape.extend_from_slice(batch_shape);
    let n_i32 = as_i32(n, OP, "n")?;
    let lda = as_i32(n, OP, "lda")?;
    let batch_total = batch_count(OP, batch_shape)?;
    let matrix_stride = checked_mul_usize(OP, "eigh_values matrix stride", n, n)?;
    let values_stride = n;
    let routine = select_eigh_driver(driver, batch_total);
    let batch_total_i32 = as_i32(batch_total, OP, "batch_total")?;
    // The 64-bit batched entry point takes i64 dimensions.
    let n_i64 = i64::from(n_i32);
    let lda_i64 = i64::from(lda);
    let batch_total_i64 = i64::from(batch_total_i32);
    if has_zero_dim(input.shape()) {
        return backend.with_raw(OP, |raw| {
            // The fast path still validates residency before allocating the
            // empty output: `raw.tensor` checks the tensor is resident.
            raw.tensor(input)?;
            raw.alloc_output::<<T as LinalgScalar>::Real>(&values_shape)
        });
    }

    let values = backend.with_raw(OP, |raw| {
        let handles = raw.resource(CudaLinalgHandles::load)?;
        // SAFETY: the stream handle is valid only for this raw-session scope;
        // it is used immediately to bind the cuSOLVER handle and not retained.
        let stream = raw.stream().raw_handle() as usize as CudaStream;
        handles.cusolver().set_stream(stream, OP)?;
        // The Jacobi routine needs a parameter object: it is queried for the
        // workspace size and reused by every call below. It outlives the
        // launches because the solver-status download later in this scope is
        // a host barrier, so the device work has retired before the object
        // drops at the end of the closure. This matches the `gesvdj` path.
        let syevj_params = match routine {
            CusolverEighRoutine::Syevd | CusolverEighRoutine::XsyevBatched => None,
            CusolverEighRoutine::Syevj | CusolverEighRoutine::SyevjBatched => {
                Some(handles.cusolver().create_syevj_info(OP)?)
            }
        };

        // Clone `input` into a fresh work matrix on the session stream.
        let work = raw.alloc_output::<T>(input.shape())?;
        {
            let src = raw.tensor(input)?;
            let dst = raw.tensor_mut(&work)?;
            // SAFETY: both spans were validated resident on this runtime and
            // the copy is stream-ordered; `dst` is exclusively borrowed.
            unsafe {
                raw.copy_bytes(dst.raw_ptr(), src.raw_ptr() as *const _, src.byte_len(), OP)?
            };
        }
        let values = raw.alloc_output::<<T as LinalgScalar>::Real>(&values_shape)?;

        // The batched divide-and-conquer entry point reports byte sizes, so
        // its query is separate from the element-count query below.
        let xsyev_bytes = if routine == CusolverEighRoutine::XsyevBatched {
            let a_ref = raw.tensor(&work)?;
            let values_ref = raw.tensor(&values)?;
            // SAFETY: both spans are validated device allocations on this
            // runtime; only the leading dimensions are queried here.
            let a_ptr = a_ref.raw_ptr().cast_const();
            // SAFETY: `values_ref` is a validated device span on this runtime.
            let values_ptr = values_ref.raw_ptr().cast_const();
            Some(handles.cusolver().xsyev_batched_buffer_size(
                T::DATA_TYPE,
                CusolverEigMode::NoVector,
                CublasFillMode::Lower,
                n_i64,
                a_ptr,
                lda_i64,
                values_ptr,
                batch_total_i64,
                OP,
            )?)
        } else {
            None
        };
        let lwork = if routine == CusolverEighRoutine::XsyevBatched {
            0
        } else {
            let a_ref = raw.tensor(&work)?;
            let values_ref = raw.tensor(&values)?;
            // SAFETY: both spans are validated device allocations on this
            // runtime; only the leading dimensions are queried here.
            let a_ptr = a_ref.raw_ptr().cast_const();
            // SAFETY: `values_ref` is a validated device span on this runtime; the pointer is used only within the raw-session scope.
            let values_ptr = values_ref.raw_ptr().cast_const();
            match &syevj_params {
                None => handles.cusolver().syevd_buffer_size(
                    T::DATA_TYPE,
                    CusolverEigMode::NoVector,
                    CublasFillMode::Lower,
                    n_i32,
                    a_ptr,
                    lda,
                    values_ptr,
                    OP,
                )?,
                Some(params) if routine == CusolverEighRoutine::SyevjBatched => {
                    handles.cusolver().syevj_batched_buffer_size(
                        T::DATA_TYPE,
                        CusolverEigMode::NoVector,
                        CublasFillMode::Lower,
                        n_i32,
                        a_ptr,
                        lda,
                        values_ptr,
                        params,
                        batch_total_i32,
                        OP,
                    )?
                }
                Some(params) => handles.cusolver().syevj_buffer_size(
                    T::DATA_TYPE,
                    CusolverEigMode::NoVector,
                    CublasFillMode::Lower,
                    n_i32,
                    a_ptr,
                    lda,
                    values_ptr,
                    params,
                    OP,
                )?,
            }
        };
        // `Xsyev` reports byte sizes for a device and a host workspace; the
        // legacy entry points report an element count and need device scratch
        // only. The host buffer stays uninitialized because cuSOLVER owns it
        // and Rust never reads it, and it outlives the launch because the
        // solver-status download below is a host barrier.
        let (workspace_nbytes, host_nbytes) = match xsyev_bytes {
            Some(bytes) => bytes,
            None => {
                let lwork = usize::try_from(lwork).map_err(|_| {
                    Error::invalid_argument(
                        OP,
                        "workspace_length",
                        format!("must be non-negative, got {lwork}"),
                    )
                })?;
                (
                    lwork.checked_mul(std::mem::size_of::<T>()).ok_or_else(|| {
                        Error::invalid_argument(OP, "workspace_length", "byte size overflowed")
                    })?,
                    0,
                )
            }
        };
        let workspace = raw.alloc_bytes(workspace_nbytes.max(1), OP)?;
        let mut workspace_ptr = std::ptr::null_mut::<c_void>();
        workspace.with_ptr(|ptr| workspace_ptr = ptr);
        let mut host_workspace = Vec::<std::mem::MaybeUninit<u8>>::new();
        host_workspace
            .try_reserve_exact(host_nbytes)
            .map_err(|error| Error::backend_source(OP, error))?;
        let host_ptr = if host_nbytes == 0 {
            std::ptr::null_mut::<c_void>()
        } else {
            host_workspace.as_mut_ptr().cast::<c_void>()
        };

        let info = raw.alloc_output::<i32>(&[batch_total])?;
        let a_ref = raw.tensor_mut(&work)?;
        // SAFETY: `a_ref` is a validated device span on this runtime; the pointer is used only within the raw-session scope.
        let a_ptr = a_ref.raw_ptr();
        let values_ref = raw.tensor_mut(&values)?;
        // SAFETY: `values_ref` is a validated device span on this runtime; the pointer is used only within the raw-session scope.
        let values_ptr = values_ref.raw_ptr();
        let info_ref = raw.tensor_mut(&info)?;
        // SAFETY: `info_ref` is a validated device span on this runtime; the pointer is used only within the raw-session scope.
        let info_ptr = info_ref.raw_ptr();

        if routine == CusolverEighRoutine::SyevjBatched {
            let params = syevj_params.as_ref().ok_or_else(|| {
                Error::runtime_state(
                    OP,
                    "batched Jacobi eigensolver selected without its cuSOLVER parameters",
                )
            })?;
            // SAFETY: the batch-wide matrix, eigenvalue, info, and workspace
            // allocations match the strides and lengths cuSOLVER queried, and
            // `params` is the object that query used.
            unsafe {
                handles.cusolver().syevj_batched(
                    T::DATA_TYPE,
                    CusolverEigMode::NoVector,
                    CublasFillMode::Lower,
                    n_i32,
                    a_ptr,
                    lda,
                    values_ptr,
                    workspace_ptr,
                    lwork,
                    info_ptr.cast::<i32>(),
                    params,
                    batch_total_i32,
                    OP,
                )
            }?;
        }
        if routine == CusolverEighRoutine::XsyevBatched {
            // SAFETY: the batch-wide matrix, eigenvalue, and info allocations
            // match the strides cuSOLVER expects, and both workspaces have the
            // queried sizes and stay live until the barrier below.
            unsafe {
                handles.cusolver().xsyev_batched(
                    T::DATA_TYPE,
                    CusolverEigMode::NoVector,
                    CublasFillMode::Lower,
                    n_i64,
                    a_ptr,
                    lda_i64,
                    values_ptr,
                    workspace_ptr,
                    workspace_nbytes,
                    host_ptr,
                    host_nbytes,
                    info_ptr.cast::<i32>(),
                    batch_total_i64,
                    OP,
                )
            }?;
        }
        // The batched launch above already covered every matrix.
        let per_matrix_batches = match routine {
            CusolverEighRoutine::SyevjBatched | CusolverEighRoutine::XsyevBatched => 0,
            CusolverEighRoutine::Syevd | CusolverEighRoutine::Syevj => batch_total,
        };
        for batch in 0..per_matrix_batches {
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
            match &syevj_params {
                // SAFETY: batch pointers, workspace, dimensions, and
                // stream-bound handle satisfy cuSOLVER syevd's no-vector
                // eigensolver contract.
                None => {
                    // SAFETY: the batch pointers, dimensions, workspace, and the stream-bound handle
                    // satisfy cuSOLVER's syevd contract for this arm.
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
                        )
                    }?;
                }
                // SAFETY: as above, plus `params` is the same object the
                // workspace size was queried with.
                Some(params) => {
                    unsafe {
                        handles.cusolver().syevj(
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
                            params,
                            OP,
                        )
                    }?;
                }
            }
        }

        // Host barrier (only for reading the solver diagnostics).
        let call = match routine {
            CusolverEighRoutine::Syevd => "cusolverDn*syevd",
            CusolverEighRoutine::Syevj => "cusolverDn*syevj",
            CusolverEighRoutine::SyevjBatched => "cusolverDn*syevjBatched",
            CusolverEighRoutine::XsyevBatched => "cusolverDnXsyevBatched",
        };
        let host_info = raw.download_tensor::<i32>(&info, OP)?;
        for &value in host_info.host_data()? {
            check_solver_info(OP, call, value)?;
        }
        Ok(values)
    })?;

    Ok(values)
}

#[allow(clippy::type_complexity)]
fn build_lu_outputs_device<T>(
    backend: &mut CudaEngine,
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

    let dtype = T::dtype();
    let p = backend.alloc_output(dtype, &p_shape)?;
    let l = backend.alloc_output(dtype, &l_shape)?;
    let u = backend.alloc_output(dtype, &u_shape)?;
    let parity = backend.alloc_output(dtype, batch_shape)?;
    backend.linalg_lu_extract_outputs(&p, &l, &u, &parity, lu, pivots, k as i64, "lu")?;
    let converted = |tensor: Tensor, what: &str| {
        tensor
            .into_typed::<T>()
            .map_err(|_| Error::Internal(format!("LU {what} changed dtype")))
    };
    Ok((
        converted(p, "P")?,
        converted(l, "L")?,
        converted(u, "U")?,
        converted(parity, "parity")?,
    ))
}

fn build_lu_parity_device<T>(
    backend: &mut CudaEngine,
    pivots: &TypedTensor<i32>,
    k: usize,
    batch_shape: &[usize],
) -> Result<TypedTensor<T>>
where
    T: LinalgScalar + TensorScalar,
{
    let batches = batch_shape.iter().product::<usize>() as i64;
    let parity = backend.alloc_output(T::dtype(), batch_shape)?;
    backend.linalg_lu_parity(&parity, pivots, k as i64, batches, "lu_factor")?;
    parity
        .into_typed::<T>()
        .map_err(|_| Error::Internal("LU parity changed dtype".into()))
}

fn fill_one_device_tensor<T>(
    backend: &mut CudaEngine,
    shape: &[usize],
    op: &'static str,
) -> Result<TypedTensor<T>>
where
    T: LinalgScalar + TensorScalar,
{
    let out = backend.alloc_output(T::dtype(), shape)?;
    backend.linalg_fill_one(&out, op)?;
    out.into_typed::<T>()
        .map_err(|_| Error::Internal("fill-one changed dtype".into()))
}

fn apply_lu_pivots_typed<T>(
    backend: &mut CudaEngine,
    input: &TypedTensor<T>,
    pivots: &TypedTensor<i32>,
    inverse: bool,
) -> Result<TypedTensor<T>>
where
    T: LinalgScalar + TensorScalar,
{
    let out = backend.alloc_output(T::dtype(), input.shape())?;
    if input.shape().iter().all(|&extent| extent != 0) {
        // One pivot per factorisation step, as the packed LU stores them.
        let k = pivots.shape()[0] as i64;
        backend.linalg_lu_apply_pivots(&out, input, pivots, k, inverse, "lu_solve_prepared")?;
    }
    out.into_typed::<T>()
        .map_err(|_| Error::Internal("LU pivot application changed dtype".into()))
}

fn zero_sized_lu_factor_pair<T>(
    backend: &mut CudaEngine,
    shape: &[usize],
) -> Result<(TypedTensor<T>, TypedTensor<i32>)>
where
    T: LinalgScalar + TensorScalar,
{
    let m = shape[0];
    let n = shape[1];
    let k = m.min(n);
    let mut pivot_shape = vec![k];
    pivot_shape.extend_from_slice(&shape[2..]);
    let lu = backend.alloc_output(T::dtype(), shape)?;
    let pivots = backend.alloc_output(DType::I32, &pivot_shape)?;
    Ok((
        lu.into_typed::<T>()
            .map_err(|_| Error::Internal("zero-sized LU changed dtype".into()))?,
        pivots
            .into_typed::<i32>()
            .map_err(|_| Error::Internal("zero-sized LU pivots changed dtype".into()))?,
    ))
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
    // SAFETY: the caller guarantees `base` covers `offset` further `T`
    // elements inside one device allocation (see `# Safety` above).
    unsafe { base.cast::<T>() }.add(offset).cast()
}

/// Offset an immutable device pointer by `offset` typed elements.
///
/// # Safety
///
/// `base` must point to a device allocation containing at least `offset`
/// additional `T` elements, and the resulting pointer must remain within the
/// same allocation for the lifetime required by the receiving CUDA library.
unsafe fn batch_const_ptr<T>(base: *const c_void, offset: usize) -> *const c_void {
    // SAFETY: the caller guarantees `base` covers `offset` further `T`
    // elements inside one device allocation (see `# Safety` above).
    unsafe { base.cast::<T>() }.add(offset).cast()
}

fn zero_like_linalg_device_tensor(
    backend: &mut CudaEngine,
    input: &Tensor,
    _op: &'static str,
) -> Result<Tensor> {
    backend.alloc_output(input.dtype(), input.shape())
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
