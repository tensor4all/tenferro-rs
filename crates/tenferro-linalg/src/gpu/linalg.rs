use std::ffi::c_void;
use std::ops::Neg;
use std::os::raw::c_char;

use cubecl::prelude::{ComplexCore, CubeElement, CubePrimitive};
use cubecl_cuda::CudaRuntime as CubeclCudaRuntime;
use cudarc::runtime::{result as cuda_result, sys::cudaStream_t};
use num_complex::{Complex32, Complex64};
use num_traits::{One, Zero};

use super::ffi::cusolver::{
    CublasDiagType, CublasFillMode, CublasOperation, CublasSideMode, CudaDataType,
    CudaLinalgHandles, CudaStream, CusolverEigMode,
};
use super::kernels as cubecl_linalg;
use tenferro_gpu::cuda_interop::{
    alloc_device_bytes, alloc_output, cube_count_for_len, cube_dim_1d, download_typed_tensor,
    ensure_typed_tensor_resident, flush_cubecl_client, typed_tensor_array_arg,
    typed_tensor_binding, upload_device_bytes, with_cubecl_client, with_raw_cuda_stream,
    with_typed_device_ptr as interop_with_typed_device_ptr, CudaExtensionCacheGuard,
    DeviceByteBuffer,
};
// validate_nonsingular_gpu uses backend ops (extract_diagonal, magnitude,
// reduce_min/reduce_max) then downloads scalar summaries — no bulk host
// roundtrip.
use tenferro_gpu::{download_tensor, CudaExecSession, CudaRuntime};
use tenferro_tensor::config::SliceConfig;
use tenferro_tensor::{
    DType, Error, Tensor, TensorElementwise, TensorReduction, TensorStructural, TypedTensor,
    ValidationError,
};

type Result<T> = tenferro_tensor::Result<T>;

trait LinalgScalar:
    CubeElement + CubePrimitive + Copy + Clone + One + Zero + Neg<Output = Self>
{
    type Real: CubeElement + CubePrimitive + Copy + Clone + Zero;

    const DATA_TYPE: CudaDataType;
    const NEEDS_RWORK: bool;

    fn copy_matrix_adjoint(
        rt: &CudaRuntime,
        v: &TypedTensor<Self>,
        vt_shape: &[usize],
        op: &'static str,
    ) -> Result<TypedTensor<Self>>;
}

impl LinalgScalar for f32 {
    type Real = f32;

    const DATA_TYPE: CudaDataType = CudaDataType::F32;
    const NEEDS_RWORK: bool = false;

    fn copy_matrix_adjoint(
        rt: &CudaRuntime,
        v: &TypedTensor<Self>,
        vt_shape: &[usize],
        op: &'static str,
    ) -> Result<TypedTensor<Self>> {
        copy_matrix_adjoint_real(rt, v, vt_shape, op)
    }
}

impl LinalgScalar for f64 {
    type Real = f64;

    const DATA_TYPE: CudaDataType = CudaDataType::F64;
    const NEEDS_RWORK: bool = false;

    fn copy_matrix_adjoint(
        rt: &CudaRuntime,
        v: &TypedTensor<Self>,
        vt_shape: &[usize],
        op: &'static str,
    ) -> Result<TypedTensor<Self>> {
        copy_matrix_adjoint_real(rt, v, vt_shape, op)
    }
}

impl LinalgScalar for Complex32 {
    type Real = f32;

    const DATA_TYPE: CudaDataType = CudaDataType::Complex32;
    const NEEDS_RWORK: bool = true;

    fn copy_matrix_adjoint(
        rt: &CudaRuntime,
        v: &TypedTensor<Self>,
        vt_shape: &[usize],
        op: &'static str,
    ) -> Result<TypedTensor<Self>> {
        copy_matrix_adjoint_complex(rt, v, vt_shape, op)
    }
}

impl LinalgScalar for Complex64 {
    type Real = f64;

    const DATA_TYPE: CudaDataType = CudaDataType::Complex64;
    const NEEDS_RWORK: bool = true;

    fn copy_matrix_adjoint(
        rt: &CudaRuntime,
        v: &TypedTensor<Self>,
        vt_shape: &[usize],
        op: &'static str,
    ) -> Result<TypedTensor<Self>> {
        copy_matrix_adjoint_complex(rt, v, vt_shape, op)
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

fn ensure_cubecl_resident_typed<T: 'static>(
    op: &'static str,
    input: &TypedTensor<T>,
) -> Result<()> {
    ensure_typed_tensor_resident(input, op)
}

fn linalg_handles<'a>(
    backend: &'a CudaExecSession<'_>,
) -> Result<CudaExtensionCacheGuard<'a, CudaLinalgHandles>> {
    backend
        .cuda_extension_cache()
        .get_or_try_init(CudaLinalgHandles::load)
}

struct Workspace {
    _owner: DeviceByteBuffer,
    ptr: *mut c_void,
}

impl Workspace {
    fn none() -> Self {
        Self::from_device(DeviceByteBuffer::none())
    }

    fn from_device(owner: DeviceByteBuffer) -> Self {
        let mut ptr = std::ptr::null_mut();
        owner.with_ptr(|device_ptr| ptr = device_ptr);
        Self { _owner: owner, ptr }
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

    backend.runtime().set_current_cuda_context(OP)?;
    ensure_cubecl_resident_tensor(OP, a)?;
    ensure_cubecl_resident_tensor(OP, b)?;
    ensure_supported_linalg_pair(OP, a, b)?;
    if has_zero_dim(a.shape()) || has_zero_dim(b.shape()) {
        return zero_like_linalg_device_tensor(backend.runtime(), b, OP);
    }

    let (rhs, restore_shape) = if let Some(matrix_rhs_shape) = batched_vector_rhs_shape(a, b) {
        (
            backend.reshape(b, &matrix_rhs_shape)?,
            Some(b.shape().to_vec()),
        )
    } else {
        (b.clone(), None)
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

    backend.runtime().set_current_cuda_context(OP)?;
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
        return zero_like_linalg_device_tensor(backend.runtime(), b, OP);
    }

    let (rhs, restore_shape) = if let Some(matrix_rhs_shape) = batched_vector_rhs_shape(a, b) {
        (
            backend.reshape(b, &matrix_rhs_shape)?,
            Some(b.shape().to_vec()),
        )
    } else {
        (b.clone(), None)
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
    backend: &CudaExecSession<'_>,
    input: &TypedTensor<T>,
) -> Result<TypedTensor<T>>
where
    T: LinalgScalar,
{
    const OP: &str = "cholesky";

    backend.runtime().set_current_cuda_context(OP)?;
    ensure_cubecl_resident_typed(OP, input)?;
    let n = square_matrix_dim(OP, input.shape())?;
    if has_zero_dim(input.shape()) {
        return Ok(alloc_output(backend.runtime(), input.shape())?);
    }

    let work = clone_device_tensor(backend.runtime(), input, OP)?;
    let handles = linalg_handles(backend)?;
    let stream = raw_stream(backend.runtime(), OP)?;
    handles.cusolver().set_stream(stream, OP)?;

    let batch_total = batch_count(OP, &input.shape()[2..])?;
    let first_ptr = typed_device_ptr(backend.runtime(), &work, OP)?;
    let lda = as_i32(n, OP, "lda")?;
    let n_i32 = as_i32(n, OP, "n")?;
    let lwork = handles.cusolver().potrf_buffer_size(
        T::DATA_TYPE,
        CublasFillMode::Lower,
        n_i32,
        first_ptr,
        lda,
        OP,
    )?;
    let workspace = alloc_workspace_elems::<T>(backend.runtime(), lwork, OP)?;
    let info = alloc_output::<i32>(backend.runtime(), &[batch_total])?;
    let info_ptr = typed_device_ptr(backend.runtime(), &info, OP)?;
    let matrix_stride = checked_mul_usize(OP, "cholesky matrix stride", n, n)?;

    for batch in 0..batch_total {
        let a_offset =
            checked_batch_offset(OP, "cholesky matrix batch offset", batch, matrix_stride)?;
        // SAFETY: `a_offset` and `batch` were checked against the matrix and
        // info strides, and both base pointers come from live device tensors.
        let (batch_a, batch_info) = unsafe {
            (
                batch_ptr::<T>(first_ptr, a_offset),
                batch_ptr::<i32>(info_ptr, batch).cast::<i32>(),
            )
        };
        // SAFETY: the batch pointers, workspace, dimensions, and stream-bound
        // handle satisfy cuSOLVER potrf's in-place matrix and info contracts.
        unsafe {
            handles.cusolver().potrf(
                T::DATA_TYPE,
                CublasFillMode::Lower,
                n_i32,
                batch_a,
                lda,
                workspace.ptr,
                lwork,
                batch_info,
                OP,
            )?;
        }
    }
    check_solver_info_tensor(backend.runtime(), &info, OP, "cusolverDn*potrf")?;

    backend.tril_typed(&work, 0)
}

fn triangular_solve_typed<T>(
    backend: &CudaExecSession<'_>,
    a: &TypedTensor<T>,
    b: &TypedTensor<T>,
    left_side: bool,
    lower: bool,
    transpose_a: bool,
    unit_diagonal: bool,
) -> Result<TypedTensor<T>>
where
    T: LinalgScalar,
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
    backend: &CudaExecSession<'_>,
    a: &TypedTensor<T>,
    b: &TypedTensor<T>,
    left_side: bool,
    lower: bool,
    trans: CublasOperation,
    unit_diagonal: bool,
    op: &'static str,
) -> Result<TypedTensor<T>>
where
    T: LinalgScalar,
{
    backend.runtime().set_current_cuda_context(op)?;
    ensure_cubecl_resident_typed(op, a)?;
    ensure_cubecl_resident_typed(op, b)?;
    let n = square_matrix_dim(op, a.shape())?;
    validate_triangular_rhs(op, a.shape(), b.shape(), left_side)?;
    if has_zero_dim(a.shape()) || has_zero_dim(b.shape()) {
        return Ok(alloc_output(backend.runtime(), b.shape())?);
    }

    let out = clone_device_tensor(backend.runtime(), b, op)?;
    let handles = linalg_handles(backend)?;
    let stream = raw_stream(backend.runtime(), op)?;
    handles.cublas().set_stream(stream, op)?;

    let a_ptr = typed_device_ptr(backend.runtime(), a, op)?;
    let out_ptr = typed_device_ptr(backend.runtime(), &out, op)?;
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
        let a_array = upload_pointer_array(backend.runtime(), &a_pointers, op)?;
        let b_array = upload_pointer_array(backend.runtime(), &b_pointers, op)?;
        // SAFETY: uploaded pointer arrays contain one valid matrix/RHS device
        // pointer per batch, and scalar dimensions/leading dimensions are validated.
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
                a_array.ptr.cast_const(),
                lda,
                b_array.ptr,
                ldb,
                as_i32(batch_total, op, "batch_count")?,
                op,
            )?;
        }
    } else {
        // SAFETY: zero offset points at the first element of the live matrix
        // and RHS device allocations already validated for this single batch.
        let (batch_a, batch_b) = unsafe {
            (
                batch_const_ptr::<T>(a_ptr.cast_const(), 0),
                batch_ptr::<T>(out_ptr, 0),
            )
        };
        // SAFETY: pointers, scalar dimensions, and leading dimensions satisfy
        // cuBLAS trsm for the validated single-batch triangular solve.
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
}

fn lu_typed<T>(
    backend: &CudaExecSession<'_>,
    input: &TypedTensor<T>,
) -> Result<(
    TypedTensor<T>,
    TypedTensor<T>,
    TypedTensor<T>,
    TypedTensor<T>,
)>
where
    T: LinalgScalar,
{
    const OP: &str = "lu";

    let (packed_lu, pivots, parity) = lu_factor_typed(backend, input)?;
    let (m, n) = matrix_dims(OP, input.shape())?;
    let (p, l, u, _extracted_parity) = build_lu_outputs_device(
        backend.runtime(),
        &packed_lu,
        &pivots,
        m,
        n,
        &input.shape()[2..],
    )?;
    Ok((p, l, u, parity))
}

fn lu_factor_typed<T>(
    backend: &CudaExecSession<'_>,
    input: &TypedTensor<T>,
) -> Result<(TypedTensor<T>, TypedTensor<i32>, TypedTensor<T>)>
where
    T: LinalgScalar,
{
    const OP: &str = "lu_factor";

    backend.runtime().set_current_cuda_context(OP)?;
    ensure_cubecl_resident_typed(OP, input)?;
    let (m, n) = matrix_dims(OP, input.shape())?;
    let k = m.min(n);
    if has_zero_dim(input.shape()) {
        return zero_sized_lu_factor_outputs(backend.runtime(), input.shape());
    }

    let work = clone_device_tensor(backend.runtime(), input, OP)?;
    let handles = linalg_handles(backend)?;
    let stream = raw_stream(backend.runtime(), OP)?;
    handles.cusolver().set_stream(stream, OP)?;

    let batch_total = batch_count(OP, &input.shape()[2..])?;
    let a_ptr = typed_device_ptr(backend.runtime(), &work, OP)?;
    let lda = as_i32(m, OP, "lda")?;
    let m_i32 = as_i32(m, OP, "m")?;
    let n_i32 = as_i32(n, OP, "n")?;
    let lwork = handles
        .cusolver()
        .getrf_buffer_size(T::DATA_TYPE, m_i32, n_i32, a_ptr, lda, OP)?;
    let workspace = alloc_workspace_elems::<T>(backend.runtime(), lwork, OP)?;
    let mut pivot_shape = vec![k];
    pivot_shape.extend_from_slice(&input.shape()[2..]);
    let pivots = alloc_output::<i32>(backend.runtime(), &pivot_shape)?;
    let info = alloc_output::<i32>(backend.runtime(), &[batch_total])?;
    let pivots_ptr = typed_device_ptr(backend.runtime(), &pivots, OP)?;
    let info_ptr = typed_device_ptr(backend.runtime(), &info, OP)?;
    let matrix_stride = checked_mul_usize(OP, "lu matrix stride", m, n)?;

    for batch in 0..batch_total {
        let a_offset = checked_batch_offset(OP, "lu matrix batch offset", batch, matrix_stride)?;
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
                workspace.ptr,
                batch_pivots.cast::<i32>(),
                batch_info.cast::<i32>(),
                OP,
            )?;
        }
    }

    let host_info = download_device_tensor(backend.runtime(), &info, OP)?;
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

    let parity = build_lu_parity_device(backend.runtime(), &pivots, k, &input.shape()[2..])?;
    Ok((work, pivots, parity))
}

fn lu_solve_prepared_typed<T>(
    backend: &CudaExecSession<'_>,
    packed_lu: &TypedTensor<T>,
    pivots: &TypedTensor<i32>,
    b: &TypedTensor<T>,
    transpose_a: bool,
    conjugate_a: bool,
) -> Result<TypedTensor<T>>
where
    T: LinalgScalar,
{
    const OP: &str = "lu_solve_prepared";

    backend.runtime().set_current_cuda_context(OP)?;
    ensure_cubecl_resident_typed(OP, packed_lu)?;
    ensure_cubecl_resident_typed(OP, pivots)?;
    ensure_cubecl_resident_typed(OP, b)?;
    validate_lu_solve_prepared_shapes(packed_lu.shape(), pivots.shape(), b.shape())?;
    if has_zero_dim(packed_lu.shape()) || has_zero_dim(b.shape()) {
        return Ok(alloc_output(backend.runtime(), b.shape())?);
    }

    match (transpose_a, conjugate_a) {
        (false, false) => {
            let pb = apply_lu_pivots_typed(backend.runtime(), b, pivots, false)?;
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
            apply_lu_pivots_typed(backend.runtime(), &y, pivots, true)
        }
        (false, true) => Err(Error::unsupported(
            OP,
            "conjugate-only prepared LU solve is unsupported on CUDA; use transpose+conjugate or solve the conjugated matrix explicitly",
        )),
    }
}

fn copy_matrix_adjoint_real<T>(
    rt: &CudaRuntime,
    v: &TypedTensor<T>,
    vt_shape: &[usize],
    op: &'static str,
) -> Result<TypedTensor<T>>
where
    T: LinalgScalar,
{
    let vt = alloc_output::<T>(rt, vt_shape)?;
    launch_matrix_adjoint_real(rt, v, &vt, op)?;
    Ok(vt)
}

fn copy_matrix_adjoint_complex<T>(
    rt: &CudaRuntime,
    v: &TypedTensor<T>,
    vt_shape: &[usize],
    op: &'static str,
) -> Result<TypedTensor<T>>
where
    T: LinalgScalar + ComplexCore,
{
    let vt = alloc_output::<T>(rt, vt_shape)?;
    launch_matrix_adjoint_complex(rt, v, &vt, op)?;
    Ok(vt)
}

fn launch_matrix_adjoint_real<T>(
    rt: &CudaRuntime,
    v: &TypedTensor<T>,
    vt: &TypedTensor<T>,
    op: &'static str,
) -> Result<()>
where
    T: LinalgScalar,
{
    if vt.n_elements() == 0 {
        return Ok(());
    }
    let vt_arg = typed_tensor_binding(vt, op)?;
    let v_arg = typed_tensor_binding(v, op)?;
    let launch_count = cube_count_for_len(vt.n_elements())?;
    // SAFETY: tensor bindings describe live CUDA tensors, and `launch_count`
    // covers exactly the output domain for the matrix-adjoint copy kernel.
    with_cubecl_client(rt, |client| unsafe {
        cubecl_linalg::matrix_adjoint_real::launch_unchecked::<T, CubeclCudaRuntime>(
            client,
            launch_count,
            cube_dim_1d(),
            vt_arg.into_tensor_arg(),
            v_arg.into_tensor_arg(),
            vt.shape().len(),
        );
    });
    flush_cubecl_client(rt, op)
}

fn launch_matrix_adjoint_complex<T>(
    rt: &CudaRuntime,
    v: &TypedTensor<T>,
    vt: &TypedTensor<T>,
    op: &'static str,
) -> Result<()>
where
    T: LinalgScalar + ComplexCore,
{
    if vt.n_elements() == 0 {
        return Ok(());
    }
    let vt_arg = typed_tensor_binding(vt, op)?;
    let v_arg = typed_tensor_binding(v, op)?;
    let launch_count = cube_count_for_len(vt.n_elements())?;
    // SAFETY: tensor bindings describe live CUDA tensors, and `launch_count`
    // covers exactly the output domain for the conjugating matrix-adjoint kernel.
    with_cubecl_client(rt, |client| unsafe {
        cubecl_linalg::matrix_adjoint_complex::launch_unchecked::<T, CubeclCudaRuntime>(
            client,
            launch_count,
            cube_dim_1d(),
            vt_arg.into_tensor_arg(),
            v_arg.into_tensor_arg(),
            vt.shape().len(),
        );
    });
    flush_cubecl_client(rt, op)
}

fn svd_typed<T>(
    backend: &CudaExecSession<'_>,
    input: &TypedTensor<T>,
) -> Result<(
    TypedTensor<T>,
    TypedTensor<<T as LinalgScalar>::Real>,
    TypedTensor<T>,
)>
where
    T: LinalgScalar,
{
    const OP: &str = "svd";

    backend.runtime().set_current_cuda_context(OP)?;
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
        return Ok((
            alloc_output(backend.runtime(), &u_shape)?,
            alloc_output(backend.runtime(), &s_shape)?,
            alloc_output(backend.runtime(), &vt_shape)?,
        ));
    }

    let s = alloc_output::<<T as LinalgScalar>::Real>(backend.runtime(), &s_shape)?;
    let batch_total = batch_count(OP, batch_shape)?;
    let a_stride = checked_mul_usize(OP, "svd input stride", m, n)?;
    let s_stride = k;

    match select_svd_driver(m, n) {
        SvdDriver::Gesvdj => {
            let work = clone_device_tensor(backend.runtime(), input, OP)?;
            let u = alloc_output::<T>(backend.runtime(), &u_shape)?;
            let mut v_shape = vec![n, k];
            v_shape.extend_from_slice(batch_shape);
            let v = alloc_output::<T>(backend.runtime(), &v_shape)?;
            {
                let handles = linalg_handles(backend)?;
                let stream = raw_stream(backend.runtime(), OP)?;
                handles.cusolver().set_stream(stream, OP)?;

                let a_ptr = typed_device_ptr(backend.runtime(), &work, OP)?;
                let u_ptr = typed_device_ptr(backend.runtime(), &u, OP)?;
                let s_ptr = typed_device_ptr(backend.runtime(), &s, OP)?;
                let v_ptr = typed_device_ptr(backend.runtime(), &v, OP)?;
                let m_i32 = as_i32(m, OP, "m")?;
                let n_i32 = as_i32(n, OP, "n")?;
                let lda = as_i32(m, OP, "lda")?;
                let ldu = as_i32(m, OP, "ldu")?;
                let ldv = as_i32(n, OP, "ldv")?;
                let params = handles.cusolver().create_gesvdj_info(OP)?;
                let lwork = handles.cusolver().gesvdj_buffer_size(
                    T::DATA_TYPE,
                    CusolverEigMode::Vector,
                    1,
                    m_i32,
                    n_i32,
                    a_ptr.cast_const(),
                    lda,
                    s_ptr.cast_const(),
                    u_ptr.cast_const(),
                    ldu,
                    v_ptr.cast_const(),
                    ldv,
                    &params,
                    OP,
                )?;
                let workspace = alloc_workspace_elems::<T>(backend.runtime(), lwork, OP)?;
                let info = alloc_output::<i32>(backend.runtime(), &[batch_total])?;
                let info_ptr = typed_device_ptr(backend.runtime(), &info, OP)?;
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
                            workspace.ptr,
                            lwork,
                            batch_info,
                            &params,
                            OP,
                        )?;
                    }
                }
                check_solver_info_tensor(backend.runtime(), &info, OP, "cusolverDn*gesvdj")?;
            }
            let vt = T::copy_matrix_adjoint(backend.runtime(), &v, &vt_shape, OP)?;
            Ok((u, s, vt))
        }
        SvdDriver::Gesvd => {
            let transpose_for_gesvd = m < n;
            let (work, gesvd_m, gesvd_n) = if transpose_for_gesvd {
                let mut work_shape = vec![n, m];
                work_shape.extend_from_slice(batch_shape);
                (
                    T::copy_matrix_adjoint(backend.runtime(), input, &work_shape, OP)?,
                    n,
                    m,
                )
            } else {
                (clone_device_tensor(backend.runtime(), input, OP)?, m, n)
            };
            let mut gesvd_u_shape = vec![gesvd_m, k];
            gesvd_u_shape.extend_from_slice(batch_shape);
            let mut gesvd_vt_shape = vec![k, gesvd_n];
            gesvd_vt_shape.extend_from_slice(batch_shape);
            let gesvd_u = alloc_output::<T>(backend.runtime(), &gesvd_u_shape)?;
            let gesvd_vt = alloc_output::<T>(backend.runtime(), &gesvd_vt_shape)?;
            let handles = linalg_handles(backend)?;
            let stream = raw_stream(backend.runtime(), OP)?;
            handles.cusolver().set_stream(stream, OP)?;

            let a_ptr = typed_device_ptr(backend.runtime(), &work, OP)?;
            let u_ptr = typed_device_ptr(backend.runtime(), &gesvd_u, OP)?;
            let s_ptr = typed_device_ptr(backend.runtime(), &s, OP)?;
            let vt_ptr = typed_device_ptr(backend.runtime(), &gesvd_vt, OP)?;
            let gesvd_m_i32 = as_i32(gesvd_m, OP, "gesvd m")?;
            let gesvd_n_i32 = as_i32(gesvd_n, OP, "gesvd n")?;
            let lda = gesvd_m_i32;
            let ldu = gesvd_m_i32;
            let ldvt = as_i32(k, OP, "ldvt")?;
            let lwork =
                handles
                    .cusolver()
                    .gesvd_buffer_size(T::DATA_TYPE, gesvd_m_i32, gesvd_n_i32, OP)?;
            let workspace = alloc_workspace_elems::<T>(backend.runtime(), lwork, OP)?;
            let rwork = if T::NEEDS_RWORK {
                alloc_workspace_elems::<<T as LinalgScalar>::Real>(
                    backend.runtime(),
                    as_i32(
                        checked_mul_usize(OP, "svd rwork length", 5, k)?,
                        OP,
                        "rwork",
                    )?,
                    OP,
                )?
            } else {
                Workspace::none()
            };
            let info = alloc_output::<i32>(backend.runtime(), &[batch_total])?;
            let info_ptr = typed_device_ptr(backend.runtime(), &info, OP)?;
            let u_stride = checked_mul_usize(OP, "svd gesvd u stride", gesvd_m, k)?;
            let vt_stride = checked_mul_usize(OP, "svd gesvd vt stride", k, gesvd_n)?;
            let job = b'S' as c_char;

            for batch in 0..batch_total {
                let a_offset = checked_batch_offset(OP, "svd input batch offset", batch, a_stride)?;
                let s_offset =
                    checked_batch_offset(OP, "svd singular value batch offset", batch, s_stride)?;
                let u_offset = checked_batch_offset(OP, "svd u batch offset", batch, u_stride)?;
                let vt_offset = checked_batch_offset(OP, "svd vt batch offset", batch, vt_stride)?;
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
                        workspace.ptr,
                        lwork,
                        rwork.ptr,
                        batch_info,
                        OP,
                    )?;
                }
            }
            check_solver_info_tensor(backend.runtime(), &info, OP, "cusolverDn*gesvd")?;
            if transpose_for_gesvd {
                let u = T::copy_matrix_adjoint(backend.runtime(), &gesvd_vt, &u_shape, OP)?;
                let vt = T::copy_matrix_adjoint(backend.runtime(), &gesvd_u, &vt_shape, OP)?;
                Ok((u, s, vt))
            } else {
                Ok((gesvd_u, s, gesvd_vt))
            }
        }
    }
}

fn svd_values_typed<T>(
    backend: &CudaExecSession<'_>,
    input: &TypedTensor<T>,
) -> Result<TypedTensor<T::Real>>
where
    T: LinalgScalar,
{
    const OP: &str = "svd_values";

    backend.runtime().set_current_cuda_context(OP)?;
    ensure_cubecl_resident_typed(OP, input)?;
    let (m, n) = matrix_dims(OP, input.shape())?;
    let k = m.min(n);
    let batch_shape = &input.shape()[2..];
    let mut s_shape = vec![k];
    s_shape.extend_from_slice(batch_shape);
    if has_zero_dim(input.shape()) {
        return Ok(alloc_output(backend.runtime(), &s_shape)?);
    }

    let s = alloc_output::<T::Real>(backend.runtime(), &s_shape)?;
    let batch_total = batch_count(OP, batch_shape)?;
    let a_stride = checked_mul_usize(OP, "svd_values input stride", m, n)?;
    let s_stride = k;

    match select_svd_driver(m, n) {
        SvdDriver::Gesvdj => {
            let work = clone_device_tensor(backend.runtime(), input, OP)?;
            let mut u_shape = vec![m, k];
            u_shape.extend_from_slice(batch_shape);
            let mut v_shape = vec![n, k];
            v_shape.extend_from_slice(batch_shape);
            let u = alloc_output::<T>(backend.runtime(), &u_shape)?;
            let v = alloc_output::<T>(backend.runtime(), &v_shape)?;
            let handles = linalg_handles(backend)?;
            let stream = raw_stream(backend.runtime(), OP)?;
            handles.cusolver().set_stream(stream, OP)?;

            let a_ptr = typed_device_ptr(backend.runtime(), &work, OP)?;
            let s_ptr = typed_device_ptr(backend.runtime(), &s, OP)?;
            let u_ptr = typed_device_ptr(backend.runtime(), &u, OP)?;
            let v_ptr = typed_device_ptr(backend.runtime(), &v, OP)?;
            let m_i32 = as_i32(m, OP, "m")?;
            let n_i32 = as_i32(n, OP, "n")?;
            let lda = as_i32(m, OP, "lda")?;
            let params = handles.cusolver().create_gesvdj_info(OP)?;
            let lwork = handles.cusolver().gesvdj_buffer_size(
                T::DATA_TYPE,
                CusolverEigMode::NoVector,
                1,
                m_i32,
                n_i32,
                a_ptr.cast_const(),
                lda,
                s_ptr.cast_const(),
                u_ptr.cast_const(),
                lda,
                v_ptr.cast_const(),
                n_i32,
                &params,
                OP,
            )?;
            let workspace = alloc_workspace_elems::<T>(backend.runtime(), lwork, OP)?;
            let info = alloc_output::<i32>(backend.runtime(), &[batch_total])?;
            let info_ptr = typed_device_ptr(backend.runtime(), &info, OP)?;
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
                        batch_ptr::<T::Real>(s_ptr, s_offset),
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
                        workspace.ptr,
                        lwork,
                        batch_info,
                        &params,
                        OP,
                    )?;
                }
            }
            check_solver_info_tensor(backend.runtime(), &info, OP, "cusolverDn*gesvdj")?;
            Ok(s)
        }
        SvdDriver::Gesvd => {
            let (work, gesvd_m, gesvd_n) = if m < n {
                let mut work_shape = vec![n, m];
                work_shape.extend_from_slice(batch_shape);
                (
                    T::copy_matrix_adjoint(backend.runtime(), input, &work_shape, OP)?,
                    n,
                    m,
                )
            } else {
                (clone_device_tensor(backend.runtime(), input, OP)?, m, n)
            };
            let handles = linalg_handles(backend)?;
            let stream = raw_stream(backend.runtime(), OP)?;
            handles.cusolver().set_stream(stream, OP)?;

            let a_ptr = typed_device_ptr(backend.runtime(), &work, OP)?;
            let s_ptr = typed_device_ptr(backend.runtime(), &s, OP)?;
            let gesvd_m_i32 = as_i32(gesvd_m, OP, "gesvd m")?;
            let gesvd_n_i32 = as_i32(gesvd_n, OP, "gesvd n")?;
            let lda = gesvd_m_i32;
            let lwork =
                handles
                    .cusolver()
                    .gesvd_buffer_size(T::DATA_TYPE, gesvd_m_i32, gesvd_n_i32, OP)?;
            let workspace = alloc_workspace_elems::<T>(backend.runtime(), lwork, OP)?;
            let rwork = if T::NEEDS_RWORK {
                alloc_workspace_elems::<T::Real>(
                    backend.runtime(),
                    as_i32(
                        checked_mul_usize(OP, "svd_values rwork length", 5, k)?,
                        OP,
                        "rwork",
                    )?,
                    OP,
                )?
            } else {
                Workspace::none()
            };
            let info = alloc_output::<i32>(backend.runtime(), &[batch_total])?;
            let info_ptr = typed_device_ptr(backend.runtime(), &info, OP)?;
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
                // SAFETY: checked offsets keep the input, singular-value, and
                // info pointers inside live device allocations for this batch.
                let (batch_a, batch_s, batch_info) = unsafe {
                    (
                        batch_ptr::<T>(a_ptr, a_offset),
                        batch_ptr::<T::Real>(s_ptr, s_offset),
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
                        workspace.ptr,
                        lwork,
                        rwork.ptr,
                        batch_info,
                        OP,
                    )?;
                }
            }
            check_solver_info_tensor(backend.runtime(), &info, OP, "cusolverDn*gesvd")?;
            Ok(s)
        }
    }
}

fn qr_typed<T>(
    backend: &CudaExecSession<'_>,
    input: &TypedTensor<T>,
) -> Result<(TypedTensor<T>, TypedTensor<T>)>
where
    T: LinalgScalar,
{
    const OP: &str = "qr";

    backend.runtime().set_current_cuda_context(OP)?;
    ensure_cubecl_resident_typed(OP, input)?;
    let (m, n) = matrix_dims(OP, input.shape())?;
    let k = m.min(n);
    let batch_shape = &input.shape()[2..];
    let mut q_shape = vec![m, k];
    q_shape.extend_from_slice(batch_shape);
    let mut r_shape = vec![k, n];
    r_shape.extend_from_slice(batch_shape);
    if has_zero_dim(input.shape()) {
        return Ok((
            alloc_output(backend.runtime(), &q_shape)?,
            alloc_output(backend.runtime(), &r_shape)?,
        ));
    }

    let work = clone_device_tensor(backend.runtime(), input, OP)?;
    let q = alloc_output::<T>(backend.runtime(), &q_shape)?;
    let handles = linalg_handles(backend)?;
    let stream = raw_stream(backend.runtime(), OP)?;
    handles.cusolver().set_stream(stream, OP)?;

    let work_ptr = typed_device_ptr(backend.runtime(), &work, OP)?;
    let q_ptr = typed_device_ptr(backend.runtime(), &q, OP)?;
    let m_i32 = as_i32(m, OP, "m")?;
    let n_i32 = as_i32(n, OP, "n")?;
    let k_i32 = as_i32(k, OP, "k")?;
    let lda = as_i32(m, OP, "lda")?;
    let geqrf_lwork =
        handles
            .cusolver()
            .geqrf_buffer_size(T::DATA_TYPE, m_i32, n_i32, work_ptr, lda, OP)?;
    let geqrf_workspace = alloc_workspace_elems::<T>(backend.runtime(), geqrf_lwork, OP)?;
    let tau_bytes = checked_mul_usize(OP, "qr tau workspace bytes", k, std::mem::size_of::<T>())?;
    let tau = alloc_workspace_bytes(backend.runtime(), tau_bytes, OP)?;
    let orgqr_lwork = handles.cusolver().orgqr_buffer_size(
        T::DATA_TYPE,
        m_i32,
        k_i32,
        k_i32,
        q_ptr.cast_const(),
        lda,
        tau.ptr.cast_const(),
        OP,
    )?;
    let orgqr_workspace = alloc_workspace_elems::<T>(backend.runtime(), orgqr_lwork, OP)?;
    let batch_total = batch_count(OP, batch_shape)?;
    let geqrf_info = alloc_output::<i32>(backend.runtime(), &[batch_total])?;
    let orgqr_info = alloc_output::<i32>(backend.runtime(), &[batch_total])?;
    let geqrf_info_ptr = typed_device_ptr(backend.runtime(), &geqrf_info, OP)?;
    let orgqr_info_ptr = typed_device_ptr(backend.runtime(), &orgqr_info, OP)?;
    let work_stride = checked_mul_usize(OP, "qr work stride", m, n)?;
    let q_stride = checked_mul_usize(OP, "qr q stride", m, k)?;

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
                tau.ptr,
                geqrf_workspace.ptr,
                geqrf_lwork,
                batch_geqrf_info,
                OP,
            )?;
        }

        copy_device_to_device(
            backend.runtime(),
            batch_q,
            batch_work.cast_const(),
            q_stride * std::mem::size_of::<T>(),
            OP,
        )?;
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
                tau.ptr.cast_const(),
                orgqr_workspace.ptr,
                orgqr_lwork,
                batch_orgqr_info,
                OP,
            )?;
        }
    }
    check_solver_info_tensor(backend.runtime(), &geqrf_info, OP, "cusolverDn*geqrf")?;
    check_solver_info_tensor(backend.runtime(), &orgqr_info, OP, "cusolverDn*orgqr")?;

    let r_input = backend.slice_typed(&work, &matrix_slice_config(input.shape(), k, n))?;
    let r = backend.triu_typed(&r_input, 0)?;
    Ok((q, r))
}

fn eigh_typed<T>(
    backend: &CudaExecSession<'_>,
    input: &TypedTensor<T>,
) -> Result<(TypedTensor<T::Real>, TypedTensor<T>)>
where
    T: LinalgScalar,
{
    const OP: &str = "eigh";

    backend.runtime().set_current_cuda_context(OP)?;
    ensure_cubecl_resident_typed(OP, input)?;
    let n = square_matrix_dim(OP, input.shape())?;
    let batch_shape = &input.shape()[2..];
    let mut values_shape = vec![n];
    values_shape.extend_from_slice(batch_shape);
    if has_zero_dim(input.shape()) {
        return Ok((
            alloc_output(backend.runtime(), &values_shape)?,
            alloc_output(backend.runtime(), input.shape())?,
        ));
    }

    let work = clone_device_tensor(backend.runtime(), input, OP)?;
    let values = alloc_output::<T::Real>(backend.runtime(), &values_shape)?;
    let handles = linalg_handles(backend)?;
    let stream = raw_stream(backend.runtime(), OP)?;
    handles.cusolver().set_stream(stream, OP)?;

    let a_ptr = typed_device_ptr(backend.runtime(), &work, OP)?;
    let values_ptr = typed_device_ptr(backend.runtime(), &values, OP)?;
    let n_i32 = as_i32(n, OP, "n")?;
    let lda = as_i32(n, OP, "lda")?;
    let lwork = handles.cusolver().syevd_buffer_size(
        T::DATA_TYPE,
        CusolverEigMode::Vector,
        CublasFillMode::Lower,
        n_i32,
        a_ptr.cast_const(),
        lda,
        values_ptr.cast_const(),
        OP,
    )?;
    let workspace = alloc_workspace_elems::<T>(backend.runtime(), lwork, OP)?;
    let batch_total = batch_count(OP, batch_shape)?;
    let info = alloc_output::<i32>(backend.runtime(), &[batch_total])?;
    let info_ptr = typed_device_ptr(backend.runtime(), &info, OP)?;
    let matrix_stride = checked_mul_usize(OP, "eigh matrix stride", n, n)?;
    let values_stride = n;

    for batch in 0..batch_total {
        let a_offset = checked_batch_offset(OP, "eigh matrix batch offset", batch, matrix_stride)?;
        let values_offset =
            checked_batch_offset(OP, "eigh values batch offset", batch, values_stride)?;
        // SAFETY: checked offsets keep matrix, eigenvalue, and info pointers
        // inside live device allocations for this batch.
        let (batch_a, batch_w, batch_info) = unsafe {
            (
                batch_ptr::<T>(a_ptr, a_offset),
                batch_ptr::<T::Real>(values_ptr, values_offset),
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
                workspace.ptr,
                lwork,
                batch_info,
                OP,
            )?;
        }
    }
    check_solver_info_tensor(backend.runtime(), &info, OP, "cusolverDn*syevd")?;

    Ok((values, work))
}

fn eigh_values_typed<T>(
    backend: &CudaExecSession<'_>,
    input: &TypedTensor<T>,
) -> Result<TypedTensor<T::Real>>
where
    T: LinalgScalar,
{
    const OP: &str = "eigh_values";

    backend.runtime().set_current_cuda_context(OP)?;
    ensure_cubecl_resident_typed(OP, input)?;
    let n = square_matrix_dim(OP, input.shape())?;
    let batch_shape = &input.shape()[2..];
    let mut values_shape = vec![n];
    values_shape.extend_from_slice(batch_shape);
    if has_zero_dim(input.shape()) {
        return Ok(alloc_output(backend.runtime(), &values_shape)?);
    }

    let work = clone_device_tensor(backend.runtime(), input, OP)?;
    let values = alloc_output::<T::Real>(backend.runtime(), &values_shape)?;
    let handles = linalg_handles(backend)?;
    let stream = raw_stream(backend.runtime(), OP)?;
    handles.cusolver().set_stream(stream, OP)?;

    let a_ptr = typed_device_ptr(backend.runtime(), &work, OP)?;
    let values_ptr = typed_device_ptr(backend.runtime(), &values, OP)?;
    let n_i32 = as_i32(n, OP, "n")?;
    let lda = as_i32(n, OP, "lda")?;
    let lwork = handles.cusolver().syevd_buffer_size(
        T::DATA_TYPE,
        CusolverEigMode::NoVector,
        CublasFillMode::Lower,
        n_i32,
        a_ptr.cast_const(),
        lda,
        values_ptr.cast_const(),
        OP,
    )?;
    let workspace = alloc_workspace_elems::<T>(backend.runtime(), lwork, OP)?;
    let batch_total = batch_count(OP, batch_shape)?;
    let info = alloc_output::<i32>(backend.runtime(), &[batch_total])?;
    let info_ptr = typed_device_ptr(backend.runtime(), &info, OP)?;
    let matrix_stride = checked_mul_usize(OP, "eigh_values matrix stride", n, n)?;
    let values_stride = n;

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
                batch_ptr::<T::Real>(values_ptr, values_offset),
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
                workspace.ptr,
                lwork,
                batch_info,
                OP,
            )?;
        }
    }
    check_solver_info_tensor(backend.runtime(), &info, OP, "cusolverDn*syevd")?;

    Ok(values)
}

fn build_lu_outputs_device<T>(
    rt: &CudaRuntime,
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
    T: LinalgScalar,
{
    let k = m.min(n);
    let mut p_shape = vec![m, m];
    p_shape.extend_from_slice(batch_shape);
    let mut l_shape = vec![m, k];
    l_shape.extend_from_slice(batch_shape);
    let mut u_shape = vec![k, n];
    u_shape.extend_from_slice(batch_shape);
    let parity_shape = batch_shape.to_vec();

    let p = alloc_output::<T>(rt, &p_shape)?;
    let l = alloc_output::<T>(rt, &l_shape)?;
    let u = alloc_output::<T>(rt, &u_shape)?;
    let parity = alloc_output::<T>(rt, &parity_shape)?;
    let launch_len = p
        .n_elements()
        .max(l.n_elements())
        .max(u.n_elements())
        .max(parity.n_elements());
    let p_arg = typed_tensor_binding(&p, "lu")?;
    let l_arg = typed_tensor_binding(&l, "lu")?;
    let u_arg = typed_tensor_binding(&u, "lu")?;
    let parity_arg = typed_tensor_binding(&parity, "lu")?;
    let work_arg = typed_tensor_binding(lu, "lu")?;
    let pivots_arg = typed_tensor_array_arg(pivots, "lu")?;
    let launch_count = cube_count_for_len(launch_len)?;
    // SAFETY: tensor bindings describe live CUDA tensors, and `launch_count`
    // covers the maximum P/L/U/parity output domain consumed by the kernel.
    with_cubecl_client(rt, |client| unsafe {
        cubecl_linalg::lu_extract_outputs::launch_unchecked::<T, CubeclCudaRuntime>(
            client,
            launch_count,
            cube_dim_1d(),
            p_arg.into_tensor_arg(),
            l_arg.into_tensor_arg(),
            u_arg.into_tensor_arg(),
            parity_arg.into_tensor_arg(),
            work_arg.into_tensor_arg(),
            pivots_arg,
            k,
            lu.shape().len(),
        );
    });
    flush_cubecl_client(rt, "lu")?;

    Ok((p, l, u, parity))
}

fn build_lu_parity_device<T>(
    rt: &CudaRuntime,
    pivots: &TypedTensor<i32>,
    k: usize,
    batch_shape: &[usize],
) -> Result<TypedTensor<T>>
where
    T: LinalgScalar,
{
    let parity = alloc_output::<T>(rt, batch_shape)?;
    let parity_arg = typed_tensor_binding(&parity, "lu_factor")?;
    let pivots_arg = typed_tensor_array_arg(pivots, "lu_factor")?;
    let launch_count = cube_count_for_len(parity.n_elements())?;
    // SAFETY: tensor bindings describe live CUDA tensors, and `launch_count`
    // covers exactly the parity output domain.
    with_cubecl_client(rt, |client| unsafe {
        cubecl_linalg::lu_parity::launch_unchecked::<T, CubeclCudaRuntime>(
            client,
            launch_count,
            cube_dim_1d(),
            parity_arg.into_tensor_arg(),
            pivots_arg,
            k,
        );
    });
    flush_cubecl_client(rt, "lu_factor")?;
    Ok(parity)
}

fn fill_one_device_tensor<T>(
    rt: &CudaRuntime,
    shape: &[usize],
    op: &'static str,
) -> Result<TypedTensor<T>>
where
    T: LinalgScalar,
{
    let out = alloc_output::<T>(rt, shape)?;
    let out_arg = typed_tensor_binding(&out, op)?;
    let launch_count = cube_count_for_len(out.n_elements())?;
    // SAFETY: `out_arg` describes a live CUDA output tensor, and
    // `launch_count` covers every output element exactly once.
    with_cubecl_client(rt, |client| unsafe {
        cubecl_linalg::fill_one_kernel::launch_unchecked::<T, CubeclCudaRuntime>(
            client,
            launch_count,
            cube_dim_1d(),
            out_arg.into_tensor_arg(),
        );
    });
    flush_cubecl_client(rt, op)?;
    Ok(out)
}

fn apply_lu_pivots_typed<T>(
    rt: &CudaRuntime,
    input: &TypedTensor<T>,
    pivots: &TypedTensor<i32>,
    inverse: bool,
) -> Result<TypedTensor<T>>
where
    T: LinalgScalar,
{
    let out = alloc_output::<T>(rt, input.shape())?;
    if out.n_elements() == 0 {
        return Ok(out);
    }
    let k = pivots.shape()[0];
    let out_arg = typed_tensor_binding(&out, "lu_solve_prepared")?;
    let input_arg = typed_tensor_binding(input, "lu_solve_prepared")?;
    let pivots_arg = typed_tensor_array_arg(pivots, "lu_solve_prepared")?;
    let launch_count = cube_count_for_len(out.n_elements())?;
    // SAFETY: tensor bindings describe live CUDA tensors, pivot metadata
    // matches the validated LU shape, and `launch_count` covers the output domain.
    with_cubecl_client(rt, |client| unsafe {
        cubecl_linalg::lu_apply_pivots::launch_unchecked::<T, CubeclCudaRuntime>(
            client,
            launch_count,
            cube_dim_1d(),
            out_arg.into_tensor_arg(),
            input_arg.into_tensor_arg(),
            pivots_arg,
            k,
            input.shape().len(),
            inverse,
        );
    });
    flush_cubecl_client(rt, "lu_solve_prepared")?;
    Ok(out)
}

fn zero_sized_lu_factor_outputs<T>(
    rt: &CudaRuntime,
    shape: &[usize],
) -> Result<(TypedTensor<T>, TypedTensor<i32>, TypedTensor<T>)>
where
    T: LinalgScalar,
{
    let m = shape[0];
    let n = shape[1];
    let k = m.min(n);
    let batch_shape = &shape[2..];
    let mut pivot_shape = vec![k];
    pivot_shape.extend_from_slice(batch_shape);
    let parity_shape = batch_shape.to_vec();
    let parity = fill_one_device_tensor(rt, &parity_shape, "lu_factor")?;
    Ok((
        alloc_output(rt, shape)?,
        alloc_output(rt, &pivot_shape)?,
        parity,
    ))
}

fn raw_stream(rt: &CudaRuntime, op: &'static str) -> Result<CudaStream> {
    let mut stream = 0;
    with_raw_cuda_stream(rt, op, |value| stream = value)?;
    Ok(stream as usize as CudaStream)
}

fn sync_stream(rt: &CudaRuntime, op: &'static str) -> Result<()> {
    let stream = raw_stream(rt, op)? as cudaStream_t;
    // SAFETY: `raw_stream` returns the live CUDA stream owned by this runtime's
    // current context; synchronizing it does not outlive the runtime.
    unsafe { cuda_result::stream::synchronize(stream) }
        .map_err(|err| Error::backend_source(op, err))
}

fn alloc_workspace_bytes(rt: &CudaRuntime, nbytes: usize, op: &'static str) -> Result<Workspace> {
    alloc_device_bytes(rt, nbytes, op).map(Workspace::from_device)
}

fn alloc_workspace_elems<T>(rt: &CudaRuntime, len: i32, op: &'static str) -> Result<Workspace>
where
    T: CubeElement + Clone,
{
    let len = usize::try_from(len).map_err(|_| {
        Error::invalid_argument(
            op,
            "workspace_length",
            format!("must be non-negative, got {len}"),
        )
    })?;
    let nbytes = len
        .checked_mul(std::mem::size_of::<T>())
        .ok_or_else(|| Error::invalid_argument(op, "workspace_length", "byte size overflowed"))?;
    alloc_workspace_bytes(rt, nbytes, op)
}

fn typed_device_ptr<T: 'static>(
    rt: &CudaRuntime,
    tensor: &TypedTensor<T>,
    op: &'static str,
) -> Result<*mut c_void> {
    let mut ptr = std::ptr::null_mut();
    interop_with_typed_device_ptr(rt, tensor, op, |value| ptr = value)?;
    Ok(ptr)
}

fn clone_device_tensor<T>(
    rt: &CudaRuntime,
    tensor: &TypedTensor<T>,
    op: &'static str,
) -> Result<TypedTensor<T>>
where
    T: CubeElement + CubePrimitive + Copy + Clone,
{
    let out = alloc_output(rt, tensor.shape())?;
    if out.n_elements() == 0 {
        return Ok(out);
    }
    let src = typed_device_ptr(rt, tensor, op)?;
    let dst = typed_device_ptr(rt, &out, op)?;
    copy_device_to_device(
        rt,
        dst,
        src.cast_const(),
        out.n_elements() * std::mem::size_of::<T>(),
        op,
    )?;
    Ok(out)
}

fn upload_pointer_array(
    rt: &CudaRuntime,
    pointers: &[usize],
    op: &'static str,
) -> Result<Workspace> {
    let nbytes = std::mem::size_of_val(pointers);
    // SAFETY: `pointers` is a live CPU slice of plain usize device-address
    // values; this reinterprets only those initialized bytes for upload.
    let bytes = unsafe { std::slice::from_raw_parts(pointers.as_ptr().cast::<u8>(), nbytes) };
    upload_device_bytes(rt, bytes, op).map(Workspace::from_device)
}

fn download_device_tensor<T>(
    rt: &CudaRuntime,
    tensor: &TypedTensor<T>,
    op: &'static str,
) -> Result<TypedTensor<T>>
where
    T: CubeElement + Clone,
{
    sync_stream(rt, op)?;
    download_typed_tensor(rt, tensor, op)
}

fn copy_device_to_device(
    rt: &CudaRuntime,
    dst: *mut c_void,
    src: *const c_void,
    nbytes: usize,
    op: &'static str,
) -> Result<()> {
    if nbytes == 0 {
        return Ok(());
    }
    let stream = raw_stream(rt, op)? as cudaStream_t;
    // SAFETY: callers pass device pointers obtained from live tensors or
    // workspaces, `nbytes` is checked before zero-size return, and the stream is current.
    unsafe { cuda_result::memcpy_dtod_async(dst, src, nbytes, stream) }
        .map_err(|err| Error::backend_source(op, err))
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

fn check_solver_info_tensor(
    rt: &CudaRuntime,
    info: &TypedTensor<i32>,
    op: &'static str,
    call: &'static str,
) -> Result<()> {
    let host_info = download_device_tensor(rt, info, op)?;
    for &value in host_info.host_data()? {
        check_solver_info(op, call, value)?;
    }
    Ok(())
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
    rt: &CudaRuntime,
    input: &Tensor,
    op: &'static str,
) -> Result<Tensor> {
    match input {
        Tensor::F32(t) => Ok(Tensor::F32(alloc_output(rt, t.shape())?)),
        Tensor::F64(t) => Ok(Tensor::F64(alloc_output(rt, t.shape())?)),
        Tensor::C32(t) => Ok(Tensor::C32(alloc_output(rt, t.shape())?)),
        Tensor::C64(t) => Ok(Tensor::C64(alloc_output(rt, t.shape())?)),
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
        Tensor::C32(t) => complex32_magnitude(backend.runtime(), t).map(Tensor::F32),
        Tensor::C64(t) => complex64_magnitude(backend.runtime(), t).map(Tensor::F64),
        Tensor::I32(_) | Tensor::I64(_) | Tensor::Bool(_) => {
            Err(unsupported_linalg_dtype("validate_nonsingular_gpu", diag))
        }
    }
}

fn complex32_magnitude(
    rt: &CudaRuntime,
    input: &TypedTensor<Complex32>,
) -> Result<TypedTensor<f32>> {
    let output = alloc_output::<f32>(rt, input.shape())?;
    let input_arg = typed_tensor_array_arg(input, "validate_nonsingular_gpu")?;
    let output_arg = typed_tensor_array_arg(&output, "validate_nonsingular_gpu")?;
    if output.n_elements() == 0 {
        return Ok(output);
    }
    let launch_count = cube_count_for_len(output.n_elements())?;
    // SAFETY: bindings describe live CUDA tensors, and `launch_count` covers
    // exactly the complex32 magnitude output domain.
    with_cubecl_client(rt, |client| unsafe {
        cubecl_linalg::complex32_magnitude::launch_unchecked::<CubeclCudaRuntime>(
            client,
            launch_count,
            cube_dim_1d(),
            output_arg,
            input_arg,
        );
    });
    flush_cubecl_client(rt, "validate_nonsingular_gpu")?;
    Ok(output)
}

fn complex64_magnitude(
    rt: &CudaRuntime,
    input: &TypedTensor<Complex64>,
) -> Result<TypedTensor<f64>> {
    let output = alloc_output::<f64>(rt, input.shape())?;
    let input_arg = typed_tensor_array_arg(input, "validate_nonsingular_gpu")?;
    let output_arg = typed_tensor_array_arg(&output, "validate_nonsingular_gpu")?;
    if output.n_elements() == 0 {
        return Ok(output);
    }
    let launch_count = cube_count_for_len(output.n_elements())?;
    // SAFETY: bindings describe live CUDA tensors, and `launch_count` covers
    // exactly the complex64 magnitude output domain.
    with_cubecl_client(rt, |client| unsafe {
        cubecl_linalg::complex64_magnitude::launch_unchecked::<CubeclCudaRuntime>(
            client,
            launch_count,
            cube_dim_1d(),
            output_arg,
            input_arg,
        );
    });
    flush_cubecl_client(rt, "validate_nonsingular_gpu")?;
    Ok(output)
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
