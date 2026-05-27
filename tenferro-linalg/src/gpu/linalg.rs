use std::ffi::c_void;
use std::ops::Neg;
use std::os::raw::c_char;

use cubecl::prelude::{CubeElement, CubePrimitive};
use cubecl_cuda::CudaRuntime;
use cudarc::runtime::{result as cuda_result, sys::cudaStream_t};
use num_complex::{Complex32, Complex64};
use num_traits::{One, Zero};

use super::ffi::cusolver::{
    CublasDiagType, CublasFillMode, CublasOperation, CublasSideMode, CudaDataType,
    CudaLinalgHandles, CudaStream, CusolverEigMode,
};
use super::kernels as cubecl_linalg;
use tenferro_gpu::cubecl::dispatch::{
    alloc_output, cube_count_for_len, cube_dim_1d, cubecl_buffer, typed_from_cubecl,
    typed_tensor_array_arg, typed_tensor_binding,
};
// validate_nonsingular_gpu uses backend ops (extract_diagonal, abs, reduce_min)
// then downloads a single scalar — no bulk host roundtrip.
use tenferro_gpu::cubecl::{
    download_tensor, CubeclBackend, CubeclRuntime, CudaExtensionCacheGuard,
};
use tenferro_gpu::CubeclBuffer;
use tenferro_tensor::config::{DotGeneralConfig, SliceConfig};
use tenferro_tensor::{
    Buffer, DType, Error, Tensor, TensorDot, TensorElementwise, TensorReduction, TensorStructural,
    TypedTensor,
};

type Result<T> = tenferro_tensor::Result<T>;

trait LinalgScalar:
    CubeElement + CubePrimitive + Copy + Clone + One + Zero + Neg<Output = Self>
{
    type Real: CubeElement + CubePrimitive + Copy + Clone + Zero;

    const DATA_TYPE: CudaDataType;
    const NEEDS_RWORK: bool;
}

impl LinalgScalar for f32 {
    type Real = f32;

    const DATA_TYPE: CudaDataType = CudaDataType::F32;
    const NEEDS_RWORK: bool = false;
}

impl LinalgScalar for f64 {
    type Real = f64;

    const DATA_TYPE: CudaDataType = CudaDataType::F64;
    const NEEDS_RWORK: bool = false;
}

impl LinalgScalar for Complex32 {
    type Real = f32;

    const DATA_TYPE: CudaDataType = CudaDataType::Complex32;
    const NEEDS_RWORK: bool = true;
}

impl LinalgScalar for Complex64 {
    type Real = f64;

    const DATA_TYPE: CudaDataType = CudaDataType::Complex64;
    const NEEDS_RWORK: bool = true;
}

fn unsupported_linalg_dtype(op: &'static str, input: &Tensor) -> Error {
    Error::backend_failure(op, format!("unsupported dtype {:?}", input.dtype()))
}

fn linalg_handles(
    backend: &CubeclBackend,
) -> Result<CudaExtensionCacheGuard<'_, CudaLinalgHandles>> {
    backend
        .cuda_extension_cache()
        .get_or_try_init(CudaLinalgHandles::load)
}

struct Workspace {
    _handle: Option<cubecl_runtime::server::Handle>,
    ptr: *mut c_void,
}

impl Workspace {
    fn none() -> Self {
        Self {
            _handle: None,
            ptr: std::ptr::null_mut(),
        }
    }
}

pub(super) fn cholesky(backend: &mut CubeclBackend, input: &Tensor) -> Result<Tensor> {
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
    backend: &mut CubeclBackend,
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
        _ if a.dtype() != b.dtype() => Err(Error::DTypeMismatch {
            op: "triangular_solve",
            lhs: a.dtype(),
            rhs: b.dtype(),
        }),
        _ => Err(Error::backend_failure(
            "triangular_solve",
            format!("unsupported dtype {:?}", a.dtype()),
        )),
    }
}

pub(super) fn lu(backend: &mut CubeclBackend, input: &Tensor) -> Result<Vec<Tensor>> {
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

pub(super) fn full_piv_lu(_backend: &mut CubeclBackend, _input: &Tensor) -> Result<Vec<Tensor>> {
    Err(Error::backend_failure(
        "full_piv_lu",
        "complete-pivoting LU is not implemented for the CubeCL backend",
    ))
}

pub(super) fn full_piv_lu_solve(
    _backend: &mut CubeclBackend,
    _a: &Tensor,
    _b: &Tensor,
    _transpose_a: bool,
) -> Result<Tensor> {
    Err(Error::backend_failure(
        "full_piv_lu_solve",
        "complete-pivoting LU solve is not implemented for the CubeCL backend",
    ))
}

pub(super) fn svd(backend: &mut CubeclBackend, input: &Tensor) -> Result<Vec<Tensor>> {
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

pub(super) fn qr(backend: &mut CubeclBackend, input: &Tensor) -> Result<Vec<Tensor>> {
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

pub(super) fn eigh(backend: &mut CubeclBackend, input: &Tensor) -> Result<Vec<Tensor>> {
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

pub(super) fn eig(_backend: &mut CubeclBackend, _input: &Tensor) -> Result<Vec<Tensor>> {
    Err(Error::backend_failure(
        "eig",
        "non-symmetric eigendecomposition is not supported on the CubeCL GPU backend \
                  because cuSOLVER does not provide it. Download to CPU explicitly via \
                  `backend.download_to_host(&gpu_tensor)?` and then call `CpuBackend::eig`."
            .to_string(),
    ))
}

pub(super) fn solve(backend: &mut CubeclBackend, a: &Tensor, b: &Tensor) -> Result<Tensor> {
    if has_zero_dim(a.shape()) || has_zero_dim(b.shape()) {
        return Ok(zeros_like_tensor(b));
    }

    let (rhs, restore_shape) = if let Some(matrix_rhs_shape) = batched_vector_rhs_shape(a, b) {
        (
            backend.reshape(b, &matrix_rhs_shape)?,
            Some(b.shape().to_vec()),
        )
    } else {
        (b.clone(), None)
    };

    let outputs = lu(backend, a)?;
    let p = &outputs[0];
    let l = &outputs[1];
    let u = &outputs[2];
    validate_nonsingular_gpu(backend, u)?;

    let pb = matmul_preserve_trailing_batch(backend, p, &rhs)?;
    let z = triangular_solve(backend, l, &pb, true, true, false, true)?;
    let x = triangular_solve(backend, u, &z, true, false, false, false)?;
    if let Some(shape) = restore_shape {
        backend.reshape(&x, &shape)
    } else {
        Ok(x)
    }
}

fn cholesky_typed<T>(backend: &CubeclBackend, input: &TypedTensor<T>) -> Result<TypedTensor<T>>
where
    T: LinalgScalar,
{
    const OP: &str = "cholesky";

    backend.runtime().set_current_cuda_context(OP)?;
    let n = square_matrix_dim(OP, &input.shape)?;
    if has_zero_dim(&input.shape) {
        return Ok(alloc_output(backend.runtime(), &input.shape));
    }

    let work = clone_device_tensor(backend.runtime(), input, OP)?;
    let handles = linalg_handles(backend)?;
    let stream = raw_stream(backend.runtime(), OP)?;
    handles.cusolver().set_stream(stream, OP)?;

    let batch_total = batch_count(&input.shape[2..]);
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
    let info = alloc_workspace_bytes(backend.runtime(), std::mem::size_of::<i32>(), OP)?;
    let matrix_stride = n * n;

    for batch in 0..batch_total {
        let batch_ptr = unsafe { batch_ptr::<T>(first_ptr, batch * matrix_stride) };
        unsafe {
            handles.cusolver().potrf(
                T::DATA_TYPE,
                CublasFillMode::Lower,
                n_i32,
                batch_ptr,
                lda,
                workspace.ptr,
                lwork,
                info.ptr.cast::<i32>(),
                OP,
            )?;
        }
        let mut host_info = [0_i32; 1];
        copy_device_to_host(backend.runtime(), &mut host_info, info.ptr.cast_const(), OP)?;
        match host_info[0] {
            0 => {}
            x if x < 0 => {
                return Err(Error::backend_failure(
                    OP,
                    format!("cusolverDn*potrf reported invalid parameter {}", -x),
                ));
            }
            x => {
                return Err(Error::backend_failure(
                    OP,
                    format!("matrix is not positive definite (minor {x})"),
                ));
            }
        }
    }

    backend.tril_typed(&work, 0)
}

fn triangular_solve_typed<T>(
    backend: &CubeclBackend,
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
    const OP: &str = "triangular_solve";

    backend.runtime().set_current_cuda_context(OP)?;
    let n = square_matrix_dim(OP, &a.shape)?;
    validate_triangular_rhs(OP, &a.shape, &b.shape, left_side)?;
    if has_zero_dim(&a.shape) || has_zero_dim(&b.shape) {
        return Ok(alloc_output(backend.runtime(), &b.shape));
    }

    let out = clone_device_tensor(backend.runtime(), b, OP)?;
    let handles = linalg_handles(backend)?;
    let stream = raw_stream(backend.runtime(), OP)?;
    handles.cublas().set_stream(stream, OP)?;

    let a_ptr = typed_device_ptr(backend.runtime(), a, OP)?;
    let out_ptr = typed_device_ptr(backend.runtime(), &out, OP)?;
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
    let trans = if transpose_a {
        CublasOperation::T
    } else {
        CublasOperation::N
    };
    let diag = if unit_diagonal {
        CublasDiagType::Unit
    } else {
        CublasDiagType::NonUnit
    };
    let rows = b.shape[0];
    let cols = b.shape[1];
    let a_stride = n * n;
    let out_stride = rows * cols;
    let lda = as_i32(n, OP, "lda")?;
    let ldb = as_i32(rows, OP, "ldb")?;
    let m = as_i32(rows, OP, "m")?;
    let n_rhs = as_i32(cols, OP, "n")?;
    let alpha = T::one();

    for batch in 0..batch_count(&b.shape[2..]) {
        let batch_a = unsafe { batch_const_ptr::<T>(a_ptr.cast_const(), batch * a_stride) };
        let batch_b = unsafe { batch_ptr::<T>(out_ptr, batch * out_stride) };
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
                OP,
            )?;
        }
    }

    Ok(out)
}

fn lu_typed<T>(
    backend: &CubeclBackend,
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

    backend.runtime().set_current_cuda_context(OP)?;
    let (m, n) = matrix_dims(OP, &input.shape)?;
    let k = m.min(n);
    if has_zero_dim(&input.shape) {
        return zero_sized_lu_outputs(backend.runtime(), input.shape.as_slice());
    }

    let work = clone_device_tensor(backend.runtime(), input, OP)?;
    let handles = linalg_handles(backend)?;
    let stream = raw_stream(backend.runtime(), OP)?;
    handles.cusolver().set_stream(stream, OP)?;

    let batch_total = batch_count(&input.shape[2..]);
    let a_ptr = typed_device_ptr(backend.runtime(), &work, OP)?;
    let lda = as_i32(m, OP, "lda")?;
    let m_i32 = as_i32(m, OP, "m")?;
    let n_i32 = as_i32(n, OP, "n")?;
    let lwork = handles
        .cusolver()
        .getrf_buffer_size(T::DATA_TYPE, m_i32, n_i32, a_ptr, lda, OP)?;
    let workspace = alloc_workspace_elems::<T>(backend.runtime(), lwork, OP)?;
    let mut pivot_shape = vec![k];
    pivot_shape.extend_from_slice(&input.shape[2..]);
    let pivots = alloc_output::<u32>(backend.runtime(), &pivot_shape);
    let info = alloc_output::<i32>(backend.runtime(), &[batch_total]);
    let pivots_ptr = typed_device_ptr(backend.runtime(), &pivots, OP)?;
    let info_ptr = typed_device_ptr(backend.runtime(), &info, OP)?;
    let matrix_stride = m * n;

    for batch in 0..batch_total {
        let batch_a = unsafe { batch_ptr::<T>(a_ptr, batch * matrix_stride) };
        let batch_pivots = unsafe { batch_ptr::<u32>(pivots_ptr, batch * k) };
        let batch_info = unsafe { batch_ptr::<i32>(info_ptr, batch) };
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
    for &info_value in host_info.host_data() {
        if info_value < 0 {
            return Err(Error::backend_failure(
                OP,
                format!(
                    "cusolverDn*getrf reported invalid parameter {}",
                    -info_value
                ),
            ));
        }
    }

    build_lu_outputs_device(backend.runtime(), &work, &pivots, m, n, &input.shape[2..])
}

fn svd_typed<T>(
    backend: &CubeclBackend,
    input: &TypedTensor<T>,
) -> Result<(TypedTensor<T>, TypedTensor<T::Real>, TypedTensor<T>)>
where
    T: LinalgScalar,
{
    const OP: &str = "svd";

    backend.runtime().set_current_cuda_context(OP)?;
    let (m, n) = matrix_dims(OP, &input.shape)?;
    let k = m.min(n);
    let batch_shape = &input.shape[2..];
    let mut u_shape = vec![m, k];
    u_shape.extend_from_slice(batch_shape);
    let mut s_shape = vec![k];
    s_shape.extend_from_slice(batch_shape);
    let mut vt_shape = vec![k, n];
    vt_shape.extend_from_slice(batch_shape);
    if has_zero_dim(&input.shape) {
        return Ok((
            alloc_output(backend.runtime(), &u_shape),
            alloc_output(backend.runtime(), &s_shape),
            alloc_output(backend.runtime(), &vt_shape),
        ));
    }

    let work = clone_device_tensor(backend.runtime(), input, OP)?;
    let u = alloc_output::<T>(backend.runtime(), &u_shape);
    let s = alloc_output::<T::Real>(backend.runtime(), &s_shape);
    let vt = alloc_output::<T>(backend.runtime(), &vt_shape);
    let handles = linalg_handles(backend)?;
    let stream = raw_stream(backend.runtime(), OP)?;
    handles.cusolver().set_stream(stream, OP)?;

    let a_ptr = typed_device_ptr(backend.runtime(), &work, OP)?;
    let u_ptr = typed_device_ptr(backend.runtime(), &u, OP)?;
    let s_ptr = typed_device_ptr(backend.runtime(), &s, OP)?;
    let vt_ptr = typed_device_ptr(backend.runtime(), &vt, OP)?;
    let m_i32 = as_i32(m, OP, "m")?;
    let n_i32 = as_i32(n, OP, "n")?;
    let lda = as_i32(m, OP, "lda")?;
    let ldu = as_i32(m, OP, "ldu")?;
    let ldvt = as_i32(k.max(1), OP, "ldvt")?;
    let lwork = handles
        .cusolver()
        .gesvd_buffer_size(T::DATA_TYPE, m_i32, n_i32, OP)?;
    let workspace = alloc_workspace_elems::<T>(backend.runtime(), lwork, OP)?;
    let rwork = if T::NEEDS_RWORK {
        alloc_workspace_elems::<T::Real>(backend.runtime(), as_i32(5 * k, OP, "rwork")?, OP)?
    } else {
        Workspace::none()
    };
    let info = alloc_workspace_bytes(backend.runtime(), std::mem::size_of::<i32>(), OP)?;
    let batch_total = batch_count(batch_shape);
    let a_stride = m * n;
    let u_stride = m * k;
    let s_stride = k;
    let vt_stride = k * n;
    let job = b'S' as c_char;

    for batch in 0..batch_total {
        let batch_a = unsafe { batch_ptr::<T>(a_ptr, batch * a_stride) };
        let batch_s = unsafe { batch_ptr::<T::Real>(s_ptr, batch * s_stride) };
        let batch_u = unsafe { batch_ptr::<T>(u_ptr, batch * u_stride) };
        let batch_vt = unsafe { batch_ptr::<T>(vt_ptr, batch * vt_stride) };
        unsafe {
            handles.cusolver().gesvd(
                T::DATA_TYPE,
                job,
                job,
                m_i32,
                n_i32,
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
                info.ptr.cast::<i32>(),
                OP,
            )?;
        }
        let mut host_info = [0_i32; 1];
        copy_device_to_host(backend.runtime(), &mut host_info, info.ptr.cast_const(), OP)?;
        check_solver_info(OP, "cusolverDn*gesvd", host_info[0])?;
    }

    Ok((u, s, vt))
}

fn qr_typed<T>(
    backend: &CubeclBackend,
    input: &TypedTensor<T>,
) -> Result<(TypedTensor<T>, TypedTensor<T>)>
where
    T: LinalgScalar,
{
    const OP: &str = "qr";

    backend.runtime().set_current_cuda_context(OP)?;
    let (m, n) = matrix_dims(OP, &input.shape)?;
    let k = m.min(n);
    let batch_shape = &input.shape[2..];
    let mut q_shape = vec![m, k];
    q_shape.extend_from_slice(batch_shape);
    let mut r_shape = vec![k, n];
    r_shape.extend_from_slice(batch_shape);
    if has_zero_dim(&input.shape) {
        return Ok((
            alloc_output(backend.runtime(), &q_shape),
            alloc_output(backend.runtime(), &r_shape),
        ));
    }

    let work = clone_device_tensor(backend.runtime(), input, OP)?;
    let q = alloc_output::<T>(backend.runtime(), &q_shape);
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
    let tau = alloc_workspace_bytes(backend.runtime(), k * std::mem::size_of::<T>(), OP)?;
    let info = alloc_workspace_bytes(backend.runtime(), std::mem::size_of::<i32>(), OP)?;
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
    let batch_total = batch_count(batch_shape);
    let work_stride = m * n;
    let q_stride = m * k;

    for batch in 0..batch_total {
        let batch_work = unsafe { batch_ptr::<T>(work_ptr, batch * work_stride) };
        let batch_q = unsafe { batch_ptr::<T>(q_ptr, batch * q_stride) };
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
                info.ptr.cast::<i32>(),
                OP,
            )?;
        }
        let mut host_info = [0_i32; 1];
        copy_device_to_host(backend.runtime(), &mut host_info, info.ptr.cast_const(), OP)?;
        check_solver_info(OP, "cusolverDn*geqrf", host_info[0])?;

        copy_device_to_device(
            backend.runtime(),
            batch_q,
            batch_work.cast_const(),
            q_stride * std::mem::size_of::<T>(),
            OP,
        )?;
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
                info.ptr.cast::<i32>(),
                OP,
            )?;
        }
        copy_device_to_host(backend.runtime(), &mut host_info, info.ptr.cast_const(), OP)?;
        check_solver_info(OP, "cusolverDn*orgqr", host_info[0])?;
    }

    let r_input = backend.slice_typed(&work, &matrix_slice_config(&input.shape, k, n))?;
    let r = backend.triu_typed(&r_input, 0)?;
    Ok((q, r))
}

fn eigh_typed<T>(
    backend: &CubeclBackend,
    input: &TypedTensor<T>,
) -> Result<(TypedTensor<T::Real>, TypedTensor<T>)>
where
    T: LinalgScalar,
{
    const OP: &str = "eigh";

    backend.runtime().set_current_cuda_context(OP)?;
    let n = square_matrix_dim(OP, &input.shape)?;
    let batch_shape = &input.shape[2..];
    let mut values_shape = vec![n];
    values_shape.extend_from_slice(batch_shape);
    if has_zero_dim(&input.shape) {
        return Ok((
            alloc_output(backend.runtime(), &values_shape),
            alloc_output(backend.runtime(), &input.shape),
        ));
    }

    let work = clone_device_tensor(backend.runtime(), input, OP)?;
    let values = alloc_output::<T::Real>(backend.runtime(), &values_shape);
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
    let info = alloc_workspace_bytes(backend.runtime(), std::mem::size_of::<i32>(), OP)?;
    let batch_total = batch_count(batch_shape);
    let matrix_stride = n * n;
    let values_stride = n;

    for batch in 0..batch_total {
        let batch_a = unsafe { batch_ptr::<T>(a_ptr, batch * matrix_stride) };
        let batch_w = unsafe { batch_ptr::<T::Real>(values_ptr, batch * values_stride) };
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
                info.ptr.cast::<i32>(),
                OP,
            )?;
        }
        let mut host_info = [0_i32; 1];
        copy_device_to_host(backend.runtime(), &mut host_info, info.ptr.cast_const(), OP)?;
        check_solver_info(OP, "cusolverDn*syevd", host_info[0])?;
    }

    Ok((values, work))
}

fn build_lu_outputs_device<T>(
    rt: &CubeclRuntime,
    lu: &TypedTensor<T>,
    pivots: &TypedTensor<u32>,
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

    let p = alloc_output::<T>(rt, &p_shape);
    let l = alloc_output::<T>(rt, &l_shape);
    let u = alloc_output::<T>(rt, &u_shape);
    let parity = alloc_output::<T>(rt, &parity_shape);
    let client = rt.client();
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
    unsafe {
        cubecl_linalg::lu_extract_outputs::launch_unchecked::<T, CudaRuntime>(
            client,
            cube_count_for_len(launch_len),
            cube_dim_1d(),
            p_arg.into_tensor_arg(),
            l_arg.into_tensor_arg(),
            u_arg.into_tensor_arg(),
            parity_arg.into_tensor_arg(),
            work_arg.into_tensor_arg(),
            pivots_arg,
            k,
            lu.shape.len(),
        );
    }
    client.flush().map_err(|err| {
        Error::backend_failure("lu", format!("LU output extraction launch failed: {err:?}"))
    })?;

    Ok((p, l, u, parity))
}

fn zero_sized_lu_outputs<T>(
    rt: &CubeclRuntime,
    shape: &[usize],
) -> Result<(
    TypedTensor<T>,
    TypedTensor<T>,
    TypedTensor<T>,
    TypedTensor<T>,
)>
where
    T: LinalgScalar,
{
    let m = shape[0];
    let n = shape[1];
    let k = m.min(n);
    let batch_shape = &shape[2..];
    let mut p_shape = vec![m, m];
    p_shape.extend_from_slice(batch_shape);
    let mut l_shape = vec![m, k];
    l_shape.extend_from_slice(batch_shape);
    let mut u_shape = vec![k, n];
    u_shape.extend_from_slice(batch_shape);
    let parity_shape = batch_shape.to_vec();
    let parity_len = batch_count(batch_shape);
    let parity = upload_host_tensor(
        rt,
        TypedTensor::from_vec_col_major(parity_shape, vec![T::one(); parity_len.max(1)]),
    )?;
    Ok((
        alloc_output(rt, &p_shape),
        alloc_output(rt, &l_shape),
        alloc_output(rt, &u_shape),
        parity,
    ))
}

fn raw_stream(rt: &CubeclRuntime, op: &'static str) -> Result<CudaStream> {
    rt.raw_cuda_stream()
        .map(|stream| stream as usize as CudaStream)
        .map_err(|err| Error::backend_failure(op, err.to_string()))
}

fn sync_stream(rt: &CubeclRuntime, op: &'static str) -> Result<()> {
    let stream = raw_stream(rt, op)? as cudaStream_t;
    unsafe { cuda_result::stream::synchronize(stream) }.map_err(|err| {
        Error::backend_failure(op, format!("CUDA stream synchronize failed: {err:?}"))
    })
}

fn alloc_workspace_bytes(
    rt: &CubeclRuntime,
    nbytes: usize,
    _op: &'static str,
) -> Result<Workspace> {
    if nbytes == 0 {
        return Ok(Workspace::none());
    }
    let handle = rt.client().empty(nbytes);
    let resource = rt.client().get_resource(handle.clone()).map_err(|err| {
        Error::backend_failure(
            "cubecl_linalg",
            format!("failed to obtain workspace resource: {err:?}"),
        )
    })?;
    Ok(Workspace {
        _handle: Some(handle),
        ptr: resource.resource().ptr as usize as *mut c_void,
    })
}

fn alloc_workspace_elems<T>(rt: &CubeclRuntime, len: i32, op: &'static str) -> Result<Workspace>
where
    T: CubeElement + Clone,
{
    let len = usize::try_from(len)
        .map_err(|_| Error::backend_failure(op, format!("workspace length was negative: {len}")))?;
    alloc_workspace_bytes(rt, len * std::mem::size_of::<T>(), op)
}

fn typed_device_ptr<T: 'static>(
    rt: &CubeclRuntime,
    tensor: &TypedTensor<T>,
    op: &'static str,
) -> Result<*mut c_void> {
    let buffer = cubecl_buffer(tensor, op)?;
    let resource = rt
        .client()
        .get_resource(buffer.handle.clone())
        .map_err(|err| {
            Error::backend_failure(op, format!("failed to obtain CubeCL resource: {err:?}"))
        })?;
    Ok(resource.resource().ptr as usize as *mut c_void)
}

fn clone_device_tensor<T>(
    rt: &CubeclRuntime,
    tensor: &TypedTensor<T>,
    op: &'static str,
) -> Result<TypedTensor<T>>
where
    T: CubeElement + CubePrimitive + Copy + Clone,
{
    let out = alloc_output(rt, &tensor.shape);
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

fn upload_host_tensor<T>(rt: &CubeclRuntime, tensor: TypedTensor<T>) -> Result<TypedTensor<T>>
where
    T: CubeElement + Clone + Send + Sync + 'static,
{
    let (shape, data) = match tensor.buffer {
        Buffer::Host(data) => (tensor.shape, data),
        Buffer::Backend(_) => {
            return Err(Error::backend_failure(
                "cubecl_linalg",
                "upload_host_tensor expects host-backed tensor".to_string(),
            ));
        }
    };
    let len = data.len();
    let handle = rt.client().create_from_slice(T::as_bytes(&data));
    Ok(typed_from_cubecl(
        shape,
        CubeclBuffer::new(handle, len),
        rt.device_ordinal(),
    ))
}

fn download_device_tensor<T>(
    rt: &CubeclRuntime,
    tensor: &TypedTensor<T>,
    op: &'static str,
) -> Result<TypedTensor<T>>
where
    T: CubeElement + Clone,
{
    sync_stream(rt, op)?;
    let buffer = cubecl_buffer(tensor, op)?;
    let bytes = rt
        .client()
        .read_one(buffer.handle.clone())
        .map_err(|err| Error::backend_failure(op, format!("failed to download tensor: {err:?}")))?;
    Ok(TypedTensor::from_vec_col_major(
        tensor.shape.clone(),
        T::from_bytes(&bytes).to_vec(),
    ))
}

fn copy_device_to_device(
    rt: &CubeclRuntime,
    dst: *mut c_void,
    src: *const c_void,
    nbytes: usize,
    op: &'static str,
) -> Result<()> {
    if nbytes == 0 {
        return Ok(());
    }
    let stream = raw_stream(rt, op)? as cudaStream_t;
    unsafe { cuda_result::memcpy_dtod_async(dst, src, nbytes, stream) }
        .map_err(|err| Error::backend_failure(op, format!("cudaMemcpyAsync DtoD failed: {err:?}")))
}

fn copy_device_to_host<T: Copy>(
    rt: &CubeclRuntime,
    dst: &mut [T],
    src: *const c_void,
    op: &'static str,
) -> Result<()> {
    let stream = raw_stream(rt, op)? as cudaStream_t;
    unsafe { cuda_result::memcpy_dtoh_async(dst, src, stream) }.map_err(|err| {
        Error::backend_failure(op, format!("cudaMemcpyAsync DtoH failed: {err:?}"))
    })?;
    unsafe { cuda_result::stream::synchronize(stream) }.map_err(|err| {
        Error::backend_failure(op, format!("CUDA stream synchronize failed: {err:?}"))
    })
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
        return Err(Error::RankMismatch {
            op,
            expected: 2,
            actual: shape.len(),
        });
    }
    Ok((shape[0], shape[1]))
}

fn square_matrix_dim(op: &'static str, shape: &[usize]) -> Result<usize> {
    let (rows, cols) = matrix_dims(op, shape)?;
    if rows != cols {
        return Err(Error::InvalidConfig {
            op,
            message: format!("expected square matrix, got shape {shape:?}"),
        });
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
        return Err(Error::ShapeMismatch {
            op,
            lhs: a_shape.to_vec(),
            rhs: b_shape.to_vec(),
        });
    }
    if left_side && b_rows != n {
        return Err(Error::InvalidConfig {
            op,
            message: format!("rhs row count mismatch: expected {n}, got {b_rows}"),
        });
    }
    if !left_side && b_cols != n {
        return Err(Error::InvalidConfig {
            op,
            message: format!("rhs column count mismatch: expected {n}, got {b_cols}"),
        });
    }
    Ok(())
}

fn check_solver_info(op: &'static str, call: &'static str, info: i32) -> Result<()> {
    if info == 0 {
        return Ok(());
    }
    if info < 0 {
        return Err(Error::backend_failure(
            op,
            format!("{call} reported invalid parameter {}", -info),
        ));
    }
    Err(Error::backend_failure(
        op,
        format!("{call} failed with info={info}"),
    ))
}

fn has_zero_dim(shape: &[usize]) -> bool {
    shape.contains(&0)
}

fn batch_count(batch_shape: &[usize]) -> usize {
    batch_shape.iter().product::<usize>().max(1)
}

fn as_i32(value: usize, op: &'static str, label: &'static str) -> Result<i32> {
    i32::try_from(value)
        .map_err(|_| Error::backend_failure(op, format!("{label} does not fit in i32: {value}")))
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

fn zeros_like_tensor(input: &Tensor) -> Tensor {
    match input {
        Tensor::F32(t) => Tensor::F32(TypedTensor::zeros(t.shape.clone())),
        Tensor::F64(t) => Tensor::F64(TypedTensor::zeros(t.shape.clone())),
        Tensor::I32(t) => Tensor::I32(TypedTensor::zeros(t.shape.clone())),
        Tensor::I64(t) => Tensor::I64(TypedTensor::zeros(t.shape.clone())),
        Tensor::Bool(t) => Tensor::Bool(TypedTensor::from_vec_col_major(
            t.shape.clone(),
            vec![false; t.n_elements()],
        )),
        Tensor::C32(t) => Tensor::C32(TypedTensor::zeros(t.shape.clone())),
        Tensor::C64(t) => Tensor::C64(TypedTensor::zeros(t.shape.clone())),
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

fn matmul_preserve_trailing_batch(
    backend: &mut CubeclBackend,
    lhs: &Tensor,
    rhs: &Tensor,
) -> Result<Tensor> {
    let rank = lhs.shape().len();
    let batch_dims: Vec<usize> = (2..rank).collect();
    backend.dot_general(
        lhs,
        rhs,
        &DotGeneralConfig {
            lhs_contracting_dims: vec![1],
            rhs_contracting_dims: vec![0],
            lhs_batch_dims: batch_dims.clone(),
            rhs_batch_dims: batch_dims,
        },
    )
}

/// Validate that U's diagonal has no singular/zero entries — GPU-accelerated.
///
/// Strategy: extract_diagonal → abs → reshape to 1D → reduce_min(axis=0) →
/// download 1 scalar → check > 0.
///
/// Only the final scalar (8 bytes) is transferred to host.
fn validate_nonsingular_gpu(backend: &mut CubeclBackend, u: &Tensor) -> Result<()> {
    let diag = backend.extract_diagonal(u, 0, 1)?;

    // abs — convert complex to real first (complex abs not supported)
    let abs_diag = match &diag {
        Tensor::C32(_) | Tensor::C64(_) => {
            let real_diag = backend.convert(&diag, DType::F64)?;
            backend.abs(&real_diag)?
        }
        _ => backend.abs(&diag)?,
    };

    // Flatten to 1D then reduce_min on axis 0 to get a single scalar.
    let total: usize = abs_diag.shape().iter().product();
    let flat = backend.reshape(&abs_diag, &[total])?;
    let min_val = backend.reduce_min(&flat, &[0])?;

    // Download single scalar
    let host_min = download_tensor(backend.runtime(), &min_val)?;
    let is_singular = match &host_min {
        Tensor::F64(t) => t.host_data()[0] == 0.0 || !t.host_data()[0].is_finite(),
        Tensor::F32(t) => t.host_data()[0] == 0.0 || !t.host_data()[0].is_finite(),
        _ => {
            return Err(Error::backend_failure(
                "solve",
                "unexpected dtype after abs reduction",
            ));
        }
    };

    if is_singular {
        Err(Error::backend_failure(
            "solve",
            "singular matrix: zero or non-finite diagonal entry in U",
        ))
    } else {
        Ok(())
    }
}
