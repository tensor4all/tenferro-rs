use std::ffi::c_void;
use std::os::raw::c_char;
use std::sync::Arc;

use libloading::Library;
use num_complex::{Complex32, Complex64};
use tenferro_device::{Error, Result};

use super::runtime::{check_cublas_status, check_cusolver_status};
use super::scalar_type::CudaDataType;

pub(super) type CublasStatus = i32;
pub(super) type CusolverStatus = i32;
pub(super) type CublasOperation = i32;

type CublasHandle = *mut c_void;
type CusolverDnHandle = *mut c_void;
type CudaStream = *mut c_void;

pub(super) const CUBLAS_OP_N: CublasOperation = 0;
pub(super) const CUBLAS_SIDE_LEFT: i32 = 0;
pub(super) const CUBLAS_FILL_MODE_LOWER: i32 = 0;
pub(super) const CUBLAS_FILL_MODE_UPPER: i32 = 1;
pub(super) const CUBLAS_DIAG_NON_UNIT: i32 = 0;

type FnCublasCreate = unsafe extern "C" fn(*mut CublasHandle) -> CublasStatus;
type FnCublasDestroy = unsafe extern "C" fn(CublasHandle) -> CublasStatus;
type FnCublasSetStream = unsafe extern "C" fn(CublasHandle, CudaStream) -> CublasStatus;
type FnCublasStrsm = unsafe extern "C" fn(
    CublasHandle,
    i32,
    i32,
    CublasOperation,
    i32,
    i32,
    i32,
    *const f32,
    *const f32,
    i32,
    *mut f32,
    i32,
) -> CublasStatus;
type FnCublasDtrsm = unsafe extern "C" fn(
    CublasHandle,
    i32,
    i32,
    CublasOperation,
    i32,
    i32,
    i32,
    *const f64,
    *const f64,
    i32,
    *mut f64,
    i32,
) -> CublasStatus;

type FnCusolverDnCreate = unsafe extern "C" fn(*mut CusolverDnHandle) -> CusolverStatus;
type FnCusolverDnDestroy = unsafe extern "C" fn(CusolverDnHandle) -> CusolverStatus;
type FnCusolverDnSetStream = unsafe extern "C" fn(CusolverDnHandle, CudaStream) -> CusolverStatus;
type FnCusolverDnSgetrfBufferSize =
    unsafe extern "C" fn(CusolverDnHandle, i32, i32, *mut f32, i32, *mut i32) -> CusolverStatus;
type FnCusolverDnDgetrfBufferSize =
    unsafe extern "C" fn(CusolverDnHandle, i32, i32, *mut f64, i32, *mut i32) -> CusolverStatus;
type FnCusolverDnCgetrfBufferSize = unsafe extern "C" fn(
    CusolverDnHandle,
    i32,
    i32,
    *mut Complex32,
    i32,
    *mut i32,
) -> CusolverStatus;
type FnCusolverDnZgetrfBufferSize = unsafe extern "C" fn(
    CusolverDnHandle,
    i32,
    i32,
    *mut Complex64,
    i32,
    *mut i32,
) -> CusolverStatus;
type FnCusolverDnSgeqrfBufferSize =
    unsafe extern "C" fn(CusolverDnHandle, i32, i32, *mut f32, i32, *mut i32) -> CusolverStatus;
type FnCusolverDnDgeqrfBufferSize =
    unsafe extern "C" fn(CusolverDnHandle, i32, i32, *mut f64, i32, *mut i32) -> CusolverStatus;
type FnCusolverDnCgeqrfBufferSize = unsafe extern "C" fn(
    CusolverDnHandle,
    i32,
    i32,
    *mut Complex32,
    i32,
    *mut i32,
) -> CusolverStatus;
type FnCusolverDnZgeqrfBufferSize = unsafe extern "C" fn(
    CusolverDnHandle,
    i32,
    i32,
    *mut Complex64,
    i32,
    *mut i32,
) -> CusolverStatus;
type FnCusolverDnSgesvdBufferSize =
    unsafe extern "C" fn(CusolverDnHandle, i32, i32, *mut i32) -> CusolverStatus;
type FnCusolverDnDgesvdBufferSize =
    unsafe extern "C" fn(CusolverDnHandle, i32, i32, *mut i32) -> CusolverStatus;
type FnCusolverDnSgetrf = unsafe extern "C" fn(
    CusolverDnHandle,
    i32,
    i32,
    *mut f32,
    i32,
    *mut f32,
    *mut i32,
    *mut i32,
) -> CusolverStatus;
type FnCusolverDnDgetrf = unsafe extern "C" fn(
    CusolverDnHandle,
    i32,
    i32,
    *mut f64,
    i32,
    *mut f64,
    *mut i32,
    *mut i32,
) -> CusolverStatus;
type FnCusolverDnCgetrf = unsafe extern "C" fn(
    CusolverDnHandle,
    i32,
    i32,
    *mut Complex32,
    i32,
    *mut Complex32,
    *mut i32,
    *mut i32,
) -> CusolverStatus;
type FnCusolverDnZgetrf = unsafe extern "C" fn(
    CusolverDnHandle,
    i32,
    i32,
    *mut Complex64,
    i32,
    *mut Complex64,
    *mut i32,
    *mut i32,
) -> CusolverStatus;
type FnCusolverDnSgetrs = unsafe extern "C" fn(
    CusolverDnHandle,
    CublasOperation,
    i32,
    i32,
    *const f32,
    i32,
    *const i32,
    *mut f32,
    i32,
    *mut i32,
) -> CusolverStatus;
type FnCusolverDnDgetrs = unsafe extern "C" fn(
    CusolverDnHandle,
    CublasOperation,
    i32,
    i32,
    *const f64,
    i32,
    *const i32,
    *mut f64,
    i32,
    *mut i32,
) -> CusolverStatus;
type FnCusolverDnCgetrs = unsafe extern "C" fn(
    CusolverDnHandle,
    CublasOperation,
    i32,
    i32,
    *const Complex32,
    i32,
    *const i32,
    *mut Complex32,
    i32,
    *mut i32,
) -> CusolverStatus;
type FnCusolverDnZgetrs = unsafe extern "C" fn(
    CusolverDnHandle,
    CublasOperation,
    i32,
    i32,
    *const Complex64,
    i32,
    *const i32,
    *mut Complex64,
    i32,
    *mut i32,
) -> CusolverStatus;
type FnCusolverDnSgeqrf = unsafe extern "C" fn(
    CusolverDnHandle,
    i32,
    i32,
    *mut f32,
    i32,
    *mut f32,
    *mut f32,
    i32,
    *mut i32,
) -> CusolverStatus;
type FnCusolverDnDgeqrf = unsafe extern "C" fn(
    CusolverDnHandle,
    i32,
    i32,
    *mut f64,
    i32,
    *mut f64,
    *mut f64,
    i32,
    *mut i32,
) -> CusolverStatus;
type FnCusolverDnCgeqrf = unsafe extern "C" fn(
    CusolverDnHandle,
    i32,
    i32,
    *mut Complex32,
    i32,
    *mut Complex32,
    *mut Complex32,
    i32,
    *mut i32,
) -> CusolverStatus;
type FnCusolverDnZgeqrf = unsafe extern "C" fn(
    CusolverDnHandle,
    i32,
    i32,
    *mut Complex64,
    i32,
    *mut Complex64,
    *mut Complex64,
    i32,
    *mut i32,
) -> CusolverStatus;
type FnCusolverDnSorgqrBufferSize = unsafe extern "C" fn(
    CusolverDnHandle,
    i32,
    i32,
    i32,
    *mut f32,
    i32,
    *mut f32,
    *mut i32,
) -> CusolverStatus;
type FnCusolverDnDorgqrBufferSize = unsafe extern "C" fn(
    CusolverDnHandle,
    i32,
    i32,
    i32,
    *mut f64,
    i32,
    *mut f64,
    *mut i32,
) -> CusolverStatus;
type FnCusolverDnCungqrBufferSize = unsafe extern "C" fn(
    CusolverDnHandle,
    i32,
    i32,
    i32,
    *mut Complex32,
    i32,
    *mut Complex32,
    *mut i32,
) -> CusolverStatus;
type FnCusolverDnZungqrBufferSize = unsafe extern "C" fn(
    CusolverDnHandle,
    i32,
    i32,
    i32,
    *mut Complex64,
    i32,
    *mut Complex64,
    *mut i32,
) -> CusolverStatus;
type FnCusolverDnSorgqr = unsafe extern "C" fn(
    CusolverDnHandle,
    i32,
    i32,
    i32,
    *mut f32,
    i32,
    *mut f32,
    *mut f32,
    i32,
    *mut i32,
) -> CusolverStatus;
type FnCusolverDnDorgqr = unsafe extern "C" fn(
    CusolverDnHandle,
    i32,
    i32,
    i32,
    *mut f64,
    i32,
    *mut f64,
    *mut f64,
    i32,
    *mut i32,
) -> CusolverStatus;
type FnCusolverDnCungqr = unsafe extern "C" fn(
    CusolverDnHandle,
    i32,
    i32,
    i32,
    *mut Complex32,
    i32,
    *mut Complex32,
    *mut Complex32,
    i32,
    *mut i32,
) -> CusolverStatus;
type FnCusolverDnZungqr = unsafe extern "C" fn(
    CusolverDnHandle,
    i32,
    i32,
    i32,
    *mut Complex64,
    i32,
    *mut Complex64,
    *mut Complex64,
    i32,
    *mut i32,
) -> CusolverStatus;
type FnCusolverDnSpotrfBufferSize = unsafe extern "C" fn(
    CusolverDnHandle,
    CublasOperation,
    i32,
    *mut f32,
    i32,
    *mut i32,
) -> CusolverStatus;
type FnCusolverDnDpotrfBufferSize = unsafe extern "C" fn(
    CusolverDnHandle,
    CublasOperation,
    i32,
    *mut f64,
    i32,
    *mut i32,
) -> CusolverStatus;
type FnCusolverDnCpotrfBufferSize = unsafe extern "C" fn(
    CusolverDnHandle,
    CublasOperation,
    i32,
    *mut Complex32,
    i32,
    *mut i32,
) -> CusolverStatus;
type FnCusolverDnZpotrfBufferSize = unsafe extern "C" fn(
    CusolverDnHandle,
    CublasOperation,
    i32,
    *mut Complex64,
    i32,
    *mut i32,
) -> CusolverStatus;
type FnCusolverDnSpotrf = unsafe extern "C" fn(
    CusolverDnHandle,
    CublasOperation,
    i32,
    *mut f32,
    i32,
    *mut f32,
    i32,
    *mut i32,
) -> CusolverStatus;
type FnCusolverDnDpotrf = unsafe extern "C" fn(
    CusolverDnHandle,
    CublasOperation,
    i32,
    *mut f64,
    i32,
    *mut f64,
    i32,
    *mut i32,
) -> CusolverStatus;
type FnCusolverDnCpotrf = unsafe extern "C" fn(
    CusolverDnHandle,
    CublasOperation,
    i32,
    *mut Complex32,
    i32,
    *mut Complex32,
    i32,
    *mut i32,
) -> CusolverStatus;
type FnCusolverDnZpotrf = unsafe extern "C" fn(
    CusolverDnHandle,
    CublasOperation,
    i32,
    *mut Complex64,
    i32,
    *mut Complex64,
    i32,
    *mut i32,
) -> CusolverStatus;
type FnCusolverDnSgesvd = unsafe extern "C" fn(
    CusolverDnHandle,
    c_char,
    c_char,
    i32,
    i32,
    *mut f32,
    i32,
    *mut f32,
    *mut f32,
    i32,
    *mut f32,
    i32,
    *mut f32,
    i32,
    *mut f32,
    *mut i32,
) -> CusolverStatus;
type FnCusolverDnDgesvd = unsafe extern "C" fn(
    CusolverDnHandle,
    c_char,
    c_char,
    i32,
    i32,
    *mut f64,
    i32,
    *mut f64,
    *mut f64,
    i32,
    *mut f64,
    i32,
    *mut f64,
    i32,
    *mut f64,
    *mut i32,
) -> CusolverStatus;

pub(super) struct CublasApi {
    _lib: Arc<Library>,
    create: FnCublasCreate,
    destroy: FnCublasDestroy,
    set_stream: FnCublasSetStream,
    strsm: FnCublasStrsm,
    dtrsm: FnCublasDtrsm,
}

pub(super) struct CusolverDnApi {
    _lib: Arc<Library>,
    create: FnCusolverDnCreate,
    destroy: FnCusolverDnDestroy,
    set_stream: FnCusolverDnSetStream,
    sgetrf_buffer_size: FnCusolverDnSgetrfBufferSize,
    dgetrf_buffer_size: FnCusolverDnDgetrfBufferSize,
    cgetrf_buffer_size: FnCusolverDnCgetrfBufferSize,
    zgetrf_buffer_size: FnCusolverDnZgetrfBufferSize,
    sgeqrf_buffer_size: FnCusolverDnSgeqrfBufferSize,
    dgeqrf_buffer_size: FnCusolverDnDgeqrfBufferSize,
    cgeqrf_buffer_size: FnCusolverDnCgeqrfBufferSize,
    zgeqrf_buffer_size: FnCusolverDnZgeqrfBufferSize,
    sgesvd_buffer_size: FnCusolverDnSgesvdBufferSize,
    dgesvd_buffer_size: FnCusolverDnDgesvdBufferSize,
    sgetrf: FnCusolverDnSgetrf,
    dgetrf: FnCusolverDnDgetrf,
    cgetrf: FnCusolverDnCgetrf,
    zgetrf: FnCusolverDnZgetrf,
    sgeqrf: FnCusolverDnSgeqrf,
    dgeqrf: FnCusolverDnDgeqrf,
    cgeqrf: FnCusolverDnCgeqrf,
    zgeqrf: FnCusolverDnZgeqrf,
    sgesvd: FnCusolverDnSgesvd,
    dgesvd: FnCusolverDnDgesvd,
    sgetrs: FnCusolverDnSgetrs,
    dgetrs: FnCusolverDnDgetrs,
    cgetrs: FnCusolverDnCgetrs,
    zgetrs: FnCusolverDnZgetrs,
    sorgqr_buffer_size: FnCusolverDnSorgqrBufferSize,
    dorgqr_buffer_size: FnCusolverDnDorgqrBufferSize,
    cungqr_buffer_size: FnCusolverDnCungqrBufferSize,
    zungqr_buffer_size: FnCusolverDnZungqrBufferSize,
    sorgqr: FnCusolverDnSorgqr,
    dorgqr: FnCusolverDnDorgqr,
    cungqr: FnCusolverDnCungqr,
    zungqr: FnCusolverDnZungqr,
    spotrf_buffer_size: FnCusolverDnSpotrfBufferSize,
    dpotrf_buffer_size: FnCusolverDnDpotrfBufferSize,
    cpotrf_buffer_size: FnCusolverDnCpotrfBufferSize,
    zpotrf_buffer_size: FnCusolverDnZpotrfBufferSize,
    spotrf: FnCusolverDnSpotrf,
    dpotrf: FnCusolverDnDpotrf,
    cpotrf: FnCusolverDnCpotrf,
    zpotrf: FnCusolverDnZpotrf,
}

#[allow(dead_code)]
pub(super) struct CublasHandleWrapper {
    pub raw: CublasHandle,
    destroy: FnCublasDestroy,
}

#[allow(dead_code)]
pub(super) struct CusolverDnHandleWrapper {
    pub raw: CusolverDnHandle,
    destroy: FnCusolverDnDestroy,
}

impl CublasApi {
    pub(super) fn load(candidates: &[String]) -> Result<Self> {
        let lib = Arc::new(load_first_library(candidates, "cuBLAS")?);
        Ok(Self {
            create: load_symbol(&lib, "cublasCreate_v2")?,
            destroy: load_symbol(&lib, "cublasDestroy_v2")?,
            set_stream: load_symbol(&lib, "cublasSetStream_v2")?,
            strsm: load_symbol(&lib, "cublasStrsm_v2")?,
            dtrsm: load_symbol(&lib, "cublasDtrsm_v2")?,
            _lib: lib,
        })
    }

    pub(super) fn create_handle(&self, stream: CudaStream) -> Result<CublasHandleWrapper> {
        let mut raw = std::ptr::null_mut();
        check_cublas_status(unsafe { (self.create)(&mut raw) }, "cublasCreate_v2")?;
        let handle = CublasHandleWrapper {
            raw,
            destroy: self.destroy,
        };
        check_cublas_status(
            unsafe { (self.set_stream)(handle.raw, stream) },
            "cublasSetStream_v2",
        )?;
        Ok(handle)
    }

    pub(super) fn trsm(
        &self,
        dtype: CudaDataType,
        handle: CublasHandle,
        side: i32,
        uplo: i32,
        trans: CublasOperation,
        diag: i32,
        m: i32,
        n: i32,
        alpha: *const c_void,
        a: *const c_void,
        lda: i32,
        b: *mut c_void,
        ldb: i32,
    ) -> Result<()> {
        match dtype {
            CudaDataType::F32 => check_cublas_status(
                unsafe {
                    (self.strsm)(
                        handle,
                        side,
                        uplo,
                        trans,
                        diag,
                        m,
                        n,
                        alpha.cast::<f32>(),
                        a.cast::<f32>(),
                        lda,
                        b.cast::<f32>(),
                        ldb,
                    )
                },
                "cublasStrsm_v2",
            ),
            CudaDataType::F64 => check_cublas_status(
                unsafe {
                    (self.dtrsm)(
                        handle,
                        side,
                        uplo,
                        trans,
                        diag,
                        m,
                        n,
                        alpha.cast::<f64>(),
                        a.cast::<f64>(),
                        lda,
                        b.cast::<f64>(),
                        ldb,
                    )
                },
                "cublasDtrsm_v2",
            ),
            _ => Err(Error::DeviceError(format!(
                "CUDA triangular solve currently supports only f32/f64, got {dtype:?}"
            ))),
        }
    }
}

impl CusolverDnApi {
    pub(super) fn load(candidates: &[String]) -> Result<Self> {
        let lib = Arc::new(load_first_library(candidates, "cuSOLVER")?);
        Ok(Self {
            create: load_symbol(&lib, "cusolverDnCreate")?,
            destroy: load_symbol(&lib, "cusolverDnDestroy")?,
            set_stream: load_symbol(&lib, "cusolverDnSetStream")?,
            sgetrf_buffer_size: load_symbol(&lib, "cusolverDnSgetrf_bufferSize")?,
            dgetrf_buffer_size: load_symbol(&lib, "cusolverDnDgetrf_bufferSize")?,
            cgetrf_buffer_size: load_symbol(&lib, "cusolverDnCgetrf_bufferSize")?,
            zgetrf_buffer_size: load_symbol(&lib, "cusolverDnZgetrf_bufferSize")?,
            sgeqrf_buffer_size: load_symbol(&lib, "cusolverDnSgeqrf_bufferSize")?,
            dgeqrf_buffer_size: load_symbol(&lib, "cusolverDnDgeqrf_bufferSize")?,
            cgeqrf_buffer_size: load_symbol(&lib, "cusolverDnCgeqrf_bufferSize")?,
            zgeqrf_buffer_size: load_symbol(&lib, "cusolverDnZgeqrf_bufferSize")?,
            sgesvd_buffer_size: load_symbol(&lib, "cusolverDnSgesvd_bufferSize")?,
            dgesvd_buffer_size: load_symbol(&lib, "cusolverDnDgesvd_bufferSize")?,
            sgetrf: load_symbol(&lib, "cusolverDnSgetrf")?,
            dgetrf: load_symbol(&lib, "cusolverDnDgetrf")?,
            cgetrf: load_symbol(&lib, "cusolverDnCgetrf")?,
            zgetrf: load_symbol(&lib, "cusolverDnZgetrf")?,
            sgeqrf: load_symbol(&lib, "cusolverDnSgeqrf")?,
            dgeqrf: load_symbol(&lib, "cusolverDnDgeqrf")?,
            cgeqrf: load_symbol(&lib, "cusolverDnCgeqrf")?,
            zgeqrf: load_symbol(&lib, "cusolverDnZgeqrf")?,
            sgesvd: load_symbol(&lib, "cusolverDnSgesvd")?,
            dgesvd: load_symbol(&lib, "cusolverDnDgesvd")?,
            sgetrs: load_symbol(&lib, "cusolverDnSgetrs")?,
            dgetrs: load_symbol(&lib, "cusolverDnDgetrs")?,
            cgetrs: load_symbol(&lib, "cusolverDnCgetrs")?,
            zgetrs: load_symbol(&lib, "cusolverDnZgetrs")?,
            sorgqr_buffer_size: load_symbol(&lib, "cusolverDnSorgqr_bufferSize")?,
            dorgqr_buffer_size: load_symbol(&lib, "cusolverDnDorgqr_bufferSize")?,
            cungqr_buffer_size: load_symbol(&lib, "cusolverDnCungqr_bufferSize")?,
            zungqr_buffer_size: load_symbol(&lib, "cusolverDnZungqr_bufferSize")?,
            sorgqr: load_symbol(&lib, "cusolverDnSorgqr")?,
            dorgqr: load_symbol(&lib, "cusolverDnDorgqr")?,
            cungqr: load_symbol(&lib, "cusolverDnCungqr")?,
            zungqr: load_symbol(&lib, "cusolverDnZungqr")?,
            spotrf_buffer_size: load_symbol(&lib, "cusolverDnSpotrf_bufferSize")?,
            dpotrf_buffer_size: load_symbol(&lib, "cusolverDnDpotrf_bufferSize")?,
            cpotrf_buffer_size: load_symbol(&lib, "cusolverDnCpotrf_bufferSize")?,
            zpotrf_buffer_size: load_symbol(&lib, "cusolverDnZpotrf_bufferSize")?,
            spotrf: load_symbol(&lib, "cusolverDnSpotrf")?,
            dpotrf: load_symbol(&lib, "cusolverDnDpotrf")?,
            cpotrf: load_symbol(&lib, "cusolverDnCpotrf")?,
            zpotrf: load_symbol(&lib, "cusolverDnZpotrf")?,
            _lib: lib,
        })
    }

    pub(super) fn create_handle(&self, stream: CudaStream) -> Result<CusolverDnHandleWrapper> {
        let mut raw = std::ptr::null_mut();
        check_cusolver_status(unsafe { (self.create)(&mut raw) }, "cusolverDnCreate")?;
        let handle = CusolverDnHandleWrapper {
            raw,
            destroy: self.destroy,
        };
        check_cusolver_status(
            unsafe { (self.set_stream)(handle.raw, stream) },
            "cusolverDnSetStream",
        )?;
        Ok(handle)
    }

    pub(super) fn getrf_buffer_size(
        &self,
        dtype: CudaDataType,
        handle: CusolverDnHandle,
        m: i32,
        n: i32,
        a: *mut c_void,
        lda: i32,
    ) -> Result<i32> {
        let mut lwork = 0;
        match dtype {
            CudaDataType::F32 => check_cusolver_status(
                unsafe {
                    (self.sgetrf_buffer_size)(handle, m, n, a.cast::<f32>(), lda, &mut lwork)
                },
                "cusolverDnSgetrf_bufferSize",
            )?,
            CudaDataType::F64 => check_cusolver_status(
                unsafe {
                    (self.dgetrf_buffer_size)(handle, m, n, a.cast::<f64>(), lda, &mut lwork)
                },
                "cusolverDnDgetrf_bufferSize",
            )?,
            CudaDataType::Complex32 => check_cusolver_status(
                unsafe {
                    (self.cgetrf_buffer_size)(handle, m, n, a.cast::<Complex32>(), lda, &mut lwork)
                },
                "cusolverDnCgetrf_bufferSize",
            )?,
            CudaDataType::Complex64 => check_cusolver_status(
                unsafe {
                    (self.zgetrf_buffer_size)(handle, m, n, a.cast::<Complex64>(), lda, &mut lwork)
                },
                "cusolverDnZgetrf_bufferSize",
            )?,
        }
        Ok(lwork)
    }

    pub(super) fn getrf(
        &self,
        dtype: CudaDataType,
        handle: CusolverDnHandle,
        m: i32,
        n: i32,
        a: *mut c_void,
        lda: i32,
        workspace: *mut c_void,
        pivots: *mut i32,
        info: *mut i32,
    ) -> Result<()> {
        match dtype {
            CudaDataType::F32 => check_cusolver_status(
                unsafe {
                    (self.sgetrf)(
                        handle,
                        m,
                        n,
                        a.cast::<f32>(),
                        lda,
                        workspace.cast::<f32>(),
                        pivots,
                        info,
                    )
                },
                "cusolverDnSgetrf",
            ),
            CudaDataType::F64 => check_cusolver_status(
                unsafe {
                    (self.dgetrf)(
                        handle,
                        m,
                        n,
                        a.cast::<f64>(),
                        lda,
                        workspace.cast::<f64>(),
                        pivots,
                        info,
                    )
                },
                "cusolverDnDgetrf",
            ),
            CudaDataType::Complex32 => check_cusolver_status(
                unsafe {
                    (self.cgetrf)(
                        handle,
                        m,
                        n,
                        a.cast::<Complex32>(),
                        lda,
                        workspace.cast::<Complex32>(),
                        pivots,
                        info,
                    )
                },
                "cusolverDnCgetrf",
            ),
            CudaDataType::Complex64 => check_cusolver_status(
                unsafe {
                    (self.zgetrf)(
                        handle,
                        m,
                        n,
                        a.cast::<Complex64>(),
                        lda,
                        workspace.cast::<Complex64>(),
                        pivots,
                        info,
                    )
                },
                "cusolverDnZgetrf",
            ),
        }
    }

    pub(super) fn getrs(
        &self,
        dtype: CudaDataType,
        handle: CusolverDnHandle,
        n: i32,
        nrhs: i32,
        a: *const c_void,
        lda: i32,
        pivots: *const i32,
        b: *mut c_void,
        ldb: i32,
        info: *mut i32,
    ) -> Result<()> {
        match dtype {
            CudaDataType::F32 => check_cusolver_status(
                unsafe {
                    (self.sgetrs)(
                        handle,
                        CUBLAS_OP_N,
                        n,
                        nrhs,
                        a.cast::<f32>(),
                        lda,
                        pivots,
                        b.cast::<f32>(),
                        ldb,
                        info,
                    )
                },
                "cusolverDnSgetrs",
            ),
            CudaDataType::F64 => check_cusolver_status(
                unsafe {
                    (self.dgetrs)(
                        handle,
                        CUBLAS_OP_N,
                        n,
                        nrhs,
                        a.cast::<f64>(),
                        lda,
                        pivots,
                        b.cast::<f64>(),
                        ldb,
                        info,
                    )
                },
                "cusolverDnDgetrs",
            ),
            CudaDataType::Complex32 => check_cusolver_status(
                unsafe {
                    (self.cgetrs)(
                        handle,
                        CUBLAS_OP_N,
                        n,
                        nrhs,
                        a.cast::<Complex32>(),
                        lda,
                        pivots,
                        b.cast::<Complex32>(),
                        ldb,
                        info,
                    )
                },
                "cusolverDnCgetrs",
            ),
            CudaDataType::Complex64 => check_cusolver_status(
                unsafe {
                    (self.zgetrs)(
                        handle,
                        CUBLAS_OP_N,
                        n,
                        nrhs,
                        a.cast::<Complex64>(),
                        lda,
                        pivots,
                        b.cast::<Complex64>(),
                        ldb,
                        info,
                    )
                },
                "cusolverDnZgetrs",
            ),
        }
    }

    pub(super) fn geqrf_buffer_size(
        &self,
        dtype: CudaDataType,
        handle: CusolverDnHandle,
        m: i32,
        n: i32,
        a: *mut c_void,
        lda: i32,
    ) -> Result<i32> {
        let mut lwork = 0;
        match dtype {
            CudaDataType::F32 => check_cusolver_status(
                unsafe {
                    (self.sgeqrf_buffer_size)(handle, m, n, a.cast::<f32>(), lda, &mut lwork)
                },
                "cusolverDnSgeqrf_bufferSize",
            )?,
            CudaDataType::F64 => check_cusolver_status(
                unsafe {
                    (self.dgeqrf_buffer_size)(handle, m, n, a.cast::<f64>(), lda, &mut lwork)
                },
                "cusolverDnDgeqrf_bufferSize",
            )?,
            CudaDataType::Complex32 => check_cusolver_status(
                unsafe {
                    (self.cgeqrf_buffer_size)(handle, m, n, a.cast::<Complex32>(), lda, &mut lwork)
                },
                "cusolverDnCgeqrf_bufferSize",
            )?,
            CudaDataType::Complex64 => check_cusolver_status(
                unsafe {
                    (self.zgeqrf_buffer_size)(handle, m, n, a.cast::<Complex64>(), lda, &mut lwork)
                },
                "cusolverDnZgeqrf_bufferSize",
            )?,
        }
        Ok(lwork)
    }

    pub(super) fn geqrf(
        &self,
        dtype: CudaDataType,
        handle: CusolverDnHandle,
        m: i32,
        n: i32,
        a: *mut c_void,
        lda: i32,
        tau: *mut c_void,
        workspace: *mut c_void,
        lwork: i32,
        info: *mut i32,
    ) -> Result<()> {
        match dtype {
            CudaDataType::F32 => check_cusolver_status(
                unsafe {
                    (self.sgeqrf)(
                        handle,
                        m,
                        n,
                        a.cast::<f32>(),
                        lda,
                        tau.cast::<f32>(),
                        workspace.cast::<f32>(),
                        lwork,
                        info,
                    )
                },
                "cusolverDnSgeqrf",
            ),
            CudaDataType::F64 => check_cusolver_status(
                unsafe {
                    (self.dgeqrf)(
                        handle,
                        m,
                        n,
                        a.cast::<f64>(),
                        lda,
                        tau.cast::<f64>(),
                        workspace.cast::<f64>(),
                        lwork,
                        info,
                    )
                },
                "cusolverDnDgeqrf",
            ),
            CudaDataType::Complex32 => check_cusolver_status(
                unsafe {
                    (self.cgeqrf)(
                        handle,
                        m,
                        n,
                        a.cast::<Complex32>(),
                        lda,
                        tau.cast::<Complex32>(),
                        workspace.cast::<Complex32>(),
                        lwork,
                        info,
                    )
                },
                "cusolverDnCgeqrf",
            ),
            CudaDataType::Complex64 => check_cusolver_status(
                unsafe {
                    (self.zgeqrf)(
                        handle,
                        m,
                        n,
                        a.cast::<Complex64>(),
                        lda,
                        tau.cast::<Complex64>(),
                        workspace.cast::<Complex64>(),
                        lwork,
                        info,
                    )
                },
                "cusolverDnZgeqrf",
            ),
        }
    }

    pub(super) fn orgqr_buffer_size(
        &self,
        dtype: CudaDataType,
        handle: CusolverDnHandle,
        m: i32,
        n: i32,
        k: i32,
        a: *mut c_void,
        lda: i32,
        tau: *mut c_void,
    ) -> Result<i32> {
        let mut lwork = 0;
        match dtype {
            CudaDataType::F32 => check_cusolver_status(
                unsafe {
                    (self.sorgqr_buffer_size)(
                        handle,
                        m,
                        n,
                        k,
                        a.cast::<f32>(),
                        lda,
                        tau.cast::<f32>(),
                        &mut lwork,
                    )
                },
                "cusolverDnSorgqr_bufferSize",
            )?,
            CudaDataType::F64 => check_cusolver_status(
                unsafe {
                    (self.dorgqr_buffer_size)(
                        handle,
                        m,
                        n,
                        k,
                        a.cast::<f64>(),
                        lda,
                        tau.cast::<f64>(),
                        &mut lwork,
                    )
                },
                "cusolverDnDorgqr_bufferSize",
            )?,
            CudaDataType::Complex32 => check_cusolver_status(
                unsafe {
                    (self.cungqr_buffer_size)(
                        handle,
                        m,
                        n,
                        k,
                        a.cast::<Complex32>(),
                        lda,
                        tau.cast::<Complex32>(),
                        &mut lwork,
                    )
                },
                "cusolverDnCungqr_bufferSize",
            )?,
            CudaDataType::Complex64 => check_cusolver_status(
                unsafe {
                    (self.zungqr_buffer_size)(
                        handle,
                        m,
                        n,
                        k,
                        a.cast::<Complex64>(),
                        lda,
                        tau.cast::<Complex64>(),
                        &mut lwork,
                    )
                },
                "cusolverDnZungqr_bufferSize",
            )?,
        }
        Ok(lwork)
    }

    pub(super) fn orgqr(
        &self,
        dtype: CudaDataType,
        handle: CusolverDnHandle,
        m: i32,
        n: i32,
        k: i32,
        a: *mut c_void,
        lda: i32,
        tau: *mut c_void,
        workspace: *mut c_void,
        lwork: i32,
        info: *mut i32,
    ) -> Result<()> {
        match dtype {
            CudaDataType::F32 => check_cusolver_status(
                unsafe {
                    (self.sorgqr)(
                        handle,
                        m,
                        n,
                        k,
                        a.cast::<f32>(),
                        lda,
                        tau.cast::<f32>(),
                        workspace.cast::<f32>(),
                        lwork,
                        info,
                    )
                },
                "cusolverDnSorgqr",
            ),
            CudaDataType::F64 => check_cusolver_status(
                unsafe {
                    (self.dorgqr)(
                        handle,
                        m,
                        n,
                        k,
                        a.cast::<f64>(),
                        lda,
                        tau.cast::<f64>(),
                        workspace.cast::<f64>(),
                        lwork,
                        info,
                    )
                },
                "cusolverDnDorgqr",
            ),
            CudaDataType::Complex32 => check_cusolver_status(
                unsafe {
                    (self.cungqr)(
                        handle,
                        m,
                        n,
                        k,
                        a.cast::<Complex32>(),
                        lda,
                        tau.cast::<Complex32>(),
                        workspace.cast::<Complex32>(),
                        lwork,
                        info,
                    )
                },
                "cusolverDnCungqr",
            ),
            CudaDataType::Complex64 => check_cusolver_status(
                unsafe {
                    (self.zungqr)(
                        handle,
                        m,
                        n,
                        k,
                        a.cast::<Complex64>(),
                        lda,
                        tau.cast::<Complex64>(),
                        workspace.cast::<Complex64>(),
                        lwork,
                        info,
                    )
                },
                "cusolverDnZungqr",
            ),
        }
    }

    pub(super) fn gesvd_buffer_size(
        &self,
        dtype: CudaDataType,
        handle: CusolverDnHandle,
        m: i32,
        n: i32,
    ) -> Result<i32> {
        let mut lwork = 0;
        match dtype {
            CudaDataType::F32 => check_cusolver_status(
                unsafe { (self.sgesvd_buffer_size)(handle, m, n, &mut lwork) },
                "cusolverDnSgesvd_bufferSize",
            )?,
            CudaDataType::F64 => check_cusolver_status(
                unsafe { (self.dgesvd_buffer_size)(handle, m, n, &mut lwork) },
                "cusolverDnDgesvd_bufferSize",
            )?,
            _ => {
                return Err(Error::DeviceError(format!(
                    "CUDA svdvals currently supports only f32/f64, got {dtype:?}"
                )));
            }
        }
        Ok(lwork)
    }

    pub(super) fn gesvd(
        &self,
        dtype: CudaDataType,
        handle: CusolverDnHandle,
        jobu: c_char,
        jobvt: c_char,
        m: i32,
        n: i32,
        a: *mut c_void,
        lda: i32,
        s: *mut c_void,
        u: *mut c_void,
        ldu: i32,
        vt: *mut c_void,
        ldvt: i32,
        workspace: *mut c_void,
        lwork: i32,
        rwork: *mut c_void,
        info: *mut i32,
    ) -> Result<()> {
        match dtype {
            CudaDataType::F32 => check_cusolver_status(
                unsafe {
                    (self.sgesvd)(
                        handle,
                        jobu,
                        jobvt,
                        m,
                        n,
                        a.cast::<f32>(),
                        lda,
                        s.cast::<f32>(),
                        u.cast::<f32>(),
                        ldu,
                        vt.cast::<f32>(),
                        ldvt,
                        workspace.cast::<f32>(),
                        lwork,
                        rwork.cast::<f32>(),
                        info,
                    )
                },
                "cusolverDnSgesvd",
            ),
            CudaDataType::F64 => check_cusolver_status(
                unsafe {
                    (self.dgesvd)(
                        handle,
                        jobu,
                        jobvt,
                        m,
                        n,
                        a.cast::<f64>(),
                        lda,
                        s.cast::<f64>(),
                        u.cast::<f64>(),
                        ldu,
                        vt.cast::<f64>(),
                        ldvt,
                        workspace.cast::<f64>(),
                        lwork,
                        rwork.cast::<f64>(),
                        info,
                    )
                },
                "cusolverDnDgesvd",
            ),
            _ => Err(Error::DeviceError(format!(
                "CUDA svdvals currently supports only f32/f64, got {dtype:?}"
            ))),
        }
    }

    pub(super) fn potrf_buffer_size(
        &self,
        dtype: CudaDataType,
        handle: CusolverDnHandle,
        uplo: CublasOperation,
        n: i32,
        a: *mut c_void,
        lda: i32,
    ) -> Result<i32> {
        let mut lwork = 0;
        match dtype {
            CudaDataType::F32 => check_cusolver_status(
                unsafe {
                    (self.spotrf_buffer_size)(handle, uplo, n, a.cast::<f32>(), lda, &mut lwork)
                },
                "cusolverDnSpotrf_bufferSize",
            )?,
            CudaDataType::F64 => check_cusolver_status(
                unsafe {
                    (self.dpotrf_buffer_size)(handle, uplo, n, a.cast::<f64>(), lda, &mut lwork)
                },
                "cusolverDnDpotrf_bufferSize",
            )?,
            CudaDataType::Complex32 => check_cusolver_status(
                unsafe {
                    (self.cpotrf_buffer_size)(
                        handle,
                        uplo,
                        n,
                        a.cast::<Complex32>(),
                        lda,
                        &mut lwork,
                    )
                },
                "cusolverDnCpotrf_bufferSize",
            )?,
            CudaDataType::Complex64 => check_cusolver_status(
                unsafe {
                    (self.zpotrf_buffer_size)(
                        handle,
                        uplo,
                        n,
                        a.cast::<Complex64>(),
                        lda,
                        &mut lwork,
                    )
                },
                "cusolverDnZpotrf_bufferSize",
            )?,
        }
        Ok(lwork)
    }

    pub(super) fn potrf(
        &self,
        dtype: CudaDataType,
        handle: CusolverDnHandle,
        uplo: CublasOperation,
        n: i32,
        a: *mut c_void,
        lda: i32,
        workspace: *mut c_void,
        lwork: i32,
        info: *mut i32,
    ) -> Result<()> {
        match dtype {
            CudaDataType::F32 => check_cusolver_status(
                unsafe {
                    (self.spotrf)(
                        handle,
                        uplo,
                        n,
                        a.cast::<f32>(),
                        lda,
                        workspace.cast::<f32>(),
                        lwork,
                        info,
                    )
                },
                "cusolverDnSpotrf",
            ),
            CudaDataType::F64 => check_cusolver_status(
                unsafe {
                    (self.dpotrf)(
                        handle,
                        uplo,
                        n,
                        a.cast::<f64>(),
                        lda,
                        workspace.cast::<f64>(),
                        lwork,
                        info,
                    )
                },
                "cusolverDnDpotrf",
            ),
            CudaDataType::Complex32 => check_cusolver_status(
                unsafe {
                    (self.cpotrf)(
                        handle,
                        uplo,
                        n,
                        a.cast::<Complex32>(),
                        lda,
                        workspace.cast::<Complex32>(),
                        lwork,
                        info,
                    )
                },
                "cusolverDnCpotrf",
            ),
            CudaDataType::Complex64 => check_cusolver_status(
                unsafe {
                    (self.zpotrf)(
                        handle,
                        uplo,
                        n,
                        a.cast::<Complex64>(),
                        lda,
                        workspace.cast::<Complex64>(),
                        lwork,
                        info,
                    )
                },
                "cusolverDnZpotrf",
            ),
        }
    }
}

impl Drop for CublasHandleWrapper {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe {
                (self.destroy)(self.raw);
            }
        }
    }
}

impl Drop for CusolverDnHandleWrapper {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe {
                (self.destroy)(self.raw);
            }
        }
    }
}

pub(super) fn load_first_library(candidates: &[String], label: &str) -> Result<Library> {
    let mut errors = Vec::new();
    for candidate in candidates {
        match unsafe { Library::new(candidate) } {
            Ok(lib) => return Ok(lib),
            Err(err) => errors.push(format!("{candidate}: {err}")),
        }
    }

    let detail = if errors.is_empty() {
        "no candidate paths were provided".to_string()
    } else {
        errors.join("; ")
    };
    Err(Error::DeviceError(format!(
        "failed to load {label} runtime library: {detail}"
    )))
}

fn load_symbol<T: Copy>(lib: &Library, symbol: &str) -> Result<T> {
    let mut symbol_bytes = Vec::with_capacity(symbol.len() + 1);
    symbol_bytes.extend_from_slice(symbol.as_bytes());
    symbol_bytes.push(0);

    unsafe {
        lib.get::<T>(&symbol_bytes)
            .map(|loaded| *loaded)
            .map_err(|err| Error::DeviceError(format!("failed to load symbol {symbol}: {err}")))
    }
}
