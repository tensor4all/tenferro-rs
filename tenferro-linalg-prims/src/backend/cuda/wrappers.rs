use std::ffi::c_void;
use std::sync::Arc;

use libloading::Library;
use tenferro_device::{Error, Result};

use super::runtime::{check_cublas_status, check_cusolver_status};
use super::scalar_type::CudaDataType;

pub(super) type CublasStatus = i32;
pub(super) type CusolverStatus = i32;
pub(super) type CublasOperation = i32;

type CublasHandle = *mut c_void;
type CusolverDnHandle = *mut c_void;
type CudaStream = *mut c_void;

const CUBLAS_OP_N: CublasOperation = 0;

type FnCublasCreate = unsafe extern "C" fn(*mut CublasHandle) -> CublasStatus;
type FnCublasDestroy = unsafe extern "C" fn(CublasHandle) -> CublasStatus;
type FnCublasSetStream = unsafe extern "C" fn(CublasHandle, CudaStream) -> CublasStatus;

type FnCusolverDnCreate = unsafe extern "C" fn(*mut CusolverDnHandle) -> CusolverStatus;
type FnCusolverDnDestroy = unsafe extern "C" fn(CusolverDnHandle) -> CusolverStatus;
type FnCusolverDnSetStream = unsafe extern "C" fn(CusolverDnHandle, CudaStream) -> CusolverStatus;
type FnCusolverDnSgetrfBufferSize =
    unsafe extern "C" fn(CusolverDnHandle, i32, i32, *mut f32, i32, *mut i32) -> CusolverStatus;
type FnCusolverDnDgetrfBufferSize =
    unsafe extern "C" fn(CusolverDnHandle, i32, i32, *mut f64, i32, *mut i32) -> CusolverStatus;
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

pub(super) struct CublasApi {
    _lib: Arc<Library>,
    create: FnCublasCreate,
    destroy: FnCublasDestroy,
    set_stream: FnCublasSetStream,
}

pub(super) struct CusolverDnApi {
    _lib: Arc<Library>,
    create: FnCusolverDnCreate,
    destroy: FnCusolverDnDestroy,
    set_stream: FnCusolverDnSetStream,
    sgetrf_buffer_size: FnCusolverDnSgetrfBufferSize,
    dgetrf_buffer_size: FnCusolverDnDgetrfBufferSize,
    sgetrf: FnCusolverDnSgetrf,
    dgetrf: FnCusolverDnDgetrf,
    sgetrs: FnCusolverDnSgetrs,
    dgetrs: FnCusolverDnDgetrs,
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
            sgetrf: load_symbol(&lib, "cusolverDnSgetrf")?,
            dgetrf: load_symbol(&lib, "cusolverDnDgetrf")?,
            sgetrs: load_symbol(&lib, "cusolverDnSgetrs")?,
            dgetrs: load_symbol(&lib, "cusolverDnDgetrs")?,
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
            _ => {
                return Err(Error::DeviceError(format!(
                    "CUDA solve currently supports only f32/f64, got {dtype:?}"
                )));
            }
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
            _ => Err(Error::DeviceError(format!(
                "CUDA solve currently supports only f32/f64, got {dtype:?}"
            ))),
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
            _ => Err(Error::DeviceError(format!(
                "CUDA solve currently supports only f32/f64, got {dtype:?}"
            ))),
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
