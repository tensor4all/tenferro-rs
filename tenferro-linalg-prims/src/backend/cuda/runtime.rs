use std::ffi::c_void;
use std::ptr;

#[cfg(feature = "cuda")]
use cudarc::runtime::result as cuda_result;
use tenferro_device::{Error, LogicalMemorySpace, Result};
use tenferro_tensor::Tensor;

use super::wrappers::{
    CublasApi, CublasHandleWrapper, CublasStatus, CusolverDnApi, CusolverDnHandleWrapper,
    CusolverStatus,
};

const CUBLAS_STATUS_SUCCESS: CublasStatus = 0;
const CUSOLVER_STATUS_SUCCESS: CusolverStatus = 0;

pub(super) struct CudaLibraryCandidates {
    pub cublas: Vec<String>,
    pub cusolver: Vec<String>,
}

#[allow(dead_code)]
pub(super) struct CudaLinalgRuntime {
    pub cublas_handle: CublasHandleWrapper,
    pub cusolver_handle: CusolverDnHandleWrapper,
    // Keep libraries alive until after the raw handles are destroyed.
    _cublas_api: CublasApi,
    _cusolver_api: CusolverDnApi,
}

#[cfg(feature = "cuda")]
pub(super) struct DeviceAllocation {
    ptr: *mut c_void,
    device_id: usize,
}

pub(super) fn unsupported<T>(op: &str) -> Result<T> {
    Err(Error::DeviceError(format!(
        "CUDA linalg backend is not yet implemented for {op}"
    )))
}

pub(super) fn check_cublas_status(status: CublasStatus, op: &str) -> Result<()> {
    if status == CUBLAS_STATUS_SUCCESS {
        Ok(())
    } else {
        Err(Error::DeviceError(format!(
            "{op} failed with cuBLAS status {status}"
        )))
    }
}

pub(super) fn check_cusolver_status(status: CusolverStatus, op: &str) -> Result<()> {
    if status == CUSOLVER_STATUS_SUCCESS {
        Ok(())
    } else {
        Err(Error::DeviceError(format!(
            "{op} failed with cuSOLVER status {status}"
        )))
    }
}

pub(super) fn default_library_candidates() -> CudaLibraryCandidates {
    let mut cublas = env_candidates("TENFERRO_CUBLAS_LIB");
    cublas.extend(
        [
            "/lib/x86_64-linux-gnu/libcublas.so.12",
            "/usr/lib/x86_64-linux-gnu/libcublas.so.12",
            "/lib/x86_64-linux-gnu/libcublas.so.11",
            "/usr/lib/x86_64-linux-gnu/libcublas.so.11",
            "libcublas.so.12",
            "libcublas.so.11",
            "libcublas.so",
        ]
        .into_iter()
        .map(str::to_owned),
    );

    let mut cusolver = env_candidates("TENFERRO_CUSOLVER_LIB");
    cusolver.extend(
        [
            "/lib/x86_64-linux-gnu/libcusolver.so.12",
            "/usr/lib/x86_64-linux-gnu/libcusolver.so.12",
            "/lib/x86_64-linux-gnu/libcusolver.so.11",
            "/usr/lib/x86_64-linux-gnu/libcusolver.so.11",
            "/opt/cuda/11.6/targets/x86_64-linux/lib/libcusolver.so.11",
            "libcusolver.so.12",
            "libcusolver.so.11",
            "libcusolver.so",
        ]
        .into_iter()
        .map(str::to_owned),
    );

    CudaLibraryCandidates { cublas, cusolver }
}

#[allow(dead_code)]
pub(super) fn load_runtime(ctx: &tenferro_prims::CudaContext) -> Result<CudaLinalgRuntime> {
    ctx.bind_to_device()?;
    let candidates = default_library_candidates();
    let cublas_api = CublasApi::load(&candidates.cublas)?;
    let cusolver_api = CusolverDnApi::load(&candidates.cusolver)?;

    let cublas_handle = cublas_api.create_handle(ptr::null_mut())?;
    let cusolver_handle = cusolver_api.create_handle(ptr::null_mut())?;

    Ok(CudaLinalgRuntime {
        cublas_handle,
        cusolver_handle,
        _cublas_api: cublas_api,
        _cusolver_api: cusolver_api,
    })
}

impl CudaLinalgRuntime {
    pub(super) fn cusolver_api(&self) -> &CusolverDnApi {
        &self._cusolver_api
    }
}

#[cfg(feature = "cuda")]
impl DeviceAllocation {
    pub(super) fn alloc(
        ctx: &tenferro_prims::CudaContext,
        num_bytes: usize,
        label: &str,
    ) -> Result<Self> {
        ctx.bind_to_device()?;
        let ptr = unsafe { cuda_result::malloc_sync(num_bytes) }
            .map_err(|err| Error::DeviceError(format!("{label} failed: {err:?}")))?;
        Ok(Self {
            ptr,
            device_id: ctx.device_id(),
        })
    }

    pub(super) fn as_mut_ptr(&self) -> *mut c_void {
        self.ptr
    }
}

#[cfg(feature = "cuda")]
impl Drop for DeviceAllocation {
    fn drop(&mut self) {
        if self.ptr.is_null() {
            return;
        }
        let _ = cuda_result::device::set(self.device_id as i32);
        let _ = unsafe { cuda_result::free_sync(self.ptr) };
    }
}

#[cfg(feature = "cuda")]
pub(super) fn copy_device_to_host<T: Copy>(
    ctx: &tenferro_prims::CudaContext,
    src: *const c_void,
    dst: &mut [T],
    label: &str,
) -> Result<()> {
    ctx.bind_to_device()?;
    unsafe { cuda_result::memcpy_dtoh_sync(dst, src) }
        .map_err(|err| Error::DeviceError(format!("{label} failed: {err:?}")))
}

pub(super) fn device_ptr<T: crate::KernelLinalgScalar>(
    tensor: &Tensor<T>,
    label: &str,
) -> Result<*mut c_void> {
    let _ = tensor_device_id(tensor, label)?;
    tensor
        .buffer()
        .as_device_ptr()
        .map(|ptr| ptr as *mut c_void)
        .ok_or_else(|| Error::DeviceError(format!("{label} tensor has no GPU device pointer")))
}

#[allow(dead_code)]
pub(super) fn context_device_ptr<T>(
    ctx: &tenferro_prims::CudaContext,
    tensor: &Tensor<T>,
    label: &str,
) -> Result<*mut c_void>
where
    T: crate::KernelLinalgScalar,
{
    let tensor_device_id = tensor_device_id(tensor, label)?;
    if tensor_device_id != ctx.device_id() {
        return Err(Error::DeviceError(format!(
            "{label} tensor is on cuda:{tensor_device_id}, but linalg context is cuda:{}",
            ctx.device_id()
        )));
    }
    device_ptr(tensor, label)
}

fn tensor_device_id<T: crate::KernelLinalgScalar>(
    tensor: &Tensor<T>,
    label: &str,
) -> Result<usize> {
    match tensor.logical_memory_space() {
        LogicalMemorySpace::GpuMemory { device_id } => Ok(device_id),
        _ => Err(Error::DeviceError(format!(
            "{label} tensor is not resident on GPU"
        ))),
    }
}

fn env_candidates(var: &str) -> Vec<String> {
    std::env::var_os(var)
        .into_iter()
        .map(|value| value.to_string_lossy().into_owned())
        .collect()
}
