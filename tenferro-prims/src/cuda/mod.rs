//! Real CUDA backend using cuTENSOR v2 via cudarc + libloading.
//!
//! This module is compiled only when `feature = "cuda"` is enabled.
//! It replaces the stub types from `gpu_stubs.rs`.
//!
//! cudarc is used with its default `dynamic-loading` feature, so **no
//! CUDA SDK is required at compile time**. The CUDA driver is loaded
//! via dlopen at runtime.
//!
//! # Examples
//!
//! ```ignore
//! use tenferro_algebra::Standard;
//! use tenferro_prims::{CudaBackend, CudaContext, SemiringCoreDescriptor, TensorSemiringCore};
//!
//! let (backend, mut ctx) = CudaBackend::load("/usr/lib/libcutensor.so").unwrap();
//! let _plan = <CudaBackend as TensorSemiringCore<Standard<f64>>>::plan(
//!     &mut ctx,
//!     &SemiringCoreDescriptor::MakeContiguous,
//!     &[&[2, 2], &[2, 2]],
//! )
//! .unwrap();
//! ```

use std::marker::PhantomData;
use std::ptr;
use std::sync::{Arc, OnceLock};

use cudarc::driver::{LaunchConfig, PushKernelArg};
use cudarc::nvrtc::{compile_ptx, Ptx};
#[cfg(unix)]
use libloading::os::unix::{Library as UnixLibrary, RTLD_GLOBAL, RTLD_NOW};
use num_complex::{Complex32, Complex64};
use tenferro_algebra::{Conjugate, Scalar, Standard};
use tenferro_device::{cuda::runtime as device_cuda, Error, LogicalMemorySpace, Result};
use tenferro_tensor::MemoryOrder;
use tenferro_tensor::Tensor;

use crate::cuda_ffi::*;
use crate::{
    SemiringCoreDescriptor, SemiringFastPathDescriptor, TensorSemiringCore, TensorSemiringFastPath,
};

mod analytic;
mod complex_real;
mod execution;
mod planning;
mod runtime;
mod scalar;
mod scalar_type;
mod wrappers;

pub use analytic::CudaAnalyticPlan;
pub use complex_real::CudaComplexRealPlan;
use execution::{execute_plan, has_fast_path, plan_core_descriptor, plan_fast_descriptor};
use planning::{check_status, NativeCutensorPlan};
pub use scalar::CudaScalarPlan;
use wrappers::HandleWrapper;

const RESOLVE_CONJ_KERNEL_NAME_C32: &str = "resolve_conj_complex32";
const RESOLVE_CONJ_KERNEL_NAME_C64: &str = "resolve_conj_complex64";
const RESOLVE_CONJ_CUDA_SRC: &str = r#"
typedef struct { float re; float im; } complex32_t;
typedef struct { double re; double im; } complex64_t;

extern "C" __global__ void resolve_conj_complex32(
    const complex32_t* src,
    complex32_t* dst,
    unsigned long long len
) {
    unsigned long long idx =
        (unsigned long long)blockIdx.x * (unsigned long long)blockDim.x +
        (unsigned long long)threadIdx.x;
    if (idx >= len) {
        return;
    }
    dst[idx].re = src[idx].re;
    dst[idx].im = -src[idx].im;
}

extern "C" __global__ void resolve_conj_complex64(
    const complex64_t* src,
    complex64_t* dst,
    unsigned long long len
) {
    unsigned long long idx =
        (unsigned long long)blockIdx.x * (unsigned long long)blockDim.x +
        (unsigned long long)threadIdx.x;
    if (idx >= len) {
        return;
    }
    dst[idx].re = src[idx].re;
    dst[idx].im = -src[idx].im;
}
"#;

fn resolve_conj_ptx() -> Result<Ptx> {
    static PTX: OnceLock<std::result::Result<Ptx, String>> = OnceLock::new();
    PTX.get_or_init(|| {
        compile_ptx(RESOLVE_CONJ_CUDA_SRC)
            .map_err(|err| format!("NVRTC compile failed for resolve_conj kernel: {err:?}"))
    })
    .clone()
    .map_err(Error::DeviceError)
}

/// CUDA execution context backed by cudarc + cuTENSOR vtable.
///
/// Encapsulates GPU-side execution resources: a cuTENSOR handle, a CUDA device
/// ordinal, and the loaded cuTENSOR function table.
///
/// # Examples
///
/// ```ignore
/// // Created internally by CudaBackend::load()
/// use tenferro_prims::CudaContext;
/// ```
pub struct CudaContext {
    /// cuTENSOR library handle (RAII — Drop calls cutensorDestroy).
    pub(in crate::cuda) handle: HandleWrapper,
    /// CUDA device ordinal used for runtime API calls.
    pub(super) device_id: usize,
    /// cuTENSOR function pointer vtable loaded via libloading.
    pub(super) vtable: Arc<CutensorVtable>,
    /// Shared Layer 0 CUDA runtime handle reused across crates.
    pub(super) shared_runtime: Arc<device_cuda::CudaRuntime>,
}

impl CudaContext {
    /// Return the CUDA device ordinal associated with this context.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_prims::CudaBackend;
    ///
    /// let (_backend, ctx) = CudaBackend::load("/usr/lib/libcutensor.so").unwrap();
    /// assert_eq!(ctx.device_id(), 0);
    /// ```
    pub fn device_id(&self) -> usize {
        self.device_id
    }

    /// Bind this context's CUDA device as the current runtime device.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_prims::CudaBackend;
    ///
    /// let (_backend, ctx) = CudaBackend::load("/usr/lib/libcutensor.so").unwrap();
    /// ctx.bind_to_device().unwrap();
    /// ```
    pub fn bind_to_device(&self) -> Result<()> {
        cudarc::runtime::result::device::set(self.device_id as i32)
            .map_err(|e| Error::DeviceError(format!("CUDA runtime set-device failed: {e:?}")))
    }

    /// Returns the shared Layer 0 runtime handle used by this context.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_prims::CudaBackend;
    ///
    /// let (_backend, ctx) = CudaBackend::load("/usr/lib/libcutensor.so").unwrap();
    /// assert_eq!(ctx.shared_runtime().device_id(), 0);
    /// ```
    pub fn shared_runtime(&self) -> &Arc<device_cuda::CudaRuntime> {
        &self.shared_runtime
    }
}

/// CUDA plan — wraps a cuTENSOR operation plan handle.
///
/// # Examples
///
/// ```ignore
/// use tenferro_algebra::Standard;
/// use tenferro_prims::{CudaBackend, SemiringCoreDescriptor, TensorSemiringCore};
///
/// let (_, mut ctx) = CudaBackend::load("/usr/lib/libcutensor.so").unwrap();
/// let plan = <CudaBackend as TensorSemiringCore<Standard<f64>>>::plan(
///     &mut ctx,
///     &SemiringCoreDescriptor::MakeContiguous,
///     &[&[3, 2], &[3, 2]],
/// )
/// .unwrap();
/// ```
#[derive(Clone, Debug)]
enum CudaPlanDescriptor {
    Core(SemiringCoreDescriptor),
    Fast(SemiringFastPathDescriptor),
}

#[derive(Clone, Debug)]
enum CudaPlanStorage {
    Compiled(NativeCutensorPlan),
    DeferredMakeContiguous,
}

pub struct CudaPlan<T: Scalar> {
    /// Compiled cuTENSOR plan (RAII — Drop calls cutensorDestroyPlan).
    plan: CudaPlanStorage,
    /// Family descriptor (for execute dispatch).
    desc: CudaPlanDescriptor,
    _marker: PhantomData<T>,
}

/// CUDA backend using cuTENSOR v2 via cudarc + libloading.
///
/// Loaded at runtime from a user-provided `.so` path. cudarc uses
/// `dynamic-loading` by default — no compile-time CUDA SDK dependency.
///
/// # Examples
///
/// ```ignore
/// use tenferro_prims::{BackendRegistry, CudaBackend};
///
/// let mut registry = BackendRegistry::new();
/// registry.load_cutensor("/usr/lib/libcutensor.so").unwrap();
/// ```
pub struct CudaBackend {
    #[cfg(unix)]
    _cudart_global: UnixLibrary,
    #[cfg(unix)]
    _lib: DynamicLibrary,
}

impl CudaBackend {
    /// Load cuTENSOR library and initialize CUDA context.
    ///
    /// Opens the cuTENSOR shared library at `path` via `libloading`, populates
    /// the function-pointer vtable, creates a cuTENSOR handle, and initializes
    /// a CUDA device and stream via cudarc.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_prims::CudaBackend;
    ///
    /// let (backend, ctx) = CudaBackend::load("/usr/lib/libcutensor.so").unwrap();
    /// ```
    pub fn load(path: &str) -> Result<(Self, CudaContext)> {
        #[cfg(unix)]
        let cudart_global = {
            let candidates = [
                "/usr/lib/x86_64-linux-gnu/libcudart.so.12",
                "/usr/lib/x86_64-linux-gnu/libcudart.so",
            ];
            let mut loaded = None;
            for candidate in candidates {
                match unsafe { UnixLibrary::open(Some(candidate), RTLD_NOW | RTLD_GLOBAL) } {
                    Ok(lib) => {
                        loaded = Some(lib);
                        break;
                    }
                    Err(_) => continue,
                }
            }
            loaded.ok_or_else(|| {
                Error::DeviceError("Failed to load libcudart with RTLD_GLOBAL for cuTENSOR".into())
            })?
        };

        let lib = unsafe { DynamicLibrary::open(path, RTLD_NOW) }
            .map_err(|e| Error::DeviceError(format!("Failed to load cuTENSOR: {e}")))?;
        let vtable = unsafe { CutensorVtable::load(lib.handle()) }
            .map_err(|e| Error::DeviceError(format!("Failed to load cuTENSOR symbols: {e}")))?;
        let vtable = Arc::new(vtable);

        let shared_runtime = device_cuda::get_or_init(0)?;
        shared_runtime
            .context()
            .bind_to_thread()
            .map_err(|e| Error::DeviceError(format!("CUDA runtime init failed: {e:?}")))?;

        let mut handle_raw: cutensorHandle_t = ptr::null_mut();
        let status = unsafe { (vtable.create)(&mut handle_raw) };
        check_status(status, "cutensorCreate")?;
        let handle = HandleWrapper {
            raw: handle_raw,
            vtable: Arc::clone(&vtable),
        };

        let ctx = CudaContext {
            handle,
            device_id: 0,
            vtable: Arc::clone(&vtable),
            shared_runtime,
        };

        Ok((
            CudaBackend {
                #[cfg(unix)]
                _cudart_global: cudart_global,
                #[cfg(unix)]
                _lib: lib,
            },
            ctx,
        ))
    }

    /// Materialize a lazily-conjugated tensor on GPU.
    pub fn resolve_conj<T: Scalar + Conjugate>(ctx: &mut CudaContext, src: &Tensor<T>) -> Tensor<T>
    where
        T: 'static,
    {
        if !src.is_conjugated() {
            return src.clone();
        }

        let contiguous = src.contiguous(MemoryOrder::ColumnMajor);
        match contiguous.logical_memory_space() {
            LogicalMemorySpace::GpuMemory { device_id } => {
                let resolved = Tensor::<T>::zeros(
                    src.dims(),
                    LogicalMemorySpace::GpuMemory { device_id },
                    MemoryOrder::ColumnMajor,
                );
                let Some(src_ptr) = contiguous.buffer().as_device_ptr() else {
                    return src.clone();
                };
                let Some(dst_ptr) = resolved.buffer().as_device_ptr() else {
                    return src.clone();
                };

                let copy_result =
                    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<Complex32>() {
                        unsafe {
                            launch_resolve_conj_kernel::<Complex32>(
                                ctx,
                                RESOLVE_CONJ_KERNEL_NAME_C32,
                                src_ptr.cast::<Complex32>(),
                                dst_ptr.cast::<Complex32>() as *mut Complex32,
                                src.len(),
                            )
                        }
                    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<Complex64>() {
                        unsafe {
                            launch_resolve_conj_kernel::<Complex64>(
                                ctx,
                                RESOLVE_CONJ_KERNEL_NAME_C64,
                                src_ptr.cast::<Complex64>(),
                                dst_ptr.cast::<Complex64>() as *mut Complex64,
                                src.len(),
                            )
                        }
                    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
                        unsafe {
                            ctx.shared_runtime.copy_dtod_raw(
                                src_ptr.cast::<f32>(),
                                dst_ptr.cast::<f32>() as *mut f32,
                                src.len(),
                            )
                        }
                    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
                        unsafe {
                            ctx.shared_runtime.copy_dtod_raw(
                                src_ptr.cast::<f64>(),
                                dst_ptr.cast::<f64>() as *mut f64,
                                src.len(),
                            )
                        }
                    } else {
                        Err(Error::DeviceError(format!(
                            "CUDA resolve_conj does not support scalar type {}",
                            std::any::type_name::<T>()
                        )))
                    };

                if copy_result.is_ok() {
                    resolved
                } else {
                    src.clone()
                }
            }
            _ => {
                let Some(data) = contiguous.buffer().as_slice() else {
                    return src.clone();
                };
                let conjugated_data: Vec<T> = data.iter().map(|&v| v.conj()).collect();
                Tensor::from_slice(&conjugated_data, src.dims(), MemoryOrder::ColumnMajor)
                    .unwrap_or_else(|_| src.clone())
            }
        }
    }
}

unsafe fn launch_resolve_conj_kernel<T>(
    ctx: &CudaContext,
    kernel_name: &str,
    src: *const T,
    dst: *mut T,
    len: usize,
) -> Result<()> {
    if len == 0 {
        return Ok(());
    }

    let runtime = ctx.shared_runtime();
    let cuda_ctx = runtime.context();
    cuda_ctx
        .bind_to_thread()
        .map_err(|err| Error::DeviceError(format!("CUDA context bind failed: {err:?}")))?;
    let stream = cuda_ctx.default_stream();
    let module = cuda_ctx
        .load_module(resolve_conj_ptx()?)
        .map_err(|err| Error::DeviceError(format!("CUDA module load failed: {err:?}")))?;
    let kernel = module
        .load_function(kernel_name)
        .map_err(|err| Error::DeviceError(format!("CUDA load function failed: {err:?}")))?;

    let len_u64 = u64::try_from(len)
        .map_err(|_| Error::DeviceError("resolve_conj length exceeds u64 range".into()))?;
    let len_u32 = u32::try_from(len).map_err(|_| {
        Error::DeviceError("resolve_conj currently requires len <= u32::MAX".into())
    })?;
    let src_ptr = src as u64;
    let dst_ptr = dst as u64;
    let config = LaunchConfig {
        grid_dim: (len_u32.div_ceil(256), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&kernel)
            .arg(&src_ptr)
            .arg(&dst_ptr)
            .arg(&len_u64)
            .launch(config)
            .map_err(|err| {
                Error::DeviceError(format!("CUDA resolve_conj launch failed: {err:?}"))
            })?;
    }
    stream
        .synchronize()
        .map_err(|err| Error::DeviceError(format!("CUDA stream synchronize failed: {err:?}")))
}

impl<S: Scalar> TensorSemiringCore<Standard<S>> for CudaBackend {
    type Plan = CudaPlan<S>;
    type Context = CudaContext;

    fn plan(
        ctx: &mut CudaContext,
        desc: &SemiringCoreDescriptor,
        shapes: &[&[usize]],
    ) -> Result<CudaPlan<S>> {
        plan_core_descriptor::<S>(ctx, desc, shapes)
    }

    fn execute(
        ctx: &mut CudaContext,
        plan: &CudaPlan<S>,
        alpha: S,
        inputs: &[&Tensor<S>],
        beta: S,
        output: &mut Tensor<S>,
    ) -> Result<()> {
        execute_plan(ctx, plan, alpha, inputs, beta, output)
    }
}

impl<S: Scalar> TensorSemiringFastPath<Standard<S>> for CudaBackend {
    type Plan = CudaPlan<S>;
    type Context = CudaContext;

    fn plan(
        ctx: &mut CudaContext,
        desc: &SemiringFastPathDescriptor,
        shapes: &[&[usize]],
    ) -> Result<CudaPlan<S>> {
        plan_fast_descriptor::<S>(ctx, desc, shapes)
    }

    fn execute(
        ctx: &mut CudaContext,
        plan: &CudaPlan<S>,
        alpha: S,
        inputs: &[&Tensor<S>],
        beta: S,
        output: &mut Tensor<S>,
    ) -> Result<()> {
        <Self as TensorSemiringCore<Standard<S>>>::execute(ctx, plan, alpha, inputs, beta, output)
    }

    fn has_fast_path(desc: SemiringFastPathDescriptor) -> bool {
        has_fast_path(desc)
    }
}

#[cfg(test)]
mod tests;
