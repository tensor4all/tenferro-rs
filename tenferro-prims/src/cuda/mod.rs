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
use std::sync::Arc;

#[cfg(unix)]
use libloading::os::unix::{Library as UnixLibrary, RTLD_GLOBAL, RTLD_NOW};
use tenferro_algebra::{Conjugate, Scalar, Standard};
use tenferro_device::cuda::runtime as device_cuda;
use tenferro_device::{Error, Result};
use tenferro_tensor::Tensor;

use crate::cpu::TempPool;
use crate::cuda_ffi::*;
use crate::{
    SemiringCoreDescriptor, SemiringFastPathDescriptor, TensorSemiringCore, TensorSemiringFastPath,
    TensorTempPoolContext,
};

mod analytic;
mod complex_real;
mod complex_scale;
mod custom;
mod execution;
mod family_common;
mod indexing;
mod metadata;
mod metadata_cast;
mod planning;
mod pointwise_ops;
mod rng;
mod runtime;
mod scalar;
mod scalar_type;
mod sort;
mod wrappers;

pub use analytic::CudaAnalyticPlan;
pub use complex_real::CudaComplexRealPlan;
pub use complex_scale::CudaComplexScalePlan;
use custom::CustomCudaRuntime;
use execution::{execute_plan, has_fast_path, plan_core_descriptor, plan_fast_descriptor};
use planning::{check_status, NativeCutensorPlan};
pub use rng::CudaRngPlan;
pub use scalar::CudaScalarPlan;
use wrappers::HandleWrapper;

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
    /// Helper runtime for custom pointwise complex kernels.
    custom: Arc<CustomCudaRuntime>,
    temp_pool: TempPool,
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

impl TensorTempPoolContext for CudaContext {
    fn take_temp_vec<T: Send + 'static>(&mut self, len: usize) -> Vec<T> {
        self.temp_pool.take_vec::<T>(len)
    }

    fn put_temp_vec<T: Send + 'static>(&mut self, vec: Vec<T>) {
        self.temp_pool.put_vec(vec);
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
        let custom = Arc::new(CustomCudaRuntime::new(0)?);

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
            custom,
            temp_pool: TempPool::default(),
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
    ///
    /// Delegates to the tensor-layer logical combine substrate by routing
    /// through singleton `stack` plus `squeeze_dim(0)`, which keeps the
    /// output on device and resolves logical values without reimplementing
    /// copy logic here.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_prims::{CudaBackend, CudaContext};
    /// use tenferro_tensor::Tensor;
    ///
    /// # fn demo(ctx: &mut CudaContext, x: &Tensor<f32>) {
    /// let resolved = CudaBackend::resolve_conj(ctx, x);
    /// assert!(!resolved.is_conjugated());
    /// # }
    /// ```
    pub fn resolve_conj<T: Scalar + Conjugate>(
        _ctx: &mut CudaContext,
        src: &Tensor<T>,
    ) -> Tensor<T> {
        if !src.is_conjugated() {
            return src.clone();
        }

        Tensor::stack(&[src], 0)
            .and_then(|tensor| tensor.squeeze_dim(0))
            .unwrap_or_else(|_| src.clone())
    }
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
