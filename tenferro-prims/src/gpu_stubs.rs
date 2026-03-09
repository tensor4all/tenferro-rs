use std::ffi::c_void;
use std::marker::PhantomData;

use tenferro_algebra::{Conjugate, Scalar, Standard};
use tenferro_device::{Error, Result};
use tenferro_tensor::Tensor;

use crate::{Extension, PlanCache, PrimDescriptor, TensorPrims};

// ===========================================================================
// CUDA stub types (used when `cuda` feature is NOT enabled)
// ===========================================================================

/// CUDA execution context (stub).
///
/// **Status: Stub.** This type exists as an API placeholder when the `cuda`
/// feature is not enabled. All operations on [`CudaBackend`] return errors.
/// Enable the `cuda` feature for the real implementation.
///
/// # Examples
///
/// ```ignore
/// // Aspirational API — not yet functional without `cuda` feature.
/// use tenferro_prims::CudaContext;
/// ```
#[cfg(not(feature = "cuda"))]
pub struct CudaContext {
    _stream: *mut c_void,
    _workspace: Vec<u8>,
    _plan_cache: PlanCache,
}

#[cfg(not(feature = "cuda"))]
impl CudaContext {
    /// Create a stub CUDA context (no-op).
    pub fn new() -> Self {
        Self {
            _stream: std::ptr::null_mut(),
            _workspace: Vec::new(),
            _plan_cache: PlanCache::new(),
        }
    }
}

#[cfg(not(feature = "cuda"))]
impl Default for CudaContext {
    fn default() -> Self {
        Self::new()
    }
}

/// CUDA plan (stub) — placeholder when `cuda` feature is not enabled.
///
/// **Status: Stub.** Enable the `cuda` feature for the real implementation.
#[cfg(not(feature = "cuda"))]
pub struct CudaPlan<T: Scalar> {
    _handle: *mut c_void,
    _workspace_size: usize,
    _marker: PhantomData<T>,
}

/// CUDA backend (stub) — placeholder when `cuda` feature is not enabled.
///
/// **Status: Stub.** All methods return errors. Enable the `cuda` feature
/// for the real implementation backed by cuTENSOR + cudarc.
///
/// # Examples
///
/// ```ignore
/// // Aspirational API — enable `cuda` feature for real backend.
/// use tenferro_prims::{CudaBackend, BackendRegistry};
///
/// let mut registry = BackendRegistry::new();
/// registry.load_cutensor("/usr/lib/libcutensor.so").unwrap();
/// ```
#[cfg(not(feature = "cuda"))]
pub struct CudaBackend {
    _handle: *mut c_void,
    _lib: libloading::Library,
}

#[cfg(not(feature = "cuda"))]
impl CudaBackend {
    /// Materialize a lazily-conjugated tensor on GPU.
    ///
    /// **Status: Stub fallback.** If data is CPU-accessible, this materializes
    /// conjugation into a new non-conjugated tensor. Otherwise it returns a
    /// clone of `src`.
    pub fn resolve_conj<T: Scalar + Conjugate>(
        _ctx: &mut CudaContext,
        src: &tenferro_tensor::Tensor<T>,
    ) -> tenferro_tensor::Tensor<T> {
        if !src.is_conjugated() {
            return src.clone();
        }

        let contiguous = src.contiguous(tenferro_tensor::MemoryOrder::ColumnMajor);
        let Some(data) = contiguous.buffer().as_slice() else {
            return src.clone();
        };
        let conjugated_data: Vec<T> = data.iter().map(|&v| v.conj()).collect();
        tenferro_tensor::Tensor::from_slice(
            &conjugated_data,
            src.dims(),
            tenferro_tensor::MemoryOrder::ColumnMajor,
        )
        .unwrap_or_else(|_| src.clone())
    }
}

#[cfg(not(feature = "cuda"))]
impl<S: Scalar> TensorPrims<Standard<S>> for CudaBackend {
    type Plan = CudaPlan<S>;
    type Context = CudaContext;

    fn plan(
        _ctx: &mut CudaContext,
        _desc: &PrimDescriptor,
        _shapes: &[&[usize]],
    ) -> Result<CudaPlan<S>> {
        Err(Error::DeviceError(
            "CUDA backend not available: load cuTENSOR library first".into(),
        ))
    }

    fn execute(
        _ctx: &mut CudaContext,
        _plan: &CudaPlan<S>,
        _alpha: S,
        _inputs: &[&Tensor<S>],
        _beta: S,
        _output: &mut Tensor<S>,
    ) -> Result<()> {
        Err(Error::DeviceError(
            "CUDA backend not available: load cuTENSOR library first".into(),
        ))
    }

    fn has_extension_for(_ext: Extension) -> bool {
        false
    }
}

// ===========================================================================
// ROCm stub types (always present — no real ROCm backend yet)
// ===========================================================================

/// ROCm execution context.
///
/// **Status: Not yet implemented.** This type exists as an API placeholder.
/// All operations on [`RocmBackend`] currently return errors.
///
/// When implemented, will encapsulate ROCm-side execution resources: a HIP
/// stream, GPU workspace buffer, and plan cache. Analogous to hipTENSOR's
/// handle.
///
/// # Examples
///
/// ```ignore
/// // Aspirational API — not yet functional.
/// use tenferro_prims::RocmContext;
///
/// // Created internally by RocmBackend::load_hiptensor()
/// ```
pub struct RocmContext {
    _stream: *mut c_void,
    _workspace: Vec<u8>,
    _plan_cache: PlanCache,
}

impl RocmContext {
    /// Create a stub ROCm context (no-op).
    pub fn new() -> Self {
        Self {
            _stream: std::ptr::null_mut(),
            _workspace: Vec::new(),
            _plan_cache: PlanCache::new(),
        }
    }
}

impl Default for RocmContext {
    fn default() -> Self {
        Self::new()
    }
}

/// ROCm plan — wraps a hipTENSOR plan handle.
///
/// **Status: Not yet implemented.** This type exists as an API placeholder.
///
/// Created by [`RocmBackend::plan`](TensorPrims::plan) and consumed by
/// [`RocmBackend::execute`](TensorPrims::execute).
pub struct RocmPlan<T: Scalar> {
    _handle: *mut c_void,
    _workspace_size: usize,
    _marker: PhantomData<T>,
}

/// ROCm backend using hipTENSOR via runtime dlopen.
///
/// **Status: Not yet implemented.** All methods currently return errors.
/// The type exists to define the intended API surface. `plan()` and
/// `execute()` return `Err(DeviceError)`. `load_hiptensor()` on
/// [`crate::BackendRegistry`] also returns an error.
///
/// When implemented, will be loaded at runtime from a user-provided `.so`
/// path with no compile-time ROCm SDK dependency. Will implement
/// [`TensorPrims<Standard<T>>`](TensorPrims) for standard arithmetic on
/// AMD GPUs.
///
/// hipTENSOR natively supports `Contract`, `Permute`, `Reduce`, and
/// `ElementwiseMul`. `AntiTrace`/`AntiDiag` will be composed via
/// `Contract(eye, dC)`.
///
/// # Examples
///
/// ```ignore
/// // Aspirational API — not yet functional.
/// use tenferro_prims::{RocmBackend, BackendRegistry};
///
/// let mut registry = BackendRegistry::new();
/// registry.load_hiptensor("/usr/lib/libhiptensor.so").unwrap();
/// ```
pub struct RocmBackend {
    _handle: *mut c_void,
    _lib: libloading::Library,
}

impl RocmBackend {
    /// Materialize a lazily-conjugated tensor on GPU.
    ///
    /// **Status: Stub fallback.** If data is CPU-accessible, this materializes
    /// conjugation into a new non-conjugated tensor. Otherwise it returns a
    /// clone of `src`.
    ///
    /// When implemented, will use `ElementwiseUnary(Conj)` via hipTENSOR
    /// to produce a new tensor with `conjugated = false`.
    pub fn resolve_conj<T: Scalar + Conjugate>(
        _ctx: &mut RocmContext,
        src: &tenferro_tensor::Tensor<T>,
    ) -> tenferro_tensor::Tensor<T> {
        if !src.is_conjugated() {
            return src.clone();
        }

        let contiguous = src.contiguous(tenferro_tensor::MemoryOrder::ColumnMajor);
        let Some(data) = contiguous.buffer().as_slice() else {
            return src.clone();
        };
        let conjugated_data: Vec<T> = data.iter().map(|&v| v.conj()).collect();
        tenferro_tensor::Tensor::from_slice(
            &conjugated_data,
            src.dims(),
            tenferro_tensor::MemoryOrder::ColumnMajor,
        )
        .unwrap_or_else(|_| src.clone())
    }
}

impl<S: Scalar> TensorPrims<Standard<S>> for RocmBackend {
    type Plan = RocmPlan<S>;
    type Context = RocmContext;

    fn plan(
        _ctx: &mut RocmContext,
        _desc: &PrimDescriptor,
        _shapes: &[&[usize]],
    ) -> Result<RocmPlan<S>> {
        Err(Error::DeviceError(
            "ROCm backend not available: load hipTENSOR library first".into(),
        ))
    }

    fn execute(
        _ctx: &mut RocmContext,
        _plan: &RocmPlan<S>,
        _alpha: S,
        _inputs: &[&Tensor<S>],
        _beta: S,
        _output: &mut Tensor<S>,
    ) -> Result<()> {
        Err(Error::DeviceError(
            "ROCm backend not available: load hipTENSOR library first".into(),
        ))
    }

    fn has_extension_for(_ext: Extension) -> bool {
        // Not yet implemented. When available, hipTENSOR will support
        // Contract and ElementwiseMul for f32/f64/Complex.
        false
    }
}

#[cfg(test)]
mod tests;
