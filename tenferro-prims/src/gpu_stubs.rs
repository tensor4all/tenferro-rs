use std::ffi::c_void;
use std::marker::PhantomData;

use strided_traits::ScalarBase;
use strided_view::{StridedView, StridedViewMut};
use tenferro_algebra::{Scalar, Standard};
use tenferro_device::{Error, Result};

use crate::{Extension, PlanCache, PrimDescriptor, TensorPrims};

// ===========================================================================
// GPU Backends (not yet implemented — runtime dlopen via libloading)
// ===========================================================================

/// CUDA execution context.
///
/// **Status: Not yet implemented.** This type exists as an API placeholder.
/// All operations on [`CudaBackend`] currently return errors.
///
/// When implemented, will encapsulate CUDA-side execution resources: a CUDA
/// stream, GPU workspace buffer, and plan cache. Analogous to cuTENSOR's
/// `cutensorHandle_t`.
///
/// # Examples
///
/// ```ignore
/// // Aspirational API — not yet functional.
/// use tenferro_prims::CudaContext;
///
/// // Created internally by CudaBackend::load_cutensor()
/// ```
pub struct CudaContext {
    _stream: *mut c_void,
    _workspace: Vec<u8>,
    _plan_cache: PlanCache,
}

/// CUDA plan — wraps a cuTENSOR plan handle.
///
/// **Status: Not yet implemented.** This type exists as an API placeholder.
///
/// Created by [`CudaBackend::plan`](TensorPrims::plan) and consumed by
/// [`CudaBackend::execute`](TensorPrims::execute).
pub struct CudaPlan<T: ScalarBase> {
    _handle: *mut c_void,
    _workspace_size: usize,
    _marker: PhantomData<T>,
}

/// CUDA backend using cuTENSOR via runtime dlopen.
///
/// **Status: Not yet implemented.** All methods currently return errors.
/// The type exists to define the intended API surface. `plan()` and
/// `execute()` return `Err(DeviceError)`. `load_cutensor()` on
/// [`BackendRegistry`] also returns an error.
///
/// When implemented, will be loaded at runtime from a user-provided `.so`
/// path with no compile-time CUDA SDK dependency. Will implement
/// [`TensorPrims<Standard<T>>`](TensorPrims) for standard arithmetic on
/// NVIDIA GPUs.
///
/// cuTENSOR natively supports `Contract`, `Permute`, `Reduce`, and
/// `ElementwiseMul`. `AntiTrace`/`AntiDiag` will be composed via
/// `Contract(eye, dC)`.
///
/// # Examples
///
/// ```ignore
/// // Aspirational API — not yet functional.
/// use tenferro_prims::{CudaBackend, BackendRegistry};
///
/// let mut registry = BackendRegistry::new();
/// registry.load_cutensor("/usr/lib/libcutensor.so").unwrap();
/// ```
pub struct CudaBackend {
    _handle: *mut c_void,
    _lib: libloading::Library,
}

impl CudaBackend {
    /// Materialize a lazily-conjugated tensor on GPU.
    ///
    /// **Status: Not yet implemented.** Currently panics with
    /// `unimplemented!`.
    ///
    /// When implemented, will use `ElementwiseUnary(Conj)` via cuTENSOR
    /// to produce a new tensor with `conjugated = false`.
    pub fn resolve_conj<T: Scalar>(
        _ctx: &mut CudaContext,
        _src: &tenferro_tensor::Tensor<T>,
    ) -> tenferro_tensor::Tensor<T> {
        unimplemented!("CUDA backend not available: load cuTENSOR library first")
    }
}

impl<S: Scalar> TensorPrims<Standard<S>> for CudaBackend {
    type Plan<T: ScalarBase> = CudaPlan<T>;
    type Context = CudaContext;

    fn plan<T: ScalarBase>(
        _ctx: &mut CudaContext,
        _desc: &PrimDescriptor,
        _shapes: &[&[usize]],
    ) -> Result<CudaPlan<T>> {
        Err(Error::DeviceError(
            "CUDA backend not available: load cuTENSOR library first".into(),
        ))
    }

    fn execute<T: ScalarBase>(
        _ctx: &mut CudaContext,
        _plan: &CudaPlan<T>,
        _alpha: T,
        _inputs: &[&StridedView<T>],
        _beta: T,
        _output: &mut StridedViewMut<T>,
    ) -> Result<()> {
        Err(Error::DeviceError(
            "CUDA backend not available: load cuTENSOR library first".into(),
        ))
    }

    fn has_extension_for<T: ScalarBase>(_ext: Extension) -> bool {
        // Not yet implemented. When available, cuTENSOR will support
        // Contract and ElementwiseMul for f32/f64/Complex.
        false
    }
}

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

/// ROCm plan — wraps a hipTENSOR plan handle.
///
/// **Status: Not yet implemented.** This type exists as an API placeholder.
///
/// Created by [`RocmBackend::plan`](TensorPrims::plan) and consumed by
/// [`RocmBackend::execute`](TensorPrims::execute).
pub struct RocmPlan<T: ScalarBase> {
    _handle: *mut c_void,
    _workspace_size: usize,
    _marker: PhantomData<T>,
}

/// ROCm backend using hipTENSOR via runtime dlopen.
///
/// **Status: Not yet implemented.** All methods currently return errors.
/// The type exists to define the intended API surface. `plan()` and
/// `execute()` return `Err(DeviceError)`. `load_hiptensor()` on
/// [`BackendRegistry`] also returns an error.
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
    /// **Status: Not yet implemented.** Currently panics with
    /// `unimplemented!`.
    ///
    /// When implemented, will use `ElementwiseUnary(Conj)` via hipTENSOR
    /// to produce a new tensor with `conjugated = false`.
    pub fn resolve_conj<T: Scalar>(
        _ctx: &mut RocmContext,
        _src: &tenferro_tensor::Tensor<T>,
    ) -> tenferro_tensor::Tensor<T> {
        unimplemented!("ROCm backend not available: load hipTENSOR library first")
    }
}

impl<S: Scalar> TensorPrims<Standard<S>> for RocmBackend {
    type Plan<T: ScalarBase> = RocmPlan<T>;
    type Context = RocmContext;

    fn plan<T: ScalarBase>(
        _ctx: &mut RocmContext,
        _desc: &PrimDescriptor,
        _shapes: &[&[usize]],
    ) -> Result<RocmPlan<T>> {
        Err(Error::DeviceError(
            "ROCm backend not available: load hipTENSOR library first".into(),
        ))
    }

    fn execute<T: ScalarBase>(
        _ctx: &mut RocmContext,
        _plan: &RocmPlan<T>,
        _alpha: T,
        _inputs: &[&StridedView<T>],
        _beta: T,
        _output: &mut StridedViewMut<T>,
    ) -> Result<()> {
        Err(Error::DeviceError(
            "ROCm backend not available: load hipTENSOR library first".into(),
        ))
    }

    fn has_extension_for<T: ScalarBase>(_ext: Extension) -> bool {
        // Not yet implemented. When available, hipTENSOR will support
        // Contract and ElementwiseMul for f32/f64/Complex.
        false
    }
}
