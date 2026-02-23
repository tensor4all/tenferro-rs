//! Real CUDA backend using cuTENSOR v2 via cudarc + libloading.
//!
//! This module is compiled only when `feature = "cuda"` is enabled.
//! It replaces the stub types from `gpu_stubs.rs`.
//!
//! cudarc is used with its default `dynamic-loading` feature, so **no
//! CUDA SDK is required at compile time**. The CUDA driver is loaded
//! via dlopen at runtime.
//!
//! **Status: API skeleton only.** All function bodies use `todo!()`.
//! Real implementations will be added during GPU testing.
//!
//! # Examples
//!
//! ```ignore
//! // Aspirational API — not yet functional.
//! use tenferro_prims::{CudaBackend, CudaContext, TensorPrims, PrimDescriptor};
//!
//! let (backend, mut ctx) = CudaBackend::load("/usr/lib/libcutensor.so").unwrap();
//! ```

use std::marker::PhantomData;
use std::sync::Arc;

use tenferro_algebra::{Scalar, Standard};
use tenferro_device::Result;
use tenferro_tensor::Tensor;

use crate::cuda_ffi::CutensorVtable;
use crate::{Extension, PlanCache, PrimDescriptor, TensorPrims};

/// CUDA execution context backed by cudarc + cuTENSOR vtable.
///
/// Encapsulates GPU-side execution resources: a cudarc device handle,
/// cuTENSOR function vtable, workspace buffer, and plan cache.
///
/// **Status: API skeleton only.** Created by [`CudaBackend::load`].
///
/// # Examples
///
/// ```ignore
/// // Created internally by CudaBackend::load()
/// use tenferro_prims::CudaContext;
/// ```
pub struct CudaContext {
    /// cudarc stream handle for GPU memory and stream management.
    _stream: Arc<cudarc::driver::CudaStream>,
    /// cuTENSOR function pointer vtable loaded via libloading.
    _vtable: CutensorVtable,
    /// GPU workspace buffer for cuTENSOR operations.
    _workspace: Vec<u8>,
    /// Plan cache for reusing compiled plans.
    _plan_cache: PlanCache,
}

/// CUDA plan — wraps a cuTENSOR operation plan handle.
///
/// **Status: API skeleton only.** Created by [`CudaBackend::plan`](TensorPrims::plan)
/// and consumed by [`CudaBackend::execute`](TensorPrims::execute).
///
/// # Examples
///
/// ```ignore
/// use tenferro_prims::{CudaBackend, CudaContext, TensorPrims, PrimDescriptor};
///
/// let (_, mut ctx) = CudaBackend::load("/usr/lib/libcutensor.so").unwrap();
/// let plan = CudaBackend::plan::<f64>(&mut ctx, &desc, &shapes).unwrap();
/// ```
pub struct CudaPlan<T: Scalar> {
    _desc: PrimDescriptor,
    _workspace_size: usize,
    _marker: PhantomData<T>,
}

/// CUDA backend using cuTENSOR v2 via cudarc + libloading.
///
/// Loaded at runtime from a user-provided `.so` path. cudarc uses
/// `dynamic-loading` by default — no compile-time CUDA SDK dependency.
///
/// **Status: API skeleton only.** `load()` uses `todo!()`.
/// `plan()` and `execute()` use `todo!()`.
/// `has_extension_for` returns `true` for `Contract` and `ElementwiseMul`.
///
/// # Examples
///
/// ```ignore
/// use tenferro_prims::{CudaBackend, BackendRegistry};
///
/// let mut registry = BackendRegistry::new();
/// registry.load_cutensor("/usr/lib/libcutensor.so").unwrap();
/// ```
pub struct CudaBackend {
    _lib: libloading::Library,
}

impl CudaBackend {
    /// Load cuTENSOR library and initialize CUDA context.
    ///
    /// **Status: Not yet implemented.** Currently uses `todo!()`.
    ///
    /// When implemented, will:
    /// 1. Open the shared library via `libloading`
    /// 2. Populate [`CutensorVtable`] with function pointers
    /// 3. Initialize CUDA device
    /// 4. Return `(CudaBackend, CudaContext)` pair
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_prims::CudaBackend;
    ///
    /// let (backend, ctx) = CudaBackend::load("/usr/lib/libcutensor.so").unwrap();
    /// ```
    pub fn load(_path: &str) -> Result<(Self, CudaContext)> {
        todo!("CudaBackend::load — cuTENSOR runtime loading not yet implemented")
    }

    /// Materialize a lazily-conjugated tensor on GPU.
    ///
    /// **Status: Not yet implemented.** Currently uses `todo!()`.
    ///
    /// When implemented, will use `ElementwiseUnary(Conj)` via cuTENSOR
    /// to produce a new tensor with `conjugated = false`.
    pub fn resolve_conj<T: Scalar>(_ctx: &mut CudaContext, _src: &Tensor<T>) -> Tensor<T> {
        todo!("CudaBackend::resolve_conj — not yet implemented")
    }
}

impl<S: Scalar> TensorPrims<Standard<S>> for CudaBackend {
    type Plan<T: Scalar> = CudaPlan<T>;
    type Context = CudaContext;

    fn plan<T: Scalar>(
        _ctx: &mut CudaContext,
        _desc: &PrimDescriptor,
        _shapes: &[&[usize]],
    ) -> Result<CudaPlan<T>> {
        todo!("CudaBackend::plan — cuTENSOR plan creation not yet implemented")
    }

    fn execute<T: Scalar>(
        _ctx: &mut CudaContext,
        _plan: &CudaPlan<T>,
        _alpha: T,
        _inputs: &[&Tensor<T>],
        _beta: T,
        _output: &mut Tensor<T>,
    ) -> Result<()> {
        todo!("CudaBackend::execute — cuTENSOR execution not yet implemented")
    }

    fn has_extension_for<T: Scalar>(ext: Extension) -> bool {
        // cuTENSOR natively supports Contract and ElementwiseMul
        matches!(ext, Extension::Contract | Extension::ElementwiseMul)
    }
}
