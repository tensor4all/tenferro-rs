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
//! use tenferro_prims::{CudaBackend, CudaContext, TensorPrims, PrimDescriptor};
//!
//! let (backend, mut ctx) = CudaBackend::load("/usr/lib/libcutensor.so").unwrap();
//! ```

use std::ffi::c_void;
use std::marker::PhantomData;
use std::ptr;
use std::sync::Arc;

use num_complex::{Complex32, Complex64};
use tenferro_algebra::{Scalar, Standard};
use tenferro_device::{Error, Result};
use tenferro_tensor::Tensor;

use crate::cuda_ffi::*;
use crate::{Extension, PlanCache, PrimDescriptor, TensorPrims};

// ============================================================================
// Error helper
// ============================================================================

/// Check a cuTENSOR status code, converting non-success to Error.
fn check_status(status: cutensorStatus_t, context: &str) -> Result<()> {
    if status == CUTENSOR_STATUS_SUCCESS {
        Ok(())
    } else {
        Err(Error::DeviceError(format!(
            "cuTENSOR error {status} in {context}"
        )))
    }
}

// ============================================================================
// RAII wrappers
// ============================================================================

/// RAII wrapper for `cutensorHandle_t`. Drop calls `cutensorDestroy`.
pub(crate) struct HandleWrapper {
    pub(crate) raw: cutensorHandle_t,
    vtable: Arc<CutensorVtable>,
}

impl Drop for HandleWrapper {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe {
                (self.vtable.destroy)(self.raw);
            }
        }
    }
}

/// RAII wrapper for `cutensorTensorDescriptor_t`.
pub(crate) struct TensorDescWrapper {
    pub(crate) raw: cutensorTensorDescriptor_t,
    vtable: Arc<CutensorVtable>,
}

impl Drop for TensorDescWrapper {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe {
                (self.vtable.destroy_tensor_descriptor)(self.raw);
            }
        }
    }
}

/// RAII wrapper for `cutensorOperationDescriptor_t`.
pub(crate) struct OpDescWrapper {
    pub(crate) raw: cutensorOperationDescriptor_t,
    vtable: Arc<CutensorVtable>,
}

impl Drop for OpDescWrapper {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe {
                (self.vtable.destroy_operation_descriptor)(self.raw);
            }
        }
    }
}

/// RAII wrapper for `cutensorPlanPreference_t`.
pub(crate) struct PlanPrefWrapper {
    pub(crate) raw: cutensorPlanPreference_t,
    vtable: Arc<CutensorVtable>,
}

impl Drop for PlanPrefWrapper {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe {
                (self.vtable.destroy_plan_preference)(self.raw);
            }
        }
    }
}

/// RAII wrapper for `cutensorPlan_t`.
pub(crate) struct PlanWrapper {
    pub(crate) raw: cutensorPlan_t,
    vtable: Arc<CutensorVtable>,
}

impl Drop for PlanWrapper {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe {
                (self.vtable.destroy_plan)(self.raw);
            }
        }
    }
}

// ============================================================================
// CutensorType trait — maps Rust scalars to cuTENSOR types
// ============================================================================

/// Maps a Rust scalar type to its cuTENSOR data type and compute descriptor.
pub(crate) trait CutensorType: Scalar {
    fn data_type() -> CutensorDataType;
    fn compute_descriptor(vtable: &CutensorVtable) -> cutensorComputeDescriptor_t;
}

impl CutensorType for f32 {
    fn data_type() -> CutensorDataType {
        CutensorDataType::R_32F
    }
    fn compute_descriptor(vtable: &CutensorVtable) -> cutensorComputeDescriptor_t {
        vtable.compute_desc_32f
    }
}

impl CutensorType for f64 {
    fn data_type() -> CutensorDataType {
        CutensorDataType::R_64F
    }
    fn compute_descriptor(vtable: &CutensorVtable) -> cutensorComputeDescriptor_t {
        vtable.compute_desc_64f
    }
}

impl CutensorType for Complex32 {
    fn data_type() -> CutensorDataType {
        CutensorDataType::C_32F
    }
    fn compute_descriptor(vtable: &CutensorVtable) -> cutensorComputeDescriptor_t {
        vtable.compute_desc_32f
    }
}

impl CutensorType for Complex64 {
    fn data_type() -> CutensorDataType {
        CutensorDataType::C_64F
    }
    fn compute_descriptor(vtable: &CutensorVtable) -> cutensorComputeDescriptor_t {
        vtable.compute_desc_64f
    }
}

// ============================================================================
// Public types
// ============================================================================

/// CUDA execution context backed by cudarc + cuTENSOR vtable.
///
/// Encapsulates GPU-side execution resources: a cudarc device handle,
/// cuTENSOR function vtable, workspace buffer, and plan cache.
///
/// # Examples
///
/// ```ignore
/// // Created internally by CudaBackend::load()
/// use tenferro_prims::CudaContext;
/// ```
pub struct CudaContext {
    /// cuTENSOR library handle (RAII — Drop calls cutensorDestroy).
    handle: HandleWrapper,
    /// cudarc stream handle for GPU memory and stream management.
    stream: Arc<cudarc::driver::CudaStream>,
    /// cuTENSOR function pointer vtable loaded via libloading.
    vtable: Arc<CutensorVtable>,
    /// GPU workspace buffer for cuTENSOR operations.
    workspace: Vec<u8>,
    /// Plan cache for reusing compiled plans.
    plan_cache: PlanCache,
}

/// CUDA plan — wraps a cuTENSOR operation plan handle.
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
    /// Compiled cuTENSOR plan (RAII — Drop calls cutensorDestroyPlan).
    plan: PlanWrapper,
    /// Operation descriptor (for cache key matching and execute dispatch).
    desc: PrimDescriptor,
    /// Required workspace size in bytes.
    workspace_size: u64,
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
        matches!(ext, Extension::Contract | Extension::ElementwiseMul)
    }
}
