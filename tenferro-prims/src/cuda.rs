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
use num_complex::{Complex32, Complex64};
use tenferro_algebra::{Conjugate, Scalar, Standard};
use tenferro_device::{Error, Result};
use tenferro_tensor::Tensor;

use crate::cuda_ffi::*;
use crate::typed_dispatch::{
    dispatch_complex_scalar_type, dispatch_real_scalar_type, dispatch_standard_scalar_type,
};
use crate::{
    validate_execute_inputs, validate_shape_count, PlanCache, SemiringBinaryOp,
    SemiringCoreDescriptor, SemiringFastPathDescriptor, TensorSemiringCore, TensorSemiringFastPath,
};

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
        CUTENSOR_R_32F
    }
    fn compute_descriptor(vtable: &CutensorVtable) -> cutensorComputeDescriptor_t {
        vtable.compute_desc_32f
    }
}

impl CutensorType for f64 {
    fn data_type() -> CutensorDataType {
        CUTENSOR_R_64F
    }
    fn compute_descriptor(vtable: &CutensorVtable) -> cutensorComputeDescriptor_t {
        vtable.compute_desc_64f
    }
}

impl CutensorType for Complex32 {
    fn data_type() -> CutensorDataType {
        CUTENSOR_C_32F
    }
    fn compute_descriptor(vtable: &CutensorVtable) -> cutensorComputeDescriptor_t {
        vtable.compute_desc_32f
    }
}

impl CutensorType for Complex64 {
    fn data_type() -> CutensorDataType {
        CUTENSOR_C_64F
    }
    fn compute_descriptor(vtable: &CutensorVtable) -> cutensorComputeDescriptor_t {
        vtable.compute_desc_64f
    }
}

// ============================================================================
// Helper functions
// ============================================================================

/// Compute default column-major strides for a given shape.
fn default_col_major_strides(shape: &[usize]) -> Vec<isize> {
    let n = shape.len();
    if n == 0 {
        return vec![];
    }
    let mut strides = vec![0isize; n];
    strides[0] = 1;
    for i in 1..n {
        strides[i] = strides[i - 1] * shape[i - 1] as isize;
    }
    strides
}

/// Create a cuTENSOR tensor descriptor from shape and strides.
fn create_tensor_desc(
    handle: cutensorHandle_t,
    vtable: &Arc<CutensorVtable>,
    shape: &[usize],
    strides: &[isize],
    data_type: CutensorDataType,
) -> Result<TensorDescWrapper> {
    let num_modes = shape.len() as u32;
    let extent: Vec<i64> = shape.iter().map(|&d| d as i64).collect();
    let stride: Vec<i64> = strides.iter().map(|&s| s as i64).collect();
    let mut raw: cutensorTensorDescriptor_t = ptr::null_mut();
    let status = unsafe {
        (vtable.create_tensor_descriptor)(
            handle,
            &mut raw,
            num_modes,
            extent.as_ptr(),
            stride.as_ptr(),
            data_type,
            128, // alignment requirement
        )
    };
    check_status(status, "cutensorCreateTensorDescriptor")?;
    Ok(TensorDescWrapper {
        raw,
        vtable: Arc::clone(vtable),
    })
}

/// Create a cuTENSOR plan (shared logic for all operation types).
///
/// Takes an already-created operation descriptor, creates plan preference,
/// estimates workspace, and builds the plan.
fn build_cutensor_plan(
    handle: cutensorHandle_t,
    vtable: &Arc<CutensorVtable>,
    op_desc: &OpDescWrapper,
) -> Result<(PlanWrapper, u64)> {
    // Create plan preference
    let mut pref_raw: cutensorPlanPreference_t = ptr::null_mut();
    let status = unsafe {
        (vtable.create_plan_preference)(
            handle,
            &mut pref_raw,
            CUTENSOR_ALGO_DEFAULT,
            CUTENSOR_JIT_MODE_NONE,
        )
    };
    check_status(status, "cutensorCreatePlanPreference")?;
    let _pref = PlanPrefWrapper {
        raw: pref_raw,
        vtable: Arc::clone(vtable),
    };

    // Estimate workspace
    let mut workspace_size: u64 = 0;
    let status = unsafe {
        (vtable.estimate_workspace_size)(
            handle,
            op_desc.raw,
            _pref.raw,
            CUTENSOR_WORKSPACE_DEFAULT,
            &mut workspace_size,
        )
    };
    check_status(status, "cutensorEstimateWorkspaceSize")?;

    // Create plan
    let mut plan_raw: cutensorPlan_t = ptr::null_mut();
    let status = unsafe {
        (vtable.create_plan)(
            handle,
            &mut plan_raw,
            op_desc.raw,
            _pref.raw,
            workspace_size,
        )
    };
    check_status(status, "cutensorCreatePlan")?;

    Ok((
        PlanWrapper {
            raw: plan_raw,
            vtable: Arc::clone(vtable),
        },
        workspace_size,
    ))
    // _pref RAII wrapper drops here, freeing cuTENSOR plan preference
}

/// Get cuTENSOR data type for a Scalar type, returning an error if unsupported.
fn scalar_data_type<T: Scalar>() -> Result<CutensorDataType> {
    macro_rules! cutensor_data_type_for {
        (f32) => {
            CUTENSOR_R_32F
        };
        (f64) => {
            CUTENSOR_R_64F
        };
        (Complex32) => {
            CUTENSOR_C_32F
        };
        (Complex64) => {
            CUTENSOR_C_64F
        };
    }

    dispatch_standard_scalar_type!(T, Concrete, {
        return Ok(cutensor_data_type_for!(Concrete));
    });

    Err(Error::DeviceError(
        "Unsupported scalar type for CUDA backend".into(),
    ))
}

/// Get cuTENSOR compute descriptor for a Scalar type, returning an error if unsupported.
fn scalar_compute_descriptor<T: Scalar>(
    vtable: &CutensorVtable,
) -> Result<cutensorComputeDescriptor_t> {
    dispatch_real_scalar_type!(T, Concrete, {
        let _ = std::marker::PhantomData::<Concrete>;
        return if std::mem::size_of::<Concrete>() == std::mem::size_of::<f32>() {
            Ok(vtable.compute_desc_32f)
        } else {
            Ok(vtable.compute_desc_64f)
        };
    });
    dispatch_complex_scalar_type!(T, Concrete, {
        let _ = std::marker::PhantomData::<Concrete>;
        return if std::mem::size_of::<Concrete>() == std::mem::size_of::<Complex32>() {
            Ok(vtable.compute_desc_32f)
        } else {
            Ok(vtable.compute_desc_64f)
        };
    });

    Err(Error::DeviceError(
        "Unsupported scalar type for CUDA backend".into(),
    ))
}

/// Create a cuTENSOR contraction plan.
///
/// Used by Contract, BatchedGemm, Trace, AntiTrace, AntiDiag — all map to
/// `cutensorCreateContraction`.
fn plan_contraction(
    ctx: &mut CudaContext,
    data_type: CutensorDataType,
    compute: cutensorComputeDescriptor_t,
    modes_a: &[i32],
    shape_a: &[usize],
    strides_a: &[isize],
    modes_b: &[i32],
    shape_b: &[usize],
    strides_b: &[isize],
    modes_c: &[i32],
    shape_c: &[usize],
    strides_c: &[isize],
) -> Result<(PlanWrapper, u64)> {
    let vtable = &ctx.vtable;
    let handle = ctx.handle.raw;

    let desc_a = create_tensor_desc(handle, vtable, shape_a, strides_a, data_type)?;
    let desc_b = create_tensor_desc(handle, vtable, shape_b, strides_b, data_type)?;
    let desc_c = create_tensor_desc(handle, vtable, shape_c, strides_c, data_type)?;
    // D descriptor same as C (in-place output: D = alpha * contract(A, B) + beta * C)
    let desc_d = create_tensor_desc(handle, vtable, shape_c, strides_c, data_type)?;

    let mut op_raw: cutensorOperationDescriptor_t = ptr::null_mut();
    let status = unsafe {
        (vtable.create_contraction)(
            handle,
            &mut op_raw,
            desc_a.raw,
            modes_a.as_ptr(),
            CUTENSOR_OP_IDENTITY,
            desc_b.raw,
            modes_b.as_ptr(),
            CUTENSOR_OP_IDENTITY,
            desc_c.raw,
            modes_c.as_ptr(),
            CUTENSOR_OP_IDENTITY,
            desc_d.raw,
            modes_c.as_ptr(),
            compute,
        )
    };
    check_status(status, "cutensorCreateContraction")?;
    let op_desc = OpDescWrapper {
        raw: op_raw,
        vtable: Arc::clone(vtable),
    };

    build_cutensor_plan(handle, vtable, &op_desc)
    // desc_a, desc_b, desc_c, desc_d, op_desc all drop here
}

/// Create a cuTENSOR permutation plan.
fn plan_permutation(
    ctx: &mut CudaContext,
    data_type: CutensorDataType,
    compute: cutensorComputeDescriptor_t,
    modes_a: &[i32],
    shape_a: &[usize],
    strides_a: &[isize],
    modes_b: &[i32],
    shape_b: &[usize],
    strides_b: &[isize],
) -> Result<(PlanWrapper, u64)> {
    let vtable = &ctx.vtable;
    let handle = ctx.handle.raw;
    let desc_a = create_tensor_desc(handle, vtable, shape_a, strides_a, data_type)?;
    let desc_b = create_tensor_desc(handle, vtable, shape_b, strides_b, data_type)?;

    let mut op_raw: cutensorOperationDescriptor_t = ptr::null_mut();
    let status = unsafe {
        (vtable.create_permutation)(
            handle,
            &mut op_raw,
            desc_a.raw,
            modes_a.as_ptr(),
            CUTENSOR_OP_IDENTITY,
            desc_b.raw,
            modes_b.as_ptr(),
            compute,
        )
    };
    check_status(status, "cutensorCreatePermutation")?;
    let op_desc = OpDescWrapper {
        raw: op_raw,
        vtable: Arc::clone(vtable),
    };
    build_cutensor_plan(handle, vtable, &op_desc)
}

/// Create a cuTENSOR reduction plan.
fn plan_reduction(
    ctx: &mut CudaContext,
    data_type: CutensorDataType,
    compute: cutensorComputeDescriptor_t,
    modes_a: &[i32],
    shape_a: &[usize],
    strides_a: &[isize],
    modes_c: &[i32],
    shape_c: &[usize],
    strides_c: &[isize],
    reduce_op: CutensorOperator,
) -> Result<(PlanWrapper, u64)> {
    let vtable = &ctx.vtable;
    let handle = ctx.handle.raw;
    let desc_a = create_tensor_desc(handle, vtable, shape_a, strides_a, data_type)?;
    let desc_c = create_tensor_desc(handle, vtable, shape_c, strides_c, data_type)?;
    // D descriptor same as C
    let desc_d = create_tensor_desc(handle, vtable, shape_c, strides_c, data_type)?;

    let mut op_raw: cutensorOperationDescriptor_t = ptr::null_mut();
    let status = unsafe {
        (vtable.create_reduction)(
            handle,
            &mut op_raw,
            desc_a.raw,
            modes_a.as_ptr(),
            CUTENSOR_OP_IDENTITY,
            desc_c.raw,
            modes_c.as_ptr(),
            CUTENSOR_OP_IDENTITY,
            desc_d.raw,
            modes_c.as_ptr(),
            reduce_op,
            compute,
        )
    };
    check_status(status, "cutensorCreateReduction")?;
    let op_desc = OpDescWrapper {
        raw: op_raw,
        vtable: Arc::clone(vtable),
    };
    build_cutensor_plan(handle, vtable, &op_desc)
}

/// Create a cuTENSOR elementwise binary plan.
fn plan_elementwise_binary(
    ctx: &mut CudaContext,
    data_type: CutensorDataType,
    compute: cutensorComputeDescriptor_t,
    modes: &[i32],
    shape: &[usize],
    strides: &[isize],
    op: CutensorOperator,
) -> Result<(PlanWrapper, u64)> {
    let vtable = &ctx.vtable;
    let handle = ctx.handle.raw;
    let desc_a = create_tensor_desc(handle, vtable, shape, strides, data_type)?;
    // C and D same shape/strides as A
    let desc_c = create_tensor_desc(handle, vtable, shape, strides, data_type)?;
    let desc_d = create_tensor_desc(handle, vtable, shape, strides, data_type)?;

    let mut op_raw: cutensorOperationDescriptor_t = ptr::null_mut();
    let status = unsafe {
        (vtable.create_elementwise_binary)(
            handle,
            &mut op_raw,
            desc_a.raw,
            modes.as_ptr(),
            CUTENSOR_OP_IDENTITY,
            desc_c.raw,
            modes.as_ptr(),
            CUTENSOR_OP_IDENTITY,
            desc_d.raw,
            modes.as_ptr(),
            op,
            compute,
        )
    };
    check_status(status, "cutensorCreateElementwiseBinary")?;
    let op_desc = OpDescWrapper {
        raw: op_raw,
        vtable: Arc::clone(vtable),
    };
    build_cutensor_plan(handle, vtable, &op_desc)
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
    /// CUDA device ordinal used for runtime API calls.
    device_id: usize,
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

pub struct CudaPlan<T: Scalar> {
    /// Compiled cuTENSOR plan (RAII — Drop calls cutensorDestroyPlan).
    plan: PlanWrapper,
    /// Family descriptor (for execute dispatch).
    desc: CudaPlanDescriptor,
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

        // 1. Open shared library
        let lib = unsafe { DynamicLibrary::open(path, RTLD_NOW) }
            .map_err(|e| Error::DeviceError(format!("Failed to load cuTENSOR: {e}")))?;

        // 2. Populate vtable
        let vtable = unsafe { CutensorVtable::load(lib.handle()) }
            .map_err(|e| Error::DeviceError(format!("Failed to load cuTENSOR symbols: {e}")))?;
        let vtable = Arc::new(vtable);

        // 3. Initialize CUDA runtime state before creating the cuTENSOR handle.
        cudarc::runtime::result::device::set(0)
            .map_err(|e| Error::DeviceError(format!("CUDA runtime init failed: {e:?}")))?;

        // 4. Initialize cuTENSOR handle now that the CUDA context is active.
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
            workspace: Vec::new(),
            plan_cache: PlanCache::new(),
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
    pub fn resolve_conj<T: Scalar + Conjugate>(
        _ctx: &mut CudaContext,
        src: &Tensor<T>,
    ) -> Tensor<T> {
        if !src.is_conjugated() {
            return src.clone();
        }

        let contiguous = src.contiguous(tenferro_tensor::MemoryOrder::ColumnMajor);
        let Some(data) = contiguous.buffer().as_slice() else {
            return src.clone();
        };
        let conjugated_data: Vec<T> = data.iter().map(|&v| v.conj()).collect();
        Tensor::from_slice(
            &conjugated_data,
            src.dims(),
            tenferro_tensor::MemoryOrder::ColumnMajor,
        )
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
        cudarc::runtime::result::device::set(ctx.device_id as i32)
            .map_err(|e| Error::DeviceError(format!("CUDA runtime set-device failed: {e:?}")))?;
        // Resolve cuTENSOR data type and compute descriptor for the algebra scalar.
        // This uses TypeId dispatch since the trait bound is Scalar (not CutensorType).
        let data_type = scalar_data_type::<S>()?;
        let compute = scalar_compute_descriptor::<S>(&ctx.vtable)?;

        match desc {
            SemiringCoreDescriptor::BatchedGemm {
                batch_dims,
                m: _,
                n: _,
                k: _,
            } => {
                validate_shape_count(shapes, 3, "BatchedGemm")?;
                let nb = batch_dims.len() as u32;
                let mut modes_a = Vec::new();
                let mut modes_b = Vec::new();
                let mut modes_c = Vec::new();
                // batch modes: 0..nb
                for i in 0..nb {
                    modes_a.push(i as i32);
                    modes_b.push(i as i32);
                    modes_c.push(i as i32);
                }
                // m mode = nb, k mode = nb+1, n mode = nb+2
                let m_mode = nb as i32;
                let k_mode = (nb + 1) as i32;
                let n_mode = (nb + 2) as i32;
                modes_a.extend([m_mode, k_mode]);
                modes_b.extend([k_mode, n_mode]);
                modes_c.extend([m_mode, n_mode]);
                let strides_a = default_col_major_strides(shapes[0]);
                let strides_b = default_col_major_strides(shapes[1]);
                let strides_c = default_col_major_strides(shapes[2]);
                let (plan, ws) = plan_contraction(
                    ctx, data_type, compute, &modes_a, shapes[0], &strides_a, &modes_b, shapes[1],
                    &strides_b, &modes_c, shapes[2], &strides_c,
                )?;
                Ok(CudaPlan {
                    plan,
                    desc: CudaPlanDescriptor::Core(desc.clone()),
                    workspace_size: ws,
                    _marker: PhantomData,
                })
            }

            SemiringCoreDescriptor::Trace { .. }
            | SemiringCoreDescriptor::AntiTrace { .. }
            | SemiringCoreDescriptor::AntiDiag { .. } => {
                // These operations contract the input with an identity tensor.
                // For Trace: C = sum_over_diag(A) via contraction with eye tensor
                // For AntiTrace/AntiDiag: reverse operation
                // The identity tensor must be created at execute time on GPU.
                // For now, return an error — the einsum layer falls back to
                // the non-extension (core) path which decomposes these into
                // simpler operations.
                Err(Error::DeviceError(
                    "Trace/AntiTrace/AntiDiag not yet supported on CUDA backend".into(),
                ))
            }

            SemiringCoreDescriptor::ReduceAdd { modes_a, modes_c } => {
                validate_shape_count(shapes, 2, "ReduceAdd")?;
                let modes_a_i32: Vec<i32> = modes_a.iter().map(|&m| m as i32).collect();
                let modes_c_i32: Vec<i32> = modes_c.iter().map(|&m| m as i32).collect();
                let strides_a = default_col_major_strides(shapes[0]);
                let strides_c = default_col_major_strides(shapes[1]);
                let (plan, ws) = plan_reduction(
                    ctx,
                    data_type,
                    compute,
                    &modes_a_i32,
                    shapes[0],
                    &strides_a,
                    &modes_c_i32,
                    shapes[1],
                    &strides_c,
                    CUTENSOR_OP_ADD,
                )?;
                Ok(CudaPlan {
                    plan,
                    desc: CudaPlanDescriptor::Core(desc.clone()),
                    workspace_size: ws,
                    _marker: PhantomData,
                })
            }

            SemiringCoreDescriptor::MakeContiguous => {
                // Identity permutation: modes_a == modes_b = [0, 1, ..., ndim-1]
                validate_shape_count(shapes, 2, "MakeContiguous")?;
                let ndim = shapes[0].len();
                let modes: Vec<i32> = (0..ndim as i32).collect();
                let strides_a = default_col_major_strides(shapes[0]);
                let strides_b = default_col_major_strides(shapes[1]);
                let (plan, ws) = plan_permutation(
                    ctx, data_type, compute, &modes, shapes[0], &strides_a, &modes, shapes[1],
                    &strides_b,
                )?;
                Ok(CudaPlan {
                    plan,
                    desc: CudaPlanDescriptor::Core(desc.clone()),
                    workspace_size: ws,
                    _marker: PhantomData,
                })
            }
        }
    }

    fn execute(
        ctx: &mut CudaContext,
        plan: &CudaPlan<S>,
        alpha: S,
        inputs: &[&Tensor<S>],
        beta: S,
        output: &mut Tensor<S>,
    ) -> Result<()> {
        use std::ffi::c_void;

        cudarc::runtime::result::device::set(ctx.device_id as i32)
            .map_err(|e| Error::DeviceError(format!("CUDA runtime set-device failed: {e:?}")))?;
        let handle = ctx.handle.raw;
        // Use null stream (default CUDA stream)
        let stream: *mut c_void = ptr::null_mut();
        // Workspace: pass null with 0 size (cuTENSOR works without workspace, just slower)
        let ws_ptr: *mut c_void = ptr::null_mut();
        let ws_size: u64 = 0;

        let alpha_ptr = &alpha as *const S as *const c_void;
        let beta_ptr = &beta as *const S as *const c_void;

        match &plan.desc {
            CudaPlanDescriptor::Core(SemiringCoreDescriptor::BatchedGemm { .. })
            | CudaPlanDescriptor::Fast(SemiringFastPathDescriptor::Contract { .. }) => {
                validate_execute_inputs(inputs, 2, "Contraction")?;
                let a_ptr = inputs[0]
                    .buffer()
                    .as_device_ptr()
                    .ok_or_else(|| Error::DeviceError("input A not on GPU".into()))?
                    as *const c_void;
                let b_ptr = inputs[1]
                    .buffer()
                    .as_device_ptr()
                    .ok_or_else(|| Error::DeviceError("input B not on GPU".into()))?
                    as *const c_void;
                let c_ptr = output
                    .buffer()
                    .as_device_ptr()
                    .ok_or_else(|| Error::DeviceError("output not on GPU".into()))?
                    as *const c_void;
                let d_ptr = c_ptr as *mut c_void;

                let status = unsafe {
                    (ctx.vtable.contract)(
                        handle,
                        plan.plan.raw,
                        alpha_ptr,
                        a_ptr,
                        b_ptr,
                        beta_ptr,
                        c_ptr,
                        d_ptr,
                        ws_ptr,
                        ws_size,
                        stream,
                    )
                };
                check_status(status, "cutensorContract")
            }

            CudaPlanDescriptor::Core(SemiringCoreDescriptor::MakeContiguous) => {
                validate_execute_inputs(inputs, 1, "MakeContiguous")?;
                let a_ptr = inputs[0]
                    .buffer()
                    .as_device_ptr()
                    .ok_or_else(|| Error::DeviceError("input A not on GPU".into()))?
                    as *const c_void;
                let b_ptr = output
                    .buffer()
                    .as_device_ptr()
                    .ok_or_else(|| Error::DeviceError("output not on GPU".into()))?
                    as *const c_void as *mut c_void;

                let status = unsafe {
                    (ctx.vtable.permute)(handle, plan.plan.raw, alpha_ptr, a_ptr, b_ptr, stream)
                };
                check_status(status, "cutensorPermute")
            }

            CudaPlanDescriptor::Core(SemiringCoreDescriptor::ReduceAdd { .. }) => {
                validate_execute_inputs(inputs, 1, "ReduceAdd")?;
                let a_ptr = inputs[0]
                    .buffer()
                    .as_device_ptr()
                    .ok_or_else(|| Error::DeviceError("input A not on GPU".into()))?
                    as *const c_void;
                let c_ptr = output
                    .buffer()
                    .as_device_ptr()
                    .ok_or_else(|| Error::DeviceError("output not on GPU".into()))?
                    as *const c_void;
                let d_ptr = c_ptr as *mut c_void;

                let status = unsafe {
                    (ctx.vtable.reduce)(
                        handle,
                        plan.plan.raw,
                        alpha_ptr,
                        a_ptr,
                        beta_ptr,
                        c_ptr,
                        d_ptr,
                        ws_ptr,
                        ws_size,
                        stream,
                    )
                };
                check_status(status, "cutensorReduce")
            }

            CudaPlanDescriptor::Fast(SemiringFastPathDescriptor::ElementwiseBinary { .. }) => {
                validate_execute_inputs(inputs, 2, "ElementwiseBinary")?;
                let a_ptr = inputs[0]
                    .buffer()
                    .as_device_ptr()
                    .ok_or_else(|| Error::DeviceError("input A not on GPU".into()))?
                    as *const c_void;
                // C = inputs[1] (the second operand in element-wise binary)
                let c_ptr = inputs[1]
                    .buffer()
                    .as_device_ptr()
                    .ok_or_else(|| Error::DeviceError("input C not on GPU".into()))?
                    as *const c_void;
                let d_ptr = output
                    .buffer()
                    .as_device_ptr()
                    .ok_or_else(|| Error::DeviceError("output not on GPU".into()))?
                    as *const c_void as *mut c_void;

                // gamma = beta for the C input scaling
                let gamma_ptr = beta_ptr;

                let status = unsafe {
                    (ctx.vtable.elementwise_binary_execute)(
                        handle,
                        plan.plan.raw,
                        alpha_ptr,
                        a_ptr,
                        gamma_ptr,
                        c_ptr,
                        d_ptr,
                        stream,
                    )
                };
                check_status(status, "cutensorElementwiseBinaryExecute")
            }
        }
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
        cudarc::runtime::result::device::set(ctx.device_id as i32)
            .map_err(|e| Error::DeviceError(format!("CUDA runtime set-device failed: {e:?}")))?;
        let data_type = scalar_data_type::<S>()?;
        let compute = scalar_compute_descriptor::<S>(&ctx.vtable)?;

        match desc {
            SemiringFastPathDescriptor::Contract {
                modes_a,
                modes_b,
                modes_c,
            } => {
                validate_shape_count(shapes, 3, "Contract")?;
                let modes_a_i32: Vec<i32> = modes_a.iter().map(|&m| m as i32).collect();
                let modes_b_i32: Vec<i32> = modes_b.iter().map(|&m| m as i32).collect();
                let modes_c_i32: Vec<i32> = modes_c.iter().map(|&m| m as i32).collect();
                let strides_a = default_col_major_strides(shapes[0]);
                let strides_b = default_col_major_strides(shapes[1]);
                let strides_c = default_col_major_strides(shapes[2]);
                let (plan, ws) = plan_contraction(
                    ctx,
                    data_type,
                    compute,
                    &modes_a_i32,
                    shapes[0],
                    &strides_a,
                    &modes_b_i32,
                    shapes[1],
                    &strides_b,
                    &modes_c_i32,
                    shapes[2],
                    &strides_c,
                )?;
                Ok(CudaPlan {
                    plan,
                    desc: CudaPlanDescriptor::Fast(desc.clone()),
                    workspace_size: ws,
                    _marker: PhantomData,
                })
            }
            SemiringFastPathDescriptor::ElementwiseBinary { op } => {
                validate_shape_count(shapes, 3, "ElementwiseBinary")?;
                let ndim = shapes[0].len();
                let modes: Vec<i32> = (0..ndim as i32).collect();
                let strides = default_col_major_strides(shapes[0]);
                let cutensor_op = match op {
                    SemiringBinaryOp::Add => CUTENSOR_OP_ADD,
                    SemiringBinaryOp::Mul => CUTENSOR_OP_MUL,
                };
                let (plan, ws) = plan_elementwise_binary(
                    ctx,
                    data_type,
                    compute,
                    &modes,
                    shapes[0],
                    &strides,
                    cutensor_op,
                )?;
                Ok(CudaPlan {
                    plan,
                    desc: CudaPlanDescriptor::Fast(desc.clone()),
                    workspace_size: ws,
                    _marker: PhantomData,
                })
            }
        }
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
        matches!(
            desc,
            SemiringFastPathDescriptor::Contract { .. }
                | SemiringFastPathDescriptor::ElementwiseBinary {
                    op: SemiringBinaryOp::Add | SemiringBinaryOp::Mul,
                }
        )
    }
}

#[cfg(test)]
mod tests;
