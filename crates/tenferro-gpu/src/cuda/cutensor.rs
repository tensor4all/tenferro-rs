//! cuTENSOR contraction on the runtime's own stream, with shape-keyed plans.
//!
//! The handle is created once per runtime. A contraction's plan and workspace depend only
//! on its descriptor signature — dtype, modes, extents and strides — so they are cached by
//! exactly that signature: the first call with a signature pays descriptor creation, the
//! workspace estimate and the plan build (a *cold* plan), and every later call with the
//! same signature reuses both (a *warm* hit). A warm plan runs no cuTENSOR planning work at
//! all, which is what U3's "zero compile, zero planning on the warm path" invariant needs
//! from the vendor side.
//!
//! Executes inside the runtime's raw scope on the runtime's stream, so the four operands
//! and the workspace are registered in that scope and the library call is ordered with the
//! kernels that produced or will consume them.

use std::collections::HashMap;
use std::ffi::c_void;

use cudarc::cutensor::sys as ct;
use t4a_cuda_kernels::DType;
use t4a_cuda_runtime::{Buffer as DeviceBuffer, RawResource};

use crate::gpu::GpuError;

use super::runtime::CudaRuntime;

/// The cuTENSOR entry points this adapter calls.
///
/// Loaded from the library file [`load_library`] found, which keeps the search order under
/// this crate's control (`TENFERRO_CUTENSOR_PATH`, then the distribution's subdirectory)
/// instead of a fixed list of sonames.
struct Vtable {
    create: unsafe extern "C" fn(*mut ct::cutensorHandle_t) -> ct::cutensorStatus_t,
    destroy: unsafe extern "C" fn(ct::cutensorHandle_t) -> ct::cutensorStatus_t,
    create_tensor_descriptor: unsafe extern "C" fn(
        ct::cutensorHandle_t,
        *mut ct::cutensorTensorDescriptor_t,
        u32,
        *const i64,
        *const i64,
        ct::cudaDataType_t,
        u32,
    ) -> ct::cutensorStatus_t,
    destroy_tensor_descriptor:
        unsafe extern "C" fn(ct::cutensorTensorDescriptor_t) -> ct::cutensorStatus_t,
    create_contraction: unsafe extern "C" fn(
        ct::cutensorHandle_t,
        *mut ct::cutensorOperationDescriptor_t,
        ct::cutensorTensorDescriptor_t,
        *const i32,
        ct::cutensorOperator_t,
        ct::cutensorTensorDescriptor_t,
        *const i32,
        ct::cutensorOperator_t,
        ct::cutensorTensorDescriptor_t,
        *const i32,
        ct::cutensorOperator_t,
        ct::cutensorTensorDescriptor_t,
        *const i32,
        ct::cutensorComputeDescriptor_t,
    ) -> ct::cutensorStatus_t,
    destroy_operation_descriptor:
        unsafe extern "C" fn(ct::cutensorOperationDescriptor_t) -> ct::cutensorStatus_t,
    create_plan_preference: unsafe extern "C" fn(
        ct::cutensorHandle_t,
        *mut ct::cutensorPlanPreference_t,
        ct::cutensorAlgo_t,
        ct::cutensorJitMode_t,
    ) -> ct::cutensorStatus_t,
    destroy_plan_preference:
        unsafe extern "C" fn(ct::cutensorPlanPreference_t) -> ct::cutensorStatus_t,
    estimate_workspace_size: unsafe extern "C" fn(
        ct::cutensorHandle_t,
        ct::cutensorOperationDescriptor_t,
        ct::cutensorPlanPreference_t,
        ct::cutensorWorksizePreference_t,
        *mut u64,
    ) -> ct::cutensorStatus_t,
    create_plan: unsafe extern "C" fn(
        ct::cutensorHandle_t,
        *mut ct::cutensorPlan_t,
        ct::cutensorOperationDescriptor_t,
        ct::cutensorPlanPreference_t,
        u64,
    ) -> ct::cutensorStatus_t,
    destroy_plan: unsafe extern "C" fn(ct::cutensorPlan_t) -> ct::cutensorStatus_t,
    contract: unsafe extern "C" fn(
        ct::cutensorHandle_t,
        ct::cutensorPlan_t,
        *const c_void,
        *const c_void,
        *const c_void,
        *const c_void,
        *const c_void,
        *mut c_void,
        *mut c_void,
        u64,
        ct::cudaStream_t,
    ) -> ct::cutensorStatus_t,
}

impl Vtable {
    fn load(library: &libloading::Library) -> Result<Vtable, GpuError> {
        // SAFETY: each symbol is bound with its declared cuTENSOR signature and stays
        // valid for as long as the library does (which is process-lifetime here).
        unsafe {
            let symbol = |name: &[u8]| -> Result<*mut c_void, GpuError> {
                library
                    .get::<*mut c_void>(name)
                    .map(|value| *value)
                    .map_err(|error| GpuError::Unavailable {
                        detail: format!("{}: {error}", String::from_utf8_lossy(name)),
                    })
            };
            macro_rules! bind {
                ($name:literal, $ty:ty) => {{
                    let raw = symbol(concat!($name, "\0").as_bytes())?;
                    std::mem::transmute::<*mut c_void, $ty>(raw)
                }};
            }
            Ok(Vtable {
                create: bind!(
                    "cutensorCreate",
                    unsafe extern "C" fn(*mut ct::cutensorHandle_t) -> ct::cutensorStatus_t
                ),
                destroy: bind!(
                    "cutensorDestroy",
                    unsafe extern "C" fn(ct::cutensorHandle_t) -> ct::cutensorStatus_t
                ),
                create_tensor_descriptor: bind!(
                    "cutensorCreateTensorDescriptor",
                    unsafe extern "C" fn(
                        ct::cutensorHandle_t,
                        *mut ct::cutensorTensorDescriptor_t,
                        u32,
                        *const i64,
                        *const i64,
                        ct::cudaDataType_t,
                        u32,
                    ) -> ct::cutensorStatus_t
                ),
                destroy_tensor_descriptor: bind!(
                    "cutensorDestroyTensorDescriptor",
                    unsafe extern "C" fn(ct::cutensorTensorDescriptor_t) -> ct::cutensorStatus_t
                ),
                create_contraction: bind!(
                    "cutensorCreateContraction",
                    unsafe extern "C" fn(
                        ct::cutensorHandle_t,
                        *mut ct::cutensorOperationDescriptor_t,
                        ct::cutensorTensorDescriptor_t,
                        *const i32,
                        ct::cutensorOperator_t,
                        ct::cutensorTensorDescriptor_t,
                        *const i32,
                        ct::cutensorOperator_t,
                        ct::cutensorTensorDescriptor_t,
                        *const i32,
                        ct::cutensorOperator_t,
                        ct::cutensorTensorDescriptor_t,
                        *const i32,
                        ct::cutensorComputeDescriptor_t,
                    ) -> ct::cutensorStatus_t
                ),
                destroy_operation_descriptor: bind!(
                    "cutensorDestroyOperationDescriptor",
                    unsafe extern "C" fn(ct::cutensorOperationDescriptor_t) -> ct::cutensorStatus_t
                ),
                create_plan_preference: bind!(
                    "cutensorCreatePlanPreference",
                    unsafe extern "C" fn(
                        ct::cutensorHandle_t,
                        *mut ct::cutensorPlanPreference_t,
                        ct::cutensorAlgo_t,
                        ct::cutensorJitMode_t,
                    ) -> ct::cutensorStatus_t
                ),
                destroy_plan_preference: bind!(
                    "cutensorDestroyPlanPreference",
                    unsafe extern "C" fn(ct::cutensorPlanPreference_t) -> ct::cutensorStatus_t
                ),
                estimate_workspace_size: bind!(
                    "cutensorEstimateWorkspaceSize",
                    unsafe extern "C" fn(
                        ct::cutensorHandle_t,
                        ct::cutensorOperationDescriptor_t,
                        ct::cutensorPlanPreference_t,
                        ct::cutensorWorksizePreference_t,
                        *mut u64,
                    ) -> ct::cutensorStatus_t
                ),
                create_plan: bind!(
                    "cutensorCreatePlan",
                    unsafe extern "C" fn(
                        ct::cutensorHandle_t,
                        *mut ct::cutensorPlan_t,
                        ct::cutensorOperationDescriptor_t,
                        ct::cutensorPlanPreference_t,
                        u64,
                    ) -> ct::cutensorStatus_t
                ),
                destroy_plan: bind!(
                    "cutensorDestroyPlan",
                    unsafe extern "C" fn(ct::cutensorPlan_t) -> ct::cutensorStatus_t
                ),
                contract: bind!(
                    "cutensorContract",
                    unsafe extern "C" fn(
                        ct::cutensorHandle_t,
                        ct::cutensorPlan_t,
                        *const c_void,
                        *const c_void,
                        *const c_void,
                        *const c_void,
                        *const c_void,
                        *mut c_void,
                        *mut c_void,
                        u64,
                        ct::cudaStream_t,
                    ) -> ct::cutensorStatus_t
                ),
            })
        }
    }
}

/// A one-line adapter so a vendor call site reads `(call)(..).pipe(check_vendor(name))`.
trait Pipe: Sized {
    fn pipe<R>(self, then: impl FnOnce(Self) -> R) -> R {
        then(self)
    }
}
impl<T> Pipe for T {}

fn check_vendor(call: &'static str) -> impl FnOnce(ct::cutensorStatus_t) -> Result<(), GpuError> {
    move |status| check(status, "contract", call)
}

/// Refuse a non-success cuTENSOR status with the call that produced it.
fn check(
    status: ct::cutensorStatus_t,
    op: &'static str,
    call: &'static str,
) -> Result<(), GpuError> {
    if status == ct::cutensorStatus_t::CUTENSOR_STATUS_SUCCESS {
        Ok(())
    } else {
        Err(GpuError::Unsupported {
            op,
            detail: format!("{call} returned {status:?}"),
        })
    }
}

/// Library file names tried in order; `TENFERRO_CUTENSOR_PATH` (`:` separated) overrides.
const DEFAULT_PATHS: &[&str] = &[
    "/usr/lib/x86_64-linux-gnu/libcutensor/12/libcutensor.so.2",
    "libcutensor.so.2",
    "libcutensor.so",
];

/// One contraction operand: its modes, extents and element strides.
#[derive(Debug, Clone, Copy)]
#[doc(hidden)]
pub struct Operand<'a> {
    /// The device allocation.
    pub buffer: &'a DeviceBuffer,
    /// Mode label per dimension, as cuTENSOR indexes them.
    pub modes: &'a [i32],
    /// Extent per dimension.
    pub extents: &'a [i64],
    /// Element stride per dimension.
    pub strides: &'a [i64],
}

/// A contraction `d = alpha * a * b + beta * c` in the caller's modes.
#[derive(Debug, Clone, Copy)]
#[doc(hidden)]
pub struct Contraction<'a> {
    /// Dtype of all four operands.
    pub dtype: DType,
    /// Left operand.
    pub a: Operand<'a>,
    /// Right operand.
    pub b: Operand<'a>,
    /// Addend operand; `None` means `beta` is ignored and `d` is overwritten.
    pub c: Option<Operand<'a>>,
    /// Destination.
    pub d: Operand<'a>,
    /// Scale of the product, as `(re, im)`; the library reads it at the compute type's width.
    pub alpha: (f64, f64),
    /// Scale of the addend, as `(re, im)`; non-zero requires `c`.
    pub beta: (f64, f64),
    /// Whether the left operand is conjugated.
    pub conj_a: bool,
    /// Whether the right operand is conjugated.
    pub conj_b: bool,
}

impl Contraction<'_> {
    /// The coefficient pair the library reads, at the width the compute type expects.
    ///
    /// For a complex dtype the library wants a two-field complex scalar, and for a real one
    /// a single scalar of that width — passing a wider value makes it read the wrong bytes.
    fn coefficient(dtype: DType, value: (f64, f64)) -> Result<Coefficient, GpuError> {
        let (re, im) = value;
        Ok(match dtype {
            DType::F32 => Coefficient::F32([re as f32]),
            DType::F64 => Coefficient::F64([re]),
            DType::C32 => Coefficient::C32([re as f32, im as f32]),
            DType::C64 => Coefficient::C64([re, im]),
            other => {
                return Err(GpuError::unsupported(
                    "contract",
                    format!("cuTENSOR serves F32, F64, C32 and C64 here, not {other:?}"),
                ))
            }
        })
    }
}

/// A cuTENSOR scalar operand, laid out as the compute type expects.
#[derive(Clone, Copy)]
enum Coefficient {
    /// One `float`.
    F32([f32; 1]),
    /// One `double`.
    F64([f64; 1]),
    /// `cuFloatComplex`.
    C32([f32; 2]),
    /// `cuDoubleComplex`.
    C64([f64; 2]),
}

impl Coefficient {
    fn as_ptr(&self) -> *const std::ffi::c_void {
        match self {
            Coefficient::F32(value) => value.as_ptr().cast(),
            Coefficient::F64(value) => value.as_ptr().cast(),
            Coefficient::C32(value) => value.as_ptr().cast(),
            Coefficient::C64(value) => value.as_ptr().cast(),
        }
    }
}

/// How many contraction plans (and their device workspaces) one handle retains.
const MAX_CACHED_PLANS: usize = 32;

/// A cached plan and the workspace it was planned with.
struct CachedPlan {
    descriptors: [ct::cutensorTensorDescriptor_t; 4],
    plan: ct::cutensorPlan_t,
    preference: ct::cutensorPlanPreference_t,
    operation: ct::cutensorOperationDescriptor_t,
    /// Bytes this plan needs; the handle's single workspace is at least this large.
    workspace_bytes: u64,
}

/// A cuTENSOR handle bound to one runtime, with its shape-keyed plan cache.
#[doc(hidden)]
pub struct Cutensor {
    handle: ct::cutensorHandle_t,
    runtime: CudaRuntime,
    plans: HashMap<Vec<i64>, CachedPlan>,
    /// Insertion order of the cached plans, so the oldest plan retires when the cache is full.
    order: std::collections::VecDeque<Vec<i64>>,
    /// One workspace for every plan of this handle, grown to the largest requirement seen.
    /// A workspace per plan would keep a device allocation alive for each shape.
    workspace: Option<DeviceBuffer>,
    workspace_bytes: u64,
    cold_plans: u64,
    warm_hits: u64,
    library: &'static libloading::Library,
    vtable: Vtable,
}

// SAFETY: a handle carries no submission authority; every call runs inside the owning
// runtime's raw scope on its stream, and the plan cache is used through `&mut self`.
unsafe impl Send for Cutensor {}

impl Cutensor {
    /// Create the handle; the first contraction pays for the library load.
    ///
    /// # Errors
    ///
    /// [`GpuError::Unavailable`] when the library or a compute descriptor cannot be
    /// loaded, and the runtime's typed admission failure.
    pub fn new(runtime: &CudaRuntime) -> Result<Cutensor, GpuError> {
        let library = load_library()?;
        let vtable = Vtable::load(library)?;
        let handle = {
            let mut raw = std::ptr::null_mut();
            // SAFETY: `cutensorCreate` takes no tensor arguments and returns an opaque
            // handle owned by the caller; the scope activates the context it belongs to.
            let status = unsafe {
                runtime
                    .with_raw_locked(Vec::new(), |_raw| (vtable.create)(&mut raw))
                    .map_err(|error| GpuError::Unavailable {
                        detail: error.to_string(),
                    })?
            };
            check(status, "contract", "cutensorCreate")?;
            raw
        };
        Ok(Cutensor {
            handle,
            runtime: runtime.clone(),
            plans: HashMap::new(),
            order: std::collections::VecDeque::new(),
            workspace: None,
            workspace_bytes: 0,
            cold_plans: 0,
            warm_hits: 0,
            library,
            vtable,
        })
    }

    /// Plans built so far (the cold count).
    pub fn cold_plans(&self) -> u64 {
        self.cold_plans
    }

    /// Calls served from a cached plan (the warm count).
    pub fn warm_hits(&self) -> u64 {
        self.warm_hits
    }

    /// Number of cached plans.
    pub fn cached_plans(&self) -> usize {
        self.plans.len()
    }

    /// Execute one contraction, planning it only if its signature is new.
    ///
    /// # Errors
    ///
    /// # Panics
    ///
    /// Only if the plan cache and the workspace it just stored disagree, which is an internal
    /// invariant: the plan is inserted before it is read back.
    ///
    /// # Errors
    ///
    /// [`GpuError::Unsupported`] for operands whose mode, extent and stride lists disagree
    /// or whose signature is already cached with a different workspace, and the library's
    /// typed failure.
    pub fn contract(&mut self, request: &Contraction<'_>) -> Result<(), GpuError> {
        for operand in [request.a, request.b, request.d]
            .into_iter()
            .chain(request.c)
        {
            if operand.modes.len() != operand.extents.len()
                || operand.modes.len() != operand.strides.len()
            {
                return Err(GpuError::unsupported(
                    "contract",
                    "an operand's mode, extent and stride lists disagree",
                ));
            }
        }
        let signature = signature_of(request);
        if self.plans.contains_key(&signature) {
            self.warm_hits += 1;
        } else {
            let plan = self.build_plan(request)?;
            // A varying shape would otherwise retain one device workspace per shape for the
            // lifetime of the engine; the oldest plan retires instead.
            if self.order.len() >= MAX_CACHED_PLANS {
                if let Some(oldest) = self.order.pop_front() {
                    if let Some(evicted) = self.plans.remove(&oldest) {
                        // The plan's handles and workspace are native resources: an evicted plan
                        // has to be destroyed, or a varying-shape workload leaks one bundle per
                        // shape. Nothing queued may still use it, so the stream is drained first.
                        self.runtime
                            .synchronize()
                            .map_err(|error| GpuError::Unavailable {
                                detail: error.to_string(),
                            })?;
                        self.destroy_plan(evicted);
                    }
                }
            }
            self.plans.insert(signature.clone(), plan);
            self.order.push_back(signature.clone());
            self.cold_plans += 1;
        }
        let cached = self.plans.get(&signature).expect("plan inserted above");
        let (a, b, c, d) = (
            request.a.buffer,
            request.b.buffer,
            request.c.map(|operand| operand.buffer),
            request.d.buffer,
        );
        let workspace = self.workspace.as_ref().expect("planned with a workspace");
        let plan = cached.plan;
        let workspace_bytes = cached.workspace_bytes;
        let handle = self.handle;
        let mut resources = vec![
            RawResource::Buffer(a),
            RawResource::Buffer(b),
            RawResource::Buffer(d),
            RawResource::Buffer(workspace),
        ];
        if let Some(c) = c {
            resources.push(RawResource::Buffer(c));
        }
        if request.c.is_none() && request.beta != (0.0, 0.0) {
            return Err(GpuError::unsupported(
                "contract",
                "a non-zero beta needs an addend operand",
            ));
        }
        let alpha = Contraction::coefficient(request.dtype, request.alpha)?;
        let beta = Contraction::coefficient(request.dtype, request.beta)?;
        let vtable = &self.vtable;
        // SAFETY: the raw scope registered every operand and the workspace in this
        // runtime, so the addresses belong to it and stay alive until the scope's
        // completion; the coefficients are live stack values in host pointer mode.
        unsafe {
            self.runtime
                .with_raw_locked(resources, |raw| {
                    let address = |buffer: &DeviceBuffer| -> Result<*mut c_void, GpuError> {
                        raw.device_ptr(buffer)
                            .map(|value| value as usize as *mut c_void)
                            .map_err(|error| GpuError::Unavailable {
                                detail: error.to_string(),
                            })
                    };
                    let (a, b, d) = (address(a)?, address(b)?, address(d)?);
                    // Without an addend the plan's C descriptor is the destination, so the
                    // execution passes the destination pointer too; `beta` is zero, which
                    // cuTENSOR guarantees never reads C.
                    let c_ptr = match c {
                        Some(c) => address(c)?,
                        None => d,
                    };
                    let workspace_ptr = address(workspace)?;
                    let stream = raw.stream() as usize as ct::cudaStream_t;
                    (vtable.contract)(
                        handle,
                        plan,
                        alpha.as_ptr(),
                        a,
                        b,
                        beta.as_ptr(),
                        c_ptr,
                        d,
                        workspace_ptr,
                        workspace_bytes,
                        stream,
                    )
                    .pipe(check_vendor("cutensorContract"))
                })
                .map_err(|error| GpuError::Unavailable {
                    detail: error.to_string(),
                })?
        }
    }

    fn build_plan(&mut self, request: &Contraction<'_>) -> Result<CachedPlan, GpuError> {
        let handle = self.handle;
        let dtype = request.dtype;
        let data_type = data_type_of(dtype)?;
        let compute = self.compute_descriptor(dtype)?;
        let vtable = &self.vtable;
        let mut operands: [Option<Operand<'_>>; 4] =
            [Some(request.a), Some(request.b), request.c, Some(request.d)];
        let mut descriptors = [std::ptr::null_mut(); 4];
        let (a_buffer, b_buffer, d_buffer) = (request.a.buffer, request.b.buffer, request.d.buffer);
        let mut resources = vec![
            RawResource::Buffer(a_buffer),
            RawResource::Buffer(b_buffer),
            RawResource::Buffer(d_buffer),
        ];
        if let Some(c) = request.c {
            resources.push(RawResource::Buffer(c.buffer));
        }
        let workspace_bytes_and_plan = {
            // SAFETY: descriptors and the operation descriptor are created from the
            // caller's signature and destroyed with the cached plan; the raw scope
            // activates the context the handle belongs to, and the workspace allocation
            // happens under the same guard.
            unsafe {
                self.runtime
                    .with_raw_locked(resources, |_raw| {
                        for (index, operand) in operands.iter_mut().enumerate() {
                            let Some(operand) = operand else { continue };
                            check(
                                (vtable.create_tensor_descriptor)(
                                    handle,
                                    &mut descriptors[index],
                                    operand.modes.len() as u32,
                                    operand.extents.as_ptr(),
                                    operand.strides.as_ptr(),
                                    data_type,
                                    // 16-byte vectorisation: every tenferro allocation is at
                                    // least 256-byte aligned (t4a-cuda-runtime's allocator).
                                    16,
                                ),
                                "contract",
                                "cutensorCreateTensorDescriptor",
                            )?;
                        }
                        let [desc_a, desc_b, desc_c, desc_d] = descriptors;
                        let mode = |operand: &Operand<'_>| operand.modes.as_ptr();
                        // A contraction without an addend passes the destination as C: both
                        // cuTENSOR and the BLAS family guarantee that a zero `beta` never reads
                        // C, and this version rejects a null C descriptor outright.
                        let (c_modes, c_desc) = match request.c {
                            Some(c) => (mode(&c), desc_c),
                            None => (mode(&request.d), desc_d),
                        };
                        let mut operation = std::ptr::null_mut();
                        check(
                            (vtable.create_contraction)(
                                handle,
                                &mut operation,
                                desc_a,
                                mode(&request.a),
                                if request.conj_a {
                                    ct::cutensorOperator_t::CUTENSOR_OP_CONJ
                                } else {
                                    ct::cutensorOperator_t::CUTENSOR_OP_IDENTITY
                                },
                                desc_b,
                                mode(&request.b),
                                if request.conj_b {
                                    ct::cutensorOperator_t::CUTENSOR_OP_CONJ
                                } else {
                                    ct::cutensorOperator_t::CUTENSOR_OP_IDENTITY
                                },
                                c_desc,
                                c_modes,
                                ct::cutensorOperator_t::CUTENSOR_OP_IDENTITY,
                                desc_d,
                                mode(&request.d),
                                compute,
                            ),
                            "contract",
                            "cutensorCreateContraction",
                        )?;
                        let mut preference = std::ptr::null_mut();
                        check(
                            (vtable.create_plan_preference)(
                                handle,
                                &mut preference,
                                ct::cutensorAlgo_t::CUTENSOR_ALGO_DEFAULT,
                                ct::cutensorJitMode_t::CUTENSOR_JIT_MODE_NONE,
                            ),
                            "contract",
                            "cutensorCreatePlanPreference",
                        )?;
                        let mut workspace_bytes = 0u64;
                        check(
                            (vtable.estimate_workspace_size)(
                                handle,
                                operation,
                                preference,
                                ct::cutensorWorksizePreference_t::CUTENSOR_WORKSPACE_DEFAULT,
                                &mut workspace_bytes,
                            ),
                            "contract",
                            "cutensorEstimateWorkspaceSize",
                        )?;
                        Ok((workspace_bytes, operation, preference))
                    })
                    .map_err(|error| GpuError::Unavailable {
                        detail: error.to_string(),
                    })?
            }
        }?;
        let (workspace_bytes, operation, preference) = workspace_bytes_and_plan;
        // The handle owns one workspace, grown on demand: an allocation per plan would keep a
        // device buffer alive for every shape ever contracted.
        if self.workspace.is_none() || workspace_bytes > self.workspace_bytes {
            if self.workspace.is_some() {
                // Replacing the workspace drops the buffer earlier plans ran against, so wait
                // for the stream first. This happens only while the requirement grows.
                self.runtime
                    .synchronize()
                    .map_err(|error| GpuError::Unavailable {
                        detail: error.to_string(),
                    })?;
            }
            let workspace = if workspace_bytes == 0 {
                self.runtime
                    .upload_bytes(&[])
                    .map_err(|error| GpuError::Unavailable {
                        detail: error.to_string(),
                    })?
            } else {
                let mut guard = self.runtime.submit();
                guard
                    .alloc_zeroed(workspace_bytes as usize, 256)
                    .map_err(|error| GpuError::Unavailable {
                        detail: error.to_string(),
                    })?
            };
            self.workspace = Some(workspace);
            self.workspace_bytes = workspace_bytes;
        }
        // The workspace was grown above whenever this plan needs more than it holds, so it is
        // present by the time the plan is created.
        let workspace = self
            .workspace
            .as_ref()
            .expect("the workspace was grown for this plan");
        let handle = self.handle;
        let plan = {
            let mut plan = std::ptr::null_mut();
            let vtable = &self.vtable;
            // SAFETY: the handle and the operation descriptor are live, and the workspace
            // size was estimated for exactly this descriptor and preference.
            let status = unsafe {
                self.runtime
                    .with_raw_locked(vec![RawResource::Buffer(workspace)], |_raw| {
                        (vtable.create_plan)(
                            handle,
                            &mut plan,
                            operation,
                            preference,
                            workspace_bytes,
                        )
                    })
                    .map_err(|error| GpuError::Unavailable {
                        detail: error.to_string(),
                    })?
            };
            check(status, "contract", "cutensorCreatePlan")?;
            plan
        };
        Ok(CachedPlan {
            descriptors,
            plan,
            preference,
            operation,
            workspace_bytes,
        })
    }

    fn compute_descriptor(
        &self,
        dtype: DType,
    ) -> Result<ct::cutensorComputeDescriptor_t, GpuError> {
        let symbol: &[u8] = match dtype {
            DType::F32 | DType::C32 => b"CUTENSOR_COMPUTE_DESC_32F\0",
            DType::F64 | DType::C64 => b"CUTENSOR_COMPUTE_DESC_64F\0",
            other => {
                let _ = other;
                return Err(GpuError::unsupported(
                    "contract",
                    "cuTENSOR serves F32, F64, C32 and C64 here",
                ));
            }
        };
        // SAFETY: `CUTENSOR_COMPUTE_DESC_*` are exported as process-lifetime data whose
        // pointee is the descriptor value cuTENSOR expects; the library outlives the
        // handle, so the read copies a value that stays valid.
        unsafe {
            let holder = self
                .library
                .get::<*const ct::cutensorComputeDescriptor_t>(symbol)
                .map_err(|error| GpuError::Unavailable {
                    detail: format!("{}: {error}", String::from_utf8_lossy(symbol)),
                })?;
            let pointee = *holder;
            if pointee.is_null() {
                return Err(GpuError::Unavailable {
                    detail: format!(
                        "{} resolved to a null descriptor",
                        String::from_utf8_lossy(symbol)
                    ),
                });
            }
            Ok(std::ptr::read(pointee))
        }
    }
}

impl Cutensor {
    /// Destroy one plan's native resources, including the descriptors it owns.
    fn destroy_plan(&self, cached: CachedPlan) {
        // SAFETY: the raw scope activates the context the plan belongs to, and the caller proved
        // nothing still reads it.
        unsafe {
            let _ = self.runtime.with_raw_locked(Vec::new(), |_raw| {
                let _ = (self.vtable.destroy_plan)(cached.plan);
                let _ = (self.vtable.destroy_plan_preference)(cached.preference);
                let _ = (self.vtable.destroy_operation_descriptor)(cached.operation);
                for descriptor in cached.descriptors {
                    if !descriptor.is_null() {
                        let _ = (self.vtable.destroy_tensor_descriptor)(descriptor);
                    }
                }
            });
        }
    }
}

impl Drop for Cutensor {
    fn drop(&mut self) {
        let handle = self.handle;
        let plans: Vec<CachedPlan> = self.plans.drain().map(|(_, plan)| plan).collect();
        self.runtime.synchronize_best_effort();
        for cached in plans {
            self.destroy_plan(cached);
        }
        // SAFETY: the runtime was synchronised first, so nothing still reads the workspace or the
        // handle; the scope activates the context they belong to.
        unsafe {
            let _ = self.runtime.with_raw_locked(Vec::new(), |_raw| {
                drop(self.workspace.take());
                let _ = (self.vtable.destroy)(handle);
            });
        }
    }
}

fn data_type_of(dtype: DType) -> Result<ct::cudaDataType_t, GpuError> {
    Ok(match dtype {
        DType::F32 => ct::cudaDataType_t::CUDA_R_32F,
        DType::F64 => ct::cudaDataType_t::CUDA_R_64F,
        DType::C32 => ct::cudaDataType_t::CUDA_C_32F,
        DType::C64 => ct::cudaDataType_t::CUDA_C_64F,
        other => {
            let _ = other;
            return Err(GpuError::unsupported(
                "contract",
                "cuTENSOR serves F32, F64, C32 and C64 here",
            ));
        }
    })
}

/// The plan cache key: dtype, then every operand's modes, extents and strides.
fn signature_of(request: &Contraction<'_>) -> Vec<i64> {
    let mut key = Vec::with_capacity(64);
    key.push(match request.dtype {
        DType::F32 => 1,
        DType::F64 => 2,
        DType::C32 => 3,
        DType::C64 => 4,
        other => {
            let _ = other;
            0
        }
    });
    key.push(i64::from(request.conj_a));
    key.push(i64::from(request.conj_b));
    for operand in [Some(request.a), Some(request.b), request.c, Some(request.d)] {
        let Some(operand) = operand else {
            key.push(-1);
            continue;
        };
        key.push(operand.modes.len() as i64);
        key.extend(operand.modes.iter().map(|mode| i64::from(*mode)));
        key.extend_from_slice(operand.extents);
        key.extend_from_slice(operand.strides);
    }
    key
}

fn load_library() -> Result<&'static libloading::Library, GpuError> {
    static LIBRARY: std::sync::OnceLock<Result<libloading::Library, String>> =
        std::sync::OnceLock::new();
    let loaded = LIBRARY.get_or_init(|| {
        let paths: Vec<String> = match std::env::var("TENFERRO_CUTENSOR_PATH") {
            Ok(value) => value
                .split(':')
                .filter(|part| !part.is_empty())
                .map(String::from)
                .collect(),
            Err(_) => DEFAULT_PATHS.iter().map(|path| path.to_string()).collect(),
        };
        let mut failures = Vec::new();
        for path in paths {
            // SAFETY: loading a vendor library runs its initialisers; cuTENSOR's are
            // self-contained and the handle is created in the caller's context.
            match unsafe { libloading::Library::new(&path) } {
                Ok(library) => return Ok(library),
                Err(error) => failures.push(format!("{path}: {error}")),
            }
        }
        Err(failures.join("; "))
    });
    loaded.as_ref().map_err(|detail| GpuError::Unavailable {
        detail: detail.clone(),
    })
}
