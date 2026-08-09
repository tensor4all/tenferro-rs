//! cuFFT plan/workspace ownership and scoped session execution.
//!
//! Plan creation follows NVIDIA's documented manual-work-area sequence:
//! `cufftCreate`, `cufftSetAutoAllocation(plan, 0)`,
//! `cufftMakePlanMany64`, and a session-scoped workspace allocation combined
//! with `cufftSetWorkArea` inside the raw execution session. The plan entry
//! retains only the required workspace byte size; each execution allocates a
//! fresh CubeCL workspace on the public raw session so no session-independent
//! device allocation must survive across `ExtensionCacheStore` lifetimes.
//! Plan retirement synchronizes the retained CubeCL stream before destroying
//! the opaque cuFFT handle.

use std::collections::hash_map::DefaultHasher;
use std::ffi::c_void;
use std::hash::{Hash, Hasher};
use std::io::Write;
use std::mem::size_of;
use std::sync::Arc;

use tenferro_gpu::cuda::raw::DeviceBytes;
use tenferro_gpu::cuda::{CudaExecSession, CudaRuntime};
use tenferro_runtime::ExtensionCacheKey;
use tenferro_tensor::{Tensor, TensorScalar, TypedTensor};

use super::descriptor::{CufftDirection, CufftPlanDescriptor, CufftPlanKey, CufftTransformKind};
use super::error::CudaFftError;
use super::ffi::{
    map_cufft_status, CufftApi, CufftHandle, CufftLibrary, CufftStatus, CUFFT_C2C, CUFFT_C2R,
    CUFFT_D2Z, CUFFT_FORWARD, CUFFT_INVERSE, CUFFT_R2C, CUFFT_Z2D, CUFFT_Z2Z,
};
use crate::FFT_EXTENSION_FAMILY_ID;

const OP: &str = "cuda_fft";

/// Cache namespace for operation-family-owned cuFFT plans.
pub(crate) const CUFFT_CACHE_NAMESPACE: &str = "cufft-plans";

/// A cleanup context used by the plan-construction and retirement guards.
pub(crate) trait CufftCleanup {
    /// Run `f` with the retained runtime's context current, restoring the
    /// caller's previous device/context on every exit path.
    fn with_context<R2>(
        &self,
        op: &'static str,
        f: impl FnOnce() -> Result<R2, CudaFftError>,
    ) -> Result<R2, CudaFftError>;
    fn synchronize(&self) -> Result<(), CudaFftError>;
}

impl CufftCleanup for CudaRuntime {
    fn with_context<R2>(
        &self,
        op: &'static str,
        f: impl FnOnce() -> Result<R2, CudaFftError>,
    ) -> Result<R2, CudaFftError> {
        self.with_current_context(op, f)
            .map_err(|source| CudaFftError::interop("cufft_plan_cleanup_context", source))?
    }

    fn synchronize(&self) -> Result<(), CudaFftError> {
        CudaRuntime::synchronize(self)
            .map_err(|source| CudaFftError::interop("cufft_plan_cleanup_stream", source))
    }
}

#[derive(Default)]
pub(crate) struct CleanupFailures {
    pub(crate) synchronization: Vec<CudaFftError>,
    pub(crate) destroy: Option<CudaFftError>,
    /// True when cleanup could not prove retirement safety and the complete
    /// plan/lifetime-witness bundle was intentionally leaked.
    pub(crate) resources_deferred: bool,
}

impl CleanupFailures {
    #[cfg(test)]
    pub(crate) fn is_empty(&self) -> bool {
        self.synchronization.is_empty() && self.destroy.is_none() && !self.resources_deferred
    }
}

/// Retire one plan handle without panicking from `Drop` paths.
///
/// The plan is released only after the retained runtime is made current and
/// its stream synchronization succeeds. The whole retirement (synchronize +
/// destroy) runs under a context-restoring guard, so the caller's previous
/// device/context is preserved on every exit path. If any step fails, the
/// opaque plan, library witness, and runtime witness are intentionally
/// leaked: destroying or dropping any of them could race work whose
/// completion was not proven.
pub(crate) fn retire_handle<R>(
    runtime: &R,
    library: &Arc<CufftLibrary>,
    handle: &mut Option<CufftHandle>,
) -> CleanupFailures
where
    R: CufftCleanup + Clone,
{
    let mut failures = CleanupFailures::default();
    if handle.is_some() {
        let outcome = runtime.with_context("cufft_plan_cleanup", || {
            runtime.synchronize()?;
            if let Some(plan) = *handle {
                // SAFETY: `plan` was returned by the same retained cuFFT
                // library's successful `cufftCreate` call and has not been
                // destroyed before this one-shot retirement path.
                let status = unsafe { (library.api.destroy)(plan) };
                map_cufft_status("cufftDestroy", status)?;
                handle.take();
            }
            Ok(())
        });
        if let Err(error) = outcome {
            // A typed destroy failure keeps its CufftStatus; any context or
            // stream barrier failure is reported as a synchronization failure.
            match error {
                CudaFftError::CufftStatus { .. } => failures.destroy = Some(error),
                _ => failures.synchronization.push(error),
            }
            failures.resources_deferred = true;
            defer_resources(runtime, library, handle);
            return failures;
        }
    }
    failures
}

/// Owns every witness needed to keep an unretired cuFFT operation valid.
///
/// This value is deliberately forgotten only when cleanup cannot prove that
/// vendor work has completed. There is no safe retry or process-global
/// quarantine for a plan whose context/stream barrier failed.
struct DeferredCufftResources<R> {
    _handle: Option<CufftHandle>,
    _library: Arc<CufftLibrary>,
    _runtime: R,
}

fn defer_resources<R>(runtime: &R, library: &Arc<CufftLibrary>, handle: &mut Option<CufftHandle>)
where
    R: Clone,
{
    // The forgotten bundle retains the vendor plan, dynamic-library handle,
    // and CUDA context/runtime clone together until process exit. Dropping any
    // member independently could invalidate queued work after a failed cleanup
    // barrier.
    std::mem::forget(DeferredCufftResources {
        _handle: handle.take(),
        _library: Arc::clone(library),
        _runtime: runtime.clone(),
    });
}

/// Retire one owned plan, as used by `CufftPlanEntry::Drop`.
pub(crate) fn retire_entry_resources<R>(
    runtime: &R,
    library: &Arc<CufftLibrary>,
    plan: CufftHandle,
) -> CleanupFailures
where
    R: CufftCleanup + Clone,
{
    let mut handle = Some(plan);
    retire_handle(runtime, library, &mut handle)
}

#[cold]
pub(crate) fn report_cleanup_failures(failures: CleanupFailures) {
    let mut stderr = std::io::stderr();
    let resources_deferred = failures.resources_deferred;
    for error in failures.synchronization {
        let _ = writeln!(
            stderr,
            "tenferro-fft: cuFFT cleanup synchronization failed: {error}"
        );
    }
    if let Some(error) = failures.destroy {
        let _ = writeln!(
            stderr,
            "tenferro-fft: cuFFT plan destroy failed during cleanup: {error}"
        );
    }
    if resources_deferred {
        let _ = writeln!(
            stderr,
            "tenferro-fft: cuFFT plan and lifetime witnesses were intentionally retained after cleanup failure"
        );
    }
}

struct CufftConstructionGuard<R>
where
    R: CufftCleanup + Clone,
{
    library: Arc<CufftLibrary>,
    runtime: R,
    handle: Option<CufftHandle>,
}

impl<R> CufftConstructionGuard<R>
where
    R: CufftCleanup + Clone,
{
    fn new(library: Arc<CufftLibrary>, runtime: R) -> Self {
        Self {
            library,
            runtime,
            handle: None,
        }
    }

    fn disarm(mut self) -> Result<CufftHandle, CudaFftError> {
        self.handle
            .take()
            .ok_or_else(|| CudaFftError::internal("cuFFT construction completed without a handle"))
    }
}

impl<R> Drop for CufftConstructionGuard<R>
where
    R: CufftCleanup + Clone,
{
    fn drop(&mut self) {
        let failures = retire_handle(&self.runtime, &self.library, &mut self.handle);
        report_cleanup_failures(failures);
    }
}

/// Build a plan with the manual create/auto-allocation/make-plan sequence and
/// rollback guard, leaving work-area binding to each raw execution session.
pub(crate) fn build_plan<R>(
    library: Arc<CufftLibrary>,
    runtime: R,
    mut descriptor: CufftPlanDescriptor,
) -> Result<(CufftHandle, usize), CudaFftError>
where
    R: CufftCleanup + Clone,
{
    let mut guard = CufftConstructionGuard::<R>::new(Arc::clone(&library), runtime);
    let mut handle = 0;
    // SAFETY: `handle` is a valid output slot and `library` retains the exact
    // cuFFT shared object from which this function pointer was loaded.
    let status = unsafe { (library.api.create)(&mut handle) };
    map_cufft_status("cufftCreate", status)?;
    guard.handle = Some(handle);

    // SAFETY: `handle` was returned by the preceding successful create call;
    // the auto-allocation flag is the documented integer C ABI parameter.
    let status = unsafe { (library.api.set_auto_allocation)(handle, 0) };
    map_cufft_status("cufftSetAutoAllocation", status)?;

    let mut workspace_size = 0usize;
    // SAFETY: the descriptor owns rank-one arrays whose checked i64 values are
    // valid for `cufftMakePlanMany64`; the library handle remains alive.
    let status = unsafe {
        (library.api.make_plan_many_64)(
            handle,
            descriptor.rank,
            descriptor.n.as_mut_ptr(),
            descriptor.inembed.as_mut_ptr(),
            descriptor.istride,
            descriptor.idist,
            descriptor.onembed.as_mut_ptr(),
            descriptor.ostride,
            descriptor.odist,
            cufft_type(descriptor.kind),
            descriptor.batch,
            &mut workspace_size,
        )
    };
    map_cufft_status("cufftMakePlanMany64", status)?;

    let plan = guard.disarm()?;
    Ok((plan, workspace_size))
}

fn cufft_type(kind: CufftTransformKind) -> i32 {
    match kind {
        CufftTransformKind::C2c32 => CUFFT_C2C,
        CufftTransformKind::C2c64 => CUFFT_Z2Z,
        CufftTransformKind::R2c32 => CUFFT_R2C,
        CufftTransformKind::R2c64 => CUFFT_D2Z,
        CufftTransformKind::C2r32 => CUFFT_C2R,
        CufftTransformKind::C2r64 => CUFFT_Z2D,
    }
}

fn direction(direction: CufftDirection) -> i32 {
    match direction {
        CufftDirection::Forward => CUFFT_FORWARD,
        CufftDirection::Inverse => CUFFT_INVERSE,
    }
}

/// Bind a plan to the session's current stream for the next vendor execution.
pub(crate) fn bind_plan_to_stream(
    library: &CufftLibrary,
    plan: CufftHandle,
    stream: u64,
) -> Result<(), CudaFftError> {
    let stream = usize::try_from(stream)
        .map_err(|_| CudaFftError::InvalidConfiguration { field: "stream" })?
        as *mut c_void;
    // SAFETY: `plan` belongs to `library`, and `stream` is borrowed from the
    // retained runtime for the duration of the raw session callback.
    let status = unsafe { (library.api.set_stream)(plan, stream) };
    map_cufft_status("cufftSetStream", status)
}

/// Bind a plan to a fresh session-scoped work area for the next execution.
fn bind_workspace_to_plan(
    library: &CufftLibrary,
    plan: CufftHandle,
    workspace: &DeviceBytes<'_>,
) -> Result<(), CudaFftError> {
    let mut work_area_result = Ok(());
    workspace.with_ptr(|ptr| {
        // SAFETY: the plan was created and the workspace pointer is scoped to
        // the session that remains alive through the subsequent synchronized
        // vendor execution.
        let status = unsafe { (library.api.set_work_area)(plan, ptr) };
        work_area_result = map_cufft_status("cufftSetWorkArea", status);
    });
    work_area_result
}

/// Combine a vendor-call error and a synchronization error captured inside a
/// raw execution session. Synchronization is always the suppressed error when
/// both are present, matching the retired lease-witness semantics.
pub(crate) fn merge_execution_errors(
    execution_error: Option<CudaFftError>,
    synchronization_error: Option<CudaFftError>,
) -> Result<(), CudaFftError> {
    match (execution_error, synchronization_error) {
        (Some(primary), Some(suppressed)) => {
            Err(CudaFftError::with_suppressed(primary, suppressed))
        }
        (Some(error), None) | (None, Some(error)) => Err(error),
        (None, None) => Ok(()),
    }
}

/// One cached cuFFT plan and its required work-area byte size.
//
// INVARIANT: stream identity is intentionally omitted from `CufftPlanKey`;
// mutable `CudaExecSession` access serializes the runtime's one selectable
// current CubeCL stream. A selectable multi-stream runtime must make stream
// identity part of this key before concurrent plan reuse is allowed.
//
// INVARIANT: the entry retains the plan handle and the required workspace byte
// size only. The CubeCL workspace allocation is session-scoped and is created
// fresh inside each raw execution session, so no session-independent device
// allocation outlives the plan handle in this cache.
pub(crate) struct CufftPlanEntry {
    pub(crate) library: Arc<CufftLibrary>,
    pub(crate) plan: CufftHandle,
    pub(crate) workspace_bytes: usize,
    pub(crate) runtime: CudaRuntime,
    pub(crate) key: CufftPlanKey,
    retained_bytes: usize,
}

// SAFETY: the library and runtime retain their executable/context witnesses;
// mutable execution and retirement are entered through one borrowed session
// and the cache's mutable typed-entry access.
unsafe impl Send for CufftPlanEntry {}
// SAFETY: callers cannot execute through a shared reference, and the retained
// plan is only accessed through the owning mutable session boundary.
unsafe impl Sync for CufftPlanEntry {}

pub(crate) fn retained_entry_bytes() -> usize {
    // The cached entry owns only host-side metadata: the library and runtime
    // witnesses, the plan handle, the workspace-size requirement, and the key.
    // The CubeCL workspace allocation itself is session-scoped and created
    // fresh inside each raw execution session, so it is not charged here.
    size_of::<Arc<CufftLibrary>>()
        .saturating_add(size_of::<CudaRuntime>())
        .saturating_add(size_of::<usize>())
        .saturating_add(size_of::<CufftPlanKey>())
        .saturating_add(size_of::<CufftHandle>())
}

/// Run cuFFT library loading and plan creation only for a non-empty batch.
///
/// The caller supplies the complete library/cache/plan closure so a zero-batch
/// execution can return without loading cuFFT or creating a plan.
pub(crate) fn with_cufft_plan_for_batch<T>(
    batch: usize,
    load_and_create: impl FnOnce() -> Result<T, CudaFftError>,
) -> Result<Option<T>, CudaFftError> {
    if batch == 0 {
        Ok(None)
    } else {
        load_and_create().map(Some)
    }
}

impl CufftPlanEntry {
    /// Create a cached cuFFT plan entry on the exact retained runtime.
    pub(crate) fn create(
        session: &mut CudaExecSession<'_>,
        key: CufftPlanKey,
        descriptor: CufftPlanDescriptor,
    ) -> Result<Self, CudaFftError> {
        let library = CufftLibrary::load()?;
        Self::create_with_library(session, library, key, descriptor)
    }

    fn create_with_library(
        session: &mut CudaExecSession<'_>,
        library: Arc<CufftLibrary>,
        key: CufftPlanKey,
        descriptor: CufftPlanDescriptor,
    ) -> Result<Self, CudaFftError> {
        // Run the cuFFT create/make-plan sequence with the tenferro primary
        // context current, restoring the caller's previous context afterwards.
        let runtime = session.runtime().clone();
        let runtime_for_cleanup = runtime.clone();
        let build_result = runtime
            .with_current_context(OP, || {
                build_plan(Arc::clone(&library), runtime_for_cleanup, descriptor)
            })
            .map_err(|source| CudaFftError::interop("cufft_plan_context", source))?;
        let (plan, workspace_size) = build_result?;
        let workspace_bytes = workspace_size;
        let retained_bytes = retained_entry_bytes();
        Ok(Self {
            library,
            plan,
            workspace_bytes,
            runtime,
            key,
            retained_bytes,
        })
    }

    /// Execute this entry on the session's current stream. A synchronized raw
    /// session binds the plan stream and fresh work area, enqueues the cuFFT
    /// call, and blocks until completion; subsequent CUDA postprocessing may
    /// still remain queued on that stream.
    pub(crate) fn execute(
        &mut self,
        session: &mut CudaExecSession<'_>,
        input: &Tensor,
        output: &mut Tensor,
    ) -> Result<(), CudaFftError> {
        match self.key.kind {
            CufftTransformKind::C2c32 => match (input, output) {
                (Tensor::C32(input), Tensor::C32(output)) => self.execute_pair(
                    session,
                    input,
                    output,
                    |api, plan, input, output| {
                        // SAFETY: the raw-session tensor refs validate exact
                        // runtime residency and keep both buffers borrowed.
                        unsafe {
                            (api.exec_c2c)(plan, input, output, direction(self.key.direction))
                        }
                    },
                    "cufftExecC2C",
                ),
                _ => Err(CudaFftError::InvalidConfiguration { field: "dtype" }),
            },
            CufftTransformKind::C2c64 => match (input, output) {
                (Tensor::C64(input), Tensor::C64(output)) => self.execute_pair(
                    session,
                    input,
                    output,
                    |api, plan, input, output| {
                        // SAFETY: the raw-session tensor refs validate exact
                        // runtime residency and keep both buffers borrowed.
                        unsafe {
                            (api.exec_z2z)(plan, input, output, direction(self.key.direction))
                        }
                    },
                    "cufftExecZ2Z",
                ),
                _ => Err(CudaFftError::InvalidConfiguration { field: "dtype" }),
            },
            CufftTransformKind::R2c32 => match (input, output) {
                (Tensor::F32(input), Tensor::C32(output)) => self.execute_pair(
                    session,
                    input,
                    output,
                    |api, plan, input, output| {
                        // SAFETY: the raw-session tensor refs validate exact
                        // runtime residency and keep both buffers borrowed.
                        unsafe { (api.exec_r2c)(plan, input, output) }
                    },
                    "cufftExecR2C",
                ),
                _ => Err(CudaFftError::InvalidConfiguration { field: "dtype" }),
            },
            CufftTransformKind::R2c64 => match (input, output) {
                (Tensor::F64(input), Tensor::C64(output)) => self.execute_pair(
                    session,
                    input,
                    output,
                    |api, plan, input, output| {
                        // SAFETY: the raw-session tensor refs validate exact
                        // runtime residency and keep both buffers borrowed.
                        unsafe { (api.exec_d2z)(plan, input, output) }
                    },
                    "cufftExecD2Z",
                ),
                _ => Err(CudaFftError::InvalidConfiguration { field: "dtype" }),
            },
            CufftTransformKind::C2r32 => match (input, output) {
                (Tensor::C32(input), Tensor::F32(output)) => self.execute_pair(
                    session,
                    input,
                    output,
                    |api, plan, input, output| {
                        // SAFETY: the raw-session tensor refs validate exact
                        // runtime residency and keep both buffers borrowed.
                        unsafe { (api.exec_c2r)(plan, input, output) }
                    },
                    "cufftExecC2R",
                ),
                _ => Err(CudaFftError::InvalidConfiguration { field: "dtype" }),
            },
            CufftTransformKind::C2r64 => match (input, output) {
                (Tensor::C64(input), Tensor::F64(output)) => self.execute_pair(
                    session,
                    input,
                    output,
                    |api, plan, input, output| {
                        // SAFETY: the raw-session tensor refs validate exact
                        // runtime residency and keep both buffers borrowed.
                        unsafe { (api.exec_z2d)(plan, input, output) }
                    },
                    "cufftExecZ2D",
                ),
                _ => Err(CudaFftError::InvalidConfiguration { field: "dtype" }),
            },
        }
    }

    fn execute_pair<T, U>(
        &self,
        session: &mut CudaExecSession<'_>,
        input: &TypedTensor<T>,
        output: &mut TypedTensor<U>,
        mut call: impl FnMut(&CufftApi, CufftHandle, *mut c_void, *mut c_void) -> CufftStatus,
        function: &'static str,
    ) -> Result<(), CudaFftError>
    where
        T: TensorScalar + 'static,
        U: TensorScalar + 'static,
    {
        let mut execution_error = None;
        let mut synchronization_error = None;
        session
            .with_raw(OP, |raw| {
                // SAFETY: the stream handle is valid only for this raw-session
                // scope; it is used immediately to bind the cuFFT plan stream.
                let stream = unsafe { raw.stream().raw_handle() };
                if let Err(error) = bind_plan_to_stream(&self.library, self.plan, stream) {
                    execution_error = Some(error);
                    return Ok::<(), tenferro_tensor::Error>(());
                }
                let workspace = match raw.alloc_bytes(self.workspace_bytes, OP) {
                    Ok(workspace) => workspace,
                    Err(error) => {
                        execution_error =
                            Some(CudaFftError::interop("cufft_workspace_allocate", error));
                        return Ok::<(), tenferro_tensor::Error>(());
                    }
                };
                if let Err(error) = bind_workspace_to_plan(&self.library, self.plan, &workspace) {
                    execution_error = Some(error);
                    return Ok::<(), tenferro_tensor::Error>(());
                }
                // Retention guards hold reference-counted clones of the input
                // and output allocation handles. On a failed synchronization
                // barrier they are forgotten below so the allocations cannot be
                // reclaimed while the vendor may still be writing to them.
                let input_retained = match raw.retain_tensor(input, OP) {
                    Ok(guard) => guard,
                    Err(error) => {
                        execution_error =
                            Some(CudaFftError::interop("cufft_retain_input_handle", error));
                        return Ok::<(), tenferro_tensor::Error>(());
                    }
                };
                let output_retained = match raw.retain_tensor(output, OP) {
                    Ok(guard) => guard,
                    Err(error) => {
                        execution_error =
                            Some(CudaFftError::interop("cufft_retain_output_handle", error));
                        return Ok::<(), tenferro_tensor::Error>(());
                    }
                };
                let input_ref = match raw.tensor(input) {
                    Ok(reference) => reference,
                    Err(error) => {
                        execution_error =
                            Some(CudaFftError::interop("cufft_execute_input_pointer", error));
                        return Ok::<(), tenferro_tensor::Error>(());
                    }
                };
                let output_ref = match raw.tensor_mut(output) {
                    Ok(reference) => reference,
                    Err(error) => {
                        execution_error =
                            Some(CudaFftError::interop("cufft_execute_output_pointer", error));
                        return Ok::<(), tenferro_tensor::Error>(());
                    }
                };
                // SAFETY: both refs are validated device spans on this
                // runtime; the sizes were checked against the plan descriptor.
                let status = unsafe {
                    call(
                        &self.library.api,
                        self.plan,
                        input_ref.raw_ptr(),
                        output_ref.raw_ptr(),
                    )
                };
                if let Err(error) = map_cufft_status(function, status) {
                    execution_error = Some(error);
                }
                if let Err(error) = raw.synchronize() {
                    synchronization_error =
                        Some(CudaFftError::interop("cufft_execute_synchronize", error));
                }
                // INVARIANT: failed synchronization cannot prove vendor
                // completion, so the workspace and the input/output retention
                // guards are forgotten. The CubeCL allocations are then
                // intentionally retained until process exit, mirroring the
                // retired lease-witness retention (issue #967).
                if synchronization_error.is_some() {
                    std::mem::forget(workspace);
                    std::mem::forget(input_retained);
                    std::mem::forget(output_retained);
                }
                Ok::<(), tenferro_tensor::Error>(())
            })
            .map_err(|source| CudaFftError::interop("cufft_execute_session", source))?;

        merge_execution_errors(execution_error, synchronization_error)
    }

    pub(crate) fn retained_bytes(&self) -> usize {
        self.retained_bytes
    }

    pub(crate) fn matches_key(&self, key: &CufftPlanKey) -> bool {
        plan_key_discriminator_matches(&self.key, key)
    }
}

impl Drop for CufftPlanEntry {
    fn drop(&mut self) {
        let failures = retire_entry_resources(&self.runtime, &self.library, self.plan);
        report_cleanup_failures(failures);
    }
}

/// Add the exact runtime witness to the cache discriminator while retaining
/// the full equality-bearing key in every entry. The identity component is
/// only a discriminator: collisions cannot cause unsafe reuse because
/// `CufftPlanEntry::matches_key` compares the retained runtime identity.
pub(crate) fn extension_plan_key_for_runtime(key: &CufftPlanKey) -> ExtensionCacheKey {
    let mut hasher = DefaultHasher::new();
    key.hash(&mut hasher);
    key.runtime_identity.hash(&mut hasher);
    ExtensionCacheKey::new(
        FFT_EXTENSION_FAMILY_ID,
        CUFFT_CACHE_NAMESPACE,
        hasher.finish(),
    )
}

pub(crate) fn plan_key_discriminator_matches<I: PartialEq>(
    stored: &CufftPlanKey<I>,
    requested: &CufftPlanKey<I>,
) -> bool {
    stored == requested
}
