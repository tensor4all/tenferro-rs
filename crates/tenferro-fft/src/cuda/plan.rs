//! cuFFT plan/workspace ownership and scoped execution.
//!
//! Plan creation follows NVIDIA's documented manual-work-area sequence:
//! `cufftCreate`, `cufftSetAutoAllocation(plan, 0)`,
//! `cufftMakePlanMany64`, workspace allocation, and `cufftSetWorkArea`.
//! Workspace and plan retirement synchronize the retained CubeCL stream before
//! destroying the opaque cuFFT handle or dropping the CubeCL allocation.

use std::collections::hash_map::DefaultHasher;
use std::ffi::c_void;
use std::hash::{Hash, Hasher};
use std::io::Write;
use std::mem::size_of;
use std::sync::Arc;

use tenferro_gpu::cuda::interop::{
    alloc_device_bytes, with_raw_cuda_stream, with_typed_device_ptr, DeviceByteBuffer,
};
use tenferro_gpu::cuda::CudaRuntime;
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

/// Cache namespace for operation-family-owned cuFFT plans and workspaces.
pub(crate) const CUFFT_CACHE_NAMESPACE: &str = "cufft-plans";

/// A cleanup context used by the plan-construction and retirement guards.
pub(crate) trait CufftCleanup {
    fn set_current(&self) -> Result<(), CudaFftError>;
    fn synchronize(&self) -> Result<(), CudaFftError>;
}

impl CufftCleanup for CudaRuntime {
    fn set_current(&self) -> Result<(), CudaFftError> {
        self.set_current_cuda_context("cufft_plan_cleanup")
            .map_err(|source| CudaFftError::interop("cufft_plan_cleanup_context", source))
    }

    fn synchronize(&self) -> Result<(), CudaFftError> {
        CudaRuntime::synchronize(self)
            .map_err(|source| CudaFftError::interop("cufft_plan_cleanup_stream", source))
    }
}

/// A workspace owner that can expose a pointer only for one scoped FFI call.
pub(crate) trait CufftWorkspaceOwner: Sized {
    fn empty() -> Self;
    fn with_ptr(&self, f: impl FnOnce(*mut c_void));
}

/// CubeCL-owned workspace retained by one cuFFT plan entry.
pub(crate) struct CufftWorkspace {
    _owner: DeviceByteBuffer,
    bytes: usize,
}

impl CufftWorkspace {
    fn from_device(owner: DeviceByteBuffer, bytes: usize) -> Self {
        Self {
            _owner: owner,
            bytes,
        }
    }

    fn bytes(&self) -> usize {
        self.bytes
    }
}

impl CufftWorkspaceOwner for CufftWorkspace {
    fn empty() -> Self {
        Self::from_device(DeviceByteBuffer::none(), 0)
    }

    fn with_ptr(&self, f: impl FnOnce(*mut c_void)) {
        self._owner.with_ptr(f);
    }
}

#[derive(Default)]
pub(crate) struct CleanupFailures {
    pub(crate) synchronization: Vec<CudaFftError>,
    pub(crate) destroy: Option<CudaFftError>,
    /// True when cleanup could not prove retirement safety and the complete
    /// plan/workspace/lifetime-witness bundle was intentionally leaked.
    pub(crate) resources_deferred: bool,
}

impl CleanupFailures {
    pub(crate) fn is_empty(&self) -> bool {
        self.synchronization.is_empty() && self.destroy.is_none() && !self.resources_deferred
    }
}

/// Retire one handle and its workspace without panicking from `Drop` paths.
///
/// The plan and workspace are released only after the retained runtime is made
/// current and its stream synchronization succeeds. If either step fails, the
/// opaque plan, workspace, library witness, and runtime witness are
/// intentionally leaked: destroying or dropping any of them could race work
/// whose completion was not proven.
pub(crate) fn retire_handle<R, W>(
    runtime: &R,
    library: &Arc<CufftLibrary>,
    handle: &mut Option<CufftHandle>,
    workspace: &mut Option<W>,
) -> CleanupFailures
where
    R: CufftCleanup + Clone,
    W: CufftWorkspaceOwner,
{
    let mut failures = CleanupFailures::default();
    if handle.is_some() || workspace.is_some() {
        if let Err(error) = runtime.set_current() {
            failures.synchronization.push(error);
            failures.resources_deferred = true;
            defer_resources(runtime, library, handle, workspace);
            return failures;
        }
        if let Err(error) = runtime.synchronize() {
            failures.synchronization.push(error);
            failures.resources_deferred = true;
            defer_resources(runtime, library, handle, workspace);
            return failures;
        }
    }

    if let Some(plan) = *handle {
        // SAFETY: `plan` was returned by the same retained cuFFT library's
        // successful `cufftCreate` call and has not been destroyed before this
        // one-shot retirement path.
        let status = unsafe { (library.api.destroy)(plan) };
        if let Err(error) = map_cufft_status("cufftDestroy", status) {
            failures.destroy = Some(error);
            failures.resources_deferred = true;
            // A failed destroy leaves the opaque plan's ownership uncertain;
            // keep the complete lifetime bundle with the intentionally leaked
            // plan and workspace.
            defer_resources(runtime, library, handle, workspace);
            return failures;
        }
        handle.take();
    }

    let workspace = workspace.take();
    drop(workspace);
    failures
}

/// Owns every witness needed to keep an unretired cuFFT operation valid.
///
/// This value is deliberately forgotten only when cleanup cannot prove that
/// vendor work has completed. There is no safe retry or process-global
/// quarantine for a plan whose context/stream barrier failed.
struct DeferredCufftResources<R, W> {
    _handle: Option<CufftHandle>,
    _workspace: Option<W>,
    _library: Arc<CufftLibrary>,
    _runtime: R,
}

fn defer_resources<R, W>(
    runtime: &R,
    library: &Arc<CufftLibrary>,
    handle: &mut Option<CufftHandle>,
    workspace: &mut Option<W>,
) where
    R: Clone,
    W: CufftWorkspaceOwner,
{
    // The forgotten bundle retains the vendor plan, workspace allocation,
    // dynamic-library handle, and CUDA context/runtime clone together until
    // process exit. Dropping any member independently could invalidate queued
    // work after a failed cleanup barrier.
    std::mem::forget(DeferredCufftResources {
        _handle: handle.take(),
        _workspace: workspace.take(),
        _library: Arc::clone(library),
        _runtime: runtime.clone(),
    });
}

/// Retire one owned plan/workspace pair, as used by `CufftPlanEntry::Drop`.
pub(crate) fn retire_entry_resources<R, W>(
    runtime: &R,
    library: &Arc<CufftLibrary>,
    plan: CufftHandle,
    workspace: W,
) -> CleanupFailures
where
    R: CufftCleanup + Clone,
    W: CufftWorkspaceOwner,
{
    let mut handle = Some(plan);
    let mut workspace = Some(workspace);
    retire_handle(runtime, library, &mut handle, &mut workspace)
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

struct CufftConstructionGuard<R, W>
where
    R: CufftCleanup + Clone,
    W: CufftWorkspaceOwner,
{
    library: Arc<CufftLibrary>,
    runtime: R,
    handle: Option<CufftHandle>,
    workspace: Option<W>,
}

impl<R, W> CufftConstructionGuard<R, W>
where
    R: CufftCleanup + Clone,
    W: CufftWorkspaceOwner,
{
    fn new(library: Arc<CufftLibrary>, runtime: R) -> Self {
        Self {
            library,
            runtime,
            handle: None,
            workspace: None,
        }
    }

    fn disarm(mut self) -> Result<(CufftHandle, W), CudaFftError> {
        let Some(handle) = self.handle.take() else {
            return Err(CudaFftError::internal(
                "cuFFT construction completed without a handle",
            ));
        };
        let Some(workspace) = self.workspace.take() else {
            // Restore the handle so the guard's Drop path still destroys it if
            // an internal construction invariant ever fails.
            self.handle = Some(handle);
            return Err(CudaFftError::internal(
                "cuFFT construction completed without workspace",
            ));
        };
        Ok((handle, workspace))
    }
}

impl<R, W> Drop for CufftConstructionGuard<R, W>
where
    R: CufftCleanup + Clone,
    W: CufftWorkspaceOwner,
{
    fn drop(&mut self) {
        let failures = retire_handle(
            &self.runtime,
            &self.library,
            &mut self.handle,
            &mut self.workspace,
        );
        report_cleanup_failures(failures);
    }
}

/// Build a plan with the manual work-area sequence and rollback guard.
pub(crate) fn build_plan<R, W, F>(
    library: Arc<CufftLibrary>,
    runtime: R,
    mut descriptor: CufftPlanDescriptor,
    mut allocate_workspace: F,
) -> Result<(CufftHandle, W), CudaFftError>
where
    R: CufftCleanup + Clone,
    W: CufftWorkspaceOwner,
    F: FnMut(usize) -> Result<W, CudaFftError>,
{
    let mut guard = CufftConstructionGuard::<R, W>::new(Arc::clone(&library), runtime);
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

    let workspace = if workspace_size == 0 {
        W::empty()
    } else {
        allocate_workspace(workspace_size)?
    };
    guard.workspace = Some(workspace);

    let mut work_area_result = Ok(());
    if let Some(workspace) = guard.workspace.as_ref() {
        workspace.with_ptr(|ptr| {
            // SAFETY: the plan was created and the workspace pointer is scoped
            // to the owner that remains alive in `guard` until destroy.
            let status = unsafe { (library.api.set_work_area)(handle, ptr) };
            work_area_result = map_cufft_status("cufftSetWorkArea", status);
        });
    }
    work_area_result?;
    guard.disarm()
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

/// Bind a plan to the current stream without waiting for queued work.
pub(crate) fn bind_plan_to_stream(
    library: &CufftLibrary,
    plan: CufftHandle,
    stream: u64,
) -> Result<(), CudaFftError> {
    let stream = usize::try_from(stream)
        .map_err(|_| CudaFftError::InvalidConfiguration { field: "stream" })?
        as *mut c_void;
    // SAFETY: `plan` belongs to `library`, and `stream` is borrowed from the
    // retained runtime for the duration of this callback.
    let status = unsafe { (library.api.set_stream)(plan, stream) };
    map_cufft_status("cufftSetStream", status)
}

pub(crate) trait CufftExecutionScopes {
    fn with_stream(&self, callback: impl FnOnce(u64)) -> Result<(), CudaFftError>;
    fn with_input_ptr(&self, callback: impl FnOnce(*mut c_void)) -> Result<(), CudaFftError>;
    fn with_output_ptr(&self, callback: impl FnOnce(*mut c_void)) -> Result<(), CudaFftError>;
}

struct TypedExecutionScopes<'a, T, U> {
    runtime: &'a CudaRuntime,
    input: &'a TypedTensor<T>,
    output: &'a TypedTensor<U>,
}

impl<T, U> CufftExecutionScopes for TypedExecutionScopes<'_, T, U>
where
    T: TensorScalar + 'static,
    U: TensorScalar + 'static,
{
    fn with_stream(&self, callback: impl FnOnce(u64)) -> Result<(), CudaFftError> {
        with_raw_cuda_stream(self.runtime, OP, callback)
            .map_err(|source| CudaFftError::interop("cufft_execute_stream", source))
    }

    fn with_input_ptr(&self, callback: impl FnOnce(*mut c_void)) -> Result<(), CudaFftError> {
        with_typed_device_ptr(self.runtime, self.input, OP, callback)
            .map_err(|source| CudaFftError::interop("cufft_execute_input_pointer", source))
    }

    fn with_output_ptr(&self, callback: impl FnOnce(*mut c_void)) -> Result<(), CudaFftError> {
        with_typed_device_ptr(self.runtime, self.output, OP, callback)
            .map_err(|source| CudaFftError::interop("cufft_execute_output_pointer", source))
    }
}

pub(crate) fn enqueue_plan_execution<S, C>(
    scopes: &S,
    library: &CufftLibrary,
    plan: CufftHandle,
    mut call: C,
    function: &'static str,
) -> Result<(), CudaFftError>
where
    S: CufftExecutionScopes,
    C: FnMut(&CufftApi, CufftHandle, *mut c_void, *mut c_void) -> CufftStatus,
{
    let mut execution_result = Ok(());
    scopes.with_stream(|stream| {
        if let Err(error) = bind_plan_to_stream(library, plan, stream) {
            execution_result = Err(error);
            return;
        }

        let mut pointer_error = None;
        let input_result = scopes.with_input_ptr(|input_ptr| {
            if let Err(error) = scopes.with_output_ptr(|output_ptr| {
                let status = call(&library.api, plan, input_ptr, output_ptr);
                execution_result = map_cufft_status(function, status);
            }) {
                pointer_error = Some(error);
            }
        });
        if let Err(error) = input_result {
            execution_result = Err(error);
        } else if let Some(error) = pointer_error {
            execution_result = Err(error);
        }
    })?;
    execution_result
}

/// One cached cuFFT plan and its manually owned CubeCL work area.
//
// INVARIANT: stream identity is intentionally omitted from `CufftPlanKey`;
// mutable `CudaExecSession` access serializes the runtime's one selectable
// current CubeCL stream. A selectable multi-stream runtime must make stream
// identity part of this key before concurrent plan reuse is allowed.
pub(crate) struct CufftPlanEntry {
    pub(crate) library: Arc<CufftLibrary>,
    pub(crate) plan: CufftHandle,
    pub(crate) workspace: CufftWorkspace,
    pub(crate) runtime: CudaRuntime,
    pub(crate) key: CufftPlanKey,
    retained_bytes: usize,
}

// SAFETY: the library and runtime retain their executable/context witnesses;
// mutable execution and retirement are entered through one borrowed session
// and the cache's mutable typed-entry access.
unsafe impl Send for CufftPlanEntry {}
// SAFETY: callers cannot execute through a shared reference, and the retained
// workspace/plan are only accessed through the owning mutable session boundary.
unsafe impl Sync for CufftPlanEntry {}

pub(crate) fn retained_bytes_for_workspace(workspace_bytes: usize) -> usize {
    size_of::<Arc<CufftLibrary>>()
        .saturating_add(size_of::<CudaRuntime>())
        .saturating_add(size_of::<usize>())
        .saturating_add(size_of::<CufftPlanKey>())
        .saturating_add(size_of::<CufftHandle>())
        .saturating_add(size_of::<CufftWorkspace>())
        .saturating_add(workspace_bytes)
}

/// Run cuFFT library loading and plan creation only for a non-empty batch.
///
/// Task 5 must put its complete `CufftLibrary::load`/plan-cache closure here so
/// a zero-batch execution can return without loading cuFFT or creating a plan.
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
        runtime: &CudaRuntime,
        key: CufftPlanKey,
        descriptor: CufftPlanDescriptor,
    ) -> Result<Self, CudaFftError> {
        let library = CufftLibrary::load()?;
        Self::create_with_library(runtime, library, key, descriptor)
    }

    fn create_with_library(
        runtime: &CudaRuntime,
        library: Arc<CufftLibrary>,
        key: CufftPlanKey,
        descriptor: CufftPlanDescriptor,
    ) -> Result<Self, CudaFftError> {
        runtime
            .set_current_cuda_context(OP)
            .map_err(|source| CudaFftError::interop("cufft_plan_context", source))?;
        let runtime_for_cleanup = runtime.clone();
        let (plan, workspace) = build_plan(
            Arc::clone(&library),
            runtime_for_cleanup,
            descriptor,
            |bytes| {
                alloc_device_bytes(runtime, bytes, OP)
                    .map(|workspace| CufftWorkspace::from_device(workspace, bytes))
                    .map_err(|source| CudaFftError::interop("cufft_workspace_allocate", source))
            },
        )?;
        let retained_bytes = retained_bytes_for_workspace(workspace.bytes());
        Ok(Self {
            library,
            plan,
            workspace,
            runtime: runtime.clone(),
            key,
            retained_bytes,
        })
    }

    /// Execute this entry on the retained runtime's currently selected stream.
    pub(crate) fn execute(
        &mut self,
        input: &Tensor,
        output: &mut Tensor,
    ) -> Result<(), CudaFftError> {
        self.runtime
            .set_current_cuda_context(OP)
            .map_err(|source| CudaFftError::interop("cufft_execute_context", source))?;

        match self.key.kind {
            CufftTransformKind::C2c32 => match (input, output) {
                (Tensor::C32(input), Tensor::C32(output)) => self.execute_pair(
                    input,
                    output,
                    |api, plan, input, output| {
                        // SAFETY: the tensor-pointer scopes validate exact
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
                    input,
                    output,
                    |api, plan, input, output| {
                        // SAFETY: the tensor-pointer scopes validate exact
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
                    input,
                    output,
                    |api, plan, input, output| {
                        // SAFETY: the tensor-pointer scopes validate exact
                        // runtime residency and keep both buffers borrowed.
                        unsafe { (api.exec_r2c)(plan, input, output) }
                    },
                    "cufftExecR2C",
                ),
                _ => Err(CudaFftError::InvalidConfiguration { field: "dtype" }),
            },
            CufftTransformKind::R2c64 => match (input, output) {
                (Tensor::F64(input), Tensor::C64(output)) => self.execute_pair(
                    input,
                    output,
                    |api, plan, input, output| {
                        // SAFETY: the tensor-pointer scopes validate exact
                        // runtime residency and keep both buffers borrowed.
                        unsafe { (api.exec_d2z)(plan, input, output) }
                    },
                    "cufftExecD2Z",
                ),
                _ => Err(CudaFftError::InvalidConfiguration { field: "dtype" }),
            },
            CufftTransformKind::C2r32 => match (input, output) {
                (Tensor::C32(input), Tensor::F32(output)) => self.execute_pair(
                    input,
                    output,
                    |api, plan, input, output| {
                        // SAFETY: the tensor-pointer scopes validate exact
                        // runtime residency and keep both buffers borrowed.
                        unsafe { (api.exec_c2r)(plan, input, output) }
                    },
                    "cufftExecC2R",
                ),
                _ => Err(CudaFftError::InvalidConfiguration { field: "dtype" }),
            },
            CufftTransformKind::C2r64 => match (input, output) {
                (Tensor::C64(input), Tensor::F64(output)) => self.execute_pair(
                    input,
                    output,
                    |api, plan, input, output| {
                        // SAFETY: the tensor-pointer scopes validate exact
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
        input: &TypedTensor<T>,
        output: &TypedTensor<U>,
        call: impl FnMut(&CufftApi, CufftHandle, *mut c_void, *mut c_void) -> CufftStatus,
        function: &'static str,
    ) -> Result<(), CudaFftError>
    where
        T: TensorScalar + 'static,
        U: TensorScalar + 'static,
    {
        let scopes = TypedExecutionScopes {
            runtime: &self.runtime,
            input,
            output,
        };
        enqueue_plan_execution(&scopes, &self.library, self.plan, call, function)
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
        let workspace = std::mem::replace(&mut self.workspace, CufftWorkspace::empty());
        let failures = retire_entry_resources(&self.runtime, &self.library, self.plan, workspace);
        report_cleanup_failures(failures);
    }
}

/// Add the exact runtime owner to the cache discriminator while retaining the
/// full equality-bearing key in every entry. The runtime address is only a
/// discriminator: pointer reuse or collision cannot cause unsafe reuse because
/// `CufftPlanEntry::matches_key` compares the retained runtime identity.
pub(crate) fn extension_plan_key_for_runtime(
    key: &CufftPlanKey,
    runtime: &CudaRuntime,
) -> ExtensionCacheKey {
    let mut hasher = DefaultHasher::new();
    key.hash(&mut hasher);
    hasher.write_usize(runtime as *const CudaRuntime as usize);
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
