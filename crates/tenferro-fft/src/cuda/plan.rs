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
use std::mem::size_of;
use std::sync::Arc;

use tenferro_gpu::cuda::interop::{
    alloc_device_bytes, with_raw_cuda_stream, with_typed_device_ptr, DeviceByteBuffer,
};
use tenferro_gpu::cuda::CudaRuntime;
use tenferro_runtime::ExtensionCacheKey;
use tenferro_tensor::{Tensor, TensorScalar, TypedTensor};

use super::descriptor::{
    CufftDirection, CufftPlanDescriptor, CufftPlanKey, CufftPlanStructuralKey, CufftTransformKind,
};
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
    ptr: *mut c_void,
    bytes: usize,
}

impl CufftWorkspace {
    fn from_device(owner: DeviceByteBuffer, bytes: usize) -> Self {
        let mut ptr = std::ptr::null_mut();
        owner.with_ptr(|device_ptr| ptr = device_ptr);
        Self {
            _owner: owner,
            ptr,
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
        // The pointer is only passed during this callback; `_owner` keeps the
        // CubeCL allocation alive for the whole plan-entry lifetime.
        f(self.ptr);
    }
}

// SAFETY: `owner` retains the CubeCL allocation and the pointer is never
// exposed except through the scoped `with_ptr` callback. Plan execution and
// retirement are serialized by the mutable FFT execution/cache boundary.
unsafe impl Send for CufftWorkspace {}
// SAFETY: immutable references only permit another scoped pointer borrow; the
// owning execution session serializes the actual cuFFT calls.
unsafe impl Sync for CufftWorkspace {}

#[derive(Default)]
pub(crate) struct CleanupFailures {
    pub(crate) synchronization: Vec<CudaFftError>,
    pub(crate) destroy: Option<CudaFftError>,
}

impl CleanupFailures {
    pub(crate) fn is_empty(&self) -> bool {
        self.synchronization.is_empty() && self.destroy.is_none()
    }
}

/// Retire one handle and its workspace without panicking from `Drop` paths.
///
/// The workspace is dropped only after the stream barrier and destroy call have
/// both been attempted. Any errors are returned so callers that can report
/// them do not silently discard cleanup failures; `Drop` callers log them via
/// [`report_cleanup_failures`].
pub(crate) fn retire_handle<R, W>(
    runtime: &R,
    library: &CufftLibrary,
    handle: &mut Option<CufftHandle>,
    workspace: &mut Option<W>,
) -> CleanupFailures
where
    R: CufftCleanup,
    W: CufftWorkspaceOwner,
{
    let mut failures = CleanupFailures::default();
    if handle.is_some() || workspace.is_some() {
        if let Err(error) = runtime.set_current() {
            failures.synchronization.push(error);
        }
        if let Err(error) = runtime.synchronize() {
            failures.synchronization.push(error);
        }
    }

    if let Some(handle) = handle.take() {
        // SAFETY: `handle` was returned by the same retained cuFFT library's
        // successful `cufftCreate` call and has not been destroyed before this
        // one-shot retirement path.
        let status = unsafe { (library.api.destroy)(handle) };
        if let Err(error) = map_cufft_status("cufftDestroy", status) {
            failures.destroy = Some(error);
        }
    }

    // Keep this explicit: cuFFT may still refer to the caller-owned work area
    // until destroy returns, so releasing it earlier would race queued work.
    let workspace = workspace.take();
    drop(workspace);
    failures
}

#[cold]
pub(crate) fn report_cleanup_failures(failures: CleanupFailures) {
    for error in failures.synchronization {
        eprintln!("tenferro-fft: cuFFT cleanup synchronization failed: {error}");
    }
    if let Some(error) = failures.destroy {
        eprintln!("tenferro-fft: cuFFT plan destroy failed during cleanup: {error}");
    }
}

struct CufftConstructionGuard<R, W>
where
    R: CufftCleanup,
    W: CufftWorkspaceOwner,
{
    library: Arc<CufftLibrary>,
    runtime: R,
    handle: Option<CufftHandle>,
    workspace: Option<W>,
}

impl<R, W> CufftConstructionGuard<R, W>
where
    R: CufftCleanup,
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
    R: CufftCleanup,
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
    R: CufftCleanup,
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
        let retained_bytes = size_of::<CufftPlanKey>()
            .saturating_add(size_of::<CufftHandle>())
            .saturating_add(size_of::<CufftWorkspace>())
            .saturating_add(workspace.bytes());
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
        let mut stream_result = Ok(());
        with_raw_cuda_stream(&self.runtime, OP, |stream| {
            let stream = match usize::try_from(stream) {
                Ok(stream) => stream as *mut c_void,
                Err(_) => {
                    stream_result = Err(CudaFftError::InvalidConfiguration { field: "stream" });
                    return;
                }
            };
            // SAFETY: the plan belongs to `self.library`, and the stream is
            // borrowed only for this scoped callback from the retained runtime.
            let status = unsafe { (self.library.api.set_stream)(self.plan, stream) };
            stream_result = map_cufft_status("cufftSetStream", status);
        })
        .map_err(|source| CudaFftError::interop("cufft_execute_stream", source))?;
        stream_result?;

        match self.key.kind {
            CufftTransformKind::C2c32 => match (input, output) {
                (Tensor::C32(input), Tensor::C32(output)) => self.execute_pair(
                    input,
                    output,
                    |api, plan, input, output| {
                        // SAFETY: the tensor-pointer callbacks validate exact
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
                        // SAFETY: the tensor-pointer callbacks validate exact
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
                        // SAFETY: the tensor-pointer callbacks validate exact
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
                        // SAFETY: the tensor-pointer callbacks validate exact
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
                        // SAFETY: the tensor-pointer callbacks validate exact
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
                        // SAFETY: the tensor-pointer callbacks validate exact
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
        call: impl FnOnce(&CufftApi, CufftHandle, *mut c_void, *mut c_void) -> CufftStatus,
        function: &'static str,
    ) -> Result<(), CudaFftError>
    where
        T: TensorScalar + 'static,
        U: TensorScalar + 'static,
    {
        let mut call_result = Ok(());
        let mut pointer_error = None;
        let input_result = with_typed_device_ptr(&self.runtime, input, OP, |input_ptr| {
            if let Err(error) = with_typed_device_ptr(&self.runtime, output, OP, |output_ptr| {
                let status = call(&self.library.api, self.plan, input_ptr, output_ptr);
                call_result = map_cufft_status(function, status);
            }) {
                pointer_error = Some(error);
            }
        });
        input_result
            .map_err(|source| CudaFftError::interop("cufft_execute_input_pointer", source))?;
        if let Some(source) = pointer_error {
            return Err(CudaFftError::interop(
                "cufft_execute_output_pointer",
                source,
            ));
        }
        call_result
    }

    pub(crate) fn retained_bytes(&self) -> usize {
        self.retained_bytes
    }

    pub(crate) fn matches_key(&self, key: &CufftPlanKey) -> bool {
        self.key == *key
    }
}

impl Drop for CufftPlanEntry {
    fn drop(&mut self) {
        let mut handle = Some(self.plan);
        let mut workspace = Some(std::mem::replace(
            &mut self.workspace,
            CufftWorkspace::empty(),
        ));
        let failures = retire_handle(&self.runtime, &self.library, &mut handle, &mut workspace);
        report_cleanup_failures(failures);
    }
}

pub(crate) fn extension_plan_key(key: &CufftPlanKey) -> ExtensionCacheKey {
    let mut hasher = DefaultHasher::new();
    key.hash(&mut hasher);
    ExtensionCacheKey::new(
        FFT_EXTENSION_FAMILY_ID,
        CUFFT_CACHE_NAMESPACE,
        hasher.finish(),
    )
}

pub(crate) fn plan_key_discriminator_matches(
    stored: &CufftPlanStructuralKey,
    requested: &CufftPlanStructuralKey,
) -> bool {
    stored == requested
}
