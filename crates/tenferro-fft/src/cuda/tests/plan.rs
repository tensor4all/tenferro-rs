use std::ffi::c_void;
use std::hash::Hasher;
use std::mem::size_of;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

use tenferro_gpu::cuda::CudaRuntime;

use super::super::descriptor::{
    CufftDirection, CufftPlanDescriptor, CufftPlanKey, CufftTransformKind,
};
use super::super::error::CudaFftError;
use super::super::ffi::{
    CufftApi, CufftHandle, CufftLibrary, CufftStatus, CUFFT_ALLOC_FAILED, CUFFT_EXEC_FAILED,
    CUFFT_INVALID_PLAN, CUFFT_SUCCESS,
};
use super::super::plan::{
    build_plan, enqueue_plan_execution, plan_key_discriminator_matches, report_cleanup_failures,
    retained_bytes_for_workspace, retire_entry_resources, retire_handle, with_cufft_plan_for_batch,
    CufftCleanup, CufftExecutionScopes, CufftPlanEntry, CufftWorkspace, CufftWorkspaceOwner,
};

#[derive(Default)]
struct FakeState {
    calls: Vec<&'static str>,
    failure: Option<&'static str>,
    cleanup_failure: Option<&'static str>,
    workspace_bytes: usize,
    allocations: usize,
    work_area_ptr: Option<usize>,
}

static TEST_LOCK: Mutex<()> = Mutex::new(());
static TEST_STATE: OnceLock<Mutex<FakeState>> = OnceLock::new();

fn state() -> &'static Mutex<FakeState> {
    TEST_STATE.get_or_init(|| Mutex::new(FakeState::default()))
}

fn test_lock() -> MutexGuard<'static, ()> {
    TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn reset_state(workspace_bytes: usize, failure: Option<&'static str>) {
    reset_state_with_cleanup(workspace_bytes, failure, None);
}

fn reset_state_with_cleanup(
    workspace_bytes: usize,
    failure: Option<&'static str>,
    cleanup_failure: Option<&'static str>,
) {
    let mut state = state()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    *state = FakeState {
        workspace_bytes,
        failure,
        cleanup_failure,
        ..FakeState::default()
    };
}

fn record(call: &'static str) {
    state()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .calls
        .push(call);
}

fn failed(stage: &'static str) -> bool {
    state()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .failure
        .is_some_and(|failure| failure == stage || failure == "both")
}

fn cleanup_failed(stage: &'static str) -> bool {
    state()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .cleanup_failure
        .is_some_and(|failure| failure == stage || failure == "both")
}

extern "C" fn fake_create(handle: *mut CufftHandle) -> CufftStatus {
    record("create");
    if failed("create") {
        return CUFFT_INVALID_PLAN;
    }
    // SAFETY: the fake test table receives the valid output slot supplied by
    // the construction helper, just like cufftCreate's C ABI contract.
    unsafe { *handle = 7 };
    CUFFT_SUCCESS
}

extern "C" fn fake_set_auto_allocation(_handle: CufftHandle, _enabled: i32) -> CufftStatus {
    record("set_auto_allocation");
    if failed("set_auto_allocation") {
        CUFFT_ALLOC_FAILED
    } else {
        CUFFT_SUCCESS
    }
}

extern "C" fn fake_make_plan_many_64(
    _handle: CufftHandle,
    _rank: i32,
    _n: *mut i64,
    _inembed: *mut i64,
    _istride: i64,
    _idist: i64,
    _onembed: *mut i64,
    _ostride: i64,
    _odist: i64,
    _kind: i32,
    _batch: i64,
    work_size: *mut usize,
) -> CufftStatus {
    record("make_plan_many_64");
    if failed("make_plan_many_64") {
        return CUFFT_INVALID_PLAN;
    }
    let workspace_bytes = state()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .workspace_bytes;
    // SAFETY: the fake test table receives the valid output slot supplied by
    // the construction helper, just like cufftMakePlanMany64's C ABI contract.
    unsafe { *work_size = workspace_bytes };
    CUFFT_SUCCESS
}

extern "C" fn fake_set_work_area(_handle: CufftHandle, workspace: *mut c_void) -> CufftStatus {
    state()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .work_area_ptr = Some(workspace as usize);
    record("set_work_area");
    if failed("set_work_area") {
        CUFFT_EXEC_FAILED
    } else {
        CUFFT_SUCCESS
    }
}

extern "C" fn fake_set_stream(_handle: CufftHandle, _stream: *mut c_void) -> CufftStatus {
    record("set_stream");
    CUFFT_SUCCESS
}

extern "C" fn fake_exec_c2c(
    _handle: CufftHandle,
    _input: *mut c_void,
    _output: *mut c_void,
    _direction: i32,
) -> CufftStatus {
    record("exec_c2c");
    CUFFT_SUCCESS
}

extern "C" fn fake_exec_r2c(
    _handle: CufftHandle,
    _input: *mut c_void,
    _output: *mut c_void,
) -> CufftStatus {
    CUFFT_SUCCESS
}

extern "C" fn fake_exec_c2r(
    _handle: CufftHandle,
    _input: *mut c_void,
    _output: *mut c_void,
) -> CufftStatus {
    CUFFT_SUCCESS
}

extern "C" fn fake_exec_z2z(
    _handle: CufftHandle,
    _input: *mut c_void,
    _output: *mut c_void,
    _direction: i32,
) -> CufftStatus {
    CUFFT_SUCCESS
}

extern "C" fn fake_exec_d2z(
    _handle: CufftHandle,
    _input: *mut c_void,
    _output: *mut c_void,
) -> CufftStatus {
    CUFFT_SUCCESS
}

extern "C" fn fake_exec_z2d(
    _handle: CufftHandle,
    _input: *mut c_void,
    _output: *mut c_void,
) -> CufftStatus {
    CUFFT_SUCCESS
}

extern "C" fn fake_destroy(_handle: CufftHandle) -> CufftStatus {
    record("destroy");
    if failed("destroy") {
        CUFFT_INVALID_PLAN
    } else {
        CUFFT_SUCCESS
    }
}

fn fake_api() -> CufftApi {
    CufftApi {
        create: fake_create,
        set_auto_allocation: fake_set_auto_allocation,
        make_plan_many_64: fake_make_plan_many_64,
        set_work_area: fake_set_work_area,
        set_stream: fake_set_stream,
        exec_c2c: fake_exec_c2c,
        exec_r2c: fake_exec_r2c,
        exec_c2r: fake_exec_c2r,
        exec_z2z: fake_exec_z2z,
        exec_d2z: fake_exec_d2z,
        exec_z2d: fake_exec_z2d,
        destroy: fake_destroy,
    }
}

#[derive(Clone)]
struct FakeRuntime;

impl CufftCleanup for FakeRuntime {
    fn set_current(&self) -> Result<(), CudaFftError> {
        record("set_current");
        if cleanup_failed("set_current") {
            Err(CudaFftError::test_interop("set_current"))
        } else {
            Ok(())
        }
    }

    fn synchronize(&self) -> Result<(), CudaFftError> {
        record("synchronize");
        if cleanup_failed("synchronize") {
            Err(CudaFftError::test_interop("synchronize"))
        } else {
            Ok(())
        }
    }
}

#[derive(Clone)]
struct TrackedRuntime {
    witness: Arc<()>,
}

impl TrackedRuntime {
    fn new(witness: Arc<()>) -> Self {
        Self { witness }
    }
}

impl CufftCleanup for TrackedRuntime {
    fn set_current(&self) -> Result<(), CudaFftError> {
        let _ = &self.witness;
        record("set_current");
        if cleanup_failed("set_current") {
            Err(CudaFftError::test_interop("set_current"))
        } else {
            Ok(())
        }
    }

    fn synchronize(&self) -> Result<(), CudaFftError> {
        record("synchronize");
        if cleanup_failed("synchronize") {
            Err(CudaFftError::test_interop("synchronize"))
        } else {
            Ok(())
        }
    }
}

struct FakeExecutionScopes;

impl CufftExecutionScopes for FakeExecutionScopes {
    fn with_stream(&self, callback: impl FnOnce(u64)) -> Result<(), CudaFftError> {
        callback(0);
        Ok(())
    }

    fn with_input_ptr(&self, callback: impl FnOnce(*mut c_void)) -> Result<(), CudaFftError> {
        record("input_pointer_scope");
        callback(std::ptr::null_mut());
        Ok(())
    }

    fn with_output_ptr(&self, callback: impl FnOnce(*mut c_void)) -> Result<(), CudaFftError> {
        record("output_pointer_scope");
        callback(std::ptr::null_mut());
        Ok(())
    }
}

struct FakeWorkspace {
    ptr: *mut c_void,
}

impl CufftWorkspaceOwner for FakeWorkspace {
    fn empty() -> Self {
        Self {
            ptr: std::ptr::null_mut(),
        }
    }

    fn with_ptr(&self, f: impl FnOnce(*mut c_void)) {
        f(self.ptr);
    }
}

impl Drop for FakeWorkspace {
    fn drop(&mut self) {
        record("workspace_drop");
    }
}

fn descriptor() -> CufftPlanDescriptor {
    CufftPlanDescriptor::new(CufftTransformKind::C2c32, CufftDirection::Forward, 8, 3)
        .unwrap_or_else(|_| unreachable!("test descriptor is valid"))
}

fn calls() -> Vec<&'static str> {
    state()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .calls
        .clone()
}

#[test]
fn zero_workspace_uses_a_null_pointer_without_allocating() {
    let _lock = test_lock();
    reset_state(0, None);
    let library = super::super::ffi::CufftLibrary::from_api_for_tests(fake_api());

    let (_handle, workspace) = build_plan::<FakeRuntime, FakeWorkspace, _>(
        Arc::clone(&library),
        FakeRuntime,
        descriptor(),
        |_bytes| unreachable!("zero-byte plans must not allocate workspace"),
    )
    .unwrap_or_else(|_| unreachable!("fake zero-workspace plan should build"));

    let state = state()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    assert_eq!(state.allocations, 0);
    assert_eq!(state.work_area_ptr, Some(0));
    drop(state);
    drop(workspace);
}

#[test]
fn successful_construction_uses_required_cufft_order() {
    let _lock = test_lock();
    reset_state(128, None);
    let library = super::super::ffi::CufftLibrary::from_api_for_tests(fake_api());

    let (handle, workspace) =
        build_plan(Arc::clone(&library), FakeRuntime, descriptor(), |_bytes| {
            record("allocate_workspace");
            state()
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .allocations += 1;
            Ok(FakeWorkspace {
                ptr: 0x1234usize as *mut c_void,
            })
        })
        .unwrap_or_else(|_| unreachable!("fake plan should build"));

    assert_eq!(
        calls(),
        vec![
            "create",
            "set_auto_allocation",
            "make_plan_many_64",
            "allocate_workspace",
            "set_work_area"
        ]
    );
    let mut handle = Some(handle);
    let mut workspace = Some(workspace);
    let failures = retire_handle(&FakeRuntime, &library, &mut handle, &mut workspace);
    assert!(failures.is_empty());
    assert_eq!(
        calls(),
        vec![
            "create",
            "set_auto_allocation",
            "make_plan_many_64",
            "allocate_workspace",
            "set_work_area",
            "set_current",
            "synchronize",
            "destroy",
            "workspace_drop",
        ]
    );
    assert!(handle.is_none());
    assert!(workspace.is_none());
}

#[test]
fn fake_plan_execution_uses_shared_enqueue_helper_without_synchronizing() {
    let _lock = test_lock();
    reset_state(0, None);
    let library = super::super::ffi::CufftLibrary::from_api_for_tests(fake_api());
    let scopes = FakeExecutionScopes;

    enqueue_plan_execution(
        &scopes,
        &library,
        7,
        |api, plan, input, output| {
            // SAFETY: the fake function table records the call and does not
            // dereference these test-only placeholder pointers.
            unsafe { (api.exec_c2c)(plan, input, output, 1) }
        },
        "cufftExecC2C",
    )
    .unwrap_or_else(|_| unreachable!("fake plan execution should enqueue"));

    assert_eq!(
        calls(),
        vec![
            "set_stream",
            "input_pointer_scope",
            "output_pointer_scope",
            "exec_c2c"
        ]
    );
    assert!(!calls().contains(&"synchronize"));
}

#[test]
fn zero_batch_plan_gate_skips_invalid_loader_and_plan_creation() {
    let mut load_calls = 0;
    let mut plan_calls = 0;
    let plan = with_cufft_plan_for_batch(0, || {
        load_calls += 1;
        plan_calls += 1;
        Err::<(), _>(CudaFftError::NoLibraryCandidates)
    })
    .unwrap_or_else(|_| unreachable!("zero-batch gate should not fail"));

    assert!(plan.is_none());
    assert_eq!(load_calls, 0);
    assert_eq!(plan_calls, 0);
}

#[test]
fn nonzero_batch_plan_gate_loads_and_creates_once() {
    let mut load_calls = 0;
    let mut plan_calls = 0;
    let plan = with_cufft_plan_for_batch(1, || {
        load_calls += 1;
        plan_calls += 1;
        Ok::<_, CudaFftError>(7)
    })
    .unwrap_or_else(|_| unreachable!("nonzero-batch gate should invoke the closure"));

    assert_eq!(plan, Some(7));
    assert_eq!(load_calls, 1);
    assert_eq!(plan_calls, 1);
}

#[test]
fn handle_cleanup_runs_after_stream_synchronization_and_workspace_release() {
    let _lock = test_lock();
    reset_state(128, None);
    let library = super::super::ffi::CufftLibrary::from_api_for_tests(fake_api());
    let (handle, workspace) =
        build_plan(Arc::clone(&library), FakeRuntime, descriptor(), |_bytes| {
            record("allocate_workspace");
            Ok(FakeWorkspace {
                ptr: 0x1234usize as *mut c_void,
            })
        })
        .unwrap_or_else(|_| unreachable!("fake plan should build"));

    let mut handle = Some(handle);
    let mut workspace = Some(workspace);
    let failures = retire_handle(&FakeRuntime, &library, &mut handle, &mut workspace);

    assert!(failures.is_empty());
    assert_eq!(
        calls(),
        vec![
            "create",
            "set_auto_allocation",
            "make_plan_many_64",
            "allocate_workspace",
            "set_work_area",
            "set_current",
            "synchronize",
            "destroy",
            "workspace_drop"
        ]
    );
    assert!(handle.is_none());
    assert!(workspace.is_none());
}

#[test]
fn retirement_retains_resources_when_context_or_stream_synchronization_fails() {
    for cleanup_failure in ["set_current", "synchronize"] {
        let _lock = test_lock();
        reset_state(128, None);
        let library = super::super::ffi::CufftLibrary::from_api_for_tests(fake_api());
        let (handle, workspace) =
            build_plan(Arc::clone(&library), FakeRuntime, descriptor(), |_bytes| {
                record("allocate_workspace");
                Ok(FakeWorkspace {
                    ptr: 0x1234usize as *mut c_void,
                })
            })
            .unwrap_or_else(|_| unreachable!("fake plan should build"));

        reset_state_with_cleanup(128, None, Some(cleanup_failure));
        let mut handle = Some(handle);
        let mut workspace = Some(workspace);
        let failures = retire_handle(&FakeRuntime, &library, &mut handle, &mut workspace);

        assert!(!failures.synchronization.is_empty());
        assert_eq!(calls().iter().filter(|&&call| call == "destroy").count(), 0);
        assert_eq!(
            calls()
                .iter()
                .filter(|&&call| call == "workspace_drop")
                .count(),
            0
        );
    }
}

#[test]
fn construction_drop_defers_resources_when_cleanup_barrier_fails() {
    for cleanup_failure in ["set_current", "synchronize"] {
        let _lock = test_lock();
        reset_state_with_cleanup(128, Some("set_work_area"), Some(cleanup_failure));
        let library = super::super::ffi::CufftLibrary::from_api_for_tests(fake_api());
        let witness = Arc::new(());
        let runtime = TrackedRuntime::new(Arc::clone(&witness));
        let result = build_plan(Arc::clone(&library), runtime, descriptor(), |_bytes| {
            record("allocate_workspace");
            Ok(FakeWorkspace {
                ptr: 0x1234usize as *mut c_void,
            })
        });

        assert!(result.is_err());
        assert_eq!(Arc::strong_count(&library), 2);
        assert_eq!(Arc::strong_count(&witness), 2);
        assert_eq!(calls().iter().filter(|&&call| call == "destroy").count(), 0);
        assert_eq!(
            calls()
                .iter()
                .filter(|&&call| call == "workspace_drop")
                .count(),
            0
        );
    }
}

#[test]
fn retirement_failure_leaks_complete_lifetime_witness_bundle() {
    for cleanup_failure in ["set_current", "synchronize"] {
        let _lock = test_lock();
        reset_state(128, None);
        let library = super::super::ffi::CufftLibrary::from_api_for_tests(fake_api());
        let (handle, workspace) =
            build_plan(Arc::clone(&library), FakeRuntime, descriptor(), |_bytes| {
                record("allocate_workspace");
                Ok(FakeWorkspace {
                    ptr: 0x1234usize as *mut c_void,
                })
            })
            .unwrap_or_else(|_| unreachable!("fake plan should build"));

        let witness = Arc::new(());
        let runtime = TrackedRuntime::new(Arc::clone(&witness));
        reset_state_with_cleanup(128, None, Some(cleanup_failure));
        let mut handle = Some(handle);
        let mut workspace = Some(workspace);
        let failures = retire_handle(&runtime, &library, &mut handle, &mut workspace);

        assert!(!failures.synchronization.is_empty());
        assert!(failures.resources_deferred);
        assert_eq!(Arc::strong_count(&library), 2);
        assert_eq!(Arc::strong_count(&witness), 3);
        assert!(handle.is_none());
        assert!(workspace.is_none());
        assert_eq!(calls().iter().filter(|&&call| call == "destroy").count(), 0);
        assert_eq!(
            calls()
                .iter()
                .filter(|&&call| call == "workspace_drop")
                .count(),
            0
        );
    }
}

#[test]
fn plan_entry_drop_retirement_leaks_complete_lifetime_witness_bundle() {
    for cleanup_failure in ["set_current", "synchronize"] {
        let _lock = test_lock();
        reset_state(128, None);
        let library = super::super::ffi::CufftLibrary::from_api_for_tests(fake_api());
        let (handle, workspace) =
            build_plan(Arc::clone(&library), FakeRuntime, descriptor(), |_bytes| {
                record("allocate_workspace");
                Ok(FakeWorkspace {
                    ptr: 0x1234usize as *mut c_void,
                })
            })
            .unwrap_or_else(|_| unreachable!("fake plan should build"));

        let witness = Arc::new(());
        let runtime = TrackedRuntime::new(Arc::clone(&witness));
        reset_state_with_cleanup(128, None, Some(cleanup_failure));
        let failures = retire_entry_resources(&runtime, &library, handle, workspace);

        assert!(!failures.synchronization.is_empty());
        assert!(failures.resources_deferred);
        assert_eq!(Arc::strong_count(&library), 2);
        assert_eq!(Arc::strong_count(&witness), 3);
        assert_eq!(calls().iter().filter(|&&call| call == "destroy").count(), 0);
        assert_eq!(
            calls()
                .iter()
                .filter(|&&call| call == "workspace_drop")
                .count(),
            0
        );
    }
}

#[test]
fn successful_retirement_releases_resources_and_lifetime_witnesses() {
    let _lock = test_lock();
    reset_state(128, None);
    let library = super::super::ffi::CufftLibrary::from_api_for_tests(fake_api());
    let (handle, workspace) =
        build_plan(Arc::clone(&library), FakeRuntime, descriptor(), |_bytes| {
            record("allocate_workspace");
            Ok(FakeWorkspace {
                ptr: 0x1234usize as *mut c_void,
            })
        })
        .unwrap_or_else(|_| unreachable!("fake plan should build"));

    let witness = Arc::new(());
    let runtime = TrackedRuntime::new(Arc::clone(&witness));
    let failures = retire_entry_resources(&runtime, &library, handle, workspace);

    assert!(failures.is_empty());
    assert_eq!(Arc::strong_count(&library), 1);
    assert_eq!(Arc::strong_count(&witness), 2);
    assert_eq!(calls().iter().filter(|&&call| call == "destroy").count(), 1);
    assert_eq!(
        calls()
            .iter()
            .filter(|&&call| call == "workspace_drop")
            .count(),
        1
    );
}

#[test]
fn construction_rolls_back_every_completed_stage_without_double_destroy() {
    for failure in [
        "set_auto_allocation",
        "make_plan_many_64",
        "allocate",
        "set_work_area",
    ] {
        let _lock = test_lock();
        reset_state(128, Some(failure));
        let library = super::super::ffi::CufftLibrary::from_api_for_tests(fake_api());
        let result = build_plan(Arc::clone(&library), FakeRuntime, descriptor(), |_bytes| {
            record("allocate_workspace");
            state()
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .allocations += 1;
            if failed("allocate") {
                Err(CudaFftError::test_interop("allocate_workspace"))
            } else {
                Ok(FakeWorkspace {
                    ptr: 0x1234usize as *mut c_void,
                })
            }
        });
        assert!(result.is_err(), "stage {failure} should fail");
        let calls = calls();
        assert_eq!(calls.iter().filter(|&&call| call == "destroy").count(), 1);
        assert_eq!(
            calls
                .iter()
                .filter(|&&call| call == "workspace_drop")
                .count(),
            usize::from(failure == "set_work_area")
        );
        assert_eq!(
            calls
                .iter()
                .filter(|&&call| call == "make_plan_many_64")
                .count(),
            usize::from(failure != "set_auto_allocation")
        );
        assert_eq!(
            calls
                .iter()
                .filter(|&&call| call == "set_work_area")
                .count(),
            usize::from(failure == "set_work_area")
        );
        assert_eq!(
            state()
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .allocations,
            usize::from(failure != "set_auto_allocation" && failure != "make_plan_many_64")
        );
    }
}

#[test]
fn failure_before_handle_creation_does_not_destroy_or_synchronize() {
    let _lock = test_lock();
    reset_state(128, Some("create"));
    let library = super::super::ffi::CufftLibrary::from_api_for_tests(fake_api());
    let result = build_plan::<FakeRuntime, FakeWorkspace, _>(
        Arc::clone(&library),
        FakeRuntime,
        descriptor(),
        |_bytes| unreachable!("allocation must not run"),
    );

    assert!(result.is_err());
    assert_eq!(calls(), vec!["create"]);
}

#[test]
fn plan_many_failure_does_not_allocate_workspace() {
    let _lock = test_lock();
    reset_state(128, Some("make_plan_many_64"));
    let library = super::super::ffi::CufftLibrary::from_api_for_tests(fake_api());
    let result = build_plan::<FakeRuntime, FakeWorkspace, _>(
        Arc::clone(&library),
        FakeRuntime,
        descriptor(),
        |_bytes| unreachable!("allocation must not run"),
    );

    assert!(result.is_err());
    assert_eq!(
        state()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .allocations,
        0
    );
    assert_eq!(calls().iter().filter(|&&call| call == "destroy").count(), 1);
}

#[test]
fn work_area_failure_releases_workspace_after_destroy() {
    let _lock = test_lock();
    reset_state(128, Some("set_work_area"));
    let library = super::super::ffi::CufftLibrary::from_api_for_tests(fake_api());
    let result = build_plan(Arc::clone(&library), FakeRuntime, descriptor(), |_bytes| {
        record("allocate_workspace");
        Ok(FakeWorkspace {
            ptr: 0x1234usize as *mut c_void,
        })
    });

    assert!(result.is_err());
    let calls = calls();
    assert!(
        calls.iter().position(|&call| call == "destroy").unwrap()
            < calls
                .iter()
                .position(|&call| call == "workspace_drop")
                .unwrap()
    );
}

#[test]
fn retirement_errors_are_reported_without_panicking() {
    let _lock = test_lock();
    reset_state(128, Some("destroy"));
    let library = super::super::ffi::CufftLibrary::from_api_for_tests(fake_api());
    let (handle, workspace) =
        build_plan(Arc::clone(&library), FakeRuntime, descriptor(), |_bytes| {
            record("allocate_workspace");
            Ok(FakeWorkspace {
                ptr: 0x1234usize as *mut c_void,
            })
        })
        .unwrap_or_else(|_| unreachable!("fake plan should build"));
    let mut handle = Some(handle);
    let mut workspace = Some(workspace);
    reset_state(128, Some("destroy"));

    let result = catch_unwind(AssertUnwindSafe(|| {
        retire_handle(&FakeRuntime, &library, &mut handle, &mut workspace)
    }));
    assert!(result.is_ok());
    let failures = result.unwrap_or_else(|_| unreachable!("retirement must not panic"));
    assert!(failures.synchronization.is_empty());
    assert!(failures.destroy.is_some());
    assert!(failures.resources_deferred);
    assert!(handle.is_none());
    assert!(workspace.is_none());
    assert_eq!(
        calls()
            .iter()
            .filter(|&&call| call == "workspace_drop")
            .count(),
        0
    );

    let report_result = catch_unwind(AssertUnwindSafe(|| report_cleanup_failures(failures)));
    assert!(report_result.is_ok(), "cleanup reporting must not panic");
}

#[derive(Default)]
struct ConstantHasher;

impl std::hash::Hasher for ConstantHasher {
    fn finish(&self) -> u64 {
        0
    }

    fn write(&mut self, _bytes: &[u8]) {}
}

fn constant_hash<T: std::hash::Hash>(value: &T) -> u64 {
    let mut hasher = ConstantHasher;
    value.hash(&mut hasher);
    hasher.finish()
}

#[test]
fn retained_bytes_include_entry_metadata_and_workspace() {
    let workspace_bytes = 128;
    let expected = size_of::<Arc<CufftLibrary>>()
        + size_of::<CudaRuntime>()
        + size_of::<usize>()
        + size_of::<CufftPlanKey>()
        + size_of::<CufftHandle>()
        + size_of::<CufftWorkspace>()
        + workspace_bytes;

    assert_eq!(retained_bytes_for_workspace(workspace_bytes), expected);
    assert!(size_of::<CufftPlanEntry>() >= size_of::<CufftWorkspace>());
}

#[test]
fn workspace_and_execution_keep_ffi_pointers_scoped() {
    let source = include_str!("../plan.rs");
    let workspace_section = source
        .split_once("pub(crate) struct CufftWorkspace")
        .and_then(|(_, rest)| rest.split_once("#[derive(Default)]"))
        .map(|(section, _)| section)
        .unwrap_or_else(|| unreachable!("workspace source section should exist"));
    assert!(!workspace_section.contains("ptr: *mut c_void"));
    assert!(!workspace_section.contains("unsafe impl Send for CufftWorkspace"));
    assert!(!workspace_section.contains("unsafe impl Sync for CufftWorkspace"));

    let enqueue_section = source
        .split_once("pub(crate) fn enqueue_plan_execution")
        .and_then(|(_, rest)| rest.split_once("/// One cached cuFFT plan"))
        .map(|(section, _)| section)
        .unwrap_or_else(|| unreachable!("enqueue helper source section should exist"));
    assert!(enqueue_section.contains("bind_plan_to_stream"));
    assert!(enqueue_section.contains("with_input_ptr"));
    assert!(enqueue_section.contains("with_output_ptr"));
    assert!(!enqueue_section.contains("synchronize"));

    let execute_pair_section = source
        .split_once("fn execute_pair")
        .and_then(|(_, rest)| rest.split_once("    pub(crate) fn retained_bytes"))
        .map(|(section, _)| section)
        .unwrap_or_else(|| unreachable!("execute pair source section should exist"));
    assert!(execute_pair_section.contains("enqueue_plan_execution"));
}

#[test]
fn exact_plan_key_match_rejects_collisions_and_runtime_identity_mismatch() {
    let first = CufftPlanKey::<usize> {
        runtime_identity: 1,
        device_ordinal: 0,
        kind: CufftTransformKind::C2c32,
        direction: CufftDirection::Forward,
        n: 8,
        batch: 3,
        istride: 3,
        idist: 1,
        ostride: 3,
        odist: 1,
    };
    let different_shape = CufftPlanKey {
        n: 7,
        ..first.clone()
    };
    let different_runtime = CufftPlanKey {
        runtime_identity: 2,
        ..first.clone()
    };

    assert_eq!(constant_hash(&first), constant_hash(&different_shape));
    assert_ne!(first, different_shape);
    assert!(!plan_key_discriminator_matches(&first, &different_shape));
    assert_ne!(first, different_runtime);
    assert!(!plan_key_discriminator_matches(&first, &different_runtime));
}

#[test]
fn cuda_sources_do_not_cross_the_explicit_transfer_boundary() {
    // Keep this list explicit: every CUDA production module must be reviewed
    // when it is added so transfer and host-payload calls cannot bypass this
    // contract by falling outside a recursive source scan.
    let sources = [
        ("mod.rs", include_str!("../mod.rs")),
        ("descriptor.rs", include_str!("../descriptor.rs")),
        ("error.rs", include_str!("../error.rs")),
        ("ffi.rs", include_str!("../ffi.rs")),
        ("hermitian.rs", include_str!("../hermitian.rs")),
        ("plan.rs", include_str!("../plan.rs")),
    ];
    let forbidden = [
        concat!("upload", "_tensor("),
        concat!("download", "_tensor("),
        concat!("host", "_data("),
        concat!("host", "_data_mut("),
    ];
    for (path, source) in sources {
        for pattern in forbidden {
            assert!(
                !source.contains(pattern),
                "CUDA production source {path} must not contain {pattern}"
            );
        }
    }
}
