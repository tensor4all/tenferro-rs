use std::ffi::c_void;
use std::hash::Hasher;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

use super::super::descriptor::{CufftDirection, CufftPlanDescriptor, CufftTransformKind};
use super::super::error::CudaFftError;
use super::super::ffi::{
    CufftApi, CufftHandle, CufftStatus, CUFFT_ALLOC_FAILED, CUFFT_EXEC_FAILED, CUFFT_INVALID_PLAN,
    CUFFT_SUCCESS,
};
use super::super::plan::{
    build_plan, plan_key_discriminator_matches, retire_handle, CufftCleanup, CufftWorkspaceOwner,
};

#[derive(Default)]
struct FakeState {
    calls: Vec<&'static str>,
    failure: Option<&'static str>,
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
    let mut state = state()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    *state = FakeState {
        workspace_bytes,
        failure,
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
        if failed("set_current") {
            Err(CudaFftError::test_interop("set_current"))
        } else {
            Ok(())
        }
    }

    fn synchronize(&self) -> Result<(), CudaFftError> {
        record("synchronize");
        if failed("synchronize") {
            Err(CudaFftError::test_interop("synchronize"))
        } else {
            Ok(())
        }
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
            "set_work_area"
        ]
    );
    drop(workspace);
    assert_eq!(handle, 7);
}

#[test]
fn handle_cleanup_runs_after_stream_synchronization_and_workspace_release() {
    let _lock = test_lock();
    reset_state(128, None);
    let library = super::super::ffi::CufftLibrary::from_api_for_tests(fake_api());
    let (handle, workspace) =
        build_plan(Arc::clone(&library), FakeRuntime, descriptor(), |_bytes| {
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
            Ok(FakeWorkspace {
                ptr: 0x1234usize as *mut c_void,
            })
        })
        .unwrap_or_else(|_| unreachable!("fake plan should build"));
    let mut handle = Some(handle);
    let mut workspace = Some(workspace);
    reset_state(128, Some("both"));

    let result = catch_unwind(AssertUnwindSafe(|| {
        retire_handle(&FakeRuntime, &library, &mut handle, &mut workspace)
    }));
    assert!(result.is_ok());
    let failures = result.unwrap_or_else(|_| unreachable!("retirement must not panic"));
    assert!(!failures.synchronization.is_empty());
    assert!(failures.destroy.is_some());
    assert!(handle.is_none());
    assert!(workspace.is_none());
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
fn exact_plan_discriminator_match_rejects_colliding_structural_keys() {
    let first = super::super::descriptor::CufftPlanStructuralKey::new(
        0,
        CufftTransformKind::C2c32,
        CufftDirection::Forward,
        8,
        3,
        3,
        1,
        3,
        1,
    );
    let second = super::super::descriptor::CufftPlanStructuralKey::new(
        0,
        CufftTransformKind::C2c32,
        CufftDirection::Forward,
        7,
        3,
        3,
        1,
        3,
        1,
    );
    assert_eq!(constant_hash(&first), constant_hash(&second));
    assert!(!plan_key_discriminator_matches(&first, &second));
}
