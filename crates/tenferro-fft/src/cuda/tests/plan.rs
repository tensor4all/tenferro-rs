// INVARIANT: The remaining tests cover the private `CufftPlanEntry` lifecycle,
// its shared fake cuFFT state, and directly coupled key/retained-byte metadata;
// independent production-source contract scans live in `source_contract.rs`.
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
    build_plan, merge_execution_errors, plan_key_discriminator_matches, report_cleanup_failures,
    retained_bytes_for_workspace, retire_entry_resources, retire_handle, with_cufft_plan_for_batch,
    CufftCleanup, CufftPlanEntry,
};

#[derive(Default)]
struct FakeState {
    calls: Vec<&'static str>,
    failure: Option<&'static str>,
    cleanup_failure: Option<&'static str>,
    workspace_bytes: usize,
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

extern "C" fn fake_set_work_area(_handle: CufftHandle, _workspace: *mut c_void) -> CufftStatus {
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
fn successful_construction_uses_required_cufft_order() {
    let _lock = test_lock();
    reset_state(128, None);
    let library = super::super::ffi::CufftLibrary::from_api_for_tests(fake_api());

    let (handle, workspace_bytes) = build_plan(Arc::clone(&library), FakeRuntime, descriptor())
        .unwrap_or_else(|_| unreachable!("fake plan should build"));

    assert_eq!(workspace_bytes, 128);
    assert_eq!(
        calls(),
        vec!["create", "set_auto_allocation", "make_plan_many_64"]
    );
    let mut handle = Some(handle);
    let failures = retire_handle(&FakeRuntime, &library, &mut handle);
    assert!(failures.is_empty());
    assert_eq!(
        calls(),
        vec![
            "create",
            "set_auto_allocation",
            "make_plan_many_64",
            "set_current",
            "synchronize",
            "destroy",
        ]
    );
    assert!(handle.is_none());
}

#[test]
fn merge_execution_errors_combines_primary_and_suppressed() {
    let execution_error = Some(CudaFftError::CufftStatus {
        function: "cufftExecC2C",
        status: CUFFT_EXEC_FAILED,
    });
    let synchronization_error = Some(CudaFftError::test_interop("cufft_execute_synchronize"));

    let result = merge_execution_errors(execution_error, synchronization_error);
    let CudaFftError::WithSuppressed {
        primary,
        suppressed,
    } = result.expect_err("both captured errors must be combined")
    else {
        panic!("expected primary and suppressed typed errors");
    };
    assert!(matches!(
        *primary,
        CudaFftError::CufftStatus {
            function: "cufftExecC2C",
            status: CUFFT_EXEC_FAILED,
        }
    ));
    assert!(matches!(*suppressed, CudaFftError::Interop { .. }));
}

#[test]
fn merge_execution_errors_returns_execution_error_only() {
    let result = merge_execution_errors(
        Some(CudaFftError::test_interop("cufft_execute_vendor")),
        None,
    );
    let error = result.expect_err("the execution error alone must be returned");
    assert!(matches!(error, CudaFftError::Interop { .. }));
}

#[test]
fn merge_execution_errors_returns_synchronization_error_only() {
    let result = merge_execution_errors(
        None,
        Some(CudaFftError::test_interop("cufft_execute_synchronize")),
    );
    let error = result.expect_err("the synchronization error alone must be returned");
    assert!(matches!(error, CudaFftError::Interop { .. }));
}

#[test]
fn merge_execution_errors_ok_when_both_none() {
    assert!(merge_execution_errors(None, None).is_ok());
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
fn handle_cleanup_runs_after_stream_synchronization() {
    let _lock = test_lock();
    reset_state(128, None);
    let library = super::super::ffi::CufftLibrary::from_api_for_tests(fake_api());
    let (handle, _) = build_plan(Arc::clone(&library), FakeRuntime, descriptor())
        .unwrap_or_else(|_| unreachable!("fake plan should build"));

    let mut handle = Some(handle);
    let failures = retire_handle(&FakeRuntime, &library, &mut handle);

    assert!(failures.is_empty());
    assert_eq!(
        calls(),
        vec![
            "create",
            "set_auto_allocation",
            "make_plan_many_64",
            "set_current",
            "synchronize",
            "destroy"
        ]
    );
    assert!(handle.is_none());
}

#[test]
fn retirement_retains_resources_when_context_or_stream_synchronization_fails() {
    for cleanup_failure in ["set_current", "synchronize"] {
        let _lock = test_lock();
        reset_state(128, None);
        let library = super::super::ffi::CufftLibrary::from_api_for_tests(fake_api());
        let (handle, _) = build_plan(Arc::clone(&library), FakeRuntime, descriptor())
            .unwrap_or_else(|_| unreachable!("fake plan should build"));

        reset_state_with_cleanup(128, None, Some(cleanup_failure));
        let mut handle = Some(handle);
        let failures = retire_handle(&FakeRuntime, &library, &mut handle);

        assert!(!failures.synchronization.is_empty());
        assert_eq!(calls().iter().filter(|&&call| call == "destroy").count(), 0);
    }
}

#[test]
fn construction_drop_defers_resources_when_cleanup_barrier_fails() {
    for cleanup_failure in ["set_current", "synchronize"] {
        let _lock = test_lock();
        reset_state_with_cleanup(128, Some("make_plan_many_64"), Some(cleanup_failure));
        let library = super::super::ffi::CufftLibrary::from_api_for_tests(fake_api());
        let witness = Arc::new(());
        let runtime = TrackedRuntime::new(Arc::clone(&witness));
        let result = build_plan(Arc::clone(&library), runtime, descriptor());

        assert!(result.is_err());
        assert_eq!(Arc::strong_count(&library), 2);
        assert_eq!(Arc::strong_count(&witness), 2);
        assert_eq!(calls().iter().filter(|&&call| call == "destroy").count(), 0);
    }
}

#[test]
fn retirement_failure_leaks_complete_lifetime_witness_bundle() {
    for cleanup_failure in ["set_current", "synchronize"] {
        let _lock = test_lock();
        reset_state(128, None);
        let library = super::super::ffi::CufftLibrary::from_api_for_tests(fake_api());
        let (handle, _) = build_plan(Arc::clone(&library), FakeRuntime, descriptor())
            .unwrap_or_else(|_| unreachable!("fake plan should build"));

        let witness = Arc::new(());
        let runtime = TrackedRuntime::new(Arc::clone(&witness));
        reset_state_with_cleanup(128, None, Some(cleanup_failure));
        let failures = retire_entry_resources(&runtime, &library, handle);

        assert!(!failures.synchronization.is_empty());
        assert!(failures.resources_deferred);
        assert_eq!(Arc::strong_count(&library), 2);
        assert_eq!(Arc::strong_count(&witness), 3);
        assert_eq!(calls().iter().filter(|&&call| call == "destroy").count(), 0);
    }
}

#[test]
fn plan_entry_drop_retirement_leaks_complete_lifetime_witness_bundle() {
    for cleanup_failure in ["set_current", "synchronize"] {
        let _lock = test_lock();
        reset_state(128, None);
        let library = super::super::ffi::CufftLibrary::from_api_for_tests(fake_api());
        let (handle, _) = build_plan(Arc::clone(&library), FakeRuntime, descriptor())
            .unwrap_or_else(|_| unreachable!("fake plan should build"));

        let witness = Arc::new(());
        let runtime = TrackedRuntime::new(Arc::clone(&witness));
        reset_state_with_cleanup(128, None, Some(cleanup_failure));
        let failures = retire_entry_resources(&runtime, &library, handle);

        assert!(!failures.synchronization.is_empty());
        assert!(failures.resources_deferred);
        assert_eq!(Arc::strong_count(&library), 2);
        assert_eq!(Arc::strong_count(&witness), 3);
        assert_eq!(calls().iter().filter(|&&call| call == "destroy").count(), 0);
    }
}

#[test]
fn successful_retirement_releases_resources_and_lifetime_witnesses() {
    let _lock = test_lock();
    reset_state(128, None);
    let library = super::super::ffi::CufftLibrary::from_api_for_tests(fake_api());
    let (handle, _) = build_plan(Arc::clone(&library), FakeRuntime, descriptor())
        .unwrap_or_else(|_| unreachable!("fake plan should build"));

    let witness = Arc::new(());
    let runtime = TrackedRuntime::new(Arc::clone(&witness));
    let failures = retire_entry_resources(&runtime, &library, handle);

    assert!(failures.is_empty());
    assert_eq!(Arc::strong_count(&library), 1);
    assert_eq!(Arc::strong_count(&witness), 2);
    assert_eq!(calls().iter().filter(|&&call| call == "destroy").count(), 1);
}

#[test]
fn construction_rolls_back_every_completed_stage_without_double_destroy() {
    for failure in ["set_auto_allocation", "make_plan_many_64"] {
        let _lock = test_lock();
        reset_state(128, Some(failure));
        let library = super::super::ffi::CufftLibrary::from_api_for_tests(fake_api());
        let result = build_plan(Arc::clone(&library), FakeRuntime, descriptor());
        assert!(result.is_err(), "stage {failure} should fail");
        assert_eq!(calls().iter().filter(|&&call| call == "destroy").count(), 1);
        assert_eq!(
            calls()
                .iter()
                .filter(|&&call| call == "make_plan_many_64")
                .count(),
            usize::from(failure != "set_auto_allocation")
        );
    }
}

#[test]
fn failure_before_handle_creation_does_not_destroy_or_synchronize() {
    let _lock = test_lock();
    reset_state(128, Some("create"));
    let library = super::super::ffi::CufftLibrary::from_api_for_tests(fake_api());
    let result = build_plan(Arc::clone(&library), FakeRuntime, descriptor());

    assert!(result.is_err());
    assert_eq!(calls(), vec!["create"]);
}

#[test]
fn plan_many_failure_returns_error_and_rolls_back_handle() {
    let _lock = test_lock();
    reset_state(128, Some("make_plan_many_64"));
    let library = super::super::ffi::CufftLibrary::from_api_for_tests(fake_api());
    let result = build_plan(Arc::clone(&library), FakeRuntime, descriptor());

    assert!(result.is_err());
    assert_eq!(calls().iter().filter(|&&call| call == "destroy").count(), 1);
}

#[test]
fn retirement_errors_are_reported_without_panicking() {
    let _lock = test_lock();
    reset_state(128, Some("destroy"));
    let library = super::super::ffi::CufftLibrary::from_api_for_tests(fake_api());
    let (handle, _) = build_plan(Arc::clone(&library), FakeRuntime, descriptor())
        .unwrap_or_else(|_| unreachable!("fake plan should build"));
    let mut handle = Some(handle);
    reset_state(128, Some("destroy"));

    let result = catch_unwind(AssertUnwindSafe(|| {
        retire_handle(&FakeRuntime, &library, &mut handle)
    }));
    assert!(result.is_ok());
    let failures = result.unwrap_or_else(|_| unreachable!("retirement must not panic"));
    assert!(failures.synchronization.is_empty());
    assert!(failures.destroy.is_some());
    assert!(failures.resources_deferred);
    assert!(handle.is_none());

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
        + workspace_bytes;

    assert_eq!(retained_bytes_for_workspace(workspace_bytes), expected);
    assert!(size_of::<CufftPlanEntry>() >= size_of::<usize>());
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
