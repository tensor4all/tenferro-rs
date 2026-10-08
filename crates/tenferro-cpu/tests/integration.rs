// The provider-pointer registry is process-global, so every test that
// registers or observes an injected CBLAS/LAPACK symbol serializes on this lock.
#[cfg(all(feature = "blas", feature = "provider-inject"))]
pub(crate) static PROVIDER_INJECT_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[path = "integration/backend_capability_contracts.rs"]
mod backend_capability_contracts;
#[path = "integration/inject_dual_abi_tests.rs"]
mod inject_dual_abi_tests;
#[path = "integration/inject_tests.rs"]
mod inject_tests;
#[path = "integration/runtime_error_tests.rs"]
mod runtime_error_tests;
#[path = "integration/static_replay.rs"]
mod static_replay;
