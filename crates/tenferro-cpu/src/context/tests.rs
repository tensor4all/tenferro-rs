use super::CpuContext;

#[test]
fn with_threads_rejects_zero() {
    assert!(CpuContext::with_threads(0).is_err());
}
