use super::CpuContext;

#[test]
fn try_with_threads_rejects_zero() {
    assert!(CpuContext::try_with_threads(0).is_err());
}
