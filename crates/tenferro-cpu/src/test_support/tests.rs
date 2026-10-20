use super::*;

#[test]
fn cpu_engine_rejects_a_zero_thread_count_without_building_a_pool() {
    let error = cpu_engine(0).expect_err("zero threads must be refused");
    assert!(matches!(
        error,
        crate::Error::Validation {
            op: "test_support::cpu_engine",
            source: crate::ValidationError::InvalidArgument {
                argument: "configuration",
                ..
            },
        }
    ));
}

#[test]
fn test_engine_helpers_forward_through_a_mutable_reference_engine() {
    let mut engine = cpu_engine(1).expect("serial test engine");
    let has_native_engine = on_engine(&mut &mut engine, |erased| erased.native_engine().is_some())
        .expect("erased engine call");
    assert!(has_native_engine);

    let borrowed = &mut engine;
    let value = borrowed.run_erased(|_| 7usize).expect("erased run");
    assert_eq!(value, 7);
}

#[test]
fn with_cpu_session_runs_on_the_concrete_engine() {
    let mut engine = cpu_engine(1).expect("serial test engine");
    let threads = with_cpu_session(&mut engine, |cpu| cpu.num_threads());
    assert_eq!(threads, 1);
}
