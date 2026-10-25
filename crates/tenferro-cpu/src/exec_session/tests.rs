use super::*;

#[test]
fn integer_dot_is_rejected_without_mutating_output() {
    let lhs = Tensor::from_vec_col_major(vec![2], vec![2_i64, 3]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![2], vec![5_i64, 7]).unwrap();
    let config = DotGeneralConfig {
        lhs_contracting_dims: [].as_slice().into(),
        rhs_contracting_dims: [].as_slice().into(),
        lhs_batch_dims: [0].as_slice().into(),
        rhs_batch_dims: [0].as_slice().into(),
    };
    let mut backend = crate::test_support::cpu_engine(1).unwrap();
    assert!(
        crate::test_support::on_engine(&mut backend, |session| session.dot_general_read(
            TensorRead::from_tensor(&lhs),
            TensorRead::from_tensor(&rhs),
            &config
        ))
        .unwrap()
        .is_err()
    );
    let mut target = Tensor::from_vec_col_major(vec![2], vec![1_i64, 2]).unwrap();
    assert!(
        crate::test_support::on_engine(&mut backend, |session| session.dot_general_read_into(
            TensorRead::from_tensor(&lhs),
            TensorRead::from_tensor(&rhs),
            &config,
            TensorWrite::from_tensor(&mut target)
        ))
        .unwrap()
        .is_err()
    );
    assert_eq!(target.as_slice::<i64>().unwrap(), [1, 2]);
}

#[test]
fn backend_controls_clear_and_bound_owned_nary_workspace() {
    let mut backend = crate::test_support::cpu_engine(1).unwrap();
    let baseline = backend.runtime_cache_stats();
    let populate = |backend: &mut crate::CpuEngine| {
        crate::test_support::on_engine(backend, |session| {
            crate::with_cpu_engine(session, |cpu| {
                cpu.with_contraction_exec(|_, buffers, workspace| {
                    workspace.with_scratch::<f64, _>(
                        16,
                        buffers.max_retained_capacity_bytes(),
                        |_| Ok(()),
                    )
                })
            })
            .unwrap()
        })
        .unwrap()
        .unwrap();
    };
    populate(&mut backend);
    let filled = backend.runtime_cache_stats();
    assert_eq!(filled.entries, baseline.entries + 1);
    assert_eq!(filled.retained_bytes, baseline.retained_bytes + 128);
    backend.reset_buffer_pool();
    assert_eq!(
        backend.runtime_cache_stats().retained_bytes,
        baseline.retained_bytes
    );
    populate(&mut backend);
    backend.set_buffer_pool_limit_bytes(0);
    assert_eq!(
        backend.runtime_cache_stats().retained_bytes,
        baseline.retained_bytes
    );
    backend.set_buffer_pool_limit_bytes(1024);
    populate(&mut backend);
    backend.clear_runtime_caches();
    assert_eq!(
        backend.runtime_cache_stats().retained_bytes,
        baseline.retained_bytes
    );
}

#[test]
fn linalg_resource_callback_and_result_stay_on_the_caller_without_send() {
    use std::rc::Rc;

    let caller = std::thread::current().id();
    let value = Rc::new(37);
    let mut backend = crate::test_support::cpu_engine(2).unwrap();
    let result = crate::test_support::on_engine(&mut backend, |session| {
        crate::with_cpu_engine(session, |cpu| {
            cpu.with_linalg_pool(|_, _| {
                assert_eq!(std::thread::current().id(), caller);
                Ok(Rc::clone(&value))
            })
        })
        .unwrap()
    })
    .unwrap()
    .unwrap();
    assert!(Rc::ptr_eq(&value, &result));
}

#[cfg(not(feature = "provider-inject"))]
#[test]
fn native_operation_starts_on_the_caller_and_fans_out_on_the_engine_pool() {
    let mut backend = crate::test_support::cpu_engine(2).unwrap();

    let caller = std::thread::current().id();
    let (same_thread, participants) = crate::tests::with_cpu_session(&mut backend, |cpu| {
        cpu.with_linalg_pool(|exec, _| {
            Ok((
                std::thread::current().id() == caller,
                crate::tests::native_participants::run_unscoped_native_map(true, exec.token()),
            ))
        })
    })
    .unwrap();
    // The operation itself stays on the calling thread (#2206); the token
    // enters the engine's two-worker pool only for the part that fans out.
    assert!(same_thread);
    assert_eq!(participants.max_active(), 2);
    assert_eq!(participants.thread_count(), 2);
    assert!(
        !participants.thread_ids().contains(&caller),
        "the fan-out ran on the engine's workers, not on the caller"
    );
}
