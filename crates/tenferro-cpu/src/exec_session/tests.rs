use super::*;

#[test]
fn integer_dot_is_rejected_without_mutating_output() {
    use tenferro_tensor::BackendSessionHost;
    let lhs = Tensor::from_vec_col_major(vec![2], vec![2_i64, 3]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![2], vec![5_i64, 7]).unwrap();
    let config = DotGeneralConfig {
        lhs_contracting_dims: [].as_slice().into(),
        rhs_contracting_dims: [].as_slice().into(),
        lhs_batch_dims: [0].as_slice().into(),
        rhs_batch_dims: [0].as_slice().into(),
    };
    let mut backend = crate::CpuBackend::with_threads(1).unwrap();
    assert!(backend
        .with_backend_session(|session| session.dot_general_read(
            TensorRead::from_tensor(&lhs),
            TensorRead::from_tensor(&rhs),
            &config
        ))
        .unwrap()
        .is_err());
    let mut target = Tensor::from_vec_col_major(vec![2], vec![1_i64, 2]).unwrap();
    assert!(backend
        .with_backend_session(|session| session.dot_general_read_into(
            TensorRead::from_tensor(&lhs),
            TensorRead::from_tensor(&rhs),
            &config,
            TensorWrite::from_tensor(&mut target)
        ))
        .unwrap()
        .is_err());
    assert_eq!(target.as_slice::<i64>().unwrap(), [1, 2]);
}

#[test]
fn backend_controls_clear_and_bound_owned_nary_workspace() {
    use tenferro_tensor::BackendSessionHost;
    let mut backend = crate::CpuBackend::with_threads(1).unwrap();
    let baseline = backend.runtime_cache_stats().unwrap();
    let populate = |backend: &mut crate::CpuBackend| {
        backend
            .with_backend_session(|session| {
                crate::with_cpu_exec_session(session, |cpu| {
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
    let filled = backend.runtime_cache_stats().unwrap();
    assert_eq!(filled.entries, baseline.entries + 1);
    assert_eq!(filled.retained_bytes, baseline.retained_bytes + 128);
    backend.reset_buffer_pool().unwrap();
    assert_eq!(
        backend.runtime_cache_stats().unwrap().retained_bytes,
        baseline.retained_bytes
    );
    populate(&mut backend);
    backend.set_buffer_pool_limit_bytes(0).unwrap();
    assert_eq!(
        backend.runtime_cache_stats().unwrap().retained_bytes,
        baseline.retained_bytes
    );
    backend.set_buffer_pool_limit_bytes(1024).unwrap();
    populate(&mut backend);
    backend.clear_runtime_caches().unwrap();
    assert_eq!(
        backend.runtime_cache_stats().unwrap().retained_bytes,
        baseline.retained_bytes
    );
}

#[test]
fn linalg_resource_callback_and_result_stay_on_the_caller_without_send() {
    use std::rc::Rc;
    use tenferro_tensor::BackendSessionHost;

    let caller = std::thread::current().id();
    let value = Rc::new(37);
    let mut backend = crate::CpuBackend::with_threads(2).unwrap();
    let result = backend
        .with_backend_session(|session| {
            crate::with_cpu_exec_session(session, |cpu| {
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
fn native_operation_enters_the_selected_rayon_executor() {
    let mut backend = crate::CpuBackend::with_threads(2).unwrap();

    assert!(rayon::current_thread_index().is_none());
    let (worker, pool_size, participants) = crate::tests::with_cpu_session(&mut backend, |cpu| {
        cpu.with_linalg_pool(|context, _| {
            context.with_native_parallelism(|| {
                Ok((
                    rayon::current_thread_index(),
                    rayon::current_num_threads(),
                    crate::tests::native_participants::run_unscoped_native_map(true),
                ))
            })
        })
    })
    .unwrap();
    assert!(matches!(worker, Some(0 | 1)));
    assert_eq!(pool_size, 2);
    assert_eq!(participants.max_active(), 2);
    assert_eq!(participants.thread_count(), 2);
    assert!(rayon::current_thread_index().is_none());
}
