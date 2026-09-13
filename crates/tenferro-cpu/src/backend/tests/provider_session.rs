use super::*;

#[test]
fn blas_sessions_reuse_entered_context_for_native_gemm_and_linalg() {
    for threads in [1, 4] {
        let context = Arc::new(CpuContext::with_threads(threads).unwrap());
        let mut backend = CpuBackend::from_context(Arc::clone(&context));
        assert_eq!(backend.kind(), CpuBackendKind::Blas);
        let mut cache = gemm::GemmAnalysisCache::default();
        let lhs = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
        let rhs = Tensor::from_vec_col_major(vec![2, 2], vec![5.0_f64, 6.0, 7.0, 8.0]).unwrap();
        let config = DotGeneralConfig {
            lhs_contracting_dims: vec![1],
            rhs_contracting_dims: vec![0],
            lhs_batch_dims: vec![],
            rhs_batch_dims: vec![],
        };
        for cached in [false, true] {
            let before = context.executor_install_calls_for_test();
            let run = |session: &mut dyn BackendSession| {
                let session_thread = std::thread::current().id();
                if threads > 1 {
                    assert!(rayon::current_thread_index().is_some());
                    assert_eq!(rayon::current_num_threads(), threads);
                }
                for _ in 0..3 {
                    crate::with_cpu_exec_session(session, |cpu| {
                        let entered = cpu
                            .entered
                            .as_ref()
                            .expect("BLAS session must already be entered");
                        assert_eq!(entered.domain_id(), cpu.domain_id());
                        cpu.with_linalg_pool(|context, _| {
                            assert_eq!(std::thread::current().id(), session_thread);
                            assert_eq!(
                                context.parallel_mode(),
                                if threads == 1 {
                                    crate::ParallelMode::Sequential
                                } else {
                                    crate::ParallelMode::Inner
                                }
                            );
                            Ok(())
                        })
                        .unwrap();
                    })
                    .unwrap();
                    let sum = session.add(&lhs, &rhs).unwrap();
                    assert_eq!(sum.as_slice::<f64>().unwrap(), &[6.0, 8.0, 10.0, 12.0]);
                    let product = session.dot_general(&lhs, &rhs, &config).unwrap();
                    assert_eq!(
                        product.as_slice::<f64>().unwrap(),
                        &[23.0, 34.0, 31.0, 46.0]
                    );
                }
            };
            if cached {
                backend.with_backend_session_cached(&mut cache, run);
            } else {
                backend.with_backend_session(run);
            }
            assert_eq!(context.executor_install_calls_for_test() - before, 1);
        }
    }
}

#[test]
fn blas_session_preserves_exclusion_and_recovers_after_callback_panic() {
    for threads in [1, 4] {
        let mut backend = CpuBackend::with_threads_and_kind(threads, CpuBackendKind::Blas).unwrap();
        let other = CpuBackend::with_threads_and_kind(threads, CpuBackendKind::Blas).unwrap();
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            backend.with_backend_session(|_| {
                let blocked = std::thread::scope(|scope| {
                    scope
                        .spawn(|| {
                            other
                                .try_acquire_execution_permit_for_test()
                                .unwrap()
                                .is_none()
                        })
                        .join()
                        .unwrap()
                });
                assert!(
                    blocked,
                    "provider exclusion must cover the whole session callback"
                );
                panic!("session callback failure");
            });
        }));
        assert_eq!(
            panic_message(outcome.unwrap_err()),
            "session callback failure"
        );
        backend.with_backend_session(|session| {
            crate::with_cpu_exec_session(session, |cpu| assert!(cpu.entered.is_some())).unwrap();
        });
        // Successful admission to an independent backend proves that unwind released the permit.
        assert_eq!(other.install(|| 17_u32), 17);
    }
}
