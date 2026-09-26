use super::*;
use tenferro_tensor::BackendSessionHost;
use tenferro_tensor::TensorRead;

fn input() -> Tensor {
    Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap()
}

#[test]
fn shared_scope_installs_once_and_reuses_resources_across_operations() {
    let kinds = &[
        #[cfg(feature = "cpu-faer")]
        CpuBackendKind::Faer,
        #[cfg(feature = "cpu-blas")]
        CpuBackendKind::Blas,
    ];
    for &kind in kinds {
        for threads in [1, 4] {
            let context = Arc::new(CpuContext::with_threads(threads).unwrap());
            let owner = CpuBackend::from_context_with_buffer_pool_limit_and_kind(
                Arc::clone(&context),
                1 << 20,
                kind,
            );
            let mut backend = owner.clone();
            let x = input();
            let before = context.executor_install_calls_for_test();
            owner
                .with_execution_scope(|| {
                    let thread = std::thread::current().id();
                    let wrong = Tensor::from_vec_col_major(vec![3], vec![1.0_f64; 3]).unwrap();
                    let mut cache = gemm::GemmAnalysisCache::default();
                    let config = DotGeneralConfig {
                        lhs_contracting_dims: [1].as_slice().into(),
                        rhs_contracting_dims: [0].as_slice().into(),
                        lhs_batch_dims: [].as_slice().into(),
                        rhs_batch_dims: [].as_slice().into(),
                    };
                    for iteration in 0..16 {
                        assert!(backend
                            .with_backend_session(|__s| __s.add_read(
                                TensorRead::from_tensor(&x),
                                TensorRead::from_tensor(&wrong)
                            ))
                            .is_err());
                        let y = backend
                            .with_backend_session(|__s| {
                                __s.add_read(
                                    TensorRead::from_tensor(&x),
                                    TensorRead::from_tensor(&x),
                                )
                            })
                            .unwrap();
                        assert_eq!(y.as_slice::<f64>().unwrap(), &[2.0, 4.0, 6.0, 8.0]);
                        backend.reclaim_buffer(y);
                        let run = |session: &mut dyn BackendSession| {
                            let y = session
                                .mul_read(TensorRead::from_tensor(&x), TensorRead::from_tensor(&x))
                                .unwrap();
                            assert_eq!(y.as_slice::<f64>().unwrap(), &[1.0, 4.0, 9.0, 16.0]);
                            let product = session.dot_general(&x, &x, &config).unwrap();
                            assert_eq!(
                                product.as_slice::<f64>().unwrap(),
                                &[7.0, 10.0, 15.0, 22.0]
                            );
                            crate::with_cpu_exec_session(session, |cpu| {
                                assert!(cpu.entered.is_some());
                                cpu.with_linalg_pool(|entered, _| {
                                    assert_eq!(entered.thread_budget().get(), threads);
                                    assert_eq!(std::thread::current().id(), thread);
                                    Ok(())
                                })
                                .unwrap();
                            })
                            .unwrap();
                        };
                        if iteration % 2 == 0 {
                            backend.with_backend_session_cached(&mut cache, run);
                        } else {
                            backend.with_backend_session(run);
                        }
                        assert_eq!(backend.install(|| std::thread::current().id()), thread);
                    }
                })
                .unwrap();
            assert_eq!(context.executor_install_calls_for_test() - before, 1);
            assert_eq!(
                backend
                    .with_backend_session(|__s| __s
                        .add_read(TensorRead::from_tensor(&x), TensorRead::from_tensor(&x)))
                    .unwrap()
                    .as_slice::<f64>()
                    .unwrap(),
                &[2.0, 4.0, 6.0, 8.0]
            );
            assert_eq!(context.executor_install_calls_for_test() - before, 2);
        }
    }
}

#[test]
fn shared_scope_rejects_wrong_witness_and_nested_scopes() {
    let owner = CpuBackend::with_threads(1).unwrap();
    let mut other = CpuBackend::with_threads(1).unwrap();
    let mut backend = owner.clone();
    let x = input();
    owner
        .with_execution_scope(|| {
            // The deleted one-shot entry was the fallible admission path; pin
            // that policy directly now that the session route (below) retains
            // the infallible panic boundary instead.
            let error = match other.execution_admission() {
                Ok(_) => panic!("a backend outside the active scope must be rejected"),
                Err(error) => error,
            };
            assert!(matches!(error, crate::Error::RuntimeState { .. }));
            let invalid_session = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                other.with_backend_session(|_| ());
            }));
            assert!(invalid_session.is_err());
            assert!(owner.with_execution_scope(|| ()).is_err());
            #[cfg(all(feature = "cpu-blas", feature = "cpu-faer"))]
            {
                let different_kind = if owner.kind() == CpuBackendKind::Blas {
                    CpuBackendKind::Faer
                } else {
                    CpuBackendKind::Blas
                };
                let mut different_provider =
                    CpuBackend::with_threads_and_kind(1, different_kind).unwrap();
                assert!(matches!(
                    different_provider.execution_admission(),
                    Err(crate::Error::RuntimeState { .. })
                ));
            }
            assert_eq!(
                backend
                    .with_backend_session(|__s| __s
                        .add_read(TensorRead::from_tensor(&x), TensorRead::from_tensor(&x)))
                    .unwrap()
                    .as_slice::<f64>()
                    .unwrap(),
                &[2.0, 4.0, 6.0, 8.0]
            );
        })
        .unwrap();
    assert!(other
        .with_backend_session(
            |__s| __s.add_read(TensorRead::from_tensor(&x), TensorRead::from_tensor(&x))
        )
        .is_ok());
}

#[test]
fn shared_scope_preserves_borrowed_session_reentry_guard_and_unwind_recovery() {
    let owner = CpuBackend::with_threads(1).unwrap();
    let mut backend = owner.clone();
    let mut reentrant = owner.clone();
    let x = input();
    let failed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        owner
            .with_execution_scope(|| {
                backend.with_backend_session(|_| {
                    reentrant
                        .with_backend_session(|__s| {
                            __s.add_read(TensorRead::from_tensor(&x), TensorRead::from_tensor(&x))
                        })
                        .unwrap()
                });
            })
            .unwrap();
    }));
    assert!(panic_message(failed.unwrap_err()).contains(crate::arbiter::BACKEND_REENTRY_PANIC));
    assert!(backend
        .with_backend_session(
            |__s| __s.add_read(TensorRead::from_tensor(&x), TensorRead::from_tensor(&x))
        )
        .is_ok());
    let returned = owner
        .with_execution_scope(|| -> std::result::Result<(), &'static str> { Err("callback error") })
        .unwrap();
    assert_eq!(returned, Err("callback error"));
    assert_eq!(owner.with_execution_scope(|| 17).unwrap(), 17);
}

#[test]
fn shared_scope_does_not_admit_child_worker_backend_reentry() {
    let owner = CpuBackend::with_threads(4).unwrap();
    let mut backend = owner.clone();
    owner
        .with_execution_scope(|| {
            let (send, receive) = std::sync::mpsc::channel();
            let worker = &mut backend;
            rayon::scope(move |scope| {
                scope.spawn(move |_| {
                    let failed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        worker
                            .with_backend_session(|__s| {
                                __s.add_read(
                                    TensorRead::from_tensor(&input()),
                                    TensorRead::from_tensor(&input()),
                                )
                            })
                            .unwrap();
                    }));
                    send.send(failed.is_err()).unwrap();
                });
                // Blocking the callback thread forces the child onto another worker.
                assert!(receive.recv_timeout(Duration::from_secs(10)).unwrap());
            });
        })
        .unwrap();
    assert!(backend
        .with_backend_session(|__s| __s.add_read(
            TensorRead::from_tensor(&input()),
            TensorRead::from_tensor(&input())
        ))
        .is_ok());
}

#[test]
#[cfg(feature = "cpu-blas")]
fn shared_scope_holds_blas_exclusion_until_callback_unwinds() {
    let owner = CpuBackend::with_threads_and_kind(1, CpuBackendKind::Blas).unwrap();
    let other = CpuBackend::with_threads_and_kind(1, CpuBackendKind::Blas).unwrap();
    let failed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        owner
            .with_execution_scope(|| {
                std::thread::scope(|scope| {
                    assert!(scope
                        .spawn(|| other
                            .try_acquire_execution_permit_for_test()
                            .unwrap()
                            .is_none())
                        .join()
                        .unwrap());
                });
                panic!("scope callback panic");
            })
            .unwrap();
    }));
    assert_eq!(panic_message(failed.unwrap_err()), "scope callback panic");
    assert_eq!(other.install(|| 23), 23);
}
