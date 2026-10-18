use super::*;
use tenferro_tensor::BackendSessionHost;
use tenferro_tensor::DotGeneralConfig;
use tenferro_tensor::TensorRead;

fn input() -> Tensor {
    Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap()
}

#[test]
fn admission_only_callbacks_and_results_need_not_be_send() {
    let mut backend = CpuBackend::with_threads(2).unwrap();
    let caller = std::thread::current().id();
    let value = std::rc::Rc::new(7);
    let result = backend
        .with_backend_session(|_| {
            assert_eq!(std::thread::current().id(), caller);
            std::rc::Rc::clone(&value)
        })
        .unwrap();
    assert!(std::rc::Rc::ptr_eq(&value, &result));
    let result = backend
        .with_execution_scope(|| {
            assert_eq!(std::thread::current().id(), caller);
            std::rc::Rc::clone(&value)
        })
        .unwrap();
    assert!(std::rc::Rc::ptr_eq(&value, &result));
}

#[cfg(target_os = "linux")]
#[test]
fn all_allowed_admission_preserves_narrowed_caller_affinity() {
    let mut backend = CpuBackend::with_threads(1).unwrap();
    let previous = crate::process_cpu_affinity().unwrap();
    let narrowed = CpuSet::new([previous.as_slice()[0]]).unwrap();
    let affinity = crate::affinity::CallerAffinityGuard::enter(Some(&narrowed)).unwrap();
    backend
        .with_backend_session(|_| {
            assert_eq!(crate::process_cpu_affinity().unwrap(), narrowed);
        })
        .unwrap();
    backend
        .with_execution_scope(|| {
            assert_eq!(crate::process_cpu_affinity().unwrap(), narrowed);
        })
        .unwrap();
    assert_eq!(crate::process_cpu_affinity().unwrap(), narrowed);
    affinity.finish().unwrap();
    assert_eq!(crate::process_cpu_affinity().unwrap(), previous);
}

#[test]
fn global_and_foreign_pool_children_do_not_wait_for_parent_admission() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    for foreign in [false, true] {
        // A private admission arbiter keeps this test's admission state
        // independent of other backends the parallel test run creates.
        let mut owner = CpuBackend::with_threads_isolated_arbiter_for_test(1);
        let mut child = owner.clone();
        let (tx, rx) = std::sync::mpsc::channel();
        // Timeout returns from the callback and releases admission even if a
        // regression parks the child, so the test itself does not deadlock.
        let received = owner
            .with_backend_session(|_| {
                let job = move || {
                    let _ = tx.send(child.with_backend_session(|_| ()));
                };
                if foreign {
                    pool.spawn(job);
                } else {
                    rayon::spawn(job);
                }
                rx.recv_timeout(std::time::Duration::from_secs(5))
            })
            .unwrap();
        assert!(matches!(
            received,
            Ok(Err(SessionEntryError::Contended { .. }))
        ));
        // Unrelated idle worker calls are admitted rather than blanket-rejected.
        pool.install(|| owner.with_backend_session(|_| ())).unwrap();
    }
}

#[test]
fn shared_scope_is_admission_only_and_reuses_resources_across_operations() {
    for threads in [1, 4] {
        {
            let context = Arc::new(CpuContext::with_threads(threads).unwrap());
            let owner =
                CpuBackend::from_context_with_buffer_pool_limit(Arc::clone(&context), 1 << 20);
            let mut backend = owner.clone();
            let x = input();
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
                            .unwrap()
                            .is_err());
                        let y = backend
                            .with_backend_session(|__s| {
                                __s.add_read(
                                    TensorRead::from_tensor(&x),
                                    TensorRead::from_tensor(&x),
                                )
                            })
                            .unwrap()
                            .unwrap();
                        assert_eq!(y.as_slice::<f64>().unwrap(), &[2.0, 4.0, 6.0, 8.0]);
                        backend
                            .with_backend_session(|__s| __s.reclaim_buffer(y))
                            .unwrap();
                        let run = |session: &mut dyn BackendSession| {
                            let y = session
                                .mul_read(TensorRead::from_tensor(&x), TensorRead::from_tensor(&x))
                                .unwrap();
                            assert_eq!(y.as_slice::<f64>().unwrap(), &[1.0, 4.0, 9.0, 16.0]);
                            let product = session
                                .dot_general_read(
                                    TensorRead::from_tensor(&x),
                                    TensorRead::from_tensor(&x),
                                    &config,
                                )
                                .unwrap();
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
                            backend
                                .with_backend_session_cached(&mut cache, run)
                                .unwrap();
                        } else {
                            backend.with_backend_session(run).unwrap();
                        }
                        assert_eq!(
                            backend.install(|| std::thread::current().id()).unwrap(),
                            thread
                        );
                    }
                })
                .unwrap();
            assert_eq!(
                backend
                    .with_backend_session(|__s| __s
                        .add_read(TensorRead::from_tensor(&x), TensorRead::from_tensor(&x)))
                    .unwrap()
                    .unwrap()
                    .as_slice::<f64>()
                    .unwrap(),
                &[2.0, 4.0, 6.0, 8.0]
            );
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
            let error = match other.execution_admission() {
                Ok(_) => panic!("a backend outside the active scope must be rejected"),
                Err(error) => error,
            };
            assert!(matches!(
                error,
                tenferro_tensor::SessionEntryError::IncompatibleContext { .. }
            ));
            let mut ran = false;
            let invalid_session = other.with_backend_session(|_| ran = true);
            assert!(matches!(
                invalid_session,
                Err(tenferro_tensor::SessionEntryError::IncompatibleContext { .. })
            ));
            assert!(!ran);
            assert!(owner.with_execution_scope(|| ()).is_err());
            assert_eq!(
                backend
                    .with_backend_session(|__s| __s
                        .add_read(TensorRead::from_tensor(&x), TensorRead::from_tensor(&x)))
                    .unwrap()
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
        .unwrap()
        .is_ok());
}

#[test]
fn shared_scope_preserves_borrowed_session_reentry_guard_and_unwind_recovery() {
    let owner = CpuBackend::with_threads(1).unwrap();
    let mut backend = owner.clone();
    let mut reentrant = owner.clone();
    let x = input();
    let nested = owner
        .with_execution_scope(|| {
            backend
                .with_backend_session(|_| {
                    reentrant.with_backend_session(|__s| {
                        __s.add_read(TensorRead::from_tensor(&x), TensorRead::from_tensor(&x))
                    })
                })
                .unwrap()
        })
        .unwrap();
    assert!(
        matches!(
            nested,
            Err(tenferro_tensor::SessionEntryError::Reentered { .. })
        ),
        "{nested:?}"
    );
    let failed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        owner
            .with_execution_scope(|| {
                backend
                    .with_backend_session(|_| panic!("session callback panic"))
                    .unwrap();
            })
            .unwrap();
    }));
    assert_eq!(panic_message(failed.unwrap_err()), "session callback panic");
    assert!(backend
        .with_backend_session(
            |__s| __s.add_read(TensorRead::from_tensor(&x), TensorRead::from_tensor(&x))
        )
        .unwrap()
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
                            .unwrap()
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
        .unwrap()
        .is_ok());
}

#[test]
#[cfg(feature = "blas")]
fn shared_scope_holds_blas_exclusion_until_callback_unwinds() {
    let owner = CpuBackend::with_threads(1).unwrap();
    let other = CpuBackend::with_threads(1).unwrap();
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
    assert_eq!(other.install(|| 23).unwrap(), 23);
}
