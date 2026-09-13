use super::*;

#[test]
fn workflow_scope_reuses_executor_across_owned_cached_and_borrowed_operations() {
    for threads in [1, 4] {
        let context = Arc::new(CpuContext::with_threads(threads).unwrap());
        let scope = CpuBackend::from_context(Arc::clone(&context));
        let mut backend = scope.clone();
        let x = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
        let config = DotGeneralConfig {
            lhs_contracting_dims: vec![1],
            rhs_contracting_dims: vec![0],
            lhs_batch_dims: vec![],
            rhs_batch_dims: vec![],
        };
        let mut cache = gemm::GemmAnalysisCache::default();
        let before = context.executor_install_calls_for_test();
        scope.with_execution_scope(|| {
            for _ in 0..3 {
                let doubled = backend.add(&x, &x).unwrap();
                assert_eq!(doubled.as_slice::<f64>().unwrap(), &[2.0, 4.0, 6.0, 8.0]);
                let squared = backend
                    .dot_general_cached(&mut cache, None, &x, &x, &config)
                    .unwrap();
                assert_eq!(squared.as_slice::<f64>().unwrap(), &[7.0, 10.0, 15.0, 22.0]);
                scope.with_execution_scope(|| {
                    backend.with_backend_session(|session| {
                        let result = session.mul(&x, &x).unwrap();
                        assert_eq!(result.as_slice::<f64>().unwrap(), &[1.0, 4.0, 9.0, 16.0]);
                    });
                });
            }
        });
        assert_eq!(context.executor_install_calls_for_test() - before, 1);
        assert_eq!(scope.install(|| 7), 7);
    }
}

#[test]
fn workflow_scope_preserves_recursive_entry_guard_and_recovers_on_unwind() {
    for threads in [1, 4] {
        let scope = CpuBackend::with_threads(threads).unwrap();
        let mut backend = scope.clone();
        scope.with_execution_scope(|| {
            let failed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                backend.with_backend_session(|_| scope.install(|| 0));
            }));
            assert!(failed.is_err());
            assert_eq!(scope.install(|| 9), 9);
        });
        let failed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            scope.with_execution_scope(|| panic!("workflow unwind"));
        }));
        assert!(failed.is_err());
        assert_eq!(scope.with_execution_scope(|| scope.install(|| 11)), 11);
    }
}

#[test]
fn workflow_scope_rejects_a_different_domain_without_leaking_admission() {
    let scope = CpuBackend::with_threads(1).unwrap();
    let other = CpuBackend::with_threads(1).unwrap();
    // Numeric domain IDs are coordinator-local; equal IDs must not share engines.
    assert_eq!(
        scope.execution_info().domain_id(),
        other.execution_info().domain_id()
    );
    scope.with_execution_scope(|| {
        let failed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| other.install(|| 1)));
        assert!(failed.is_err());
        assert_eq!(scope.install(|| 2), 2);
    });
    assert_eq!(other.install(|| 3), 3);
}

#[test]
fn prepared_runtime_workflow_uses_one_executor_entry_for_multiple_operations() {
    use tenferro_runtime::{DType, GraphCompiler, Runtime, TracedTensor};
    for threads in [1, 4] {
        let context = Arc::new(CpuContext::with_threads(threads).unwrap());
        let backend = CpuBackend::from_context(Arc::clone(&context));
        let mut builder = Runtime::builder();
        builder
            .register_engine(crate::runtime_engine_registration(&backend).unwrap())
            .unwrap();
        let runtime = builder.build().unwrap();
        let x = TracedTensor::input_concrete_shape(DType::F64, &[2, 2]).unwrap();
        let y = x
            .dot_general(
                &x,
                DotGeneralConfig {
                    lhs_contracting_dims: vec![1],
                    rhs_contracting_dims: vec![0],
                    lhs_batch_dims: vec![],
                    rhs_batch_dims: vec![],
                },
            )
            .unwrap();
        let z = (&y + &x).unwrap();
        let program = GraphCompiler::new()
            .compile_with_input_specs(&z, &[(&x, DType::F64, &[2, 2])])
            .unwrap();
        let input = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
        let prepared = runtime.prepare_compiled(&program, &[&input]).unwrap();
        let before = context.executor_install_calls_for_test();
        let outputs = runtime.run_prepared(&prepared, &[&input]).unwrap();
        assert_eq!(
            outputs[0].as_slice::<f64>().unwrap(),
            &[8.0, 12.0, 18.0, 26.0]
        );
        assert_eq!(context.executor_install_calls_for_test() - before, 1);
        let before = context.executor_install_calls_for_test();
        backend.with_execution_scope(|| {
            for _ in 0..3 {
                let outputs = runtime.run_compiled(&program, &[&input]).unwrap();
                assert_eq!(
                    outputs[0].as_slice::<f64>().unwrap(),
                    &[8.0, 12.0, 18.0, 26.0]
                );
                let outputs = runtime.run_compiled_values(&program, &[&input]).unwrap();
                assert_eq!(
                    outputs[0].as_tensor().unwrap().as_slice::<f64>().unwrap(),
                    &[8.0, 12.0, 18.0, 26.0]
                );
            }
        });
        assert_eq!(context.executor_install_calls_for_test() - before, 1);
    }
}

#[test]
fn workflow_scope_holds_admission_between_operations() {
    let backend = CpuBackend::with_threads(4).unwrap();
    let other = CpuBackend::with_threads(4).unwrap();
    backend.with_execution_scope(|| {
        assert_eq!(backend.install(|| 1), 1);
        let blocked = std::thread::scope(|threads| {
            threads
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
            "admission must remain held between operation sessions"
        );
        assert_eq!(backend.install(|| 2), 2);
    });
    assert_eq!(other.install(|| 3), 3);
}

#[test]
fn workflow_scope_rejects_backend_entry_from_parallel_operation_children() {
    {
        let backend = CpuBackend::with_threads(4).unwrap();
        backend.with_execution_scope(|| {
            backend.install(|| {
                let attempt = || {
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| backend.install(|| 0)))
                        .is_err()
                };
                let (left, right) = rayon::join(attempt, attempt);
                assert!(left && right);
            });
            assert_eq!(backend.install(|| 7), 7);
        });
    }
}
