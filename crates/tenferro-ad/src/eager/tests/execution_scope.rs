use super::*;

#[test]
fn eager_workflow_scope_covers_forward_and_all_derivative_entry_points() {
    for threads in [1, 4] {
        let ctx =
            EagerRuntime::with_cpu_backend(CpuBackend::with_threads(threads).unwrap()).unwrap();
        let input = |data| {
            EagerTensor::requires_grad_in(
                Tensor::from_vec_col_major(vec![2], data).unwrap(),
                Arc::clone(&ctx),
            )
            .unwrap()
        };
        let x = input(vec![2.0_f64, 3.0]);
        let seed = input(vec![1.0_f64, 2.0]);
        ctx.with_execution_scope(|| {
            let y = x.mul(&x).unwrap();
            for derivative in [
                ctx.vjp(&y, &x, &seed).unwrap(),
                ctx.jvp(&y, &x, &seed).unwrap(),
            ] {
                assert_eq!(
                    derivative.to_tensor().unwrap().as_slice::<f64>().unwrap(),
                    &[4.0, 12.0]
                );
            }
            let loss = y.reduce_sum(Some(&[0])).unwrap();
            let gradient = ctx.grad(&loss, &x).unwrap();
            assert_eq!(
                gradient.to_tensor().unwrap().as_slice::<f64>().unwrap(),
                &[4.0, 6.0]
            );
            ctx.with_execution_scope(|| loss.backward().unwrap())
                .unwrap();
            assert_eq!(
                x.grad().unwrap().unwrap().as_slice::<f64>().unwrap(),
                &[4.0, 6.0]
            );
        })
        .unwrap();
        // Automatic derivative scopes also work without a user workflow scope.
        let other = input(vec![5.0_f64, 7.0]);
        other.mul(&other).unwrap().backward_with(&seed).unwrap();
        assert_eq!(
            other.grad().unwrap().unwrap().as_slice::<f64>().unwrap(),
            &[10.0, 28.0]
        );
    }
}

#[test]
fn eager_workflow_scope_releases_backend_between_operations_and_after_errors() {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::with_threads(4).unwrap()).unwrap();
    let error = ctx
        .with_execution_scope(|| -> std::result::Result<(), &'static str> {
            ctx.with_execution_session(|_| ()).unwrap();
            Err("callback failure")
        })
        .unwrap();
    assert_eq!(error, Err("callback failure"));
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        ctx.with_execution_scope(|| panic!("workflow failure"))
            .unwrap();
    }));
    assert!(panic.is_err());
    assert_eq!(
        ctx.with_execution_scope(|| ctx.with_execution_session(|_| 7).unwrap())
            .unwrap(),
        7
    );
}

#[test]
fn eager_workflow_scope_rejects_entry_from_an_active_borrowed_session() {
    for threads in [1, 4] {
        let ctx =
            EagerRuntime::with_cpu_backend(CpuBackend::with_threads(threads).unwrap()).unwrap();
        ctx.with_execution_session(|_| {
            let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                ctx.with_execution_scope(|| ()).unwrap();
            }));
            assert!(panic.is_err());
        })
        .unwrap();
        assert_eq!(ctx.with_execution_scope(|| 3).unwrap(), 3);
    }
}

#[test]
fn eager_workflow_and_ordinary_calls_share_a_runtime_without_lock_inversion() {
    for threads in [1, 4] {
        let ctx =
            EagerRuntime::with_cpu_backend(CpuBackend::with_threads(threads).unwrap()).unwrap();
        let x = EagerTensor::from_tensor_in(
            Tensor::from_vec_col_major(vec![2], vec![2.0_f64, 3.0]).unwrap(),
            Arc::clone(&ctx),
        )
        .unwrap();
        let (entered_tx, entered_rx) = std::sync::mpsc::channel();
        let (attempt_tx, attempt_rx) = std::sync::mpsc::channel();
        std::thread::scope(|workers| {
            let x = &x;
            let ordinary = workers.spawn(move || {
                entered_rx.recv().unwrap();
                attempt_tx.send(()).unwrap();
                for _ in 0..20 {
                    let result = x.mul(x).unwrap();
                    assert_eq!(
                        result.to_tensor().unwrap().as_slice::<f64>().unwrap(),
                        &[4.0, 9.0]
                    );
                }
            });
            ctx.with_execution_scope(move || {
                entered_tx.send(()).unwrap();
                attempt_rx.recv().unwrap();
                for _ in 0..20 {
                    std::thread::yield_now();
                    let result = x.add(x).unwrap();
                    assert_eq!(
                        result.to_tensor().unwrap().as_slice::<f64>().unwrap(),
                        &[4.0, 6.0]
                    );
                }
            })
            .unwrap();
            ordinary.join().unwrap();
        });
    }
}
