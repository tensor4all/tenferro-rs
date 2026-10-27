use super::*;
use crate::Pool;
use crate::{Tensor, TensorRead};
use tenferro_tensor::{TensorDeviceTransfer, TensorElementwise, TensorStructural};

fn pool(threads: usize) -> Pool {
    Pool::builder().threads(threads).build().expect("test pool")
}

fn vector(values: &[f64]) -> Tensor {
    Tensor::from_vec_col_major(vec![values.len()], values.to_vec()).expect("vector")
}

fn engine_error(error: &crate::Error) -> Option<CpuEngineError> {
    let mut source: Option<&(dyn std::error::Error + 'static)> = Some(error);
    while let Some(current) = source {
        if let Some(found) = current.downcast_ref::<CpuEngineError>() {
            return Some(*found);
        }
        source = current.source();
    }
    None
}

#[test]
fn serial_engine_has_no_pool_and_budget_one() {
    let mut engine = CpuEngine::serial();
    assert_eq!(engine.num_threads(), 1);
    assert_eq!(engine.exec().expect("serial token").budget(), 1);
}

#[test]
fn pooled_engine_budget_is_the_pool_size() {
    let mut engine = CpuEngine::new(pool(3));
    assert_eq!(engine.num_threads(), 3);
    assert_eq!(engine.exec().expect("pooled token").budget(), 3);
}

#[test]
fn operation_from_a_worker_of_a_different_pool_runs_on_the_engine() {
    let engine = std::sync::Mutex::new(CpuEngine::new(pool(2)));
    let a = vector(&[1.0, 2.0]);
    let outer = pool(1);
    // A worker of a different pool may call the engine; a worker of its own
    // pool may too, because the token enters the pool in place (no refusal).
    let ok = std::thread::scope(|scope| {
        scope
            .spawn(|| {
                // `Pool` has no `install`; hand the closure to its only worker
                // through a token fork on a one-slot token.
                let token = threadteam::Exec::new(&outer, 1).unwrap();
                let slot = std::sync::Mutex::new(None);
                token.fork(1, |_, _| {
                    *slot.lock().unwrap() = Some(
                        engine
                            .lock()
                            .unwrap()
                            .add_read(TensorRead::from_tensor(&a), TensorRead::from_tensor(&a))
                            .map(|tensor| tensor.as_slice::<f64>().map(<[f64]>::to_vec)),
                    );
                });
                slot.into_inner().unwrap().expect("the job ran")
            })
            .join()
            .expect("foreign pool caller")
    });
    assert_eq!(ok.expect("add").expect("f64"), vec![2.0, 4.0]);
}

#[test]
fn panicking_operation_breaks_the_engine() {
    let mut engine = CpuEngine::serial();
    let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = engine.with_linalg_pool::<()>(|_, buffers| {
            let _scratch = <f64 as crate::buffer_pool::PoolScalar>::pool_acquire_zeroed(buffers, 8);
            panic!("exercise the broken flag");
        });
    }));
    assert!(unwind.is_err());
    assert!(engine.is_broken());
    let a = vector(&[1.0]);
    let error = engine
        .neg_read(TensorRead::from_tensor(&a))
        .expect_err("broken engine refuses work");
    assert_eq!(engine_error(&error), Some(CpuEngineError::Broken));
}

#[test]
fn engines_share_a_plan_cache_only_when_given_one() {
    let shared = CpuPlanCache::new();
    let first = CpuEngine::serial().with_plan_cache(shared.clone());
    let second = CpuEngine::serial().with_plan_cache(shared.clone());
    let private = CpuEngine::serial();
    assert!(first.plan_cache().shares_with(second.plan_cache()));
    assert!(!first.plan_cache().shares_with(private.plan_cache()));
}

#[test]
fn buffer_limit_and_reset_apply_to_this_engine() {
    let mut engine = CpuEngine::serial();
    let a = Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]).expect("matrix");
    let transposed = engine
        .transpose_read(TensorRead::from_tensor(&a), &[1, 0])
        .expect("transpose");
    tenferro_tensor::TensorBuffer::reclaim_buffer(&mut engine, transposed);
    assert!(engine.buffer_pool_len() > 0);
    engine.set_buffer_pool_limit_bytes(0);
    assert_eq!(engine.buffer_pool_limit_bytes(), 0);
    engine.reset_buffer_pool();
    assert_eq!(engine.buffer_pool_len(), 0);
    engine.clear_runtime_caches();
    assert_eq!(engine.runtime_cache_stats().clears, 1);
}

#[test]
fn batch_policy_setter_returns_previous() {
    let mut engine = CpuEngine::serial();
    let previous = engine.set_batch_policy(crate::CpuBatchPolicy::new(
        crate::CpuBatchStrategy::Sequential,
    ));
    assert_eq!(previous.strategy, crate::CpuBatchStrategy::Auto);
    assert_eq!(
        engine.batch_policy().strategy,
        crate::CpuBatchStrategy::Sequential
    );
}

#[test]
fn nested_engines_keep_their_pools_independent_and_match_the_serial_result() {
    use std::collections::HashSet;
    use std::sync::Barrier;

    const LARGE: usize = crate::tests::native_participants::LARGE_NATIVE_TEST_LEN;
    let mut serial = CpuEngine::serial();
    let (_, expected) = serial
        .with_linalg_pool(|exec, _| {
            Ok(crate::tests::native_participants::run_native_map_of_len(
                LARGE,
                false,
                exec.token(),
            ))
        })
        .expect("serial map");

    for k in [2_usize, 3] {
        for m in [2_usize, 4] {
            let barrier = std::sync::Arc::new(Barrier::new(k));
            let handles: Vec<_> = (0..k)
                .map(|_| {
                    let barrier = std::sync::Arc::clone(&barrier);
                    std::thread::spawn(move || {
                        let mut engine = CpuEngine::new(pool(m));
                        barrier.wait();
                        let (participants, output) = engine
                            .with_linalg_pool(|exec, _| {
                                Ok(crate::tests::native_participants::run_native_map_of_len(
                                    LARGE,
                                    true,
                                    exec.token(),
                                ))
                            })
                            .expect("nested map");
                        // The ledger counts live tasks per token: never more
                        // than the engine's budget, at any moment.
                        #[cfg(feature = "ledger")]
                        assert!(engine.take_ledger_max_live() <= m);
                        (participants, output)
                    })
                })
                .collect();

            let mut workers = HashSet::new();
            for handle in handles {
                let (participants, output) = handle.join().expect("nested caller");
                assert_eq!(output, expected, "k={k} m={m}");
                assert!(
                    participants.max_active() <= m,
                    "k={k} m={m}: {} workers ran at once",
                    participants.max_active()
                );
                assert!(
                    participants.thread_count() <= m,
                    "k={k} m={m}: {} workers participated",
                    participants.thread_count()
                );
                workers.extend(participants.thread_ids());
            }
            assert!(
                workers.len() <= k * m,
                "k={k} m={m}: {} tenferro-controlled workers across {k} engines",
                workers.len()
            );
        }
    }
}

#[test]
fn a_small_strided_operation_stays_on_the_calling_thread_without_a_pool_entry() {
    const SMALL: usize = 64;
    let mut serial = CpuEngine::serial();
    let (_, expected) = serial
        .with_linalg_pool(|exec, _| {
            Ok(crate::tests::native_participants::run_native_map_of_len(
                SMALL,
                false,
                exec.token(),
            ))
        })
        .expect("serial map");
    // A serial engine mints a serial token: no root, no pool.
    #[cfg(feature = "ledger")]
    assert!(serial.take_ledger_roots().is_empty());

    let mut engine = CpuEngine::new(pool(4));
    let caller = std::thread::current().id();
    let (participants, actual) = engine
        .with_linalg_pool(|exec, _| {
            Ok(crate::tests::native_participants::run_native_map_of_len(
                SMALL,
                false,
                exec.token(),
            ))
        })
        .expect("pooled map");
    assert_eq!(actual, expected);
    assert_eq!(participants.thread_ids(), vec![caller]);
    assert_eq!(participants.max_active(), 1);
    assert_eq!(participants.thread_count(), 1);
    #[cfg(feature = "ledger")]
    {
        let roots = engine.take_ledger_roots();
        assert!(!roots.is_empty());
        for root in roots {
            assert_eq!(threadteam::ledger::installs(root), 0);
        }
    }
    assert_eq!(std::thread::current().id(), caller);
}

#[cfg(feature = "native")]
#[test]
fn a_small_direct_faer_call_stays_on_the_calling_thread_without_a_pool_entry() {
    let mut engine = CpuEngine::new(pool(4));
    let caller = std::thread::current().id();
    let (same_thread, sequential, width) = engine
        .with_linalg_pool(|exec, _| {
            let width = exec
                .batch_decision(&cpueinsum::BatchQuery::new(1, 8.0))
                .item_threads;
            let observed = exec
                .token()
                .limit(std::num::NonZeroUsize::new(width).unwrap())
                .external(|granted| {
                    (
                        std::thread::current().id() == caller,
                        granted.get() == 1,
                        granted.get(),
                    )
                });
            Ok((observed.0, observed.1, width))
        })
        .expect("direct faer decision");
    assert!(same_thread);
    assert!(sequential);
    assert_eq!(width, 1);
    #[cfg(feature = "ledger")]
    assert_eq!(engine.take_ledger_installs(), 0);
}

#[test]
fn elementwise_families_match_serial_on_pooled_engines_at_one_two_and_four_threads() {
    let a = Tensor::from_vec_col_major(
        vec![4096],
        (0..4096)
            .map(|index| index as f64 * 0.25 - 7.0)
            .collect::<Vec<_>>(),
    )
    .expect("a");
    let b = Tensor::from_vec_col_major(
        vec![4096],
        (0..4096)
            .map(|index| (index % 13) as f64 + 1.5)
            .collect::<Vec<_>>(),
    )
    .expect("b");

    let mut serial = CpuEngine::serial();
    let serial_add = serial
        .add_read(TensorRead::from_tensor(&a), TensorRead::from_tensor(&b))
        .expect("serial add");
    let serial_mul = serial
        .mul_read(TensorRead::from_tensor(&a), TensorRead::from_tensor(&b))
        .expect("serial mul");

    for threads in [1, 2, 4] {
        let mut engine = CpuEngine::new(pool(threads));
        let add = engine
            .add_read(TensorRead::from_tensor(&a), TensorRead::from_tensor(&b))
            .expect("pooled add");
        let mul = engine
            .mul_read(TensorRead::from_tensor(&a), TensorRead::from_tensor(&b))
            .expect("pooled mul");
        assert_eq!(
            add.as_slice::<f64>().unwrap(),
            serial_add.as_slice::<f64>().unwrap()
        );
        assert_eq!(
            mul.as_slice::<f64>().unwrap(),
            serial_mul.as_slice::<f64>().unwrap()
        );
    }
}

#[test]
fn fast_paths_refuse_a_broken_engine_before_touching_storage() {
    let mut engine = CpuEngine::serial();
    let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = engine.with_linalg_pool::<()>(|_, _| panic!("break the engine"));
    }));
    assert!(unwind.is_err());
    assert!(engine.is_broken());

    let input = vector(&[1.0, 2.0]);
    let read = TensorRead::from_tensor(&input);
    let upload = engine
        .upload_host_tensor(read.clone())
        .expect_err("a broken engine refuses upload");
    let download = engine
        .download_to_host(read.clone())
        .expect_err("a broken engine refuses download");
    let reshape = engine
        .reshape_read(read, &[2])
        .expect_err("a broken engine refuses the small reshape fast path");
    for error in [upload, download, reshape] {
        assert_eq!(engine_error(&error), Some(CpuEngineError::Broken));
    }
}
