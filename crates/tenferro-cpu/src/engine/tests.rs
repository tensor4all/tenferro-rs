use super::*;
use crate::{Tensor, TensorRead};
use tenferro_tensor::{TensorElementwise, TensorStructural};

fn pool(threads: usize) -> rayon::ThreadPool {
    thread_pool_builder()
        .num_threads(threads)
        .build()
        .expect("test pool")
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
    assert!(engine.pool().is_none());
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
fn operation_from_own_pool_worker_is_rejected() {
    let mut engine = CpuEngine::new(pool(2));
    let a = vector(&[1.0, 2.0]);
    let pool = thread_pool_builder()
        .num_threads(1)
        .build()
        .expect("outer pool");
    // A worker of a different pool may call the engine.
    let ok = pool.install(|| {
        engine
            .add_read(TensorRead::from_tensor(&a), TensorRead::from_tensor(&a))
            .map(|tensor| tensor.as_slice::<f64>().map(<[f64]>::to_vec))
    });
    assert_eq!(
        ok.expect("foreign pool caller").expect("f64"),
        vec![2.0, 4.0]
    );

    // A worker of the engine's own pool may not: every operation and `exec`
    // start with this entry check.
    let engine = &engine;
    let error = engine
        .pool()
        .expect("pool")
        .install(|| engine.check_entry("test"))
        .expect_err("own-pool caller must be rejected");
    assert_eq!(engine_error(&error), Some(CpuEngineError::CallerInOwnPool));
    assert!(engine.check_entry("test").is_ok());
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

    const LARGE: usize = 1 << 17;
    let mut serial = CpuEngine::serial();
    let (_, expected) = serial
        .with_linalg_pool(|context, _| {
            Ok(context.with_native_parallelism(|| {
                crate::tests::native_participants::run_native_map_of_len(LARGE, false)
            }))
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
                            .with_linalg_pool(|context, _| {
                                Ok(context.with_native_parallelism(|| {
                                    crate::tests::native_participants::run_native_map_of_len(
                                        LARGE, true,
                                    )
                                }))
                            })
                            .expect("nested map");
                        assert_eq!(engine.pool_stats().entries, 0);
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
        .with_linalg_pool(|context, _| {
            Ok(context.with_native_parallelism(|| {
                crate::tests::native_participants::run_native_map_of_len(SMALL, false)
            }))
        })
        .expect("serial map");
    assert_eq!(serial.pool_stats().entries, 0);

    let mut engine = CpuEngine::new(pool(4));
    let caller = std::thread::current().id();
    let (participants, actual) = engine
        .with_linalg_pool(|context, _| {
            Ok(context.with_native_parallelism(|| {
                crate::tests::native_participants::run_native_map_of_len(SMALL, false)
            }))
        })
        .expect("pooled map");
    assert_eq!(actual, expected);
    assert_eq!(participants.thread_ids(), vec![caller]);
    assert_eq!(participants.max_active(), 1);
    assert_eq!(participants.thread_count(), 1);
    assert_eq!(engine.pool_stats().entries, 0);
    assert_eq!(std::thread::current().id(), caller);
}

#[cfg(feature = "native")]
#[test]
fn a_small_direct_faer_call_stays_on_the_calling_thread_without_a_pool_entry() {
    let mut engine = CpuEngine::new(pool(4));
    let caller = std::thread::current().id();
    let (same_thread, sequential, width) = engine
        .with_linalg_pool(|context, _| {
            let width = context
                .contraction_exec()?
                .batch_decision(&cpueinsum::BatchQuery::new(1, 8.0))
                .map_err(|error| crate::Error::backend_source("test_faer_width", error))?
                .item_threads;
            let observed = context.with_faer_width(width, |parallel| {
                (
                    std::thread::current().id() == caller,
                    parallel == crate::faer::Par::Seq,
                )
            });
            Ok((observed.0, observed.1, width))
        })
        .expect("direct faer decision");
    assert!(same_thread);
    assert!(sequential);
    assert_eq!(width, 1);
    assert_eq!(engine.pool_stats().entries, 0);
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
