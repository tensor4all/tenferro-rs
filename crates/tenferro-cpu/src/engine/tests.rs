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
