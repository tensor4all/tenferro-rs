use std::num::NonZeroUsize;

use tenferro_ad::{AdContext, AdTransformCacheLimits, EagerRuntime, EagerTensor, Tensor};
use tenferro_cpu::CpuBackend;
use tenferro_runtime::{GraphCompiler, TracedTensor};

use crate::support::{cpu_runtime, run_compiled_one};

#[test]
fn compiler_clear_caches_clears_extension_entries() {
    let mut compiler = GraphCompiler::new();

    let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    let y = (&x + &x).unwrap();
    let _ = compiler.compile(&y).expect("compile");

    let before = compiler.cache_stats();
    assert_eq!(before.entries, 0);

    compiler.clear_caches();

    let after = compiler.cache_stats();
    assert_eq!(after.entries, 0);
}

#[test]
fn executor_clear_caches_leaves_no_extension_entries_without_extensions() {
    let mut compiler = GraphCompiler::new();
    let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    let y = (&x + &x).unwrap();
    let program = compiler.compile(&y).expect("compile");
    let executor = cpu_runtime();

    let _ = run_compiled_one(&executor, &program, &[]).expect("run");

    let before = executor.cache_stats().expect("cache stats");
    assert_eq!(before.extensions.entries, 0);

    executor.clear_caches().expect("clear caches");

    let after = executor.cache_stats().expect("cache stats");
    assert_eq!(after.extensions.entries, 0);
}

#[test]
fn executor_clear_caches_clears_executor_owned_runtime_caches() {
    let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    let y = (&x + &x).unwrap();
    let mut compiler = GraphCompiler::new();
    let program = compiler.compile(&y).expect("compile");
    let executor = cpu_runtime();

    let _ = run_compiled_one(&executor, &program, &[]).expect("run");
    executor.clear_caches().expect("clear caches");

    let after = executor.cache_stats().expect("cache stats");
    assert_eq!(after.extensions.entries, 0);
    assert_eq!(after.engines.entries, 0);
    assert_eq!(after.prepared_plans.entries, 0);
}

#[test]
fn ad_context_transform_cache_limits_stats_and_clear_are_public() {
    let ad = AdContext::builder().build().unwrap();
    let default_limits = ad.ad_transform_cache_limits().unwrap();
    assert!(default_limits.max_entries().get() > 0);
    assert!(default_limits.max_retained_bytes().is_some());

    let limits = AdTransformCacheLimits::new(NonZeroUsize::new(1).unwrap())
        .with_max_retained_bytes(NonZeroUsize::new(1024).unwrap());
    ad.set_ad_transform_cache_limits(limits).unwrap();
    assert_eq!(ad.ad_transform_cache_limits().unwrap(), limits);

    let stats = ad.ad_transform_cache_stats().unwrap();
    assert_eq!(stats.entries, 0);
    assert_eq!(stats.retained_bytes, 0);

    ad.clear_ad_transform_caches().unwrap();
    assert_eq!(ad.ad_transform_cache_stats().unwrap().entries, 0);
}

#[test]
fn eager_runtime_built_from_ad_context_uses_shared_transform_cache() {
    let ad = AdContext::builder().build().unwrap();
    let ctx = EagerRuntime::with_cpu_backend_and_ad_context(CpuBackend::new(), &ad).unwrap();
    let x = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![2], vec![2.0_f64, 3.0]).unwrap(),
        ctx.clone(),
    )
    .unwrap();
    let seed = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 1.0]).unwrap(),
        ctx.clone(),
    )
    .unwrap();
    let y = x.mul(&x).unwrap();

    assert_eq!(ad.ad_transform_cache_stats().unwrap().entries, 0);
    let _ = ctx.vjp(&y, &x, &seed).unwrap();
    assert!(ad.ad_transform_cache_stats().unwrap().entries > 0);

    ad.clear_ad_transform_caches().unwrap();
    assert_eq!(ctx.cache_stats().unwrap().ad_transforms.entries, 0);
}

#[test]
fn ad_transform_cache_entry_limit_evicts_lru_entries() {
    let ad = AdContext::builder().build().unwrap();
    ad.set_ad_transform_cache_limits(AdTransformCacheLimits::new(NonZeroUsize::new(1).unwrap()))
        .unwrap();
    let ctx = EagerRuntime::with_cpu_backend_and_ad_context(CpuBackend::new(), &ad).unwrap();

    let x0 = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![1], vec![2.0_f64]).unwrap(),
        ctx.clone(),
    )
    .unwrap();
    let x1 = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![1], vec![4.0_f64]).unwrap(),
        ctx.clone(),
    )
    .unwrap();
    let seed = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![1], vec![1.0_f64]).unwrap(),
        ctx.clone(),
    )
    .unwrap();

    let _ = ctx.vjp(&x0.mul(&x0).unwrap(), &x0, &seed).unwrap();
    let _ = ctx.vjp(&x1.mul(&x1).unwrap(), &x1, &seed).unwrap();

    assert_eq!(ad.ad_transform_cache_stats().unwrap().entries, 1);
}

#[test]
fn ad_context_traced_vjp_reuses_transform_cache() {
    let ad = AdContext::builder().build().unwrap();
    let x = TracedTensor::from_vec_col_major(vec![2], vec![2.0_f64, 3.0]).unwrap();
    let y = (&x * &x).unwrap();
    let seed = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 1.0]).unwrap();

    assert_eq!(ad.ad_transform_cache_stats().unwrap().entries, 0);
    let _ = ad.vjp(&y, &x, &seed).unwrap();
    let after_first = ad.ad_transform_cache_stats().unwrap();
    assert!(after_first.entries > 0);
    assert!(after_first.retained_bytes > 0);
    assert_eq!(after_first.hits, 0);
    assert_eq!(after_first.misses, 1);

    let _ = ad.vjp(&y, &x, &seed).unwrap();
    let after_second = ad.ad_transform_cache_stats().unwrap();
    assert_eq!(after_second.entries, after_first.entries);
    assert_eq!(after_second.retained_bytes, after_first.retained_bytes);
    assert_eq!(after_second.hits, after_first.hits + 1);
    assert_eq!(after_second.misses, after_first.misses);

    ad.clear_ad_transform_caches().unwrap();
    let after_clear = ad.ad_transform_cache_stats().unwrap();
    assert_eq!(after_clear.entries, 0);
    assert_eq!(after_clear.retained_bytes, 0);
    assert_eq!(after_clear.clears, after_second.clears + 1);
}

#[test]
fn eager_backward_shape_churn_keeps_transform_cache_shape_specific() {
    struct Fixture {
        x: EagerTensor,
        loss: EagerTensor,
    }

    fn tensor(shape: Vec<usize>, seed: usize) -> Tensor {
        let len = shape.iter().product();
        let data = (0..len)
            .map(|index| ((index * 23 + seed * 41 + 17) % 997) as f64 / 997.0 - 0.5)
            .collect();
        Tensor::from_vec_col_major(shape, data).unwrap()
    }

    fn fixture(ctx: &std::sync::Arc<EagerRuntime>, shape: Vec<usize>, seed: usize) -> Fixture {
        let x = EagerTensor::requires_grad_in(tensor(shape.clone(), seed), ctx.clone()).unwrap();
        let weight = EagerTensor::from_tensor_in(tensor(shape, seed + 1000), ctx.clone()).unwrap();
        let loss = x.mul(&weight).unwrap().mul(&x).unwrap();
        let axes: Vec<_> = (0..loss.shape().len()).collect();
        let loss = loss.reduce_sum(Some(&axes)).unwrap();
        Fixture { x, loss }
    }

    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();
    let fixtures = [
        fixture(&ctx, vec![8, 4, 10], 0),
        fixture(&ctx, vec![10, 4, 12], 1),
    ];

    for fixture in &fixtures {
        ctx.clear_grads().unwrap();
        fixture.loss.backward().unwrap();
        let grad = fixture.x.grad().unwrap().unwrap();
        assert_eq!(grad.shape(), fixture.x.shape());
    }
}
