use std::sync::Arc;

use crate::{EagerRuntime, EagerTensor, Tensor};
use tenferro_cpu::CpuBackend;

#[test]
fn backward_shares_one_vjp_across_leaves_and_separates_active_masks() {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::with_threads(1).unwrap()).unwrap();
    let leaf = |values| {
        EagerTensor::requires_grad_in(
            Tensor::from_vec_col_major(vec![2], values).unwrap(),
            Arc::clone(&ctx),
        )
        .unwrap()
    };
    let x = leaf(vec![2.0_f64, 3.0]);
    let y = leaf(vec![5.0_f64, 7.0]);
    let unused = leaf(vec![11.0_f64, 13.0]);
    let loss = x.mul(&y).unwrap().reduce_sum(Some(&[0])).unwrap();
    let first = loss.backward().unwrap();
    assert_eq!(first.len(), 2);
    assert_eq!(
        x.grad().unwrap().unwrap().as_slice::<f64>().unwrap(),
        &[5.0, 7.0]
    );
    assert_eq!(
        y.grad().unwrap().unwrap().as_slice::<f64>().unwrap(),
        &[2.0, 3.0]
    );
    assert!(unused.grad().unwrap().is_none());
    let stats = ctx.cache_stats().unwrap().prepared_derivatives;
    assert_eq!(stats.entries, 1, "all leaves must share one prepared VJP");
    assert_eq!(stats.misses, 1);

    // A single-input VJP must not reuse the two-input derivative's output map.
    let gx = ctx.grad(&loss, &x).unwrap();
    assert_eq!(gx.value().unwrap().as_slice::<f64>().unwrap(), &[5.0, 7.0]);
    assert_eq!(ctx.cache_stats().unwrap().prepared_derivatives.entries, 2);
    let before = ctx.cache_stats().unwrap().prepared_derivatives;
    let second = loss.backward().unwrap();
    assert_eq!(second.len(), 2);
    let after = ctx.cache_stats().unwrap().prepared_derivatives;
    assert_eq!(after.hits, before.hits + 1);
    assert_eq!(after.misses, before.misses);
    assert_eq!(
        x.grad().unwrap().unwrap().as_slice::<f64>().unwrap(),
        &[10.0, 14.0]
    );
    assert_eq!(
        y.grad().unwrap().unwrap().as_slice::<f64>().unwrap(),
        &[4.0, 6.0]
    );
}
