use std::collections::HashSet;
use std::sync::Arc;

use crate::{EagerRuntime, EagerTensor, Tensor};
use tenferro_cpu::CpuBackend;
use tenferro_runtime::program::{CoreSemanticOp, SemanticOpRef};

fn assert_no_live_transcendental(ctx: &EagerRuntime) {
    let cache = ctx.lock_prepared_derivative_cache().unwrap();
    for (_, entry) in cache.entries.iter() {
        let program = entry.value.execution_program.program();
        let mut live: HashSet<_> = program.outputs().iter().copied().collect();
        for operation in program.operations().collect::<Vec<_>>().into_iter().rev() {
            if operation
                .outputs()
                .iter()
                .any(|output| live.contains(output))
            {
                assert!(
                    !matches!(
                        operation.op(),
                        SemanticOpRef::Core(CoreSemanticOp::Exp | CoreSemanticOp::Tanh)
                    ),
                    "backward must consume the saved output, not recompute it"
                );
                live.extend(operation.inputs().iter().copied());
            }
        }
    }
}

#[test]
fn exp_tanh_reuse_saved_outputs_with_dropped_handles_and_correct_higher_derivatives() {
    for tanh in [false, true] {
        let ctx = EagerRuntime::with_cpu_backend(CpuBackend::with_threads(1).unwrap()).unwrap();
        for values in [vec![0.3_f64, -0.4], vec![0.7_f64, -0.2]] {
            let x = EagerTensor::requires_grad_in(
                Tensor::from_vec_col_major(vec![2], values.clone()).unwrap(),
                Arc::clone(&ctx),
            )
            .unwrap();
            let y = if tanh { x.tanh() } else { x.exp() }.unwrap();
            let loss = y.reduce_sum(Some(&[0])).unwrap();
            let saved = loss.trace.as_ref().unwrap().collect();
            assert_eq!(saved.len(), 1);
            let weak = Arc::downgrade(&saved[0].value);
            drop(saved);
            drop(y);
            assert!(weak.upgrade().is_some());
            let first = ctx.grad(&loss, &x).unwrap();
            let expected = values
                .iter()
                .map(|&x| {
                    if tanh {
                        1.0 - x.tanh().powi(2)
                    } else {
                        x.exp()
                    }
                })
                .collect::<Vec<_>>();
            for (actual, expected) in first
                .value()
                .unwrap()
                .as_slice::<f64>()
                .unwrap()
                .iter()
                .zip(expected)
            {
                assert!((actual - expected).abs() < 1e-12);
            }
            // Only inspect first-order execution plans; higher-order programs
            // retain the complete mathematical producer graph deliberately.
            assert_no_live_transcendental(&ctx);
            let second = ctx
                .grad(&first.reduce_sum(Some(&[0])).unwrap(), &x)
                .unwrap();
            for (actual, &x) in second
                .value()
                .unwrap()
                .as_slice::<f64>()
                .unwrap()
                .iter()
                .zip(&values)
            {
                let expected = if tanh {
                    -2.0 * x.tanh() * (1.0 - x.tanh().powi(2))
                } else {
                    x.exp()
                };
                assert!((actual - expected).abs() < 1e-11, "{actual} != {expected}");
            }
            drop(second);
            drop(first);
            drop(loss);
            assert!(
                weak.upgrade().is_none(),
                "program caches must not retain residual payloads"
            );
            ctx.clear_prepared_derivative_cache().unwrap();
        }
    }
}

#[test]
fn cached_execution_rebinds_residuals_for_each_forward() {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::with_threads(1).unwrap()).unwrap();
    for (iteration, value) in [0.2_f64, 0.8].into_iter().enumerate() {
        let x = EagerTensor::requires_grad_in(
            Tensor::from_vec_col_major(vec![1], vec![value]).unwrap(),
            Arc::clone(&ctx),
        )
        .unwrap();
        let loss = x.exp().unwrap().reduce_sum(Some(&[0])).unwrap();
        let gradient = ctx.grad(&loss, &x).unwrap();
        assert!(
            (gradient.value().unwrap().as_slice::<f64>().unwrap()[0] - value.exp()).abs() < 1e-12
        );
        let stats = ctx.cache_stats().unwrap().prepared_derivatives;
        assert_eq!(stats.entries, 1);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.hits, iteration as u64);
    }
}

#[test]
fn intermediate_inputs_are_retained_without_retaining_all_outputs() {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::with_threads(1).unwrap()).unwrap();
    let x = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![2], vec![2.0_f64, 3.0]).unwrap(),
        Arc::clone(&ctx),
    )
    .unwrap();
    let square = x.mul(&x).unwrap();
    assert!(square.trace.as_ref().unwrap().collect().is_empty());
    let fourth = square.mul(&square).unwrap();
    assert_eq!(fourth.trace.as_ref().unwrap().collect().len(), 1);
    let loss = fourth.reduce_sum(Some(&[0])).unwrap();
    drop(square);
    drop(fourth);
    let gradient = ctx.grad(&loss, &x).unwrap();
    assert_eq!(
        gradient.value().unwrap().as_slice::<f64>().unwrap(),
        &[32.0, 108.0]
    );
}
