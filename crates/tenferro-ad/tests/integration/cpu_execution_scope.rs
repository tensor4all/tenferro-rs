use std::sync::Arc;

use tenferro_ad::{AdContext, EagerRuntime, EagerTensor};
use tenferro_cpu::CpuBackend;
use tenferro_runtime::{GraphCompiler, TracedTensor};
use tenferro_tensor::{DotGeneralConfig, Tensor};

use crate::support::runtime_from_cpu_backend;

fn tensor(shape: &[usize], values: Vec<f64>) -> Tensor {
    Tensor::from_vec_col_major(shape.to_vec(), values).unwrap()
}

fn assert_values(actual: &Tensor, expected: &[f64]) {
    let actual = actual.as_slice::<f64>().unwrap();
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= 1e-12 * e.abs().max(1.0),
            "element {i}: {a} != {e}"
        );
    }
}

#[test]
fn shared_scope_eager_and_prepared_matmul_primal_jvp_vjp() {
    for threads in [1, 4] {
        for (n, batch) in [(2, 16), (4, 3), (16, 1)] {
            let owner = CpuBackend::with_threads(threads).unwrap();
            let eager = EagerRuntime::with_cpu_backend(owner.clone()).unwrap();
            let runtime = runtime_from_cpu_backend(&owner);
            let other_runtime = runtime_from_cpu_backend(&owner);
            let shape = [n, n, batch];
            let output_shape = [n, n, batch];
            let count = n * n * batch;
            let a: Vec<_> = (0..count)
                .map(|i| 0.25 + (i % 11) as f64 * 0.03125)
                .collect();
            let b: Vec<_> = (0..count)
                .map(|i| 0.125 + (i % 7) as f64 * 0.0625)
                .collect();
            let mut primal = vec![0.0; count];
            let mut jvp = vec![0.0; count];
            let mut vjp = vec![0.0; count];
            for z in 0..batch {
                for j in 0..n {
                    for i in 0..n {
                        for k in 0..n {
                            let output = i + n * (j + n * z);
                            let lhs = i + n * (k + n * z);
                            let rhs = k + n * (j + n * z);
                            primal[output] += a[lhs] * b[rhs];
                            jvp[output] += b[rhs];
                            vjp[lhs] += b[rhs];
                        }
                    }
                }
            }
            let config = DotGeneralConfig {
                lhs_contracting_dims: [1].as_slice().into(),
                rhs_contracting_dims: [0].as_slice().into(),
                lhs_batch_dims: [2].as_slice().into(),
                rhs_batch_dims: [2].as_slice().into(),
            };
            let x = EagerTensor::requires_grad_in(tensor(&shape, a.clone()), Arc::clone(&eager))
                .unwrap();
            let y =
                EagerTensor::from_tensor_in(tensor(&shape, b.clone()), Arc::clone(&eager)).unwrap();
            let tangent =
                EagerTensor::from_tensor_in(tensor(&shape, vec![1.0; count]), Arc::clone(&eager))
                    .unwrap();
            let seed = EagerTensor::from_tensor_in(
                tensor(&output_shape, vec![1.0; count]),
                Arc::clone(&eager),
            )
            .unwrap();
            let tx = TracedTensor::from_tensor_concrete_shape(tensor(&shape, a.clone())).unwrap();
            let ty = TracedTensor::from_tensor_concrete_shape(tensor(&shape, b)).unwrap();
            let td =
                TracedTensor::from_tensor_concrete_shape(tensor(&shape, vec![1.0; count])).unwrap();
            let ts =
                TracedTensor::from_tensor_concrete_shape(tensor(&output_shape, vec![1.0; count]))
                    .unwrap();
            let traced = tx.dot_general(&ty, config.clone()).unwrap();
            let ad = AdContext::builder().build().unwrap();
            let traced_jvp = ad.jvp(&traced, &tx, &td).unwrap();
            let traced_vjp = ad.vjp(&traced, &tx, &ts).unwrap();
            let mut compiler = GraphCompiler::new();
            let prepared: Vec<_> = [&traced, &traced_jvp, &traced_vjp]
                .iter()
                .map(|output| {
                    let compiled = compiler.compile(output).unwrap();
                    runtime.prepare_compiled(&compiled, &[]).unwrap()
                })
                .collect();
            let check = || {
                let output = x.dot_general(&y, config.clone()).unwrap();
                assert_eq!(output.shape(), output_shape);
                assert_values(&output.to_tensor().unwrap(), &primal);
                assert_values(
                    &eager
                        .jvp(&output, &x, &tangent)
                        .unwrap()
                        .to_tensor()
                        .unwrap(),
                    &jvp,
                );
                assert_values(
                    &eager.vjp(&output, &x, &seed).unwrap().to_tensor().unwrap(),
                    &vjp,
                );
                x.clear_grad().unwrap();
                output.reduce_sum(None).unwrap().backward().unwrap();
                assert_eq!(
                    x.grad().unwrap().unwrap().as_slice::<f64>().unwrap(),
                    vjp.as_slice()
                );
                x.clear_grad().unwrap();
                for (plan, expected) in prepared.iter().zip([&primal, &jvp, &vjp]) {
                    assert!(other_runtime.run_prepared(plan, &[]).is_err());
                    let results = runtime.run_prepared(plan, &[]).unwrap();
                    assert_eq!(results.len(), 1);
                    assert_values(&results[0], expected);
                }
                assert_values(&x.to_tensor().unwrap(), &a);
            };
            check();
            owner
                .with_execution_scope(|| {
                    check();
                    check();
                })
                .unwrap();
            check();
        }
    }
}

#[test]
fn shared_scope_elementwise_reduction_ad_and_error_recovery() {
    for threads in [1, 4] {
        let owner = CpuBackend::with_threads(threads).unwrap();
        let eager = EagerRuntime::with_cpu_backend(owner.clone()).unwrap();
        for n in [3, 64, 1024] {
            let data: Vec<_> = (0..n).map(|i| (i % 13) as f64 * 0.125).collect();
            let x = EagerTensor::requires_grad_in(tensor(&[n], data.clone()), Arc::clone(&eager))
                .unwrap();
            let tangent =
                EagerTensor::from_tensor_in(tensor(&[n], vec![1.0; n]), Arc::clone(&eager))
                    .unwrap();
            let seed =
                EagerTensor::from_tensor_in(tensor(&[], vec![1.0]), Arc::clone(&eager)).unwrap();
            let check = || {
                let loss = x.mul(&x).unwrap().reduce_sum(None).unwrap();
                assert_values(
                    &loss.to_tensor().unwrap(),
                    &[data.iter().map(|v| v * v).sum()],
                );
                assert_values(
                    &eager.jvp(&loss, &x, &tangent).unwrap().to_tensor().unwrap(),
                    &[2.0 * data.iter().sum::<f64>()],
                );
                assert_values(
                    &eager.vjp(&loss, &x, &seed).unwrap().to_tensor().unwrap(),
                    &data.iter().map(|v| 2.0 * v).collect::<Vec<_>>(),
                );
                assert!(x.reshape([n + 1]).is_err());
                assert_values(&x.to_tensor().unwrap(), &data);
            };
            owner.with_execution_scope(check).unwrap();
            check();
        }
    }
}
