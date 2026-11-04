//! Command census of the #2226 reproducer.
//!
//! `grad_sum_qr_vjp` 2x2 is the traced program whose per-command cost #2186
//! and #2226 measure. Its metadata-only operations (`Transpose`,
//! `BroadcastInDim`) with one reading consumer fold into that consumer's
//! command; the broadcast with two consumers and the transposed program output
//! keep theirs. This test pins the count and checks the VJP that the folded
//! program computes against central finite differences.

#![cfg(feature = "autodiff")]

use tenferro_ad::AdContext;
use tenferro_cpu::CpuEngine;
use tenferro_linalg::TracedTensorLinalgExt;
use tenferro_runtime::{GraphCompiler, TracedTensor};
use tenferro_tensor::{Tensor, TypedTensor};

use super::support;

/// Scheduled commands of the 2x2 reproducer after the metadata view fold.
const REPRODUCER_COMMANDS: usize = 20;

fn f64_tensor(shape: Vec<usize>, data: Vec<f64>) -> Tensor {
    Tensor::from_typed::<f64>(TypedTensor::from_vec_col_major(shape, data).unwrap())
}

fn traced(shape: &[usize], data: Vec<f64>) -> TracedTensor {
    TracedTensor::from_tensor_concrete_shape(f64_tensor(shape.to_vec(), data)).unwrap()
}

/// The benchmark's diagonal-dominant 2x2 matrix (`well_conditioned_matrix`).
fn well_conditioned_matrix(n: usize, seed: u64) -> Vec<f64> {
    let mut data = vec![0.0; n * n];
    for col in 0..n {
        for row in 0..n {
            let mixed = (row as u64)
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add((col as u64).wrapping_mul(1_442_695_040_888_963_407))
                .wrapping_add(seed);
            let mut value = (mixed % 2048) as f64 / 2048.0 - 0.5;
            value *= 0.05;
            if row == col {
                value += 2.0 + row as f64 / n as f64;
            }
            data[row + n * col] = value;
        }
    }
    data
}

/// sum(Q) + sum(R) of the QR factors of `a`, and its VJP with respect to `a`.
fn loss_and_gradient(a: &TracedTensor, ad: &AdContext) -> (TracedTensor, TracedTensor) {
    let (q, r) = a.qr().unwrap();
    let loss =
        (&q.reduce_sum(Some(&[0, 1])).unwrap() + &r.reduce_sum(Some(&[0, 1])).unwrap()).unwrap();
    let seed = traced(&[], vec![1.0]);
    let gradient = ad.vjp(&loss, a, &seed).unwrap();
    (loss, gradient)
}

fn ad_context() -> AdContext {
    AdContext::builder()
        .with_semantic_extension_rules(tenferro_linalg::semantic_ad_rules().unwrap())
        .unwrap()
        .build()
        .unwrap()
}

fn evaluate(
    engine: &mut CpuEngine,
    runtime: &tenferro_runtime::Runtime,
    data: Vec<f64>,
) -> (f64, Vec<f64>, usize) {
    let a = traced(&[2, 2], data);
    let (loss, gradient) = loss_and_gradient(&a, &ad_context());
    let program = GraphCompiler::new()
        .compile_many(&[&loss, &gradient])
        .unwrap();
    let prepared = runtime
        .prepare(
            &program,
            &[],
            &[tenferro_runtime::ExecutionEngine::target(engine)],
        )
        .unwrap();
    let commands = prepared.execution_command_count();
    let outputs = support::run_program(runtime, engine, &program, &[]).unwrap();
    let loss = outputs[0].as_slice::<f64>().unwrap()[0];
    let gradient = outputs[1].as_slice::<f64>().unwrap().to_vec();
    (loss, gradient, commands)
}

#[test]
fn grad_sum_qr_vjp_2x2_census_and_finite_differences() {
    let mut engine = CpuEngine::serial();
    let runtime = support::cpu_runtime_with_linalg(&engine).unwrap();
    let base = well_conditioned_matrix(2, 4);

    let (_, gradient, commands) = evaluate(&mut engine, &runtime, base.clone());
    assert_eq!(commands, REPRODUCER_COMMANDS);

    let step = 1e-6;
    for entry in 0..4 {
        let mut plus = base.clone();
        plus[entry] += step;
        let mut minus = base.clone();
        minus[entry] -= step;
        let (loss_plus, _, _) = evaluate(&mut engine, &runtime, plus);
        let (loss_minus, _, _) = evaluate(&mut engine, &runtime, minus);
        let central = (loss_plus - loss_minus) / (2.0 * step);
        assert!(
            (central - gradient[entry]).abs() < 1e-6,
            "entry {entry}: finite difference {central} vs vjp {}",
            gradient[entry]
        );
    }
}
