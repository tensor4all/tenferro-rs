//! Regression test for tenferro-rs#2643: `Runtime::prepare` on a reverse-mode
//! graph of a solver unrolled over several steps.
//!
//! The AD transform bakes concrete extents into `BroadcastInDim` / `Reshape`
//! while the compiler's analysis shape stream starts from symbolic
//! `InputDim`s, so every broadcast combined a constant with an input-derived
//! expression into a `Max`. The two operands were distinct, already grown
//! trees, and the expression roughly doubled per RK4 step: two steps took
//! 20 s to prepare and three were OOM-killed at 20 GB. The graph below is the
//! neural-ODE window from tensor4all/tenferro-examples#12 (2-state RK4, a
//! `[2, 50, 2]` tanh MLP applied to the elementwise cube, ten steps).

use tenferro_ad::TracedTensorAdExt;
use tenferro_runtime::{
    DType, DotGeneralConfig, ExecutionEngine, GraphCompiler, Tensor, TensorRead, TraceContext,
    TracedTensor,
};

use crate::support::runtime_from_cpu_backend;

const BATCH: usize = 64;
const HIDDEN: usize = 50;
const STEPS: usize = 10;
const H: f64 = 25.0 / 999.0;

fn linear(
    x: &TracedTensor,
    weight: &TracedTensor,
    bias: &TracedTensor,
    out: usize,
) -> TracedTensor {
    let y = x
        .dot_general(
            weight,
            DotGeneralConfig {
                lhs_contracting_dims: [1].as_slice().into(),
                rhs_contracting_dims: [0].as_slice().into(),
                lhs_batch_dims: [].as_slice().into(),
                rhs_batch_dims: [].as_slice().into(),
            },
        )
        .unwrap();
    y.add(&bias.reshape([1, out]).unwrap()).unwrap()
}

/// Deterministic values in `[-0.5, 0.5)`, so the test needs no RNG.
fn spread(n: usize, offset: usize) -> Vec<f64> {
    (0..n)
        .map(|k| (((k + offset) as f64) * 0.754_877_666_246_692_7).fract() - 0.5)
        .collect()
}

#[test]
fn unrolled_rk4_value_and_grad_prepares_and_runs() {
    let trace = TraceContext::new();
    let params = [
        trace
            .input_concrete_shape(DType::F64, &[2, HIDDEN])
            .unwrap(),
        trace.input_concrete_shape(DType::F64, &[HIDDEN]).unwrap(),
        trace
            .input_concrete_shape(DType::F64, &[HIDDEN, 2])
            .unwrap(),
        trace.input_concrete_shape(DType::F64, &[2]).unwrap(),
    ];
    let z0 = trace.input_concrete_shape(DType::F64, &[BATCH, 2]).unwrap();
    let target = trace.input_concrete_shape(DType::F64, &[BATCH, 2]).unwrap();

    let rhs = |z: &TracedTensor| -> TracedTensor {
        let cubed = z.mul(z).unwrap().mul(z).unwrap();
        let hidden = linear(&cubed, &params[0], &params[1], HIDDEN)
            .tanh()
            .unwrap();
        linear(&hidden, &params[2], &params[3], 2)
    };
    let step = |z: &TracedTensor| -> TracedTensor {
        let k1 = rhs(z);
        let k2 = rhs(&z.add(&k1.scale_real(0.5 * H).unwrap()).unwrap());
        let k3 = rhs(&z.add(&k2.scale_real(0.5 * H).unwrap()).unwrap());
        let k4 = rhs(&z.add(&k3.scale_real(H).unwrap()).unwrap());
        let stages = k1
            .add(&k2.scale_real(2.0).unwrap())
            .unwrap()
            .add(&k3.scale_real(2.0).unwrap())
            .unwrap();
        z.add(&k4.add(&stages).unwrap().scale_real(H / 6.0).unwrap())
            .unwrap()
    };

    let mut z = z0.clone();
    let mut loss: Option<TracedTensor> = None;
    for _ in 0..STEPS {
        z = step(&z);
        let residual = z.sub(&target).unwrap();
        let term = residual.mul(&residual).unwrap().reduce_sum(None).unwrap();
        loss = Some(match loss {
            Some(total) => total.add(&term).unwrap(),
            None => term,
        });
    }
    let loss = loss.unwrap();
    let leaves: Vec<&TracedTensor> = params.iter().collect();
    let (_, grads) = loss.value_and_grad(&leaves).unwrap();
    let grads: Vec<TracedTensor> = grads.into_iter().map(Option::unwrap).collect();
    let mut outputs = vec![&loss];
    outputs.extend(grads.iter());
    let program = GraphCompiler::new().compile_many(&outputs).unwrap();

    let values = [
        Tensor::from_vec_col_major(vec![2, HIDDEN], spread(2 * HIDDEN, 0)).unwrap(),
        Tensor::from_vec_col_major(vec![HIDDEN], spread(HIDDEN, 1)).unwrap(),
        Tensor::from_vec_col_major(vec![HIDDEN, 2], spread(2 * HIDDEN, 2)).unwrap(),
        Tensor::from_vec_col_major(vec![2], spread(2, 3)).unwrap(),
        Tensor::from_vec_col_major(vec![BATCH, 2], spread(2 * BATCH, 4)).unwrap(),
        Tensor::from_vec_col_major(vec![BATCH, 2], spread(2 * BATCH, 5)).unwrap(),
    ];
    let reads: Vec<TensorRead<'_>> = values.iter().map(TensorRead::from_tensor).collect();
    let mut engine = tenferro_cpu::CpuEngine::serial();
    let runtime = runtime_from_cpu_backend(&engine);
    let prepared = runtime
        .prepare(&program, &reads, std::slice::from_ref(&engine.target()))
        .expect("the unrolled reverse-mode graph must prepare");
    let results = runtime
        .run(&mut [&mut engine], &prepared, &reads)
        .expect("the prepared program must run");

    assert_eq!(results.len(), 1 + params.len());
    for result in results {
        let tensor = result.into_tensor(&mut engine).unwrap();
        assert!(tensor
            .as_slice::<f64>()
            .unwrap()
            .iter()
            .all(|v| v.is_finite()));
    }
}
