//! Signed axes on the traced layer (#2552 §3.2): `-1` is the last axis.

use tenferro_cpu::CpuEngine;
use tenferro_runtime::{ExecutionEngine, GraphCompiler, Runtime, TraceContext, TracedTensor};
use tenferro_tensor::Tensor;

fn run(outputs: &[&TracedTensor]) -> Vec<Tensor> {
    let program = GraphCompiler::new().compile_many(outputs).unwrap();
    let mut engine = CpuEngine::serial();
    let mut builder = Runtime::builder();
    builder
        .register_engine(tenferro_cpu::runtime_engine_registration(&engine).unwrap())
        .unwrap();
    let runtime = builder.build().unwrap();
    let prepared = runtime.prepare(&program, &[], &[engine.target()]).unwrap();
    let results = {
        let mut engines: [&mut dyn ExecutionEngine; 1] = [&mut engine];
        runtime.run(&mut engines, &prepared, &[]).unwrap()
    };
    results
        .into_iter()
        .map(|r| r.into_tensor(&mut engine).unwrap())
        .collect()
}

fn values(t: &Tensor) -> (Vec<usize>, Vec<u64>) {
    (
        t.shape().to_vec(),
        t.as_slice::<f64>()
            .unwrap()
            .iter()
            .map(|x| x.to_bits())
            .collect(),
    )
}

#[test]
fn negative_axes_equal_rank_minus_one_on_the_traced_layer() {
    let trace = TraceContext::new();
    let x = trace
        .from_vec_col_major(vec![2, 3], vec![0.5_f64, -1.25, 2.0, 0.75, 3.5, -0.25])
        .unwrap();
    let outs = [
        x.reduce_sum(Some(&[-1])).unwrap(),
        x.reduce_sum(Some(&[1])).unwrap(),
        x.softmax(-1).unwrap(),
        x.softmax(1).unwrap(),
        TracedTensor::concatenate(&[&x, &x], -1).unwrap(),
        TracedTensor::concatenate(&[&x, &x], 1).unwrap(),
        x.transpose(&[-1, 0]).unwrap(),
        x.transpose(&[1, 0]).unwrap(),
    ];
    let refs: Vec<&TracedTensor> = outs.iter().collect();
    let tensors = run(&refs);
    for pair in tensors.chunks(2) {
        assert_eq!(values(&pair[0]), values(&pair[1]));
    }
    assert!(x.reduce_sum(Some(&[-3])).is_err());
    assert!(x.softmax(2).is_err());
}
