//! A `dot_general` with no contracting and no batch dims is an outer product.
//!
//! `dot_general_output_shape` makes that form's output `lhs ++ rhs`, because
//! every operand axis is free. Two shapes that happen to be equal are not a
//! Hadamard: that form needs every axis to be a *batch* axis. The classifier
//! answered `Hadamard` for equal shapes without batch dims, and both consumers
//! of it then produced `a.shape` where the graph declared `lhs ++ rhs`, which is
//! the rank/payload mismatch tensor4all-rs reported in #2165.
//!
//! The eager and CUDA paths share the classifier; the compiler's degenerate
//! rewrite does too, so this exercises the compiled path end to end and pins the
//! Hadamard form as well, which must keep working.
use tenferro_cpu::{runtime_engine_registration, CpuEngine};
use tenferro_runtime::{CompiledGraph, ExecutionEngine, GraphCompiler, Runtime, TracedTensor};
use tenferro_tensor::{DType, DotGeneralConfig, Tensor, TensorRead};

fn cpu_engine() -> CpuEngine {
    CpuEngine::serial()
}

fn runtime(engine: &CpuEngine) -> Runtime {
    let mut builder = Runtime::builder();
    builder
        .register_engine(runtime_engine_registration(engine).expect("engine"))
        .expect("register the CPU engine");
    builder.build().expect("runtime")
}

fn run_compiled(
    engine: &mut CpuEngine,
    runtime: &Runtime,
    program: &CompiledGraph,
    inputs: &[&Tensor],
) -> Vec<Tensor> {
    let reads: Vec<TensorRead<'_>> = inputs
        .iter()
        .map(|input| TensorRead::from_tensor(input))
        .collect();
    let prepared = runtime
        .prepare(program, &reads, &[engine.target()])
        .expect("prepare");
    runtime
        .run(&mut [engine], &prepared, &reads)
        .expect("run")
        .into_iter()
        .map(|output| output.into_tensor(engine).expect("own output"))
        .collect()
}

fn config(lhs_batch: &[usize], rhs_batch: &[usize]) -> DotGeneralConfig {
    DotGeneralConfig {
        lhs_contracting_dims: [].as_slice().into(),
        rhs_contracting_dims: [].as_slice().into(),
        lhs_batch_dims: lhs_batch.iter().copied().collect(),
        rhs_batch_dims: rhs_batch.iter().copied().collect(),
    }
}

/// Column-major reference for `out[i, j, k, l] = a[i, j] * b[k, l]`: `a` is one
/// column-major block per leading pair of output axes.
fn outer_product_reference(a: &[f64], b: &[f64]) -> Vec<f64> {
    b.iter()
        .flat_map(|&factor| a.iter().map(move |&value| value * factor))
        .collect()
}

#[test]
fn equal_shaped_operands_without_batch_dims_form_an_outer_product() {
    const N: usize = 2;
    let a = Tensor::from_vec_col_major(vec![N, N], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let b = Tensor::from_vec_col_major(vec![N, N], vec![5.0_f64, 6.0, 7.0, 8.0]).unwrap();

    let mut compiler = GraphCompiler::new();
    let lhs = TracedTensor::input_concrete_shape(DType::F64, &[N, N]).unwrap();
    let rhs = TracedTensor::input_concrete_shape(DType::F64, &[N, N]).unwrap();
    let product = lhs.dot_general(&rhs, config(&[], &[])).unwrap();
    assert_eq!(product.rank, 4, "the graph declares the concatenated rank");
    let program = compiler
        .compile_with_input_specs(
            &product,
            &[(&lhs, DType::F64, &[N, N]), (&rhs, DType::F64, &[N, N])],
        )
        .unwrap();

    let mut engine = cpu_engine();
    let runtime = runtime(&engine);
    let mut out = run_compiled(&mut engine, &runtime, &program, &[&a, &b]);
    let out = out.pop().unwrap();
    assert_eq!(out.shape(), &[N, N, N, N]);
    assert_eq!(
        out.as_slice::<f64>().unwrap(),
        outer_product_reference(a.as_slice::<f64>().unwrap(), b.as_slice::<f64>().unwrap())
            .as_slice()
    );
}

/// The form that really is a Hadamard: every axis batched, so the output is the
/// shared shape and the compiled result is one elementwise product.
#[test]
fn fully_batched_equal_shaped_operands_form_a_hadamard() {
    const N: usize = 2;
    let a = Tensor::from_vec_col_major(vec![N, N], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let b = Tensor::from_vec_col_major(vec![N, N], vec![5.0_f64, 6.0, 7.0, 8.0]).unwrap();

    let mut compiler = GraphCompiler::new();
    let lhs = TracedTensor::input_concrete_shape(DType::F64, &[N, N]).unwrap();
    let rhs = TracedTensor::input_concrete_shape(DType::F64, &[N, N]).unwrap();
    let product = lhs.dot_general(&rhs, config(&[0, 1], &[0, 1])).unwrap();
    assert_eq!(product.rank, 2);
    let program = compiler
        .compile_with_input_specs(
            &product,
            &[(&lhs, DType::F64, &[N, N]), (&rhs, DType::F64, &[N, N])],
        )
        .unwrap();

    let mut engine = cpu_engine();
    let runtime = runtime(&engine);
    let mut out = run_compiled(&mut engine, &runtime, &program, &[&a, &b]);
    let out = out.pop().unwrap();
    assert_eq!(out.shape(), &[N, N]);
    let a = a.as_slice::<f64>().unwrap();
    let b = b.as_slice::<f64>().unwrap();
    let expected: Vec<f64> = a.iter().zip(b).map(|(x, y)| x * y).collect();
    assert_eq!(out.as_slice::<f64>().unwrap(), expected.as_slice());
}

/// Batch axis `k` pairs `lhs_batch_dims[k]` with `rhs_batch_dims[k]`, so a
/// permuted batch list is not an elementwise product: `a[i, j] * b[j, i]`. The
/// classifier answered `Hadamard` for every fully batched equal-shaped pair,
/// and the compiled path then multiplied the operands elementwise.
#[test]
fn permuted_batch_dims_are_not_an_elementwise_product() {
    const N: usize = 2;
    let a = Tensor::from_vec_col_major(vec![N, N], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let b = Tensor::from_vec_col_major(vec![N, N], vec![5.0_f64, 6.0, 7.0, 8.0]).unwrap();

    let mut compiler = GraphCompiler::new();
    let lhs = TracedTensor::input_concrete_shape(DType::F64, &[N, N]).unwrap();
    let rhs = TracedTensor::input_concrete_shape(DType::F64, &[N, N]).unwrap();
    let product = lhs.dot_general(&rhs, config(&[0, 1], &[1, 0])).unwrap();
    let program = compiler
        .compile_with_input_specs(
            &product,
            &[(&lhs, DType::F64, &[N, N]), (&rhs, DType::F64, &[N, N])],
        )
        .unwrap();

    let mut engine = cpu_engine();
    let runtime = runtime(&engine);
    let mut out = run_compiled(&mut engine, &runtime, &program, &[&a, &b]);
    let out = out.pop().unwrap();
    let a = a.as_slice::<f64>().unwrap();
    let b = b.as_slice::<f64>().unwrap();
    let mut expected = vec![0.0; N * N];
    for i in 0..N {
        for j in 0..N {
            expected[i + N * j] = a[i + N * j] * b[j + N * i];
        }
    }
    assert_eq!(out.as_slice::<f64>().unwrap(), expected.as_slice());
}
