use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use tenferro_cpu::CpuEngine;
use tenferro_runtime::Runtime;
use tenferro_runtime::{
    CompiledGraph, DType, ExecutionEngine, GraphCompiler, Tensor, TensorRead, TracedTensor,
};

fn cpu_engine_and_runtime() -> (CpuEngine, Runtime) {
    let engine = CpuEngine::serial();
    let mut builder = Runtime::builder();
    builder
        .register_engine(tenferro_cpu::runtime_engine_registration(&engine).unwrap())
        .unwrap();
    (engine, builder.build().unwrap())
}

fn input_tensor(n: usize, scale: f64) -> Tensor {
    let data = (0..n)
        .map(|i| ((i % 97) as f64 + 1.0) * scale)
        .collect::<Vec<_>>();
    Tensor::from_vec_col_major(vec![n], data).expect("benchmark tensor")
}

fn compile_add_mul(n: usize) -> CompiledGraph {
    let a = TracedTensor::input_concrete_shape(DType::F64, &[n]).expect("a placeholder");
    let b = TracedTensor::input_concrete_shape(DType::F64, &[n]).expect("b placeholder");
    let sum = (&a + &b).expect("sum graph");
    let out = (&sum * &a).expect("multiply graph");
    let mut compiler = GraphCompiler::new();
    compiler
        .compile_with_input_specs(&out, &[(&a, DType::F64, &[n]), (&b, DType::F64, &[n])])
        .expect("compiled add-mul graph")
}

fn compile_broadcast_mul(rows: usize, cols: usize) -> CompiledGraph {
    let a = TracedTensor::input_concrete_shape(DType::F64, &[rows]).expect("a placeholder");
    let b = TracedTensor::input_concrete_shape(DType::F64, &[cols]).expect("b placeholder");
    let a_bc = a
        .broadcast_in_dim(&[rows, cols], &[0])
        .expect("lhs broadcast graph");
    let b_bc = b
        .broadcast_in_dim(&[rows, cols], &[1])
        .expect("rhs broadcast graph");
    let out = (&a_bc * &b_bc).expect("multiply graph");
    let mut compiler = GraphCompiler::new();
    compiler
        .compile_with_input_specs(
            &out,
            &[(&a, DType::F64, &[rows]), (&b, DType::F64, &[cols])],
        )
        .expect("compiled broadcast-multiply graph")
}

fn compile_broadcast_mul_add(rows: usize, cols: usize) -> CompiledGraph {
    let a = TracedTensor::input_concrete_shape(DType::F64, &[rows]).expect("a placeholder");
    let b = TracedTensor::input_concrete_shape(DType::F64, &[cols]).expect("b placeholder");
    let a_bc = a
        .broadcast_in_dim(&[rows, cols], &[0])
        .expect("lhs broadcast graph");
    let b_bc = b
        .broadcast_in_dim(&[rows, cols], &[1])
        .expect("rhs broadcast graph");
    let product = (&a_bc * &b_bc).expect("multiply graph");
    let out = (&product + &a_bc).expect("add graph");
    let mut compiler = GraphCompiler::new();
    compiler
        .compile_with_input_specs(
            &out,
            &[(&a, DType::F64, &[rows]), (&b, DType::F64, &[cols])],
        )
        .expect("compiled broadcast-multiply-add graph")
}

/// Benchmark one graph through the prepared path.
///
/// Every production run now goes through the scheduled executor that
/// `prepared_graph` measures, so there is one timing per case: the former
/// `unprepared_graph` label measured the same executor.
fn bench_both_paths(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    program: &CompiledGraph,
    inputs: &[&Tensor],
    case: String,
) {
    group.bench_function(BenchmarkId::new("prepared_graph", case), |bench| {
        let (mut engine, runtime) = cpu_engine_and_runtime();
        let reads: Vec<TensorRead<'_>> = inputs
            .iter()
            .map(|input| TensorRead::from_tensor(input))
            .collect();
        let prepared = runtime
            .prepare(program, &reads, &[engine.target()])
            .expect("graph should prepare");
        bench.iter(|| {
            let out = runtime
                .run(&mut [&mut engine], black_box(&prepared), &reads)
                .expect("graph run");
            black_box(out);
        });
    });
}

fn bench_runtime_add_mul(c: &mut Criterion) {
    let mut group = c.benchmark_group("runtime_elementwise_chain/f64/add_mul");
    for &n in &[4_096_usize, 65_536, 1_048_576] {
        let program = compile_add_mul(n);
        let a = input_tensor(n, 0.5);
        let b = input_tensor(n, 1.25);
        group.throughput(criterion::Throughput::Elements(n as u64));
        bench_both_paths(&mut group, &program, &[&a, &b], n.to_string());
    }
    group.finish();
}

fn bench_runtime_broadcast_mul(c: &mut Criterion) {
    let mut group = c.benchmark_group("runtime_elementwise_chain/f64/broadcast_mul");
    for &(rows, cols) in &[(256_usize, 256_usize), (1024, 1024)] {
        let program = compile_broadcast_mul(rows, cols);
        let a = input_tensor(rows, 0.5);
        let b = input_tensor(cols, 1.25);
        let elements = rows * cols;
        group.throughput(criterion::Throughput::Elements(elements as u64));
        bench_both_paths(&mut group, &program, &[&a, &b], format!("{rows}x{cols}"));
    }
    group.finish();
}

fn bench_runtime_broadcast_mul_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("runtime_elementwise_chain/f64/broadcast_mul_add");
    for &(rows, cols) in &[(256_usize, 256_usize), (1024, 1024)] {
        let program = compile_broadcast_mul_add(rows, cols);
        let a = input_tensor(rows, 0.5);
        let b = input_tensor(cols, 1.25);
        let elements = rows * cols;
        group.throughput(criterion::Throughput::Elements(elements as u64));
        bench_both_paths(&mut group, &program, &[&a, &b], format!("{rows}x{cols}"));
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_runtime_add_mul,
    bench_runtime_broadcast_mul,
    bench_runtime_broadcast_mul_add
);
criterion_main!(benches);
