use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use tenferro_cpu::CpuBackend;
use tenferro_runtime::Runtime;
use tenferro_runtime::{CompiledGraph, DType, GraphCompiler, Tensor, TracedTensor};

fn cpu_runtime() -> Runtime {
    let mut builder = Runtime::builder();
    builder
        .register_engine(tenferro_cpu::runtime_engine_registration(&CpuBackend::new()).unwrap())
        .unwrap();
    builder.build().unwrap()
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

fn bench_runtime_add_mul(c: &mut Criterion) {
    let mut group = c.benchmark_group("runtime_elementwise_chain/f64/add_mul");
    for &n in &[4_096_usize, 65_536, 1_048_576] {
        let program = compile_add_mul(n);
        let a = input_tensor(n, 0.5);
        let b = input_tensor(n, 1.25);
        group.throughput(criterion::Throughput::Elements(n as u64));
        group.bench_function(BenchmarkId::new("segmented_graph", n), |bench| {
            let runtime = cpu_runtime();
            let prepared = runtime
                .prepare_compiled(&program, &[&a, &b])
                .expect("graph should prepare");
            bench.iter(|| {
                let out = runtime
                    .run_prepared(black_box(&prepared), &[black_box(&a), black_box(&b)])
                    .expect("graph run");
                black_box(out);
            });
        });
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
        group.bench_function(
            BenchmarkId::new("segmented_graph", format!("{rows}x{cols}")),
            |bench| {
                let runtime = cpu_runtime();
                let prepared = runtime
                    .prepare_compiled(&program, &[&a, &b])
                    .expect("graph should prepare");
                bench.iter(|| {
                    let out = runtime
                        .run_prepared(black_box(&prepared), &[black_box(&a), black_box(&b)])
                        .expect("graph run");
                    black_box(out);
                });
            },
        );
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
        group.bench_function(
            BenchmarkId::new("segmented_graph", format!("{rows}x{cols}")),
            |bench| {
                let runtime = cpu_runtime();
                let prepared = runtime
                    .prepare_compiled(&program, &[&a, &b])
                    .expect("graph should prepare");
                bench.iter(|| {
                    let out = runtime
                        .run_prepared(black_box(&prepared), &[black_box(&a), black_box(&b)])
                        .expect("graph run");
                    black_box(out);
                });
            },
        );
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
