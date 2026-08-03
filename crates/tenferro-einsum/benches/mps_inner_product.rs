use std::sync::Arc;

use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use num_complex::Complex64;
use tenferro_cpu::CpuBackend;
use tenferro_einsum::TraceContextEinsumExt;
use tenferro_ops::dim_expr::DimExpr;
use tenferro_runtime::program::{CoreSemanticOp, ProgramInputSpec};
use tenferro_runtime::{CompiledGraph, GraphCompiler, Runtime, Tensor, TraceContext, TypedTensor};

const L: usize = 32;
const PHYS_DIM: usize = 2;
const CHIS: &[usize] = &[4, 8, 16, 32, 64];

struct MpsFixture {
    bra_tensors: Vec<Tensor>,
    ket_tensors: Vec<Tensor>,
}

fn mps_shapes(sites: usize, phys_dim: usize, bond_dim: usize) -> Vec<Vec<usize>> {
    (0..sites)
        .map(|site| {
            let left = if site == 0 { 1 } else { bond_dim };
            let right = if site + 1 == sites { 1 } else { bond_dim };
            vec![left, phys_dim, right]
        })
        .collect()
}

fn complex_tensor(shape: &[usize], seed: usize) -> Tensor {
    let len = shape.iter().product();
    let data = (0..len)
        .map(|idx| {
            let real = ((idx * 17 + seed * 13 + 3) % 97) as f64 / 97.0 - 0.5;
            let imag = ((idx * 29 + seed * 7 + 5) % 89) as f64 / 89.0 - 0.5;
            Complex64::new(real, imag)
        })
        .collect();
    Tensor::C64(TypedTensor::from_vec_col_major(shape.to_vec(), data).unwrap())
}

fn build_mps_fixture(sites: usize, phys_dim: usize, bond_dim: usize) -> MpsFixture {
    let shapes = mps_shapes(sites, phys_dim, bond_dim);
    let bra_tensors = shapes
        .iter()
        .enumerate()
        .map(|(site, shape)| complex_tensor(shape, site + 1))
        .collect();
    let ket_tensors = shapes
        .iter()
        .enumerate()
        .map(|(site, shape)| complex_tensor(shape, sites + site + 1))
        .collect();

    MpsFixture {
        bra_tensors,
        ket_tensors,
    }
}

fn trace_default(trace: &mut TraceContext, tensor: &Tensor) -> tenferro_runtime::TraceValue {
    trace
        .input_with_default(
            ProgramInputSpec::new(tensor.dtype(), DimExpr::from_concrete(tensor.shape())),
            Arc::new(tensor.duplicate().unwrap()),
        )
        .unwrap()
}

fn build_inner_product_graph(
    trace: &mut TraceContext,
    fixture: &MpsFixture,
) -> tenferro_runtime::TraceValue {
    let env_tensor =
        Tensor::from_vec_col_major(vec![1, 1], vec![Complex64::new(1.0, 0.0)]).unwrap();
    let mut env = trace_default(trace, &env_tensor);
    let bra = fixture
        .bra_tensors
        .iter()
        .map(|tensor| trace_default(trace, tensor))
        .collect::<Vec<_>>();
    let ket = fixture
        .ket_tensors
        .iter()
        .map(|tensor| trace_default(trace, tensor))
        .collect::<Vec<_>>();
    for (bra_core, ket_core) in bra.iter().zip(ket) {
        let bra_core = trace.add_op(CoreSemanticOp::Conj, &[*bra_core]).unwrap()[0];
        env = trace
            .einsum(&[env, bra_core, ket_core], "ab,acr,bcs->rs")
            .expect("MPS inner-product contraction should build");
    }
    env
}

fn compile_mps_inner_product(fixture: &MpsFixture) -> CompiledGraph {
    let mut trace = TraceContext::new();
    let output = build_inner_product_graph(&mut trace, fixture);
    let graph = trace
        .finish(&[output])
        .expect("MPS inner-product trace should finish");
    let mut compiler = GraphCompiler::new();
    compiler
        .compile_traced_graph(&graph)
        .expect("MPS inner-product semantic graph should compile")
}

fn cpu_runtime_with_einsum_one_thread() -> Runtime {
    let backend = CpuBackend::with_threads(1).unwrap();
    let mut builder = Runtime::builder();
    builder
        .register_engine(tenferro_cpu::runtime_engine_registration(&backend).unwrap())
        .unwrap();
    builder
        .install_extension_module(
            tenferro_einsum::extension_module::<CpuBackend>(
                tenferro_cpu::runtime_engine_id().unwrap(),
            )
            .unwrap(),
        )
        .unwrap();
    builder.build().unwrap()
}

fn bench_mps_inner_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("mps_inner_product/c64/one_thread");
    for &chi in CHIS {
        let fixture = build_mps_fixture(L, PHYS_DIM, chi);
        let compiled = compile_mps_inner_product(&fixture);
        let runtime = cpu_runtime_with_einsum_one_thread();
        let params = format!("L_{L}_chi_{chi}_d_{PHYS_DIM}");

        let warmup_outputs = runtime
            .run_compiled(&compiled, &[])
            .expect("warmup evaluation should succeed");
        black_box(warmup_outputs);

        group.bench_function(BenchmarkId::new("eval_only", &params), move |b| {
            b.iter_batched(
                || compiled.clone(),
                |program| {
                    let outputs = runtime
                        .run_compiled(black_box(&program), &[])
                        .expect("MPS inner-product evaluation should succeed");
                    black_box(outputs);
                },
                BatchSize::SmallInput,
            );
        });

        group.bench_function(
            BenchmarkId::new("compile_and_first_eval", &params),
            move |b| {
                b.iter(|| {
                    let compiled = compile_mps_inner_product(black_box(&fixture));
                    let runtime = cpu_runtime_with_einsum_one_thread();
                    let outputs = runtime
                        .run_compiled(black_box(&compiled), &[])
                        .expect("MPS inner-product compile+eval should succeed");
                    black_box(outputs);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_mps_inner_product);
criterion_main!(benches);
