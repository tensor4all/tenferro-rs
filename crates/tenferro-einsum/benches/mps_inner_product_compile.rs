use std::sync::Arc;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use num_complex::Complex64;
use tenferro_einsum::TraceContextEinsumExt;
use tenferro_ops::dim_expr::DimExpr;
use tenferro_runtime::program::ProgramInputSpec;
use tenferro_runtime::{CompiledGraph, DType, GraphCompiler, Tensor, TraceContext, TracedGraph};

const PHYS_DIM: usize = 2;
const CHI: usize = 32;
const LS: &[usize] = &[4, 8, 16, 32, 64];

struct CompileCase {
    graph: TracedGraph,
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

fn build_inner_product_graph(
    trace: &mut TraceContext,
    shapes: &[Vec<usize>],
) -> tenferro_runtime::TraceValue {
    let env_tensor =
        Tensor::from_vec_col_major(vec![1, 1], vec![Complex64::new(1.0, 0.0)]).unwrap();
    let mut env = trace
        .input_with_default(
            ProgramInputSpec::new(DType::C64, DimExpr::from_concrete(env_tensor.shape())),
            Arc::new(env_tensor),
        )
        .unwrap();
    let bra = shapes
        .iter()
        .map(|shape| {
            trace
                .input(ProgramInputSpec::new(
                    DType::C64,
                    DimExpr::from_concrete(shape),
                ))
                .unwrap()
        })
        .collect::<Vec<_>>();
    let ket = shapes
        .iter()
        .map(|shape| {
            trace
                .input(ProgramInputSpec::new(
                    DType::C64,
                    DimExpr::from_concrete(shape),
                ))
                .unwrap()
        })
        .collect::<Vec<_>>();
    for (bra_core, ket_core) in bra.iter().zip(ket) {
        let bra_core = trace
            .add_op(
                tenferro_runtime::program::CoreSemanticOp::Conj,
                &[*bra_core],
            )
            .unwrap()[0];
        env = trace
            .einsum(&[env, bra_core, ket_core], "ab,acr,bcs->rs")
            .expect("MPS inner-product contraction should build");
    }
    env
}

fn build_compile_case(sites: usize, chi: usize) -> CompileCase {
    let shapes = mps_shapes(sites, PHYS_DIM, chi);
    let mut trace = TraceContext::new();
    let output = build_inner_product_graph(&mut trace, &shapes);
    let graph = trace
        .finish(&[output])
        .expect("MPS inner-product trace should finish");
    CompileCase { graph }
}

fn compile_case(case: &CompileCase) -> CompiledGraph {
    let mut compiler = GraphCompiler::new();
    compiler
        .compile_traced_graph(&case.graph)
        .expect("MPS inner-product semantic graph should compile")
}

fn bench_mps_inner_product_compile(c: &mut Criterion) {
    let mut group = c.benchmark_group("mps_inner_product_compile/c64/one_thread");

    for &l in LS {
        let params = format!("L_{l}_chi_{CHI}_d_{PHYS_DIM}");

        group.bench_function(BenchmarkId::new("build_graph_only", &params), |b| {
            b.iter(|| {
                let case = build_compile_case(black_box(l), black_box(CHI));
                black_box(case.graph.program().operations().count());
            });
        });

        group.bench_function(BenchmarkId::new("build_graph_and_compile", &params), |b| {
            b.iter(|| {
                let case = build_compile_case(black_box(l), black_box(CHI));
                let program = compile_case(black_box(&case));
                black_box(program.output_count());
            });
        });

        let case = build_compile_case(l, CHI);
        group.bench_function(BenchmarkId::new("compile_existing_graph", &params), |b| {
            b.iter(|| {
                let program = compile_case(black_box(&case));
                black_box(program.output_count());
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_mps_inner_product_compile);
criterion_main!(benches);
