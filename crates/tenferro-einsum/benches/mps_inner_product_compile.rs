use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use num_complex::Complex64;
use tenferro_einsum::traced_tensor::einsum;
use tenferro_runtime::{DType, GraphCompiler, GraphProgram, TracedTensor};

const PHYS_DIM: usize = 2;
const CHI: usize = 32;
const LS: &[usize] = &[4, 8, 16, 32, 64];

struct CompileCase {
    shapes: Vec<Vec<usize>>,
    bra_placeholders: Vec<TracedTensor>,
    ket_placeholders: Vec<TracedTensor>,
    output: TracedTensor,
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
    compiler: &mut GraphCompiler,
    bra: &[TracedTensor],
    ket: &[TracedTensor],
) -> TracedTensor {
    let mut env =
        TracedTensor::from_vec_col_major(vec![1, 1], vec![Complex64::new(1.0, 0.0)]).unwrap();
    for (bra_core, ket_core) in bra.iter().zip(ket) {
        let bra_core = bra_core.conj();
        env = einsum(compiler, &[&env, &bra_core, ket_core], "ab,acr,bcs->rs")
            .expect("MPS inner-product contraction should build");
    }
    env.reshape(&[])
}

fn build_compile_case(sites: usize, chi: usize) -> CompileCase {
    let shapes = mps_shapes(sites, PHYS_DIM, chi);
    let bra_placeholders = shapes
        .iter()
        .map(|shape| TracedTensor::input_concrete_shape(DType::C64, shape).unwrap())
        .collect::<Vec<_>>();
    let ket_placeholders = shapes
        .iter()
        .map(|shape| TracedTensor::input_concrete_shape(DType::C64, shape).unwrap())
        .collect::<Vec<_>>();
    let mut compiler = GraphCompiler::new();
    let output = build_inner_product_graph(&mut compiler, &bra_placeholders, &ket_placeholders);

    CompileCase {
        shapes,
        bra_placeholders,
        ket_placeholders,
        output,
    }
}

fn compile_case(case: &CompileCase) -> GraphProgram {
    let mut specs = Vec::with_capacity(case.shapes.len() * 2);
    for site in 0..case.shapes.len() {
        let shape = case.shapes[site].as_slice();
        specs.push((&case.bra_placeholders[site], DType::C64, shape));
        specs.push((&case.ket_placeholders[site], DType::C64, shape));
    }

    let mut compiler = GraphCompiler::new();
    compiler
        .compile_with_input_specs(&case.output, &specs)
        .expect("MPS inner-product graph should compile from specs")
}

fn bench_mps_inner_product_compile(c: &mut Criterion) {
    let mut group = c.benchmark_group("mps_inner_product_compile/c64/one_thread");

    for &l in LS {
        let params = format!("L_{l}_chi_{CHI}_d_{PHYS_DIM}");

        group.bench_function(BenchmarkId::new("build_graph_only", &params), |b| {
            b.iter(|| {
                let case = build_compile_case(black_box(l), black_box(CHI));
                black_box(case.output.rank);
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
