use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use num_complex::Complex64;
use tenferro::traced_tensor::einsum;
use tenferro::{CpuBackend, DType, Engine, Tensor, TracedTensor, TypedTensor};

const L: usize = 32;
const PHYS_DIM: usize = 2;
const CHIS: &[usize] = &[4, 8, 16, 32, 64];

struct MpsFixture {
    shapes: Vec<Vec<usize>>,
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
    Tensor::C64(TypedTensor::from_vec(shape.to_vec(), data))
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
        shapes,
        bra_tensors,
        ket_tensors,
    }
}

fn build_inner_product_graph(
    engine: &mut Engine<CpuBackend>,
    bra: &[TracedTensor],
    ket: &[TracedTensor],
) -> TracedTensor {
    let mut env = TracedTensor::from_vec(vec![1, 1], vec![Complex64::new(1.0, 0.0)]);
    for (bra_core, ket_core) in bra.iter().zip(ket) {
        let bra_core = bra_core.conj();
        env = einsum(engine, &[&env, &bra_core, ket_core], "ab,acr,bcs->rs")
            .expect("MPS inner-product contraction should build");
    }
    env.reshape(&[])
}

fn compile_mps_inner_product(fixture: &MpsFixture) -> tenferro::traced::CompiledTracedTensor {
    let bra_placeholders: Vec<_> = fixture
        .shapes
        .iter()
        .map(|shape| TracedTensor::input_concrete_shape(DType::C64, shape))
        .collect();
    let ket_placeholders: Vec<_> = fixture
        .shapes
        .iter()
        .map(|shape| TracedTensor::input_concrete_shape(DType::C64, shape))
        .collect();

    let mut build_engine = Engine::new(CpuBackend::with_threads(1));
    let output = build_inner_product_graph(&mut build_engine, &bra_placeholders, &ket_placeholders);

    let mut bindings = Vec::with_capacity(fixture.shapes.len() * 2);
    for site in 0..fixture.shapes.len() {
        bindings.push((&bra_placeholders[site], &fixture.bra_tensors[site]));
        bindings.push((&ket_placeholders[site], &fixture.ket_tensors[site]));
    }

    output
        .compile_with_inputs(&bindings)
        .expect("MPS inner-product graph should compile")
}

fn bench_mps_inner_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("mps_inner_product/c64/one_thread");
    for &chi in CHIS {
        let fixture = build_mps_fixture(L, PHYS_DIM, chi);
        let compiled = compile_mps_inner_product(&fixture);
        let mut engine = Engine::new(CpuBackend::with_threads(1));
        let params = format!("L_{L}_chi_{chi}_d_{PHYS_DIM}");

        let warmup_outputs = engine
            .eval_exec_ir(&compiled.program, compiled.inputs.clone())
            .expect("warmup evaluation should succeed");
        black_box(warmup_outputs);

        group.bench_function(BenchmarkId::new("eval_only", &params), move |b| {
            b.iter_batched(
                || compiled.inputs.clone(),
                |inputs| {
                    let outputs = engine
                        .eval_exec_ir(black_box(&compiled.program), black_box(inputs))
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
                    let mut engine = Engine::new(CpuBackend::with_threads(1));
                    let outputs = engine
                        .eval_exec_ir(black_box(&compiled.program), black_box(compiled.inputs))
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
