use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};
use tenferro_tensor::{
    backend::accumulate_dot_result_into, ContractionScalar, DotGeneralAccumulation, Tensor,
    TensorWrite,
};

const ELEMENT_COUNT: usize = 4096;

fn compact_dot_accumulation(c: &mut Criterion) {
    let dot = Tensor::from_vec_col_major([ELEMENT_COUNT], vec![1.0_f64; ELEMENT_COUNT]).unwrap();
    let accumulation = DotGeneralAccumulation {
        lhs_conj: false,
        rhs_conj: false,
        alpha: ContractionScalar::F64(1.0),
        beta: ContractionScalar::F64(1.0),
    };

    c.bench_function("dot_accumulation/compact_f64/4096", |b| {
        b.iter_batched(
            || Tensor::from_vec_col_major([ELEMENT_COUNT], vec![2.0_f64; ELEMENT_COUNT]).unwrap(),
            |mut out| {
                let mut write = TensorWrite::from_tensor(&mut out);
                accumulate_dot_result_into(black_box(&dot), accumulation, &mut write).unwrap();
                black_box(out);
            },
            BatchSize::SmallInput,
        );
    });
}

criterion_group!(benches, compact_dot_accumulation);
criterion_main!(benches);
