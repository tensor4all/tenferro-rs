use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use tenferro_cpu::CpuBackend;
use tenferro_tensor::{TensorViewCanonicalization, TypedTensorView};

const TN_24D_PERM: [usize; 24] = [
    0, 1, 2, 3, 22, 4, 23, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
];

const TN_24D_SCATTERED_STRIDES: [isize; 24] = [
    1, 2, 4, 8, 4_194_304, 16, 8_388_608, 32, 64, 128, 256, 512, 1_024, 2_048, 4_096, 8_192,
    16_384, 32_768, 65_536, 131_072, 262_144, 524_288, 1_048_576, 2_097_152,
];

struct MaterializationCase {
    name: &'static str,
    storage: Vec<f64>,
    source_shape: Vec<usize>,
    source_strides: Vec<isize>,
    permutation: Option<Vec<usize>>,
    offset: isize,
}

#[derive(Clone, Copy)]
enum MaterializationSpec {
    Compact3d,
    Permuted3d,
    HighRankContiguousPermutation,
    Scattered24dExplicitStride,
    TinyTranspose,
}

const CASES: [MaterializationSpec; 5] = [
    MaterializationSpec::Compact3d,
    MaterializationSpec::Permuted3d,
    MaterializationSpec::HighRankContiguousPermutation,
    MaterializationSpec::Scattered24dExplicitStride,
    MaterializationSpec::TinyTranspose,
];

fn compact_strides(shape: &[usize]) -> Vec<isize> {
    let mut strides = Vec::with_capacity(shape.len());
    let mut stride = 1isize;
    for &extent in shape {
        strides.push(stride);
        stride = stride
            .checked_mul(isize::try_from(extent).expect("benchmark extent fits in isize"))
            .expect("benchmark compact stride fits in isize");
    }
    strides
}

fn permute_layout(shape: &[usize], strides: &[isize], perm: &[usize]) -> (Vec<usize>, Vec<isize>) {
    (
        perm.iter().map(|&axis| shape[axis]).collect(),
        perm.iter().map(|&axis| strides[axis]).collect(),
    )
}

fn storage_len(shape: &[usize], strides: &[isize], offset: isize) -> usize {
    let max_offset = shape
        .iter()
        .zip(strides)
        .try_fold(offset, |position, (&extent, &stride)| {
            let last = isize::try_from(extent.saturating_sub(1)).ok()?;
            position.checked_add(last.checked_mul(stride)?)
        })
        .expect("benchmark layout has a representable reachable range");
    usize::try_from(max_offset)
        .expect("benchmark layout has a non-negative reachable range")
        .checked_add(1)
        .expect("benchmark storage length fits in usize")
}

fn make_case(
    name: &'static str,
    source_shape: Vec<usize>,
    source_strides: Vec<isize>,
    perm: Option<&[usize]>,
) -> MaterializationCase {
    let storage_len = storage_len(&source_shape, &source_strides, 0);
    let storage = (0..storage_len).map(|index| index as f64).collect();
    MaterializationCase {
        name,
        storage,
        source_shape,
        source_strides,
        permutation: perm.map(<[usize]>::to_vec),
        offset: 0,
    }
}

fn build_case(spec: MaterializationSpec) -> MaterializationCase {
    match spec {
        MaterializationSpec::Compact3d => {
            let shape = vec![128, 128, 128];
            let strides = compact_strides(&shape);
            make_case("compact_3d", shape, strides, None)
        }
        MaterializationSpec::Permuted3d => {
            let shape = vec![128, 128, 128];
            let strides = compact_strides(&shape);
            make_case("permuted_3d", shape, strides, Some(&[2, 0, 1]))
        }
        MaterializationSpec::HighRankContiguousPermutation => {
            let shape = vec![2; 24];
            let strides = compact_strides(&shape);
            make_case(
                "high_rank_contiguous_permutation",
                shape,
                strides,
                Some(&TN_24D_PERM),
            )
        }
        MaterializationSpec::Scattered24dExplicitStride => make_case(
            "scattered_24d_explicit_stride",
            vec![2; 24],
            TN_24D_SCATTERED_STRIDES.to_vec(),
            Some(&TN_24D_PERM),
        ),
        MaterializationSpec::TinyTranspose => make_case(
            "tiny_transpose",
            vec![16, 16],
            compact_strides(&[16, 16]),
            Some(&[1, 0]),
        ),
    }
}

fn output_shape(case: &MaterializationCase) -> Vec<usize> {
    case.permutation.as_ref().map_or_else(
        || case.source_shape.clone(),
        |permutation| {
            permutation
                .iter()
                .map(|&source_axis| case.source_shape[source_axis])
                .collect()
        },
    )
}

fn expected_physical_offset(case: &MaterializationCase, output_index: &[usize]) -> isize {
    output_index
        .iter()
        .enumerate()
        .try_fold(case.offset, |position, (output_axis, &coordinate)| {
            let source_axis = case
                .permutation
                .as_ref()
                .map_or(output_axis, |permutation| permutation[output_axis]);
            position.checked_add(
                isize::try_from(coordinate)
                    .expect("benchmark coordinate fits in isize")
                    .checked_mul(case.source_strides[source_axis])
                    .expect("benchmark physical offset fits in isize"),
            )
        })
        .expect("benchmark physical offset fits in isize")
}

fn verify_exact_output(case: &MaterializationCase, output: &[f64]) {
    let shape = output_shape(case);
    let elements = shape.iter().product::<usize>();
    assert_eq!(output.len(), elements);
    let mut index = vec![0usize; shape.len()];
    for (logical_offset, &actual) in output.iter().enumerate() {
        let expected = expected_physical_offset(case, &index) as f64;
        assert_eq!(actual, expected, "logical offset {logical_offset}");

        for axis in 0..index.len() {
            index[axis] += 1;
            if index[axis] < shape[axis] {
                break;
            }
            index[axis] = 0;
        }
    }
}

fn verify_case_once(
    backend: &mut CpuBackend,
    view: &TypedTensorView<'_, f64>,
    case: &MaterializationCase,
) {
    let checked = backend
        .to_contiguous(view)
        .expect("pre-timing materialization succeeds");
    verify_exact_output(
        case,
        checked
            .as_slice()
            .expect("CPU materialization returns host storage"),
    );
}

fn bench_view_materialization(c: &mut Criterion) {
    for threads in [1, 4] {
        let mut group = c.benchmark_group(format!("view_materialization/{threads}_threads"));
        group.sample_size(10);

        for spec in CASES {
            let case = build_case(spec);
            let (view_shape, view_strides) = case.permutation.as_ref().map_or_else(
                || (case.source_shape.clone(), case.source_strides.clone()),
                |permutation| permute_layout(&case.source_shape, &case.source_strides, permutation),
            );
            let view =
                TypedTensorView::from_slice(&view_shape, &view_strides, case.offset, &case.storage)
                    .expect("benchmark view layout is valid");
            let mut backend = CpuBackend::with_threads(threads)
                .expect("benchmark CPU thread configuration is valid");

            // The checked tensor is dropped when this helper returns, before
            // Criterion starts measuring allocation-inclusive materialization.
            verify_case_once(&mut backend, &view, &case);

            group.throughput(Throughput::Elements(
                view_shape.iter().product::<usize>() as u64
            ));
            group.bench_function(case.name, |b| {
                b.iter(|| {
                    let materialized = backend
                        .to_contiguous(black_box(&view))
                        .expect("timed materialization succeeds");
                    black_box(materialized);
                });
            });
        }
        group.finish();
    }
}

criterion_group!(benches, bench_view_materialization);
criterion_main!(benches);
