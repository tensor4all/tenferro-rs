use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use tenferro_tensor::{Rank, Tensor, TypedTensor, TypedTensorView};

const INDEX_COUNT: usize = 4096;

/// # Panics
///
/// Panics if the benchmark is called with a non-compact or non-host tensor;
/// the probe is intentionally restricted to the validated benchmark fixture.
#[inline(never)]
#[no_mangle]
pub extern "C" fn tensor_static_rank_read_probe(tensor: &TypedTensor<f64, Rank<2>>) -> f64 {
    tensor
        .host_col_major_view()
        .expect("the probe uses a compact host tensor")
        .axis0_lanes()
        .flatten()
        .fold(0.0, |sum, value| sum + *value)
}

/// # Panics
///
/// Panics if the benchmark is called with a non-host tensor; the probe is
/// intentionally restricted to the validated benchmark fixture.
#[inline(never)]
#[no_mangle]
pub extern "C" fn tensor_static_rank_write_probe(tensor: &mut TypedTensor<f64, Rank<2>>) {
    for value in tensor
        .host_col_major_view_mut()
        .expect("the probe uses a host tensor")
        .iter_mut()
    {
        *value *= 2.0;
    }
}

fn typed_tensor<const R: usize>(shape: [usize; R]) -> TypedTensor<f64> {
    let len = shape.iter().product();
    let data = (0..len).map(|value| value as f64).collect();
    TypedTensor::from_vec_col_major(shape.to_vec(), data).unwrap()
}

fn static_tensor<const R: usize>(shape: [usize; R]) -> TypedTensor<f64, Rank<R>> {
    let len = shape.iter().product();
    let data = (0..len).map(|value| value as f64).collect();
    TypedTensor::from_vec_col_major(shape, data).unwrap()
}

fn dynamic_tensor<const R: usize>(shape: [usize; R]) -> Tensor {
    let len = shape.iter().product();
    let data = (0..len).map(|value| value as f64).collect();
    Tensor::from_vec_col_major(shape.to_vec(), data).unwrap()
}

fn index_workload<const R: usize>(shape: [usize; R]) -> Vec<[usize; R]> {
    let mut indices = Vec::with_capacity(INDEX_COUNT);
    for n in 0..INDEX_COUNT {
        let mut idx = [0usize; R];
        for axis in 0..R {
            let factor = 17 + axis * 12;
            idx[axis] = (n.wrapping_mul(factor) + axis * 7) % shape[axis];
        }
        indices.push(idx);
    }
    indices
}

fn col_major_offset<const R: usize>(shape: &[usize; R], index: &[usize; R]) -> usize {
    let mut offset = 0usize;
    let mut stride = 1usize;
    for axis in 0..R {
        offset += index[axis] * stride;
        stride *= shape[axis];
    }
    offset
}

fn bench_rank<const R: usize>(c: &mut Criterion, name: &str, shape: [usize; R]) {
    let indices = index_workload(shape);
    let tensor = typed_tensor(shape);
    let offsets = indices
        .iter()
        .map(|index| col_major_offset(&shape, index))
        .collect::<Vec<_>>();
    let mut group = c.benchmark_group(format!("element_access/{name}/col_major"));

    group.bench_function(BenchmarkId::new("direct_slice", INDEX_COUNT), |b| {
        let data = tensor.as_slice().unwrap();
        b.iter(|| {
            let mut sum = 0.0;
            for &offset in &offsets {
                sum += black_box(data[black_box(offset)]);
            }
            black_box(sum)
        });
    });

    group.bench_function(
        BenchmarkId::new("direct_slice_unchecked", INDEX_COUNT),
        |b| {
            let data = tensor.as_slice().unwrap();
            b.iter(|| {
                let mut sum = 0.0;
                for &offset in &offsets {
                    // SAFETY: offsets are generated from in-bounds indices and the
                    // compact fixture has exactly shape.product() elements.
                    sum += black_box(unsafe { data.get_unchecked(black_box(offset)) });
                }
                black_box(sum)
            });
        },
    );

    group.bench_function(BenchmarkId::new("direct_slice_mut", INDEX_COUNT), |b| {
        b.iter_batched(
            || tensor.duplicate().unwrap(),
            |mut tensor| {
                let data = tensor.host_data_mut().unwrap();
                let mut sum = 0.0;
                for &offset in &offsets {
                    data[black_box(offset)] += black_box(1.0);
                    sum += data[offset];
                }
                black_box(sum)
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function(BenchmarkId::new("linear_offset", INDEX_COUNT), |b| {
        b.iter(|| {
            let mut sum = 0usize;
            for index in &indices {
                sum = sum.wrapping_add(tensor.linear_offset(black_box(index.as_slice())).unwrap());
            }
            black_box(sum)
        });
    });

    group.bench_function(BenchmarkId::new("get", INDEX_COUNT), |b| {
        b.iter(|| {
            let mut sum = 0.0;
            for index in &indices {
                sum += *black_box(tensor.get(black_box(index.as_slice())).unwrap());
            }
            black_box(sum)
        });
    });

    group.bench_function(BenchmarkId::new("get_unchecked", INDEX_COUNT), |b| {
        b.iter(|| {
            let mut sum = 0.0;
            for index in &indices {
                sum += *black_box(unsafe {
                    tensor.get_unchecked(black_box(index.as_slice())).unwrap()
                });
            }
            black_box(sum)
        });
    });

    group.bench_function(BenchmarkId::new("get_mut", INDEX_COUNT), |b| {
        b.iter_batched(
            || tensor.duplicate().unwrap(),
            |mut tensor| {
                let mut sum = 0.0;
                for index in &indices {
                    let value = tensor.get_mut(black_box(index.as_slice())).unwrap();
                    *value += black_box(1.0);
                    sum += *value;
                }
                black_box(sum)
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function(BenchmarkId::new("get_unchecked_mut", INDEX_COUNT), |b| {
        b.iter_batched(
            || tensor.duplicate().unwrap(),
            |mut tensor| {
                let mut sum = 0.0;
                for index in &indices {
                    unsafe {
                        let value = tensor
                            .get_unchecked_mut(black_box(index.as_slice()))
                            .unwrap();
                        *value += black_box(1.0);
                        sum += *value;
                    }
                }
                black_box(sum)
            },
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

fn element_access(c: &mut Criterion) {
    bench_rank(c, "2d", [64, 64]);
    bench_rank(c, "3d", [32, 16, 8]);
    bench_rank(c, "4d", [16, 8, 8, 4]);
    bench_rank2_fixed(c);
    bench_rank3_fixed(c);
    bench_linear_iteration(c);
    bench_strided_traversal(c);
}

fn bench_rank2_fixed(c: &mut Criterion) {
    let shape = [64usize, 64];
    let indices = index_workload(shape);
    let tensor = typed_tensor(shape);
    let static_tensor = static_tensor(shape);
    let mut group = c.benchmark_group("rank_fixed/2d/col_major");

    group.bench_function(BenchmarkId::new("linear_offset2", INDEX_COUNT), |b| {
        b.iter(|| {
            let mut sum = 0usize;
            for [i, j] in &indices {
                sum =
                    sum.wrapping_add(tensor.linear_offset2(black_box(*i), black_box(*j)).unwrap());
            }
            black_box(sum)
        });
    });

    group.bench_function(BenchmarkId::new("get2", INDEX_COUNT), |b| {
        b.iter(|| {
            let mut sum = 0.0;
            for [i, j] in &indices {
                sum += *black_box(tensor.get2(black_box(*i), black_box(*j)).unwrap());
            }
            black_box(sum)
        });
    });

    group.bench_function("nested_get2", |b| {
        b.iter(|| {
            let mut sum = 0.0;
            for j in 0..shape[1] {
                for i in 0..shape[0] {
                    sum += *black_box(tensor.get2(i, j).unwrap());
                }
            }
            black_box(sum)
        });
    });

    group.bench_function("nested_direct_slice_unchecked", |b| {
        b.iter(|| {
            let data = tensor.as_slice().unwrap();
            let mut sum = 0.0;
            for j in 0..shape[1] {
                for i in 0..shape[0] {
                    let offset = i + shape[0] * j;
                    // SAFETY: nested loops stay within the compact fixture.
                    sum += black_box(unsafe { data.get_unchecked(offset) });
                }
            }
            black_box(sum)
        });
    });

    group.bench_function("nested_col_major_get", |b| {
        let view = static_tensor.host_col_major_view().unwrap();
        b.iter(|| {
            let mut sum = 0.0;
            for j in 0..shape[1] {
                for i in 0..shape[0] {
                    sum += black_box(view.get([black_box(i), black_box(j)]).unwrap());
                }
            }
            black_box(sum)
        });
    });

    group.bench_function("nested_col_major_unchecked", |b| {
        let view = static_tensor.host_col_major_view().unwrap();
        b.iter(|| {
            let mut sum = 0.0;
            for j in 0..shape[1] {
                for i in 0..shape[0] {
                    // SAFETY: both loops use the validated shape bounds.
                    sum += black_box(unsafe { view.get_unchecked([i, j]) });
                }
            }
            black_box(sum)
        });
    });

    group.bench_function("col_major_axis0_lanes", |b| {
        let view = static_tensor.host_col_major_view().unwrap();
        b.iter(|| {
            let mut sum = 0.0;
            for lane in view.axis0_lanes() {
                for value in lane {
                    sum += black_box(*value);
                }
            }
            black_box(sum)
        });
    });

    group.bench_function("col_major_axis0_lanes_mut", |b| {
        b.iter_batched(
            || static_tensor.duplicate().unwrap(),
            |mut tensor| {
                let mut view = tensor.host_col_major_view_mut().unwrap();
                let mut sum = 0.0;
                for lane in view.axis0_lanes_mut() {
                    for value in lane {
                        *value += black_box(1.0);
                        sum += *value;
                    }
                }
                black_box(sum)
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function(BenchmarkId::new("get_mut2", INDEX_COUNT), |b| {
        b.iter_batched(
            || tensor.duplicate().unwrap(),
            |mut tensor| {
                let mut sum = 0.0;
                for [i, j] in &indices {
                    let value = tensor.get_mut2(black_box(*i), black_box(*j)).unwrap();
                    *value += black_box(1.0);
                    sum += *value;
                }
                black_box(sum)
            },
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

fn bench_rank3_fixed(c: &mut Criterion) {
    let shape = [32usize, 16, 8];
    let indices = index_workload(shape);
    let tensor = typed_tensor(shape);
    let static_tensor = static_tensor(shape);
    let mut group = c.benchmark_group("rank_fixed/3d/col_major");

    group.bench_function(BenchmarkId::new("linear_offset3", INDEX_COUNT), |b| {
        b.iter(|| {
            let mut sum = 0usize;
            for [i, j, k] in &indices {
                sum = sum.wrapping_add(
                    tensor
                        .linear_offset3(black_box(*i), black_box(*j), black_box(*k))
                        .unwrap(),
                );
            }
            black_box(sum)
        });
    });

    group.bench_function(BenchmarkId::new("get3", INDEX_COUNT), |b| {
        b.iter(|| {
            let mut sum = 0.0;
            for [i, j, k] in &indices {
                sum += *black_box(
                    tensor
                        .get3(black_box(*i), black_box(*j), black_box(*k))
                        .unwrap(),
                );
            }
            black_box(sum)
        });
    });

    group.bench_function(BenchmarkId::new("get_mut3", INDEX_COUNT), |b| {
        b.iter_batched(
            || tensor.duplicate().unwrap(),
            |mut tensor| {
                let mut sum = 0.0;
                for [i, j, k] in &indices {
                    let value = tensor
                        .get_mut3(black_box(*i), black_box(*j), black_box(*k))
                        .unwrap();
                    *value += black_box(1.0);
                    sum += *value;
                }
                black_box(sum)
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("nested_direct_slice_unchecked", |b| {
        let data = static_tensor.as_slice().unwrap();
        b.iter(|| {
            let mut sum = 0.0;
            for k in 0..shape[2] {
                for j in 0..shape[1] {
                    for i in 0..shape[0] {
                        let offset = i + shape[0] * (j + shape[1] * k);
                        // SAFETY: nested loops stay within the compact fixture.
                        sum += black_box(unsafe { data.get_unchecked(offset) });
                    }
                }
            }
            black_box(sum)
        });
    });

    group.bench_function("nested_col_major_get", |b| {
        let view = static_tensor.host_col_major_view().unwrap();
        b.iter(|| {
            let mut sum = 0.0;
            for k in 0..shape[2] {
                for j in 0..shape[1] {
                    for i in 0..shape[0] {
                        sum += black_box(
                            view.get([black_box(i), black_box(j), black_box(k)])
                                .unwrap(),
                        );
                    }
                }
            }
            black_box(sum)
        });
    });

    group.bench_function("nested_col_major_unchecked", |b| {
        let view = static_tensor.host_col_major_view().unwrap();
        b.iter(|| {
            let mut sum = 0.0;
            for k in 0..shape[2] {
                for j in 0..shape[1] {
                    for i in 0..shape[0] {
                        // SAFETY: all loops use the validated shape bounds.
                        sum += black_box(unsafe { view.get_unchecked([i, j, k]) });
                    }
                }
            }
            black_box(sum)
        });
    });

    group.bench_function("col_major_axis0_lanes", |b| {
        let view = static_tensor.host_col_major_view().unwrap();
        b.iter(|| {
            let mut sum = 0.0;
            for lane in view.axis0_lanes() {
                for value in lane {
                    sum += black_box(*value);
                }
            }
            black_box(sum)
        });
    });

    group.finish();
}

fn bench_linear_iteration(c: &mut Criterion) {
    let shape = [256usize, 256];
    let tensor = typed_tensor(shape);
    let dynamic_tensor = dynamic_tensor(shape);
    let mut group = c.benchmark_group("linear_iteration/col_major");

    group.bench_function("as_slice_iter", |b| {
        b.iter(|| {
            let sum = tensor
                .as_slice()
                .unwrap()
                .iter()
                .fold(0.0, |acc, value| acc + black_box(*value));
            black_box(sum)
        });
    });

    group.bench_function("tensor_iter", |b| {
        b.iter(|| {
            let sum = tensor
                .iter()
                .unwrap()
                .fold(0.0, |acc, value| acc + black_box(*value));
            black_box(sum)
        });
    });

    group.bench_function("dynamic_tensor_iter", |b| {
        b.iter(|| {
            let sum = dynamic_tensor
                .iter::<f64>()
                .unwrap()
                .fold(0.0, |acc, value| acc + black_box(*value));
            black_box(sum)
        });
    });

    group.bench_function("tensor_iter_mut", |b| {
        b.iter_batched(
            || tensor.duplicate().unwrap(),
            |mut tensor| {
                let mut sum = 0.0;
                for value in tensor.iter_mut().unwrap() {
                    *value += black_box(1.0);
                    sum += *value;
                }
                black_box(sum)
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("dynamic_tensor_iter_mut", |b| {
        b.iter_batched(
            || dynamic_tensor.duplicate().unwrap(),
            |mut tensor| {
                let mut sum = 0.0;
                for value in tensor.iter_mut::<f64>().unwrap() {
                    *value += black_box(1.0);
                    sum += *value;
                }
                black_box(sum)
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("empty", |b| {
        b.iter(|| {
            let values: &[f64] = &[];
            black_box(values.iter().copied().sum::<f64>())
        });
    });

    group.finish();
}

fn bench_strided_traversal(c: &mut Criterion) {
    let source_shape = [48usize, 80];
    let view_shape = [source_shape[1], source_shape[0]];
    let element_count = view_shape[0] * view_shape[1];
    let mut indices = Vec::with_capacity(element_count);
    for column in 0..view_shape[1] {
        for row in 0..view_shape[0] {
            indices.push([row, column]);
        }
    }
    let tensor = typed_tensor(source_shape);
    let view = TypedTensorView::from_col_major(&source_shape, tensor.as_slice().unwrap())
        .unwrap()
        .transpose_view([1, 0])
        .unwrap();
    let mut group = c.benchmark_group("strided_traversal/rectangular_transpose");

    group.bench_function(BenchmarkId::new("logical_order_get", element_count), |b| {
        b.iter(|| {
            let mut sum = 0.0;
            for index in &indices {
                sum += *black_box(view.get(black_box(index.as_slice())).unwrap());
            }
            black_box(sum)
        });
    });

    group.finish();
}

criterion_group!(benches, element_access);
criterion_main!(benches);
