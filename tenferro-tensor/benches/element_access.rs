use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use tenferro_tensor::{MemoryOrder, Tensor, TypedTensor};

const INDEX_COUNT: usize = 4096;

fn tensor_with_order<const R: usize>(shape: [usize; R], order: MemoryOrder) -> TypedTensor<f64> {
    let len = shape.iter().product();
    let data = (0..len).map(|value| value as f64).collect();
    TypedTensor::from_vec_with_order(shape.to_vec(), data, order)
}

fn dynamic_tensor_with_order<const R: usize>(shape: [usize; R], order: MemoryOrder) -> Tensor {
    let len = shape.iter().product();
    let data = (0..len).map(|value| value as f64).collect();
    Tensor::from_vec_with_order(shape.to_vec(), data, order)
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

fn offset_for_order<const R: usize>(
    shape: &[usize; R],
    index: &[usize; R],
    order: MemoryOrder,
) -> usize {
    match order {
        MemoryOrder::ColMajor => {
            let mut offset = 0usize;
            let mut stride = 1usize;
            for axis in 0..R {
                offset += index[axis] * stride;
                stride *= shape[axis];
            }
            offset
        }
        MemoryOrder::RowMajor => {
            let mut offset = 0usize;
            let mut stride = 1usize;
            for axis in (0..R).rev() {
                offset += index[axis] * stride;
                stride *= shape[axis];
            }
            offset
        }
    }
}

fn bench_rank<const R: usize>(c: &mut Criterion, name: &str, shape: [usize; R]) {
    let indices = index_workload(shape);

    for order in [MemoryOrder::ColMajor, MemoryOrder::RowMajor] {
        let order_name = match order {
            MemoryOrder::ColMajor => "col_major",
            MemoryOrder::RowMajor => "row_major",
        };
        let tensor = tensor_with_order(shape, order);
        let offsets = indices
            .iter()
            .map(|index| offset_for_order(&shape, index, order))
            .collect::<Vec<_>>();
        let mut group = c.benchmark_group(format!("element_access/{name}/{order_name}"));

        group.bench_function(BenchmarkId::new("direct_slice", INDEX_COUNT), |b| {
            let data = tensor.as_slice();
            b.iter(|| {
                let mut sum = 0.0;
                for &offset in &offsets {
                    sum += black_box(data[black_box(offset)]);
                }
                black_box(sum)
            });
        });

        group.bench_function(BenchmarkId::new("direct_slice_mut", INDEX_COUNT), |b| {
            b.iter_batched(
                || tensor.clone(),
                |mut tensor| {
                    let data = tensor.host_data_mut();
                    for &offset in &offsets {
                        data[black_box(offset)] += black_box(1.0);
                    }
                    black_box(data[0])
                },
                BatchSize::SmallInput,
            );
        });

        group.bench_function(BenchmarkId::new("linear_offset", INDEX_COUNT), |b| {
            b.iter(|| {
                let mut sum = 0usize;
                for index in &indices {
                    sum = sum.wrapping_add(tensor.linear_offset(black_box(index.as_slice())));
                }
                black_box(sum)
            });
        });

        group.bench_function(BenchmarkId::new("get", INDEX_COUNT), |b| {
            b.iter(|| {
                let mut sum = 0.0;
                for index in &indices {
                    sum += *black_box(tensor.get(black_box(index.as_slice())));
                }
                black_box(sum)
            });
        });

        group.bench_function(BenchmarkId::new("try_get", INDEX_COUNT), |b| {
            b.iter(|| {
                let mut sum = 0.0;
                for index in &indices {
                    sum += *black_box(tensor.try_get(black_box(index.as_slice())).unwrap());
                }
                black_box(sum)
            });
        });

        group.bench_function(BenchmarkId::new("get_unchecked", INDEX_COUNT), |b| {
            b.iter(|| {
                let mut sum = 0.0;
                for index in &indices {
                    sum += *black_box(unsafe { tensor.get_unchecked(black_box(index.as_slice())) });
                }
                black_box(sum)
            });
        });

        group.bench_function(BenchmarkId::new("get_mut", INDEX_COUNT), |b| {
            b.iter_batched(
                || tensor.clone(),
                |mut tensor| {
                    for index in &indices {
                        *tensor.get_mut(black_box(index.as_slice())) += black_box(1.0);
                    }
                    black_box(tensor.as_slice()[0])
                },
                BatchSize::SmallInput,
            );
        });

        group.bench_function(BenchmarkId::new("try_get_mut", INDEX_COUNT), |b| {
            b.iter_batched(
                || tensor.clone(),
                |mut tensor| {
                    for index in &indices {
                        *tensor.try_get_mut(black_box(index.as_slice())).unwrap() += black_box(1.0);
                    }
                    black_box(tensor.as_slice()[0])
                },
                BatchSize::SmallInput,
            );
        });

        group.bench_function(BenchmarkId::new("get_unchecked_mut", INDEX_COUNT), |b| {
            b.iter_batched(
                || tensor.clone(),
                |mut tensor| {
                    for index in &indices {
                        unsafe {
                            *tensor.get_unchecked_mut(black_box(index.as_slice())) +=
                                black_box(1.0);
                        }
                    }
                    black_box(tensor.as_slice()[0])
                },
                BatchSize::SmallInput,
            );
        });

        group.finish();
    }
}

fn element_access(c: &mut Criterion) {
    bench_rank(c, "2d", [64, 64]);
    bench_rank(c, "3d", [32, 16, 8]);
    bench_rank(c, "4d", [16, 8, 8, 4]);
    bench_rank2_fixed(c);
    bench_rank3_fixed(c);
    bench_linear_iteration(c);
}

fn bench_rank2_fixed(c: &mut Criterion) {
    let shape = [64usize, 64];
    let indices = index_workload(shape);
    for order in [MemoryOrder::ColMajor, MemoryOrder::RowMajor] {
        let order_name = match order {
            MemoryOrder::ColMajor => "col_major",
            MemoryOrder::RowMajor => "row_major",
        };
        let tensor = tensor_with_order(shape, order);
        let mut group = c.benchmark_group(format!("rank_fixed/2d/{order_name}"));

        group.bench_function(BenchmarkId::new("linear_offset2", INDEX_COUNT), |b| {
            b.iter(|| {
                let mut sum = 0usize;
                for [i, j] in &indices {
                    sum = sum.wrapping_add(tensor.linear_offset2(black_box(*i), black_box(*j)));
                }
                black_box(sum)
            });
        });

        group.bench_function(BenchmarkId::new("get2", INDEX_COUNT), |b| {
            b.iter(|| {
                let mut sum = 0.0;
                for [i, j] in &indices {
                    sum += *black_box(tensor.get2(black_box(*i), black_box(*j)));
                }
                black_box(sum)
            });
        });

        group.bench_function(BenchmarkId::new("get_mut2", INDEX_COUNT), |b| {
            b.iter_batched(
                || tensor.clone(),
                |mut tensor| {
                    for [i, j] in &indices {
                        *tensor.get_mut2(black_box(*i), black_box(*j)) += black_box(1.0);
                    }
                    black_box(tensor.as_slice()[0])
                },
                BatchSize::SmallInput,
            );
        });

        group.finish();
    }
}

fn bench_rank3_fixed(c: &mut Criterion) {
    let shape = [32usize, 16, 8];
    let indices = index_workload(shape);
    for order in [MemoryOrder::ColMajor, MemoryOrder::RowMajor] {
        let order_name = match order {
            MemoryOrder::ColMajor => "col_major",
            MemoryOrder::RowMajor => "row_major",
        };
        let tensor = tensor_with_order(shape, order);
        let mut group = c.benchmark_group(format!("rank_fixed/3d/{order_name}"));

        group.bench_function(BenchmarkId::new("linear_offset3", INDEX_COUNT), |b| {
            b.iter(|| {
                let mut sum = 0usize;
                for [i, j, k] in &indices {
                    sum = sum.wrapping_add(tensor.linear_offset3(
                        black_box(*i),
                        black_box(*j),
                        black_box(*k),
                    ));
                }
                black_box(sum)
            });
        });

        group.bench_function(BenchmarkId::new("get3", INDEX_COUNT), |b| {
            b.iter(|| {
                let mut sum = 0.0;
                for [i, j, k] in &indices {
                    sum += *black_box(tensor.get3(black_box(*i), black_box(*j), black_box(*k)));
                }
                black_box(sum)
            });
        });

        group.bench_function(BenchmarkId::new("get_mut3", INDEX_COUNT), |b| {
            b.iter_batched(
                || tensor.clone(),
                |mut tensor| {
                    for [i, j, k] in &indices {
                        *tensor.get_mut3(black_box(*i), black_box(*j), black_box(*k)) +=
                            black_box(1.0);
                    }
                    black_box(tensor.as_slice()[0])
                },
                BatchSize::SmallInput,
            );
        });

        group.finish();
    }
}

fn bench_linear_iteration(c: &mut Criterion) {
    let shape = [256usize, 256];
    for order in [MemoryOrder::ColMajor, MemoryOrder::RowMajor] {
        let order_name = match order {
            MemoryOrder::ColMajor => "col_major",
            MemoryOrder::RowMajor => "row_major",
        };
        let tensor = tensor_with_order(shape, order);
        let dynamic_tensor = dynamic_tensor_with_order(shape, order);
        let mut group = c.benchmark_group(format!("linear_iteration/{order_name}"));

        group.bench_function("as_slice_iter", |b| {
            b.iter(|| {
                let sum = tensor
                    .as_slice()
                    .iter()
                    .fold(0.0, |acc, value| acc + black_box(*value));
                black_box(sum)
            });
        });

        group.bench_function("tensor_iter", |b| {
            b.iter(|| {
                let sum = tensor
                    .iter()
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
                || tensor.clone(),
                |mut tensor| {
                    for value in tensor.iter_mut() {
                        *value += black_box(1.0);
                    }
                    black_box(tensor.as_slice()[0])
                },
                BatchSize::SmallInput,
            );
        });

        group.bench_function("dynamic_tensor_iter_mut", |b| {
            b.iter_batched(
                || dynamic_tensor.clone(),
                |mut tensor| {
                    for value in tensor.iter_mut::<f64>().unwrap() {
                        *value += black_box(1.0);
                    }
                    black_box(tensor.as_slice::<f64>().unwrap()[0])
                },
                BatchSize::SmallInput,
            );
        });

        group.finish();
    }
}

criterion_group!(benches, element_access);
criterion_main!(benches);
